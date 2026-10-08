"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It implements retry logic with exponential backoff, rate limiting,
circuit breaker pattern for fault tolerance, metrics collection, request tracing,
and comprehensive error handling.
"""

import asyncio
import logging
import time
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    PgMcpError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.tracing import generate_request_id, request_context
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation. It implements retry logic with exponential backoff, rate
    limiting for queries and LLM calls, circuit breaker pattern for fault
    tolerance, optional per-database validators/executors (enforcing
    database-specific security policies), metrics collection, request
    tracing, and comprehensive error handling.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     sql_executor=executor,
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ...     sql_validators={"mydb": validator},   # optional per-database policies
        ...     sql_executors={"mydb": executor},     # optional per-database executors
        ...     rate_limiter=rate_limiter,            # optional concurrency control
        ...     metrics=metrics,                      # optional observability
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executor: SQLExecutor,
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        sql_validators: dict[str, SQLValidator] | None = None,
        sql_executors: dict[str, SQLExecutor] | None = None,
        rate_limiter: MultiRateLimiter | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: Default SQL validation service.
            sql_executor: Default SQL execution service.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries, circuit
                breaker, and rate limiting.
            validation_config: Validation configuration including thresholds.
            sql_validators: Optional per-database validators enforcing
                database-specific security policies (blocked tables/columns,
                EXPLAIN policy). Takes precedence over ``sql_validator``.
            sql_executors: Optional per-database executors, each bound to its
                own connection pool and security config. Takes precedence
                over ``sql_executor``.
            rate_limiter: Optional rate limiter guarding concurrent queries
                and LLM calls.
            metrics: Optional metrics collector for observability.
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.sql_executor = sql_executor
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config
        self.sql_validators = sql_validators
        self.sql_executors = sql_executors
        self.rate_limiter = rate_limiter
        self.metrics = metrics

        # Create circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    def _get_validator(self, database: str) -> SQLValidator:
        """Get the SQL validator for a database.

        Prefers the per-database validator (which enforces that database's
        security policy) and falls back to the shared default validator.

        Args:
            database: Resolved database name.

        Returns:
            SQLValidator: Validator to use for this database.

        Raises:
            DatabaseError: If per-database validators are configured but none
                exists for this database.
        """
        if self.sql_validators is not None:
            validator = self.sql_validators.get(database)
            if validator is None:
                raise DatabaseError(
                    message=f"No SQL validator configured for database '{database}'",
                    details={"database": database},
                )
            return validator
        return self.sql_validator

    def _get_executor(self, database: str) -> SQLExecutor:
        """Get the SQL executor for a database.

        Prefers the per-database executor (bound to that database's pool and
        security config) and falls back to the shared default executor. This
        guarantees queries always run against the resolved database instead
        of whatever single executor the server was booted with.

        Args:
            database: Resolved database name.

        Returns:
            SQLExecutor: Executor to use for this database.

        Raises:
            DatabaseError: If per-database executors are configured but none
                exists for this database.
        """
        if self.sql_executors is not None:
            executor = self.sql_executors.get(database)
            if executor is None:
                raise DatabaseError(
                    message=f"No SQL executor configured for database '{database}'",
                    details={"database": database},
                )
            return executor
        return self.sql_executor

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method wraps the pipeline with cross-cutting concerns and then
        delegates to :meth:`_execute_query_impl`:

        1. Establish request tracing context (request ID propagation)
        2. Acquire a query rate-limiter slot (reject on timeout)
        3. Run the full pipeline (generation, validation, execution, result
           validation) with per-database security policies
        4. Record metrics for the request outcome
        5. Always release the rate-limiter slot

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.

        Example:
            >>> response = await orchestrator.execute_query(
            ...     QueryRequest(question="Count all users", return_type="result")
            ... )
            >>> if response.success:
            ...     print(f"Found {response.data.row_count} rows")
        """
        request_id = generate_request_id()

        # Rate limiting: reject instead of queueing indefinitely
        if self.rate_limiter is not None:
            acquired = await self.rate_limiter.query_limiter.acquire(
                timeout=self.resilience_config.rate_limit_timeout,
            )
            if not acquired:
                logger.warning(
                    "Query rejected: rate limit exceeded",
                    extra={"request_id": request_id},
                )
                if self.metrics is not None:
                    self.metrics.increment_query_request(
                        status="rate_limited",
                        database=request.database or "unknown",
                    )
                error = RateLimitExceededError(
                    message="Too many concurrent queries, please retry later",
                    details={"timeout_seconds": self.resilience_config.rate_limit_timeout},
                )
                return QueryResponse(
                    success=False,
                    generated_sql=None,
                    validation=None,
                    data=None,
                    error=ErrorDetail(
                        code=error.code.value,
                        message=error.message,
                        details=error.details,
                    ),
                    confidence=0,
                    tokens_used=0,
                )

        database_name = request.database or "unknown"
        start_time = self._get_current_time_ms()
        response: QueryResponse | None = None
        try:
            # request_context propagates request_id to all downstream operations
            async with request_context(request_id):
                resolved: dict[str, str] = {}
                response = await self._execute_query_impl(request, request_id, resolved)
                database_name = resolved.get("database", database_name)
                return response
        finally:
            if self.rate_limiter is not None:
                self.rate_limiter.query_limiter.release()
            if self.metrics is not None:
                if response is not None and response.success:
                    status = "success"
                elif response is not None and response.error is not None:
                    status = response.error.code
                else:
                    status = "error"
                self.metrics.increment_query_request(status=status, database=database_name)
                self.metrics.query_duration.observe(
                    (self._get_current_time_ms() - start_time) / 1000
                )

    async def _execute_query_impl(
        self,
        request: QueryRequest,
        request_id: str,
        resolved: dict[str, str],
    ) -> QueryResponse:
        """Run the query pipeline: resolve database, generate, validate, execute.

        Args:
            request: Query request containing question and parameters.
            request_id: Request ID for tracing.
            resolved: Mutable dict updated with the resolved database name
                (key ``"database"``) once known, for metrics reporting.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.
        """
        try:
            # Step 1: Resolve database name
            database_name = self._resolve_database(request.database)
            resolved["database"] = database_name
            logger.debug(
                "Resolved database",
                extra={"request_id": request_id, "database": database_name},
            )

            # Step 2: Resolve per-database components (security policies)
            validator = self._get_validator(database_name)
            executor = self._get_executor(database_name)

            # Step 3: Get schema from cache
            schema = self.schema_cache.get(database_name)
            if schema is None:
                # Schema not in cache, load it
                pool = self.pools.get(database_name)
                if pool is None:
                    raise DatabaseError(
                        message=f"No connection pool available for database '{database_name}'",
                        details={"database": database_name},
                    )
                try:
                    schema = await self.schema_cache.load(database_name, pool)
                except Exception as e:
                    raise SchemaLoadError(
                        message=f"Failed to load schema for database '{database_name}': {e!s}",
                        details={"database": database_name, "error": str(e)},
                    ) from e

            logger.debug(
                "Schema loaded",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "tables": len(schema.tables),
                },
            )

            # Step 4: Generate and validate SQL with retry logic
            generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
                question=request.question,
                schema=schema,
                request_id=request_id,
                validator=validator,
            )

            # Step 5: If return_type is SQL, return early
            if request.return_type == ReturnType.SQL:
                logger.info(
                    "Returning SQL only",
                    extra={"request_id": request_id, "sql_length": len(generated_sql)},
                )
                return QueryResponse(
                    success=True,
                    generated_sql=generated_sql,
                    validation=validation_result,
                    data=None,
                    error=None,
                    confidence=100,
                    tokens_used=tokens_used,
                )

            # Step 6: Execute SQL
            logger.debug("Executing SQL", extra={"request_id": request_id})
            start_time = self._get_current_time_ms()

            results, total_count = await self._execute_sql(executor, generated_sql)

            execution_time_ms = self._get_current_time_ms() - start_time
            logger.info(
                "SQL executed successfully",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "row_count": total_count,
                    "execution_time_ms": execution_time_ms,
                },
            )

            # Step 7: Validate results (non-blocking, failures don't fail the request)
            result_confidence = await self._validate_results_safely(
                question=request.question,
                sql=generated_sql,
                results=results,
                row_count=total_count,
                request_id=request_id,
            )

            # Step 8: Build successful response
            query_result = QueryResult(
                columns=list(results[0].keys()) if results else [],
                rows=results,
                row_count=len(results),  # Limited row count (after max_rows applied)
                execution_time_ms=execution_time_ms,
            )

            return QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=query_result,
                error=None,
                confidence=result_confidence,
                tokens_used=tokens_used,
            )

        except PgMcpError as e:
            # Handle known application errors
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "error_code": e.code,
                    "error_message": str(e),
                },
            )
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=e.code.value,
                    message=e.message,
                    details=e.details,
                ),
                confidence=0,
                tokens_used=None,
            )
        except Exception as e:
            # Handle unexpected errors
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id},
            )
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR.value,
                    message=f"Internal server error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
                confidence=0,
                tokens_used=None,
            )

    async def _execute_sql(
        self,
        executor: SQLExecutor,
        sql: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """Execute SQL with database query duration metrics.

        Args:
            executor: The executor bound to the target database.
            sql: Validated SQL query.

        Returns:
            tuple: (results, total_row_count).
        """
        if self.metrics is not None:
            start = time.perf_counter()
            results, total_count = await executor.execute(sql)
            self.metrics.observe_db_query_duration(time.perf_counter() - start)
            return results, total_count
        return await executor.execute(sql)

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.

        Example:
            >>> name = orchestrator._resolve_database("mydb")  # Validates "mydb" exists
            >>> name = orchestrator._resolve_database(None)  # Auto-selects if only one DB
        """
        if database is not None:
            # Validate specified database exists
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        # Auto-select if only one database available
        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
        validator: SQLValidator | None = None,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic on validation failures.

        This method implements a retry loop that:
        1. Checks circuit breaker state
        2. Acquires an LLM rate-limiter slot
        3. Generates SQL using LLM (recording call counts and latency)
        4. Validates the generated SQL using the provided (per-database) validator
        5. On validation failure, backs off exponentially and retries with error feedback
        6. Records success/failure to circuit breaker

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.
            validator: Validator enforcing the target database's security
                policy. Defaults to the shared validator.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed.

        Example:
            >>> sql, validation, tokens = await orchestrator._generate_sql_with_retry(
            ...     question="Count users",
            ...     schema=db_schema,
            ...     request_id="123",
            ... )
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        effective_validator = validator or self.sql_validator
        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                # Generate SQL, guarded by the LLM rate limiter
                if self.metrics is not None:
                    self.metrics.increment_llm_call("generate_sql")

                llm_start = time.perf_counter()
                try:
                    if self.rate_limiter is not None:
                        async with self.rate_limiter.for_llm(
                            timeout=self.resilience_config.rate_limit_timeout
                        ):
                            generated_sql = await self.sql_generator.generate(
                                question=question,
                                schema=schema,
                                previous_attempt=previous_sql,
                                error_feedback=error_feedback,
                            )
                    else:
                        generated_sql = await self.sql_generator.generate(
                            question=question,
                            schema=schema,
                            previous_attempt=previous_sql,
                            error_feedback=error_feedback,
                        )
                except TimeoutError as e:
                    raise RateLimitExceededError(
                        message="LLM call rejected: rate limit exceeded",
                        details={"timeout_seconds": self.resilience_config.rate_limit_timeout},
                    ) from e

                if self.metrics is not None:
                    self.metrics.observe_llm_latency(
                        "generate_sql", time.perf_counter() - llm_start
                    )
                    if tokens_used is not None:
                        self.metrics.increment_llm_tokens("generate_sql", tokens_used)

                # Note: tokens_used would come from OpenAI response metadata if available
                # For now, we don't extract it, but it can be added later

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL against the target database's security policy
                try:
                    effective_validator.validate_or_raise(generated_sql)
                except (SecurityViolationError, SQLParseError) as validation_error:
                    if self.metrics is not None:
                        self.metrics.increment_sql_rejected(
                            validation_error.code.value
                        )
                    if attempt < max_retries:
                        # Record as failure, back off, and retry with feedback
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)

                        # Exponential backoff between attempts
                        delay = self.resilience_config.retry_delay * (
                            self.resilience_config.backoff_factor**attempt
                        )
                        logger.debug(
                            "Backing off before retry",
                            extra={"request_id": request_id, "delay_seconds": delay},
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Out of retries, record failure and raise
                        self.circuit_breaker.record_failure()
                        logger.error(
                            "SQL validation failed after all retries",
                            extra={
                                "request_id": request_id,
                                "attempts": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build validation result
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=True,
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_used

            except (LLMError, RateLimitExceededError, SecurityViolationError, SQLParseError):
                # Re-raise known errors
                raise
            except Exception as e:
                # Unexpected error during generation
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        # Should not reach here, but just in case
        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using LLM, but failures
        don't cause the overall query to fail. Returns a confidence score.

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100). Returns 100 if validation disabled/fails.

        Example:
            >>> confidence = await orchestrator._validate_results_safely(
            ...     question="Count users",
            ...     sql="SELECT COUNT(*) FROM users",
            ...     results=[{"count": 42}],
            ...     row_count=1,
            ...     request_id="123",
            ... )
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            validation_result = await self.result_validator.validate(
                question=question,
                sql=sql,
                results=results,
                row_count=row_count,
            )

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence

        except Exception as e:
            # Log but don't fail the query
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            return 100  # Default to high confidence if validation fails

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds.

        Returns:
            float: Current time in milliseconds since epoch.
        """
        return time.time() * 1000
