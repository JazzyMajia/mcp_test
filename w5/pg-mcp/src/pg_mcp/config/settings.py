"""Configuration management for PostgreSQL MCP Server.

This module defines all configuration settings using Pydantic for validation
and type safety. Configuration is loaded from environment variables with
sensible defaults.

Multi-database support:
    Besides the primary database (``DATABASE_*`` environment variables),
    additional databases can be configured via the ``DATABASES`` environment
    variable containing a JSON array (or object keyed by name) of database
    configs, e.g.::

        DATABASES='[{"name": "analytics", "host": "db.example.com"}, ...]'
"""

import json
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """PostgreSQL database connection configuration."""

    model_config = SettingsConfigDict(env_prefix="DATABASE_")

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="postgres", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")

    # Connection pool settings
    min_pool_size: int = Field(default=5, ge=1, le=100, description="Minimum pool size")
    max_pool_size: int = Field(default=20, ge=1, le=100, description="Maximum pool size")
    pool_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Pool acquire timeout in seconds"
    )
    command_timeout: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Command execution timeout in seconds"
    )

    @property
    def dsn(self) -> str:
        """Build PostgreSQL DSN connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def safe_dsn(self) -> str:
        """Build DSN with masked password for logging."""
        return f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.name}"


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_")

    api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    model: str = Field(default="gpt-4o-mini", description="Model to use for SQL generation")
    max_tokens: int = Field(default=2000, ge=100, le=4096, description="Maximum tokens in response")
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Temperature for response randomness"
    )
    timeout: float = Field(
        default=30.0, ge=5.0, le=120.0, description="API request timeout in seconds"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: SecretStr) -> SecretStr:
        """Validate API key is not empty and has correct format."""
        api_key_str = v.get_secret_value()
        if not api_key_str or not api_key_str.strip():
            raise ValueError("OpenAI API key must not be empty")
        if not api_key_str.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v


class DatabaseSecurityPolicy(BaseSettings):
    """Per-database security policy.

    Policies narrow access for a specific database on top of the global
    :class:`SecurityConfig`: blocked tables/columns are additive (union with
    the global lists) and ``allow_explain`` can only tighten or relax the
    EXPLAIN policy for that database.
    """

    model_config = SettingsConfigDict(env_prefix="SECURITY_DB_")

    blocked_tables: list[str] = Field(
        default_factory=list,
        description="Additional table names blocked for this database",
    )
    blocked_columns: list[str] = Field(
        default_factory=list,
        description="Additional column names blocked for this database",
    )
    allow_explain: bool = Field(
        default=False, description="Whether EXPLAIN statements are allowed for this database"
    )

    @field_validator("blocked_tables", "blocked_columns", mode="before")
    @classmethod
    def parse_csv_list(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string or list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


class SecurityConfig(BaseSettings):
    """Security and access control configuration."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_")

    allow_write_operations: bool = Field(
        default=False, description="Allow write operations (INSERT, UPDATE, DELETE)"
    )
    blocked_functions: list[str] = Field(
        default_factory=lambda: [
            "pg_sleep",
            "pg_read_file",
            "pg_write_file",
            "lo_import",
            "lo_export",
        ],
        description="List of blocked PostgreSQL functions",
    )
    blocked_tables: list[str] = Field(
        default_factory=list,
        description="Table names that must never be accessed (applies to all databases)",
    )
    blocked_columns: list[str] = Field(
        default_factory=list,
        description="Column names that must never be accessed (applies to all databases)",
    )
    allow_explain: bool = Field(
        default=False, description="Whether EXPLAIN statements are allowed"
    )
    database_policies: dict[str, DatabaseSecurityPolicy] = Field(
        default_factory=dict,
        description=(
            "Per-database security policies keyed by database name. "
            "Accepts a JSON object via SECURITY_DATABASE_POLICIES, e.g. "
            '\'{"analytics": {"blocked_tables": ["secrets"], "allow_explain": true}}\''
        ),
    )
    max_rows: int = Field(default=10000, ge=1, le=100000, description="Maximum rows to return")
    max_execution_time: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Maximum query execution time in seconds"
    )
    readonly_role: str | None = Field(
        default=None, description="PostgreSQL role to switch to for read-only access"
    )
    safe_search_path: str = Field(
        default="public", description="Safe search_path to set during query execution"
    )

    @field_validator("blocked_functions", "blocked_tables", "blocked_columns", mode="before")
    @classmethod
    def parse_blocked_list(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated string or list."""
        if isinstance(v, str):
            return [f.strip() for f in v.split(",") if f.strip()]
        return v

    @field_validator("database_policies", mode="before")
    @classmethod
    def parse_database_policies(
        cls, v: dict[str, DatabaseSecurityPolicy] | str | None
    ) -> dict[str, Any]:
        """Parse JSON string or plain dicts into DatabaseSecurityPolicy entries."""
        if v is None:
            return {}
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, dict):
            return {
                name: (
                    policy
                    if isinstance(policy, DatabaseSecurityPolicy)
                    else DatabaseSecurityPolicy(**policy)
                )
                for name, policy in v.items()
            }
        raise ValueError(
            "database_policies must be a mapping of database name to policy, "
            f"got {type(v).__name__}"
        )

    def policy_for(self, database: str) -> DatabaseSecurityPolicy:
        """Get the effective security policy for a database.

        The per-database policy (if any) is combined with the global config:
        blocked tables/columns are unioned, and ``allow_explain`` comes from
        the per-database policy when one exists, otherwise from the global
        setting.

        Args:
            database: Database name.

        Returns:
            DatabaseSecurityPolicy: Effective policy for the database.
        """
        override = self.database_policies.get(database)
        return DatabaseSecurityPolicy(
            blocked_tables=[*self.blocked_tables, *(override.blocked_tables if override else [])],
            blocked_columns=[
                *self.blocked_columns,
                *(override.blocked_columns if override else []),
            ],
            allow_explain=override.allow_explain if override else self.allow_explain,
        )


class ValidationConfig(BaseSettings):
    """Query validation configuration."""

    model_config = SettingsConfigDict(env_prefix="VALIDATION_")

    max_question_length: int = Field(
        default=10000, ge=1, le=50000, description="Maximum question length in characters"
    )

    # Result validation settings
    enabled: bool = Field(default=True, description="Enable result validation using LLM")
    sample_rows: int = Field(
        default=5, ge=1, le=100, description="Number of sample rows to include in validation"
    )
    timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=60.0, description="Result validation timeout in seconds"
    )
    confidence_threshold: int = Field(
        default=70, ge=0, le=100, description="Minimum confidence for acceptable results"
    )


class CacheConfig(BaseSettings):
    """Schema cache configuration."""

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    schema_ttl: int = Field(
        default=3600, ge=60, le=86400, description="Schema cache TTL in seconds"
    )
    max_size: int = Field(default=100, ge=1, le=1000, description="Maximum cache entries")
    enabled: bool = Field(default=True, description="Enable schema caching")


class ResilienceConfig(BaseSettings):
    """Resilience and fault tolerance configuration."""

    model_config = SettingsConfigDict(env_prefix="RESILIENCE_")

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Initial retry delay in seconds"
    )
    backoff_factor: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Exponential backoff factor"
    )
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, le=100, description="Failures before circuit opens"
    )
    circuit_breaker_timeout: float = Field(
        default=60.0, ge=10.0, le=300.0, description="Circuit breaker timeout in seconds"
    )
    max_concurrent_queries: int = Field(
        default=10, ge=1, le=1000, description="Maximum concurrent query requests"
    )
    max_concurrent_llm_calls: int = Field(
        default=5, ge=1, le=1000, description="Maximum concurrent LLM API calls"
    )
    rate_limit_timeout: float = Field(
        default=5.0, ge=0.1, le=300.0,
        description="Max seconds to wait for a rate limiter slot before rejecting",
    )


class ObservabilityConfig(BaseSettings):
    """Observability and monitoring configuration."""

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_")

    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(
        default=9090, ge=1024, le=65535, description="Metrics HTTP server port"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(default="text", description="Log format")


class Settings(BaseSettings):
    """Main application settings aggregating all config sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )

    # Nested configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    databases: dict[str, DatabaseConfig] = Field(
        default_factory=dict,
        description=(
            "Additional database configurations keyed by name. Loaded from the "
            "DATABASES environment variable as a JSON array of configs or a JSON "
            "object keyed by name, e.g. "
            '\'[{"name": "analytics", "host": "db.example.com"}]\''
        ),
    )
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @field_validator("databases", mode="before")
    @classmethod
    def parse_databases(
        cls, v: dict[str, DatabaseConfig] | list[dict[str, Any]] | str | None
    ) -> dict[str, Any]:
        """Parse the DATABASES environment variable (JSON) or structured input.

        Accepts a JSON string (array of database configs, or object keyed by
        name), a list of config dicts, or a dict of already-built configs.
        """
        if v is None:
            return {}
        if isinstance(v, str):
            v = json.loads(v)

        if isinstance(v, list):
            result: dict[str, Any] = {}
            for item in v:
                config = item if isinstance(item, DatabaseConfig) else DatabaseConfig(**item)
                result[config.name] = config
            return result

        if isinstance(v, dict):
            return {
                name: (
                    config
                    if isinstance(config, DatabaseConfig)
                    else DatabaseConfig(**{**config, "name": name})
                )
                for name, config in v.items()
            }

        raise ValueError(f"databases must be a list or mapping, got {type(v).__name__}")

    @model_validator(mode="after")
    def validate_databases(self) -> "Settings":
        """Ensure database names are unique and non-empty across all sources."""
        for name in self.databases:
            if not name or not name.strip():
                raise ValueError("Database names must not be empty")
        return self

    @property
    def all_database_configs(self) -> dict[str, DatabaseConfig]:
        """Get all configured databases keyed by name.

        Merges the primary ``database`` config with the additional
        ``databases`` entries. On a name conflict the primary config wins.

        Returns:
            dict[str, DatabaseConfig]: Mapping of database name to config.
        """
        merged = dict(self.databases)
        merged[self.database.name] = self.database
        return merged

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create global settings instance.

    Returns:
        Settings: The global settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset global settings instance. Useful for testing."""
    global _settings
    _settings = None
