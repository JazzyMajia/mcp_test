"""Unit tests for configuration management.

Tests for all configuration classes to ensure proper validation,
defaults, and environment variable parsing.
"""

import os

import pytest
from pydantic import ValidationError

from pg_mcp.config.settings import (
    CacheConfig,
    DatabaseConfig,
    DatabaseSecurityPolicy,
    ObservabilityConfig,
    OpenAIConfig,
    ResilienceConfig,
    SecurityConfig,
    Settings,
    ValidationConfig,
    get_settings,
    reset_settings,
)


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "postgres"
        assert config.user == "postgres"
        assert config.min_pool_size == 5
        assert config.max_pool_size == 20

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            name="mydb",
            user="myuser",
            password="secret",
        )
        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.name == "mydb"
        assert config.user == "myuser"
        assert config.password == "secret"

    def test_dsn_generation(self) -> None:
        """Test DSN string generation."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            name="testdb",
            user="testuser",
            password="testpass",
        )
        dsn = config.dsn
        assert dsn == "postgresql://testuser:testpass@localhost:5432/testdb"

    def test_safe_dsn_masks_password(self) -> None:
        """Test safe DSN masks password."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            name="testdb",
            user="testuser",
            password="secret123",
        )
        safe_dsn = config.safe_dsn
        assert "secret123" not in safe_dsn
        assert "***" in safe_dsn
        assert "testuser" in safe_dsn

    def test_invalid_port(self) -> None:
        """Test invalid port number is rejected."""
        with pytest.raises(ValidationError):
            DatabaseConfig(port=0)

        with pytest.raises(ValidationError):
            DatabaseConfig(port=99999)

    def test_invalid_pool_size(self) -> None:
        """Test invalid pool size is rejected."""
        with pytest.raises(ValidationError):
            DatabaseConfig(min_pool_size=0)

        with pytest.raises(ValidationError):
            DatabaseConfig(max_pool_size=101)


class TestOpenAIConfig:
    """Tests for OpenAIConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = OpenAIConfig(api_key="sk-test123")
        assert config.model == "gpt-4o-mini"
        assert config.max_tokens == 2000
        assert config.temperature == 0.0
        assert config.timeout == 30.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = OpenAIConfig(
            api_key="sk-custom",
            model="gpt-4",
            max_tokens=4000,
            temperature=0.7,
            timeout=60.0,
        )
        assert config.model == "gpt-4"
        assert config.max_tokens == 4000
        assert config.temperature == 0.7
        assert config.timeout == 60.0

    def test_empty_api_key_rejected(self) -> None:
        """Test empty API key is rejected."""
        with pytest.raises(ValidationError, match="must not be empty"):
            OpenAIConfig(api_key="")

    def test_whitespace_api_key_rejected(self) -> None:
        """Test whitespace-only API key is rejected."""
        with pytest.raises(ValidationError, match="must not be empty"):
            OpenAIConfig(api_key="   ")

    def test_invalid_api_key_format(self) -> None:
        """Test API key must start with sk-."""
        with pytest.raises(ValidationError, match="must start with 'sk-'"):
            OpenAIConfig(api_key="invalid-key")

    def test_invalid_max_tokens(self) -> None:
        """Test invalid max_tokens is rejected."""
        with pytest.raises(ValidationError):
            OpenAIConfig(api_key="sk-test", max_tokens=50)

        with pytest.raises(ValidationError):
            OpenAIConfig(api_key="sk-test", max_tokens=5000)

    def test_invalid_temperature(self) -> None:
        """Test invalid temperature is rejected."""
        with pytest.raises(ValidationError):
            OpenAIConfig(api_key="sk-test", temperature=-0.1)

        with pytest.raises(ValidationError):
            OpenAIConfig(api_key="sk-test", temperature=2.1)


class TestSecurityConfig:
    """Tests for SecurityConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SecurityConfig()
        assert config.allow_write_operations is False
        assert config.max_rows == 10000
        assert config.max_execution_time == 30.0
        assert "pg_sleep" in config.blocked_functions
        assert "pg_read_file" in config.blocked_functions

    def test_custom_blocked_functions(self) -> None:
        """Test custom blocked functions."""
        config = SecurityConfig(
            blocked_functions=["func1", "func2"],
        )
        assert config.blocked_functions == ["func1", "func2"]

    def test_parse_blocked_functions_from_string(self) -> None:
        """Test parsing blocked functions from comma-separated string."""
        config = SecurityConfig(
            blocked_functions="func1, func2, func3",  # type: ignore
        )
        assert "func1" in config.blocked_functions
        assert "func2" in config.blocked_functions
        assert "func3" in config.blocked_functions

    def test_allow_write_operations(self) -> None:
        """Test enabling write operations."""
        config = SecurityConfig(allow_write_operations=True)
        assert config.allow_write_operations is True

    def test_invalid_max_rows(self) -> None:
        """Test invalid max_rows is rejected."""
        with pytest.raises(ValidationError):
            SecurityConfig(max_rows=0)

        with pytest.raises(ValidationError):
            SecurityConfig(max_rows=100001)


class TestValidationConfig:
    """Tests for ValidationConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ValidationConfig()
        assert config.max_question_length == 10000
        assert config.confidence_threshold == 70

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ValidationConfig(
            max_question_length=5000,
            confidence_threshold=80,
        )
        assert config.max_question_length == 5000
        assert config.confidence_threshold == 80

    def test_invalid_confidence_threshold(self) -> None:
        """Test invalid confidence threshold is rejected."""
        with pytest.raises(ValidationError):
            ValidationConfig(confidence_threshold=-1)

        with pytest.raises(ValidationError):
            ValidationConfig(confidence_threshold=101)


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CacheConfig()
        assert config.schema_ttl == 3600
        assert config.max_size == 100
        assert config.enabled is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CacheConfig(
            schema_ttl=7200,
            max_size=200,
            enabled=False,
        )
        assert config.schema_ttl == 7200
        assert config.max_size == 200
        assert config.enabled is False

    def test_invalid_ttl(self) -> None:
        """Test invalid TTL is rejected."""
        with pytest.raises(ValidationError):
            CacheConfig(schema_ttl=30)

        with pytest.raises(ValidationError):
            CacheConfig(schema_ttl=90000)


class TestResilienceConfig:
    """Tests for ResilienceConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ResilienceConfig()
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.backoff_factor == 2.0
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_timeout == 60.0
        assert config.max_concurrent_queries == 10
        assert config.max_concurrent_llm_calls == 5
        assert config.rate_limit_timeout == 5.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ResilienceConfig(
            max_retries=5,
            retry_delay=2.0,
            backoff_factor=3.0,
            max_concurrent_queries=20,
            max_concurrent_llm_calls=8,
            rate_limit_timeout=2.5,
        )
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.backoff_factor == 3.0
        assert config.max_concurrent_queries == 20
        assert config.max_concurrent_llm_calls == 8
        assert config.rate_limit_timeout == 2.5

    def test_invalid_values(self) -> None:
        """Test invalid values are rejected."""
        with pytest.raises(ValidationError):
            ResilienceConfig(max_retries=-1)

        with pytest.raises(ValidationError):
            ResilienceConfig(backoff_factor=0.5)

        with pytest.raises(ValidationError):
            ResilienceConfig(max_concurrent_queries=0)

        with pytest.raises(ValidationError):
            ResilienceConfig(rate_limit_timeout=0)


class TestDatabaseSecurityPolicy:
    """Tests for DatabaseSecurityPolicy."""

    def test_default_values(self) -> None:
        """Test default policy allows everything the global config allows."""
        policy = DatabaseSecurityPolicy()
        assert policy.blocked_tables == []
        assert policy.blocked_columns == []
        assert policy.allow_explain is False

    def test_csv_parsing(self) -> None:
        """Test comma-separated string parsing for blocked lists."""
        policy = DatabaseSecurityPolicy(
            blocked_tables="secrets, audit_log",
            blocked_columns="password",
        )
        assert policy.blocked_tables == ["secrets", "audit_log"]
        assert policy.blocked_columns == ["password"]


class TestSecurityConfigPolicies:
    """Tests for SecurityConfig per-database policy resolution."""

    def test_default_no_policies(self) -> None:
        """Test policy_for with no per-database overrides."""
        config = SecurityConfig()
        policy = config.policy_for("anydb")
        assert policy.blocked_tables == []
        assert policy.blocked_columns == []
        assert policy.allow_explain is False

    def test_global_lists_apply_to_all_databases(self) -> None:
        """Test global blocked lists apply to every database."""
        config = SecurityConfig(blocked_tables="secrets", allow_explain=True)
        policy = config.policy_for("db1")
        assert policy.blocked_tables == ["secrets"]
        assert policy.allow_explain is True

    def test_per_database_policy_is_additive(self) -> None:
        """Test per-database policy extends the global lists."""
        config = SecurityConfig(
            blocked_tables=["secrets"],
            database_policies={
                "analytics": DatabaseSecurityPolicy(blocked_tables=["raw_events"]),
            },
        )
        policy = config.policy_for("analytics")
        assert sorted(policy.blocked_tables) == ["raw_events", "secrets"]
        # Other databases keep only the global policy
        assert config.policy_for("otherdb").blocked_tables == ["secrets"]

    def test_per_database_allow_explain_override(self) -> None:
        """Test per-database allow_explain overrides the global value."""
        config = SecurityConfig(
            allow_explain=False,
            database_policies={
                "devdb": DatabaseSecurityPolicy(allow_explain=True),
            },
        )
        assert config.policy_for("devdb").allow_explain is True
        assert config.policy_for("otherdb").allow_explain is False

    def test_database_policies_from_json_string(self) -> None:
        """Test database_policies parses from a JSON string (env var)."""
        import json

        raw = json.dumps(
            {
                "analytics": {"blocked_tables": ["internal"], "allow_explain": True},
            }
        )
        config = SecurityConfig(database_policies=raw)
        assert "analytics" in config.database_policies
        policy = config.policy_for("analytics")
        assert policy.blocked_tables == ["internal"]
        assert policy.allow_explain is True

    def test_invalid_database_policies_rejected(self) -> None:
        """Test invalid database_policies values are rejected."""
        with pytest.raises(ValidationError):
            SecurityConfig(database_policies=[{"name": "x"}])

    def test_allow_write_operations_flag(self) -> None:
        """Test allow_write_operations is a wired config field."""
        config = SecurityConfig(allow_write_operations=True)
        assert config.allow_write_operations is True


class TestObservabilityConfig:
    """Tests for ObservabilityConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ObservabilityConfig()
        # metrics_enabled 在测试环境可能被禁用以避免启动 HTTP 服务器
        # 生产环境应该通过环境变量显式设置
        assert config.metrics_port == 9090
        assert config.log_level == "INFO"
        assert config.log_format == "text"

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ObservabilityConfig(
            metrics_enabled=False,
            metrics_port=8080,
            log_level="DEBUG",
            log_format="text",
        )
        assert config.metrics_enabled is False
        assert config.metrics_port == 8080
        assert config.log_level == "DEBUG"
        assert config.log_format == "text"

    def test_invalid_log_level(self) -> None:
        """Test invalid log level is rejected."""
        with pytest.raises(ValidationError):
            ObservabilityConfig(log_level="INVALID")  # type: ignore

    def test_invalid_log_format(self) -> None:
        """Test invalid log format is rejected."""
        with pytest.raises(ValidationError):
            ObservabilityConfig(log_format="xml")  # type: ignore


class TestSettings:
    """Tests for main Settings class."""

    def test_default_settings(self) -> None:
        """Test default settings initialization."""
        settings = Settings(openai=OpenAIConfig(api_key="sk-test"))
        assert settings.environment == "development"
        assert settings.database is not None
        assert settings.openai is not None
        assert settings.security is not None
        assert settings.validation is not None
        assert settings.cache is not None
        assert settings.resilience is not None
        assert settings.observability is not None

    def test_is_production(self) -> None:
        """Test production environment check."""
        settings = Settings(
            environment="production",
            openai=OpenAIConfig(api_key="sk-test"),
        )
        assert settings.is_production
        assert not settings.is_development

    def test_is_development(self) -> None:
        """Test development environment check."""
        settings = Settings(
            environment="development",
            openai=OpenAIConfig(api_key="sk-test"),
        )
        assert settings.is_development
        assert not settings.is_production

    def test_nested_config_override(self) -> None:
        """Test overriding nested configurations."""
        settings = Settings(
            openai=OpenAIConfig(api_key="sk-test"),
            database=DatabaseConfig(
                host="custom.host",
                port=5433,
            ),
            security=SecurityConfig(
                allow_write_operations=True,
            ),
        )
        assert settings.database.host == "custom.host"
        assert settings.database.port == 5433
        assert settings.security.allow_write_operations is True

    def test_databases_from_json_list(self) -> None:
        """Test DATABASES as a JSON array of database configs."""
        import json

        raw = json.dumps(
            [
                {"name": "analytics", "host": "analytics.example.com"},
                {"name": "billing", "host": "billing.example.com", "port": 5433},
            ]
        )
        settings = Settings(
            openai=OpenAIConfig(api_key="sk-test"),
            databases=raw,
        )
        assert set(settings.databases) == {"analytics", "billing"}
        assert settings.databases["analytics"].host == "analytics.example.com"
        assert settings.databases["billing"].port == 5433

    def test_databases_from_json_object(self) -> None:
        """Test DATABASES as a JSON object keyed by database name."""
        import json

        raw = json.dumps(
            {
                "analytics": {"host": "analytics.example.com"},
                "billing": {"host": "billing.example.com"},
            }
        )
        settings = Settings(
            openai=OpenAIConfig(api_key="sk-test"),
            databases=raw,
        )
        assert set(settings.databases) == {"analytics", "billing"}
        assert settings.databases["analytics"].name == "analytics"

    def test_all_database_configs_merges_primary(self) -> None:
        """Test all_database_configs merges primary database with extras."""
        settings = Settings(
            openai=OpenAIConfig(api_key="sk-test"),
            database=DatabaseConfig(name="main", host="main.example.com"),
            databases=[{"name": "analytics", "host": "analytics.example.com"}],
        )
        configs = settings.all_database_configs
        assert set(configs) == {"main", "analytics"}
        assert configs["main"].host == "main.example.com"
        assert configs["analytics"].host == "analytics.example.com"

    def test_all_database_configs_primary_wins_on_conflict(self) -> None:
        """Test primary database config wins over a same-name DATABASES entry."""
        settings = Settings(
            openai=OpenAIConfig(api_key="sk-test"),
            database=DatabaseConfig(name="main", host="primary.example.com"),
            databases=[{"name": "main", "host": "override.example.com"}],
        )
        configs = settings.all_database_configs
        assert configs["main"].host == "primary.example.com"

    def test_empty_database_name_rejected(self) -> None:
        """Test empty database names in databases are rejected."""
        with pytest.raises(ValidationError):
            Settings(
                openai=OpenAIConfig(api_key="sk-test"),
                databases={"": {"host": "h"}},
            )


class TestSettingsGlobalInstance:
    """Tests for global settings instance management."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_settings()
        # Clean up environment variables
        for key in list(os.environ.keys()):
            if key.startswith(("DATABASE_", "OPENAI_", "SECURITY_")):
                del os.environ[key]

    def test_get_settings_creates_instance(self) -> None:
        """Test get_settings creates instance."""
        # Set required env var
        os.environ["OPENAI_API_KEY"] = "sk-test123"

        settings = get_settings()
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_get_settings_returns_same_instance(self) -> None:
        """Test get_settings returns singleton."""
        os.environ["OPENAI_API_KEY"] = "sk-test123"

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_reset_settings(self) -> None:
        """Test reset_settings clears instance."""
        os.environ["OPENAI_API_KEY"] = "sk-test123"

        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()
        assert settings1 is not settings2

    def test_settings_from_environment(self) -> None:
        """Test loading settings from environment variables."""
        os.environ["OPENAI_API_KEY"] = "sk-env-key"
        os.environ["OPENAI_MODEL"] = "gpt-4"
        os.environ["DATABASE_HOST"] = "env.host.com"
        os.environ["SECURITY_MAX_ROWS"] = "5000"

        reset_settings()
        settings = get_settings()

        # Use get_secret_value() to access SecretStr content
        assert settings.openai.api_key.get_secret_value() == "sk-env-key"
        assert settings.openai.model == "gpt-4"
        assert settings.database.host == "env.host.com"
        assert settings.security.max_rows == 5000
