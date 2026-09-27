import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_ENV = PROJECT_ROOT / "streamlit" / ".env"
ROOT_ENV = PROJECT_ROOT / ".env"

PLACEHOLDER_VALUES = {
    "your_account_name",
    "your_username",
    "your_password",
    "ton_user",
    "ton_password",
    "account",
    "user",
    "password",
}


for env_file in (ROOT_ENV, STREAMLIT_ENV):
    if env_file.exists():
        load_dotenv(env_file, override=(env_file == STREAMLIT_ENV))


def _is_placeholder_value(value: str | None) -> bool:
    if value is None:
        return True

    normalized = value.strip().lower()
    return not normalized or normalized in PLACEHOLDER_VALUES or normalized.startswith("your_") or normalized.startswith("ton_")


def get_snowflake_config() -> Dict[str, str]:
    """Return validated Snowflake configuration from environment variables."""
    required_vars = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise RuntimeError(
            "Missing required Snowflake environment variables: "
            + ", ".join(missing_vars)
        )

    config = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "HARDTEC_DB"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
        "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
    }

    invalid_values = [
        name for name, value in {
            "account": config["account"],
            "user": config["user"],
            "password": config["password"],
        }.items() if _is_placeholder_value(value)
    ]

    if invalid_values:
        raise RuntimeError(
            "Snowflake credentials in the environment are still placeholders. "
            "Set the real SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, and SNOWFLAKE_PASSWORD "
            "values in the active environment or in streamlit/.env."
        )

    return config
