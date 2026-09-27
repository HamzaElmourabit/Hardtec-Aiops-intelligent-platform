import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.config import get_snowflake_config


def test_get_snowflake_config_requires_required_vars():
    required_vars = (
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
    )
    environment = {key: value for key, value in os.environ.items() if key not in required_vars}

    with patch.dict(os.environ, environment, clear=True):
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "Missing required Snowflake environment variables"
        ):
            get_snowflake_config()


def test_get_snowflake_config_reads_values_from_environment():
    environment = {
        "SNOWFLAKE_ACCOUNT": "demo_account",
        "SNOWFLAKE_USER": "demo_user",
        "SNOWFLAKE_PASSWORD": "demo_password",
        "SNOWFLAKE_WAREHOUSE": "COMPUTE_WH",
        "SNOWFLAKE_DATABASE": "HARDTEC_DB",
        "SNOWFLAKE_SCHEMA": "PUBLIC",
        "SNOWFLAKE_ROLE": "ACCOUNTADMIN",
    }

    with patch.dict(os.environ, environment, clear=False):
        config = get_snowflake_config()

    assert config["account"] == "demo_account"
    assert config["user"] == "demo_user"
    assert config["password"] == "demo_password"
    assert config["warehouse"] == "COMPUTE_WH"


def test_streamlit_connection_uses_environment_values():
    environment = {
        "SNOWFLAKE_ACCOUNT": "demo_account",
        "SNOWFLAKE_USER": "demo_user",
        "SNOWFLAKE_PASSWORD": "demo_password",
        "SNOWFLAKE_WAREHOUSE": "COMPUTE_WH",
        "SNOWFLAKE_DATABASE": "HARDTEC_DB",
        "SNOWFLAKE_SCHEMA": "PUBLIC",
        "SNOWFLAKE_ROLE": "ACCOUNTADMIN",
    }

    module_path = Path(__file__).resolve().parents[1] / "streamlit" / "snowflake_connection.py"
    spec = importlib.util.spec_from_file_location("local_streamlit_snowflake_connection", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with patch.dict(os.environ, environment, clear=True):
        with patch("snowflake.connector.connect") as mock_connect:
            module.get_connection()

    mock_connect.assert_called_once_with(
        account="demo_account",
        user="demo_user",
        password="demo_password",
        warehouse="COMPUTE_WH",
        database="HARDTEC_DB",
        schema="PUBLIC",
        role="ACCOUNTADMIN",
    )


def test_streamlit_prediction_query_reads_sqlite_history(tmp_path):
    streamlit_dir = Path(__file__).resolve().parents[1] / "streamlit"
    connection_module_path = streamlit_dir / "snowflake_connection.py"
    connection_spec = importlib.util.spec_from_file_location(
        "snowflake_connection",
        connection_module_path,
    )
    connection_module = importlib.util.module_from_spec(connection_spec)
    sys.modules["snowflake_connection"] = connection_module
    connection_spec.loader.exec_module(connection_module)

    query_module_path = streamlit_dir / "queries.py"
    spec = importlib.util.spec_from_file_location("local_streamlit_queries", query_module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_streamlit_queries"] = module
    spec.loader.exec_module(module)

    sqlite_path = tmp_path / "ticket_predictions.db"
    sqlite_path.touch()
    mock_conn = unittest.mock.Mock()
    with patch.object(module, "SQLITE_PATH", sqlite_path):
        with patch.object(
            module,
            "get_sqlite_connection",
            return_value=mock_conn,
        ) as mock_get_connection:
            with patch(
                "pandas.read_sql_query",
                return_value=pd.DataFrame(),
            ) as mock_read_sql:
                module.get_ai_predictions()

    mock_get_connection.assert_called_once()
    mock_conn.close.assert_called_once()
    assert "FROM ticket_predictions" in mock_read_sql.call_args[0][0]
