import os

import snowflake.connector

from src.config import get_snowflake_config


def get_connection():
    config = get_snowflake_config()

    return snowflake.connector.connect(
        account=config["account"],
        user=config["user"],
        password=config["password"],
        warehouse=config["warehouse"],
        database=config["database"],
        schema=config["schema"],
        role=config["role"],
    )