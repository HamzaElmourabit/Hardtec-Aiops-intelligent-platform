import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from src.api.security import enforce_api_key, validate_api_configuration


def make_request(path="/predict", api_key=None):
    headers = []
    if api_key is not None:
        headers.append((b"x-api-key", api_key.encode()))

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers,
        "query_string": b"",
        "server": ("testserver", 80),
        "scheme": "http",
        "client": ("testclient", 50000),
    }
    return Request(scope)


class SecurityTests(unittest.TestCase):
    def test_authentication_is_disabled_by_default(self):
        with patch.dict(os.environ, {"API_AUTH_ENABLED": "false"}, clear=False):
            enforce_api_key(make_request())

    def test_invalid_key_is_rejected(self):
        environment = {
            "API_AUTH_ENABLED": "true",
            "API_SECRET_KEY": "test-secret",
        }
        with patch.dict(os.environ, environment, clear=False):
            with self.assertRaises(HTTPException) as context:
                enforce_api_key(make_request(api_key="wrong-secret"))
            self.assertEqual(context.exception.status_code, 401)

    def test_valid_key_is_accepted(self):
        environment = {
            "API_AUTH_ENABLED": "true",
            "API_SECRET_KEY": "test-secret",
        }
        with patch.dict(os.environ, environment, clear=False):
            enforce_api_key(make_request(api_key="test-secret"))

    def test_production_auth_rejects_placeholder_secret(self):
        environment = {
            "API_AUTH_ENABLED": "true",
            "API_SECRET_KEY": "your_super_secret_key_change_this_in_production",
        }
        with patch.dict(os.environ, environment, clear=False):
            with self.assertRaises(RuntimeError):
                validate_api_configuration()


if __name__ == "__main__":
    unittest.main()