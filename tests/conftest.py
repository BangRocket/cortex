"""Pytest configuration and fixtures."""

import pytest


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
