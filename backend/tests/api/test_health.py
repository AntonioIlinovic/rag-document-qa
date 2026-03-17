"""Tests for the health check endpoint."""

import pytest

class TestHealthEndpoint:
    """Test cases for the health endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
