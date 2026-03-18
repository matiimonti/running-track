from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app

client = TestClient(app)


def test_health_returns_200():
    with patch("app.main.aredis.from_url") as mock_redis:
        mock_instance = AsyncMock()
        mock_instance.ping = AsyncMock()
        mock_instance.aclose = AsyncMock()
        mock_redis.return_value = mock_instance
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
