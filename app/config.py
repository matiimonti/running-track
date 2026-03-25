from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Literal


class Settings(BaseSettings):
    # Environment
    environment: Literal["local", "staging", "production"] = "local"

    # API keys
    strava_client_id: str = ""
    strava_client_secret: str = ""
    claude_api_key: str = ""
    mapbox_token: str = ""
    graphhopper_api_key: str = ""
    openrouteservice_api_key: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379"

    # CORS
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Rate limiting (requests per minute)
    rate_limit_generate: int = 30
    rate_limit_read: int = 100

    # Logging
    log_level: str = "INFO"

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("log_level", mode="before")
    @classmethod
    def set_log_level_by_env(cls, v, info):
        env = info.data.get("environment", "local")
        if env == "production":
            return "WARNING"
        if env == "staging":
            return "INFO"
        return "DEBUG"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
