from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    """Database settings; uses the same variable names as .env and docker-compose.yml."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    postgres_host: str = "localhost"
    # 5432 is taken by a local PostgreSQL service on this machine
    postgres_port: int = 5433
    postgres_db: str
    postgres_user: str
    postgres_password: str

    @property
    def database_url(self) -> URL:
        # URL.create escapes special characters in the password
        return URL.create(
            "postgresql+psycopg2",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
