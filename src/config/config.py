from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    models_s3_bucket_name: str

    class Config:
        env_file = ".env"


settings = Settings()
