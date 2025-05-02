from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    models_s3_bucket_name: str


settings = Settings()
