from typing import Optional, Tuple
import boto3
from botocore.exceptions import ClientError, EndpointConnectionError
from dotenv import load_dotenv
import logging
import os
from io import BytesIO

from src.config.config import settings


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class S3Client:
    def __init__(self, bucket_name: str):
        try:
            # Use default credentials provider which will automatically use instance profile in AWS
            self.s3_client = boto3.client("s3")
            self.bucket_name = bucket_name
            # Test connection on initialization
            self._test_connection()
        except Exception as e:
            logger.error(f"Error initializing S3 client: {e}")
            raise

    def _test_connection(self) -> bool:
        """
        Test the S3 connection by attempting to access a test object in the bucket
        :return: True if connection successful, False otherwise
        """
        try:
            # Try to get metadata of a test object
            # We use a test object that should exist in the bucket
            test_object_key = "2095"
            self.s3_client.head_object(Bucket=self.bucket_name, Key=test_object_key)
            logger.info("Successfully connected to S3 bucket and accessed test object")
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                logger.error(f"Test object not found in bucket {self.bucket_name}")
            elif error_code == "403":
                logger.error("Access denied to S3 object. Check IAM role permissions.")
            else:
                logger.error(f"Error connecting to S3: {e}")
            return False
        except EndpointConnectionError:
            logger.error(
                "Could not connect to S3 endpoint. Check your network connection."
            )
            return False
        except Exception as e:
            logger.error(f"Unexpected error testing S3 connection: {e}")
            return False

    def upload_file(self, file_path: str, model_id: str) -> Tuple[bool, str]:
        """
        Upload a file to S3 using model_id as the key
        :param file_path: Local path to the file
        :param model_id: Model ID to use as the S3 key
        :return: Tuple of (success boolean, error message if any)
        """
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, model_id)
            logger.info(f"Successfully uploaded file for model {model_id}")
            return True, ""
        except ClientError as e:
            error_msg = f"Error uploading file to S3: {e}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error uploading file: {e}"
            logger.error(error_msg)
            return False, error_msg

    def download_file(self, model_id: str) -> Tuple[Optional[BytesIO], str]:
        """
        Download a file from S3 into memory using model_id as the key
        :param model_id: Model ID used as the S3 key
        :return: Tuple of (BytesIO object containing the file data or None, error message if any)
        """
        try:
            # Create a BytesIO object to store the file in memory
            file_obj = BytesIO()

            # Download the file directly into memory
            self.s3_client.download_fileobj(self.bucket_name, model_id, file_obj)

            # Reset the file pointer to the beginning
            file_obj.seek(0)

            logger.info(
                f"Successfully downloaded file for model {model_id} into memory"
            )
            return file_obj, ""
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                error_msg = f"Model {model_id} not found in S3"
            else:
                error_msg = f"Error downloading file from S3: {e}"
            logger.error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error downloading file: {e}"
            logger.error(error_msg)
            return None, error_msg

    def get_file_url(
        self, model_id: str, expires_in: int = 3600
    ) -> Tuple[Optional[str], str]:
        """
        Generate a pre-signed URL for a model file in S3
        :param model_id: Model ID used as the S3 key
        :param expires_in: URL expiration time in seconds (default: 1 hour)
        :return: Tuple of (pre-signed URL or None, error message if any)
        """
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": model_id},
                ExpiresIn=expires_in,
            )
            logger.info(f"Successfully generated pre-signed URL for model {model_id}")
            return url, ""
        except ClientError as e:
            error_msg = f"Error generating pre-signed URL: {e}"
            logger.error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error generating pre-signed URL: {e}"
            logger.error(error_msg)
            return None, error_msg
