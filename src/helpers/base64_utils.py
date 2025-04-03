from base64 import b64decode
import io
from PIL.Image import Image


def decode(raw_model):
    return b64decode(raw_model)


def decode_and_validate_image(b64_string):
    try:
        image_bytes = b64decode(b64_string)
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Validate format, throws if not valid image
        img = Image.open(io.BytesIO(image_bytes))  # Re-open for actual use
        return img
    except Exception as e:
        raise ValueError("Invalid image input") from e
