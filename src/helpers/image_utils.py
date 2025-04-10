import base64
import io
import torch
from PIL import Image, UnidentifiedImageError
from PIL.ImageFile import ImageFile
import torchvision.transforms as T


def validate_and_decode_image(b64_string: str) -> ImageFile:
    try:
        image_bytes = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Validate format, throws if not valid image
        img = Image.open(io.BytesIO(image_bytes))  # Reopen image to use
        return img
    except Exception as e:
        raise ValueError("Invalid image input") from e


def tensor_to_base64_img(tensor: torch.Tensor) -> str:
    """
    Converts a 3D tensor (C, H, W) to a base64 PNG image after validating it.
    """
    if tensor.ndim != 3 or tensor.size(0) not in [1, 3]:
        raise ValueError("Expected tensor of shape (C, H, W) with 1 or 3 channels")

    # Clamp values to [0, 1] and convert to PIL image
    tensor = tensor.detach().cpu().clamp(0, 1)
    to_pil = T.ToPILImage()
    image = to_pil(tensor)

    # Save to in-memory buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    try:
        validated = Image.open(buffer)
        validated.verify()  # Raises error if corrupt
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Generated image is not valid") from e

    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"
