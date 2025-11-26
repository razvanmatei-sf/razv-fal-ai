"""
Fal AI Upload Image Node for ComfyUI

Uploads images to fal.media CDN for use with fal.ai API endpoints.
"""
import tempfile
import os
import fal_client
from .fal_api.utils import tensor2images, pil_to_bytes


class FalAIUploadImage:
    """
    Fal AI Upload Image Node

    Uploads a ComfyUI IMAGE tensor to fal.media CDN and returns the URL.
    This URL can then be used as input to other Fal AI nodes.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("FAL_AI_API_CLIENT",),
                "image": ("IMAGE",),
            },
            "optional": {
                "filename": ("STRING", {
                    "default": "image",
                    "tooltip": "Base filename for the uploaded image (without extension)"
                }),
                "format": (["png", "jpeg", "webp"], {
                    "default": "png",
                    "tooltip": "Image format for upload"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_url",)

    FUNCTION = "upload"

    CATEGORY = "FalAI"

    def upload(self, client, image, filename="image", format="png"):
        """
        Upload an image to fal.media CDN

        Args:
            client: Fal AI API client
            image: ComfyUI IMAGE tensor
            filename: Base filename for the uploaded image
            format: Image format (png, jpeg, webp)

        Returns:
            URL of the uploaded image
        """
        # Set the API key
        os.environ["FAL_KEY"] = client["api_key"]

        # Convert tensor to PIL image (take first image if batch)
        pil_images = tensor2images(image)
        pil_image = pil_images[0]

        # Convert format name to PIL format
        format_map = {
            "png": "PNG",
            "jpeg": "JPEG",
            "webp": "WEBP"
        }
        pil_format = format_map.get(format, "PNG")

        # Extension map
        ext_map = {
            "png": ".png",
            "jpeg": ".jpg",
            "webp": ".webp"
        }
        ext = ext_map.get(format, ".png")

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, prefix=f"{filename}_") as f:
            pil_image.save(f, format=pil_format)
            temp_path = f.name

        try:
            # Upload to fal.media
            url = fal_client.upload_file(temp_path)
            print(f"[FalAI Upload] Image uploaded successfully: {url}")
            return (url,)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FalAI Upload Image": FalAIUploadImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAI Upload Image": "FalAI Upload Image"
}
