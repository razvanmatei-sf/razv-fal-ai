"""
Fal AI API Client wrapper for ComfyUI custom nodes
"""
import fal_client


class FalClient:
    """
    A wrapper class for the fal_client library to handle API calls to fal.ai
    """

    def __init__(self, api_key: str):
        """
        Initialize the Fal AI client

        Args:
            api_key: The fal.ai API key
        """
        self.api_key = api_key
        # Set the API key for fal_client
        import os
        os.environ["FAL_KEY"] = api_key

    def run(self, model_id: str, arguments: dict, timeout: int = 300):
        """
        Run a model synchronously and wait for the result

        Args:
            model_id: The fal.ai model identifier (e.g., "rundiffusion-fal/juggernaut-flux-lora/inpainting")
            arguments: Dictionary of input arguments for the model
            timeout: Maximum time to wait for result in seconds

        Returns:
            The result dictionary from the API
        """
        result = fal_client.run(model_id, arguments=arguments)
        return result

    def submit(self, model_id: str, arguments: dict):
        """
        Submit a model request asynchronously

        Args:
            model_id: The fal.ai model identifier
            arguments: Dictionary of input arguments for the model

        Returns:
            A handler object that can be used to get the result
        """
        handler = fal_client.submit(model_id, arguments=arguments)
        return handler

    def subscribe(self, model_id: str, arguments: dict, with_logs: bool = False):
        """
        Subscribe to a model and get results with optional logging

        Args:
            model_id: The fal.ai model identifier
            arguments: Dictionary of input arguments for the model
            with_logs: Whether to include logs in the response

        Returns:
            The result dictionary from the API
        """
        result = fal_client.subscribe(
            model_id,
            arguments=arguments,
            with_logs=with_logs
        )
        return result

    def upload_file(self, file_path: str) -> str:
        """
        Upload a file to fal.media CDN

        Args:
            file_path: Path to the file to upload

        Returns:
            URL of the uploaded file
        """
        return fal_client.upload_file(file_path)

    def upload_image(self, image_data: bytes, content_type: str = "image/png") -> str:
        """
        Upload image data to fal.media CDN

        Args:
            image_data: Raw image bytes
            content_type: MIME type of the image

        Returns:
            URL of the uploaded image
        """
        import tempfile
        import os

        # Determine file extension from content type
        ext_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp"
        }
        ext = ext_map.get(content_type, ".png")

        # Write to temporary file and upload
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(image_data)
            temp_path = f.name

        try:
            url = fal_client.upload_file(temp_path)
            return url
        finally:
            os.unlink(temp_path)
