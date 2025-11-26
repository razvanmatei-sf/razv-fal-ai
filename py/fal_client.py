"""
Fal AI API Client Node for ComfyUI

This node creates a client for connecting to the fal.ai API.
"""
import os
import configparser


class FalAIAPIClient:
    """
    Fal AI API Client Node

    This node creates a client for connecting to the fal.ai API.
    The API key can be provided directly, via config.ini, or via FAL_KEY environment variable.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Fal AI API key. Leave empty to use config.ini or FAL_KEY environment variable."
                }),
            },
        }

    RETURN_TYPES = ("FAL_AI_API_CLIENT",)
    RETURN_NAMES = ("client",)

    FUNCTION = "create_client"

    CATEGORY = "FalAI"

    def create_client(self, api_key):
        """
        Create a Fal AI API client

        Args:
            api_key: Fal AI API key (optional if set in config.ini or environment)

        Returns:
            Dictionary containing the API key for use by other nodes
        """
        fal_api_key = ""

        if api_key == "":
            # Try to read from config.ini
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                config_path = os.path.join(parent_dir, 'config.ini')

                if os.path.exists(config_path):
                    config = configparser.ConfigParser()
                    config.read(config_path)
                    fal_api_key = config.get('API', 'api_key', fallback='')

                if not fal_api_key:
                    # Try environment variable
                    fal_api_key = os.environ.get('FAL_KEY', '')

                if not fal_api_key:
                    raise ValueError('API_KEY is empty. Please provide an API key, set it in config.ini, or set FAL_KEY environment variable.')

            except Exception as e:
                raise ValueError(f'Unable to find API_KEY: {str(e)}')

        else:
            fal_api_key = api_key

        return ({
            "api_key": fal_api_key
        },)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FalAI Client": FalAIAPIClient
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAI Client": "FalAI Client"
}
