"""
Juggernaut Flux LoRA Inpainting Node for ComfyUI

Uses the fal.ai API to perform inpainting using the Juggernaut Flux LoRA model.
API Documentation: https://fal.ai/models/rundiffusion-fal/juggernaut-flux-lora/inpainting/api
"""
from .fal_api.client import FalClient
from .fal_api.utils import imageurl2tensor


class JuggernautFluxInpainting:
    """
    Juggernaut Flux LoRA Inpainting Node

    Performs inpainting using the Juggernaut Flux LoRA model via fal.ai API.
    Requires an image URL and mask URL as inputs.
    """

    # Image size presets
    IMAGE_SIZES = [
        "square_hd",
        "square",
        "portrait_4_3",
        "portrait_16_9",
        "landscape_4_3",
        "landscape_16_9"
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("FAL_AI_API_CLIENT",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The prompt to generate an image from"
                }),
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "URL of the image to inpaint (connect from Upload Image node)",
                    "forceInput": True
                }),
                "mask_url": ("STRING", {
                    "default": "",
                    "tooltip": "URL of the mask image. White areas will be inpainted.",
                    "forceInput": True
                }),
            },
            "optional": {
                "lora_1_path": ("STRING", {
                    "default": "",
                    "tooltip": "First LoRA model URL or path (e.g., 'https://civitai.com/api/download/models/...')"
                }),
                "lora_1_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "First LoRA influence scale (0.0 to 4.0)"
                }),
                "lora_2_path": ("STRING", {
                    "default": "",
                    "tooltip": "Second LoRA model URL or path (optional)"
                }),
                "lora_2_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.1,
                    "tooltip": "Second LoRA influence scale (0.0 to 4.0)"
                }),
                "num_inference_steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Number of inference steps (1-50)"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 35.0,
                    "step": 0.1,
                    "tooltip": "Guidance scale for generation (0-35)"
                }),
                "strength": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Strength of the inpainting effect (0.01-1.0)"
                }),
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "tooltip": "Number of images to generate (1-4)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed for reproducible results. -1 for random."
                }),
                "image_size": (s.IMAGE_SIZES, {
                    "default": "landscape_4_3",
                    "tooltip": "Output image size preset"
                }),
                "output_format": (["jpeg", "png"], {
                    "default": "jpeg",
                    "tooltip": "Output image format"
                }),
                "enable_safety_checker": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable the safety checker for content moderation"
                }),
                "sync_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, wait for generation to complete before returning"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    CATEGORY = "FalAI"
    FUNCTION = "execute"

    def execute(
        self,
        client,
        prompt,
        image_url,
        mask_url,
        lora_1_path="",
        lora_1_scale=1.0,
        lora_2_path="",
        lora_2_scale=1.0,
        num_inference_steps=28,
        guidance_scale=3.5,
        strength=0.85,
        num_images=1,
        seed=-1,
        image_size="landscape_4_3",
        output_format="jpeg",
        enable_safety_checker=True,
        sync_mode=False
    ):
        """
        Execute the Juggernaut Flux LoRA Inpainting model

        Args:
            client: Fal AI API client
            prompt: Text prompt for inpainting
            image_url: URL of the source image
            mask_url: URL of the mask image (white = inpaint area)
            lora_1_path: First LoRA model URL or path
            lora_1_scale: First LoRA influence scale (0-4)
            lora_2_path: Second LoRA model URL or path
            lora_2_scale: Second LoRA influence scale (0-4)
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale for generation
            strength: Strength of inpainting effect
            num_images: Number of images to generate
            seed: Random seed (-1 for random)
            image_size: Output image size preset
            output_format: Output format (jpeg or png)
            enable_safety_checker: Whether to enable content safety checker
            sync_mode: Whether to wait for completion

        Returns:
            Generated image tensor
        """

        # Create the actual client object
        real_client = FalClient(api_key=client["api_key"])

        # Build LoRA list from inputs
        lora_list = []
        if lora_1_path and lora_1_path.strip():
            lora_list.append({
                "path": lora_1_path.strip(),
                "scale": lora_1_scale
            })
        if lora_2_path and lora_2_path.strip():
            lora_list.append({
                "path": lora_2_path.strip(),
                "scale": lora_2_scale
            })

        # Build the request arguments
        arguments = {
            "prompt": prompt,
            "image_url": image_url,
            "mask_url": mask_url,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "num_images": num_images,
            "image_size": image_size,
            "output_format": output_format,
            "enable_safety_checker": enable_safety_checker,
            "sync_mode": sync_mode
        }

        # Only add seed if it's not -1 (random)
        if seed != -1:
            arguments["seed"] = seed

        # Add LoRAs if any are specified
        if lora_list:
            arguments["loras"] = lora_list

        # Model identifier for Juggernaut Flux LoRA Inpainting
        model_id = "rundiffusion-fal/juggernaut-flux-lora/inpainting"

        try:
            # Use subscribe for reliable results with logging
            result = real_client.subscribe(model_id, arguments, with_logs=True)

            # Extract image URLs from result
            if "images" in result and result["images"]:
                image_urls = [img["url"] for img in result["images"]]
                return (imageurl2tensor(image_urls),)
            else:
                raise Exception(f"No images received from API. Response: {result}")

        except Exception as e:
            print(f"Error in Juggernaut Flux Inpainting: {str(e)}")
            raise e


# Node registration
NODE_CLASS_MAPPINGS = {
    "FalAI Juggernaut Flux Inpainting": JuggernautFluxInpainting
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAI Juggernaut Flux Inpainting": "FalAI Juggernaut Flux Inpainting"
}
