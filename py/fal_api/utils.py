"""
Utility functions for Fal AI ComfyUI nodes
"""
import requests
import io
import numpy
import PIL.Image
import torch
from typing import List


def imageurl2tensor(image_urls: List[str]):
    """
    Convert a list of image URLs to a PyTorch tensor

    Args:
        image_urls: List of image URLs to download and convert

    Returns:
        PyTorch tensor of images in ComfyUI format (B, H, W, C)
    """
    images = []
    if not image_urls:
        return torch.zeros((1, 1, 1, 3))
    for url in image_urls:
        image_data = fetch_image(url)
        image = decode_image(image_data)
        images.append(image)
    return images2tensor(images)


def fetch_image(url: str, stream: bool = True) -> bytes:
    """
    Fetch image data from a URL

    Args:
        url: URL of the image to fetch
        stream: Whether to use streaming mode

    Returns:
        Raw image bytes
    """
    response = requests.get(url, stream=stream)
    response.raise_for_status()
    return response.content


def decode_image(data_bytes: bytes, rtn_mask: bool = False) -> PIL.Image.Image:
    """
    Decode image bytes to a PIL Image

    Args:
        data_bytes: Raw image bytes
        rtn_mask: Whether to return the alpha channel as a mask

    Returns:
        PIL Image object
    """
    with io.BytesIO(data_bytes) as bytes_io:
        img = PIL.Image.open(bytes_io)
        img.load()  # Force load to prevent issues with closed file
        if not rtn_mask:
            img = img.convert('RGB')
        elif 'A' in img.getbands():
            img = img.getchannel('A')
        else:
            img = None
    return img


def images2tensor(images):
    """
    Convert a list of PIL Images to a PyTorch tensor

    Args:
        images: List of PIL Image objects or single PIL Image

    Returns:
        PyTorch tensor in ComfyUI format (B, H, W, C)
    """
    from collections.abc import Iterable
    if isinstance(images, Iterable) and not isinstance(images, PIL.Image.Image):
        return torch.stack([torch.from_numpy(numpy.array(image)).float() / 255.0 for image in images])
    return torch.from_numpy(numpy.array(images)).unsqueeze(0).float() / 255.0


def tensor2images(tensor):
    """
    Convert a PyTorch tensor to a list of PIL Images

    Args:
        tensor: PyTorch tensor in ComfyUI format (B, H, W, C)

    Returns:
        List of PIL Image objects
    """
    np_imgs = numpy.clip(tensor.cpu().numpy() * 255.0, 0.0, 255.0).astype(numpy.uint8)
    return [PIL.Image.fromarray(np_img) for np_img in np_imgs]


def encode_image(img: PIL.Image.Image, mask: PIL.Image.Image = None) -> bytes:
    """
    Encode a PIL Image to bytes

    Args:
        img: PIL Image to encode
        mask: Optional alpha mask to apply

    Returns:
        Image bytes in PNG or JPEG format
    """
    if mask is not None:
        img = img.copy()
        img.putalpha(mask)
    with io.BytesIO() as bytes_io:
        if mask is not None:
            img.save(bytes_io, format='PNG')
        else:
            img.save(bytes_io, format='JPEG')
        data_bytes = bytes_io.getvalue()
    return data_bytes


def tensor_to_pil(tensor) -> List[PIL.Image.Image]:
    """
    Convert ComfyUI IMAGE tensor to list of PIL Images

    Args:
        tensor: IMAGE tensor from ComfyUI (B, H, W, C)

    Returns:
        List of PIL Image objects
    """
    return tensor2images(tensor)


def pil_to_bytes(image: PIL.Image.Image, format: str = "PNG") -> bytes:
    """
    Convert PIL Image to bytes

    Args:
        image: PIL Image object
        format: Output format (PNG, JPEG, etc.)

    Returns:
        Image bytes
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()
