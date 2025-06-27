"""
RembgBackgroundRemovalNode - ComfyUI Node
AI-powered background removal using rembg

Generated from: https://github.com/danielgatis/rembg
"""

import torch
import numpy as np
from PIL import Image
import io

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: rembg library not found. Please install with: pip install rembg")

class RembgBackgroundRemovalNode:
    """ComfyUI Node for AI-powered background removal using rembg"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ([
                    "u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg",
                    "silueta", "isnet-general-use", "isnet-anime",
                    "birefnet-general", "birefnet-general-lite", "birefnet-portrait"
                ], {"default": "u2net"}),
                "return_mask": ("BOOLEAN", {"default": False}),
                "alpha_matting": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "image/background"

    def __init__(self):
        self.session_cache = {}

    def remove_background(self, image, model, return_mask, alpha_matting):
        """Remove background from image using rembg"""

        if not REMBG_AVAILABLE:
            raise Exception("rembg library is not installed. Please install with: pip install rembg")

        # Convert ComfyUI tensor to PIL Image
        img_array = (image.squeeze().cpu().numpy() * 255).astype(np.uint8)
        if len(img_array.shape) == 3:
            pil_image = Image.fromarray(img_array, 'RGB')
        else:
            pil_image = Image.fromarray(img_array, 'L').convert('RGB')

        # Get or create session for the model
        if model not in self.session_cache:
            try:
                self.session_cache[model] = new_session(model)
            except Exception as e:
                print(f"Warning: Failed to load model '{model}', falling back to 'u2net': {e}")
                if 'u2net' not in self.session_cache:
                    self.session_cache['u2net'] = new_session('u2net')
                model = 'u2net'

        session = self.session_cache[model]

        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # Remove background
        try:
            output_bytes = remove(img_bytes, session=session, alpha_matting=alpha_matting)
        except Exception as e:
            raise Exception(f"Background removal failed: {e}")

        # Convert result back to PIL Image
        output_image = Image.open(io.BytesIO(output_bytes)).convert('RGBA')

        # Extract RGB and alpha channels
        rgb_array = np.array(output_image)[:, :, :3]
        alpha_array = np.array(output_image)[:, :, 3]

        # Convert back to ComfyUI tensors
        rgb_tensor = torch.from_numpy(rgb_array.astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy(alpha_array.astype(np.float32) / 255.0).unsqueeze(0)

        if return_mask:
            return (image, mask_tensor)
        else:
            return (rgb_tensor, mask_tensor)

NODE_CLASS_MAPPINGS = {
    "RembgBackgroundRemovalNode": RembgBackgroundRemovalNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RembgBackgroundRemovalNode": "Rembg Background Removal"
}
