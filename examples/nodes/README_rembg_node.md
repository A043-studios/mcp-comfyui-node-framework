# Rembg Background Removal Node for ComfyUI

## Overview

This ComfyUI node provides AI-powered background removal using the rembg library. It supports multiple models and advanced features like alpha matting.

**Generated from:** https://github.com/danielgatis/rembg

## Features

- **Multiple AI Models**: Support for various background removal models including:
  - u2net (default)
  - u2netp
  - u2net_human_seg
  - u2net_cloth_seg
  - silueta
  - isnet-general-use
  - isnet-anime
  - birefnet-general
  - birefnet-general-lite
  - birefnet-portrait

- **Advanced Options**:
  - Alpha matting for better edge quality
  - Option to return original image with mask or processed image
  - Model caching for improved performance

- **ComfyUI Integration**:
  - Standard IMAGE input/output types
  - MASK output for compositing
  - Proper error handling
  - Category: "image/background"

## Installation

### Prerequisites

1. ComfyUI installed and working
2. Python 3.8 or higher

### Steps

1. **Install rembg library**:
   ```bash
   pip install rembg
   ```

2. **Copy the node file**:
   - Save `rembg_background_removal_node.py` to your ComfyUI `custom_nodes` directory
   - The path should be: `ComfyUI/custom_nodes/rembg_background_removal_node.py`

3. **Restart ComfyUI**:
   - Restart your ComfyUI server
   - The node will appear under "image/background" category

## Usage

### Basic Usage

1. Add the "Rembg Background Removal" node to your workflow
2. Connect an IMAGE input to the node
3. Select your preferred model (u2net is recommended for general use)
4. Execute the workflow

### Parameters

- **image**: Input image (IMAGE type)
- **model**: AI model to use for background removal
- **return_mask**: If True, returns original image + mask; if False, returns processed image + mask
- **alpha_matting**: Enable for better edge quality (slower processing)

### Outputs

- **image**: Either the original image (if return_mask=True) or the background-removed image
- **mask**: Alpha mask showing the removed background areas

## Model Recommendations

- **u2net**: Best general-purpose model
- **u2net_human_seg**: Optimized for human subjects
- **u2net_cloth_seg**: Optimized for clothing
- **isnet-anime**: Best for anime/cartoon images
- **birefnet-portrait**: Optimized for portrait photos

## Troubleshooting

### Common Issues

1. **"rembg library not found" error**:
   - Install rembg: `pip install rembg`
   - Restart ComfyUI

2. **Model loading errors**:
   - The node will automatically fall back to u2net if a model fails to load
   - Check your internet connection (models are downloaded on first use)

3. **Memory issues**:
   - Use smaller images or restart ComfyUI to clear model cache
   - Consider using lighter models like u2netp

### Performance Tips

- Models are cached after first use for better performance
- u2netp is faster but slightly less accurate than u2net
- Alpha matting improves quality but increases processing time

## License

This node is based on the rembg library. Please check the original repository for licensing information:
https://github.com/danielgatis/rembg

## Support

For issues related to:
- **Node functionality**: Check ComfyUI logs and ensure rembg is properly installed
- **rembg library**: Visit the original repository at https://github.com/danielgatis/rembg
- **ComfyUI integration**: Check ComfyUI documentation and community forums
