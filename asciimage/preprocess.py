#!/usr/bin/env python
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageOps
import torch
from torchvision import transforms


def preprocess_image(image: Image, output_size: int) -> torch.Tensor:
    """Performs necessary preprocessing steps on the input image.
    """
    # Increase contrast
    contraster = ImageEnhance.Contrast(image)
    contrasted_image = contraster.enhance(1.5)
    # Convert to grayscale
    grayscale_image = contrasted_image.convert(mode="L")
    # Invert colors
    inverted_image = ImageOps.invert(grayscale_image)
    # Resize to yield NxN blocks of size 32x32
    resized_image = inverted_image.resize((output_size * 32, output_size * 32))
    # Convert to Tensor
    return transforms.ToTensor()(resized_image).unsqueeze(0)
