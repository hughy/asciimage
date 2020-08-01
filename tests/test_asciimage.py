from PIL import Image
import torch

from asciimage import __version__
from asciimage import convert
from asciimage import preprocess


def test_version():
    assert __version__ == "0.1.0"


def test_preprocess():
    test_output_size = 32
    fixture_tensor = torch.load(f"fixtures/cat0_tensor{test_output_size}.pth")

    test_image = Image.open("images/cat0.jpg")
    preprocessed_test_image = preprocess.preprocess_image(test_image, test_output_size)

    assert torch.all(torch.eq(fixture_tensor, preprocessed_test_image))


def test_preprocess_shape():
    test_image = Image.open("images/cat0.jpg")
    width, height = test_image.size

    test_output_size = 16
    preprocessed_test_image = preprocess.preprocess_image(test_image, test_output_size)
    # Output must be 4D Tensor
    # Represents images of 32xoutput_size pixels on each side
    assert preprocessed_test_image.shape == (
        1,
        1,
        test_output_size * 32,
        test_output_size * 32,
    )


def test_convert():
    test_output_size = 32
    with open(f"fixtures/cat0_str{test_output_size}.txt") as f:
        fixture_output = f.read()

    test_converson = convert.convert("images/cat0.jpg", test_output_size)
    assert fixture_output == test_converson
