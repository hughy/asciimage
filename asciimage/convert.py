import argparse
import string

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from asciimage.preprocess import preprocess_image


class CharacterPool(nn.Module):
    """
    Implements conversion from image to ASCII characters as a PyTorch Module.

    The `forward` pass over the image executes the following steps:

    1. Use `unfold` to extract `n` blocks of size 32x32 from image
    2. Batch blocks into nx1x32x32 Tensor
    3. Evaluate trained character recognition model on batch
    4. Identify predicted class for each block
    """

    def __init__(self, character_model: nn.Module) -> None:
        super().__init__()
        self.character_model = character_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = F.unfold(x, kernel_size=32, stride=32)
        _, _, n = blocks.shape
        block_batch = blocks.permute(2, 0, 1).reshape(n, 1, 32, 32)
        y_hat = self.character_model(block_batch)
        return torch.argmax(y_hat, dim=1)


CHARACTER_CLASSES = string.digits + string.ascii_uppercase + string.ascii_lowercase


def get_output_string(character_predictions: torch.Tensor, size: int) -> str:
    """Converts a Tensor of character predictions to a string.
    """
    prediction_matrix = character_predictions.reshape(size, size)
    return "\n".join(
        [
            "".join([CHARACTER_CLASSES[c] for c in row])
            for row in prediction_matrix.tolist()
        ]
    )


def convert_image(image_path: str, output_size: int) -> str:
    """Converts the image at the given path to a string of characters.
    """
    image = Image.open(image_path)
    preprocessed = preprocess_image(image, output_size)

    # Load TorchScript model
    character_model = torch.jit.load("models/lenet_5_emnist.pt")
    character_model.eval()
    pooler = CharacterPool(character_model)

    character_predictions = pooler(preprocessed)
    return get_output_string(character_predictions, output_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Converts images to strings of ASCII characters by applying a deep learning
        character recognition model to image blocks.
    """
    )
    parser.add_argument(
        "-i",
        "--image-path",
        type=str,
        default="images/cat0.jpg",
        help="Path to the image to convert",
    )
    parser.add_argument(
        "-o",
        "--output-size",
        type=int,
        default=32,
        help="The size of the output string. `convert` will generate a string with this number of rows and columns.",
    )
    args = parser.parse_args()
    print(convert_image(args.image_path, args.output_size))
