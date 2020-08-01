"""
Implements conversion from image to ASCII characters as a PyTorch Module.

1. Use `unfold` to extract N blocks of size 32x32 from image
2. Batch blocks into Nx1x32x32 Tensor
3. Evaluate trained character recognition model on batch
4. Get best class for each block
5. Output NxN Tensor of predicted characters

"""
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from asciimage.preprocess import preprocess_image


def load_model(filepath: str) -> torch.nn.Module:
    """Loads saved PyTorch model stored at the given filepath.
    """
    return torch.load(filepath)


class CharacterPool(nn.Module):
    """
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


def get_output_string(character_predictions: torch.Tensor, size: int) -> str:
    """Converts a Tensor of character predictions to a string.
    """
    prediction_matrix = character_predictions.reshape(size, size)
    return "\n".join(
        ["".join([str(c) for c in row]) for row in prediction_matrix.tolist()]
    )


def convert(image_path: str, output_size: int) -> None:
    output_image_size = 32
    image = Image.open(image_path)
    preprocessed = preprocess_image(image, output_image_size)

    character_model = load_model("models/lenet_5.pickle")
    character_model.eval()
    pooler = CharacterPool(character_model)

    character_predictions = pooler(preprocessed)
    return get_output_string(character_predictions, output_image_size)


if __name__ == "__main__":
    print(convert("images/cat0.jpg", 32))
