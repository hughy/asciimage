# asciimage

This project implements a toy program that uses deep learning models to convert images to ASCII art.
## Implementation

Image conversion works by dividing the input image into equally-sized 'blocks' and treating each block as an independent image. The next step in conversion passes each block to a convolutional neural network that predicts the most likely character class for that image as if the image were an image of a character.

### CharacterPool module

The division of an image into equally-sized blocks and 'downsampling' each block to a single value sounds a lot like a pooling layer in a convolutional neural network. I leverage that similarity and use PyTorch to implement conversion from image to text as a neural network `Module` class: `CharacterPool`. The `CharacterPool` module operates in much the same way as `AvgPool` or `MaxPool` by using `unfold` to yield image blocks but uses a character classification model as its downsampling function.

### Character classifier

I use a convolutional neural network modeled after [LeNet-5](http://yann.lecun.com/exdb/lenet/) and trained on the [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) dataset of handwritten characters to predict the character for each 'block' in the input image.

### Preprocessing

Each input image undergoes the following steps before conversion to text:

1. Cropping the image to a square
2. Increasing contrast
3. Conversion to grayscale
4. Color inversion
5. Resizing for desired output size

The character recognition models that I used necessitate both the conversion to grayscale and color inversion of images in order to match the input format of the training images. Similarly, resizing input images ensures that the input image will consist of 32x32 px 'blocks'.

## Example
Given the input image below of a cat, the converter produces a string of characters like those pictured beside the cat:

![Test cat image](images/cat0.jpg) ![Example output](docs/cat0_example.png)

The converter module generated the example output above using an output size of 128x128 characters. The image above shows a cropped selection of the output text for detail. 

The text below provides a more representative example of converter output at lower resolutions such as 32x32 as in this case:

<div style="font-size:5pt">

    11111111111111111111111111111111
    11111111111111111111111111111111
    11111111111111111111111111111111
    11111111M71111111111111111111111
    111111DD6QIM11111111111111111111
    1111jBBBBBB37111D1111WY111111111
    1111jaBBBBBBBT116811jB9111111111
    111LWBBBBBBBBP1168BBBBR111111111
    1111eBeQ6BBBB811WBBBBBQT11111111
    11116811WBBBB91LBBBWBBB711111111
    1111111jBBBBBBDDQQeBBBB811111111
    1111111BWeBBBBBBWBBBBWBB71111111
    1111111S6eBBBBBBBB3BBBBBT1111111
    11111111VUBBBBBBQ9WWeeBB71111111
    111111111WBBBeBBBBBBeBeBT1111111
    111111111WBBBBBBBBBBBBBB71111111
    1111111116BBBBeBBBBQeBQB71111111
    1111111116BBBBBBBBB88QBB11111111
    111111111iBBBBBBBBBBBB8R11111111
    111111111LBBBBBBB8BBBBBT11111111
    11111111116BBBBeBQBBBBe111111111
    1111111111dBBBBeBBBeBBP111111111
    1111111111eBBBBe8BBBB81111111111
    1111111111LeQBBBeBBBBT1111111111
    11111111111WBRjBBBBQQ71111111111
    11111111111WBe1WBeBQR11111111111
    11111111111WBMDDBBBBR11111111111
    11111111111WeR8BR6BBB11111111111
    11111111111111111bBBR11111111111
    111111111111111111WBBT1111111111
    1111111111111111111VV11111111111
    11111111111111111111111111111111

</div>

## Usage

## Limitations

- The example shown above suggests that the character pooling module primarily distinguishes between black (class `B`) and white (class `1`). Alternate classes near the edges of the cat figure and in the area of its face suggest some texture, but output text only contains detail beyond the silhouette of the cat at higher resolutions (i.e., 128x128 and larger).

- Since the converter downsamples each 'block' in the input image to a single character only monospace fonts will display text in such a way that a viewer can recognize the input image.

- The character classifier that I used to generate the example only recognizes digits and uppercase and lowercase letters. Additional characters such as a blank space or period might produce more detailed text-images.

- The converter cannot manage large output image size (i.e., larger than 256x256). Dividing the blocks resulting from `unfold`ing the input image into smaller batches before passing them to the character classifier may resolve this.