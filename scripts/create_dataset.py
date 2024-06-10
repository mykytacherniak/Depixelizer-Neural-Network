from glob import glob
from os import path
from typing import Optional
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if pil_image.ndim == 2:
        return pil_image.copy()[None]
    if pil_image.ndim != 3:
        raise ValueError("image must have either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"image has shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")

    rgb = pil_image / 255
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]

    grayscale = np.where(
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255

    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]


def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim < 3 or image.shape[-3] != 1:

        raise ValueError("image must have shape (..., 1, H, W)")
    if width < 2 or height < 2 or size < 2:
        raise ValueError("width/height/size must be >= 2")
    if x < 0 or (x + width) > image.shape[-1]:
        raise ValueError(f"x={x} and width={width} do not fit into the image width={image.shape[-1]}")
    if y < 0 or (y + height) > image.shape[-2]:
        raise ValueError(f"y={y} and height={height} do not fit into the image height={image.shape[-2]}")

    # The (height, width) slices to extract the area that should be pixelated.
    area = (..., slice(y, y + height), slice(x, x + width))

    pixelated_image = pixelate(image, x, y, width, height, size)

    known_array = np.ones_like(image, dtype=bool)
    known_array[area] = False

    # Create a copy to avoid that "target_array" and "image" point to the same array
    target_array = image.copy()

    return pixelated_image, known_array, target_array


def pixelate(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> np.ndarray:
    image = image.copy()
    curr_x = x

    while curr_x < x + width:
        curr_y = y
        while curr_y < y + height:
            block = (..., slice(curr_y, min(curr_y + size, y + height)), slice(curr_x, min(curr_x + size, x + width)))
            image[block] = image[block].mean()
            curr_y += size
        curr_x += size

    return image


class RandomImagePixelationDataset(Dataset):

    def __init__(
            self,
            image_dir,
            width_range: tuple[int, int],
            height_range: tuple[int, int],
            size_range: tuple[int, int],
            dtype: Optional[type] = None
    ):
        RandomImagePixelationDataset._check_range(width_range, "width")
        RandomImagePixelationDataset._check_range(height_range, "height")
        RandomImagePixelationDataset._check_range(size_range, "size")
        self.image_files = sorted(path.abspath(f) for f in glob(path.join(image_dir, "**", "*.jpg"), recursive=True))
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype

    @staticmethod
    def _check_range(r: tuple[int, int], name: str):
        if r[0] < 2:
            raise ValueError(f"minimum {name} must be >= 2")
        if r[0] > r[1]:
            raise ValueError(f"minimum {name} must be <= maximum {name}")

    def __getitem__(self, index):
        with Image.open(self.image_files[index]) as im:
            image = np.array(im, dtype=self.dtype)
        image = to_grayscale(image)  # Image shape is now (1, H, W)
        image_width = image.shape[-1]
        image_height = image.shape[-2]

        rng = np.random.default_rng(seed=index)

        # Both width and height can be arbitrary, but they must not exceed the
        # actual image width and height
        width = min(rng.integers(low=self.width_range[0], high=self.width_range[1], endpoint=True), image_width)
        height = min(rng.integers(low=self.height_range[0], high=self.height_range[1], endpoint=True), image_height)

        # Ensure that x and y always fit with the randomly chosen width and
        # height (and not throw an error in "prepare_image")
        x = rng.integers(image_width - width, endpoint=True)
        y = rng.integers(image_height - height, endpoint=True)

        size = rng.integers(low=self.size_range[0], high=self.size_range[1], endpoint=True)

        pixelated_image, known_array, target_array = prepare_image(image, x, y, width, height, size)
        combined_array = torch.cat((torch.from_numpy(pixelated_image), torch.from_numpy(known_array)),
                                   0)
        return combined_array, known_array, target_array, self.image_files[index]

    def __len__(self):
        return len(self.image_files)


def stack(batch_as_list: list):
    pixelated_images = []
    known_arrays = []
    target_arrays = []
    image_files = []

    for pixelated_image, known_array, target_array, image_file in batch_as_list:
        pixelated_images.append(pixelated_image)
        known_arrays.append(known_array)
        target_arrays.append(torch.from_numpy(target_array))
        image_files.append(image_file)

    stacked_pixelated_images = torch.stack(pixelated_images)
    stacked_known_arrays = torch.stack([torch.from_numpy(arr) for arr in known_arrays])
    target_tensors = torch.stack(target_arrays)

    return stacked_pixelated_images, stacked_known_arrays, target_tensors
