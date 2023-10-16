import torch
import albumentations as albu
import numpy as np
from PIL import Image
import io
from fastapi import UploadFile

def standardize(x, mean, std):
    """Standardize the given image.
    Args:
        x (np.ndarray): Image to standardize.
    Returns:
        np.ndarray: Standardized image.
    """
    x = x.reshape(1,240,240)
    x = x/255
    x -= mean
    x /= std
    return x.astype(np.float32)

resize = albu.Resize(240, 240, always_apply=True)

def preprocess_t1(t1image):
    transform = resize(image=t1image)
    t1image = transform['image']
    t1image = standardize(t1image, 0.0999, 0.23646)
    return t1image

def preprocess_t1ce(t1ceimage):
    transform = resize(image=t1ceimage)
    t1ceimage = transform['image']
    t1ceimage = standardize(t1ceimage, 0.05345, 0.13268)
    return t1ceimage

def preprocess_t2(t2image):
    transform = resize(image=t2image)
    t2image = transform['image']
    t2image = standardize(t2image, 0.0999, 0.23646)
    return t2image

def preprocess_flair(flairimage):
    transform = resize(image=flairimage)
    flairimage = transform['image']
    flairimage = standardize(flairimage, 0.0999, 0.23646)
    return flairimage


def read_image(file: UploadFile) -> np.ndarray:
    """Read the uploaded file into a grayscale numpy array and reshape it to 240,240,1"""
    image_stream = Image.open(file.file).convert('L')
    image_resized = image_stream.resize((240, 240))
    return np.array(image_resized).reshape(240, 240, 1)

def pre_process_brats(t1: UploadFile, t1c: UploadFile, t2: UploadFile, flair: UploadFile):
    """
    Preprocess the uploaded files and return them separately.

    Args:
    - t1, t1c, t2, flair (UploadFile): The uploaded files.

    Returns:
    - 4 torch.Tensors: The preprocessed images as separate tensors.
    """

    if t1 is not None:
      t1_image = preprocess_t1(read_image(t1))
      t1 = torch.from_numpy(t1_image.reshape(1, 1, 240, 240))
    if t1c is not None:
      t1c_image = preprocess_t1ce(read_image(t1c))
      t1c = torch.from_numpy(t1c_image.reshape(1, 1, 240, 240))
    if t2 is not None:
      t2_image = preprocess_t2(read_image(t2))
      t2 = torch.from_numpy(t2_image.reshape(1, 1, 240, 240))
    if flair is not None:
      flair_image = preprocess_flair(read_image(flair))
      flair = torch.from_numpy(flair_image.reshape(1, 1, 240, 240))


    return t1, t1c, t2, flair

color_map = {
    0: [0, 0, 0],      # Black
    1: [255, 0, 0],    # Red
    2: [0, 255, 0],    # Green
    3: [0, 0, 255]     # Blue
}

def post_process_brats(prediction):
    """
    Postprocess the model output to get the mask image.

    Args:
    - prediction (torch.Tensor): The model's output tensor.

    Returns:
    - np.ndarray: The postprocessed mask image.
    """

    # Convert the tensor to numpy array and squeeze out the batch dimension
    # This is a simple example; you might need more specific post-processing steps depending on your model's output
    prediction = torch.argmax(prediction,dim=1 )
    colored_prediction = torch.zeros((*prediction.shape, 3), dtype=torch.uint8)  # Shape: [BATCH, H, W, 3]

    for value, color in color_map.items():
        mask = prediction == value
        colored_prediction[mask] = torch.tensor(color, dtype=torch.uint8)
    mask = colored_prediction.squeeze().numpy()

    # Convert the mask to a PIL Image and then to bytes
    output_image = Image.fromarray((mask).astype(np.uint8))

    byte_io = io.BytesIO()
    output_image.save(byte_io, "PNG")
    byte_io.seek(0)

    return byte_io

