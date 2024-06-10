import os
from torchvision import transforms
from PIL import Image

im_shape = 64
resize_transforms = transforms.Compose([
    transforms.Resize(size=im_shape),
    transforms.CenterCrop(size=(im_shape, im_shape)),
])

training_path = 'data/training'

resized_path = 'data/resized_training'

if not os.path.exists(resized_path):
    os.makedirs(resized_path)

for folder in os.listdir(training_path):
    folder_path = os.path.join(training_path, folder)

    if not os.path.isdir(folder_path):
        continue

    resized_folder_path = os.path.join(resized_path, folder)
    if not os.path.exists(resized_folder_path):
        os.makedirs(resized_folder_path)

    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        with Image.open(image_path) as image:
            resized_image = resize_transforms(image)
            resized_image_path = os.path.join(resized_folder_path, image_file)
            resized_image.save(resized_image_path)
