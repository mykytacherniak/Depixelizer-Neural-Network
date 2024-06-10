# Depixelizer Neural Network

This repository contains a neural network implementation for depixelizing images. The project involves several steps including data preparation, model training, and evaluation.

## Project Structure

- `data/`: Contains original and resized training images.
- `models/`: Stores the saved model.
- `scripts/`: Contains all Python scripts for various tasks.
- `requirements.txt`: Lists the dependencies required to run the project.
- `README.md`: Project documentation.

## Installation

1. **Clone the repository**:

    ```sh
    git clone https://github.com/mykytacherniak/Depixelizer-Neural-Network.git
    cd Depixelizer-Neural-Network
    ```

2. **Create a virtual environment and install dependencies**:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

### Step 1: Resize Images

Use `resize.py` to resize the training images.

   ```sh
       python scripts/resize.py
   ```


### Step 2: Train the Model

Use `main.py` to train the model.

   ```sh
   python scripts/main.py
   ```

### Step 3: Get Predictions

Use `get_predictions.py` to generate predictions on the test set in the pkl format.

   ```sh
   python scripts/get_predictions.py
   ```

### Step 4: Visualize Predictions

Use `visualize_predictions.py` to visualize the results.

   ```sh
   python scripts/visualize_predictions.py
   ```
    

### Step 5: Test on a Subset

Use `test_on_a_subset.py` to test the model on a subset of the data.

   ```sh
   python scripts/test_on_a_subset.py
   ```

## Data Preparation

### Collecting Images

1. Collect images for the dataset. Ensure the images are in JPEG format and do not exceed 250kB each.
2. Place them in the `data/training/` directory.

### Preprocessing

1. **Resizing Images**: Use the `resize.py` script to resize the images to a standard shape suitable for training.

    ```sh
    python scripts/resize.py
    ```

### Dataset Preparation

The `create_dataset.py` script contains functions to convert images to grayscale, pixelate images, and create datasets suitable for training the model.

## Model Architecture

The model is implemented in `architecture.py` and consists of a ResNet architecture with several residual blocks. The main components are:
- Initial convolutional layer
- Residual blocks
- Output convolutional layer

## Training the Model

The `main.py` script is used to train the model. It includes the following functions:
- `get_dataset()`: Loads and splits the dataset into training and validation sets.
- `training_step()`: Performs a single training step.
- `eval_step()`: Performs a single evaluation step.
- `training_loop()`: Manages the overall training loop.

## Generating and Visualizing Predictions

After training the model, use `get_predictions.py` to generate predictions on the test set and `visualize_predictions.py` to visualize these predictions.