# Convolutional Neural Network (CNN) for Dog & Cat Classification

This project contains a Convolutional Neural Network (CNN) model for classifying images of dogs and cats. The model is trained on a dataset of labeled dog and cat images and can predict whether a given image contains a dog or a cat.

## Dependencies

- Python 3.x
- Numpy
- Pandas
- Tensorflow
- Keras
- PIL (Python Imaging Library)

## Dataset

The dataset used for training the CNN model consists of two classes: "dog" and "cat." The dataset is stored in separate folders for each class: dog/ and cat/. Additionally, there is a test folder test/ containing images for prediction.

## Setup

Install the required dependencies mentioned above.
Clone this repository or download the code files.

## Usage

Set the paths to the image folders dog_folder, cat_folder, and test_folder in the code.
Adjust the image_size if needed to match the input size of the CNN model.
Run the script CCN_Dogs_Cats.py to train the model on the provided dataset.

## Model Architecture

The CNN model architecture consists of three convolutional layers, each followed by batch normalization and max-pooling. After that, the flattened feature maps are passed through three fully connected layers, and finally, an output layer is used to predict the class probabilities.

## Training

The model is trained using the Adam optimizer and Binary Crossentropy loss function. The training process involves 10 epochs with a batch size of 32.

## Evaluation

The accuracy of the model on the test dataset is evaluated after training, and the predictions are saved in the results DataFrame.

## Predictions

To make predictions on new images, place the images in the test/ folder and run the model. The predictions will be saved in the results DataFrame.

## Results

The results DataFrame contains the predicted labels for the test images along with the corresponding image filenames.
