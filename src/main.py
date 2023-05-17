""" main.py
Author: 
    Anton Drasbæk Schiønning (202008161), GitHub: @drasbaek

Desc:
    This script contains the code for a VGG16 based model for classifying images of Indonesian fashion.

Usage:
    $ python src/main.py
"""

# general packages
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json

# tf tools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 VGG16)
from tensorflow.keras.layers import (Flatten, 
                                     Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


def define_paths():
    '''
    Define paths for input and output data.
    
    Returns:
    -   inpath (pathlib.PosixPath): Path to input data.
    -   outpath (pathlib.PosixPath): Path to where the classified data should be saved.
    '''

    # define paths
    path = Path(__file__)

    # define input dir
    inpath = path.parents[1] / "images"

    # define output dir
    outpath = path.parents[1] / "out"

    return inpath, outpath

def make_dataframe_from_json(inpath, data_subset):
    '''
    Loads json file and turns it into a dataframe 
    Inspired from https://www.kaggle.com/code/vencerlanz09/indo-fashion-classification-using-efficientnetb0

    Args:
    -   inpath (pathlib.PosixPath): Path to input data.
    -   data_subset (str): Which subset of the data to load (train, val, test).

    Returns:
    -   data (pandas.DataFrame): Dataframe containing the data.
    '''

    # define paths
    path = inpath / "metadata" / f"{data_subset}_data.json"

    # load JSON data into dictionaries
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))

    # convert dictionaries to dataframe
    data = pd.DataFrame(data)

    return data


def load_image_data(label_data, batch_size=32, shuffle=True):
    '''
    Uses the ImageDataGenerator to load images specified in the dataframe.
    It rescales the images and uses the preprocess_input function from the VGG16 model.
    This function saves memory by only loading one batch of images at a time.

    Args:
    -   label_data (pandas.DataFrame): Dataframe containing the data.
    -   batch_size (int): Size of the batch (defaults to 32).
    -   shuffle (bool): Whether to shuffle the data (defaults to True).

    Returns:
    -   images (tensorflow.python.keras.preprocessing.image.DataFrameIterator): Iterator containing the images.
    '''

    # create an instance of the ImageDataGenerator class for each dataset
    generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    # create a generator
    images = generator.flow_from_dataframe(
    dataframe=label_data,
    x_col='image_path',
    y_col='class_label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=shuffle
)
    return images


def build_model():
    '''
    Builds the neural network model using the VGG16 model as a base.

    Returns:
    -   model (tensorflow.python.keras.engine.functional.Functional): Model object.

    '''
    
    # define model without top layer
    print("Building model...")
    model = VGG16(weights="imagenet", include_top=False, pooling = 'avg', input_shape=(224, 224, 3))

    # disable convolutional layers
    for layer in model.layers:
        layer.trainable = False

    # add new classification layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu')(flat1)
    class2 = Dense(64, activation='relu')(class1)
    output = Dense(15, activation='softmax')(class2)

    # define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    # define inputs and outputs
    model = Model(inputs=model.inputs, outputs=output)

    # compile model
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def fit_model(model, train_images, val_images, outpath):
    '''
    Fits model to training data and saves the model weights.
    Uses early stopping to prevent overfitting.

    Args:
    -   model (tensorflow.python.keras.engine.functional.Functional): Model object.
    -   train_images (tensorflow.python.keras.preprocessing.image.DataFrameIterator): Iterator containing the training images.
    -   val_images (tensorflow.python.keras.preprocessing.image.DataFrameIterator): Iterator containing the validation images.
    -   outpath (pathlib.PosixPath): Path to where the model weights should be saved.

    Returns:
    -   history (tensorflow.python.keras.callbacks.History): History object containing the training and validation loss.
    '''

    # fit model with early stopping
    print("Fitting model...")
    history = model.fit(train_images,
                        epochs=10,
                        batch_size=64,
                        validation_data=val_images,
                        verbose=1,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
    
    # save model weights
    model.save_weights(outpath / "indofashion_model_weights.h5")
    
    return history


def plot_history(history, outpath):
    '''
    Creates and saves a plot displaying the training and validation loss.

    Args:
    -   history (tensorflow.python.keras.callbacks.History): History object containing the training and validation loss.
    -   outpath (pathlib.PosixPath): Path to where the plots should be saved.
    '''

    # plot training loss history
    plt.plot(history.history['loss'], label='train')

    # plot validation loss history
    plt.plot(history.history['val_loss'], label='val')

    # add title
    plt.title('Training and validation loss')

    # add x label
    plt.xlabel('Epoch')

    # add y label
    plt.ylabel('Loss')

    # add legend
    plt.legend()

    # save plot
    plt.savefig(outpath / "loss.png")


def save_classification_report(outpath, y_true, y_pred, target_names):
    '''
    Saves a classification report as a text file.

    Args:
    -   outpath (pathlib.PosixPath): Path to where the classification report should be saved.
    -   y_true (numpy.ndarray): Array containing the true labels.
    -   y_pred (numpy.ndarray): Array containing the predicted labels.
    -   target_names (list): List containing the names of the classes (neccessary for the classification to look nice)

    '''

    # create classification report
    report = classification_report(y_true, y_pred, target_names=target_names)

    # save report
    with open(outpath / "classification_report.txt", "w") as f:
        f.write(report)


def main():
    # define paths
    inpath, outpath = define_paths()

    # load labels
    print("Loading labels...")
    train_data = make_dataframe_from_json(inpath, "train")
    val_data = make_dataframe_from_json(inpath, "val")
    test_data = make_dataframe_from_json(inpath, "test")

    # load image data
    print("Loading image data...")
    train_images = load_image_data(train_data)
    val_images = load_image_data(val_data)
    test_images = load_image_data(test_data, shuffle = False) # we make sure to not shuffle the test data

    # build model
    model = build_model()

    # fit model
    history = fit_model(model, train_images, val_images, outpath)
    
    # plot history
    plot_history(history, outpath)

    # predict
    predictions = model.predict(test_images)

    # save classification report
    save_classification_report(outpath, test_images.classes, predictions.argmax(axis=1), target_names=test_images.class_indices.keys())

    print("Run Finished! Output saved to: " + str(outpath))

if __name__ == "__main__":
    # run main
    main()