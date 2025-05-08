# Khanh Nguyen Cong Tran
# 1002046419


import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist

def create_and_train_model(training_inputs, training_labels, blocks, 
                               filter_size, filter_number, region_size, 
                               epochs, cnn_activation) -> any: 

    # Argument training_inputs is a numpy array (you decide the shape), where training_inputs[i] is the i-th training input pattern.

    # Argument training_labels is a numpy column vector. That means it is a 2D numpy array, with a single column. 
    #   The value stored in training_labels[i,0] is the class label for the input pattern stored at training_inputs[i].

    # Argument blocks specifies how many convolutional layers your model will have. Every convolutional layer must be followed by a max pool layer. 
    #   The total number of layers should be 2*blocks + 2, to account for the input layer, and the output layer. 
    #   The output layer should be fully connected and use the softmax activation function. 
    #   Except for the output layer, there should be no other fully connected layers.

    # Argument filter_size specifies the number of rows of each 2D convolutional filter. 
    #   The number of columns should be equal to the number of rows, so it is also specified by filter_size. 
    #   For example, if filter_size = 3, each 2D filter has 3 rows and 3 columns.

    # Argument filter_number specifies the number of 2D convolutional filters used at each convolutional layer. 
    #   For example, if filter_number = 5, each convolutional layer applies 5 2D filters.

    # Argument region_size specifies the size of the region for the max pool layer. 
    #   For example, if region_size = 2, each output of a max pool layer should be the max value of a 2x2 region (i.e., 2 rows and 2 columns).

    # Argument epochs specifies the number of epochs (i.e., number of training rounds) for the training process.

    # Argument cnn_activation specifies the activation function that should be used in convolutional layers. 
    #   This argument is a single string (not a list), that can be "sigmoid", "tanh", or "relu". 
    #   All convolutional layers will have the same activation function.

    # Return value model is the model that the function has created and trained.

    input_shape = training_inputs[0].shape
    number_of_classes = np.max(training_labels) + 1

    # create the model
    model = tf.keras.Sequential()

    # add the input layer
    model.add(tf.keras.layers.Input(shape=input_shape))

    #add the blocks 
    for _ in range(blocks): 
        #each block will have a convolutional layer followed by a max pooling layer
        model.add(tf.keras.layers.Conv2D(filter_number, kernel_size=(filter_size, filter_size), activation=cnn_activation))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(region_size, region_size)))

    # add the output layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation='softmax'))

    # compile
    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                  metrics=['accuracy'])

    model.summary()

    # training
    model.fit(training_inputs, training_labels, epochs=epochs)

    return model


def load_mnist() -> tuple: 

    #laod the mnist dataset
    training_set, test_set = mnist.load_data()
    training_data, training_labels = training_set
    test_data, test_labels = test_set

    # find the absolute max value
    absolute_max_value = np.max(np.abs(training_data))

    #normalized the training and test set
    n_training_data = training_data / absolute_max_value
    n_test_data = test_data / absolute_max_value

    # resahape into a 4d matrix
    fourD_training_data = np.expand_dims(n_training_data, -1)
    fourD_test_data = np.expand_dims(n_test_data, -1)

    return (fourD_training_data, training_labels, fourD_test_data, test_labels)