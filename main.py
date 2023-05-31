#https://data-flair.training/blogs/cats-dogs-classification-deep-learning-project-beginners/
#Remember that this code is runned in tensor flow, alternative you can use the google open program
# Colab




#Importing the Libraries that is needed.

'''
Summaries of text the Libaries that is needed for the project.

    numpy: Used for numerical operations and array manipulation.
    pandas: Used for data manipulation and analysis, especially for handling tabular data.
    keras.preprocessing.image: Used for image preprocessing tasks such as loading and augmenting images.
    keras.utils: Used for converting labels into categorical format.
    sklearn.model_selection: Used for splitting the dataset into training and testing sets.
    matplotlib.pyplot: Used for visualizing images and plots.
    random: Used for generating random numbers or selecting random elements from a list.
    os: Used for interacting with the operating system, such as reading file directories and paths.



I was a little bit confused why i need numpy so i asked chatgpt for the answer:

    Array Manipulation: numpy provides efficient data structures and functions for working with multidimensional arrays. In the context of image classification, you can use numpy arrays to store and manipulate image data.

    Numerical Operations: numpy offers a wide range of mathematical and numerical operations that are essential for preprocessing and manipulating image data. This includes operations such as resizing, normalizing, and transforming images.

    Compatibility with Keras: Keras, the library you're using for building the classifier, relies heavily on numpy arrays as input for training and inference. Therefore, you need numpy to convert and prepare your data in a format compatible with Keras.

Overall, numpy is a fundamental library for scientific computing in Python and is particularly useful in machine learning projects that involve numerical operations and array manipulation, such as image classification.

'''
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os



#2: Define Image properties
"""
    Image_Width = 128: This line sets the width of the images to 128 pixels.
    Image_Height = 128: This line sets the height of the images to 128 pixels.
    Image_Size = (Image_Width, Image_Height): This line creates a tuple Image_Size that stores the dimensions of the images, using the values of Image_Width and Image_Height.
    Image_Channels = 3: This line indicates that the images have 3 color channels, typically representing the red, green, and blue color channels (RGB).

"""

Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width,Image_Height)
Image_Channels=3 #The color Channels (RGB)

#3 Prepare dataset for Training Model

filenames = os.listdir("./dogs-vs-cat/train") #Getting the file by OS command

#Create a Dataframe for input and output
categories = []
for f_name in filenames:
    category = f_name.split('.')[0] #Split filename by period and assign first part as category label.
    if category =='cat':
        categories.append(1)
    else:
        categories.append(0)

    df=pd.DataFrame({
        'filename': filenames,
        'category': categories #Using the iformation that was in the foor loop
    })




#4 Creating the Neural net Model
'''
I asked the ChatGpt about the imported from keras model what they do and this is the answer i got 

In the code snippet you provided, the Keras library is being imported, and specific modules and classes are being imported from Keras. Here's what each line represents:

    from keras.models import Sequential: This line imports the Sequential class from Keras. The Sequential class is a linear stack of layers, allowing you to build a deep learning model by stacking layers on top of each other.

    from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization: This line imports various layer classes from Keras that will be used to construct the model. Here's a breakdown of each layer class:

        Conv2D: This layer represents a two-dimensional convolutional layer. It performs convolution operations on the input data to extract features.

        MaxPooling2D: This layer performs max pooling operations on the input data. It helps reduce the spatial dimensions of the data while retaining the most important features.

        Dropout: This layer applies dropout regularization to the input data. It randomly sets a fraction of input units to 0 during training to prevent overfitting.

        Flatten: This layer flattens the input data into a 1-dimensional array. It is typically used to transition from convolutional layers to fully connected layers.

        Dense: This layer represents a fully connected layer. It connects every neuron in the current layer to every neuron in the subsequent layer.

        Activation: This layer applies an activation function to the output of the previous layer. Common activation functions include ReLU, sigmoid, and softmax.

        BatchNormalization: This layer performs batch normalization on the input data. It normalizes the activations of the previous layer, improving the stability and performance of the model.

These imported modules and classes from Keras will be used to define the architecture of the deep learning model for the cat vs. dog classification task.

'''
#Imported libraries from Kreas
from keras.models import Sequential # Allows me to build layer of classess to construct a model
from keras.layers import Conv2D,MaxPooling2D, \
    Dropout,Flatten,Dense,Activation,\
    BatchNormalization


model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape =(Image_Width,Image_Height,Image_Channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization)
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

model.complie(loss='categorical_crosstropy',
              optimizer='rmsprop',metrices=['accuracy'])

#5 Analyzing the Model:
model.summary()


#6 Define Callbacks and learning rate:
from keras.callacks import EarlyStopping, ReduceLROnPlateau # Callback functions
earlystop = EarlyStopping(patience=10) #Patience that stops at value 10 validation loss
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 2, verbose = 1 , factor = 0.5, min_lr=0.0001)
callbacks = [earlystop, learning_rate_reduction]