import streamlit as st
import pandas as pd
import numpy as np
import tensorflow.keras
from PIL import Image, ImageOps

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#open labels.txt file which has the labels of the model output
with open('labels.txt','r') as f:
    labels = f.read().splitlines()

#labels dictionary 
labels_dict = {int(label.split()[0]): label.split()[1] for label in labels}
#st.text(labels_dict)

#Page containers
siteHeader = st.beta_container()
ClassifyingImage = st.beta_container()

with siteHeader:
    st.title('ICA-LAB project!')
    st.text('''In this project, a deep learning model is applied to a set of canyoning knots images.

To test this page, click on "Browse files" and select a list of canyoning knots images.

At the moment, the available knots are:
            - ''')
    

with ClassifyingImage:
    st.header('Select a list of images')

# Replace this with the path to your image
uploaded_files = st.file_uploader("", accept_multiple_files=True)
if len(uploaded_files) > 0:
    cols = st.beta_columns(len(uploaded_files))

    for uploaded_file, col in zip(uploaded_files, cols):
        image = Image.open(uploaded_file)

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)

        # just accept predictions with 75% certaintly
        if max(max(prediction)) > 0.75:
            # getting the maximum value index
            idx_prediction = np.argmax(prediction)
            result = "{}".format(labels_dict[idx_prediction])
        else:
            result = "Unknown"

        # display the resized image
        with col:
            st.header(result)
            st.image(image, use_column_width=True)


