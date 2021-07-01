import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
def yoyo(img, s="bird_classifier_model.h5"):
    model = tf.keras.models.load_model(s)
    #creating an input data in the required shape
    data = np.ndarray(shape=(1,224,224,3),dtype = np.float32)
    #resizing the image
    image = img
    image = ImageOps.fit(image,(224,224))

    img_array = np.asarray(image)

    #no need to normalise our model Efficient net will do it automatically
    data[0] = img_array
    prediction = model.predict(data)
    p = np.argmax(prediction)

    return p