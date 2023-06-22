import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
import tensorflow_hub as hub
from tensorflow import keras
import base64
from PIL import Image
import io
import numpy as np

def base64_to_bytes(base64_string):
    base64_string = base64_string.split(',')[1]
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    return image

def bytes_to_np_array(image):
    np_array = keras.preprocessing.image.img_to_array(image)
    np_array /= 255
    return np_array

def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.reshape(image,[-1, 224,224,3])
    return image

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile = False)
    return model

def predict_image(model, image):
    decoded_predictions = model.predict(image)
    return decoded_predictions

def process_image(model_path, img_str):
    img = base64_to_bytes(img_str)
    img = bytes_to_np_array(img)
    img = preprocess_image(img)
    mod = load_model(model_path)
    res = predict_image(mod, img)
    return res

def output(result, age):
    cats = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
    prob = {}
    for i, p in enumerate(result[0]):
        prob[cats[i]] = float(p)
    m = max(prob, key=prob.get)
    o = 'yes'
    if age < 18:
        if m == 'hentai' or m == 'porn':
            o = 'no'
    return o

def main(model_path, img_str, age=1):
    res = process_image(model_path, img_str)
    out = output(res, age)
    return out
