import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Function to preprocess the image
def load_and_preprocess_image(image_path):
    # Load image
    response = requests.get(image_path)
    image = Image.open(BytesIO(response.content))
    image = image.resize((224, 224))

    # Convert to numpy array
    image = np.array(image)
    # Scale image pixels
    image = image / 255.0
    # Expand dimensions to fit the model input
    image = np.expand_dims(image, axis=0)
    return image

def get_labels_from_image(image_url):

    # URL of the image
    #image_url = "https://i.pinimg.com/564x/b3/9b/fa/b39bfa3fa1e84058dc94a5326446c1dc.jpg"

    # Preprocess the image
    image = load_and_preprocess_image(image_url)

    # Load the MobileNetV2 model
    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=(224,224,3))
    ])

    # Predict the image
    predictions = model.predict(image)
    labels = []
    for prediction in predictions:
        # get the args of the top 5 predictions
        predicted_labels = np.argsort(prediction)[-5:][::-1]
        print(predicted_labels)
        for label in predicted_labels:
            # Decode the predictions
            labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
            imagenet_labels = np.array(open(labels_path).read().splitlines())
            # Print the top label
            print("Predicted label:", imagenet_labels[label])
            labels.append(imagenet_labels[label])
    return labels