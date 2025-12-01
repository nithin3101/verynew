#*****************************
#TRANSFER LEARNING
#***************************


from google.colab import drive
drive.mount('/content/drive')

import zipfile

zip_path = "/content/drive/MyDrive/Tomato_Leaf_Disease_Classification_Dataset.zip"
extract_path = "/content/dataset/"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset extracted to:", extract_path)

import os
import numpy as np

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from keras.models import load_model
from keras import backend as K

from io import BytesIO
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import colors

import requests

# Load model
model = load_model('/content/drive/MyDrive/transfer_densenet169_tomato.hdf5')

# Image generator
eval_datagen = ImageDataGenerator(rescale=1./255)
eval_dir = '/content/dataset/Tomato_Leaf_Disease_Classification_Dataset/valid'

eval_generator = eval_datagen.flow_from_directory(
    eval_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate model
loss = model.evaluate(eval_generator, steps=len(eval_generator))

# Print metric scores
for index, name in enumerate(model.metrics_names):
    print(name, loss[index])


# Utility functions
classes = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot','Spider_mites Two-spotted_spider_mite','Target_spot','Tomato_Yellow_Leaf_Curl_Virus','Tomato_mosaic_virus','healthy','powdery_mildew']
# Preprocess the input
# Rescale the values to the same range that was used during training 
def preprocess_input(x):
    x = img_to_array(x) / 255.
    return np.expand_dims(x, axis=0) 

# Prediction for an image path in the local directory
def predict_from_image_path(image_path):
    return predict_image(load_img(image_path, target_size=(224, 224)))

# Prediction for an image URL path
def predict_from_image_url(image_url):
    res = requests.get(image_url)
    im = Image.open(BytesIO(res.content))
    return predict_from_image_path(im.fp)
    
# Predict an image
def predict_image(im):
    x = preprocess_input(im)
    pred = np.argmax(model.predict(x))
    return pred, classes[pred]

# -----------------------------
# Grad-CAM for DenseNet169
# -----------------------------
def grad_CAM(image_path, last_conv_layer_name='conv5_block32_concat', alpha=0.5):
    # Load image
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img) / 255.
    x = np.expand_dims(x, axis=0)

    # Get last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    
    # Create model mapping input -> last conv layer output + predictions
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of top predicted class w.r.t conv outputs
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    # Weight the conv outputs by pooled grads
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Superimpose heatmap on original image
    orig_img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(orig_img, 1-alpha, heatmap, alpha, 0)

    plt.figure(figsize=(12,8))
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


    # Predict a local image
idx, label = predict_from_image_path('/content/dataset/Tomato_Leaf_Disease_Classification_Dataset/valid/Target_Spot/003a5321-0430-42dd-a38d-30ac4563f4ba___Com.G_TgS_FL 8121_180deg.JPG')
print(idx, label)

# Show Grad-CAM
grad_CAM('/content/dataset/Tomato_Leaf_Disease_Classification_Dataset/valid/Target_Spot/003a5321-0430-42dd-a38d-30ac4563f4ba___Com.G_TgS_FL 8121_180deg.JPG')


import os

# Base folder of your test dataset
base_folder = '/content/dataset/Tomato_Leaf_Disease_Classification_Dataset/valid'

for i, c in enumerate(classes):
    folder = os.path.join(base_folder, c)
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        continue  # Skip missing class folders

    count = 0
    for file in os.listdir(folder):
        if file.lower().endswith('.jpeg'):
            image_path = os.path.join(folder, file)
            p, class_name = predict_from_image_path(image_path)
            
            if p == i:
                print(file, p, class_name)
            else:
                print(file, p, class_name, '**INCORRECT PREDICTION**')
                grad_CAM(image_path)

            count += 1
            if count == 100:
                break

