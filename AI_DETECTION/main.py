import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
print(tf.__version__)
img = cv2.imread('images/woman_cat.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.pad(img, ((158, 158), (0, 0), (0, 0)), mode='edge')
plt.imshow(img)

# Define the VGG16 model architecture
model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(244, 244, 3))

def predict(img):
    im = cv2.resize(img, (244, 244))
    im = tf.keras.applications.vgg16.preprocess_input(im)
    pr = model.predict(np.expand_dims(im, axis=0))[0]
    return np.sum(pr[281:294])

def predict_map(img, n):
    dx = img.shape[0] // n
    res = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            im = img[dx*i:dx*(i+1), dx*j:dx*(j+1)]
            r = predict(im)
            res[i, j] = r
    return res

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[1].imshow(img)
ax[0].imshow(predict_map(img, 10))
plt.show()
