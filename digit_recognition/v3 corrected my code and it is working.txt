import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('digit_recognition/handwritten_model.keras')

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        
        # added resizing process to coreect errors
        img = cv2.resize(img, (28, 28)) 
        img = np.invert(np.array([img]))

        #adding normalization process
        img = img/255.0
        
        prediction = model.predict(img)
        print(f"This digit maybe is  a {np.argmax(prediction)}")

        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()

    except Exception as e:
        print(f"Error:{e}")
    finally:
        image_number += 1
