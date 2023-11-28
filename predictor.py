import os
import cv2
import pickle
import numpy as np
from time import time
from keras import utils
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
           'W', 'X', 'Y', 'Z', 'nothing', 'space', 'del']
# Load the model

#with open ('CS-IA-/data.pickle', 'rb') as f:
#    test = pickle.load(f)

#x_train = test[0]
#x_test = test[1]
#y_train = test[2]
#y_test = test[3]


model = load_model('model.h5')
    
batch = 128
epochs = 5
learning_rate = 0.001
adam = Adam(lr=learning_rate)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
#c:\Users\Luo\Downloads\D10.jpgclasses = 29
batch = 128
epochs = 2
learning_rate = 0.001

#history = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=0.1, shuffle = True, verbose=1)
from keras.preprocessing import image
import keras.utils as image
# Load the image
img = image.load_img("6.jpg", target_size=(32, 32))

#img1 = image.load_img("CNNtranslator/W11.png", target_size=(32, 32))
# Convert the image to a numpy array





img_array = image.img_to_array(img)
#img_array = img_array.reshape(32, 32)  # Assumes the input shape is (width, height, 3)
#img_array1 = image.img_to_array(img1)
# Add an extra dimension because the model expects batches of images
img_batch = np.expand_dims(img_array, axis=0)
#img_batch2 = np.expand_dims(img_array1, axis=0)
# Normalize image
#img_batch /= 255.
#img_batch2 /= 255.
# Predict
prediction = (model.predict(img_batch))   
#prediction1 = (model.predict(img_batch2))   

print(prediction)
#print(prediction1)

predicted_class = np.argmax(prediction)

# Map the predicted class with its corresponding label
predicted_label = classes[predicted_class]

#max_index1 = np.argmax(prediction1)


print("Predicted class: ", predicted_label)
#print("Predicted class: ", classes[max_index1])



# Create a list of image paths
image_paths = ["6.jpg", "2272.jpg","5.jpg","4.jpg","3.jpg"]

# Load and preprocess the images
image_list = []
for path in image_paths:
    img = image.load_img(path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    image_list.append(img_batch)

# Stack the image arrays
image_batch = np.vstack(image_list)

# Normalize image
image_batch /= 255.

# Predict
predictions = model.predict(image_batch)

# Get predicted labels
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = [classes[i] for i in predicted_classes]

print(predicted_labels)

folder_path = ("B")
image_list = []
label_list = []

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  
        img_path = os.path.join(folder_path, filename)
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        image_list.append(img_array)
        label_list.append(filename)

image_array = np.array(image_list)
label_array = np.array(label_list)

image_array = image_array / 255.0


for i in range(len(image_array)-2900):
    img_batch = np.expand_dims(image_array[i], axis=0)
    prediction = model.predict(img_batch)
    predicted_class = np.argmax(prediction)
    predicted_label = classes[predicted_class]

    print("Image:", i+1)
    print("Label:", label_array[i])
    print("Predicted class:", predicted_label)
    print("--------------------------------------")