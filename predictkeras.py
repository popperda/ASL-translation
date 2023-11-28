import os
import cv2
import pickle
import numpy as np
from time import time
#from keras import utils
#from keras.optimizers import Adam
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model
classes = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
           'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S','space', 'T', 'U', 'V', 
           'W', 'X', 'Y', 'Z' ]
# Load the model
#pitrain_dir = 'CNNtranslator/asl_alphabet_train/asl_alphabet_train'
test_dir = '6'

#with open ('CNNtranslator/data.pickle', 'rb') as f:
#    test = pickle.load(f)

#x_train = test[0]
#x_test = test[1]
##y_train = test[2]
#y_test = test[3]


# THIS IS WHERE WE INPUT DATA
size = 32,32
tests = []
for image1 in os.listdir(test_dir):
    temp_img = cv2.imread(test_dir + '/' + image1)
    temp_img = cv2.resize(temp_img, size)  
    tests.append(temp_img)
tests = np.array(tests)
tests = tests.astype('float32')/255.0



model = load_model('model4abc.keras')

#c:\Users\Luo\Downloads\D10.jpgclasses = 29
batch = 128
epochs = 2
learning_rate = 0.001

#history = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_split=0.1, shuffle = True, verbose=1)
from keras.preprocessing import image

# Load the image

# how to predict
predictiontest = model.predict(tests)
print(np.argmax(predictiontest,axis =1))

for n in range (0,predictiontest.size-2):

    print(classes[np.argmax(predictiontest,axis =1)[n]])