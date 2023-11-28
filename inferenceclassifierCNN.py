import cv2
import mediapipe as mp
import os
import pickle
import h5py
import numpy as np
from time import time
from keras import utils
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('C:/Users/Jared XIN/Downloads/Compsci Code/CS-IA-/model4abc.keras' )


labels_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11 : 'L', 12 : 'M', 13 : 'N', 14 : 'O', 15 : 'P', 16   : 'Q', 17 : '   R', 18 : 'S', 19 : ' T', 20 : ' U', 21 : ' V', 22 : ' W', 23 : ' X', 24 : ' Y', 25 : ' Z'}
classes = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
           'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S','space', 'T', 'U', 'V', 
           'W', 'X', 'Y', 'Z' ]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands()
 


 
cap = cv2.VideoCapture(0)
while(True):
   


    # Capture the video frame
    # by frame
    x_ = []
    likelinessarray = []
    y_ = []
    x_crop = []
    y_crop = []
    data_aux = []
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape
    

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                    frame, #draw what?
                    hand_landmarks, #model output
                    mp_hands.HAND_CONNECTIONS, #hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        for hand_landmarks in results.multi_hand_landmarks:        
            for i in range(len(hand_landmarks.landmark)):
                #print(hand_landmarks.landmark[i])
                xcrop = hand_landmarks.landmark[i].x
                ycrop = hand_landmarks.landmark[i].y
                x_crop.append(xcrop)
                y_crop.append(ycrop)
        x1crop = int(min(x_crop) * W) - 100
        y1crop = int(min(y_crop) * H) - 100


        x2crop = int(max(x_crop) * W) + 100
        y2crop = int(max(y_crop) * H) + 100
        if x1crop < 0:
            x1crop = 0
        if x2crop < 0:
            x2crop = 0
        if y1crop < 0:
            y1crop = 0
        if y2crop < 0:
            y2crop = 0
        #print(x1crop, y1crop, x2crop, y2crop)
        ret, frame_rgb = cap.read()
        ogframe = frame_rgb
        #frame_rgb = np.array(frame_rgb)
        try:
            frame_rgb1 = frame_rgb[y1crop:y2crop, x1crop:x2crop]
            #cv2.imshow('frame_rgb',frame_rgb1)
        except cv2.error:
            print("")
        #frame_rgb2 = cv2.cvtColor(frame_rgb1, cv2.COLOR_BGR2RGB)
        
        results = hands.process(frame_rgb1)
        #print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            #print("Hello, hand!")


           
        
            for hand_landmarks in results.multi_hand_landmarks:        
                for i in range(len(hand_landmarks.landmark)):
                    #print(hand_landmarks.landmark[i])
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10


            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10



            size = 32,32
            tests = []
            sized = 200,200

            frame200 = cv2.resize(frame_rgb1, sized)
            cv2.imshow("200",frame200)
            framepredict = cv2.resize(frame200, size)  
            #cv2.imshow("image",framepredict)
            tests.append(framepredict)
            tests = np.array(tests)
            tests = tests.astype('float32')/255.0
            
            try:
                prediction = model.predict(tests)
            except ValueError:
                print("...")
            #print(prediction)
            print(np.argmax(prediction,axis=1))
            predicted_character = classes[np.argmax(prediction,axis=1)[0]]
            likelinessarray.append(predicted_character)
            test = []
            #print(predicted_character)


            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

            #cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            if cv2.waitKey(1) & 0xFF == ord('e'):
                
                print((max(set(likelinessarray), key=likelinessarray.count)))    
                likelinessarray = []
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break
    
       

    
    cv2.imshow('frame',frame)
    cv2.waitKey(10)

#WAY TOO INACCURATE
# MNIST DATA SET?
