from flask import Flask, render_template, Response
import cv2
import warnings
warnings.simplefilter('ignore')
#Initialize the Flask app
app = Flask(__name__)

camera = cv2.VideoCapture(0)
'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''
#def process():

import mediapipe as mp
import pickle
import numpy as np


model_dict = pickle.load(open('./model.p','rb'))
model = model_dict['model']

def grammar_check(text):
    return text

import keyboard
labels_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11 : 'L', 12 : 'M', 13 : 'N', 14 : 'O', 15 : 'P', 16   : 'Q', 17 : '   R', 18 : 'S', 19 : ' T', 20 : ' U', 21 : ' V', 22 : ' W', 23 : ' X', 24 : ' Y', 25 : ' Z'}


message = []
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands()

def predict():
    success, frame = camera.read()  # read the camera frame

        

    # Capture the video frame
    # by frame
    x_ = []
    likelinessarray = []
    y_ = []
    x_crop = []
    y_crop = []
    data_aux = []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape

    try:
        results = hands.process(frame_rgb)
    except ValueError:
        results = None
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
                # (hand_landmarks.landmark[i])
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
        ret, frame_rgb = camera.read()
        #frame_rgb = np.array(frame_rgb)
        try:
            frame_rgb1 = frame_rgb[y1crop:y2crop, x1crop:x2crop]
            #cv2.imshow('frame_rgb',frame_rgb1)
        except cv2.error:
            pass
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



            
            try:
                prediction = model.predict([np.asarray(data_aux)])
                #print(labels_dict[int(prediction[0])])
            except ValueError:
                prediction = [0,0]
            predicted_character = labels_dict[int(prediction[0])]
            return(predicted_character)
            #likelinessarray.append(predicted_character)


def gen_frames():  
    message = []
    predicted_character = ""
    string = ""
    while True:
        success, frame = camera.read()  # read the camera frame

        

    # Capture the video frame
    # by frame
        x_ = []
        likelinessarray = []
        y_ = []
        x_crop = []
        y_crop = []
        data_aux = []
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
            ret, frame_rgb = camera.read()
            #frame_rgb = np.array(frame_rgb)
            try:
                frame_rgb1 = frame_rgb[y1crop:y2crop, x1crop:x2crop]
                #cv2.imshow('frame_rgb',frame_rgb1)
            except cv2.error:
                pass
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



                
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    #print(labels_dict[int(prediction[0])])
                except ValueError:
                    pass
                predicted_character = labels_dict[int(prediction[0])]
                likelinessarray.append(predicted_character)

                #print(predicted_character)


                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                if keyboard.is_pressed('e'):
                    message.append(predicted_character.strip())
                    textsize = cv2.getTextSize((max(set(likelinessarray), key=likelinessarray.count)), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_X_coord = (frame.shape[1] - textsize[0]) // 2
                    #print((max(set(likelinessarray), key=likelinessarray.count)))  
                    print(message) 
                    string = "".join(message)
                    #cv2.putText(frame, string, (text_X_coord, 270),
                     #      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 
                    #likelinessarray = []
                if keyboard.is_pressed('r'):
                    message = []
                    string = ""
                if keyboard.is_pressed('f'):
                    message.append(" ")
                if keyboard.is_pressed('backspace'):
                    try:
                        message.pop()
                    except IndexError:
                        print("Keyboard")
                
        if not success:
            break
        else:
            
            stringws = "".join(message)
            cv2.putText(frame, stringws, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def update_text():
    print("Hello")
    global message
    text =""
    try:
        text = predict()
    except ValueError:
        print("Try again")
    print("guess") 
    print(text) 
    message.append(text)
    #massage = " ".join(message)
    return render_template('index.html', message=message)



@app.route('/video_feed')
def video_feed():
    
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)