import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb, #draw what?
                    hand_landmarks, #model output
                    mp_hands.HAND_CONNECTIONS, #hadn connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                for i in range(len(hand_landmarks.landmark)):
                    print(hand_landmarks.landmark[i])
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                data.append(data_aux)
                labels.append(dir)
                



#        plt.figure()
#        plt.imshow(img_rgb)
#plt.show()


f = open('data.pickle','wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()