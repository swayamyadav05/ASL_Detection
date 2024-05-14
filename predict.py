import mediapipe as mp
import pickle
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_hands=2)
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

def prediction(frame, results):

    data_aux = []
    x_ = []
    y_ = []
    H, W, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        '''x1 = int(min(x_) * W)-20
        y1 = int(min(y_) * H)-20
        x2 = int(max(x_) * W)+20
        y2 = int(max(y_) * H)+20'''

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]
        return predicted_character







