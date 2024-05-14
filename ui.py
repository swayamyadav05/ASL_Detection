import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
from predict import *


def click():
    var.set(txt)
    mylabel.config(text=var)
    mylabel.after(10,click)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_hands=1)

# Initialize Tkinter window
window = tk.Tk()
window.title("Hand Landmark Detection")

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)


# Set up function to continuously draw hand landmarks on video frames
def draw_hand_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Set up function to continuously update video feed
def update_video_feed():
    _, frame = cap.read()
    frame.flags.writeable = False
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    draw_hand_landmarks(frame_rgb, results)
    frame.flags.writeable = True

    global txt
    txt=prediction(frame, results)
    image = Image.fromarray(frame_rgb)
    image = ImageTk.PhotoImage(image=image)
    label.configure(image=image)
    label.image = image
    print(txt)
    window.after(10, update_video_feed)


var=tk.StringVar()
myfont=('times',52,'bold')
mylabel=tk.Label(window,textvariable=var,font=myfont)
mylabel.pack()
var.set("hello")
button=tk.Button(window,text="click",command=click)
button.pack()
# Set up video display in Tkinter window
label = tk.Label(window)
label.pack()

# Start updating video feed
update_video_feed()

# Start Tkinter event loop
window.mainloop()
