import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

# Load MoveNet model
# Ensure you have a MoveNet model saved locally. Replace 'movenet_model.tflite' with your model's file path.
movenet_model = 'movenet_model.tflite'
interpreter = tf.lite.Interpreter(model_path=movenet_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def movenet_inference(image):
    input_image = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])[0]
    return keypoints

def draw_keypoints(image, keypoints):
    h, w, _ = image.shape
    for kp in keypoints:
        x, y = int(kp[1] * w), int(kp[0] * h)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image

st.title('Hand Gesture Recognition using MoveNet')

st.sidebar.title("Select Input Source")
input_source = st.sidebar.radio("Input Source", ("Webcam", "Video File"))

if input_source == "Webcam":
    st.write("Using webcam for real-time gesture recognition.")
    cap = cv2.VideoCapture(0)
else:
    video_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if video_file is not None:
        st.write("Using uploaded video for gesture recognition.")
        cap = cv2.VideoCapture(video_file.name)

if cap is not None:
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints = movenet_inference(frame_rgb)
        frame_with_keypoints = draw_keypoints(frame_rgb, keypoints)
        
        stframe.image(frame_with_keypoints, channels='RGB')
        
    cap.release()
else:
    st.write("Please upload a video file or enable your webcam.")

st.write("Hand gesture recognition using MoveNet model. The model detects keypoints on the hand to recognize gestures.")
