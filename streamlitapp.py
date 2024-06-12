import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os

# Function to download and load the MoveNet model
def load_movenet_model():
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    model_path = "movenet_lightning.tflite"
    
    if not os.path.exists(model_path):
        st.write("Downloading the MoveNet model...")
        model = hub.load(model_url)
        tf.saved_model.save(model, model_path)
        st.write("Model downloaded and saved to", model_path)
    else:
        st.write("Using cached model at", model_path)
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Failed to load the model: {str(e)}")
        return None

# Load the MoveNet model
interpreter = load_movenet_model()

if interpreter:
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

    cap = None
    if input_source == "Webcam":
        st.write("Using webcam for real-time gesture recognition.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Please ensure your webcam is connected and accessible.")
            cap.release()
            cap = None
    else:
        video_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if video_file is not None:
            st.write("Using uploaded video for gesture recognition.")
            temp_file = "uploaded_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(video_file.getvalue())
            cap = cv2.VideoCapture(temp_file)
        else:
            st.write("Please upload a video file.")

    if cap is not None and cap.isOpened():
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
        st.write("No valid input source available. Please check your webcam or upload a video file.")

    st.write("Hand gesture recognition using MoveNet model. The model detects keypoints on the hand to recognize gestures.")
