import streamlit as st
import cv2
import numpy as np
import tempfile
import requests
import os

# Function to download video from URL
def download_video(url, output_path):
    r = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

# Function to enable/disable webcam
def toggle_webcam():
    global capture
    if st.session_state.is_webcam_enabled:
        capture = cv2.VideoCapture(0)
    else:
        if capture is not None:
            capture.release()

# Function to perform hand gesture recognition
def recognize_gesture(frame):
    # Add your hand gesture recognition code here using Movenet or any other model
    # This function should return the processed frame with overlays or annotations
    
    # Placeholder code to display a rectangle on the frame
    processed_frame = cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)
    
    return processed_frame

# Main function to run the Streamlit app
def main():
    st.title("Hand Gesture Recognition App")
    
    # Download the video from URL
    video_url = "https://drive.google.com/uc?id=1QJS0yZMu8zNGRyJr_jDUuIW1WT4kpZBM"
    video_path = "temp_video.mp4"
    download_video(video_url, video_path)
    
    # Display the video
    st.video(video_path)
    
    # Add checkbox to enable/disable webcam
    st.sidebar.markdown("## Webcam Control")
    st.session_state.is_webcam_enabled = st.sidebar.checkbox("Enable Webcam", False, key="webcam_checkbox")
    
    # Toggle webcam based on checkbox state
    toggle_webcam()
    
    # Main loop to process webcam feed
    while st.session_state.is_webcam_enabled:
        ret, frame = capture.read()
        if ret:
            # Process frame for hand gesture recognition
            processed_frame = recognize_gesture(frame)
            st.image(processed_frame, channels="BGR")
        
    # Release the webcam capture when done
    if capture is not None:
        capture.release()

if __name__ == "__main__":
    # Initialize webcam capture
    capture = None
    main()
