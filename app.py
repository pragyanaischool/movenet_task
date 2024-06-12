import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Function to detect hand gestures using MoveNet
def detect_hand_gesture(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the MoveNet model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    # Detect hand landmarks
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        # Extract hand landmarks
        landmarks = results.multi_hand_landmarks[0].landmark
        # Your gesture recognition logic here
        
    return image

# Streamlit app
def main():
    st.title("Hand Gesture Recognition using MoveNet")

    # Option to select video source
    video_source = st.radio("Select Video Source:", ("Desktop Video", "Webcam"))

    if video_source == "Desktop Video":
        # Path to desktop video file
        video_path = "path_to_desktop_video.mp4"
        # Read desktop video
        cap = cv2.VideoCapture(video_path)
    else:
        # Open webcam
        cap = cv2.VideoCapture(0)

    # Loop to process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect hand gestures
        frame = detect_hand_gesture(frame)

        # Display the frame
        st.image(frame, channels="BGR", use_column_width=True)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
