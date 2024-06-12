import streamlit as st
import cv2
import numpy as np
import tempfile
import requests
import os
import tensorflow as tf
import tensorflow_hub as hub
import imageio

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
    try:
        if st.session_state.is_webcam_enabled:
            camera_index = 0  # Replace with the index you found
            capture = cv2.VideoCapture(camera_index)
        else:
            if capture is not None:
                capture.release()
    except Exception as e:
        st.error(f"Error accessing webcam: {e}")

# Function to perform hand gesture recognition
def recognize_gesture(frame):
    movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    # Define the mapping of keypoints to hand parts
    keypoint_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'open_hand', 'close_hand', 'thumb', 'index',
                      'middle', 'ring', 'pinky']
    # Define the connections between keypoints to draw lines for visualization
    connections = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                   (5, 6), (5, 11), (6, 12), (11, 12)]
    # Function to perform pose detection on an image sequence or GIF
def detect_pose_sequence(video_path):
    # Load the GIF
    gif = cv2.VideoCapture(video_path)
    frames = []
    # Read frames from the GIF
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    # Initialize an empty list to store keypoints for each frame
    all_keypoints = []
    # Iterate through each frame
    for frame in frames:
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize frame to the expected input size of MoveNet
        frame_resized = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, axis=0), 192, 192) # 192 for lightning
        # Convert the resized frame tensor to a NumPy array with dtype uint8
        frame_np = frame_resized.numpy().astype(np.int32)
        # Perform inference
        outputs = movenet.signatures["serving_default"](tf.constant(frame_np))
        # Extract the keypoints
        keypoints = outputs['output_0'].numpy()
        # Append keypoints to the list
        all_keypoints.append(keypoints)
    # Return keypoints for all frames
    return all_keypoints
    # Function to visualize keypoints on an image sequence or GIF and create a new GIF
def visualize_and_create_pose_sequence(video_path, keypoints_list, output_video_path, default_fps=10):
    # Load the GIF
    video = imageio.get_reader(video_path)
    # Initialize list to store frames with keypoints overlay
    frames_with_keypoints = []
    # Loop through each frame and its corresponding keypoints
    for frame_index, (frame, keypoints) in enumerate(zip(video, keypoints_list)):
        # Convert keypoints to numpy array
        keypoints = np.array(keypoints)
        # Ensure keypoints array has the expected shape
        if keypoints.shape == (1, 1, 17, 3):
            # Extract keypoints from the array
            keypoints = keypoints[0, 0]
            # Loop through each keypoint
            for kp_index, kp in enumerate(keypoints):
                # Extract x and y coordinates of the keypoint
                x = int(kp[1] * frame.shape[1])
                y = int(kp[0] * frame.shape[0])
                # Check if the keypoint is critical
                if keypoint_names[kp_index] in ['open_hand', 'close_hand', 'thumb', 'index',
'middle', 'ring', 'pinky']:
                    # Calculate the average position of neighboring keypoints
                    neighbor_indices = [0, 1, 5, 6]
                    neighbor_positions = []
                    for connection in neighbor_indices:
                        neighbor_kp_index = connection[0] if connection[1] == kp_index else connection[1]
                        neighbor_positions.append(keypoints[neighbor_kp_index])
                    neighbor_positions = np.array(neighbor_positions)
                    average_x = int(np.mean(neighbor_positions[:, 1]) * frame.shape[1])
                    average_y = int(np.mean(neighbor_positions[:, 0]) * frame.shape[0])
                    # Update the position of the critical keypoint
                    x = average_x
                    y = average_y
                # Draw a circle at the adjusted keypoint position
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)  # Increase thickness and change color to blue
            # Draw lines connecting keypoints
            for connection in connections:
                start_point = (int(keypoints[connection[0], 1] * frame.shape[1]),
                               int(keypoints[connection[0], 0] * frame.shape[0]))
                end_point = (int(keypoints[connection[1], 1] * frame.shape[1]),
                             int(keypoints[connection[1], 0] * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (0, 0, 255), 1)  # Increase thickness and change color to red
            # Append the frame with keypoints overlay to the list
            frames_with_keypoints.append(frame)
        else:
            print("Unexpected shape of keypoints array for frame", frame_index + 1)
    # Remove the last frame if it's a black frame
    if np.all(frames_with_keypoints[-1] == [0, 0, 0]):
        frames_with_keypoints.pop()
    # Get the frame rate from the metadata if available, otherwise use the default frame rate
    try:
        fps = video.get_meta_data()['fps']
    except KeyError:
        fps = default_fps
    # Save the frames with keypoints overlay as a new GIF
    imageio.mimsave(output_video_path, frames_with_keypoints, fps=fps)
    import gdown

    # URL to the Google Drive file
    url = "https://drive.google.com/uc?id=1QJS0yZMu8zNGRyJr_jDUuIW1WT4kpZBM"
    output = "temp_video.mp4"

    # Download the file
    gdown.download(url, output, quiet=False)
    
    # Placeholder code to display a rectangle on the frame
    processed_frame = cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), 2)

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

