import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# Load the Keras model
model = load_model(r"C:\Users\HP\Contacts\Desktop\Gesture sense\keras_model.h5")

# Define the hand gesture labels
labels = ["like", "dislike", "Fist", "one", "two up", "three", "four", "rock", "peace", "stop", "call"]

# Function to preprocess and predict hand gesture
def predict_gesture(frame):
    input_width = 224
    input_height = 224
    # Preprocess the image if needed
    resized_image = cv2.resize(frame, (input_width, input_height))
    normalized_image = resized_image / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)

    # Make predictions using the loaded model
    prediction = model.predict(input_image)

    # Get the predicted gesture label
    predicted_label_index = np.argmax(prediction)
    predicted_gesture = labels[predicted_label_index]

    return predicted_gesture

# Define a video processor class
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: np.ndarray) -> np.ndarray:
        # Perform hand gesture recognition
        gesture_prediction = predict_gesture(frame)

        # Display the frame and predicted gesture
        st.image(frame, channels="BGR", caption="Hand Gesture Recognition")
        st.write(f"Predicted Gesture: {gesture_prediction}")

# Main Streamlit app
def main():
    st.title("Hand Gesture Recognition")
    st.write("Perform hand gestures in front of the camera.")

    # Launch WebRTC streamer
    webrtc_streamer(key="example", video_processor_factory=VideoProcessor)

if __name__ == "__main__":
    main()
