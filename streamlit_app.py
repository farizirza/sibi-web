import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image
import tempfile
import os
from collections import deque

# Page config
st.set_page_config(
    page_title="SIBI Real-time Sign Language Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SIBIStreamlitDetector:
    def __init__(self, model_path='models/sibiv3.pt'):
        """
        Initialize SIBI detector for Streamlit with sentence building capability
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize camera and model
        self.cap = None
        self.model = None
        self.current_prediction = None
        self.current_confidence = 0.0
        self.is_running = False

        # Load the model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            raise Exception(f"Failed to load model: {e}")

        # Detection parameters
        self.confidence_threshold = 0.5
        self.prediction_history = []
        self.history_size = 5

        # Sentence building features
        self.detected_words = deque(maxlen=50)  # Store last 50 detected words
        self.current_sentence = ""
        self.last_detection_time = 0
        self.word_timeout = 2.0  # seconds between words
        self.stable_detection_count = 0
        self.stable_threshold = 3  # need 3 stable detections to add word

    def start_camera(self):
        """Start camera capture"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            # Set camera properties for better performance and smaller size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.is_running = True

    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
    def predict(self, frame):
        """
        Make prediction on the frame using YOLO (same as Flask version)
        """
        try:
            results = self.model(frame, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the best detection
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                
                # Get highest confidence detection
                best_idx = confidences.argmax()
                predicted = int(classes[best_idx])
                confidence = float(confidences[best_idx])
                
                # Get bounding box
                bbox = boxes.xyxy[best_idx].cpu().numpy()
                
                return predicted, confidence, bbox
            else:
                return None, 0.0, None
        
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, 0.0, None
    
    def smooth_predictions(self, prediction, confidence):
        """
        Smooth predictions using history to reduce noise (same as Flask version)
        """
        if confidence > self.confidence_threshold:
            self.prediction_history.append(prediction)
        
        # Keep only recent predictions
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Return most common prediction if we have enough history
        if len(self.prediction_history) >= 3:
            most_common = max(set(self.prediction_history), 
                            key=self.prediction_history.count)
            return most_common
        
        return prediction if confidence > self.confidence_threshold else None

    def add_word_to_sentence(self, word):
        """
        Add detected word to sentence with timing logic
        """
        current_time = time.time()

        # Check if enough time has passed since last detection
        if current_time - self.last_detection_time > self.word_timeout:
            # Add word to sentence
            if self.current_sentence:
                self.current_sentence += " " + word
            else:
                self.current_sentence = word

            # Add to detected words history
            self.detected_words.append({
                'word': word,
                'timestamp': current_time,
                'confidence': self.current_confidence
            })

            self.last_detection_time = current_time
            self.stable_detection_count = 0
            return True
        return False

    def clear_sentence(self):
        """Clear the current sentence"""
        self.current_sentence = ""
        self.detected_words.clear()
        self.stable_detection_count = 0

    def get_sentence_info(self):
        """Get current sentence and word history"""
        return {
            'sentence': self.current_sentence,
            'word_count': len(self.current_sentence.split()) if self.current_sentence else 0,
            'last_words': list(self.detected_words)[-5:] if self.detected_words else []
        }

    def draw_info(self, frame, prediction, confidence, bbox=None):
        """
        Draw prediction information on frame with sentence building
        """
        width = frame.shape[1]

        # Draw bounding box if available
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw prediction info
        if prediction is not None and confidence > self.confidence_threshold:
            # Get class name from model
            class_names = self.model.names
            label = class_names.get(prediction, f"Class_{prediction}")

            # Check for stable detection for sentence building
            if label == self.current_prediction:
                self.stable_detection_count += 1
            else:
                self.stable_detection_count = 1

            # Add to sentence if stable enough
            if self.stable_detection_count >= self.stable_threshold:
                self.add_word_to_sentence(label)

            # Background for text (larger for sentence)
            cv2.rectangle(frame, (10, 10), (min(width-10, 600), 150), (0, 0, 0), -1)

            # Current prediction text
            cv2.putText(frame, f"Current: {label}",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}",
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Sentence text
            sentence_text = self.current_sentence if self.current_sentence else "..."
            if len(sentence_text) > 40:  # Truncate long sentences
                sentence_text = sentence_text[:37] + "..."
            cv2.putText(frame, f"Sentence: {sentence_text}",
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Stability indicator
            stability_text = f"Stability: {self.stable_detection_count}/{self.stable_threshold}"
            cv2.putText(frame, stability_text,
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Update current prediction for display
            self.current_prediction = label
            self.current_confidence = confidence
        else:
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.putText(frame, "No detection",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Still show sentence if available
            if self.current_sentence:
                sentence_text = self.current_sentence
                if len(sentence_text) > 40:
                    sentence_text = sentence_text[:37] + "..."
                cv2.putText(frame, f"Sentence: {sentence_text}",
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Reset stability counter
            self.stable_detection_count = 0
            self.current_prediction = None
            self.current_confidence = 0.0

        return frame

@st.cache_resource
def load_detector():
    """Load detector with caching"""
    try:
        return SIBIStreamlitDetector()
    except Exception as e:
        st.error(f"Failed to initialize detector: {e}")
        return None

def process_image(detector, image_array, confidence_threshold):
    """Process image and return results"""
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Make prediction
    prediction, confidence, bbox = detector.predict(image_bgr)
    smoothed_prediction = detector.smooth_predictions(prediction, confidence)
    
    # Update confidence threshold
    detector.confidence_threshold = confidence_threshold
    
    # Draw information on frame
    annotated_frame = detector.draw_info(image_bgr, smoothed_prediction, confidence, bbox)
    
    # Convert back to RGB for display
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame_rgb, smoothed_prediction, confidence

def main():
    st.title("🤟 SIBI Real-time Sign Language Detector")
    st.markdown("**Position your hand in front of the camera to detect SIBI sign language and build sentences.**")

    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust detection sensitivity"
    )

    # Sentence building settings
    st.sidebar.subheader("📝 Sentence Building")
    word_timeout = st.sidebar.slider(
        "Word Timeout (seconds)",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Time between word detections"
    )

    stable_threshold = st.sidebar.slider(
        "Stability Threshold",
        min_value=2,
        max_value=10,
        value=3,
        step=1,
        help="Number of consistent detections needed to add word"
    )

    # Load detector
    detector = load_detector()
    if detector is None:
        st.stop()

    # Update detector settings
    detector.confidence_threshold = confidence_threshold
    detector.word_timeout = word_timeout
    detector.stable_threshold = stable_threshold

    # Function to stop all camera activities when switching tabs
    def stop_all_cameras():
        if detector.cap:
            detector.stop_camera()
        st.session_state.live_detection_active = False
        st.session_state.camera_capture_active = False

    # Main interface tabs
    tab1, tab3, tab4 = st.tabs(["📷 Live Detection", "📁 Upload Image", "ℹ️ About"])
    
    with tab1:
        st.header("📷 Live Camera Detection")
        st.markdown("**Real-time SIBI detection with sentence building**")

        # Initialize session state for live detection
        if 'live_detection_active' not in st.session_state:
            st.session_state.live_detection_active = False
        if 'sentence_history' not in st.session_state:
            st.session_state.sentence_history = []
        if 'current_tab' not in st.session_state:
            st.session_state.current_tab = 0
        if 'camera_capture_active' not in st.session_state:
            st.session_state.camera_capture_active = False

        # Stop camera capture when entering Live Detection tab
        if st.session_state.camera_capture_active:
            st.session_state.camera_capture_active = False
            if detector.cap:
                detector.stop_camera()

        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎥 Start Live Detection", type="primary"):
                st.session_state.live_detection_active = True
                detector.clear_sentence()
        with col2:
            if st.button("⏹️ Stop Detection"):
                st.session_state.live_detection_active = False
        with col3:
            if st.button("🗑️ Clear Sentence"):
                detector.clear_sentence()
                st.rerun()

        # Live detection area
        if st.session_state.live_detection_active:
            st.info("🔴 Live detection is active. Position your hand to show SIBI signs.")

            # Create layout with smaller video feed
            col1, col2 = st.columns([1, 2])  # Video smaller (1/3), info larger (2/3)

            with col1:
                st.subheader("📹 Live Feed")
                video_placeholder = st.empty()

            with col2:
                st.subheader("📊 Detection Results")
                sentence_placeholder = st.empty()
                stats_placeholder = st.empty()

            # Start camera capture
            try:
                detector.start_camera()

                # Live detection loop
                frame_count = 0
                while st.session_state.live_detection_active and frame_count < 100:  # Limit frames to prevent infinite loop
                    ret, frame = detector.cap.read()
                    if not ret:
                        st.error("Failed to read from camera")
                        break

                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)

                    # Make prediction
                    prediction, confidence, bbox = detector.predict(frame)
                    smoothed_prediction = detector.smooth_predictions(prediction, confidence)

                    # Draw information on frame
                    annotated_frame = detector.draw_info(frame, smoothed_prediction, confidence, bbox)

                    # Convert to RGB for display
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # Update video display with fixed width
                    video_placeholder.image(frame_rgb, channels="RGB", width=320)

                    # Update sentence display
                    sentence_info = detector.get_sentence_info()
                    with sentence_placeholder.container():
                        if sentence_info['sentence']:
                            st.success(f"**Current Sentence:** {sentence_info['sentence']}")
                            st.info(f"**Word Count:** {sentence_info['word_count']}")
                        else:
                            st.warning("No sentence built yet. Show SIBI signs to start building.")

                    # Update statistics
                    with stats_placeholder.container():
                        if smoothed_prediction and confidence > confidence_threshold:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Sign", smoothed_prediction)
                            with col2:
                                st.metric("Confidence", f"{confidence:.1%}")
                            with col3:
                                st.metric("Stability", f"{detector.stable_detection_count}/{detector.stable_threshold}")

                    frame_count += 1
                    time.sleep(0.1)  # Small delay to prevent overwhelming

                detector.stop_camera()

            except Exception as e:
                st.error(f"Camera error: {e}")
                st.session_state.live_detection_active = False
                if detector.cap:
                    detector.stop_camera()
        else:
            st.info("👆 Click 'Start Live Detection' to begin real-time SIBI detection with sentence building.")

            # Show current sentence even when not active
            sentence_info = detector.get_sentence_info()
            if sentence_info['sentence']:
                st.success(f"**Last Built Sentence:** {sentence_info['sentence']}")

                # Save sentence button
                if st.button("💾 Save Sentence"):
                    st.session_state.sentence_history.append({
                        'sentence': sentence_info['sentence'],
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'word_count': sentence_info['word_count']
                    })
                    st.success("Sentence saved to history!")

        # Sentence history
        if st.session_state.sentence_history:
            st.subheader("� Sentence History")
            for i, entry in enumerate(reversed(st.session_state.sentence_history[-10:])):  # Show last 10
                with st.expander(f"Sentence {len(st.session_state.sentence_history)-i}: {entry['sentence'][:50]}..."):
                    st.write(f"**Full Sentence:** {entry['sentence']}")
                    st.write(f"**Words:** {entry['word_count']}")
                    st.write(f"**Time:** {entry['timestamp']}")

        # Instructions
        with st.expander("📖 How to Use Live Detection"):
            st.markdown("""
            **Instructions for Live Detection:**
            1. Click "Start Live Detection" to activate the camera
            2. Position your hand clearly in front of the camera
            3. Hold each SIBI sign steady for a few seconds
            4. The system will automatically add detected signs to build a sentence
            5. Use "Clear Sentence" to start over
            6. Use "Save Sentence" to store completed sentences

            **Tips:**
            - Ensure good lighting for better detection
            - Hold signs steady for the stability threshold duration
            - Adjust confidence and stability thresholds in the sidebar
            - The word timeout controls spacing between words
            """)
            
    with tab3:
        st.header("📁 Upload Image")
        st.markdown("Upload an image containing SIBI sign language")

        # Stop all camera activities when entering this tab
        if st.session_state.live_detection_active or st.session_state.camera_capture_active:
            st.session_state.live_detection_active = False
            st.session_state.camera_capture_active = False
            if detector.cap:
                detector.stop_camera()
        
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image for SIBI detection"
        )
        
        if uploaded_file is not None:
            # Load and process image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Process image
            result_image, prediction, confidence = process_image(
                detector, image_array, confidence_threshold
            )
            
            # Display results with smaller images
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Image")
                st.image(image, width=300)

            with col2:
                st.subheader("Detection Result")
                st.image(result_image, width=300)
            
            # Results display
            if prediction and confidence > confidence_threshold:
                st.success(f"**Prediction:** {prediction}")
                st.info(f"**Confidence:** {confidence:.2%}")
                st.progress(confidence)

                # Add to sentence button
                if st.button("➕ Add to Sentence", key="add_to_sentence_upload"):
                    detector.add_word_to_sentence(prediction)
                    st.success(f"Added '{prediction}' to sentence!")
                    st.rerun()

                # Download functionality
                result_pil = Image.fromarray(result_image)
                buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                result_pil.save(buf.name)

                with open(buf.name, 'rb') as f:
                    st.download_button(
                        label="📥 Download Result",
                        data=f.read(),
                        file_name=f"sibi_detection_{int(time.time())}.png",
                        mime="image/png"
                    )

                os.unlink(buf.name)
            else:
                st.warning("No detection above threshold")
                if prediction:
                    st.info(f"Low confidence: {prediction} ({confidence:.2%})")

            # Show current sentence
            sentence_info = detector.get_sentence_info()
            if sentence_info['sentence']:
                st.subheader("📝 Current Sentence")
                st.success(f"**Sentence:** {sentence_info['sentence']}")
                st.info(f"**Word Count:** {sentence_info['word_count']}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Sentence", key="save_sentence_upload"):
                        st.session_state.sentence_history.append({
                            'sentence': sentence_info['sentence'],
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'word_count': sentence_info['word_count']
                        })
                        st.success("Sentence saved!")
                with col2:
                    if st.button("🗑️ Clear Sentence", key="clear_sentence_upload"):
                        detector.clear_sentence()
                        st.rerun()
    
    with tab4:
        st.header("ℹ️ About SIBI Detector")

        # Stop all camera activities when entering this tab
        if st.session_state.live_detection_active or st.session_state.camera_capture_active:
            st.session_state.live_detection_active = False
            st.session_state.camera_capture_active = False
            if detector.cap:
                detector.stop_camera()

        st.markdown("""
        ### 🤟 Sistem Isyarat Bahasa Indonesia (SIBI)

        This enhanced Streamlit application provides advanced SIBI detection with sentence building capabilities:

        ### ✨ New Features
        - **🎥 Live Camera Detection** - Real-time continuous video streaming
        - **📝 Sentence Building** - Automatic word-to-sentence construction
        - **📚 Sentence History** - Save and manage detected sentences
        - **⚙️ Advanced Settings** - Configurable stability and timing parameters
        - **📸 Multiple Input Methods** - Live camera, photo capture, and file upload

        ### 🔧 Technology Stack
        - **Model**: Ultralytics YOLO v8 for SIBI detection
        - **Backend**: PyTorch for deep learning inference
        - **Frontend**: Streamlit for interactive web interface
        - **Computer Vision**: OpenCV for image processing
        - **Real-time Processing**: Optimized for live video streams

        ### 📊 Detection Features
        - **Prediction Smoothing** - Reduces noise with history-based filtering
        - **Confidence Thresholding** - Adjustable detection sensitivity
        - **Stability Control** - Requires consistent detections before adding words
        - **Bounding Box Visualization** - Visual feedback for detected signs
        - **Real-time Statistics** - Live confidence and stability metrics

        ### 🎯 How It Works
        1. **Detection**: YOLO model identifies SIBI signs in real-time
        2. **Smoothing**: Multiple consistent detections reduce false positives
        3. **Stability**: Words are added only after stable detection periods
        4. **Sentence Building**: Detected words are automatically combined with timing logic
        5. **History**: Completed sentences are saved for review and export
        """)

        # Model information
        if detector and detector.model:
            st.subheader("🔍 Model Information")

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Device**: {detector.device}")
                st.write(f"**Model Type**: Ultralytics YOLO")
                st.write(f"**Classes**: {len(detector.model.names)} classes")
            with col2:
                st.write(f"**Confidence Threshold**: {confidence_threshold}")
                st.write(f"**Word Timeout**: {detector.word_timeout}s")
                st.write(f"**Stability Threshold**: {detector.stable_threshold}")

            # Show available classes
            with st.expander("📋 View All Available SIBI Classes"):
                # Display classes in a more organized way
                classes = list(detector.model.names.items())
                cols = st.columns(3)
                for i, (idx, name) in enumerate(classes):
                    with cols[i % 3]:
                        st.write(f"**{idx}**: {name}")

        # Usage tips
        with st.expander("💡 Usage Tips"):
            st.markdown("""
            **For Best Results:**
            - Ensure good lighting conditions
            - Position hand clearly in camera view
            - Hold signs steady for stability threshold duration
            - Use consistent hand positioning
            - Adjust confidence threshold based on environment

            **Sentence Building Tips:**
            - Wait for word timeout between different signs
            - Use "Clear Sentence" to start over
            - Save important sentences to history
            - Adjust stability threshold for accuracy vs speed
            """)

        # Performance info
        with st.expander("⚡ Performance Information"):
            st.markdown("""
            **Live Detection Performance:**
            - Optimized for real-time processing
            - Automatic frame rate adjustment
            - Memory-efficient video streaming
            - GPU acceleration when available

            **System Requirements:**
            - Python 3.7+
            - Webcam/camera access
            - 4GB+ RAM recommended
            - GPU optional but recommended for better performance
            """)

if __name__ == "__main__":
    main()
