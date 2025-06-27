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
import threading
import queue

# Import streamlit-webrtc components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Page config
st.set_page_config(
    page_title="SIBI Real-time Sign Language Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# WebRTC configuration for better connectivity
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

class SIBIStreamlitDetector:
    def __init__(self, model_path='models/sibiv1.pt'):
        """
        Initialize SIBI detector for Streamlit with sentence building capability
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = None
        self.current_prediction = None
        self.current_confidence = 0.0

        # Load the model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            raise Exception(f"Failed to load model: {e}")

        # Detection parameters
        self.confidence_threshold = 0.3
        self.prediction_history = []
        self.history_size = 5

        # Sentence building features
        self.detected_words = deque(maxlen=50)  # Store last 50 detected words
        self.current_sentence = ""
        self.last_detection_time = 0
        self.word_timeout = 1.5  # seconds between words
        self.stable_detection_count = 0
        self.stable_threshold = 3  # need 3 stable detections to add word

        # Thread-safe variables for WebRTC
        self.lock = threading.Lock()
        self.latest_frame = None
        self.detection_results = queue.Queue(maxsize=10)

    def predict(self, frame):
        """
        Make prediction on the frame using YOLO
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
            return None, 0.0, None
    
    def smooth_predictions(self, prediction, confidence):
        """
        Smooth predictions using history to reduce noise
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
        Draw prediction information on frame with smaller, compact display
        """
        height, width = frame.shape[:2]

        # Draw bounding box if available
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw prediction info with smaller box
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

            # Smaller background for text - reduced height from 150 to 90
            cv2.rectangle(frame, (10, 10), (min(width-10, 350), 90), (0, 0, 0), -1)

            # Current prediction text - smaller font size
            cv2.putText(frame, f"{label}",
                       (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"{confidence:.2f}",
                       (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Compact sentence text
            sentence_text = self.current_sentence if self.current_sentence else "..."
            if len(sentence_text) > 25:  # Shorter truncation
                sentence_text = sentence_text[:22] + "..."
            cv2.putText(frame, f"S: {sentence_text}",
                       (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Update current prediction for display
            self.current_prediction = label
            self.current_confidence = confidence
        else:
            # Smaller "No detection" box
            cv2.rectangle(frame, (10, 10), (200, 60), (0, 0, 0), -1)
            cv2.putText(frame, "No detection",
                       (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Still show sentence if available - compact version
            if self.current_sentence:
                sentence_text = self.current_sentence
                if len(sentence_text) > 20:
                    sentence_text = sentence_text[:17] + "..."
                cv2.putText(frame, f"S: {sentence_text}",
                           (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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

def video_frame_callback(frame, detector):
    """
    Callback function for processing video frames from WebRTC
    """
    img = frame.to_ndarray(format="bgr24")
    
    # Flip frame horizontally for mirror effect
    img = cv2.flip(img, 1)
    
    # Make prediction
    prediction, confidence, bbox = detector.predict(img)
    smoothed_prediction = detector.smooth_predictions(prediction, confidence)
    
    # Draw information on frame
    annotated_frame = detector.draw_info(img, smoothed_prediction, confidence, bbox)
    
    # Store latest results for UI updates (thread-safe)
    try:
        detector.detection_results.put_nowait({
            'prediction': smoothed_prediction,
            'confidence': confidence,
            'timestamp': time.time()
        })
    except queue.Full:
        pass  # Skip if queue is full
    
    # Convert back to av.VideoFrame
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

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

    # Initialize session state
    if 'sentence_history' not in st.session_state:
        st.session_state.sentence_history = []

    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📷 Live Detection", "📁 Upload Image", "ℹ️ About"])
    
    with tab1:
        st.header("📷 Live Camera Detection")
        st.markdown("**Real-time SIBI detection with sentence building using WebRTC**")

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Sentence"):
                detector.clear_sentence()
                st.rerun()
        with col2:
            # Save sentence button
            sentence_info = detector.get_sentence_info()
            if sentence_info['sentence'] and st.button("💾 Save Sentence"):
                st.session_state.sentence_history.append({
                    'sentence': sentence_info['sentence'],
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'word_count': sentence_info['word_count']
                })
                st.success("Sentence saved to history!")

        # Create layout with smaller camera
        col1, col2 = st.columns([1, 2])  # Camera smaller (1/3), info larger (2/3)
        
        with col1:
            st.subheader("📹 Camera")
            # Create WebRTC streamer with custom styling
            webrtc_ctx = webrtc_streamer(
                key="sibi-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_frame_callback=lambda frame: video_frame_callback(frame, detector),
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 320},
                        "height": {"ideal": 240}
                    }, 
                    "audio": False
                },
                async_processing=True,
            )
        
        with col2:
            st.subheader("📊 Detection Results")
            detection_placeholder = st.empty()
            
            st.subheader("📝 Current Sentence")
            sentence_placeholder = st.empty()

        # Update detection results in real-time
        if webrtc_ctx.state.playing:
            st.success("🔴 Live detection is active. Position your hand clearly in the camera view.")
            
            # Continuously update results
            while webrtc_ctx.state.playing:
                try:
                    # Get latest detection results
                    result = detector.detection_results.get(timeout=0.1)
                    
                    with detection_placeholder.container():
                        if result['prediction'] and result['confidence'] > confidence_threshold:
                            # Compact metrics display
                            st.metric("🎯 Current Sign", result['prediction'])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("📊 Confidence", f"{result['confidence']:.1%}")  
                            with col2:
                                st.metric("⚡ Stability", f"{detector.stable_detection_count}/{detector.stable_threshold}")
                        else:
                            st.warning("⌛ Waiting for detection...")
                    
                    # Update sentence display
                    sentence_info = detector.get_sentence_info()
                    with sentence_placeholder.container():
                        if sentence_info['sentence']:
                            st.success(f"**📝 Sentence:** {sentence_info['sentence']}")
                            st.info(f"**📊 Words:** {sentence_info['word_count']}")
                            
                            # Show last few words
                            if sentence_info['last_words']:
                                recent_words = [w['word'] for w in sentence_info['last_words']]
                                st.text(f"Recent: {' → '.join(recent_words)}")
                        else:
                            st.info("💡 Show SIBI signs to start building sentences")
                            
                except queue.Empty:
                    time.sleep(0.1)
                    continue
                except:
                    break
        else:
            with col2:
                st.info("""
                👆 Click **Start** button in the camera section to begin detection.
                
                **Note**: Allow camera access when prompted by your browser.
                """)
                
                # Show current sentence even when not active
                sentence_info = detector.get_sentence_info()
                if sentence_info['sentence']:
                    st.success(f"**Last Sentence:** {sentence_info['sentence']}")
                    st.info(f"**Word Count:** {sentence_info['word_count']}")

        # Show current sentence even when not active
        sentence_info = detector.get_sentence_info()
        if sentence_info['sentence'] and not webrtc_ctx.state.playing:
            st.success(f"**Last Built Sentence:** {sentence_info['sentence']}")

        # Sentence history
        if st.session_state.sentence_history:
            st.subheader("📚 Sentence History")
            for i, entry in enumerate(reversed(st.session_state.sentence_history[-10:])):  # Show last 10
                with st.expander(f"Sentence {len(st.session_state.sentence_history)-i}: {entry['sentence'][:50]}..."):
                    st.write(f"**Full Sentence:** {entry['sentence']}")
                    st.write(f"**Words:** {entry['word_count']}")
                    st.write(f"**Time:** {entry['timestamp']}")

        # Instructions
        with st.expander("📖 How to Use Live Detection"):
            st.markdown("""
            **Instructions for Live Detection:**
            1. Click "Start" to activate the camera feed
            2. Allow camera access when prompted by your browser
            3. Position your hand clearly in front of the camera
            4. Hold each SIBI sign steady for a few seconds
            5. The system will automatically add detected signs to build a sentence
            6. Use "Clear Sentence" to start over
            7. Use "Save Sentence" to store completed sentences

            **Tips:**
            - Ensure good lighting for better detection
            - Hold signs steady for the stability threshold duration
            - Adjust confidence and stability thresholds in the sidebar
            - The word timeout controls spacing between words
            
            **Browser Compatibility:**
            - Works best with Chrome, Firefox, Safari, and Edge
            - Requires HTTPS for camera access in production
            - May need to refresh if camera doesn't start
            """)
            
    with tab2:
        st.header("📁 Upload Image")
        st.markdown("Upload an image containing SIBI sign language")
        
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
    
    with tab3:
        st.header("ℹ️ About SIBI Detector")

        st.markdown("""
        ### 🤟 Sistem Isyarat Bahasa Indonesia (SIBI)

        This Streamlit application provides advanced SIBI detection with sentence building capabilities using WebRTC for real-time camera access.

        ### ✨ Features
        - **🎥 WebRTC Live Detection** - Real-time video streaming that works in deployed apps
        - **📝 Sentence Building** - Automatic word-to-sentence construction
        - **📚 Sentence History** - Save and manage detected sentences
        - **⚙️ Advanced Settings** - Configurable stability and timing parameters
        - **📸 Multiple Input Methods** - Live camera and file upload
        - **🌐 Cloud Compatible** - Works on deployed Streamlit apps with HTTPS

        ### 🔧 Technology Stack
        - **Model**: Ultralytics YOLO v8 for SIBI detection
        - **Backend**: PyTorch for deep learning inference
        - **Frontend**: Streamlit with streamlit-webrtc for camera access
        - **Computer Vision**: OpenCV for image processing
        - **Real-time Processing**: WebRTC for low-latency video streaming

        ### 📊 Detection Features
        - **Prediction Smoothing** - Reduces noise with history-based filtering
        - **Confidence Thresholding** - Adjustable detection sensitivity
        - **Stability Control** - Requires consistent detections before adding words
        - **Bounding Box Visualization** - Visual feedback for detected signs
        - **Real-time Statistics** - Live confidence and stability metrics

        ### 🎯 How It Works
        1. **Video Capture**: WebRTC streams video directly from your browser
        2. **Detection**: YOLO model identifies SIBI signs in real-time
        3. **Smoothing**: Multiple consistent detections reduce false positives
        4. **Stability**: Words are added only after stable detection periods
        5. **Sentence Building**: Detected words are automatically combined with timing logic
        6. **History**: Completed sentences are saved for review and export

        ### 🌐 Deployment Benefits
        - **Browser Compatibility**: Works across modern browsers
        - **No Local Dependencies**: Camera access through web standards
        - **HTTPS Support**: Secure camera access in production
        - **Cross-Platform**: Works on desktop and mobile devices
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

            **Browser Setup:**
            - Allow camera permissions when prompted
            - Use Chrome, Firefox, Safari, or Edge for best compatibility
            - Ensure stable internet connection for smooth streaming
            - Refresh page if camera doesn't start immediately
            """)

        # Technical information
        with st.expander("⚡ Technical Information"):
            st.markdown("""
            **WebRTC Implementation:**
            - Real-time peer-to-peer video streaming
            - Low-latency processing pipeline
            - Automatic frame rate optimization
            - Thread-safe detection results handling

            **Performance Optimizations:**
            - Async video processing
            - Queue-based result updates
            - Memory-efficient frame handling
            - GPU acceleration when available

            **Deployment Requirements:**
            - HTTPS required for camera access in production
            - streamlit-webrtc package for WebRTC functionality
            - Modern browser with WebRTC support
            - Stable internet connection
            """)

if __name__ == "__main__":
    main()