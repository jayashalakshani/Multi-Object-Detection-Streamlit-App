import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tensorflow as tf
import pickle
import os
import sys
import pandas as pd

# Append path for local modules
sys.path.append(os.path.abspath('.'))  
from yolo_integration import YOLOObjectDetector

# --- Configuration ---
YOLO_MODEL_DIR = "C:\\Users\\acer\\OneDrive\\Jupyter\\object_detection_project\\yolo_models"
YOLO_MODEL_NAME = "yolov8s.pt"
YOLO_MODEL_PATH = os.path.join(YOLO_MODEL_DIR, YOLO_MODEL_NAME)

MY_MODEL_DIR = "C:\\Users\\acer\\OneDrive\\Jupyter\\object_detection_project"
MY_MODEL_NAME = "my_tuned_model_2.h5"
MY_MODEL_PATH = os.path.join(MY_MODEL_DIR, MY_MODEL_NAME)

# --- Load YOLO Model ---
try:
    yolo_detector = YOLOObjectDetector(custom_model_path=YOLO_MODEL_PATH)
    if yolo_detector.model is None:
        raise ValueError("❌ YOLO model failed to load.")
    print("✅ YOLO model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    yolo_detector = None

# Load my custom Keras model
try:
    keras_model = tf.keras.models.load_model(MY_MODEL_PATH)
    print("✅ Custom Keras model loaded.")
except Exception as e:
    print(f"❌ Error loading Keras model: {e}")
    keras_model = None

# --- Streamlit UI ---
st.title("🔍 Object Detection App")

# --- Image Upload ---
uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    col1, col2 = st.columns(2)

    # --- Class names
    CLASS_NAMES = [
        "airplane", "apple", "backpack", "banana", "baseball bat", "baseball glove", "bear", "bed", "bench", "bicycle",
        "bird", "boat", "book", "bottle", "bowl", "broccoli", "bus", "cake", "car", "carrot", "cat", "cell phone", "chair",
        "clock", "couch", "cow", "cup", "dining table", "dog", "donut", "elephant", "fire hydrant", "fork", "frisbee",
        "giraffe", "handbag", "horse", "hot dog", "keyboard", "kite", "knife", "laptop", "microwave", "motorcycle",
        "mouse", "orange", "oven", "parking meter", "person", "pizza", "potted plant", "refrigerator", "remote",
        "sandwich", "scissors", "sheep", "sink", "skateboard", "skis", "snowboard", "spoon", "sports ball", "stop sign",
        "suitcase", "surfboard", "teddy bear", "tennis racket", "tie", "toilet", "toothbrush", "traffic light", "train",
        "truck", "tv", "umbrella", "vase", "wine glass", "zebra"
    ]

    # --- Custom Model Detection ---
    with col1:
        st.header("🟡 Custom Model Detection")
        if keras_model:
            try:
                start_time = time.time()

                # Resize and normalize like in preprocessing
                img_resized = cv2.resize(img_np, (224, 224))
                img_processed = img_resized.astype('float32') / 255.0
                img_expanded = np.expand_dims(img_processed, axis=0)  # (1, 224, 224, 3)

                # Predict
                predictions = keras_model.predict(img_expanded)[0]
                inference_time = time.time() - start_time

                # Threshold (you can set dynamically based on tuning)
                threshold = 0.1
                detected_classes = [(CLASS_NAMES[i], float(conf)) for i, conf in enumerate(predictions) if conf > threshold]

                # Display image
                st.image(img_np, channels="RGB", use_column_width=True)

                if detected_classes:
                    st.success("Detected objects:")
                    for cls, conf in detected_classes:
                        st.markdown(f"- **{cls}** ({conf:.2f})")
                else:
                    st.warning("No objects detected with confidence > {:.4f}".format(threshold))

                st.write(f"🕒 Inference Time: {inference_time:.4f} seconds")

            except Exception as e:
                st.error(f"❌ Error during custom model inference: {e}")
        else:
            st.warning("❌ Custom model not loaded.")
    
        
        
    # --- YOLO Detection ---
    with col2:
        st.header("🟢 YOLOv8 Detection")
        if yolo_detector and yolo_detector.model:
            try:
                # Save uploaded image to temp file for YOLO
                temp_path = "temp_uploaded_image.jpg"
                img.save(temp_path)

                yolo_results, yolo_inference_time, _ = yolo_detector.detect_objects(temp_path)

                if yolo_results:
                    yolo_df = pd.DataFrame(yolo_results)
                    img_cv2 = cv2.imread(temp_path)

                    for _, row in yolo_df.iterrows():
                        xmin, ymin, xmax, ymax = map(int, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
                        label = row["name"]
                        confidence = row["confidence"]

                        cv2.rectangle(img_cv2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(img_cv2, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    yolo_img_np = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                    st.image(yolo_img_np, channels="RGB", use_column_width=True)
                    st.write(f"🕒 Inference Time: {yolo_inference_time:.4f} seconds")
                else:
                    st.warning("❌ YOLO inference failed.")
            except Exception as e:
                st.error(f"❌ Error during YOLO inference: {e}")
        else:
            st.warning("❌ YOLO model not loaded.")













