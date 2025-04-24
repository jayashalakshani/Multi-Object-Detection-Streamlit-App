import cv2
import time
import os
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

class YOLOObjectDetector:
    def __init__(self, model_name='yolov8s.pt', model_dir='yolo_models', custom_model_path=None, conf_threshold=0.5, iou_threshold=0.45):
        """
        Initializes the YOLOObjectDetector.

        Args:
            model_name (str): The name of the YOLO model.
            model_dir (str): The directory to store YOLO models.
            custom_model_path (str, optional): Path to a custom YOLO model.
            conf_threshold (float): Confidence threshold for detections.
            iou_threshold (float): IoU threshold for NMS.
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.custom_model_path = custom_model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = self.load_model()

    def load_model(self):
        """Loads a YOLOv8 model from a file or downloads it if not available locally."""
        os.makedirs(self.model_dir, exist_ok=True)
        try:
            if self.custom_model_path:
                print(f"Loading custom model from: {self.custom_model_path}")
                
                # Check if the model file exists
                if not os.path.exists(self.custom_model_path):
                    raise FileNotFoundError(f"Model not found at {self.custom_model_path}")
                
                self.model = YOLO(self.custom_model_path)
                print("✅ YOLO Model Loaded Successfully!")
                return self.model

            else:
                print(f"Loading pre-trained model: {self.model_name}")
                self.model = YOLO(self.model_name)
                return self.model

        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            return None

    def detect_objects(self, image_path):
        """Performs object detection on a given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple: (predictions list, inference time, original image)
        """
        try:
            img = Image.open(image_path)
            start_time = time.time()
            results = self.model(img)  
            end_time = time.time()

            inference_time = end_time - start_time
            predictions = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    label = self.model.names[class_id]

                    if conf >= self.conf_threshold:
                        predictions.append({'xmin': xyxy[0], 'ymin': xyxy[1], 'xmax': xyxy[2], 'ymax': xyxy[3], 'confidence': conf, 'class': class_id, 'name': label})

            return predictions, inference_time, img

        except Exception as e:
            print(f"❌ Error during YOLO inference: {e}")
            return None, None, None

if __name__ == '__main__':
    # Standalone testing
    TEST_IMAGE = "C:\\Users\\acer\\OneDrive\\Jupyter\\object_detection_project\\data\\test\\images\\000000005345_jpg.rf.48e7947456159d44cbe1a733ad832bf1.jpg"
    CUSTOM_MODEL_PATH = "C:\\Users\\acer\\OneDrive\\Jupyter\\object_detection_project\\yolo_models\\yolov8s.pt"

    yolo_detector = YOLOObjectDetector(custom_model_path=CUSTOM_MODEL_PATH)

    if yolo_detector.model is None:
        print("❌ YOLO model failed to load.")
    else:
        predictions, inference_time, img = yolo_detector.detect_objects(TEST_IMAGE)
        
        if predictions:
            print(f"✅ Inference Time: {inference_time:.4f} seconds")
            print(predictions)







# # yolo_integration.py

# import cv2
# import time
# import os
# import numpy as np
# import pandas as pd
# from PIL import Image
# from ultralytics import YOLO

# class YOLOObjectDetector:
#     def __init__(self, model_name='yolov8s.pt', model_dir='yolo_models',custom_model_path = None, conf_threshold=0.5, iou_threshold=0.45):
#         """Initializes the YOLOObjectDetector.
#         Args:
#             model_name (str): The name of the YOLO model.
#             model_dir (str): The directory to store YOLO models.
#             custom_model_path (str, optional): if you have your own model instead of yolov8
#             conf_threshold (float): Confidence threshold for detections.
#             iou_threshold (float): IoU threshold for NMS.
#         """
#         self.model_name = model_name
#         self.model_dir = model_dir
#         self.custom_model_path = custom_model_path # Set the absolute PATH or file will fail.
#         self.conf_threshold = conf_threshold
#         self.iou_threshold = iou_threshold
#         self.model = self.load_model()  # Load the model during object creation

#     def load_model(self):
#         """Loads a YOLOv8 model from a file or downloads if it is not available locally.
#         """
#         # Create the YOLO model directory if it doesn't exist
#         os.makedirs(self.model_dir, exist_ok=True)
#         try:
#             if self.custom_model_path:  # Attempt to load from the specified path
#                 print(f"Loading custom model from {self.custom_model_path}")
#                 try:
#                     self.model = YOLO(self.custom_model_path)
#                     print("Load Success")
#                     return self.model
#                 except Exception as e:
#                     print(f"Error loading custom model: {e}")
#                     return None

#             else:
#                 print(f"Attempting to download pre-trained model {self.model_name}")
#                 self.model = YOLO(self.model_name)
#                 # save to custom path for persistent loading
#                 #self.model.save(self.custom_model_path) You don't want to resave the thing
#                 return self.model

#         except Exception as e:
#             print(f"Error loading YOLO model: {e}")
#             return None


#     def detect_objects(self, image_path):
#         """Performs object detection using YOLOv8 on a given image.
#         Args:
#             image_path (str): Path to the image file.
#         Returns:
#             tuple: A tuple containing:
#                 - predictions (list): List of dictionaries, where each dictionary represents a detected object.
#                 - inference_time (float): Inference time in seconds.
#                 - img (PIL.Image): The original PIL Image.
#         """
#         try:
#             img = Image.open(image_path)
#             start_time = time.time()
#             results = self.model(img)  # Perform inference - this returns a Results object, not a list
#             end_time = time.time()

#             inference_time = end_time - start_time

#             predictions = []  # Prepare a list to store formatted predictions
#             for r in results:
#                 boxes = r.boxes
#                 for box in boxes:
#                     xyxy = box.xyxy[0].tolist()
#                     conf = box.conf[0].item()
#                     class_id = int(box.cls[0].item())
#                     label = self.model.names[class_id]
#                     if conf >= self.conf_threshold:
#                         predictions.append({'xmin': xyxy[0], 'ymin': xyxy[1], 'xmax': xyxy[2], 'ymax': xyxy[3], 'confidence': conf, 'class': class_id, 'name': label})

#             return predictions, inference_time, img

#         except Exception as e:
#             print(f"Error during YOLO inference: {e}")
#             return None, None, None


# if __name__ == '__main__':
#     # For standalone testing
#     import cv2
#     yolo_detector = YOLOObjectDetector(custom_model_path="C:\\Users\\acer\\OneDrive\\Jupyter\\object_detection_project\\yolov8s.pt")
#     TEST_IMAGE = "C:\\Users\\acer\\OneDrive\\Jupyter\\object_detection_project\\data\\test\\images\\000000005345_jpg.rf.48e7947456159d44cbe1a733ad832bf1.jpg"

#     if yolo_detector.model is None:
#         print("YOLO model failed to load.")
#     else:

#         #TEST_IMAGE = "C:\\Users\\acer\\OneDrive\\Jupyter\\object_detection_project\\data\\test\\images\\000000005345_jpg.rf.48e7947456159d44cbe1a733ad832bf1.jpg"

#         try:

#             yolo_results, yolo_inference_time, img = yolo_detector.detect_objects(TEST_IMAGE)

#             if yolo_results and yolo_inference_time and img:
#                 print("\nYOLO Results:")
#                 print(f"Inference Time: {yolo_inference_time:.4f} seconds")
#                 print(yolo_results)

#                 # Plot image with bounding boxes
#                 yolo_df = pd.DataFrame(yolo_results)
#                 img_cv2 = cv2.imread(TEST_IMAGE)

#                 for index, row in yolo_df.iterrows():
#                     xmin = int(row['xmin'])
#                     ymin = int(row['ymin'])
#                     xmax = int(row['xmax'])
#                     ymax = int(row['ymax'])
#                     label = row['name']
#                     confidence = row['confidence']

#                     cv2.rectangle(img_cv2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#                     cv2.putText(img_cv2, f'{label} {confidence:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                 # Display the image using cv2.imshow
#                 cv2.imshow('YOLO Results', img_cv2)
#                 cv2.waitKey(0)  # Wait for any key press
#             else:
#                 print("YOLO inference failed. Check the error messages above.")

#         except Exception as e:
#             print(f"An error occurred: {e}")

#         finally:
#             cv2.destroyAllWindows() # Close the window                




# # yolo_integration.ipynb

# import cv2
# import time
# import os
# import numpy as np
# import pandas as pd
# from PIL import Image
# from ultralytics import YOLO

# # --- Configuration ---
# YOLO_MODEL_NAME = 'yolov8s.pt'  # Or yolov8n.pt, yolov8m.pt, etc.
# YOLO_MODEL_DIR = 'yolo_models'
# CUSTOM_MODEL_PATH = os.path.join(YOLO_MODEL_DIR, YOLO_MODEL_NAME)
# DATASET_DIR = 'data'
# YOLO_CONF_THRESHOLD = 0.5
# YOLO_IOU_THRESHOLD = 0.45

# # Create the YOLO model directory if it doesn't exist
# os.makedirs(YOLO_MODEL_DIR, exist_ok=True)

# def load_yolo_model(model_name, model_dir, custom_model_path=None):
#     """Loads a YOLOv8 model.  Downloads it if it doesn't exist."""

#     # Download the model if it's not present
#     if not os.path.exists(custom_model_path):
#         print(f"Downloading {model_name}...")
#         try:
#             model = YOLO(model_name)  # Downloads the model if it's not already present
#             model.save(custom_model_path)
#             print("Download Complete")
#         except Exception as e:
#             print(f"Error downloading the YOLO model: {e}")
#             return None
#     else:
#         print(f"Loading {model_name} from {custom_model_path}")
#         try:
#             model = YOLO(custom_model_path)
#         except Exception as e:
#             print(f"Error loading the YOLO model from the specified path: {e}")
#             return None

#     return model

# def yolo_detect(model, image_path):
#     """Performs object detection using YOLOv8 on a given image.
#     Args:
#         model: A loaded YOLOv8 model.
#         image_path (str): Path to the image file.
#     Returns:
#         tuple: A tuple containing:
#             - predictions (list): List of dictionaries, where each dictionary represents a detected object.
#             - inference_time (float): Inference time in seconds.
#             - img (PIL.Image): The original PIL Image.
#     """
#     try:
#         img = Image.open(image_path)
#         start_time = time.time()
#         results = model(img)  # Perform inference - this returns a Results object, not a list
#         end_time = time.time()

#         inference_time = end_time - start_time

#         predictions = []  # Prepare a list to store formatted predictions
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 xyxy = box.xyxy[0].tolist()
#                 conf = box.conf[0].item()
#                 class_id = box.cls[0].item()
#                 label = model.names[class_id] #Correct Name
#                 if conf >= YOLO_CONF_THRESHOLD:
#                     predictions.append({'xmin': xyxy[0], 'ymin': xyxy[1], 'xmax': xyxy[2], 'ymax': xyxy[3], 'confidence': conf, 'class': int(class_id), 'name': label}) # int(class_id), make sure this is int before cast into str

#         return predictions, inference_time, img

#     except Exception as e:
#         print(f"Error during YOLO inference: {e}")
#         return None, None, None


# if __name__ == '__main__':
#     # For standalone testing

#     yolo_model = load_yolo_model(YOLO_MODEL_NAME, YOLO_MODEL_DIR, CUSTOM_MODEL_PATH)  # Load model
#     if yolo_model is None:
#         print("YOLO model failed to load. Skipping YOLO inference and display.")
#     else:
#         TEST_IMAGE = os.path.join(DATASET_DIR, 'test', 'images', '000000005345_jpg.rf.48e7947456159d44cbe1a733ad832bf1.jpg')
#         # Load Image
#         if not os.path.exists(TEST_IMAGE):
#             raise FileNotFoundError("Test image does not exist. Make sure it has the correct path")

#         # Perform inference
#         yolo_results, yolo_inference_time, img = yolo_detect(yolo_model, TEST_IMAGE)

#         if yolo_results and yolo_inference_time and img:
#             print("\nYOLO Results:")
#             print(f"Inference Time: {yolo_inference_time:.4f} seconds")
#             print(yolo_results)

#             # Plot image with bounding boxes
#             yolo_df = pd.DataFrame(yolo_results)
#             img_cv2 = cv2.imread(TEST_IMAGE)

#             for index, row in yolo_df.iterrows():
#                 xmin = int(row['xmin'])
#                 ymin = int(row['ymin'])
#                 xmax = int(row['xmax'])
#                 ymax = int(row['ymax'])
#                 label = row['name'] #label = row['class'] change to name
#                 confidence = row['confidence']

#                 cv2.rectangle(img_cv2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#                 cv2.putText(img_cv2, f'{label} {confidence:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Display the image using cv2.imshow
#             cv2.imshow('YOLO Results', img_cv2)
#             cv2.waitKey(0)  # Wait for any key press
#         else:
#             print("YOLO inference failed. Check the error messages above.")

#     cv2.destroyAllWindows()