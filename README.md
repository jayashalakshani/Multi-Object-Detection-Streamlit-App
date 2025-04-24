# 🧠 Multi-Model Object Detection with Streamlit

This project implements a hybrid object detection application using two powerful models:

- **YOLOv8**: For high-performance bounding box detection.
- **Custom CNN Model**: Trained for multi-label classification across 78 COCO-style object classes.

The app is deployed with **Streamlit** for an interactive interface, and leverages **Google Colab (TPU runtime)** for training the CNN model, while development and testing are done locally using **Jupyter Notebook**.

---

## 🔧 Features

- ✅ Real-time object detection using YOLOv8.
- ✅ Multi-label image classification using a custom CNN model.
- ✅ Side-by-side visual comparison of both models.
- ✅ Image augmentation for robust CNN training.
- ✅ Keras Tuner support for CNN optimization.
- ✅ Easy-to-use interface with Streamlit.

---

## 🚀 Run Locally

1. Clone the repo:
```bash
git clone https://github.com/yourusername/multi-model-object-detection-streamlit.git
cd multi-model-object-detection-streamlit
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3.Run the app
```bash
streamlit run app.py
```

## 🏋️ Model Training
- CNN was trained on Google Colab with a TPU v2-8 runtime using a balanced multi-label dataset.
- Keras Tuner was used to optimize CNN architecture.
- Model saved in .keras format.

## 📸 Dataset
- Based on COCO-style multi-label annotations.
- Each image can belong to multiple of the 78 object classes.

## 📦 Requirements
- See requirements.txt.

## 🧠 Future Enhancements
- Add model interpretability (Grad-CAM).
- Add webcam support.
- Deploy via Streamlit Cloud or Docker.
