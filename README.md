#  Plankton Detection using YOLOv8

This project detects plankton in microscope images using a custom trained YOLOv8 object detection model and a Streamlit web application.

---

## 🚀 Features

- Detect plankton from microscope images
- Web interface built with Streamlit
- Custom YOLOv8 trained model
- Adjustable confidence threshold

---

## 🧠 Model

The detection model was trained using YOLOv8 on a plankton microscopy dataset.

Framework used:
- Ultralytics YOLOv8

---

## 🖥️ Run the App Locally

Install dependencies

```
pip install -r requirements.txt
```

Run the Streamlit app

```
streamlit run app.py
```

---

## 📂 Project Structure

```
plankton-detection-ai
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
```

---

## 🔮 Future Improvements

- Add species analytics
- Support video detection
- Improve model accuracy