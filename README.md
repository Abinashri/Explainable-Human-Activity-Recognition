# Human Activity Recognition with Explainable AI (XAI)

An AI-powered video analysis application that recognizes human activities using a **CNN–LSTM deep learning model** and visualizes model attention using **Grad-CAM** for explainable predictions.

<p align="center">
  <a href="https://explainable-human-activity-recognition-gftf5jdadd93hwv42mjd5a.streamlit.app/">
    <strong>🚀 Try the Live Demo</strong>
  </a>
</p>

## 📌 Project Overview

Human Activity Recognition (HAR) is a computer vision task that identifies human actions from video sequences. Although deep learning models can classify activities, understanding why a model makes a particular prediction is also important.

This project combines **Convolutional Neural Networks (CNNs)**, **Long Short-Term Memory (LSTM)** networks, and **Explainable AI (XAI)** to recognize human activities from uploaded videos and provide visual explanations of the predictions.

The application is built with Streamlit, allowing users to interact with the model through a web interface without installing the project locally.

## ✨ Key Features

- **Video-based activity recognition:** Upload a video and obtain a predicted human activity.
- **Person detection:** Detect and extract the person from video frames.
- **CNN-based feature extraction:** Extract spatial features from video frames.
- **LSTM-based temporal learning:** Learn movement patterns across a sequence of frames.
- **Explainable AI with Grad-CAM:** Generate visual heatmaps to help interpret the model's prediction.
- **Interactive web interface:** Run the application through a browser using Streamlit.
- **Five activity classes:** Walking, running, boxing, handclapping, and handwaving.

## 🎯 Supported Activities

| Activity | Description |
|---|---|
| Walking | Recognizes walking movements |
| Running | Recognizes running movements |
| Boxing | Recognizes boxing-related movements |
| Handclapping | Recognizes clapping movements |
| Handwaving | Recognizes handwaving movements |

## 🧠 Technologies Used

- **Programming language:** Python
- **Deep learning:** PyTorch
- **Computer vision:** OpenCV, Torchvision
- **Deep learning architecture:** MobileNetV2 CNN + LSTM
- **Person detection:** Faster R-CNN
- **Explainable AI:** Grad-CAM
- **Web application:** Streamlit
- **Version control:** Git and GitHub
- **Deployment:** Streamlit Community Cloud

## 🏗️ System Architecture

The application follows this workflow:

1. **Video input:** The user uploads a video through the Streamlit interface.
2. **Frame extraction:** A sequence of frames is extracted from the video.
3. **Person detection:** A Faster R-CNN detector identifies the person in the frames.
4. **Preprocessing:** The detected person is cropped, resized, and normalized.
5. **Spatial feature extraction:** MobileNetV2 extracts visual features from each frame.
6. **Temporal learning:** An LSTM processes the sequence of frame features to learn movement patterns.
7. **Activity prediction:** The trained model predicts one of the five supported activities.
8. **Visual explanation:** Grad-CAM generates a heatmap highlighting image regions that influence the selected frame's CNN activation for the predicted class.

```text
            Input Video
                 |
                 v
          Extract Frames
                 |
                 v
        Person Detection
          (Faster R-CNN)
                 |
                 v
       Crop and Preprocess
                 |
                 v
      Spatial Feature Extraction
           (MobileNetV2)
                 |
                 v
       Temporal Sequence Model
                (LSTM)
                 |
                 v
        Activity Prediction
                 |
                 v
        Grad-CAM Heatmap
                 |
                 v
      Prediction + Explanation
```

## 🔍 Explainable AI with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize image regions associated with a model's class-specific activation.

The application generates a heatmap for a selected video frame to help users inspect which visual regions contribute to the model's prediction.

This makes the system more interpretable than displaying only an activity label.

**Note:** A Grad-CAM heatmap is an approximate visual explanation, not proof that the model has correctly understood the action or its causal reasoning.

## 🖥️ Live Demo

Try the deployed application here:

**[Human Activity Recognition + XAI — Open Live App](https://explainable-human-activity-recognition-gftf5jdadd93hwv42mjd5a.streamlit.app/)**

Upload a supported video to view the predicted activity and its visual explanation.

## ⚙️ Run Locally

### Prerequisites

- Python installed on your computer
- Git
- A compatible environment for PyTorch and its dependencies

### 1. Clone the repository

```bash
git clone https://github.com/Abinashri/Explainable-Human-Activity-Recognition.git
```

### 2. Navigate to the project directory

```bash
cd Explainable-Human-Activity-Recognition
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify the model checkpoint

Ensure that the trained model checkpoint is available at the path expected by the application:

```text
outputs/har_model.pt
```

### 6. Start the Streamlit application

```bash
streamlit run app.py
```

The application should open in your browser. If it does not, open the local URL printed in your terminal, usually `http://localhost:8501`.

## 📁 Project Structure

```text
Explainable-Human-Activity-Recognition/
│
├── app.py              # Streamlit application
├── model.py            # CNN-LSTM model definition
├── common.py           # Shared project utilities
├── gradcam.py          # Grad-CAM visualization
├── prepare_data.py     # Data preparation
├── train.py            # Model training
├── evaluate.py         # Model evaluation
├── featstore.py        # Feature storage utilities
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
│
└── outputs/
    └── har_model.pt    # Trained model checkpoint
```

*The structure above highlights the main project files; generated files and local datasets may not be included in the repository.*

## 🎓 Applications

- Human activity analysis
- Video-based action recognition
- Interpretable computer vision systems
- Research and experimentation in Explainable AI
- Educational demonstrations of CNN–LSTM architectures

## 🚀 Future Enhancements

- Improve classification accuracy across different subjects, environments, and camera angles.
- Evaluate the model on additional activity datasets.
- Extend the number of supported activity classes.
- Add quantitative evaluation metrics and confusion matrices.
- Explore additional explainability techniques.
- Optimize inference speed for longer videos and resource-constrained environments.

## 👩‍💻 Author

**Abinashri**

GitHub: [@Abinashri](https://github.com/Abinashri)

## 📄 License

No license has been specified yet. Add an appropriate open-source license if you intend to permit others to reuse or modify this project's code.

---

⭐ If you find this project interesting, consider starring the repository!
