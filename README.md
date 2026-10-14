# DeepGraph-Phys

### Deepfake Video Detection via Cross-Regional Physiological and Motion Consistency Using Dynamic Graph Signal Processing

DeepGraph-Phys is a research-oriented Python project that investigates deepfake video detection using facial-region analysis, physiological signal processing, facial motion features, and graph signal processing.

The project represents selected facial regions as graph nodes and analyzes relationships between regions over time. It explores whether physiological and motion-related inconsistencies can provide useful signals for distinguishing real videos from manipulated videos.

> **Research status:** The current evaluated classifier is a frozen, motion-only Random Forest. The reported metrics below apply to that model and should not be interpreted as results for a fully evaluated combined physiological-plus-motion classifier.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Pipeline](#system-pipeline)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [How to Run the Project](#how-to-run-the-project)
- [Evaluation Results](#evaluation-results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)
- [Author](#author)

## Overview

Deepfake videos can convincingly manipulate facial appearance and expressions. DeepGraph-Phys explores a complementary approach that analyzes temporal information and relationships among facial regions rather than relying exclusively on visual appearance.

The framework includes facial landmark detection, facial-region RGB signal extraction, remote photoplethysmography (rPPG) estimation, facial motion analysis, graph construction, graph-based feature extraction, and machine-learning evaluation.

## Key Features

- **Facial landmark detection:** Uses MediaPipe to obtain facial landmarks.
- **Facial-region analysis:** Organizes selected areas into seven regions: forehead, left cheek, right cheek, left eye, right eye, nose, and mouth.
- **Physiological signal analysis:** Extracts facial RGB signals and estimates rPPG signals from selected regions.
- **Motion feature extraction:** Analyzes temporal facial landmark changes.
- **Graph representation:** Models facial regions as nodes and selected relationships as edges.
- **Graph signal processing:** Investigates spatial and temporal characteristics of graph-based signals.
- **Machine-learning inference:** Uses a previously trained Random Forest for the evaluated motion-feature pipeline.
- **Streamlit interface:** Provides an interactive interface for testing videos when the required model and dependencies are available.

## System Pipeline

```text
Input Video
    |
    v
Video Frame Extraction
    |
    v
Facial Landmark Detection
    |
    v
Facial Region Selection
    |
    +--------------------------+
    |                          |
    v                          v
RGB / rPPG Analysis       Facial Motion Analysis
    |                          |
    +-------------+------------+
                  |
                  v
      Graph Construction and
       Graph Signal Processing
                  |
                  v
          Feature Preparation
                  |
                  v
      Trained Motion-Only Model
                  |
                  v
          Real / Fake Prediction
                  |
                  v
        Evaluation and Reporting
```

The diagram illustrates the broader research pipeline. The reported external evaluation uses the motion-only model.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Video processing |
| MediaPipe | Facial landmark detection |
| NumPy and SciPy | Numerical and signal-processing operations |
| Pandas | Feature and result tables |
| NetworkX | Graph representation and operations |
| scikit-learn | Machine-learning inference and evaluation |
| Matplotlib | Signal and result visualization |
| Streamlit | Interactive application |
| Git and GitHub | Version control |

## Repository Structure

```text
DeepGraph-Phys/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── docs/
│   └── independent_evaluation_report.md
├── models/
│   └── phase23_motion_only_rf.joblib
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── experiments/
├── scripts/
├── src/
│   ├── preprocessing/
│   ├── physiology/
│   ├── motion/
│   ├── graph/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── explainability/
├── tests/
├── requirements.txt
└── README.md
```

This is a high-level map of the project. Some folders may contain additional files, and the trained model file must be present locally for the demo to run.

## Installation

### Prerequisites

- Windows, Linux, or macOS
- Python 3.10 recommended for the tested environment
- Git
- A webcam is not required; the demo processes video files.

### 1. Clone the repository

```bash
git clone https://github.com/harshadaa10/DeepGraph-Phys.git
cd DeepGraph-Phys
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.10 -m venv .venv
.\.venv/scripts/activate
```

If PowerShell blocks environment activation, consult the Python virtual-environment documentation or use the virtual environment's Python executable directly.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the dependency versions specified by the repository where available. MediaPipe and OpenCV APIs can vary across versions.

### 4. Verify the required model

Check that this file exists:

```text
models/phase23_motion_only_rf.joblib
```

If the file is not included in the cloned repository, obtain the authorized model artifact from the project owner and place it at the expected path. Do not substitute a different model without checking feature compatibility.

## How to Run the Project

### Launch the Streamlit application

From the repository root, with the virtual environment activated, run:

```powershell
python -m streamlit run app\streamlit_app.py
```

Streamlit will display a local URL in the terminal, typically:

```text
http://localhost:8501
```

Open the URL in your browser.

### Test a video

1. Open the Streamlit application.
2. Upload a supported video file using the interface.
3. Wait for frame processing, face detection, and feature extraction.
4. Review the displayed prediction and available score or diagnostic information.
5. Test additional videos to observe how predictions vary.

The exact output depends on the application implementation and video quality. A model score is not necessarily a calibrated probability.

### Troubleshooting

- **Model file not found:** Verify that `models/phase23_motion_only_rf.joblib` exists at the expected path.
- **Module not found:** Activate the correct virtual environment and install the dependencies.
- **Face detection failure:** Try a video with a clearly visible face and adequate lighting.
- **Slow processing:** Try a shorter video or lower-resolution input if supported by the application.
- **Feature mismatch:** Verify that the loaded model and feature extractor use the same expected feature names and order.

## Evaluation Results

The frozen motion-only Random Forest was evaluated on 100 previously unused FaceForensics++ video sequence-ID samples: 50 real and 50 fake videos. The sample used the Deepfakes manipulation method and c23 quality.

| Metric | Result |
|---|---:|
| Accuracy | 68.00% |
| Precision | 0.7045 |
| Recall | 0.6200 |
| F1-score | 0.6596 |
| ROC-AUC | 0.7696 |
| PR-AUC | 0.7853 |

### Confusion Matrix

Rows represent actual classes; columns represent predicted classes.

| Actual class | Predicted real | Predicted fake |
|---|---:|---:|
| Real | 37 | 13 |
| Fake | 19 | 31 |

The model correctly classified 68 of the 100 videos. At the fixed decision threshold of 0.50, it produced 13 false positives and 19 false negatives.

The independent evaluation used the existing frozen model without retraining, recalibration, or threshold tuning on this sample.

A detailed report is available at [`docs/independent_evaluation_report.md`](docs/independent_evaluation_report.md).

## Limitations

- The external evaluation uses unseen sequence IDs from the same dataset family; it is not cross-dataset validation.
- The sample contains 100 videos and one manipulation method at c23 quality.
- The reported results belong to the motion-only Random Forest, not a fully evaluated combined physiological-plus-motion classifier.
- Performance can be affected by lighting, facial movement, video compression, source characteristics, and face-detection quality.
- The model score is not a calibrated probability.
- The results do not establish real-world deployment readiness.

## Future Work

- Evaluate on independent datasets to investigate cross-dataset generalization.
- Implement and assess combined physiological-plus-motion classification.
- Test additional manipulation methods and video-quality settings.
- Improve robustness of rPPG extraction under varying lighting and motion.
- Investigate temporal graph modeling and feature explainability.
- Evaluate on larger and more diverse samples.

## References

1. A. Rössler et al., “FaceForensics++: Learning to Detect Manipulated Facial Images,” ICCV, 2019.
2. W. Wang et al., “Algorithmic Principles of Remote PPG,” IEEE Transactions on Biomedical Engineering, 2017.
3. D. I. Shuman et al., “The Emerging Field of Signal Processing on Graphs,” IEEE Signal Processing Magazine, 2013.
4. L. Breiman, “Random Forests,” Machine Learning, 2001.
5. MediaPipe documentation: https://ai.google.dev/edge/mediapipe
6. OpenCV documentation: https://docs.opencv.org/
7. scikit-learn documentation: https://scikit-learn.org/stable/

## Author

**Harshada Suryawanshi**  
B.Tech Computer Science and Engineering  
MIT World Peace University (MIT-WPU)

GitHub: https://github.com/harshadaa10

## Disclaimer

DeepGraph-Phys is an academic research project. Its outputs are experimental and should not be used as the sole basis for deciding whether a video is authentic or manipulated.
