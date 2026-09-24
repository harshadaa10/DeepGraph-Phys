# DeepGraph-Phys

## Generalizable Deepfake Video Detection Using Cross-Regional Physiological and Motion Consistency with Dynamic Graph Signal Processing

DeepGraph-Phys is a research-oriented deepfake video detection framework designed to investigate whether physiological and facial-motion relationships remain physically consistent across facial regions.

Instead of relying only on visual artifacts, the project models facial regions as nodes in a dynamic graph and analyzes physiological and motion signals using Graph Signal Processing.

## Core Idea

Traditional approach:

Video → Visual Features → Classifier → Real/Fake

DeepGraph-Phys:

Video
→ Face Detection
→ Facial Landmarks
→ Facial Regions
→ rPPG + Motion Signals
→ Dynamic Facial Graph
→ Graph Signal Processing
→ Anomaly Detection
→ Authenticity Score
→ Real/Fake + Explanation

## Main Research Goal

Investigate whether cross-regional physiological-motion consistency can improve generalization to deepfake generation methods or datasets that were not seen during model development.

## Core Components

1. Video preprocessing
2. Face detection and facial landmarks
3. Facial ROI extraction
4. rPPG physiological signal extraction
5. Facial motion extraction
6. Dynamic facial graph construction
7. Graph Signal Processing
8. Graph consistency feature extraction
9. One-class anomaly detection
10. Cross-dataset evaluation
11. Explainability
12. Streamlit forensic demo

## Technology Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- SciPy
- Pandas
- NetworkX
- scikit-learn
- Matplotlib
- Streamlit

## Project Status

### Day 1
- [x] Project environment
- [x] Project structure
- [ ] Dependency installation
- [ ] Video preprocessing
- [ ] Face detection
- [ ] Facial landmarks
- [ ] ROI extraction
- [ ] rPPG extraction
- [ ] Motion extraction

### Day 2
- [ ] Dynamic facial graph
- [ ] Graph Signal Processing
- [ ] Feature generation
- [ ] Anomaly detection
- [ ] Evaluation
- [ ] Ablation study
- [ ] Explainability
- [ ] Streamlit application

