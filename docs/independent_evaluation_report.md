# DeepGraph-Phys: Independent Evaluation Report

## 1. Objective

Evaluate the existing frozen motion-only Random Forest on 100 previously
unused FaceForensics++ video sequence IDs without retraining or tuning.

## 2. Evaluation setup

- Dataset: FaceForensics++
- Manipulation method: Deepfakes
- Video quality: c23
- Sample size: 100 videos
- Real videos: 50
- Fake videos: 50
- Model: phase23_motion_only_random_forest
- Features: 30 motion-based graph signal processing features
- Decision threshold: 0.50, fixed before this evaluation
- Label mapping: real = 0, fake = 1
- Model retraining: No
- Threshold tuning: No
- Probability calibration: No

The model score is not a calibrated probability.

## 3. Results

| Metric | Result |
|---|---:|
| Accuracy | 0.6800 |
| Precision | 0.7045 |
| Recall | 0.6200 |
| F1 score | 0.6596 |
| ROC-AUC | 0.7696 |
| PR-AUC | 0.7853 |

## 4. Confusion matrix

Rows are actual classes; columns are predicted classes.

| Actual class | Predicted real | Predicted fake |
|---|---:|---:|
| Real | 37 | 13 |
| Fake | 19 | 31 |

- True negatives: 37
- False positives: 13
- False negatives: 19
- True positives: 31
- Correct predictions: 68
- Incorrect predictions: 32

## 5. Interpretation

The model demonstrates useful ranking ability on this independent sample,
with ROC-AUC of 0.7696. At the fixed 0.50 threshold, it correctly classified
68 of 100 videos.

The 13 false positives show that some real videos were flagged as fake.
The 19 false negatives show that some fake videos were classified as real.
The results indicate that motion-based features alone do not reliably
classify every video.

## 6. Limitations

1. This is an evaluation on unseen sequence IDs within the same dataset family,
   not a cross-dataset generalization experiment.
2. The sample contains only 100 videos and one manipulation method at c23.
3. Results may depend on video compression, source characteristics, and
   the selected sample.
4. The decision threshold was fixed at 0.50 and was not optimized here.
5. Model scores are not calibrated probabilities.
6. These results should not be presented as evidence of real-world deployment
   readiness.
7. This evaluation must remain separate from model development and tuning.

## 7. Reproducibility artifacts

All files are stored outside the Git repository in:
DeepGraphPhys_ExternalEval

- manifest.csv
- external_motion_features.csv
- external_extraction_status.csv
- external_frozen_model_predictions.csv
- external_error_analysis.csv

## 8. Integrity statement

The existing frozen model was used for inference only. No retraining,
recalibration, or threshold tuning was performed on this sample. The reserved
final-test evaluation was not accessed during this experiment.

## 9. Conclusion

The frozen motion-only model achieved 68% accuracy and ROC-AUC of 0.7696
on 100 previously unused FaceForensics++ sequence-ID videos. This provides
additional same-dataset-family evaluation evidence, while leaving cross-dataset
generalization and broader robustness for future independent studies.

