
from pathlib import Path
import tempfile

import joblib
import numpy as np
import streamlit as st

from src.features.video_pipeline import process_video_to_features


# --------------------------------------------------
# 1. Paths and demo settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "phase23_motion_only_rf.joblib"
LANDMARKER_PATH = ROOT / "models" / "mediapipe" / "face_landmarker.task"

# Provisional demo cutoff only.
# This is not a calibrated or clinically meaningful probability threshold.
DEMO_THRESHOLD = 0.50


# --------------------------------------------------
# 2. Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="DeepGraph-Phys",
    page_icon="🎥",
    layout="centered",
)

st.title("DeepGraph-Phys")
st.subheader("Research prototype for video deepfake detection")

st.warning(
    "Research demonstration only. Results can be wrong and must not be "
    "treated as definitive proof that a video is real or fake. "
    "The displayed score is a model estimate, not a calibrated probability."
)

st.caption(
    "Phase 23 motion-only Random Forest · 30 features · "
    "Provisional demo threshold: 0.50"
)


# --------------------------------------------------
# 3. Check required files and load model
# --------------------------------------------------

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not LANDMARKER_PATH.exists():
    st.error(f"MediaPipe face landmarker file not found: {LANDMARKER_PATH}")
    st.stop()

try:
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    imputation_medians = bundle["imputation_medians"]
    label_mapping = bundle["label_mapping"]

    if bundle.get("score_type") != "predict_proba_fake":
        raise ValueError("The saved model bundle has an unexpected score type.")

    if label_mapping.get("fake") != 1 or label_mapping.get("real") != 0:
        raise ValueError("The saved model bundle has an unexpected label mapping.")

    if len(feature_columns) != 30:
        raise ValueError(
            f"Expected 30 motion features, found {len(feature_columns)}."
        )

    if not hasattr(model, "predict_proba"):
        raise ValueError("The saved model does not support predict_proba().")

    model_classes = list(model.classes_)
    if 1 not in model_classes:
        raise ValueError("The model does not contain the fake class (label 1).")

    fake_class_index = model_classes.index(1)

except Exception as exc:
    st.error(f"Could not load the Phase 23 model bundle: {exc}")
    st.stop()


# --------------------------------------------------
# 4. Upload a video
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a video to analyse",
    type=["mp4", "mov", "avi"],
    help="Use a video containing a visible face for best results.",
)


# --------------------------------------------------
# 5. Extract features and predict
# --------------------------------------------------

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Analyse video", type="primary"):
        suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
        temporary_path = None

        try:
            # Save the upload temporarily so the video pipeline can read it.
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temporary_file:
                temporary_file.write(uploaded_file.getbuffer())
                temporary_path = Path(temporary_file.name)

            with st.spinner("Extracting video features and analysing..."):
                features, metadata = process_video_to_features(
                    video_path=temporary_path,
                    model_path=LANDMARKER_PATH,
                )

            # Select only the exact 30 features expected by Phase 23.
            # Extra physiology features from the extractor are intentionally ignored.
            missing_features = [
                name for name in feature_columns if name not in features
            ]

            if missing_features:
                raise ValueError(
                    "The video pipeline did not produce required features: "
                    + ", ".join(missing_features)
                )

            feature_values = []

            for name in feature_columns:
                value = features[name]

                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    numeric_value = np.nan

                if not np.isfinite(numeric_value):
                    numeric_value = float(imputation_medians[name])

                feature_values.append(numeric_value)

            feature_vector = np.asarray(
                feature_values,
                dtype=float,
            ).reshape(1, -1)

            if feature_vector.shape[1] != len(feature_columns):
                raise ValueError("The extracted feature count does not match the model.")

            # Class 1 represents fake.
            fake_score = float(
                model.predict_proba(feature_vector)[0][fake_class_index]
            )

            if not np.isfinite(fake_score):
                raise ValueError("The model returned an invalid score.")

            prediction = (
                "Potentially fake"
                if fake_score >= DEMO_THRESHOLD
                else "Potentially real"
            )

            # Display prediction and score.
            st.divider()
            st.subheader("Analysis result")

            if prediction == "Potentially fake":
                st.error(f"Prediction: {prediction}")
            else:
                st.success(f"Prediction: {prediction}")

            st.metric(
                label="Model fake-class score",
                value=f"{fake_score:.1%}",
            )

            st.progress(min(max(fake_score, 0.0), 1.0))

            st.caption(
                f"Demo rule: fake-class score ≥ {DEMO_THRESHOLD:.2f} "
                "is labelled 'Potentially fake'. This cutoff is provisional, "
                "not calibrated, and may not generalise to other videos."
            )

            # Display processing metadata when available.
            st.subheader("Video processing details")

            if isinstance(metadata, dict):
                st.json(metadata)

                face_detection_rate = metadata.get("face_detection_rate")

                if face_detection_rate is not None:
                    try:
                        rate = float(face_detection_rate)
                        st.write(f"Face detection rate: {rate:.1%}")

                        if rate < 0.80:
                            st.warning(
                                "Face detection was below 80%. The result may "
                                "be less reliable because too few frames may "
                                "have contained a detected face."
                            )
                    except (TypeError, ValueError):
                        pass
            else:
                st.write(metadata)

        except Exception as exc:
            st.error(f"Video analysis failed: {exc}")
            st.info(
                "Check that the video is readable, contains a detectable face, "
                "and can be processed by the project's feature pipeline."
            )

        finally:
            # Always remove the temporary uploaded video.
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    st.warning("Could not remove the temporary video file.")


st.divider()
st.caption(
    "DeepGraph-Phys · Phase 23 model integration · "
    "Experimental academic project"
)
