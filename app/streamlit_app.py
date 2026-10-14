from pathlib import Path
import tempfile

import joblib
import numpy as np
import streamlit as st

from src.features.video_pipeline import process_video_to_features


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "phase23_motion_only_rf.joblib"
LANDMARKER_PATH = ROOT / "models" / "mediapipe" / "face_landmarker.task"

DEMO_THRESHOLD = 0.50


# --------------------------------------------------
# 2. Page configuration and light UI
# --------------------------------------------------

st.set_page_config(
    page_title="DeepGraph-Phys",
    page_icon="🎬",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
        color: #1e293b;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(120deg, #e8f1ff, #f0fdfa);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: #1d4ed8;
        margin: 0;
        font-size: 2.2rem;
    }

    .hero p {
        color: #475569;
        margin: 0.6rem 0 0;
        line-height: 1.6;
    }

    h2, h3 {
        color: #1e293b !important;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #93c5fd;
        border-radius: 12px;
        padding: 0.8rem;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
    }

    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 2.8rem;
    }

    .app-footer {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🎬 DeepGraph-Phys</h1>
        <p>
            Video deepfake detection using facial motion analysis
            and graph-based research methods.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Research prototype only. Predictions can be wrong. "
    "The fake-class score is a model estimate, not a calibrated probability."
)

st.caption("Phase 23 · Motion-only Random Forest · 30 features")


# --------------------------------------------------
# 4. Check files and load model
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
        raise ValueError("Unexpected model score type.")

    if label_mapping.get("fake") != 1 or label_mapping.get("real") != 0:
        raise ValueError("Unexpected model label mapping.")

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
# 5. Upload and preview video
# --------------------------------------------------

st.subheader("Upload a video")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi"],
    help="For best results, use a video containing a visible face.",
)

if uploaded_file is not None:
    st.video(uploaded_file)

    st.caption(
        f"File: {uploaded_file.name} · "
        f"Size: {len(uploaded_file.getbuffer()) / (1024 * 1024):.2f} MB"
    )

analyse_clicked = (
    st.button("Analyse video", type="primary", use_container_width=True)
    if uploaded_file is not None
    else False
)


# --------------------------------------------------
# 6. Extract features and predict
# --------------------------------------------------

if uploaded_file is not None and analyse_clicked:
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(uploaded_file.getbuffer())
            temporary_path = Path(temporary_file.name)

        with st.spinner("Extracting features and analysing video..."):
            features, metadata = process_video_to_features(
                video_path=temporary_path,
                model_path=LANDMARKER_PATH,
            )

        missing_features = [
            name for name in feature_columns if name not in features
        ]

        if missing_features:
            raise ValueError(
                "Missing required features: " + ", ".join(missing_features)
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
            raise ValueError("Extracted feature count does not match the model.")

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

        # Results
        st.divider()
        st.subheader("Analysis result")

        if prediction == "Potentially fake":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        score_col, threshold_col = st.columns(2)

        with score_col:
            st.metric("Fake-class score", f"{fake_score:.1%}")

        with threshold_col:
            st.metric("Demo threshold", f"{DEMO_THRESHOLD:.0%}")

        st.progress(
            min(max(fake_score, 0.0), 1.0),
            text="Fake-class score",
        )

        st.caption(
            "The score is not a calibrated probability. The threshold is "
            "provisional and may not generalise to other videos."
        )

        # Processing diagnostics
        if isinstance(metadata, dict):
            face_detection_rate = metadata.get("face_detection_rate")

            if face_detection_rate is not None:
                try:
                    rate = float(face_detection_rate)
                    st.metric("Face detection rate", f"{rate:.1%}")

                    if rate < 0.80:
                        st.warning(
                            "Face detection was below 80%. "
                            "The result may be less reliable."
                        )
                except (TypeError, ValueError):
                    pass

            with st.expander("Processing details"):
                st.json(metadata)

        else:
            with st.expander("Processing details"):
                st.write(metadata)

    except Exception as exc:
        st.error(f"Video analysis failed: {exc}")
        st.info(
            "Check that the video is readable, contains a detectable face, "
            "and can be processed by the feature pipeline."
        )

    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                st.warning("Could not remove the temporary video file.")


# --------------------------------------------------
# 7. Footer
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="app-footer">
        DeepGraph-Phys · Experimental academic project
    </div>
    """,
    unsafe_allow_html=True,
)