from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.preprocessing.face_localizer import (
    PrimaryFaceLocalizer,
    crop_face_region,
)

from src.preprocessing.face_landmarker import (
    FaceLandmarkDetector,
    draw_landmarks,
)


VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "FaceForensics++"
    / "original_sequences"
    / "youtube"
    / "c23"
    / "videos"
    / "033.mp4"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mediapipe"
    / "face_landmarker.task"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)


def load_middle_frame():

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    frame_number = (
        total_frames // 2
    )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number,
    )

    success, frame = (
        capture.read()
    )

    capture.release()

    if not success:
        raise RuntimeError(
            "Could not read frame."
        )

    return (
        frame,
        frame_number,
    )


def main():

    print(
        "\nDeepGraph-Phys — "
        "Primary Face Localization Test"
    )

    print("-" * 65)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame, frame_number = (
        load_middle_frame()
    )

    print(
        "Frame number:",
        frame_number,
    )

    print(
        "Frame resolution:",
        frame.shape[1],
        "x",
        frame.shape[0],
    )

    localizer = (
        PrimaryFaceLocalizer()
    )

    faces = (
        localizer.detect_faces(
            frame
        )
    )

    print(
        "\nFaces detected:",
        len(faces),
    )

    for index, face in enumerate(
        faces,
        start=1,
    ):

        print(
            f"Face {index}: "
            f"x={face.x}, "
            f"y={face.y}, "
            f"w={face.width}, "
            f"h={face.height}"
        )

    primary_face = (
        localizer.select_primary_face(
            frame,
            faces,
        )
    )

    if primary_face is None:

        print(
            "\nNo primary face detected."
        )

        return

    print(
        "\nSelected primary face:"
    )

    print(
        "x:",
        primary_face.x,
    )

    print(
        "y:",
        primary_face.y,
    )

    print(
        "width:",
        primary_face.width,
    )

    print(
        "height:",
        primary_face.height,
    )

    # Draw selected face box.
    box_image = frame.copy()

    cv2.rectangle(
        box_image,
        (
            primary_face.x,
            primary_face.y,
        ),
        (
            primary_face.x2,
            primary_face.y2,
        ),
        (0, 255, 0),
        2,
    )

    box_output = (
        OUTPUT_DIR
        / "ffpp_033_primary_face_box.jpg"
    )

    cv2.imwrite(
        str(box_output),
        box_image,
    )

    # Expanded crop.
    crop, coordinates = (
        crop_face_region(
            frame,
            primary_face,
            margin=0.70,
        )
    )

    print(
        "\nExpanded crop coordinates:",
        coordinates,
    )

    print(
        "Crop resolution:",
        crop.shape[1],
        "x",
        crop.shape[0],
    )

    crop_output = (
        OUTPUT_DIR
        / "ffpp_033_auto_face_crop.jpg"
    )

    cv2.imwrite(
        str(crop_output),
        crop,
    )

    # MediaPipe landmarks.
    landmark_detector = (
        FaceLandmarkDetector(
            str(MODEL_PATH)
        )
    )

    landmarks = (
        landmark_detector.detect(
            crop
        )
    )

    landmark_detector.close()

    print(
        "\nMediaPipe face detected:",
        landmarks is not None,
    )

    if landmarks is not None:

        print(
            "Landmarks:",
            len(landmarks),
        )

        annotated = (
            draw_landmarks(
                crop,
                landmarks,
            )
        )

        landmark_output = (
            OUTPUT_DIR
            / (
                "ffpp_033_auto_"
                "face_landmarks.jpg"
            )
        )

        cv2.imwrite(
            str(landmark_output),
            annotated,
        )

        print(
            "Landmark image:",
            landmark_output,
        )

    print(
        "\nValidation checks:"
    )

    print(
        "At least one face localized:",
        len(faces) > 0,
    )

    print(
        "Primary face selected:",
        primary_face is not None,
    )

    print(
        "MediaPipe landmarks found:",
        landmarks is not None,
    )

    print(
        "478 landmarks found:",
        (
            landmarks is not None
            and len(landmarks) == 478
        ),
    )


if __name__ == "__main__":
    main()