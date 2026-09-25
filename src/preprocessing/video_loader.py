from pathlib import Path
import cv2


def get_video_info(video_path):
    """
    Read basic information about a video.

    Returns:
        Dictionary containing video metadata.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "filename": video_path.name,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


def read_video(video_path, max_frames=None):
    """
    Load video frames using OpenCV.
    """

    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        frames.append(frame)

        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    return frames


if __name__ == "__main__":
    print("DeepGraph-Phys Video Loader")
    print("Module loaded successfully.")