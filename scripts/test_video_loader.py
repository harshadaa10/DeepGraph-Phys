from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing.video_loader import get_video_info, read_video


REAL_VIDEO = PROJECT_ROOT / "data" / "raw" / "real" / "sample_real.mp4"
FAKE_VIDEO = PROJECT_ROOT / "data" / "raw" / "fake" / "sample_fake.mp4"


def test_video(video_path, label):
    print("\n" + "=" * 50)
    print(f"Testing {label} video")
    print("=" * 50)

    info = get_video_info(video_path)

    print(f"File       : {info['filename']}")
    print(f"FPS        : {info['fps']:.2f}")
    print(f"Frames     : {info['frame_count']}")
    print(f"Resolution : {info['width']} x {info['height']}")
    print(f"Duration   : {info['duration_seconds']:.2f} seconds")

    frames = read_video(video_path, max_frames=10)

    print(f"Test frames successfully loaded: {len(frames)}")

    if len(frames) > 0:
        print(f"Frame shape: {frames[0].shape}")


if __name__ == "__main__":

    print("\nDeepGraph-Phys — Video Pipeline Test")

    test_video(REAL_VIDEO, "REAL")
    test_video(FAKE_VIDEO, "FAKE")

    print("\nVideo pipeline test completed successfully.")