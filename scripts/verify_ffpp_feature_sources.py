from pathlib import Path
import pandas as pd


FEATURE_FILE = Path("data/features/ffpp_features.csv")
DATASET_ROOT = Path("data/datasets/FaceForensics++")

REAL_DIR = (
    DATASET_ROOT
    / "original_sequences"
    / "youtube"
    / "c23"
    / "videos"
)

FAKE_DIR = (
    DATASET_ROOT
    / "manipulated_sequences"
    / "Deepfakes"
    / "c23"
    / "videos"
)


def main():
    df = pd.read_csv(FEATURE_FILE)

    missing = []

    for _, row in df.iterrows():
        video_name = row["video_name"]
        label = row["label"]

        if label == "real":
            source_path = REAL_DIR / video_name
        elif label == "fake":
            source_path = FAKE_DIR / video_name
        else:
            missing.append((video_name, label, "unknown label"))
            continue

        if not source_path.is_file():
            missing.append((video_name, label, str(source_path)))

    print(f"Feature rows: {len(df)}")
    print(f"Unique video names: {df['video_name'].nunique()}")
    print(f"Missing source files: {len(missing)}")

    if missing:
        print("\nMissing files:")
        for item in missing:
            print(item)
    else:
        print("\nAll feature rows have matching source videos.")


if __name__ == "__main__":
    main()