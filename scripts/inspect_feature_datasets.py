import pandas as pd

files = [
    "data/features/ffpp_features.csv",
    "data/features/ffpp_graph_features.csv",
    "data/features/deepgraph_features.csv",
]

for file_path in files:
    df = pd.read_csv(file_path)

    print(f"\nFILE: {file_path}")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())
    print("\nFirst 2 rows:")
    print(df.head(2).to_string(index=False))