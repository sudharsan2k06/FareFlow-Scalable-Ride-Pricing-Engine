print("Script started")
import pandas as pd
import glob

files = glob.glob("data/feature_engineered/features_split_*.parquet")

for i, file in enumerate(files, 1):
    
    df = pd.read_parquet(file)
    
    sample_df = df.sample(n=100000, random_state=42)
    
    sample_df.to_parquet(f"data/sampled/sample_{i}.parquet", index=False)

    print(f"Saved sample_{i}.parquet")