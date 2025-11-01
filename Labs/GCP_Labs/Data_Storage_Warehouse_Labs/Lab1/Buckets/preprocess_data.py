import os
import pandas as pd
import seaborn as sns
from google.cloud import storage

#CONFIGURATION
BUCKET_NAME = "gcp-lab-bucket-anjali" 
DESTINATION_BLOB = "processed/processed_data.csv"  # path inside the bucket
LOCAL_PATH = "data/processed_data.csv"

#LOAD DATA
df = sns.load_dataset("titanic")
print(f"Loaded Titanic dataset with {df.shape[0]} rows and {df.shape[1]} columns")

#PREPROCESS
#Keep relevant columns
keep_cols = [
    "survived", "pclass", "sex", "age", "sibsp", "parch",
    "fare", "embarked", "class", "who", "adult_male", "alone"
]
df = df[keep_cols].copy()

#Fill missing values
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode().iloc[0])

#Convert booleans to integers
bool_cols = df.select_dtypes(include="bool").columns
for c in bool_cols:
    df[c] = df[c].astype(int)

#Create a derived column
df["family_size"] = df["sibsp"] + df["parch"] + 1

#SAVE LOCALLY
os.makedirs("data", exist_ok=True)
df.to_csv(LOCAL_PATH, index=False)
print(f"Saved processed dataset to {LOCAL_PATH}")

#UPLOAD TO GCS
def upload_to_gcs(bucket_name, local_path, dest_blob):
    """Uploads a local file to an existing GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} → gs://{bucket_name}/{dest_blob}")

upload_to_gcs(BUCKET_NAME, LOCAL_PATH, DESTINATION_BLOB)
print("Done — processed Titanic dataset uploaded successfully!")