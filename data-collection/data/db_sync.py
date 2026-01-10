import json

import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def sync_data():



    if os.path.exists('static.csv'):
        print("Syncing Static...")
        static_df = pd.read_csv('static.csv')
        static_df=static_df.replace([np.inf, -np.inf], np.nan)
        static_df=static_df.where(pd.notnull(static_df), None)

        json_data = static_df.to_json(orient='records')
        data = json.loads(json_data)

        supabase.table("static").upsert(data, on_conflict="RunID", ignore_duplicates = True).execute()

    if os.path.exists('training.csv'):
        print("Syncing Training...")
        training_df = pd.read_csv('training.csv')

        training_df=training_df.replace([np.inf, -np.inf], np.nan)
        training_df=training_df.where(pd.notnull(training_df), None)

        json_data = training_df.to_json(orient='records')
        records = json.loads(json_data)
        chunk_size = 2000
        for i in range(0, len(records), chunk_size):
            chunk = records[i: i + chunk_size]
            supabase.table("training").upsert(chunk, on_conflict="RunID,Step" , ignore_duplicates = True).execute()
            print(f"  Training: {i + len(chunk)} rows synced")


    if os.path.exists('summary.csv'):
        print("Syncing Summary...")

        summary_df = pd.read_csv('summary.csv')
        summary_df = summary_df.drop_duplicates(subset=['RunID'], keep='last')
        summary_df = summary_df.replace([np.inf, -np.inf], np.nan)
        summary_df = summary_df.where(pd.notnull(summary_df), None)

        json_data = summary_df.to_json(orient='records')
        data = json.loads(json_data)
        supabase.table("summary").upsert(data, on_conflict="RunID" , ignore_duplicates = True).execute()


if __name__ == "__main__":
    sync_data()

