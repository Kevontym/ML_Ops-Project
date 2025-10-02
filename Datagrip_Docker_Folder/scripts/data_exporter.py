# scripts/data_exporter.py
import pandas as pd
import os
from datetime import datetime
from sqlalchemy import create_engine
import json
from dotenv import load_dotenv


class DataExporter:
    def __init__(self):
        load_dotenv()
        db_user = os.getenv('POSTGRES_USER')
        db_password = os.getenv('POSTGRES_PASSWORD')
        db_name = os.getenv('POSTGRES_DB')
        db_port = os.getenv('POSTGRES_PORT')

        self.engine = create_engine(f"postgresql://{db_user}:{db_password}@localhost:{db_port}/{db_name}")
        self.export_dir = "data/exports"
        os.makedirs(self.export_dir, exist_ok=True)

    def export_latest_snapshot(self, sample_size=None):
        """Export current database state with metadata"""
        print("📤 Exporting database snapshot...")

        # Query data (with optional sampling for testing)
        if sample_size:
            query = f"SELECT * FROM patient_records ORDER BY RANDOM() LIMIT {sample_size}"
        else:
            query = "SELECT * FROM patient_records"

        df = pd.read_sql(query, self.engine)

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"patient_data_{timestamp}"

        # Export to multiple formats
        files_created = {}

        # Parquet (for Colab)
        parquet_path = f"{self.export_dir}/{base_filename}.parquet"
        df.to_parquet(parquet_path, index=False)
        files_created['parquet'] = parquet_path

        # CSV (for analysis)
        csv_path = f"{self.export_dir}/{base_filename}.csv"
        df.to_csv(csv_path, index=False)
        files_created['csv'] = csv_path

        # Metadata file
        metadata = {
            "export_timestamp": timestamp,
            "total_records": len(df),
            "data_schema": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "statistics": {
                "avg_age": float(df['age'].mean()),
                "readmission_rate": float(df['readmission_risk'].mean()),
                "record_count": len(df)
            }
        }

        metadata_path = f"{self.export_dir}/{base_filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        files_created['metadata'] = metadata_path

        print(f"✅ Exported {len(df):,} records to:")
        for file_type, path in files_created.items():
            print(f"   📁 {file_type}: {os.path.basename(path)}")

        return files_created

    def get_export_stats(self):
        """Get statistics about exports"""
        if not os.path.exists(self.export_dir):
            return {"total_exports": 0}

        parquet_files = [f for f in os.listdir(self.export_dir) if f.endswith('.parquet')]
        return {
            "total_exports": len(parquet_files),
            "latest_export": max(parquet_files) if parquet_files else None,
            "export_directory": self.export_dir
        }