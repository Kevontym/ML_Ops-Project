# scripts/run_pipeline.py
# !/usr/bin/env python3
"""
Simple Drive Sync Pipeline Runner
"""

from data_exporter import DataExporter
from drive_uploader import DriveUploader
import mlflow
import os
from datetime import datetime


def run_drive_sync_pipeline(sample_size=None):
    """Run the complete Drive Sync pipeline"""

    # Setup MLflow tracking
    mlflow.set_tracking_uri("file:///" + os.path.abspath("mlruns"))
    mlflow.set_experiment("drive_sync_pipeline")

    with mlflow.start_run(run_name=f"drive_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        print("🚀 STARTING DRIVE SYNC PIPELINE")
        print("=" * 50)

        # Step 1: Export data
        print("\n1️⃣  STEP 1: Exporting Data...")
        exporter = DataExporter()
        exported_files = exporter.export_latest_snapshot(sample_size=sample_size)

        # Log to MLflow
        mlflow.log_param("pipeline_type", "drive_sync")
        mlflow.log_param("sample_size", sample_size or "full")
        mlflow.log_metric("records_exported", exported_files.get('record_count', 0))

        # Step 2: Prepare for upload
        print("\n2️⃣  STEP 2: Preparing for Drive Upload...")
        uploader = DriveUploader()
        upload_file = uploader.prepare_for_upload()

        if upload_file:
            mlflow.log_artifact(upload_file)
            mlflow.log_artifact(upload_file.replace('.parquet', '_metadata.json'))

        # Step 3: Show available exports
        print("\n3️⃣  STEP 3: Export Summary")
        stats = exporter.get_export_stats()
        print(f"   Total exports: {stats['total_exports']}")
        if stats['latest_export']:
            print(f"   Latest export: {stats['latest_export']}")

        print("\n✅ PIPELINE COMPLETE!")
        print("💡 Next: Upload files to Google Drive and open Colab notebook")

        return exported_files


if __name__ == "__main__":
    # For testing, use small sample. Remove sample_size for full dataset.
    run_drive_sync_pipeline(sample_size=1000)  # Start with 1000 records for testing