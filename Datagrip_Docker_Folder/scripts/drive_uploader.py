# scripts/drive_uploader.py
import os
from datetime import datetime
import webbrowser


class DriveUploader:
    def __init__(self):
        self.upload_dir = "data/exports"
        os.makedirs(self.upload_dir, exist_ok=True)

    def prepare_for_upload(self):
        """Prepare files and provide upload instructions"""
        print("📤 Preparing files for Google Drive upload...")

        # Find the latest export
        parquet_files = [f for f in os.listdir(self.upload_dir) if f.endswith('.parquet')]
        if not parquet_files:
            print("❌ No export files found. Run data exporter first.")
            return None

        latest_file = max(parquet_files)
        file_path = os.path.join(self.upload_dir, latest_file)

        # Get file info
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        print(f"📁 Latest export: {latest_file}")
        print(f"📊 File size: {file_size_mb:.2f} MB")
        print(f"📍 Location: {file_path}")

        # Create upload instructions
        instructions = self._create_upload_instructions(file_path, latest_file)

        return file_path

    def _create_upload_instructions(self, file_path, filename):
        """Create step-by-step upload instructions"""
        print("\n🎯 UPLOAD INSTRUCTIONS:")
        print("=" * 50)
        print("1. Go to https://drive.google.com")
        print("2. Create a folder called 'mlops_data'")
        print("3. Upload these files from your local machine:")
        print(f"   - {file_path}")
        print(f"   - {file_path.replace('.parquet', '.csv')}")
        print(f"   - {file_path.replace('.parquet', '_metadata.json')}")
        print("\4. In Colab, use:")
        print(f"   df = pd.read_parquet('/content/drive/MyDrive/mlops_data/{filename}')")
        print("=" * 50)

        # Try to open the directory
        try:
            os.startfile(self.upload_dir)  # Windows
        except:
            try:
                os.system(f'open "{self.upload_dir}"')  # macOS
            except:
                try:
                    os.system(f'xdg-open "{self.upload_dir}"')  # Linux
                except:
                    print(f"💡 Manually navigate to: {os.path.abspath(self.upload_dir)}")

    def list_available_exports(self):
        """List all available exports"""
        if not os.path.exists(self.upload_dir):
            print("❌ No exports directory found")
            return

        files = os.listdir(self.upload_dir)
        if not files:
            print("❌ No export files found")
            return

        print("📁 Available exports:")
        for file in sorted(files):
            file_path = os.path.join(self.upload_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   {file} ({size_mb:.1f} MB)")