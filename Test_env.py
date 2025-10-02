# test_env.py - Run this to debug
from dotenv import load_dotenv
import os

load_dotenv()

print("🔍 .env Debug:")
print(f"POSTGRES_USER: {os.getenv('POSTGRES_USER')}")
print(f"POSTGRES_PASSWORD: {os.getenv('POSTGRES_PASSWORD')}")
print(f"POSTGRES_DB: {os.getenv('POSTGRES_DB')}")
print(f"POSTGRES_PORT: {os.getenv('POSTGRES_PORT')}")

# Check file location
print(f"Current directory: {os.getcwd()}")
print(f".env exists: {os.path.exists('.env')}")

if os.path.exists('.env'):
    with open('.env', 'r') as f:
        print("File content:")
        print(f.read())