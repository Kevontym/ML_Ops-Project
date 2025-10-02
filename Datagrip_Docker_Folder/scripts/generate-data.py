# !/usr/bin/env python3
"""
Massive Healthcare Data Generator for MLOps Project
FIXED VERSION - Correct SQL execution
"""

import pandas as pd
import numpy as np
import json
from faker import Faker
import random
from datetime import datetime
from sqlalchemy import create_engine, text
import sys
import os


from dotenv import load_dotenv
load_dotenv()


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fake = Faker()


# Safe configuration - uses env vars with defaults
DB_USER = os.getenv('POSTGRES_USER', 'mlops_user')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'mlops_password')
DB_NAME = os.getenv('POSTGRES_DB', 'mlops_healthcare')
DB_PORT = os.getenv('POSTGRES_PORT', '5433')



class HealthcareDataGenerator:
    def __init__(self, db_connection=None):
        self.fake = Faker()

        # Use your actual credentials from docker-compose.yml
        self.db_connection = db_connection or f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:{DB_PORT}/{DB_NAME}"

        print(f"🔗 Attempting connection to: {self.db_connection}")

        self.engine = create_engine(self.db_connection)

        # Medical conditions for realistic data
        self.conditions = [
            "hypertension", "diabetes", "asthma", "arthritis", "migraine",
            "copd", "heart_disease", "obesity", "depression", "anxiety"
        ]

        self.medications = [
            {"name": "Lisinopril", "type": "blood_pressure", "common_dosage": "10mg"},
            {"name": "Metformin", "type": "diabetes", "common_dosage": "500mg"},
            {"name": "Atorvastatin", "type": "cholesterol", "common_dosage": "20mg"},
            {"name": "Albuterol", "type": "asthma", "common_dosage": "100mcg"},
            {"name": "Sertraline", "type": "depression", "common_dosage": "50mg"}
        ]

    def test_connection(self):
        """Test database connection - FIXED VERSION"""
        try:
            with self.engine.connect() as conn:
                # Use text() for raw SQL or just execute directly
                result = conn.execute(text("SELECT 1 as test"))
                print(f"✅ Database connected successfully! Test result: {result.fetchone()[0]}")
                return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

    def generate_patient_medications(self, age, conditions):
        """Generate realistic medications based on age and conditions"""
        medications = []
        num_meds = random.randint(1, 4)

        for _ in range(num_meds):
            med = random.choice(self.medications)
            medications.append({
                "name": med["name"],
                "dosage": med["common_dosage"],
                "frequency": random.choice(["daily", "bid", "tid"]),
                "start_date": fake.date_this_year().isoformat(),
                "condition": random.choice(conditions) if conditions else "general"
            })

        return medications

    def generate_lab_results(self, age, conditions):
        """Generate realistic lab results"""
        lab_results = {
            "blood_work": {
                "wbc": round(random.uniform(4.0, 11.0), 1),
                "rbc": round(random.uniform(4.0, 6.0), 1),
                "hemoglobin": round(random.uniform(12.0, 18.0), 1),
                "hematocrit": round(random.uniform(36.0, 50.0), 1),
                "platelets": random.randint(150, 450)
            }
        }

        # Add condition-specific abnormalities
        if "diabetes" in conditions:
            lab_results["blood_work"]["glucose"] = random.randint(126, 300)
            lab_results["blood_work"]["a1c"] = round(random.uniform(6.5, 10.0), 1)

        return lab_results

    def generate_doctor_notes(self, conditions, age, medications):
        """Generate realistic doctor notes"""
        condition_desc = ", ".join(conditions) if conditions else "general health"

        note = f"Patient presents for routine follow-up. History of {condition_desc}. " + \
               f"Currently taking {len(medications)} medications. Vital signs stable."

        return note

    def calculate_readmission_risk(self, age, conditions, medications, previous_admissions):
        """Calculate realistic readmission risk"""
        risk_score = 0

        # Age risk
        if age > 65:
            risk_score += 2
        elif age > 50:
            risk_score += 1

        # Condition risk
        high_risk_conditions = ["heart_disease", "copd", "diabetes"]
        for condition in conditions:
            if condition in high_risk_conditions:
                risk_score += 2
            else:
                risk_score += 1

        # Medication risk
        if len(medications) >= 3: risk_score += 1
        if len(medications) >= 5: risk_score += 2

        # Previous admissions risk
        risk_score += min(previous_admissions, 3)

        return risk_score >= 5

    def generate_patient_record(self, patient_id):
        """Generate one complete patient record"""
        age = random.randint(18, 95)
        gender = random.choice(['M', 'F'])

        # Generate conditions based on age
        num_conditions = random.randint(0, 3)
        if age > 60:
            num_conditions = random.randint(1, 4)

        conditions = random.sample(self.conditions, num_conditions) if num_conditions > 0 else []

        # Generate dependent data
        medications = self.generate_patient_medications(age, conditions)
        lab_results = self.generate_lab_results(age, conditions)
        doctor_notes = self.generate_doctor_notes(conditions, age, medications)

        # Calculate vital signs based on conditions
        if "hypertension" in conditions:
            systolic = random.randint(140, 180)
            diastolic = random.randint(90, 110)
        else:
            systolic = random.randint(100, 140)
            diastolic = random.randint(60, 90)

        blood_pressure = f"{systolic}/{diastolic}"

        # Generate other metrics
        if "diabetes" in conditions:
            glucose_level = random.randint(126, 300)
        else:
            glucose_level = random.randint(70, 125)

        cholesterol = random.randint(150, 280)
        previous_admissions = random.randint(0, 3)
        readmission_risk = self.calculate_readmission_risk(age, conditions, medications, previous_admissions)

        return {
            'patient_id': patient_id,
            'age': age,
            'gender': gender,
            'blood_pressure': blood_pressure,
            'cholesterol': cholesterol,
            'glucose_level': glucose_level,
            'medications': json.dumps(medications),
            'doctor_notes': doctor_notes,
            'lab_results': json.dumps(lab_results),
            'previous_admissions': previous_admissions,
            'readmission_risk': readmission_risk
        }

    def generate_dataset(self, num_patients=1000, batch_size=500):
        """Generate dataset with connection testing"""
        if not self.test_connection():
            print("❌ Cannot generate data without database connection")
            return

        print(f"🚀 Generating {num_patients} patient records...")

        for batch_start in range(0, num_patients, batch_size):
            batch_end = min(batch_start + batch_size, num_patients)
            batch_size_actual = batch_end - batch_start

            records = []
            for i in range(batch_start, batch_end):
                if i % 100 == 0:
                    print(f"📊 Generated {i}/{num_patients} records...")

                records.append(self.generate_patient_record(i))

            # Convert to DataFrame and save to database
            df = pd.DataFrame(records)
            df.to_sql('patient_records', self.engine, if_exists='append', index=False, method='multi')

            print(f"✅ Batch {batch_start // batch_size + 1} loaded: {batch_size_actual} records")

        print(f"🎉 Successfully generated {num_patients} patient records!")

        # Print dataset statistics
        self.print_dataset_stats()

    def print_dataset_stats(self):
        """Print statistics about the generated dataset"""
        try:
            with self.engine.connect() as conn:
                # Get basic stats
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_patients,
                        AVG(age) as avg_age,
                        COUNT(CASE WHEN readmission_risk THEN 1 END) as high_risk_patients,
                        COUNT(CASE WHEN gender = 'M' THEN 1 END) as male_patients,
                        COUNT(CASE WHEN gender = 'F' THEN 1 END) as female_patients
                    FROM patient_records
                """))
                stats = result.fetchone()

                print("\n📊 DATASET STATISTICS:")
                print("=" * 40)
                print(f"Total Patients: {stats[0]:,}")
                print(f"Average Age: {stats[1]:.1f}")
                print(f"High Risk Patients: {stats[2]:,} ({stats[2] / stats[0] * 100:.1f}%)")
                print(f"Male Patients: {stats[3]:,}")
                print(f"Female Patients: {stats[4]:,}")
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")


def main():
    """Main function to generate and load data"""
    print("🏥 MLOps Healthcare Data Generator")
    print("=" * 50)

    generator = HealthcareDataGenerator()

    # Generate dataset - start with 100 for testing
    generator.generate_dataset(num_patients=100)


if __name__ == "__main__":
    main()