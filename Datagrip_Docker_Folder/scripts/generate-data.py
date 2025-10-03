#!/usr/bin/env python3
"""
MASSIVE Healthcare Data Generator for Deep Learning - 20K Records
Enhanced with more variability and realistic distributions
"""

import pandas as pd
import numpy as np
import json
from faker import Faker
import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fake = Faker()

# Configuration

DB_USER = os.getenv('POSTGRES_USER', 'healthcare_user')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'healthcare_pass123')
DB_NAME = os.getenv('POSTGRES_DB', 'healthcare_mlops')
DB_PORT = os.getenv('POSTGRES_PORT', '5433')


class MassiveHealthcareDataGenerator:
    def __init__(self, db_connection=None):
        self.fake = Faker()
        self.db_connection = db_connection or f"postgresql://{DB_USER}:{DB_PASSWORD}@localhost:{DB_PORT}/{DB_NAME}"
        self.engine = create_engine(self.db_connection)

        # Expanded medical conditions with prevalence weights
        self.conditions_with_weights = [
            ("hypertension", 0.35), ("diabetes", 0.15), ("asthma", 0.08),
            ("arthritis", 0.12), ("migraine", 0.06), ("copd", 0.05),
            ("heart_disease", 0.08), ("obesity", 0.25), ("depression", 0.18),
            ("anxiety", 0.15), ("hyperlipidemia", 0.20), ("osteoporosis", 0.07),
            ("chronic_kidney", 0.04), ("thyroid", 0.09), ("anemia", 0.06)
        ]

        # Expanded medications with conditions they treat
        self.medications = [
            {"name": "Lisinopril", "type": "blood_pressure", "conditions": ["hypertension", "heart_disease"]},
            {"name": "Metformin", "type": "diabetes", "conditions": ["diabetes"]},
            {"name": "Atorvastatin", "type": "cholesterol", "conditions": ["hyperlipidemia", "heart_disease"]},
            {"name": "Albuterol", "type": "asthma", "conditions": ["asthma", "copd"]},
            {"name": "Sertraline", "type": "depression", "conditions": ["depression", "anxiety"]},
            {"name": "Levothyroxine", "type": "thyroid", "conditions": ["thyroid"]},
            {"name": "Amlodipine", "type": "blood_pressure", "conditions": ["hypertension"]},
            {"name": "Omeprazole", "type": "gerd", "conditions": []},
            {"name": "Losartan", "type": "blood_pressure", "conditions": ["hypertension", "chronic_kidney"]},
            {"name": "Gabapentin", "type": "pain", "conditions": ["arthritis", "migraine"]}
        ]

    def test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                print(f"✅ Database connected successfully!")
                return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

    def get_conditions_by_age(self, age):
        """Get conditions based on age-appropriate prevalence"""
        base_conditions = []

        # Age-based condition probabilities
        if age < 30:
            weights = [0.02, 0.01, 0.08, 0.01, 0.10, 0.00, 0.00, 0.15, 0.12, 0.15, 0.01, 0.00, 0.00, 0.05, 0.04]
        elif age < 50:
            weights = [0.15, 0.08, 0.06, 0.08, 0.08, 0.02, 0.03, 0.25, 0.15, 0.18, 0.12, 0.02, 0.01, 0.08, 0.04]
        else:
            weights = [0.45, 0.25, 0.05, 0.20, 0.04, 0.10, 0.15, 0.30, 0.12, 0.10, 0.35, 0.15, 0.08, 0.12, 0.08]

        # Select conditions based on weighted probabilities
        for (condition, base_weight), weight in zip(self.conditions_with_weights, weights):
            if random.random() < weight:
                base_conditions.append(condition)

        return base_conditions

    def generate_patient_medications(self, age, conditions):
        """Generate realistic medications based on conditions"""
        medications = []

        # Always include some medications for conditions
        for condition in conditions:
            # Find medications that treat this condition
            suitable_meds = [med for med in self.medications if condition in med.get("conditions", [])]
            if suitable_meds and random.random() < 0.7:  # 70% chance of prescribing
                med = random.choice(suitable_meds)
                medications.append({
                    "name": med["name"],
                    "dosage": f"{random.choice([5, 10, 20, 40, 80])}mg",
                    "frequency": random.choice(["daily", "bid", "tid"]),
                    "condition": condition
                })

        # Add some random medications (polypharmacy effect)
        if age > 65 and random.random() < 0.4:
            extra_med = random.choice(self.medications)
            if extra_med["name"] not in [m["name"] for m in medications]:
                medications.append({
                    "name": extra_med["name"],
                    "dosage": f"{random.choice([5, 10, 20])}mg",
                    "frequency": "daily",
                    "condition": "general"
                })

        return medications if medications else [{
            "name": "Multivitamin",
            "dosage": "1 tablet",
            "frequency": "daily",
            "condition": "general"
        }]

    def generate_comprehensive_lab_results(self, age, conditions):
        """Generate comprehensive lab results with realistic patterns"""
        labs = {
            "blood_work": {
                "wbc": round(np.random.normal(7.0, 1.5), 1),
                "rbc": round(np.random.normal(4.8, 0.5), 1),
                "hemoglobin": round(np.random.normal(14.0, 1.5), 1),
                "hematocrit": round(np.random.normal(42.0, 3.0), 1),
                "platelets": int(np.random.normal(250, 50))
            },
            "metabolic": {
                "sodium": int(np.random.normal(140, 3)),
                "potassium": round(np.random.normal(4.0, 0.3), 1),
                "chloride": int(np.random.normal(102, 2)),
                "bicarbonate": int(np.random.normal(26, 2))
            }
        }

        # Condition-specific abnormalities
        if "diabetes" in conditions:
            labs["metabolic"]["glucose"] = random.randint(126, 300)
            labs["metabolic"]["a1c"] = round(np.random.normal(7.5, 1.0), 1)
        else:
            labs["metabolic"]["glucose"] = random.randint(70, 125)

        if "hyperlipidemia" in conditions or "heart_disease" in conditions:
            labs["lipid_panel"] = {
                "ldl": random.randint(130, 250),
                "hdl": random.randint(25, 55),
                "triglycerides": random.randint(150, 400)
            }
        else:
            labs["lipid_panel"] = {
                "ldl": random.randint(70, 129),
                "hdl": random.randint(40, 80),
                "triglycerides": random.randint(50, 149)
            }

        # Age-related changes
        if age > 65:
            labs["blood_work"]["hemoglobin"] = max(11.0, labs["blood_work"]["hemoglobin"] - 1.0)
            labs["renal"] = {"creatinine": round(np.random.normal(1.2, 0.3), 1)}
        else:
            labs["renal"] = {"creatinine": round(np.random.normal(0.9, 0.2), 1)}

        return labs

    def generate_detailed_doctor_notes(self, conditions, age, medications):
        """Generate detailed, realistic doctor notes"""
        condition_desc = ", ".join(conditions) if conditions else "generally good health"

        templates = [
            f"Patient presents for routine follow-up. History of {condition_desc}. " +
            f"Currently taking {len(medications)} medications. Vital signs within acceptable limits.",

            f"Follow-up visit for chronic conditions including {condition_desc}. " +
            f"Medication regimen reviewed and appears effective. No acute concerns today.",

            f"Comprehensive visit for {random.choice(['annual physical', 'medication management', 'chronic care'])}. " +
            f"Patient with history of {condition_desc}. Managing well on current therapy."
        ]

        return random.choice(templates)

    def calculate_risk_score(self, age, conditions, medications, previous_admissions):
        """Calculate detailed risk score (0-100) for stratification"""
        risk_score = 0

        # Age component (0-25 points)
        if age >= 80:
            risk_score += 25
        elif age >= 70:
            risk_score += 20
        elif age >= 60:
            risk_score += 15
        elif age >= 50:
            risk_score += 10
        elif age >= 40:
            risk_score += 5

        # Condition complexity (0-35 points)
        high_risk_conditions = ["heart_disease", "copd", "chronic_kidney", "diabetes"]
        medium_risk_conditions = ["hypertension", "hyperlipidemia", "obesity"]

        for condition in conditions:
            if condition in high_risk_conditions:
                risk_score += 8
            elif condition in medium_risk_conditions:
                risk_score += 5
            else:
                risk_score += 3

        # Medication burden (0-20 points)
        if len(medications) >= 5:
            risk_score += 20
        elif len(medications) >= 3:
            risk_score += 15
        elif len(medications) >= 1:
            risk_score += 5

        # Previous admissions (0-20 points)
        risk_score += min(previous_admissions * 5, 20)

        return min(risk_score, 100)

    def generate_patient_record(self, patient_id):
        """Generate one complete patient record with enhanced features"""
        # Age distribution weighted toward older patients (more healthcare utilization)
        age = int(np.random.choice(
            [18, 25, 35, 45, 55, 65, 75, 85],
            p=[0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05]
        )) + random.randint(0, 9)

        gender = random.choice(['M', 'F'])

        # Generate conditions based on age
        conditions = self.get_conditions_by_age(age)

        # Generate dependent data
        medications = self.generate_patient_medications(age, conditions)
        lab_results = self.generate_comprehensive_lab_results(age, conditions)
        doctor_notes = self.generate_detailed_doctor_notes(conditions, age, medications)

        # Calculate vital signs with realistic distributions
        if "hypertension" in conditions:
            systolic = int(np.random.normal(145, 10))
            diastolic = int(np.random.normal(92, 8))
        else:
            systolic = int(np.random.normal(125, 15))
            diastolic = int(np.random.normal(78, 10))

        blood_pressure = f"{max(80, systolic)}/{max(50, diastolic)}"

        # Generate other metrics with condition influence
        if "diabetes" in conditions:
            glucose_level = int(np.random.normal(180, 40))
        else:
            glucose_level = int(np.random.normal(95, 15))

        if "hyperlipidemia" in conditions:
            cholesterol = int(np.random.normal(240, 30))
        else:
            cholesterol = int(np.random.normal(190, 25))

        # Previous admissions - more likely with age and conditions
        admission_base = len(conditions) * 0.3 + (age - 50) * 0.02
        previous_admissions = np.random.poisson(max(0, admission_base))

        # Calculate risk scores
        risk_score = self.calculate_risk_score(age, conditions, medications, previous_admissions)
        readmission_risk = risk_score > 50  # Binary risk flag

        return {
            'patient_id': f"PAT{patient_id:06d}",
            'age': age,
            'gender': gender,
            'blood_pressure': blood_pressure,
            'cholesterol': cholesterol,
            'glucose_level': glucose_level,
            'medications': json.dumps(medications),
            'doctor_notes': doctor_notes,
            'lab_results': json.dumps(lab_results),
            'previous_admissions': previous_admissions,
            'readmission_risk': readmission_risk,
            'risk_score': risk_score  # New field for stratification
        }

    def generate_large_dataset(self, num_patients=20000, batch_size=1000):
        """Generate large dataset optimized for performance"""
        if not self.test_connection():
            print("❌ Cannot generate data without database connection")
            return

        print(f"🚀 Generating {num_patients:,} patient records...")
        start_time = datetime.now()

        # Create table if not exists
        self.create_table()

        total_generated = 0

        for batch_num, batch_start in enumerate(range(0, num_patients, batch_size), 1):
            batch_end = min(batch_start + batch_size, num_patients)
            batch_size_actual = batch_end - batch_start

            print(f"📦 Processing batch {batch_num}: records {batch_start:,} to {batch_end:,}")

            records = []
            for i in range(batch_start, batch_end):
                records.append(self.generate_patient_record(i))

                # Progress indicator
                if (i - batch_start + 1) % 200 == 0:
                    percent = (i - batch_start + 1) / batch_size_actual * 100
                    print(f"   ... {percent:.0f}% complete", end='\r')

            # Convert to DataFrame and save
            df = pd.DataFrame(records)
            df.to_sql('patient_records', self.engine, if_exists='append', index=False, method='multi')

            total_generated += len(records)
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = total_generated / elapsed if elapsed > 0 else 0

            print(f"✅ Batch {batch_num} complete: {len(records):,} records")
            print(f"   📊 Total: {total_generated:,} | Rate: {rate:.1f} records/sec")

        total_time = (datetime.now() - start_time).total_seconds()
        print(f"\n🎉 Successfully generated {num_patients:,} patient records in {total_time:.1f} seconds!")
        print(f"📈 Average rate: {num_patients / total_time:.1f} records/second")

        # Print detailed statistics
        self.print_detailed_stats()

    def create_table(self):
        """Create the patient_records table if it doesn't exist"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS patient_records (
                        patient_id VARCHAR(20) PRIMARY KEY,
                        age INTEGER,
                        gender CHAR(1),
                        blood_pressure VARCHAR(20),
                        cholesterol INTEGER,
                        glucose_level INTEGER,
                        medications TEXT,
                        doctor_notes TEXT,
                        lab_results TEXT,
                        previous_admissions INTEGER,
                        readmission_risk BOOLEAN,
                        risk_score INTEGER
                    )
                """))
                conn.commit()
                print("✅ Table created/verified successfully")
        except Exception as e:
            print(f"❌ Error creating table: {e}")

    def print_detailed_stats(self):
        """Print comprehensive dataset statistics"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_patients,
                        AVG(age) as avg_age,
                        STDDEV(age) as age_stddev,
                        COUNT(CASE WHEN readmission_risk THEN 1 END) as high_risk_patients,
                        COUNT(CASE WHEN gender = 'M' THEN 1 END) as male_patients,
                        COUNT(CASE WHEN gender = 'F' THEN 1 END) as female_patients,
                        AVG(risk_score) as avg_risk_score,
                        MIN(risk_score) as min_risk,
                        MAX(risk_score) as max_risk,
                        AVG(previous_admissions) as avg_admissions
                    FROM patient_records
                """))
                stats = result.fetchone()

                print("\n📊 COMPREHENSIVE DATASET STATISTICS:")
                print("=" * 50)
                print(f"Total Patients: {stats[0]:,}")
                print(f"Average Age: {stats[1]:.1f} ± {stats[2]:.1f} years")
                print(f"Gender Distribution: {stats[4]:,} Male, {stats[5]:,} Female")
                print(f"High Risk Patients: {stats[3]:,} ({stats[3] / stats[0] * 100:.1f}%)")
                print(f"Risk Score: {stats[6]:.1f} avg ({stats[7]}-{stats[8]})")
                print(f"Average Previous Admissions: {stats[9]:.2f}")

                # Risk stratification
                risk_result = conn.execute(text("""
                    SELECT 
                        COUNT(CASE WHEN risk_score < 25 THEN 1 END) as low_risk,
                        COUNT(CASE WHEN risk_score >= 25 AND risk_score < 50 THEN 1 END) as medium_risk,
                        COUNT(CASE WHEN risk_score >= 50 AND risk_score < 75 THEN 1 END) as high_risk,
                        COUNT(CASE WHEN risk_score >= 75 THEN 1 END) as critical_risk
                    FROM patient_records
                """))
                risk_stats = risk_result.fetchone()

                print(f"\n🎯 RISK STRATIFICATION:")
                print(f"Low Risk (<25): {risk_stats[0]:,} ({risk_stats[0] / stats[0] * 100:.1f}%)")
                print(f"Medium Risk (25-49): {risk_stats[1]:,} ({risk_stats[1] / stats[0] * 100:.1f}%)")
                print(f"High Risk (50-74): {risk_stats[2]:,} ({risk_stats[2] / stats[0] * 100:.1f}%)")
                print(f"Critical Risk (75+): {risk_stats[3]:,} ({risk_stats[3] / stats[0] * 100:.1f}%)")

        except Exception as e:
            print(f"❌ Error getting statistics: {e}")


def main():
    """Main function to generate large dataset"""
    print("🏥 MASSIVE Healthcare Data Generator for Deep Learning")
    print("=" * 60)

    generator = MassiveHealthcareDataGenerator()

    # Generate 20,000 records for deep learning
    generator.generate_large_dataset(num_patients=20000, batch_size=1000)


if __name__ == "__main__":
    main()