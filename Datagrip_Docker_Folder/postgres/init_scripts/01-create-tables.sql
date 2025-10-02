-- Initialize your MLOps healthcare database
CREATE TABLE IF NOT EXISTS patient_records (
    id SERIAL PRIMARY KEY,
    patient_id BIGINT UNIQUE NOT NULL,
    age INTEGER CHECK (age BETWEEN 0 AND 120),
    gender VARCHAR(1) CHECK (gender IN ('M', 'F', 'O')),
    blood_pressure VARCHAR(10),
    cholesterol INTEGER CHECK (cholesterol BETWEEN 100 AND 400),
    glucose_level INTEGER CHECK (glucose_level BETWEEN 50 AND 300),
    medications JSONB,
    doctor_notes TEXT,
    lab_results JSONB,
    previous_admissions INTEGER DEFAULT 0,
    readmission_risk BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_patient_id ON patient_records(patient_id);
CREATE INDEX IF NOT EXISTS idx_readmission_risk ON patient_records(readmission_risk);
CREATE INDEX IF NOT EXISTS idx_age ON patient_records(age);
CREATE INDEX IF NOT EXISTS idx_created_at ON patient_records(created_at);

-- Create a function to get patient statistics
CREATE OR REPLACE FUNCTION get_patient_stats()
RETURNS TABLE(
    total_patients BIGINT,
    avg_age NUMERIC,
    readmission_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) as total_patients,
        ROUND(AVG(age), 2) as avg_age,
        ROUND(100.0 * SUM(CASE WHEN readmission_risk THEN 1 ELSE 0 END) / COUNT(*), 2) as readmission_rate
    FROM patient_records;
END;
$$ LANGUAGE plpgsql;

-- Insert sample data for testing
INSERT INTO patient_records (patient_id, age, gender, blood_pressure, cholesterol, glucose_level, medications, doctor_notes, lab_results, previous_admissions, readmission_risk)
VALUES
(1, 45, 'M', '120/80', 200, 95,
 '[{"name": "Lisinopril", "dosage": "10mg", "frequency": "daily"}]',
 'Patient presents with stable vitals. No concerning symptoms noted.',
 '{"blood_work": {"wbc": 7.5, "rbc": 5.0, "hemoglobin": 15.0}}',
 0, false),
(2, 67, 'F', '140/90', 250, 145,
 '[{"name": "Metformin", "dosage": "500mg", "frequency": "bid"}, {"name": "Lipitor", "dosage": "20mg", "frequency": "daily"}]',
 'Patient with history of diabetes and hypertension. Managing well with current medications.',
 '{"blood_work": {"wbc": 6.8, "rbc": 4.8, "hemoglobin": 14.2}, "urinalysis": {"ph": 6.5, "protein": "trace"}}',
 2, true)
ON CONFLICT (patient_id) DO NOTHING;

-- Print confirmation
DO $$
BEGIN
    RAISE NOTICE 'MLOps Healthcare database initialized successfully!';
END $$;