# AegisFlow

## AI-Based Network Attack Forecasting from Network Traffic Data

AegisFlow is a temporal network-security analytics platform designed to analyze network traffic, identify security risks, track attack activity, and maintain a tamper-evident security audit trail.

Built for Smart India Hackathon 2026

- Problem Statement: SIH26153
- Theme: Blockchain & Cybersecurity
- Category: Software
- Team: CyberVanguard
- Team ID: 40


## Overview

Traditional network intrusion detection systems primarily focus on identifying suspicious activity after it has already occurred.

AegisFlow explores a temporal approach to network security by analyzing network-flow behaviour across time windows and using machine-learning models to estimate future security risk.

The platform combines:

- Temporal network-flow analysis
- Machine-learning-based risk prediction
- Host-level security monitoring
- Attack-stage mapping
- MITRE ATT&CK integration
- Network traffic replay
- Tamper-evident hash-chain auditing
- Security monitoring dashboard


## Key Features

### Network Traffic Analysis

Processes network-flow data and converts raw traffic into structured features suitable for temporal analysis and machine learning.

### Temporal Analysis

Network activity is organized into time-based windows and sequences to capture changes in traffic behaviour over time.

### AI-Based Risk Prediction

Machine-learning models analyze temporal network behaviour and generate security-risk predictions for monitored hosts.

### Host Risk Monitoring

The system aggregates network activity at the host level and provides risk information for security monitoring.

### Attack-Stage Mapping

Security activity can be associated with stages such as:

Reconnaissance
        |
        v
Initial Access
        |
        v
Lateral Movement
        |
        v
Command & Control
        |
        v
Impact

### MITRE ATT&CK Integration

AegisFlow maps supported attack stages to relevant MITRE ATT&CK tactics and techniques using a configurable mapping.

### Network Traffic Replay

Recorded network activity can be replayed through the analysis pipeline to demonstrate how the system processes traffic over time.

### Tamper-Evident Audit Ledger

Security events are stored using a SHA-256 hash chain where each event references the previous event's hash.

This provides a verifiable sequence of audit records.

### Security Dashboard

The dashboard provides a centralized view of:

- Host risk
- Security events
- Temporal activity
- Attack stages
- MITRE mappings
- Audit records
- Model evaluation


## System Architecture

Network Traffic
       |
       v
Data Ingestion & Validation
       |
       v
Feature Engineering
       |
       v
Temporal Windowing
       |
       v
ML Risk Prediction
       |
       +------------------+
       |                  |
       v                  v
   Host Risk        Attack Stage
       |                  |
       +--------+---------+
                |
                v
         MITRE ATT&CK
                |
                v
       Security Dashboard
                |
                v
       Hash-Chain Ledger


## Technology Stack

### Backend

- Python
- FastAPI
- SQLite

### Machine Learning

- PyTorch
- Scikit-learn
- Pandas
- NumPy

### Network Analysis

- CIC-IDS2017
- Network-flow processing
- Temporal traffic analysis

### Security

- MITRE ATT&CK
- SHA-256
- Hash-chain audit logging

### Frontend

- HTML
- CSS
- Vanilla JavaScript
- Chart.js

### Development & Testing

- Git
- Pytest
- YAML configuration


## Dataset

AegisFlow uses the CIC-IDS2017 network intrusion dataset.

The dataset is not included in this repository.

After obtaining the dataset, place the required CSV files under:

data/raw/cic_ids2017/

The project pipeline performs:

Raw Traffic
    |
    v
Validation
    |
    v
Cleaning
    |
    v
Feature Engineering
    |
    v
Temporal Windowing
    |
    v
Sequence Generation
    |
    v
Model Training / Replay


## Project Structure

Aegisflow-demo/
|
+-- aegisflow/
|   +-- ml/
|   +-- temporal/
|   +-- preprocessing/
|   +-- ingestion/
|   +-- schema.py
|
+-- backend/
|   +-- app/
|       +-- main.py
|       +-- replay.py
|       +-- audit.py
|       +-- static/
|           +-- index.html
|
+-- configs/
|   +-- config.yaml
|   +-- datasets.yaml
|   +-- stages.yaml
|   +-- mitre_mapping.yaml
|
+-- data/
+-- docs/
+-- reports/
+-- scripts/
+-- tests/
|
+-- pyproject.toml
+-- requirements.txt
+-- README.md


# Installation

## 1. Clone the repository

git clone https://github.com/Shirshak14/Aegisflow-demo.git
cd Aegisflow-demo

### Windows users

If Windows reports a path-length error while installing dependencies, clone the repository to a shorter path, for example:

C:\Aegisflow


## 2. Create a virtual environment

### Windows

python -m venv .venv

Activate it:

.venv\Scripts\activate

### Linux / macOS

python3 -m venv .venv
source .venv/bin/activate


## 3. Install dependencies

pip install -r requirements.txt

The project uses several machine-learning and scientific-computing packages, so the initial installation may take several minutes.


# Dataset Setup

Place the CIC-IDS2017 CSV files inside:

data/raw/cic_ids2017/

Then validate the dataset:

python -m aegisflow validate-dataset --dataset cic_ids2017


# Preprocessing

Run the preprocessing pipeline:

python -m aegisflow preprocess --dataset cic_ids2017

The pipeline generates the required processed network-flow and temporal data used by the machine-learning pipeline.

Processing time depends on the size of the dataset and the system being used.


# Model Training

After preprocessing:

python -m aegisflow train --dataset cic_ids2017

For a reproducible baseline configuration:

python -m aegisflow train --dataset cic_ids2017 --epochs 8 --batch-size 256 --learning-rate 0.001 --hidden-size 16 --dropout 0.2 --patience 3

Generated model artifacts are stored locally and are intentionally excluded from the repository.


# Running the Dashboard

Start the FastAPI application:

uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

Then open:

http://127.0.0.1:8000

The dashboard provides access to the project's monitoring and replay interface.


# Replay

AegisFlow supports replaying processed network activity through the analysis pipeline.

The replay workflow is:

Network Activity
       |
       v
Temporal Analysis
       |
       v
Risk Prediction
       |
       v
Attack Stage
       |
       v
MITRE Mapping
       |
       v
Security Event
       |
       v
Audit Ledger

Replay uses recorded project data to demonstrate the behaviour of the analysis pipeline over time.


# Audit Ledger

AegisFlow maintains a tamper-evident security event ledger using chained SHA-256 hashes.

Each event references the hash of the previous event.

Event 1
   |
   v
Hash 1
   |
   v
Event 2 + Hash 1
   |
   v
Hash 2
   |
   v
Event 3 + Hash 2
   |
   v
Hash 3

If an earlier record is modified, the subsequent chain relationships can be detected during verification.

The ledger is a tamper-evident hash chain, not a decentralized blockchain network.


# MITRE ATT&CK

AegisFlow uses a configurable MITRE ATT&CK mapping for supported attack stages.

The mapping is maintained in:

configs/mitre_mapping.yaml

This allows attack-stage information to be associated with relevant security tactics and techniques without hard-coding the mapping into the application.


# Model Evaluation

The project includes evaluation of machine-learning baselines using chronological network-traffic data.

Metrics include:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- False-positive rate
- Confusion matrix

Evaluation artifacts are generated during the training and evaluation process.

The current dataset and evaluation setup have known limitations, including class imbalance, temporal attack distribution, and limited generalization to attack classes that may not be represented in the training period.

The project reports model performance together with its evaluation context rather than presenting a single metric as proof of production-level detection capability.

Detailed evaluation information is available in the project reports.


# Testing

Run the test suite:

pytest

The tests cover areas including:

- Dataset processing
- Data validation
- Temporal sequence generation
- Temporal integrity
- Model pipeline
- API functionality
- Audit ledger behaviour


# Security & Data Considerations

The repository intentionally excludes:

- Raw datasets
- Processed datasets
- Local databases
- Virtual environments
- Trained model artifacts
- Temporary test directories
- Archived experiment data

These files can be generated locally using the project pipeline.


# Project Status

- Network-flow ingestion: Complete
- Dataset validation: Complete
- Feature engineering: Complete
- Temporal windowing: Complete
- Sequence generation: Complete
- ML baseline models: Complete
- Risk analysis: Complete
- MITRE ATT&CK mapping: Complete
- Network replay: Complete
- Hash-chain audit ledger: Complete
- Security dashboard: Complete
- Automated testing: Complete
- Advanced forecasting research: In progress


# Team

## CyberVanguard

Smart India Hackathon 2026

- Problem Statement: SIH26153
- Theme: Blockchain & Cybersecurity
- Category: Software
- Team ID: 40


# License

This project is intended for academic and research purposes.

See the repository license file for applicable licensing information.
