# AegisFlow

### AI-Based Network Attack Forecasting from Network Traffic Data

AegisFlow is a cybersecurity analytics platform designed to analyze network traffic over time, identify emerging security risks, and provide a forward-looking view of potential attack activity.

Built for **Smart India Hackathon 2026 — SIH26153**  
**Theme:** Blockchain & Cybersecurity  
**Team:** CyberVanguard · **Team ID:** 40

---

## Overview

Traditional intrusion detection systems primarily focus on identifying threats after suspicious activity has already occurred.

AegisFlow explores a temporal approach to network security by analyzing sequences of network-flow behaviour and generating risk predictions for future network states.

The platform combines:

- Temporal network-flow analysis
- Machine learning-based risk prediction
- Attack-stage mapping
- MITRE ATT&CK integration
- Host-level risk monitoring
- Replay-based security analysis
- Tamper-evident security event logging

---

## Key Features

### 🔐 Network Traffic Analysis
Processes network-flow data and converts raw traffic into structured temporal representations suitable for machine-learning analysis.

### 🧠 AI-Based Risk Prediction
Uses machine-learning models to analyze historical traffic sequences and estimate future attack risk.

### 📈 Temporal Analysis
Network activity is organized into time-based windows and sequences, allowing changes in behaviour to be analyzed over time.

### 🎯 Attack Stage Mapping
Security events can be associated with attack stages such as:

`Reconnaissance → Initial Access → Lateral Movement → Command & Control → Impact`

### 🛡️ MITRE ATT&CK Integration
Predicted security stages can be mapped to relevant MITRE ATT&CK tactics and techniques.

### 🔗 Tamper-Evident Audit Ledger
Security events are recorded using a SHA-256 hash chain, creating a verifiable sequence of audit records.

### 🔄 Traffic Replay
Recorded network traffic can be replayed through the analysis pipeline to demonstrate the system's behaviour over time.

### 📊 Security Dashboard
A lightweight web dashboard provides:

- Host risk overview
- Security alerts
- Temporal activity
- Attack-stage information
- MITRE mappings
- Audit events
- Model evaluation information

---

## System Architecture

```text
Network Traffic
       │
       ▼
┌─────────────────────┐
│ Data Ingestion      │
│ & Validation        │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Feature Engineering │
│ & Temporal Windows  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Temporal ML Model   │
│ Risk Prediction     │
└─────────┬───────────┘
          │
          ├──────────────► Host Risk
          │
          ├──────────────► Attack Stage
          │
          ├──────────────► MITRE ATT&CK
          │
          ▼
┌─────────────────────┐
│ Security Dashboard  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Hash-Chain Audit     │
│ Ledger               │
└─────────────────────┘
```

---

## Technology Stack

**Backend**
- Python
- FastAPI

**Machine Learning**
- PyTorch
- Scikit-learn
- Pandas
- NumPy

**Network Data**
- CIC-IDS2017
- Network flow analysis
- PCAP/flow-oriented processing

**Security**
- MITRE ATT&CK
- SHA-256 hash chaining
- Temporal attack analysis

**Frontend**
- HTML
- CSS
- JavaScript
- Chart.js

**Development**
- Git
- Pytest
- YAML configuration
- SQLite

---

## Dataset

The project currently uses the **CIC-IDS2017** network intrusion dataset.

The pipeline supports:

- Data validation
- Cleaning
- Feature normalization
- Temporal window generation
- Sequence construction
- Chronological evaluation

Dataset files are intentionally excluded from the repository.

---

## Project Structure

```text
AegisFlow/
│
├── aegisflow/
│   ├── ml/
│   ├── temporal/
│   ├── preprocessing/
│   ├── ingestion/
│   └── schema.py
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── replay.py
│       └── audit.py
│
├── configs/
│   ├── config.yaml
│   ├── datasets.yaml
│   ├── stages.yaml
│   └── mitre_mapping.yaml
│
├── data/
├── models/
├── reports/
├── scripts/
├── tests/
│
├── docs/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Running the Project

### 1. Clone

```bash
git clone https://github.com/Shirshak14/Aegisflow-demo.git
cd Aegisflow-demo
```

### 2. Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the required CIC-IDS2017 files under:

```text
data/raw/cic_ids2017/
```

Then validate:

```bash
python -m aegisflow validate-dataset --dataset cic_ids2017
```

### 5. Run preprocessing

```bash
python -m aegisflow preprocess --dataset cic_ids2017
```

### 6. Train the baseline models

```bash
python -m aegisflow train --dataset cic_ids2017
```

### 7. Start the dashboard

```bash
uvicorn backend.app.main:app --port 8000
```

Open:

```text
http://localhost:8000
```

---

## Security Audit Ledger

AegisFlow maintains a tamper-evident audit trail using chained SHA-256 hashes.

Each event references the hash of the previous event:

```text
Event 1
   │
   ▼
Hash 1
   │
   ▼
Event 2 + Hash 1
   │
   ▼
Hash 2
   │
   ▼
Event 3 + Hash 2
   │
   ▼
Hash 3
```

This allows the integrity of the recorded event sequence to be verified.

---

## Testing

Run the test suite with:

```bash
pytest
```

The project includes tests covering:

- Dataset processing
- Temporal sequence generation
- Data validation
- Model pipeline
- API behaviour
- Audit ledger integrity

---

## Design Goals

AegisFlow is built around four core principles:

**Temporal**  
Analyze network behaviour as a sequence rather than isolated events.

**Predictive**  
Explore forward-looking risk estimation rather than relying exclusively on reactive detection.

**Explainable**  
Connect security predictions with recognizable attack stages and MITRE ATT&CK concepts.

**Auditable**  
Maintain an integrity-verifiable record of security events.

---

## Project Status

| Component | Status |
|---|---|
| Network-flow ingestion | ✅ |
| Data validation & preprocessing | ✅ |
| Temporal windowing | ✅ |
| Sequence generation | ✅ |
| ML baseline models | ✅ |
| Risk analysis | ✅ |
| MITRE ATT&CK mapping | ✅ |
| Traffic replay | ✅ |
| Hash-chain audit ledger | ✅ |
| Security dashboard | ✅ |
| Advanced forecasting models | 🚧 |

---

## Team

### CyberVanguard

**Smart India Hackathon 2026**  
Problem Statement: **SIH26153**  
Theme: **Blockchain & Cybersecurity**  
Team ID: **40**

---

## License

This project is released under the MIT License.
