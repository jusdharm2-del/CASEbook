# ML Use Case Recommender

Beautiful web frontend + FastAPI backend powered by **AWS Bedrock** (Claude Sonnet).

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML/CSS/JS (no framework, zero build step) |
| Backend | FastAPI + Uvicorn |
| AI | AWS Bedrock — `anthropic.claude-sonnet-4-6` |
| Profiling | pandas + numpy (pure local, no AI cost) |

## User flow (5 steps)

1. **Upload Dataset** — Drop any CSV; columns are auto-detected
2. **AI Analysis** — Bedrock detects the domain, profiles the data, and surfaces a data quality report (missing values, class imbalance, skewness, leakage risk, cardinality)
3. **Select Use Case** — Choose from 4–6 ranked ML use case recommendations (scored by business importance, data readiness, and ease of implementation)
4. **Starter Code** — Bedrock generates a complete, runnable Python script; download as `.py` or export as a structured `.ipynb` notebook
5. **Post-Modelling Insights** — Bedrock generates:
   - **Next steps checklist** — phase-tagged actions (Validation, Deployment, Monitoring, Retraining) with priority levels, specific to your use case and columns
   - **Output profiling report** — expected cluster/class/segment/anomaly profiles described in terms of your actual dataset features
   - **Alternative model options** — comparison table (complexity, speed, interpretability, expected accuracy) plus expandable pro/con cards for 3 alternative algorithms

## Setup

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configure AWS credentials

The app uses boto3's standard credential chain. Any of these work:

**Option A — Environment variables (recommended for dev)**
```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-west-2"
```

**Option B — AWS CLI profile**
```bash
aws configure  # sets ~/.aws/credentials
```

**Option C — IAM Role** (EC2 / ECS / Lambda — no extra config needed)

### 3. Enable Bedrock model access

In the AWS Console → Bedrock → Model access → request access to:
- `Anthropic Claude Sonnet` (latest available version)

### 4. (Optional) Override the model
```bash
export BEDROCK_MODEL_ID="anthropic.claude-3-5-haiku-20241022-v1:0"
```

### 5. Run
```bash
uvicorn server:app --reload --port 8000
```

Open **http://localhost:8000**

## Development without AWS credentials

Set `MOCK_MODE=true` to run entirely offline — no Bedrock calls are made. All AI responses are generated locally using heuristics and templates derived from the actual uploaded dataset.

```bash
MOCK_MODE=true uvicorn server:app --reload --port 8000
```

The mock covers all 5 stages including the new Insights page, with type-aware responses for Classification, Regression, Clustering, and Anomaly Detection use cases.

## Project structure

```
project/
├── server.py               ← FastAPI app + all route handlers
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_profiler.py    ← Local CSV parsing, stats, and quality report (no AWS)
│   ├── ai_analyst.py       ← AWS Bedrock calls (analyse, generate_code, insights)
│   └── mock_analyst.py     ← Offline mock for all AI functions (MOCK_MODE=true)
└── static/
    └── index.html          ← Full SPA — HTML + CSS + JS, zero build step
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the frontend |
| `POST` | `/api/profile` | Parses CSV, returns statistical profile |
| `POST` | `/api/analyse` | Bedrock: domain detection + ranked use cases |
| `POST` | `/api/generate_code` | Bedrock: complete Python starter script |
| `POST` | `/api/insights` | Bedrock: next steps, output profiling, alt models |
| `POST` | `/api/quality_report` | Local: data quality warnings and score (A–D) |
| `POST` | `/api/export_notebook` | Local: converts generated code to `.ipynb` |
| `GET` | `/api/health` | Health check, mode (mock/bedrock), region, model |

## IAM permissions required

```json
{
  "Effect": "Allow",
  "Action": ["bedrock:InvokeModel"],
  "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-6"
}
```
