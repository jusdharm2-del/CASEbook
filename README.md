# CASEbook — ML Use Case Recommender

Beautiful web frontend + FastAPI backend powered by **AWS Bedrock** (Claude Sonnet).

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML/CSS/JS (no framework, zero build step) |
| Backend | FastAPI + Uvicorn |
| AI | AWS Bedrock — `anthropic.claude-haiku-3` |
| Profiling | pandas + numpy (pure local, no AI cost) |

## User flow

1. **Upload Dataset** — Drop one or multiple CSVs; columns are auto-detected
2. **Join Recommendations** *(multi-file only)* — AI analyses schemas and recommends the best join strategy, including chain joins across all uploaded files
3. **AI Analysis** — Bedrock detects the domain, profiles the data, and surfaces a data quality report (missing values, class imbalance, skewness, leakage risk, cardinality)
4. **Business Context** — Answer 4 dataset-specific questions to sharpen the use case recommendations
5. **Select Use Case** — Choose from 4–6 ranked ML use case recommendations (scored by business importance, data readiness, and ease of implementation)
6. **Feature Engineering** — Review and select AI-generated feature transformation recommendations before code generation
7. **Starter Code** — Bedrock generates a complete, runnable Python script; download as `.py` or export as a structured `.ipynb` notebook
8. **Post-Modelling Insights** — Bedrock generates:
   - **Next steps checklist** — phase-tagged actions (Validation, Deployment, Monitoring, Retraining) with priority levels
   - **Output profiling report** — expected cluster/class/segment/anomaly profiles in terms of your actual dataset features
   - **Alternative model options** — comparison table (complexity, speed, interpretability, expected accuracy) plus pro/con cards for 3 alternative algorithms

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

Set `MOCK_MODE=true` to run entirely offline — no Bedrock calls are made. All AI responses are generated locally using heuristics derived from the actual uploaded dataset.

```bash
MOCK_MODE=true uvicorn server:app --reload --port 8000
```

The mock covers all stages including join recommendations (with chain joins for 3+ files), feature engineering, business context questions, and insights.

## Project structure

```
project/
├── server.py               ← FastAPI app + all route handlers
├── requirements.txt
├── README.md
├── FLOW.md                 ← Detailed data flow and architecture notes
├── src/
│   ├── __init__.py
│   ├── data_profiler.py    ← Local CSV parsing, stats, and quality report (no AWS)
│   ├── ai_analyst.py       ← AWS Bedrock calls (all AI functions)
│   └── mock_analyst.py     ← Offline mock for all AI functions (MOCK_MODE=true)
└── static/
    └── index.html          ← Full SPA — HTML + CSS + JS, zero build step
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the frontend |
| `POST` | `/api/profile` | Parses single CSV, returns statistical profile |
| `POST` | `/api/profile_multi` | Parses multiple CSVs, returns a profile for each |
| `POST` | `/api/recommend_joins` | AI: recommends join strategies across multiple datasets (including chain joins) |
| `POST` | `/api/merge_and_profile` | Applies a pairwise or chain join and returns the merged profile |
| `POST` | `/api/analyse` | AI: domain detection + ranked use cases |
| `POST` | `/api/biz_questions` | AI: generates 4 dataset-specific business context questions |
| `POST` | `/api/feature_engineering` | AI: feature transformation recommendations for the selected use case |
| `POST` | `/api/generate_code` | AI: complete Python starter script |
| `POST` | `/api/insights` | AI: next steps, output profiling, alt models |
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
