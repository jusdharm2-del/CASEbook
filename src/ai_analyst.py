"""
ai_analyst.py
All AI interactions via AWS Bedrock (boto3).
Model: anthropic.claude-3-5-sonnet-20241022-v2:0  (configurable via env)

Required environment variables:
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_DEFAULT_REGION   (default: us-east-1)
  BEDROCK_MODEL_ID     (optional override)

Development / preview mode (no AWS credentials needed):
  MOCK_MODE=true       Routes all AI calls to src/mock_analyst.py
"""

from dotenv import load_dotenv
load_dotenv()

import json
import re
import os
import boto3
from botocore.exceptions import ClientError
from src.data_profiler import profile_to_text

# ── Mock mode ─────────────────────────────────────────────────────────────────
_MOCK = os.environ.get("MOCK_MODE", "").lower() in ("1", "true", "yes")

# ── Bedrock client ────────────────────────────────────────────────────────────
_AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-sonnet-4-6",
)

def _get_client():
    """Lazy Bedrock runtime client (picks up env credentials automatically)."""
    return boto3.client("bedrock-runtime", region_name=_AWS_REGION)


def _invoke(system: str, user_prompt: str, max_tokens: int = 4096) -> str:
    """
    Call Bedrock converse API with Anthropic Claude.
    Returns the raw text response.
    """
    client = _get_client()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    try:
        response = client.invoke_model(
            modelId=_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"].strip()

    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        raise RuntimeError(f"Bedrock ClientError [{code}]: {msg}") from e


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS — domain + use cases
# ─────────────────────────────────────────────────────────────────────────────

ANALYSIS_SYSTEM = """You are a senior ML consultant with deep expertise in data science, 
business strategy, and machine learning. You analyse datasets and recommend actionable 
ML use cases with clear business value.

Always respond with ONLY valid JSON — no markdown fences, no preamble, no explanation outside the JSON."""


def analyse_dataset(profile: dict, col_descriptions: str = "", business_context: dict = None) -> dict:
    """
    Uses Bedrock Claude to detect domain and generate use cases.
    Set MOCK_MODE=true to skip Bedrock (no credentials needed).
    """
    if _MOCK:
        from src.mock_analyst import analyse_dataset as _mock
        return _mock(profile, col_descriptions)
    profile_text = profile_to_text(profile)
    col_names = profile["column_names"]

    biz_ctx_text = ""
    if business_context:
        parts = []
        if business_context.get("goal"):
            parts.append(f"  Primary business goal: {business_context['goal']}")
        if business_context.get("stakeholders"):
            parts.append(f"  End users / stakeholders: {business_context['stakeholders']}")
        if business_context.get("pain_point"):
            parts.append(f"  Specific pain point to solve: {business_context['pain_point']}")
        if business_context.get("constraints"):
            parts.append(f"  Constraints / preferences: {business_context['constraints']}")
        if parts:
            biz_ctx_text = "\nBUSINESS CONTEXT (provided by user — prioritise use cases that match these):\n" + "\n".join(parts) + "\n"

    prompt = f"""Analyse this dataset and return a JSON object with ML use case recommendations.

DATASET PROFILE:
{profile_text}
{f"USER-PROVIDED COLUMN DESCRIPTIONS:{chr(10)}{col_descriptions}" if col_descriptions.strip() else ""}{biz_ctx_text}
Return ONLY this JSON structure (no markdown, no backticks):
{{
  "domain": "short domain name (e.g. E-commerce, Healthcare, Finance, HR, Logistics, etc.)",
  "domain_reason": "one sentence explaining why",
  "summary": "2-3 sentences describing what this dataset represents and its key characteristics for ML",
  "usecases": [
    {{
      "title": "concise use case name",
      "type": "Supervised Classification | Supervised Regression | Unsupervised Clustering | Anomaly Detection | Time Series | NLP | Recommendation",
      "algorithm": "specific algorithm name (e.g. XGBoost Classifier, K-Means, LSTM, etc.)",
      "business_context": "2-3 sentences: what exact business problem does this solve and what measurable value does it create",
      "target_column": "column name or null if unsupervised",
      "features": ["list", "of", "relevant", "column", "names"],
      "data_readiness": "High | Medium | Low",
      "data_readiness_reason": "one sentence",
      "ease_of_implementation": "High | Medium | Low",
      "ease_reason": "one sentence",
      "business_importance": "High | Medium | Low",
      "importance_reason": "one sentence",
      "score": 85
    }}
  ]
}}

Rules:
- Generate 4 to 6 use cases, ranked by score descending
- score is 0-100: weighted average (business_importance 40%, data_readiness 35%, ease 25%)
- High=3, Medium=2, Low=1 for scoring
- Only use column names that actually exist: {col_names}
- Be specific and realistic about what the data can support
- Do NOT make up columns that don't exist"""

    raw = _invoke(ANALYSIS_SYSTEM, prompt, max_tokens=3000)
    return _parse_json(raw, "analyse_dataset")


# ─────────────────────────────────────────────────────────────────────────────
# CODE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

CODE_SYSTEM = """You are an expert Python data scientist. You write clean, well-commented, 
production-ready Python code using pandas, scikit-learn, matplotlib, seaborn, and joblib.

Output ONLY Python code. No markdown, no backticks, no explanations outside of inline comments."""


def generate_starter_code(usecase: dict, profile: dict, col_descriptions: str = "", fe_recs: list = None) -> str:
    """
    Generates a complete Python starter script via Bedrock.
    Set MOCK_MODE=true to skip Bedrock (no credentials needed).
    """
    if _MOCK:
        from src.mock_analyst import generate_starter_code as _mock
        return _mock(usecase, profile, col_descriptions, fe_recs or [])
    ov = profile["overview"]
    col_stats = "\n".join(
        f"  {c['name']}: {c['type']}, nulls={c['null_pct']}%, unique={c['unique']}"
        for c in profile["columns"]
    )

    fe_block = ""
    if fe_recs:
        fe_lines = "\n".join(
            f"  {i+1}. [{r['type']}] {r['title']}\n     Code: {r['code']}"
            for i, r in enumerate(fe_recs)
        )
        fe_block = f"""
FEATURE ENGINEERING STEPS (user-selected — apply ALL of these):
{fe_lines}

  Important: Add a feature_engineering(df) function that applies every step above in order.
  Call it immediately after loading the data, before building the pipeline.
  Update FEATURE_COLS to include any new columns created and remove originals where replaced.
"""

    prompt = f"""Generate a complete Python script for this ML use case. Output ONLY Python code.

USE CASE:
  Title: {usecase['title']}
  Type: {usecase['type']}
  Algorithm: {usecase['algorithm']}
  Target column: {usecase.get('target_column') or 'N/A (unsupervised)'}
  Feature columns: {', '.join(usecase.get('features', []))}
  Business context: {usecase['business_context']}

DATASET INFO:
  Rows: {ov['total_rows']:,}  |  Columns: {ov['total_cols']}  |  Null rate: {ov['null_rate_pct']}%
  Column stats:
{col_stats}
{f"Column descriptions: {col_descriptions}" if col_descriptions.strip() else ""}{fe_block}

REQUIREMENTS — include ALL of these sections:

1. HEADER COMMENT — Title, description, install command (pip install ...)

2. IMPORTS — all necessary imports

3. CONFIGURATION block — easy-to-change constants:
   DATA_PATH, TARGET_COL, FEATURE_COLS, TEST_SIZE, RANDOM_STATE, model hyperparams

4. LOAD & INSPECT — read CSV, print shape and dtypes, show first rows

5. EXPLORATORY DATA ANALYSIS — distribution plots, correlation heatmap (if numeric),
   target class balance (if classification), save to 'eda_plots.png'

6. PREPROCESSING PIPELINE — handle nulls (median/mode imputation), encode categoricals
   (LabelEncoder or OneHotEncoder as appropriate), scale numerics if needed,
   use sklearn Pipeline + ColumnTransformer

7. TRAIN/TEST SPLIT (if supervised)

8. MODEL TRAINING — fit the specific algorithm with good default hyperparams,
   add comments explaining key hyperparameter choices

9. EVALUATION — appropriate metrics:
   - Classification: classification_report, confusion_matrix, ROC-AUC
   - Regression: RMSE, MAE, R², residual plot
   - Clustering: inertia, silhouette score, elbow method plot
   - Anomaly: contamination, scores distribution
   Save evaluation plots to 'results.png'

10. FEATURE IMPORTANCE / INSIGHTS — bar chart of feature importances,
    save to 'feature_importance.png'

11. PREDICT FUNCTION — def predict(new_data_path) that loads new CSV, preprocesses,
    and returns predictions

12. SAVE MODEL — save pipeline + model to 'model.joblib' using joblib

13. MAIN GUARD — if __name__ == '__main__': run everything in order with print statements

Use actual column names from the dataset. Handle edge cases (empty data, wrong dtypes).
Add helpful print statements throughout so the user can follow progress."""

    code = _invoke(CODE_SYSTEM, prompt, max_tokens=4096)
    # Strip any accidental markdown fences
    code = re.sub(r"^```python\n?", "", code)
    code = re.sub(r"^```\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code.strip()


# ─────────────────────────────────────────────────────────────────────────────
# INSIGHTS — post-modelling next steps, output profiling, alt models
# ─────────────────────────────────────────────────────────────────────────────

INSIGHTS_SYSTEM = """You are a senior ML consultant advising a data scientist who has just chosen a
machine learning approach for their dataset. You provide specific, actionable post-modelling guidance
grounded in the actual dataset and use case — not generic advice.

Always respond with ONLY valid JSON — no markdown fences, no preamble, no explanation outside the JSON."""


def generate_insights(usecase: dict, profile: dict, col_descriptions: str = "") -> dict:
    """
    Generate post-modelling insights: next steps checklist, output profiling report,
    and alternative model comparisons.
    Set MOCK_MODE=true to skip Bedrock.
    """
    if _MOCK:
        from src.mock_analyst import generate_insights as _mock
        return _mock(usecase, profile, col_descriptions)

    ov = profile["overview"]
    col_stats = "\n".join(
        f"  {c['name']}: {c['type']}, nulls={c['null_pct']}%, unique={c['unique']}"
        for c in profile["columns"]
    )

    prompt = f"""Generate post-modelling insights for this ML project.

USE CASE:
  Title: {usecase['title']}
  Type: {usecase['type']}
  Algorithm: {usecase['algorithm']}
  Target column: {usecase.get('target_column') or 'N/A (unsupervised)'}
  Features: {', '.join(usecase.get('features', []))}
  Business context: {usecase['business_context']}

DATASET:
  Rows: {ov['total_rows']:,}  |  Columns: {ov['total_cols']}  |  Null rate: {ov['null_rate_pct']}%
  Columns:
{col_stats}
{f"Column descriptions: {col_descriptions}" if col_descriptions.strip() else ""}

Return ONLY this JSON (no markdown, no backticks):
{{
  "next_steps": [
    {{
      "phase": "Validation | Deployment | Monitoring | Retraining",
      "title": "short actionable title (max 8 words)",
      "detail": "2-3 sentences of concrete, specific advice referencing actual columns or domain context",
      "priority": "High | Medium | Low"
    }}
  ],
  "profiling_report": {{
    "title": "e.g. 'Expected Cluster Profiles' or 'Predicted Class Characteristics'",
    "description": "2-3 sentences explaining what the model output will look like and how to interpret it in business terms",
    "profiles": [
      {{
        "label": "e.g. 'Cluster 1: High-Value Loyalists' or 'Predicted Churners'",
        "description": "what records in this group typically look like based on the features",
        "characteristics": ["specific characteristic using actual column names", "another characteristic", "a third one"]
      }}
    ]
  }},
  "alt_models": [
    {{
      "algorithm": "algorithm name",
      "complexity": "Low | Medium | High",
      "speed": "Slow | Medium | Fast",
      "interpretability": "Low | Medium | High",
      "expected_accuracy": "Lower | Similar | Higher",
      "when_to_use": "one concrete sentence about when to prefer this over the chosen algorithm",
      "pros": ["specific pro 1", "specific pro 2"],
      "cons": ["specific con 1", "specific con 2"],
      "vs_chosen": "one sentence directly comparing this to {usecase['algorithm']}"
    }}
  ]
}}

Rules:
- next_steps: 5-7 steps, mix of all 4 phases, ordered by priority then phase
- profiling_report.profiles: 3-4 profiles realistic for THIS use case type and domain
  - For clustering: describe expected cluster archetypes based on the feature columns
  - For classification: describe what each class typically looks like in the data
  - For regression: describe low / mid / high prediction bands
  - For anomaly: describe the normal pattern vs anomalous pattern
- alt_models: exactly 3 alternatives appropriate for {usecase['type']}
- Use actual column names in characteristics and next_steps detail where relevant
- Be specific to this domain and dataset, not generic ML boilerplate"""

    raw = _invoke(INSIGHTS_SYSTEM, prompt, max_tokens=3000)
    return _parse_json(raw, "generate_insights")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — recommendations for a selected use case
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_ENG_SYSTEM = """You are an expert feature engineer advising a data scientist preparing their
dataset for ML training. You analyse actual column statistics (skew, cardinality, null rates, types)
and the chosen use case to produce specific, actionable feature transformation recommendations.

Always respond with ONLY valid JSON — no markdown fences, no preamble, no explanation outside the JSON."""


def generate_feature_engineering(usecase: dict, profile: dict, col_descriptions: str = "") -> dict:
    """
    Generate feature engineering recommendations for a selected use case.
    Set MOCK_MODE=true to skip Bedrock.
    """
    if _MOCK:
        from src.mock_analyst import generate_feature_engineering as _mock
        return _mock(usecase, profile)

    target = usecase.get("target_column") or "N/A"
    features = usecase.get("features", [])
    relevant = set(features + ([target] if target != "N/A" else []))

    col_stats_lines = []
    for c in profile["columns"]:
        if c["name"] not in relevant:
            continue
        row = f"  {c['name']}: {c['type']}, nulls={c['null_pct']}%, unique={c['unique']}"
        if c["type"] == "numeric":
            row += f", skew={c.get('skew')}, min={c.get('min')}, max={c.get('max')}, mean={c.get('mean')}"
        elif c["type"] == "categorical":
            top = c.get("top_values", {})
            row += ", top: " + ", ".join(f"{k}({v})" for k, v in list(top.items())[:3])
        col_stats_lines.append(row)

    col_stats = "\n".join(col_stats_lines) or "  (no stats available)"

    prompt = f"""Generate feature engineering recommendations for this ML use case.

USE CASE:
  Title: {usecase['title']}
  Type: {usecase['type']}
  Algorithm: {usecase['algorithm']}
  Target column: {target}
  Feature columns: {', '.join(features)}

COLUMN STATISTICS (feature + target columns only):
{col_stats}
{f"Column descriptions: {col_descriptions}" if col_descriptions.strip() else ""}

Return ONLY this JSON (no markdown, no backticks):
{{
  "summary": "2-3 sentences describing the main engineering opportunities specific to this dataset and use case",
  "recommendations": [
    {{
      "type": "Transform | New Feature | Encoding | Imputation | Interaction",
      "columns": ["column_name"],
      "title": "Short action title (max 8 words)",
      "rationale": "1-2 sentences citing actual stats (e.g. skew=2.4, 23% nulls, 150 unique values) and why this helps {usecase['algorithm']}",
      "code": "pandas/numpy code — 1 line preferred, 3 max",
      "impact": "High | Medium | Low"
    }}
  ]
}}

Rules:
- Generate 4–7 recommendations, sorted by impact descending
- Only reference columns that exist: {sorted(relevant)}
- Be specific: cite actual stat values in the rationale
- Transform: log1p for skew > 1, sqrt for moderate skew, clip for extreme outliers
- Encoding: target encoding for unique > 20, ordinal for ordered categories, OHE for unique ≤ 10
- Imputation: add a binary missingness indicator flag when null_pct > 10%
- Interaction: only domain-relevant numeric × numeric or numeric / numeric ratios
- New Feature: datetime decomposition (year/month/dayofweek), bins (pd.qcut/pd.cut), polynomial terms
- Code must use df['col'] syntax, runnable directly on a pandas DataFrame named df
- Do NOT recommend StandardScaler — the sklearn Pipeline handles scaling"""

    raw = _invoke(FEATURE_ENG_SYSTEM, prompt, max_tokens=2000)
    return _parse_json(raw, "generate_feature_engineering")


# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS CONTEXT QUESTIONS — tailored to the dataset
# ─────────────────────────────────────────────────────────────────────────────

BIZ_QUESTIONS_SYSTEM = """You are a senior ML consultant conducting a brief discovery session with
a client who has just uploaded a dataset. Based on the dataset profile and detected domain, you
craft 4 highly specific, context-aware questions that will help you recommend the most relevant
ML use cases.

Always respond with ONLY valid JSON — no markdown fences, no preamble, no explanation outside the JSON."""


def generate_biz_questions(profile: dict, analysis: dict) -> dict:
    """
    Generate 4 dataset-specific business context questions.
    Set MOCK_MODE=true to skip Bedrock.
    """
    if _MOCK:
        from src.mock_analyst import generate_biz_questions as _mock
        return _mock(profile, analysis)

    domain = analysis.get("domain", "General Analytics")
    col_names = profile["column_names"]
    ov = profile["overview"]
    numeric_cols = [c["name"] for c in profile["columns"] if c["type"] == "numeric"][:5]
    categorical_cols = [c["name"] for c in profile["columns"] if c["type"] == "categorical"][:5]

    prompt = f"""A user has uploaded a dataset and I need to ask them 4 tailored business context questions
before recommending ML use cases. The questions must be specific to this exact dataset — not generic.

DATASET:
  Domain: {domain}
  Rows: {ov['total_rows']:,}  |  Columns: {ov['total_cols']}
  Key numeric columns: {', '.join(numeric_cols) or 'none'}
  Key categorical columns: {', '.join(categorical_cols) or 'none'}
  All columns: {', '.join(col_names)}

DETECTED USE CASES SUMMARY:
{chr(10).join(f"  - {uc['title']} ({uc['type']})" for uc in analysis.get('usecases', [])[:4])}

Generate exactly 4 questions that:
1. Reference actual column names or domain-specific terminology
2. Help prioritise between the detected use cases
3. Uncover real constraints (budget, timeline, regulatory, team size)
4. Identify the most important stakeholder outcome

Return ONLY this JSON (no markdown, no backticks):
{{
  "intro": "1-2 sentences addressed to the user, referencing the detected domain and dataset to show you've actually read it",
  "questions": [
    {{
      "label": "Specific question text referencing the domain or columns (max 12 words)",
      "placeholder": "2-3 concrete example answers specific to {domain} — separated by commas"
    }},
    {{
      "label": "...",
      "placeholder": "..."
    }},
    {{
      "label": "...",
      "placeholder": "..."
    }},
    {{
      "label": "...",
      "placeholder": "..."
    }}
  ]
}}

Question focus (one each):
1. Primary business goal — specific to the detected use cases and domain
2. End users / stakeholders — who acts on the model output in this domain
3. Specific pain point — referencing actual columns or domain metrics
4. Constraints — technical, regulatory, or resource constraints typical for this domain"""

    raw = _invoke(BIZ_QUESTIONS_SYSTEM, prompt, max_tokens=1000)
    return _parse_json(raw, "generate_biz_questions")


# ─────────────────────────────────────────────────────────────────────────────
# JOIN RECOMMENDATIONS — analyse multiple dataset profiles
# ─────────────────────────────────────────────────────────────────────────────

JOIN_SYSTEM = """You are a senior data engineer and ML consultant specialising in data integration.
You analyse schemas of multiple datasets and identify the best join strategies to create a unified
analytical dataset suitable for machine learning.

Always respond with ONLY valid JSON — no markdown fences, no preamble, no explanation outside the JSON."""


def recommend_joins(profiles: list) -> dict:
    """
    Analyse multiple dataset profiles and recommend join strategies.
    Set MOCK_MODE=true to skip Bedrock.
    """
    if _MOCK:
        from src.mock_analyst import recommend_joins as _mock
        return _mock(profiles)

    file_names = [p.get("filename", f"file_{i+1}") for i, p in enumerate(profiles)]

    dataset_summaries = []
    for p in profiles:
        fname = p.get("filename", "unknown")
        ov = p["overview"]
        col_lines = "\n".join(
            f"    {c['name']}: {c['type']}, unique={c['unique']}, nulls={c['null_pct']}%"
            + (f", sample: {', '.join(str(v) for v in c.get('sample', [])[:3])}" if c.get("sample") else "")
            for c in p["columns"]
        )
        dataset_summaries.append(
            f"File: {fname}\n"
            f"  Rows: {ov['total_rows']:,}  |  Columns: {ov['total_cols']}  |  Null rate: {ov['null_rate_pct']}%\n"
            f"  Columns:\n{col_lines}"
        )

    datasets_text = "\n\n".join(dataset_summaries)

    chain_instruction = ""
    if len(profiles) >= 3:
        chain_instruction = f"""
IMPORTANT — CHAIN JOIN (all {len(profiles)} datasets):
You MUST include at least one recommendation that joins ALL {len(profiles)} datasets together using
the "chain_steps" field. chain_steps is an array of sequential merge operations; the result of
each step is used as the left side of the next step. Use "__result__" as left_file for steps 2+.
Example for 3 files:
  "chain_steps": [
    {{"left_file": "A.csv", "right_file": "B.csv", "left_key": "id", "right_key": "id", "join_type": "inner"}},
    {{"left_file": "__result__", "right_file": "C.csv", "left_key": "id", "right_key": "product_id", "join_type": "left"}}
  ]
For chain join recommendations set left_file, right_file, left_key, right_key to null at the top level.
"""

    prompt = f"""Analyse these {len(profiles)} datasets and recommend the best join strategies.

DATASETS:
{datasets_text}
{chain_instruction}
Return ONLY this JSON (no markdown, no backticks):
{{
  "summary": "2-3 sentences describing what these datasets contain and the integration opportunity",
  "recommendations": [
    {{
      "title": "short descriptive join name (e.g. 'Customers ⋈ Orders on customer_id')",
      "left_file": "exact filename from the list above, or null for chain joins",
      "right_file": "exact filename from the list above, or null for chain joins",
      "left_key": "exact column name from left file, or null for chain joins",
      "right_key": "exact column name from right file, or null for chain joins",
      "chain_steps": null,
      "join_type": "inner | left | right | outer",
      "join_type_reason": "one sentence explaining this join type choice",
      "rationale": "2-3 sentences: why these are the best join keys, what ML value the merged dataset unlocks, and any row-explosion risk",
      "expected_rows": "rough estimate like 'similar to left dataset' or 'may grow to N× rows if keys are non-unique'",
      "use_case_hint": "one sentence — what ML use case does this join unlock?",
      "confidence": "High | Medium | Low",
      "confidence_reason": "one sentence about cardinality, naming clarity, or data quality concerns"
    }}
  ]
}}

Rules:
- Recommend 1–{min(4, len(profiles) * (len(profiles) - 1))} pairwise strategies PLUS any chain joins, ordered by confidence descending
- Only reference exact filenames: {file_names}
- Only reference exact column names that exist in the named files
- Prioritise columns named *_id, *_key, *_code, or with identical names in both files
- Also consider semantically related names (e.g. 'user_id' and 'customer_id')
- For join_type: inner preserves only matched rows; left preserves all left rows — choose based on analytical safety
- Warn in rationale if right_key has low uniqueness — the merge may explode row count
- For chain joins: chain_steps must reference exact filenames (or __result__ for step 2+), top-level left_file/right_file/left_key/right_key must be null"""

    raw = _invoke(JOIN_SYSTEM, prompt, max_tokens=2000)
    return _parse_json(raw, "recommend_joins")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(raw: str, context: str = "") -> dict:
    """Robustly parse JSON from Claude response."""
    cleaned = re.sub(r"```json\n?", "", raw)
    cleaned = re.sub(r"```\n?", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start: end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"[{context}] Failed to parse JSON from Bedrock response.\n"
            f"Error: {e}\nRaw response snippet:\n{raw[:600]}"
        ) from e
