"""
ML Use Case Recommender — FastAPI Backend
Serves the frontend and exposes API endpoints backed by AWS Bedrock.

Run:
    uvicorn server:app --reload --port 8000
"""

import json
import io
import os
import nbformat
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data_profiler import profile_dataset, quality_report
from src.ai_analyst import analyse_dataset, generate_starter_code, generate_insights, generate_biz_questions, generate_feature_engineering, recommend_joins

app = FastAPI(title="ML Use Case Recommender", version="2.0")

# Serve static files (frontend)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/profile")
async def api_profile(
    file: UploadFile = File(...),
    col_descriptions: str = Form(default=""),
):
    """Parse the CSV and return a statistical profile."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    if df.empty or df.shape[1] == 0:
        raise HTTPException(status_code=422, detail="CSV appears to be empty.")

    profile = profile_dataset(df)
    return JSONResponse({"profile": profile, "col_descriptions": col_descriptions})


@app.post("/api/analyse")
async def api_analyse(body: dict):
    """Run AI domain detection + use case generation."""
    profile = body.get("profile")
    col_descriptions = body.get("col_descriptions", "")
    business_context = body.get("business_context") or None
    if not profile:
        raise HTTPException(status_code=400, detail="Missing profile payload.")
    try:
        result = analyse_dataset(profile, col_descriptions, business_context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(result)


@app.post("/api/feature_engineering")
async def api_feature_engineering(body: dict):
    """Generate feature engineering recommendations for a selected use case."""
    usecase = body.get("usecase")
    profile = body.get("profile")
    col_descriptions = body.get("col_descriptions", "")
    if not usecase or not profile:
        raise HTTPException(status_code=400, detail="Missing usecase or profile payload.")
    try:
        result = generate_feature_engineering(usecase, profile, col_descriptions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(result)


@app.post("/api/biz_questions")
async def api_biz_questions(body: dict):
    """Generate dataset-tailored business context questions."""
    profile = body.get("profile")
    analysis = body.get("analysis")
    if not profile or not analysis:
        raise HTTPException(status_code=400, detail="Missing profile or analysis payload.")
    try:
        result = generate_biz_questions(profile, analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(result)


class CodeRequest(BaseModel):
    usecase: dict
    profile: dict
    col_descriptions: str = ""
    fe_recs: list = []


@app.post("/api/generate_code")
async def api_generate_code(body: CodeRequest):
    """Generate Python starter code for the selected use case."""
    try:
        code = generate_starter_code(body.usecase, body.profile, body.col_descriptions, body.fe_recs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"code": code})


@app.post("/api/quality_report")
async def api_quality_report(
    file: UploadFile = File(...),
):
    """Compute data quality warnings: imbalance, skew, leakage risk, cardinality."""
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")
    profile = profile_dataset(df)
    report  = quality_report(df, profile)
    return JSONResponse(report)


class NotebookRequest(BaseModel):
    code: str
    title: str
    usecase: dict


@app.post("/api/export_notebook")
async def api_export_notebook(body: NotebookRequest):
    """Convert generated Python code into a Jupyter .ipynb notebook."""
    nb = nbformat.v4.new_notebook()
    cells = []

    # ── Title + description markdown cell ─────────────────────────────────────
    cells.append(nbformat.v4.new_markdown_cell(
        f"# {body.title}\n\n"
        f"**Type:** {body.usecase.get('type', '')}  \n"
        f"**Algorithm:** {body.usecase.get('algorithm', '')}  \n\n"
        f"> {body.usecase.get('business_context', '')}\n\n"
        f"---\n"
        f"*Generated by ML Use Case Recommender*"
    ))

    # ── Split code into sections and make each a cell ─────────────────────────
    section_markers = [
        "# ── Configuration",
        "# ── 1.",
        "# ── 2.",
        "# ── 3.",
        "# ── 4.",
        "# ── 5.",
        "# ── Main",
        "if __name__",
    ]

    # Section heading → readable markdown label
    section_labels = {
        "# ── Configuration": "## ⚙️ Configuration\nEdit the constants below before running.",
        "# ── 1.": "## 1. Load & Inspect Data",
        "# ── 2.": "## 2. Exploratory Data Analysis",
        "# ── 3.": "## 3. Preprocessing Pipeline",
        "# ── 4.": "## 4. Feature Importance",
        "# ── 5.": "## 5. Predict on New Data",
        "# ── Main": "## 🚀 Run Everything",
        "if __name__": "## 🚀 Run Everything",
    }

    lines = body.code.split("\n")
    current_section_lines: list[str] = []
    current_label: str | None = None

    def flush(label, code_lines):
        """Emit a markdown header + code cell for the buffered section."""
        code_block = "\n".join(code_lines).strip()
        if not code_block:
            return
        if label:
            cells.append(nbformat.v4.new_markdown_cell(label))
        cells.append(nbformat.v4.new_code_cell(code_block))

    for line in lines:
        # Detect if this line starts a new section
        matched_marker = next((m for m in section_markers if line.strip().startswith(m)), None)
        if matched_marker:
            flush(current_label, current_section_lines)
            current_label = section_labels.get(matched_marker)
            current_section_lines = [line]
        else:
            current_section_lines.append(line)

    flush(current_label, current_section_lines)  # flush last section

    # ── Install deps reminder ─────────────────────────────────────────────────
    cells.insert(1, nbformat.v4.new_markdown_cell("## 📦 Install Dependencies"))
    cells.insert(2, nbformat.v4.new_code_cell(
        "# Run this cell first if dependencies are not installed\n"
        "# !pip install pandas scikit-learn matplotlib seaborn joblib"
    ))

    nb.cells = cells

    # Serialise and return as file download
    nb_bytes = nbformat.writes(nb).encode("utf-8")
    safe_title = body.title.lower().replace(" ", "_").replace("/", "_")
    filename   = f"{safe_title}_starter.ipynb"

    return Response(
        content=nb_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class InsightsRequest(BaseModel):
    usecase: dict
    profile: dict
    col_descriptions: str = ""


@app.post("/api/insights")
async def api_insights(body: InsightsRequest):
    """Generate post-modelling insights: next steps, output profiling, alt models."""
    try:
        result = generate_insights(body.usecase, body.profile, body.col_descriptions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(result)


@app.post("/api/profile_multi")
async def api_profile_multi(
    files: List[UploadFile] = File(...),
):
    """Parse multiple CSVs and return a profile for each."""
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 CSV files required.")

    profiles = []
    for file in files:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail=f"Only CSV files supported: {file.filename}")
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents), low_memory=False)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not parse {file.filename}: {e}")
        if df.empty or df.shape[1] == 0:
            raise HTTPException(status_code=422, detail=f"{file.filename} appears to be empty.")
        profile = profile_dataset(df)
        profile["filename"] = file.filename
        profiles.append(profile)

    return JSONResponse({"profiles": profiles})


@app.post("/api/recommend_joins")
async def api_recommend_joins(body: dict):
    """Use AI to recommend possible joins between multiple profiled datasets."""
    profiles = body.get("profiles")
    if not profiles or len(profiles) < 2:
        raise HTTPException(status_code=400, detail="At least 2 profiles required.")
    try:
        result = recommend_joins(profiles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(result)


def _apply_pairwise_merge(dfs: dict, left_file: str, right_file: str,
                           left_key: str, right_key: str, join_type: str,
                           result_df: pd.DataFrame = None) -> pd.DataFrame:
    """Merge two DataFrames by key. Pass result_df to use a prior merge result as the left side."""
    left_df = result_df if result_df is not None else dfs.get(left_file)
    right_df = dfs.get(right_file)
    if left_df is None:
        raise HTTPException(status_code=400, detail=f"File '{left_file}' not found in upload.")
    if right_df is None:
        raise HTTPException(status_code=400, detail=f"File '{right_file}' not found in upload.")
    if left_key not in left_df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{left_key}' not found in left dataset.")
    if right_key not in right_df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{right_key}' not found in {right_file}.")
    right_df = right_df.copy()
    if left_key != right_key:
        right_df = right_df.rename(columns={right_key: left_key})
    sfx_r = "_" + right_file.replace(".csv", "")[:10]
    return left_df.merge(right_df, on=left_key, how=join_type, suffixes=("", sfx_r))


@app.post("/api/merge_and_profile")
async def api_merge_and_profile(
    files: List[UploadFile] = File(...),
    join_spec: str = Form(...),
):
    """Apply a join (pairwise or chained) to datasets and return the merged profile + quality report."""
    spec = json.loads(join_spec)

    dfs = {}
    for file in files:
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents), low_memory=False)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not parse {file.filename}: {e}")
        dfs[file.filename] = df

    try:
        chain_steps = spec.get("chain_steps")
        if chain_steps:
            # Multi-table chain join: apply steps sequentially
            merged = None
            label_parts = []
            for i, step in enumerate(chain_steps):
                left_file = step["left_file"]
                right_file = step["right_file"]
                left_key = step["left_key"]
                right_key = step["right_key"]
                join_type = step.get("join_type", "inner")
                if i == 0:
                    label_parts.append(left_file)
                label_parts.append(right_file)
                merged = _apply_pairwise_merge(
                    dfs, left_file, right_file, left_key, right_key, join_type,
                    result_df=merged if i > 0 else None,
                )
            filename = " ⋈ ".join(label_parts)
        else:
            # Standard pairwise join
            left_file = spec["left_file"]
            right_file = spec["right_file"]
            left_key = spec["left_key"]
            right_key = spec["right_key"]
            join_type = spec.get("join_type", "inner")
            merged = _apply_pairwise_merge(dfs, left_file, right_file, left_key, right_key, join_type)
            filename = f"{left_file} ⋈ {right_file}"
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Merge failed: {e}")

    if merged is None or merged.empty:
        raise HTTPException(status_code=422, detail="Merged dataset is empty — no matching keys found.")

    profile = profile_dataset(merged)
    profile["filename"] = filename
    qr = quality_report(merged, profile)

    return JSONResponse({"profile": profile, "quality_report": qr})


@app.get("/api/health")
def health():
    mock = os.environ.get("MOCK_MODE", "").lower() in ("1", "true", "yes")
    return {
        "status": "ok",
        "mode": "mock (no AWS needed)" if mock else "bedrock",
        "bedrock_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "model": os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
    }
