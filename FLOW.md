# Application Flow — CASEbook ML Use Case Recommender

## Overview

The application is a wizard that takes one or more CSV files and walks the user through dataset profiling, optional join merging, AI-powered ML use case discovery, feature engineering, starter code generation, and post-modelling guidance.

### Single file path
```
Upload CSV → AI Analysis → Business Context → Select Use Case → Feature Engineering → Starter Code → Insights
```

### Multi-file path
```
Upload CSVs → Join Recommendations → Apply Join → AI Analysis → Business Context → Select Use Case → Feature Engineering → Starter Code → Insights
```

---

## Stage 1 — Upload Dataset

**What the user does:** Drops or browses for one or more CSV files. Optionally adds plain-text descriptions of columns.

- File(s) are held in the browser's `FileList` — nothing is sent until the user clicks **Analyse Dataset**
- If a single file is uploaded, the wizard proceeds directly to Stage 2 (AI Analysis)
- If two or more files are uploaded, the wizard first routes through Stage 1b (Join Recommendations)

---

## Stage 1b — Join Recommendations *(multi-file only)*

### Multi-file profiling — `POST /api/profile_multi`

All uploaded CSVs are sent in one multipart request. The server parses each with pandas and returns a statistical profile per file (same structure as the single-file `/api/profile` response, plus a `filename` field).

### Join recommendation — `POST /api/recommend_joins`

```
Browser                    server.py              ai_analyst.py              AWS Bedrock
   │                           │                       │                          │
   │── { profiles[] } ────────▶│                       │                          │
   │                           │── recommend_joins() ──▶│                          │
   │                           │                       │── build prompt:          │
   │                           │                       │   schema summary per file│
   │                           │                       │   + chain join rules     │
   │                           │                       │   (when 3+ files)        │
   │                           │                       │                          │
   │                           │                       │── invoke_model() ────────▶│
   │                           │                       │◀── raw JSON ──────────────│
   │                           │                       │── _parse_json()          │
   │◀── { summary,            │◀── parsed dict ────────│
   │     recommendations[] } ──│
```

Each recommendation includes:
- `left_file`, `right_file`, `left_key`, `right_key`, `join_type` — for a standard pairwise join
- `chain_steps[]` — for a multi-table chain join (3+ files); each step specifies which files and keys to join sequentially; `__result__` is used as `left_file` for steps after the first

When 3+ files are uploaded, the AI always includes at least one chain-join recommendation that merges all datasets.

**Rendered in the UI:** Clickable join cards. Pairwise cards show the two files and join badge. Chain-join cards show the full sequence of files and per-step join types with a "chain join" badge.

### Applying the join — `POST /api/merge_and_profile`

```
Browser                    server.py
   │                           │
   │── FormData(files[],       │
   │   join_spec JSON) ────────▶│
   │                           │── if chain_steps present:
   │                           │     apply steps sequentially,
   │                           │     passing result of each step
   │                           │     as left DataFrame for the next
   │                           │── else:
   │                           │     standard pandas merge (two files)
   │                           │
   │                           │── profile_dataset(merged_df)
   │                           │── quality_report(merged_df, profile)
   │◀── { profile, quality_   │
   │     report } ─────────────│
```

The merged profile becomes `S.profile` for all downstream AI calls, exactly as if a single file had been uploaded.

---

## Stage 2 — AI Analysis

This stage makes **two parallel calls**: one local (quality report) and one to Bedrock (AI analysis).

### 2a. Data Quality Report — `POST /api/quality_report`

Runs as a fire-and-forget alongside the AI analysis. Entirely local — no Bedrock call.

```
data_profiler.quality_report(df, profile)
  │
  ├── Missing values      → warns if null% ≥ 20% (medium) or ≥ 50% (high)
  ├── Class imbalance     → warns if dominant class ≥ 75% (medium) or ≥ 90% (high)
  ├── Numeric skewness    → warns if |skew| ≥ 1 (medium) or ≥ 3 (high)
  ├── Leakage risk        → keyword-matches column names against target/label/outcome
  │                         and post-event patterns (overdue, written_off, etc.)
  └── Cardinality         → warns on constant columns (unique=1) or high-cardinality
                            categoricals (unique > 50)

Output: score 0–100, grade A/B/C/D, list of warnings with category + severity + fix advice
```

### 2b. AI Domain Analysis — `POST /api/analyse`

```
Browser                    server.py              ai_analyst.py              AWS Bedrock
   │                           │                       │                          │
   │── { profile, col_desc } ──▶│                       │                          │
   │                           │── analyse_dataset() ──▶│                          │
   │                           │                       │── build prompt:           │
   │                           │                       │   profile_to_text()       │
   │                           │                       │   + col descriptions      │
   │                           │                       │   + business_context      │
   │                           │                       │     (if already answered) │
   │                           │                       │                           │
   │                           │                       │── invoke_model() ────────▶│
   │                           │                       │◀── raw JSON ──────────────│
   │                           │                       │── _parse_json()           │
   │◀── { domain, summary,    │◀── parsed dict ────────│
   │     usecases[] } ─────────│
```

**AI response structure:**
- `domain` — detected industry (E-commerce, Healthcare, Finance, etc.)
- `domain_reason` — one sentence explaining the classification
- `summary` — 2–3 sentence description of the dataset
- `usecases[]` — 4–6 use cases, each with:
  - `title`, `type` (Classification / Regression / Clustering / Anomaly / etc.), `algorithm`
  - `target_column`, `features[]` (only real column names)
  - `data_readiness`, `ease_of_implementation`, `business_importance` (High/Medium/Low)
  - `score` (0–100, weighted: business 40%, readiness 35%, ease 25%)

**Rendered in the UI:**
- Overview stat cards (rows, columns, null rate, duplicates, size)
- Column profile table (type badge, null bar, unique count, mean, sample values)
- Typewriter animation plays out the AI summary
- Use case cards sorted by score with pill badges

---

## Stage 2c — Business Context Questions

After the initial analysis, the AI generates 4 dataset-specific questions to sharpen recommendations.

### `POST /api/biz_questions`

```
Browser                    server.py              ai_analyst.py              AWS Bedrock
   │                           │                       │                          │
   │── { profile, analysis } ──▶│                       │                          │
   │                           │── generate_biz_       │                          │
   │                           │   questions() ────────▶│                          │
   │                           │                       │── prompt references       │
   │                           │                       │   actual column names,    │
   │                           │                       │   detected domain, and    │
   │                           │                       │   top use case titles     │
   │                           │                       │── invoke_model() ────────▶│
   │                           │                       │◀── raw JSON ──────────────│
   │◀── { intro, questions[] } │◀── parsed dict ────────│
```

Returns 4 questions, each with a `label` and domain-specific `placeholder` examples. The user's answers are sent back with the next `/api/analyse` call as `business_context`, which causes the AI to re-rank use cases around the stated goal, stakeholders, pain points, and constraints.

---

## Stage 3 — Select Use Case

**What the user does:** Clicks a use case card to select it. The card expands to show rationale fields. The user clicks **Continue**.

**No server call.** The selected use case object is stored in `S.selectedUC`.

Selecting a different use case clears `S.insights` (the Stage 5 cache) and resets the feature engineering state.

---

## Stage 3b — Feature Engineering

### `POST /api/feature_engineering`

```
Browser                    server.py              ai_analyst.py              AWS Bedrock
   │                           │                       │                          │
   │── { usecase, profile,    │                       │                          │
   │     col_desc } ──────────▶│                       │                          │
   │                           │── generate_feature_  │                          │
   │                           │   engineering() ─────▶│                          │
   │                           │                       │── filters to only the    │
   │                           │                       │   feature + target cols  │
   │                           │                       │── prompt cites actual    │
   │                           │                       │   skew, null%, cardinality│
   │                           │                       │   values per column      │
   │                           │                       │── invoke_model() ────────▶│
   │                           │                       │◀── raw JSON ──────────────│
   │◀── { summary,            │◀── parsed dict ────────│
   │     recommendations[] } ──│
```

**Each recommendation includes:**
- `type` — Transform | New Feature | Encoding | Imputation | Interaction
- `columns[]` — which columns it applies to
- `title`, `rationale` (cites actual stat values), `impact` (High/Medium/Low)
- `code` — 1–3 line pandas snippet, runnable directly on a DataFrame named `df`

**Rendered in the UI:** Selectable cards. The user picks which recommendations to apply. Selected recommendations are passed to `/api/generate_code` as `fe_recs[]` and the AI wraps them in a `feature_engineering(df)` function called before the pipeline.

---

## Stage 4 — Starter Code

### `POST /api/generate_code`

```
Browser                    server.py              ai_analyst.py              AWS Bedrock
   │                           │                       │                          │
   │── { usecase, profile,    │                       │                          │
   │     col_desc, fe_recs[] } │                       │                          │
   │   ───────────────────────▶│                       │                          │
   │                           │── generate_starter_  │                          │
   │                           │   code() ────────────▶│                          │
   │                           │                       │── prompt embeds:         │
   │                           │                       │   use case + col stats   │
   │                           │                       │   + 13 required sections │
   │                           │                       │   + fe_recs if selected  │
   │                           │                       │── invoke_model() ────────▶│
   │                           │                       │◀── raw Python code ───────│
   │                           │                       │── strip markdown fences  │
   │◀── { code: "..." } ──────│◀── clean Python str ──│
```

**The generated script always includes:**
1. Header comment with install command
2. All imports
3. Configuration block (DATA_PATH, TARGET_COL, FEATURE_COLS, hyperparams)
4. Load & inspect
5. Exploratory data analysis — plots saved to `eda_plots.png`
6. Preprocessing pipeline — sklearn Pipeline + ColumnTransformer
7. Train/test split (supervised only)
8. Model training with the chosen algorithm
9. Evaluation (type-appropriate metrics and plots) — saved to `results.png`
10. Feature importance plot — saved to `feature_importance.png`
11. `predict()` function for scoring new data
12. `joblib.dump()` to save `model.joblib`
13. `if __name__ == '__main__':` main guard

If feature engineering recommendations were selected, a `feature_engineering(df)` function is injected and called immediately after data loading.

**Download options:**
- **Download .py** — direct browser download of the raw Python file
- **Download .ipynb** — calls `POST /api/export_notebook`, splits code on section-marker comments and wraps each section in its own Jupyter code cell with a markdown header above it

---

## Stage 5 — Post-Modelling Insights

### `POST /api/insights`

```
Browser                    server.py              ai_analyst.py              AWS Bedrock
   │                           │                       │                          │
   │── { usecase, profile,    │                       │                          │
   │     col_desc } ──────────▶│                       │                          │
   │                           │── generate_insights() ▶│                          │
   │                           │                       │── invoke_model() ────────▶│
   │                           │                       │◀── raw JSON ──────────────│
   │                           │                       │── _parse_json()          │
   │◀── { next_steps[],       │◀── parsed dict ────────│
   │     profiling_report{},  │
   │     alt_models[] } ───────│
```

**AI response structure:**

```
next_steps[]        5–7 items
  ├── phase:        Validation | Deployment | Monitoring | Retraining
  ├── title, detail (references actual columns and domain)
  └── priority:     High | Medium | Low

profiling_report{}
  ├── title, description
  └── profiles[]   3–4 items
        ├── label, description
        └── characteristics[]   specific traits using actual column names

alt_models[]        exactly 3
  ├── algorithm, complexity, speed, interpretability, expected_accuracy
  ├── when_to_use, pros[], cons[]
  └── vs_chosen: direct comparison sentence
```

**Profiling report adapts by use case type:**

| Use Case Type | Profile sections |
|---|---|
| Classification | One card per predicted class |
| Regression | Low / mid-range / high prediction bands |
| Clustering | 3–4 expected cluster archetypes |
| Anomaly Detection | Normal pattern / Anomalous / Borderline |

Insights are cached in `S.insights`. If the user navigates back and returns to Stage 5 without changing the use case, the cached result is rendered instantly.

---

## Mock Mode

When `MOCK_MODE=true`, all Bedrock functions are intercepted and replaced with local implementations in `src/mock_analyst.py`. No AWS credentials are needed.

| Function | Mock behaviour |
|---|---|
| `analyse_dataset` | Keyword-matches column names to detect domain; builds use cases from column type patterns |
| `recommend_joins` | Finds common columns across files; generates pairwise + chain-join recommendations for 3+ files |
| `generate_biz_questions` | Returns domain-aware question templates |
| `generate_feature_engineering` | Returns stat-driven recommendations based on skew, null%, and cardinality |
| `generate_starter_code` | Returns a complete Python script template via f-strings with type-specific eval blocks |
| `generate_insights` | Returns pre-written next steps, profiling templates, and alt model specs per use case type |

A short `time.sleep()` is added to simulate network latency so UI loading states render correctly.

---

## Data flow summary

```
CSV file(s) (user upload)
    │
    ├── [multi-file] /api/profile_multi   ← profile each file locally
    │       │
    │       └── /api/recommend_joins      → Bedrock → pairwise + chain join options
    │               │
    │               └── /api/merge_and_profile ← pandas merge (pairwise or chained)
    │
    ▼
data_profiler.profile_dataset()           ← pure pandas, always local
    │
    ├──▶ /api/quality_report              ← local scoring, shown in Stage 2
    │
    └──▶ profile dict (S.profile)
              │
              ├──▶ /api/analyse           → Bedrock → domain + use cases     (Stage 2)
              │
              ├──▶ /api/biz_questions     → Bedrock → context questions       (Stage 2c)
              │
              ├──▶ /api/feature_engineering → Bedrock → transform recs        (Stage 3b)
              │
              ├──▶ /api/generate_code     → Bedrock → Python script           (Stage 4)
              │         │
              │         └──▶ /api/export_notebook → nbformat → .ipynb
              │
              └──▶ /api/insights          → Bedrock → next steps +            (Stage 5)
                                                       profiling + alt models
```

All Bedrock calls share the same `_invoke()` helper using `invoke_model()` with the Messages API format. Responses are cleaned of markdown fences by `_parse_json()` before being returned to the frontend.
