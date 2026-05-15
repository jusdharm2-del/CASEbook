# Application Flow — ML Use Case Recommender

## Overview

The application is a 5-step wizard that takes a raw CSV file and walks the user through dataset profiling, AI-powered ML use case discovery, starter code generation, and post-modelling guidance.

```
User uploads CSV
      │
      ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Stage 1    │───▶│  Stage 2    │───▶│   Stage 3    │───▶│   Stage 4    │───▶│   Stage 5    │
│   Upload    │    │  Analysis   │    │ Select Use   │    │ Starter Code │    │  Insights    │
│             │    │             │    │    Case      │    │              │    │              │
└─────────────┘    └─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
  Local only      Local + Bedrock       User choice         Bedrock             Bedrock
```

---

## Stage 1 — Upload Dataset

**What the user does:** Drops or browses for a CSV file. Optionally adds plain-text descriptions of columns to give the AI more context.

**What happens locally (no server call yet):**
- File is held in the browser's `FileList` — nothing is sent until the user clicks **Analyse Dataset**
- The "Analyse Dataset" button is disabled until a valid `.csv` file is selected

**Inputs to next stage:**
- The raw CSV file (binary)
- Optional free-text column descriptions

---

## Stage 2 — AI Analysis

This stage makes **two parallel calls**: one local (profile + quality) and one to Bedrock (AI analysis).

### 2a. Dataset Profiling — `POST /api/profile`

```
Browser                          server.py                    data_profiler.py
   │                                 │                               │
   │──── FormData(file, col_desc) ──▶│                               │
   │                                 │──── pd.read_csv(bytes) ──────▶│
   │                                 │                               │── overview stats
   │                                 │                               │   (rows, cols, nulls,
   │                                 │                               │    memory, duplicates)
   │                                 │                               │
   │                                 │                               │── per-column stats
   │                                 │                               │   type inference:
   │                                 │                               │     numeric / categorical /
   │                                 │                               │     datetime / text
   │                                 │                               │   null%, unique count,
   │                                 │                               │   mean/std/skew (numeric)
   │                                 │                               │   top values (categorical)
   │                                 │                               │
   │◀──── { profile, col_desc } ────│◀──── profile dict ────────────│
```

The profile dict is stored in browser state (`S.profile`) and used in every subsequent Bedrock call.

### 2b. Data Quality Report — `POST /api/quality_report`

Runs in parallel with the AI analysis (fire-and-forget, does not block). Entirely local — no Bedrock call.

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

The quality panel is shown collapsed in the UI with the grade badge visible inline.

### 2c. AI Domain Analysis — `POST /api/analyse`

```
Browser                    server.py              ai_analyst.py              AWS Bedrock
   │                           │                       │                          │
   │── { profile, col_desc } ──▶│                       │                          │
   │                           │── analyse_dataset() ──▶│                          │
   │                           │                       │── build prompt:           │
   │                           │                       │   profile_to_text()       │
   │                           │                       │   + col descriptions      │
   │                           │                       │                           │
   │                           │                       │── invoke_model() ────────▶│
   │                           │                       │                          │── Claude generates
   │                           │                       │                          │   JSON response:
   │                           │                       │◀── raw JSON text ─────────│   domain, reason,
   │                           │                       │                          │   summary,
   │                           │                       │── _parse_json()           │   4–6 use cases
   │                           │                       │   (strips markdown fences,│
   │                           │                       │    extracts JSON object)  │
   │◀── { domain, summary,    │◀── parsed dict ────────│
   │     usecases[] } ─────────│
```

**AI prompt instructs Claude to return:**
- `domain` — detected industry (E-commerce, Healthcare, Finance, etc.)
- `domain_reason` — one sentence explaining the classification
- `summary` — 2–3 sentence description of the dataset
- `usecases[]` — 4–6 use cases, each with:
  - `title`, `type` (Classification / Regression / Clustering / Anomaly / etc.), `algorithm`
  - `target_column`, `features[]` (only real column names)
  - `data_readiness`, `ease_of_implementation`, `business_importance` (High/Medium/Low)
  - `score` (0–100, weighted: business 40%, readiness 35%, ease 25%)
  - Rationale fields for each rating

**Rendered in the UI:**
- Overview stat cards (rows, columns, null rate, duplicates, size)
- Column profile table (type badge, null bar, unique count, mean, sample values)
- Typewriter animation plays out the AI summary
- Use case cards sorted by score, with pill badges for each rating

---

## Stage 3 — Select Use Case

**What the user does:** Clicks a use case card to select it. The card expands to show the rationale fields. The user clicks **Continue**.

**No server call.** The selected use case object is stored in `S.selectedUC`.

Selecting a different use case clears `S.insights` (the Stage 5 cache) and disables the **View Insights** button, ensuring stale insights from a previous selection are never shown.

---

## Stage 4 — Starter Code

### Code generation — `POST /api/generate_code`

```
Browser                    server.py              ai_analyst.py              AWS Bedrock
   │                           │                       │                          │
   │── { usecase, profile,    │                       │                          │
   │     col_desc } ──────────▶│                       │                          │
   │                           │── generate_starter_  │                          │
   │                           │   code() ────────────▶│                          │
   │                           │                       │── build prompt:          │
   │                           │                       │   use case details       │
   │                           │                       │   + per-column stats     │
   │                           │                       │   + 13 required sections │
   │                           │                       │                          │
   │                           │                       │── invoke_model() ────────▶│
   │                           │                       │                          │── Claude generates
   │                           │                       │◀── raw Python code ───────│   complete .py script
   │                           │                       │                          │
   │                           │                       │── strip markdown fences  │
   │◀── { code: "..." } ──────│◀── clean Python str ──│
```

**The generated script always includes:**
1. Header comment with install command
2. All imports
3. Configuration block (DATA_PATH, TARGET_COL, FEATURE_COLS, hyperparams)
4. `load_data()` — read CSV, print shape/dtypes/nulls
5. `run_eda()` — distribution plots, saved to `eda_plots.png`
6. `build_pipeline()` — sklearn Pipeline + ColumnTransformer (imputation, encoding, scaling)
7. Train/test split (supervised only)
8. Model training with the chosen algorithm
9. Evaluation (type-appropriate: classification_report/ROC, RMSE/R², silhouette/elbow, anomaly scores)
10. Feature importance plot, saved to `feature_importance.png`
11. `predict()` function for scoring new data
12. `joblib.dump()` to save `model.joblib`
13. `if __name__ == '__main__':` main guard

**Download options:**
- **Download .py** — direct browser download of the raw Python file
- **Download .ipynb** — calls `POST /api/export_notebook`, which splits the code on section-marker comments (`# ── 1.`, `# ── 2.`, etc.) and wraps each section in its own Jupyter code cell with a markdown header cell above it

---

## Stage 5 — Post-Modelling Insights

### Insights generation — `POST /api/insights`

```
Browser                    server.py              ai_analyst.py              AWS Bedrock
   │                           │                       │                          │
   │── { usecase, profile,    │                       │                          │
   │     col_desc } ──────────▶│                       │                          │
   │                           │── generate_insights() ▶│                          │
   │                           │                       │── build prompt:          │
   │                           │                       │   use case + column stats│
   │                           │                       │   + domain context       │
   │                           │                       │                          │
   │                           │                       │── invoke_model() ────────▶│
   │                           │                       │                          │── Claude generates
   │                           │                       │◀── raw JSON ──────────────│   JSON response
   │                           │                       │                          │
   │                           │                       │── _parse_json()          │
   │◀── { next_steps[],       │◀── parsed dict ────────│
   │     profiling_report{},  │
   │     alt_models[] } ───────│
```

**AI response structure:**

```
next_steps[]
  ├── phase: Validation | Deployment | Monitoring | Retraining
  ├── title: short actionable title
  ├── detail: 2–3 sentences referencing actual columns and domain
  └── priority: High | Medium | Low

profiling_report{}
  ├── title: e.g. "Expected Cluster Profiles"
  ├── description: how to interpret model output in business terms
  └── profiles[]
        ├── label: e.g. "Cluster 1: High-Value Loyalists"
        ├── description: what records in this group look like
        └── characteristics[]: specific traits using actual column names

alt_models[]  (exactly 3)
  ├── algorithm, complexity, speed, interpretability, expected_accuracy
  ├── when_to_use: one sentence
  ├── pros[], cons[]
  └── vs_chosen: direct comparison to the selected algorithm
```

**Profiling report adapts by use case type:**

| Use Case Type | Profile sections |
|---|---|
| Classification | One card per predicted class (positive / negative / boundary cases) |
| Regression | Three prediction bands: high / mid-range / low |
| Clustering | 3–4 expected cluster archetypes based on feature columns |
| Anomaly Detection | Normal pattern / Anomalous pattern / Borderline cases |

**Insights are cached** in `S.insights`. If the user navigates back to Stage 4 and returns to Stage 5 without changing the use case, the cached result is rendered instantly. The cache is cleared whenever a new use case is selected in Stage 2.

---

## Mock Mode

When `MOCK_MODE=true`, all three Bedrock functions are intercepted and replaced with local implementations in `src/mock_analyst.py`. No AWS credentials are needed.

| Function | Mock behaviour |
|---|---|
| `analyse_dataset` | Keyword-matches column names to detect domain; builds use cases from column type patterns (binary cat → classification, numeric target → regression, etc.) |
| `generate_starter_code` | Returns a complete Python script template built via f-strings, with type-specific eval blocks |
| `generate_insights` | Returns pre-written next steps, profiling templates, and alt model specs per use case type |

A 1–1.5 second `time.sleep()` is added to simulate network latency so the UI loading states render correctly.

---

## Data flow summary

```
CSV file (user upload)
    │
    ▼
data_profiler.profile_dataset()        ← pure pandas, always local
    │
    ├──▶ quality_report()              ← local scoring, shown in Stage 2
    │
    └──▶ profile dict (S.profile)
              │
              ├──▶ /api/analyse        → Bedrock → domain + use cases  (Stage 2)
              │
              ├──▶ /api/generate_code  → Bedrock → Python script        (Stage 4)
              │         │
              │         └──▶ /api/export_notebook → nbformat → .ipynb   (Stage 4)
              │
              └──▶ /api/insights       → Bedrock → next steps +          (Stage 5)
                                                   profiling +
                                                   alt models
```

All Bedrock calls share the same `_invoke()` helper which uses `client.invoke_model()` with the Messages API format. Responses are cleaned of any markdown fences by `_parse_json()` before being returned to the frontend.
