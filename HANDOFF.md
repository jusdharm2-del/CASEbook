# Session Handoff — ML Use Case Recommender

## What this project is

A 5-step web tool: upload a CSV → AI detects the domain and recommends ranked ML use cases → pick one → get a complete Python starter script → view post-modelling insights (next steps, output profiles, alternative algorithms).

**Stack:** FastAPI + Uvicorn · AWS Bedrock (`anthropic.claude-sonnet-4-6`) · pandas/numpy · vanilla HTML/CSS/JS SPA (zero build step).

---

## How to run

```bash
cd /Users/amoghmakam/project_1
source venv/bin/activate

# With real AWS (reads .env automatically):
uvicorn server:app --reload --port 8000

# Without AWS credentials (fully offline):
MOCK_MODE=true uvicorn server:app --reload --port 8000
```

Open http://localhost:8000

---

## Project structure

```
project_1/
├── server.py               FastAPI app — all 7 API routes
├── requirements.txt
├── .env                    ⚠ Contains live AWS credentials (see security section)
├── src/
│   ├── data_profiler.py    Local CSV profiler + quality report (no AI cost)
│   ├── ai_analyst.py       AWS Bedrock calls (analyse, generate_code, insights)
│   └── mock_analyst.py     Full offline mock — mirrors ai_analyst.py signatures
└── static/
    └── index.html          Complete SPA — HTML + CSS + JS (1507 lines)
```

---

## API surface

| Method | Path | What it does |
|--------|------|--------------|
| GET | `/` | Serves the SPA |
| POST | `/api/profile` | Parses CSV → statistical profile (local, no AI) |
| POST | `/api/analyse` | Bedrock: domain detection + ranked use cases |
| POST | `/api/generate_code` | Bedrock: full Python starter script |
| POST | `/api/quality_report` | Local: quality score A–D + warnings |
| POST | `/api/export_notebook` | Local: converts code → `.ipynb` |
| POST | `/api/insights` | Bedrock: next steps, output profiles, alt models |
| GET | `/api/health` | Mode (mock/bedrock), region, model ID |

---

## What was done this session

### 1. Full UI redesign (`static/index.html`)

**Before:** sidebar + main-content grid layout, light-on-blue indigo palette, IBM Plex Mono + Inter fonts.

**After:**
- **Layout:** wizard / centered modal — single glassmorphism card (max 880px), horizontal 5-step progress track at the top.
- **Theme:** dark glassmorphism — `#100c28` base, animated gradient orbs (indigo/purple/cyan), `rgba` glass cards with `backdrop-filter: blur`.
- **Typography:** DM Sans (body) + JetBrains Mono (code/labels).
- **Light/dark toggle:** button in topbar; persists to `localStorage`; respects OS `prefers-color-scheme` on first visit.
- **Sticky step track:** uses `overflow: clip` on the wizard card (not `overflow: hidden`) so `position: sticky` works on the header child. Topbar is also sticky.
- **All JS logic untouched** — same IDs, same API calls, same state machine.

### 2. Light mode

Full CSS override block under `[data-theme="light"]`. Key differences:
- Background: `#edeaff` (soft lavender)
- Cards: `rgba(255,255,255, 0.62–0.88)` with indigo-tinted borders
- Text: dark (`rgba(14,9,38,...)`)
- Code block stays dark intentionally

---

## ⚠ Security — action required

**The `.env` file contains live AWS credentials in plaintext.** This repo has no `.gitignore`.

```
.env → AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are exposed
```

Before doing anything with git:
1. Rotate those AWS keys in the AWS console immediately if they've been used.
2. Create a `.gitignore` with at minimum:
   ```
   .env
   venv/
   __pycache__/
   *.pyc
   .DS_Store
   ```
3. Never commit `.env`.

---

## Known issues / open items

| Issue | Severity | Location |
|-------|----------|----------|
| AWS credentials in `.env`, no `.gitignore` | **Critical** | `.env` |
| `invoke_model` used instead of newer `converse` API | Low | `src/ai_analyst.py:57` |
| `infer_datetime_format=True` deprecated in newer pandas | Low | `src/data_profiler.py:124` |
| Notebook cell splitting relies on AI following exact comment markers | Low | `server.py:128` |
| No file size limit on CSV uploads | Medium | `server.py:47` |
| No test suite | Medium | — |
| `.env` sets `BEDROCK_MODEL_ID` to Haiku but README says Sonnet | Low | `.env` / `README.md` |

---

## Bedrock model config

- Default in `ai_analyst.py`: `anthropic.claude-sonnet-4-6`
- Current `.env` override: `anthropic.claude-3-haiku-20240307-v1:0` (cheaper/faster, lower quality)
- Region: `us-west-2`
- IAM permission needed: `bedrock:InvokeModel` on `arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-6`

---

## Mock mode behaviour

`MOCK_MODE=true` bypasses all Bedrock calls. `src/mock_analyst.py` generates realistic responses from the actual dataset profile using keyword heuristics. Covers all 5 stages including insights. Use this for UI work or demos without AWS credentials.

---

## Suggested next steps

- Add a `.gitignore` and `git init` (after rotating credentials)
- Add CSV file size limit (`< 50 MB`) in `/api/profile` and `/api/quality_report`
- Migrate `invoke_model` → `converse` API in `ai_analyst.py`
- Add at least smoke tests for `data_profiler.py` (pure functions, easy to test)
- Consider adding syntax highlighting to the code viewer (Prism.js or highlight.js, zero-build CDN import)
