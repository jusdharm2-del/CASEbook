"""
mock_analyst.py
Fake AI responses for UI development — no AWS credentials needed.

Activated automatically when:  MOCK_MODE=true  (set in environment)

The mock responses are dynamically built from the actual dataset profile,
so the UI looks realistic with whatever CSV you upload.
"""

import time
import random


# ─────────────────────────────────────────────────────────────────────────────
# Domain detection heuristics (keyword-based, no AI)
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_RULES = [
    (["churn", "subscription", "plan", "tenure", "contract", "monthly_charges"], "Telecommunications"),
    (["fraud", "transaction", "amount", "merchant", "card", "payment", "balance"], "Financial Services"),
    (["salary", "department", "employee", "hire_date", "performance", "attrition", "job_role"], "Human Resources"),
    (["price", "sales", "revenue", "product", "category", "discount", "quantity", "order"], "E-commerce / Retail"),
    (["diagnosis", "patient", "age", "bmi", "glucose", "blood", "hospital", "disease"], "Healthcare"),
    (["loan", "credit", "default", "income", "debt", "interest", "mortgage"], "Banking / Credit"),
    (["species", "sepal", "petal", "genus", "sample", "measurement"], "Life Sciences"),
    (["rating", "review", "sentiment", "text", "comment", "feedback", "score"], "NLP / Feedback"),
    (["latitude", "longitude", "city", "region", "location", "distance", "zip"], "Geospatial"),
    (["clicks", "impressions", "ctr", "campaign", "ad", "conversion", "session"], "Marketing Analytics"),
]

def _detect_domain(col_names: list[str]) -> tuple[str, str]:
    lower = [c.lower() for c in col_names]
    best_domain, best_hits = "General Analytics", 0
    for keywords, domain in _DOMAIN_RULES:
        hits = sum(1 for kw in keywords if any(kw in col for col in lower))
        if hits > best_hits:
            best_hits, best_domain = hits, domain
    reason = (
        f"Column names suggest {best_domain.lower()} data patterns."
        if best_hits > 0
        else "No strong domain signals detected; treating as general tabular data."
    )
    return best_domain, reason


# ─────────────────────────────────────────────────────────────────────────────
# Use-case templates (filled in with real column names where possible)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_target(cols: list[dict], type_filter: str, exclude: set) -> str | None:
    """Return the first column matching type_filter not in exclude."""
    for c in cols:
        if c["type"] == type_filter and c["name"] not in exclude:
            return c["name"]
    return None

def _pick_features(cols: list[dict], exclude: set, max_n: int = 6) -> list[str]:
    return [c["name"] for c in cols if c["name"] not in exclude][:max_n]


def _build_usecases(profile: dict) -> list[dict]:
    cols = profile["columns"]
    col_names = profile["column_names"]
    cat_cols = [c for c in cols if c["type"] == "categorical"]
    num_cols = [c for c in cols if c["type"] == "numeric"]
    all_names = set(col_names)

    usecases = []
    used_targets: set = set()

    # ── 1. Classification (if binary categorical exists) ──────────────────
    binary = next((c for c in cat_cols if c["unique"] == 2), None)
    if binary:
        target = binary["name"]
        used_targets.add(target)
        features = _pick_features(cols, {target})
        usecases.append({
            "title": f"Predict {target.replace('_',' ').title()}",
            "type": "Supervised Classification",
            "algorithm": "XGBoost Classifier",
            "business_context": (
                f"Train a binary classifier to predict '{target}' before it occurs. "
                "Early identification allows the business to take proactive action, "
                "reducing churn or risk and improving key KPIs."
            ),
            "target_column": target,
            "features": features,
            "data_readiness": "High",
            "data_readiness_reason": f"'{target}' is a clean binary label — ideal for supervised learning.",
            "ease_of_implementation": "High",
            "ease_reason": "XGBoost handles mixed types natively with minimal preprocessing.",
            "business_importance": "High",
            "importance_reason": "Direct impact on retention or risk metrics.",
            "score": 92,
        })

    # ── 2. Regression (if numeric target-like column exists) ──────────────
    reg_candidates = [c for c in num_cols if any(kw in c["name"].lower() for kw in ["price","amount","salary","revenue","spend","value","cost","score","rate"])]
    if not reg_candidates:
        reg_candidates = num_cols[:1]
    if reg_candidates:
        target = reg_candidates[0]["name"]
        used_targets.add(target)
        features = _pick_features(cols, {target})
        usecases.append({
            "title": f"Forecast {target.replace('_',' ').title()}",
            "type": "Supervised Regression",
            "algorithm": "Gradient Boosting Regressor",
            "business_context": (
                f"Build a regression model to predict '{target}' from historical patterns. "
                "Accurate forecasts enable better resource allocation, pricing strategy, "
                "and financial planning."
            ),
            "target_column": target,
            "features": features,
            "data_readiness": "High" if reg_candidates[0]["null_pct"] < 10 else "Medium",
            "data_readiness_reason": f"Numeric target with {reg_candidates[0]['null_pct']}% nulls.",
            "ease_of_implementation": "High",
            "ease_reason": "Gradient Boosting is robust to outliers and works well out of the box.",
            "business_importance": "High",
            "importance_reason": "Forecasting drives planning and budget decisions.",
            "score": 88,
        })

    # ── 3. Customer / Record Segmentation ────────────────────────────────
    if len(num_cols) >= 2:
        features = _pick_features(num_cols, used_targets, max_n=5)
        usecases.append({
            "title": "Customer Segmentation",
            "type": "Unsupervised Clustering",
            "algorithm": "K-Means Clustering",
            "business_context": (
                "Group records into distinct segments based on behavioural or demographic features. "
                "Segments inform targeted marketing, personalised offers, and product development "
                "without needing labelled data."
            ),
            "target_column": None,
            "features": features,
            "data_readiness": "Medium",
            "data_readiness_reason": "Clustering works well once nulls are imputed and features are scaled.",
            "ease_of_implementation": "High",
            "ease_reason": "K-Means is fast, interpretable, and easy to tune with the elbow method.",
            "business_importance": "Medium",
            "importance_reason": "Segmentation enables personalisation but requires downstream action plans.",
            "score": 78,
        })

    # ── 4. Anomaly Detection ─────────────────────────────────────────────
    if num_cols:
        features = _pick_features(num_cols, used_targets, max_n=5)
        usecases.append({
            "title": "Anomaly / Fraud Detection",
            "type": "Anomaly Detection",
            "algorithm": "Isolation Forest",
            "business_context": (
                "Automatically flag unusual records that deviate from normal patterns. "
                "Useful for fraud detection, equipment fault prediction, data quality checks, "
                "or identifying high-value outlier events."
            ),
            "target_column": None,
            "features": features,
            "data_readiness": "Medium",
            "data_readiness_reason": "Isolation Forest is unsupervised — no labels needed, but domain validation is required.",
            "ease_of_implementation": "Medium",
            "ease_reason": "Threshold tuning and explainability require domain expertise.",
            "business_importance": "High",
            "importance_reason": "Anomaly detection directly reduces financial or operational risk.",
            "score": 74,
        })

    # ── 5. Multi-class classification (if categorical with 3-10 classes) ─
    multi = next((c for c in cat_cols if 3 <= c["unique"] <= 10 and c["name"] not in used_targets), None)
    if multi:
        target = multi["name"]
        features = _pick_features(cols, {target})
        usecases.append({
            "title": f"Classify {target.replace('_',' ').title()}",
            "type": "Supervised Classification",
            "algorithm": "Random Forest Classifier",
            "business_context": (
                f"Predict the category of '{target}' for new records. "
                "Multi-class classifiers automate categorisation tasks, "
                "reducing manual effort and improving consistency at scale."
            ),
            "target_column": target,
            "features": features,
            "data_readiness": "Medium",
            "data_readiness_reason": f"'{target}' has {multi['unique']} classes — check for class imbalance.",
            "ease_of_implementation": "Medium",
            "ease_reason": "Random Forest is robust but multi-class evaluation needs careful metric selection.",
            "business_importance": "Medium",
            "importance_reason": "Automation of categorisation saves analyst hours.",
            "score": 70,
        })

    # Sort by score
    usecases.sort(key=lambda u: u["score"], reverse=True)
    return usecases[:6]


# ─────────────────────────────────────────────────────────────────────────────
# Public API — mirrors ai_analyst.py signatures exactly
# ─────────────────────────────────────────────────────────────────────────────

def analyse_dataset(profile: dict, col_descriptions: str = "") -> dict:
    """Mock version — returns heuristic-based analysis instantly."""
    time.sleep(1.2)  # simulate network latency

    domain, domain_reason = _detect_domain(profile["column_names"])
    ov = profile["overview"]
    usecases = _build_usecases(profile)

    summary = (
        f"This dataset contains {ov['total_rows']:,} records across {ov['total_cols']} columns "
        f"with a {ov['null_rate_pct']}% overall null rate. "
        f"The column mix ({len([c for c in profile['columns'] if c['type']=='numeric'])} numeric, "
        f"{len([c for c in profile['columns'] if c['type']=='categorical'])} categorical) "
        f"supports a range of supervised and unsupervised ML approaches. "
        f"{'Data quality looks strong — good foundation for modelling.' if ov['null_rate_pct'] < 10 else 'Some null handling will be needed before modelling.'}"
    )

    return {
        "domain": domain,
        "domain_reason": domain_reason,
        "summary": summary,
        "usecases": usecases,
    }


def generate_starter_code(usecase: dict, profile: dict, col_descriptions: str = "", fe_recs: list = None) -> str:
    """Mock version — returns a realistic but static Python template."""
    time.sleep(1.5)  # simulate generation latency

    uc_title = usecase["title"]
    algorithm = usecase["algorithm"]
    uc_type = usecase["type"]
    target = usecase.get("target_column") or "None"
    features = list(usecase.get("features", []))
    fe_recs = fe_recs or []

    # Track new columns added by FE so FEATURE_COLS can be updated
    fe_new_cols = []
    for r in fe_recs:
        code = r.get("code", "")
        # Heuristic: extract df['new_col'] = ... assignments
        import re as _re
        for m in _re.findall(r"df\['([^']+)'\]\s*=", code):
            if m not in features and m != target:
                fe_new_cols.append(m)

    # Build augmented feature list
    aug_features = features + [c for c in fe_new_cols if c not in features]
    feature_str = str(aug_features)

    col_names = profile["column_names"]
    filename = uc_title.lower().replace(" ", "_")

    is_classification = "Classification" in uc_type
    is_regression = "Regression" in uc_type
    is_clustering = "Clustering" in uc_type
    is_anomaly = "Anomaly" in uc_type

    # ── Evaluation block ──────────────────────────────────────────────────
    if is_classification:
        eval_block = """
    # ── Evaluation ──────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    print("\\n── Classification Report ──")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        axes[1].plot([0,1],[0,1],"--", color="gray")
        axes[1].set_title("ROC Curve"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("results.png", dpi=150); plt.close()
    print("Saved results.png")"""
    elif is_regression:
        eval_block = """
    # ── Evaluation ──────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    # Residual plot
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuals, alpha=.4, s=20)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title("Residuals vs Predicted")
    axes[1].hist(residuals, bins=40, edgecolor="white")
    axes[1].set_title("Residual Distribution")
    plt.tight_layout()
    plt.savefig("results.png", dpi=150); plt.close()
    print("Saved results.png")"""
    elif is_clustering:
        eval_block = """
    # ── Evaluation ──────────────────────────────────────────────────────────
    labels = pipeline.named_steps["model"].labels_
    score  = silhouette_score(X_scaled, labels)
    print(f"  Silhouette Score : {score:.4f}")
    print(f"  Cluster counts   :\\n{pd.Series(labels).value_counts().sort_index()}")

    # Elbow method
    inertias = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(K_range, inertias, "o-")
    plt.xlabel("Number of clusters (k)"); plt.ylabel("Inertia")
    plt.title("Elbow Method"); plt.tight_layout()
    plt.savefig("results.png", dpi=150); plt.close()
    print("Saved results.png")"""
    else:  # anomaly
        eval_block = """
    # ── Evaluation ──────────────────────────────────────────────────────────
    scores = pipeline.named_steps["model"].score_samples(X_scaled)
    threshold = np.percentile(scores, 5)  # flag bottom 5% as anomalies
    anomalies = scores < threshold
    print(f"  Anomalies detected : {anomalies.sum()} / {len(anomalies)}")

    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=50, edgecolor="white")
    plt.axvline(threshold, color="red", linestyle="--", label="threshold")
    plt.legend(); plt.title("Anomaly Score Distribution")
    plt.tight_layout()
    plt.savefig("results.png", dpi=150); plt.close()
    print("Saved results.png")"""

    supervised_split = f"""
    # ── Train / test split ───────────────────────────────────────────────────
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if df[TARGET_COL].nunique() <= 10 else None
    )
    print(f"Train: {{X_train.shape}}  Test: {{X_test.shape}}")""" if not (is_clustering or is_anomaly) else ""

    fit_call = "pipeline.fit(X_train, y_train)" if not (is_clustering or is_anomaly) else "pipeline.fit(X)"

    # Build the feature_engineering() function block from selected recs
    if fe_recs:
        fe_body_lines = "\n".join(f"    {r['code']}" for r in fe_recs)
        fe_func_block = f"""

# ── Feature Engineering ───────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
{fe_body_lines}
    return df
"""
        fe_call = "    df = feature_engineering(df)\n    print(f\"  After FE: {{df.shape[1]}} columns\")"
    else:
        fe_func_block = ""
        fe_call = ""

    return f'''# =============================================================================
# {uc_title}
# Type      : {uc_type}
# Algorithm : {algorithm}
# Generated : ML Use Case Recommender (Mock Mode — replace with real Bedrock output)
#
# Install   : pip install pandas scikit-learn matplotlib seaborn joblib
# Usage     : python {filename}_starter.py
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score,
)
{"from sklearn.ensemble import GradientBoostingClassifier" if is_classification else ""}
{"from sklearn.ensemble import GradientBoostingRegressor" if is_regression else ""}
{"from sklearn.cluster import KMeans" if is_clustering else ""}
{"from sklearn.ensemble import IsolationForest" if is_anomaly else ""}

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH    = "your_data.csv"         # ← update this path
TARGET_COL   = "{target}"
FEATURE_COLS = {feature_str}
TEST_SIZE    = 0.2
RANDOM_STATE = 42
N_CLUSTERS   = 4                       # K-Means only — tune with elbow plot
CONTAMINATION = 0.05                   # Isolation Forest only — expected anomaly fraction


# ── 1. Load & inspect ─────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {{df.shape[0]:,}} rows × {{df.shape[1]}} columns")
    print(df.dtypes.to_string())
    print("\\nNull counts:")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string() or "  none")
    return df
{fe_func_block}

# ── 2. EDA ────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> None:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    n = max(len(num_cols), 1)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, num_cols[:n]):
        df[col].dropna().hist(bins=40, ax=ax, edgecolor="white", color="#4B8FE2")
        ax.set_title(col); ax.set_xlabel("")
    plt.suptitle("Numeric distributions", y=1.02)
    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved eda_plots.png")


# ── 3. Build preprocessing pipeline ─────────────────────────────────────────
def build_pipeline(df: pd.DataFrame) -> Pipeline:
    X = df[FEATURE_COLS] if FEATURE_COLS else df.drop(columns=[TARGET_COL] if TARGET_COL != "None" else [], errors="ignore")

    num_features = X.select_dtypes(include="number").columns.tolist()
    cat_features = X.select_dtypes(exclude="number").columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ], remainder="drop")

    {"model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE)" if is_classification else ""}
    {"model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE)" if is_regression else ""}
    {"model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)" if is_clustering else ""}
    {"model = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE, n_estimators=200)" if is_anomaly else ""}

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return pipeline


# ── 4. Feature importance ─────────────────────────────────────────────────────
def plot_feature_importance(pipeline: Pipeline, feature_names: list) -> None:
    try:
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            ohe_features = pipeline.named_steps["preprocessor"].get_feature_names_out()
            importances = pd.Series(model.feature_importances_, index=ohe_features)
            importances = importances.sort_values(ascending=False).head(20)
            importances.plot(kind="barh", figsize=(8, 6), color="#4B8FE2")
            plt.xlabel("Importance"); plt.title("Feature Importances")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig("feature_importance.png", dpi=150)
            plt.close()
            print("Saved feature_importance.png")
    except Exception as e:
        print(f"Could not plot feature importance: {{e}}")


# ── 5. Predict on new data ────────────────────────────────────────────────────
def predict(new_data_path: str, pipeline_path: str = "model.joblib") -> pd.Series:
    pipe = joblib.load(pipeline_path)
    new_df = pd.read_csv(new_data_path)
    X_new = new_df[FEATURE_COLS] if FEATURE_COLS else new_df
    preds = pipe.predict(X_new)
    return pd.Series(preds, name="prediction")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"  {uc_title}")
    print("=" * 60)

    # 1. Load
    df = load_data(DATA_PATH)
{fe_call}
    # 2. EDA
    print("\\n[1/5] Running EDA...")
    run_eda(df)

    # 3. Pipeline
    print("\\n[2/5] Building preprocessing pipeline...")
    pipeline = build_pipeline(df)
{supervised_split}
    # 4. Train
    print("\\n[3/5] Training {algorithm}...")
    X_scaled = pipeline.named_steps["preprocessor"].fit_transform(
        df[FEATURE_COLS] if FEATURE_COLS else df
    ) if ({"True" if is_clustering or is_anomaly else "False"}) else None

    {"pipeline.fit(X_train, y_train)" if not (is_clustering or is_anomaly) else "pipeline.fit(df[FEATURE_COLS] if FEATURE_COLS else df)"}
    print("  Training complete.")

    # 5. Evaluate
    print("\\n[4/5] Evaluating..."){eval_block}

    # 6. Feature importance
    print("\\n[5/5] Plotting feature importances...")
    plot_feature_importance(pipeline, FEATURE_COLS)

    # 7. Save
    joblib.dump(pipeline, "model.joblib")
    print("\\nModel saved to model.joblib")
    print("\\nDone! ✓")
'''


# ─────────────────────────────────────────────────────────────────────────────
# Insights helpers
# ─────────────────────────────────────────────────────────────────────────────

_NEXT_STEPS_BY_TYPE = {
    "Supervised Classification": [
        {"phase": "Validation", "title": "Run k-fold cross-validation", "detail": "Use 5-fold stratified cross-validation to get a reliable estimate of model performance. Compare fold scores — high variance across folds indicates overfitting.", "priority": "High"},
        {"phase": "Validation", "title": "Check for class imbalance impact", "detail": "Review the classification report per class. If minority-class F1 is low, apply class_weight='balanced' or SMOTE oversampling before retraining.", "priority": "High"},
        {"phase": "Deployment", "title": "Wrap model in a prediction API", "detail": "Load the saved model.joblib and expose a /predict endpoint using FastAPI or Flask. Accept JSON input matching FEATURE_COLS and return predicted class and probability.", "priority": "Medium"},
        {"phase": "Monitoring", "title": "Log predictions and ground truth", "detail": "Store each prediction alongside the eventual true label in a database. This lets you compute live accuracy, precision, and recall as real outcomes arrive.", "priority": "High"},
        {"phase": "Monitoring", "title": "Set up feature drift detection", "detail": "Use statistical tests (KS-test for numeric, chi-squared for categorical) to compare incoming feature distributions against the training baseline weekly.", "priority": "Medium"},
        {"phase": "Retraining", "title": "Define a retraining trigger", "detail": "Retrain when monitored accuracy drops more than 5% below baseline or when drift is detected in 2+ features. Automate with a scheduled pipeline.", "priority": "Low"},
    ],
    "Supervised Regression": [
        {"phase": "Validation", "title": "Cross-validate with k-fold", "detail": "Use 5-fold cross-validation and report RMSE and R² across folds. A large gap between train and test R² signals overfitting — try stronger regularisation.", "priority": "High"},
        {"phase": "Validation", "title": "Inspect residuals for patterns", "detail": "Plot residuals against predicted values. Systematic patterns (funnel shape, curve) indicate heteroscedasticity or a non-linear relationship that the model is missing.", "priority": "High"},
        {"phase": "Deployment", "title": "Build a batch scoring pipeline", "detail": "For periodic forecasting, load the saved model and run predictions on new data files on a schedule. Output results to a database or reporting dashboard.", "priority": "Medium"},
        {"phase": "Monitoring", "title": "Track prediction error over time", "detail": "Compute rolling MAE and RMSE on recent predictions vs actuals. Alert when error exceeds 1.5× the baseline RMSE to catch model staleness early.", "priority": "High"},
        {"phase": "Monitoring", "title": "Monitor input feature distributions", "detail": "Use KS-test to compare incoming numeric feature distributions to training data monthly. Shift in key features often precedes degraded model performance.", "priority": "Medium"},
        {"phase": "Retraining", "title": "Schedule periodic retraining", "detail": "Retrain on a rolling window of the most recent 12 months of data quarterly, or immediately when monitored error exceeds the alert threshold.", "priority": "Low"},
    ],
    "Unsupervised Clustering": [
        {"phase": "Validation", "title": "Tune k with the elbow method", "detail": "Run K-Means for k=2 to 12 and plot inertia. Pick the k where the curve bends. Confirm with silhouette scores — aim for >0.4 for well-separated clusters.", "priority": "High"},
        {"phase": "Validation", "title": "Profile clusters with stakeholders", "detail": "Compute mean/mode of each feature per cluster and present to domain experts. Clusters only add value when they map to recognisable, actionable business segments.", "priority": "High"},
        {"phase": "Deployment", "title": "Assign labels to new records", "detail": "Use the trained pipeline to predict cluster labels for new data via pipeline.predict(). Store cluster assignments alongside record IDs in your database.", "priority": "Medium"},
        {"phase": "Monitoring", "title": "Monitor cluster size stability", "detail": "Track the proportion of records in each cluster monthly. A cluster that grows to >60% or shrinks to <5% signals that the segmentation is no longer valid.", "priority": "Medium"},
        {"phase": "Monitoring", "title": "Check centroid drift over time", "detail": "Periodically recompute cluster centroids on recent data and compare to the original centroids using cosine distance. Large drift means the segments have shifted.", "priority": "Low"},
        {"phase": "Retraining", "title": "Retrain when cluster stability degrades", "detail": "If cluster size proportions shift by >15% or centroid drift exceeds threshold, retrain the full pipeline on the most recent 6 months of data.", "priority": "Low"},
    ],
    "Anomaly Detection": [
        {"phase": "Validation", "title": "Manually review flagged anomalies", "detail": "Sample 20-30 flagged records and have domain experts classify them as true anomalies or false positives. Use this to calibrate the contamination parameter.", "priority": "High"},
        {"phase": "Validation", "title": "Tune the contamination threshold", "detail": "The contamination parameter controls the anomaly rate. Start at 0.05 (5%) and adjust based on expert review. Lower values = fewer but higher-confidence flags.", "priority": "High"},
        {"phase": "Deployment", "title": "Deploy as a real-time scoring service", "detail": "Wrap the model in a streaming pipeline. For each new record, call pipeline.score_samples() and flag records below the calibrated threshold for review.", "priority": "Medium"},
        {"phase": "Monitoring", "title": "Track daily anomaly rate", "detail": "Plot the daily percentage of records flagged as anomalous. A sudden spike indicates either a real incident or a shift in normal data patterns requiring model recalibration.", "priority": "High"},
        {"phase": "Monitoring", "title": "Watch for normal pattern drift", "detail": "As normal behaviour evolves over time, the model's definition of 'normal' becomes stale. Compare score distributions monthly against the original training baseline.", "priority": "Medium"},
        {"phase": "Retraining", "title": "Retrain on recent normal data", "detail": "Quarterly, retrain on data confirmed as normal (excluding known anomalies). This keeps the model calibrated to current baseline behaviour.", "priority": "Low"},
    ],
}

_ALT_MODELS_BY_TYPE = {
    "Supervised Classification": [
        {
            "algorithm": "Logistic Regression",
            "complexity": "Low", "speed": "Fast", "interpretability": "High",
            "expected_accuracy": "Lower",
            "when_to_use": "Use when you need a fast, interpretable baseline or when the decision boundary is approximately linear.",
            "pros": ["Outputs calibrated probabilities", "Coefficients are directly interpretable", "Trains in seconds even on large datasets"],
            "cons": ["Underperforms on non-linear relationships", "Requires manual feature engineering for interactions"],
            "vs_chosen": "Much simpler and faster than Gradient Boosting but trades accuracy for interpretability — good as a baseline.",
        },
        {
            "algorithm": "Random Forest Classifier",
            "complexity": "Medium", "speed": "Medium", "interpretability": "Medium",
            "expected_accuracy": "Similar",
            "when_to_use": "Use when you want robustness to outliers and noisy features without extensive hyperparameter tuning.",
            "pros": ["Less prone to overfitting than single trees", "Built-in feature importance", "Handles missing values via imputation naturally"],
            "cons": ["Slower to train and predict than Gradient Boosting", "Less accurate on complex patterns"],
            "vs_chosen": "Comparable accuracy on many datasets, more robust out of the box, but Gradient Boosting usually wins on leaderboard metrics.",
        },
        {
            "algorithm": "LightGBM Classifier",
            "complexity": "Medium", "speed": "Fast", "interpretability": "Low",
            "expected_accuracy": "Higher",
            "when_to_use": "Use when dataset is large (>100k rows), training speed matters, or you want to push accuracy further.",
            "pros": ["3-10× faster to train than Gradient Boosting", "Often higher accuracy on tabular data", "Native categorical feature support"],
            "cons": ["More hyperparameters to tune", "Can overfit on small datasets without careful regularisation"],
            "vs_chosen": "Usually matches or beats Gradient Boosting Classifier in accuracy while training significantly faster.",
        },
    ],
    "Supervised Regression": [
        {
            "algorithm": "Ridge Regression",
            "complexity": "Low", "speed": "Fast", "interpretability": "High",
            "expected_accuracy": "Lower",
            "when_to_use": "Use as a fast linear baseline when feature relationships are approximately linear and interpretability is required.",
            "pros": ["Coefficients directly explain feature impact", "Handles multicollinearity via L2 regularisation", "Extremely fast to train"],
            "cons": ["Cannot capture non-linear interactions", "Sensitive to outliers in the target"],
            "vs_chosen": "Much simpler than Gradient Boosting Regressor but useful as a baseline and for understanding linear feature contributions.",
        },
        {
            "algorithm": "Random Forest Regressor",
            "complexity": "Medium", "speed": "Medium", "interpretability": "Medium",
            "expected_accuracy": "Similar",
            "when_to_use": "Use when you want a robust model that handles outliers well and does not require target transformation.",
            "pros": ["Robust to outliers in features", "Built-in feature importance via impurity", "No need to scale features"],
            "cons": ["Cannot extrapolate beyond training data range", "Predictions are averages of leaves — may underestimate extremes"],
            "vs_chosen": "Similar accuracy to Gradient Boosting on many datasets, more robust to noise but weaker on extrapolation.",
        },
        {
            "algorithm": "XGBoost Regressor",
            "complexity": "High", "speed": "Medium", "interpretability": "Low",
            "expected_accuracy": "Higher",
            "when_to_use": "Use when maximising predictive accuracy is the priority and you have time for hyperparameter tuning.",
            "pros": ["State-of-the-art accuracy on tabular regression", "Built-in regularisation (L1 + L2)", "Handles missing values natively"],
            "cons": ["Many hyperparameters require careful tuning", "Black-box — harder to explain to stakeholders"],
            "vs_chosen": "Often outperforms Gradient Boosting Regressor with proper tuning; the two share similar architecture but XGBoost is more optimised.",
        },
    ],
    "Unsupervised Clustering": [
        {
            "algorithm": "DBSCAN",
            "complexity": "Medium", "speed": "Medium", "interpretability": "Medium",
            "expected_accuracy": "Similar",
            "when_to_use": "Use when clusters have irregular shapes or when you expect noise/outliers that should not belong to any cluster.",
            "pros": ["Automatically determines number of clusters", "Robust to outliers — flags them as noise", "Finds arbitrarily shaped clusters"],
            "cons": ["Sensitive to eps and min_samples parameters", "Struggles with varying density clusters"],
            "vs_chosen": "Unlike K-Means, DBSCAN does not assume spherical clusters and handles noise naturally — better for spatial or irregular data.",
        },
        {
            "algorithm": "Hierarchical Clustering (Ward)",
            "complexity": "High", "speed": "Slow", "interpretability": "High",
            "expected_accuracy": "Similar",
            "when_to_use": "Use when you need a dendrogram to visualise cluster relationships or when the number of clusters is unknown.",
            "pros": ["Dendrogram provides intuitive visual of cluster structure", "No need to specify k upfront", "Deterministic — same result every run"],
            "cons": ["O(n²) memory — impractical for >10k rows", "Cannot assign new data points without re-fitting"],
            "vs_chosen": "More interpretable than K-Means via the dendrogram but does not scale to large datasets.",
        },
        {
            "algorithm": "Gaussian Mixture Model",
            "complexity": "High", "speed": "Medium", "interpretability": "Medium",
            "expected_accuracy": "Higher",
            "when_to_use": "Use when clusters overlap or when you need soft (probabilistic) cluster assignments rather than hard labels.",
            "pros": ["Soft cluster membership probabilities", "Handles elliptical cluster shapes", "Model selection via BIC/AIC"],
            "cons": ["Assumes Gaussian distribution within clusters", "Can be sensitive to initialisation"],
            "vs_chosen": "More flexible than K-Means — handles overlapping clusters and non-spherical shapes at the cost of more computation.",
        },
    ],
    "Anomaly Detection": [
        {
            "algorithm": "Local Outlier Factor (LOF)",
            "complexity": "Medium", "speed": "Medium", "interpretability": "Medium",
            "expected_accuracy": "Similar",
            "when_to_use": "Use when anomalies are contextual — i.e., a point is only anomalous relative to its local neighbourhood, not globally.",
            "pros": ["Context-aware: detects local density anomalies", "No assumption about global data distribution", "Works well with varying density regions"],
            "cons": ["Does not support predict() on new data without re-fitting", "Slow on large datasets (O(n²) per query)"],
            "vs_chosen": "More sensitive to local anomalies than Isolation Forest, but cannot score new data points without retraining.",
        },
        {
            "algorithm": "One-Class SVM",
            "complexity": "High", "speed": "Slow", "interpretability": "Low",
            "expected_accuracy": "Similar",
            "when_to_use": "Use when the decision boundary between normal and anomalous is complex and non-linear.",
            "pros": ["Effective in high-dimensional spaces", "Kernel trick captures non-linear boundaries", "Works well with small normal-class datasets"],
            "cons": ["Very slow to train on >10k rows", "Sensitive to kernel and nu hyperparameters", "Memory intensive"],
            "vs_chosen": "More expressive boundary than Isolation Forest but much slower — only practical for small datasets.",
        },
        {
            "algorithm": "Autoencoder (Neural Network)",
            "complexity": "High", "speed": "Medium", "interpretability": "Low",
            "expected_accuracy": "Higher",
            "when_to_use": "Use when data has complex non-linear structure and you have enough data (>50k rows) to train a deep model.",
            "pros": ["Learns complex non-linear normal patterns", "Reconstruction error is a natural anomaly score", "Scales well to high-dimensional data"],
            "cons": ["Requires deep learning setup (PyTorch/TensorFlow)", "Many hyperparameters (layers, bottleneck size, lr)", "Black-box — hard to explain"],
            "vs_chosen": "Higher ceiling accuracy than Isolation Forest on complex data, but requires significantly more setup and data.",
        },
    ],
}


def _build_profiling_mock(uc_type: str, target: str, features: list, profile: dict) -> dict:
    num_cols = [c["name"] for c in profile["columns"] if c["type"] == "numeric"]
    cat_cols = [c["name"] for c in profile["columns"] if c["type"] == "categorical"]
    feat_sample = features[:4] if features else (num_cols[:2] + cat_cols[:2])

    if "Classification" in uc_type:
        return {
            "title": f"Predicted Class Profiles — {target or 'Target'}",
            "description": (
                f"The classifier will output a binary or multi-class label for '{target}'. "
                "Each class has a distinct feature signature — understanding these helps validate "
                "model logic and design targeted interventions."
            ),
            "profiles": [
                {
                    "label": f"Positive Class ('{target}' = 1)",
                    "description": "Records predicted to belong to the positive class tend to have distinctive feature patterns.",
                    "characteristics": [
                        f"Higher values in numeric features such as {feat_sample[0]}" if feat_sample else "Higher values in key numeric features",
                        f"Specific categorical patterns in {feat_sample[-1]}" if len(feat_sample) > 1 else "Distinct category distributions",
                        "More missing values or outlier presence in one or more features",
                    ],
                },
                {
                    "label": f"Negative Class ('{target}' = 0)",
                    "description": "The majority class typically represents the baseline / normal state.",
                    "characteristics": [
                        "Values concentrated near the median across numeric features",
                        f"More uniform distribution in categorical columns like {cat_cols[0]}" if cat_cols else "More uniform categorical distributions",
                        "Lower variance overall — these records are closer to the 'normal' centroid",
                    ],
                },
                {
                    "label": "High-Confidence Boundary Cases",
                    "description": "Records near the decision boundary that the model is uncertain about.",
                    "characteristics": [
                        "Predicted probability between 0.4 and 0.6 — worth manual review",
                        "Mixed feature signals: some indicators pointing each way",
                        "Worth examining for data quality issues or feature engineering opportunities",
                    ],
                },
            ],
        }

    if "Regression" in uc_type:
        return {
            "title": f"Prediction Band Profiles — {target or 'Target'}",
            "description": (
                f"The regression model outputs a continuous value for '{target}'. "
                "Segmenting predictions into bands helps you understand which records drive "
                "high vs low predicted values and where to focus business action."
            ),
            "profiles": [
                {
                    "label": "High Predictions (top 25%)",
                    "description": f"Records predicted to have the highest '{target}' values.",
                    "characteristics": [
                        f"Elevated values in {feat_sample[0]}" if feat_sample else "Elevated values in key numeric features",
                        "Fewer missing values — well-observed records tend to score higher",
                        "Priority segment for upsell, investment, or risk management depending on domain",
                    ],
                },
                {
                    "label": "Mid-Range Predictions (middle 50%)",
                    "description": "The bulk of records fall here — average behaviour across features.",
                    "characteristics": [
                        "Feature values near their training-set medians",
                        "Model uncertainty is lowest in this band — most reliable predictions",
                        "Largest segment by volume; small improvements per record have high aggregate impact",
                    ],
                },
                {
                    "label": "Low Predictions (bottom 25%)",
                    "description": f"Records predicted to have the lowest '{target}' values.",
                    "characteristics": [
                        "Feature values at the lower end of their ranges",
                        f"May include records with high null rates in {feat_sample[-1]}" if feat_sample else "May include records with missing data",
                        "Worth investigating whether low predictions are an opportunity or a risk signal",
                    ],
                },
            ],
        }

    if "Clustering" in uc_type:
        return {
            "title": "Expected Cluster Profiles",
            "description": (
                "K-Means will partition records into k groups based on feature similarity. "
                "The cluster profiles below represent archetypal segments you are likely to discover — "
                "exact characteristics will depend on your specific data."
            ),
            "profiles": [
                {
                    "label": "Cluster 1: High-Engagement / High-Value",
                    "description": "Typically the smallest but most valuable segment — records with the highest activity or value metrics.",
                    "characteristics": [
                        f"Above-average values in {feat_sample[0]}" if feat_sample else "Above-average values in key metrics",
                        "Low null rates — well-documented, active records",
                        "Priority for retention, upsell, or premium treatment strategies",
                    ],
                },
                {
                    "label": "Cluster 2: Average / Mainstream",
                    "description": "The largest cluster — records close to the feature means across the board.",
                    "characteristics": [
                        "Feature values near their dataset medians",
                        "Broadest and most heterogeneous group",
                        "Needs further sub-segmentation for targeted action",
                    ],
                },
                {
                    "label": "Cluster 3: Low-Engagement / At-Risk",
                    "description": "Records with below-average activity or value — potential churn or disengagement risk.",
                    "characteristics": [
                        f"Below-median values in {feat_sample[1]}" if len(feat_sample) > 1 else "Below-median values in activity features",
                        "Higher proportion of missing or null feature values",
                        "Candidates for re-engagement campaigns or risk monitoring",
                    ],
                },
                {
                    "label": "Cluster 4: Outlier / Niche Segment",
                    "description": "A small, distinctive cluster with unusual feature combinations.",
                    "characteristics": [
                        "Extreme values in one or more features — often edge cases",
                        "May represent a genuine niche customer type or data quality issues",
                        "Review manually before acting — confirm whether this is a real segment or noise",
                    ],
                },
            ],
        }

    # Anomaly Detection
    return {
        "title": "Normal vs Anomalous Record Profiles",
        "description": (
            "Isolation Forest learns the boundary of 'normal' behaviour from the training data. "
            "Records that are hard to isolate (require many splits) are scored as normal; "
            "easy-to-isolate records get low scores and are flagged as anomalous."
        ),
        "profiles": [
            {
                "label": "Normal Records (low anomaly score)",
                "description": "The vast majority of records — behave consistently with the training distribution.",
                "characteristics": [
                    f"Feature values within the typical range for {feat_sample[0]}" if feat_sample else "Feature values within typical ranges",
                    "Consistent patterns across correlated features",
                    "Require many tree splits to isolate — signature of dense, normal regions",
                ],
            },
            {
                "label": "Anomalous Records (high anomaly score)",
                "description": "Records that are statistically isolated from the rest — worth investigating.",
                "characteristics": [
                    "Extreme values in one or more numeric features",
                    f"Unusual combination of {feat_sample[0]} and {feat_sample[1]}" if len(feat_sample) > 1 else "Unusual feature combinations",
                    "Can be isolated in very few tree splits — signature of sparse, outlier regions",
                ],
            },
            {
                "label": "Borderline Cases (near threshold)",
                "description": "Records with anomaly scores close to the decision threshold — uncertain classification.",
                "characteristics": [
                    "Anomaly score within ±10% of the contamination threshold",
                    "May represent rare-but-legitimate behaviour or early-stage anomalies",
                    "Recommend manual review by a domain expert before acting",
                ],
            },
        ],
    }


_BIZ_QUESTIONS_BY_DOMAIN = {
    "Telecommunications": {
        "intro": "I can see this is a telecom customer dataset with churn and subscription signals. Let me ask a few questions to surface the use cases that matter most for your situation.",
        "questions": [
            {"label": "What churn reduction target are you aiming for?", "placeholder": "e.g. Reduce monthly churn from 8% to 5%, flag customers 30 days before contract expiry, recover 20% of at-risk accounts…"},
            {"label": "Who acts on the model's predictions?", "placeholder": "e.g. Customer success team, call centre agents, automated retention campaign system, account managers…"},
            {"label": "At what point in the customer lifecycle does churn become most costly?", "placeholder": "e.g. High-value customers after month 6, customers on annual contracts, those who've called support twice…"},
            {"label": "Any technical or regulatory constraints?", "placeholder": "e.g. Must explain decisions to customers, deploy on AWS, GDPR applies, team of 2 data scientists, 2-month timeline…"},
        ],
    },
    "Financial Services": {
        "intro": "This looks like a financial transactions dataset with fraud and payment signals. A few targeted questions will help me focus on the use cases with the most business impact for you.",
        "questions": [
            {"label": "What fraud or risk outcome are you trying to reduce?", "placeholder": "e.g. Reduce false declines by 30%, catch 95% of fraud before settlement, flag suspicious merchants in real-time…"},
            {"label": "Who reviews or acts on flagged transactions?", "placeholder": "e.g. Fraud analyst team, automated decline system, risk operations, compliance officer…"},
            {"label": "What's the current false positive rate you're trying to improve on?", "placeholder": "e.g. We decline 5% of legitimate transactions, analysts review 200 cases/day but 80% are benign…"},
            {"label": "Any compliance or latency constraints?", "placeholder": "e.g. Must respond within 200ms, PCI-DSS compliant, model decisions must be auditable, no third-party data…"},
        ],
    },
    "Human Resources": {
        "intro": "I can see this is an HR dataset with employee and attrition signals. Let me ask a few questions to prioritise the use cases most relevant to your people strategy.",
        "questions": [
            {"label": "What employee outcome are you most focused on improving?", "placeholder": "e.g. Reduce voluntary attrition by 15%, identify flight risk before resignation, optimise promotion decisions…"},
            {"label": "Who will use the model output?", "placeholder": "e.g. HR business partners, people analytics team, line managers, executive dashboard, HRIS system…"},
            {"label": "Which employee segments are highest priority?", "placeholder": "e.g. Senior engineers, first-year hires, employees in high-cost-to-replace roles, those below median salary band…"},
            {"label": "Any sensitivity or fairness constraints?", "placeholder": "e.g. Must not use protected attributes, decisions must be explainable to employees, GDPR/CCPA applies…"},
        ],
    },
    "E-commerce / Retail": {
        "intro": "This looks like a retail or e-commerce dataset with sales and product signals. A few questions will help me recommend the use cases that drive the most revenue impact for you.",
        "questions": [
            {"label": "What revenue or conversion outcome are you targeting?", "placeholder": "e.g. Increase basket size by 10%, reduce cart abandonment by 20%, improve demand forecast accuracy to ±5%…"},
            {"label": "Who consumes the model's output?", "placeholder": "e.g. Merchandising team, email marketing platform, recommendation engine, pricing analyst, supply chain planner…"},
            {"label": "What's your biggest current forecasting or personalisation gap?", "placeholder": "e.g. We overstock seasonal items by 30%, email campaigns have 1% CTR, no personalisation at all today…"},
            {"label": "Any technical or timeline constraints?", "placeholder": "e.g. Must integrate with Shopify/Salesforce, real-time serving required, 3-month roadmap, 1 ML engineer…"},
        ],
    },
    "Healthcare": {
        "intro": "I can see this is a healthcare or clinical dataset with patient and diagnostic signals. A few questions will help me identify the use cases that are both impactful and feasible given your constraints.",
        "questions": [
            {"label": "What clinical or operational outcome are you trying to improve?", "placeholder": "e.g. Reduce readmissions by 20%, flag high-risk patients 48h before deterioration, automate triage scoring…"},
            {"label": "Who acts on the model's predictions?", "placeholder": "e.g. Attending physicians, nursing staff, care coordinators, hospital admin system, insurance reviewers…"},
            {"label": "What's the acceptable false negative rate for this use case?", "placeholder": "e.g. Missing a high-risk patient is not acceptable, we can tolerate 20% false positives if true positive rate is high…"},
            {"label": "What regulatory or data governance constraints apply?", "placeholder": "e.g. HIPAA compliant, model must be FDA-cleared, no PHI in logs, must be explainable to clinicians…"},
        ],
    },
    "Banking / Credit": {
        "intro": "This looks like a banking or credit dataset with loan and default signals. Let me ask a few questions to focus on the use cases that match your risk appetite and compliance requirements.",
        "questions": [
            {"label": "What credit risk or lending outcome are you optimising for?", "placeholder": "e.g. Reduce default rate by 10%, approve 15% more creditworthy applicants, automate underwriting for loans under $50k…"},
            {"label": "Who uses the model's output in the decision process?", "placeholder": "e.g. Underwriting team, automated decisioning engine, risk committee, loan officer, credit bureau integration…"},
            {"label": "What's your current biggest source of model or process error?", "placeholder": "e.g. High false denial rate for thin-file customers, manual review takes 3 days, 12% default rate on approved loans…"},
            {"label": "What regulatory or fairness constraints apply?", "placeholder": "e.g. Fair lending / ECOA compliance, must provide adverse action reasons, model must be explainable to regulators…"},
        ],
    },
    "Marketing Analytics": {
        "intro": "I can see this is a marketing analytics dataset with campaign and conversion signals. A few questions will help me recommend the use cases with the highest ROI for your marketing team.",
        "questions": [
            {"label": "What campaign metric are you most focused on improving?", "placeholder": "e.g. Increase email CTR from 2% to 5%, reduce cost-per-acquisition by 25%, improve conversion rate for paid ads…"},
            {"label": "Which team or system consumes the model's output?", "placeholder": "e.g. Email marketing platform, Google Ads, CRM system, growth team, paid acquisition manager…"},
            {"label": "What's your biggest audience targeting gap today?", "placeholder": "e.g. No lookalike modelling, campaigns sent to full list with no segmentation, high unsubscribe rate…"},
            {"label": "Any budget, timeline, or data constraints?", "placeholder": "e.g. No third-party data allowed, must work with 6 months of history, integrate with HubSpot, 1-person team…"},
        ],
    },
}

_BIZ_QUESTIONS_DEFAULT = {
    "intro": "I've analysed your dataset and detected its key characteristics. A few targeted questions will help me recommend the ML use cases that best match your specific goals and situation.",
    "questions": [
        {"label": "What business outcome are you trying to improve?", "placeholder": "e.g. Reduce operational costs by 15%, automate a manual classification task, improve forecast accuracy by 20%…"},
        {"label": "Who will use or act on the model's output?", "placeholder": "e.g. Operations team, executive dashboard, automated pipeline, front-line staff, external API consumers…"},
        {"label": "What specific problem are you trying to solve?", "placeholder": "e.g. We currently do this manually and it takes 2 days, our current model has 60% accuracy, we have no prediction capability at all…"},
        {"label": "Any technical, regulatory, or team constraints?", "placeholder": "e.g. Must be explainable, deploy on AWS, 3-month timeline, team of 2 engineers, GDPR applies…"},
    ],
}


def generate_feature_engineering(usecase: dict, profile: dict) -> dict:
    """Mock version — builds recommendations from actual column stats, no AI call."""
    import time
    time.sleep(0.8)

    target = usecase.get("target_column") or ""
    features = usecase.get("features", [])
    uc_type = usecase.get("type", "")
    algorithm = usecase.get("algorithm", "")
    relevant = set(features + ([target] if target else []))
    col_map = {c["name"]: c for c in profile["columns"] if c["name"] in relevant}

    recs = []

    # ── Transforms for skewed numeric columns ─────────────────────────────────
    for name, c in col_map.items():
        if name == target or c["type"] != "numeric":
            continue
        skew = c.get("skew") or 0
        if abs(skew) >= 2:
            recs.append({
                "type": "Transform",
                "columns": [name],
                "title": f"Log-transform {name}",
                "rationale": f"skew={skew:.2f} — strong right skew. log1p compresses the tail and helps {algorithm} converge faster.",
                "code": f"df['{name}_log'] = np.log1p(df['{name}'].clip(lower=0))",
                "impact": "High",
            })
        elif abs(skew) >= 1:
            recs.append({
                "type": "Transform",
                "columns": [name],
                "title": f"Sqrt-transform {name}",
                "rationale": f"skew={skew:.2f} — moderate skew. Square-root transform reduces skew while preserving zeros.",
                "code": f"df['{name}_sqrt'] = np.sqrt(df['{name}'].clip(lower=0))",
                "impact": "Medium",
            })

    # ── Missingness indicator flags ───────────────────────────────────────────
    for name, c in col_map.items():
        if c["null_pct"] >= 10:
            recs.append({
                "type": "Imputation",
                "columns": [name],
                "title": f"Add missingness flag for {name}",
                "rationale": f"{c['null_pct']}% nulls — a binary flag preserves the information that a value was absent, which itself may be predictive.",
                "code": f"df['{name}_missing'] = df['{name}'].isnull().astype(int)",
                "impact": "High" if c["null_pct"] >= 30 else "Medium",
            })

    # ── Encoding recommendations for high-cardinality categoricals ────────────
    for name, c in col_map.items():
        if name == target or c["type"] != "categorical":
            continue
        if c["unique"] > 20:
            recs.append({
                "type": "Encoding",
                "columns": [name],
                "title": f"Target-encode {name}",
                "rationale": f"{c['unique']} unique categories — one-hot encoding would create {c['unique']} columns. Target encoding maps each category to its mean target value with cross-validation to prevent leakage.",
                "code": f"# Use category_encoders: encoder = ce.TargetEncoder(cols=['{name}'])",
                "impact": "High",
            })
        elif c["unique"] <= 10 and c["unique"] > 2:
            recs.append({
                "type": "Encoding",
                "columns": [name],
                "title": f"One-hot encode {name}",
                "rationale": f"{c['unique']} unique categories — low enough cardinality for one-hot encoding without dimensionality concerns.",
                "code": f"df = pd.get_dummies(df, columns=['{name}'], drop_first=True)",
                "impact": "Medium",
            })

    # ── Interaction feature between two numeric feature columns ───────────────
    num_features = [n for n in features if col_map.get(n, {}).get("type") == "numeric"]
    if len(num_features) >= 2:
        a, b = num_features[0], num_features[1]
        recs.append({
            "type": "Interaction",
            "columns": [a, b],
            "title": f"Create {a} × {b} interaction",
            "rationale": f"Multiplying {a} and {b} captures joint effects that neither feature encodes alone — useful for tree-based models like {algorithm}.",
            "code": f"df['{a}_x_{b}'] = df['{a}'] * df['{b}']",
            "impact": "Medium",
        })

    # ── Binning for a numeric feature ─────────────────────────────────────────
    if num_features:
        col = num_features[0]
        recs.append({
            "type": "New Feature",
            "columns": [col],
            "title": f"Bin {col} into quartile groups",
            "rationale": f"Quantile bins for {col} create an ordinal segment feature that captures non-linear threshold effects without relying on the raw scale.",
            "code": f"df['{col}_bin'] = pd.qcut(df['{col}'], q=4, labels=['Q1','Q2','Q3','Q4'], duplicates='drop')",
            "impact": "Medium",
        })

    # ── Sort by impact and cap at 6 ────────────────────────────────────────────
    order = {"High": 0, "Medium": 1, "Low": 2}
    recs.sort(key=lambda r: order.get(r["impact"], 2))
    recs = recs[:6]

    # ── Fallback if nothing was generated ────────────────────────────────────
    if not recs:
        recs = [{
            "type": "New Feature",
            "columns": list(relevant)[:1],
            "title": "Explore polynomial features",
            "rationale": "No strong skew, null, or cardinality signals detected. Polynomial or interaction features may still improve model expressiveness.",
            "code": "from sklearn.preprocessing import PolynomialFeatures  # add to pipeline",
            "impact": "Low",
        }]

    num_cols = sum(1 for c in col_map.values() if c["type"] == "numeric")
    cat_cols = sum(1 for c in col_map.values() if c["type"] == "categorical")
    null_cols = sum(1 for c in col_map.values() if c["null_pct"] > 10)
    summary = (
        f"Found {len(recs)} feature engineering opportunities across {len(col_map)} relevant columns "
        f"({num_cols} numeric, {cat_cols} categorical). "
        + (f"{null_cols} column(s) have significant nulls — adding missingness flags is recommended. " if null_cols else "")
        + f"Applying these transforms before training {algorithm} should improve predictive performance."
    )

    return {"summary": summary, "recommendations": recs}


def generate_biz_questions(profile: dict, analysis: dict) -> dict:
    """Mock version — returns domain-heuristic questions without any AI call."""
    import time
    time.sleep(0.4)
    domain = analysis.get("domain", "")
    return _BIZ_QUESTIONS_BY_DOMAIN.get(domain, _BIZ_QUESTIONS_DEFAULT)


def generate_insights(usecase: dict, profile: dict, col_descriptions: str = "") -> dict:  # noqa: ARG001
    """Mock version — returns type-aware insights without any AI call."""
    time.sleep(1.5)

    uc_type  = usecase.get("type", "")
    target   = usecase.get("target_column") or ""
    features = usecase.get("features", [])

    # Pick the closest type key
    type_key = next((k for k in _NEXT_STEPS_BY_TYPE if k in uc_type), "Supervised Classification")

    next_steps  = _NEXT_STEPS_BY_TYPE[type_key]
    alt_models  = _ALT_MODELS_BY_TYPE.get(type_key, _ALT_MODELS_BY_TYPE["Supervised Classification"])
    profiling   = _build_profiling_mock(uc_type, target, features, profile)

    return {
        "next_steps": next_steps,
        "profiling_report": profiling,
        "alt_models": alt_models,
    }


def recommend_joins(profiles: list) -> dict:
    """Mock join recommendations — no AI call required."""
    file_names = [p.get("filename", f"file_{i+1}") for i, p in enumerate(profiles)]

    # Find columns that appear in both datasets (potential join keys)
    col_sets = [set(p["column_names"]) for p in profiles]
    common_cols = col_sets[0].intersection(*col_sets[1:]) if len(col_sets) > 1 else set()

    # Prefer ID-like columns as join keys
    id_candidates = [
        c for c in common_cols
        if any(kw in c.lower() for kw in ["_id", "_key", "_code", "id_", "key_"])
    ]
    join_key = id_candidates[0] if id_candidates else (
        list(common_cols)[0] if common_cols else profiles[0]["column_names"][0]
    )

    left_rows = profiles[0]["overview"]["total_rows"]
    right_rows = profiles[1]["overview"]["total_rows"]

    recommendations = [
        {
            "title": f"{file_names[0]} ⋈ {file_names[1]} on {join_key}",
            "left_file": file_names[0],
            "right_file": file_names[1],
            "left_key": join_key,
            "right_key": join_key,
            "chain_steps": None,
            "join_type": "inner",
            "join_type_reason": "Inner join preserves only records with matching keys in both datasets, producing a clean analytical table.",
            "rationale": (
                f"'{join_key}' is present in both datasets and follows a key-column naming pattern. "
                "Joining on this column merges the feature spaces of both sources, enabling cross-table patterns to be captured. "
                "Verify key uniqueness in both files before joining to avoid unexpected row multiplication."
            ),
            "expected_rows": f"Up to {min(left_rows, right_rows):,} rows after inner join",
            "use_case_hint": "Merged dataset enables richer supervised models by combining features from multiple sources.",
            "confidence": "Medium",
            "confidence_reason": "Key column identified by naming pattern — confirm semantic meaning matches before joining.",
        }
    ]

    if len(profiles) >= 3:
        chain_steps = [
            {
                "left_file": file_names[0],
                "right_file": file_names[1],
                "left_key": join_key,
                "right_key": join_key,
                "join_type": "inner",
            }
        ]
        for fname in file_names[2:]:
            p = next(p for p in profiles if p.get("filename") == fname)
            step_key = next(
                (c for c in p["column_names"] if c == join_key),
                p["column_names"][0]
            )
            chain_steps.append({
                "left_file": "__result__",
                "right_file": fname,
                "left_key": join_key,
                "right_key": step_key,
                "join_type": "left",
            })
        recommendations.append({
            "title": " ⋈ ".join(file_names) + f" (all {len(file_names)} datasets)",
            "left_file": None,
            "right_file": None,
            "left_key": None,
            "right_key": None,
            "chain_steps": chain_steps,
            "join_type": "inner",
            "join_type_reason": "Chain join combines all datasets sequentially for the richest feature space.",
            "rationale": (
                f"Joining all {len(file_names)} datasets together produces a unified analytical table "
                f"with features from every source. The chain starts with '{join_key}' as the common key "
                "and progressively enriches the result. This maximises ML feature coverage but verify "
                "key alignment across all files first."
            ),
            "expected_rows": f"Up to {min(p['overview']['total_rows'] for p in profiles):,} rows after inner joins",
            "use_case_hint": f"Unified {len(file_names)}-table dataset enables the most feature-rich ML models.",
            "confidence": "Medium",
            "confidence_reason": "Chain join assumes common key propagates across all datasets — verify before applying.",
        })

    return {
        "summary": (
            f"These {len(profiles)} datasets can be integrated to create a richer analytical table. "
            f"'{join_key}' appears to be a common key for joining {file_names[0]} "
            f"({left_rows:,} rows) with {file_names[1]} ({right_rows:,} rows). "
            "A joined dataset enables more powerful ML use cases by combining features from all sources."
        ),
        "recommendations": recommendations,
    }
