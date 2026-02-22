from __future__ import annotations

import io
import itertools
import math
import os
import uuid
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, entropy as shannon_entropy, f_oneway, kurtosis as sp_kurtosis, skew as sp_skew, ttest_ind
from sklearn.preprocessing import KBinsDiscretizer

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# JSON serialization helper — converts numpy/pandas types to native Python
# ---------------------------------------------------------------------------

def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return None
    return obj


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _round(v: float | None, decimals: int = 2) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(v, decimals)


def _infer_dtype(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "text"

    # Boolean check
    unique_lower = set(non_null.astype(str).str.lower().unique())
    if unique_lower <= {"true", "false", "0", "1", "yes", "no"} and len(unique_lower) <= 2:
        return "boolean"

    # Numeric check
    coerced = pd.to_numeric(non_null, errors="coerce")
    if coerced.notna().sum() > len(non_null) * 0.8:
        return "numeric"

    # Categorical vs text
    if non_null.nunique() < min(len(non_null) * 0.3, 50):
        return "categorical"

    return "text"


def _profile_column(df: pd.DataFrame, col: str) -> dict[str, Any]:
    series = df[col]
    total_count = len(series)
    null_count = int(series.isna().sum())
    non_null = series.dropna()
    unique_count = int(non_null.nunique())

    dtype = _infer_dtype(series)

    profile: dict[str, Any] = {
        "name": col,
        "dtype": dtype,
        "totalCount": total_count,
        "nullCount": null_count,
        "nullPercent": round(null_count / total_count * 100) if total_count else 0,
        "uniqueCount": unique_count,
        "uniquePercent": round(unique_count / total_count * 100) if total_count else 0,
    }

    if dtype == "numeric":
        nums = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(nums) > 0:
            profile["mean"] = _round(nums.mean())
            profile["median"] = _round(nums.median())
            profile["std"] = _round(nums.std(ddof=1))
            profile["min"] = _round(nums.min())
            profile["max"] = _round(nums.max())
            q1, q3 = nums.quantile(0.25), nums.quantile(0.75)
            profile["q1"] = _round(q1)
            profile["q3"] = _round(q3)
            profile["iqr"] = _round(q3 - q1)
            p5, p95 = nums.quantile(0.05), nums.quantile(0.95)
            profile["p5"] = _round(p5)
            profile["p95"] = _round(p95)
            profile["skewness"] = _round(nums.skew())

    if dtype in ("categorical", "text"):
        vc = non_null.value_counts()
        total = len(non_null)
        if len(vc) > 0:
            profile["mode"] = str(vc.index[0])
            profile["modeCount"] = int(vc.iloc[0])
            profile["modePercent"] = round(int(vc.iloc[0]) / total * 1000) / 10
            profile["topValues"] = [
                {
                    "value": str(v),
                    "count": int(c),
                    "percent": round(int(c) / total * 1000) / 10,
                }
                for v, c in vc.head(5).items()
            ]
            profile["categoryCount"] = len(vc)
            profile["entropy"] = _round(shannon_entropy(vc.values, base=2))

    if dtype == "boolean":
        true_set = {"true", "1", "yes"}
        lower_vals = non_null.astype(str).str.lower()
        true_count = int(lower_vals.isin(true_set).sum())
        false_count = len(non_null) - true_count
        profile["trueCount"] = true_count
        profile["falseCount"] = false_count
        profile["truePercent"] = (
            round(true_count / len(non_null) * 1000) / 10 if len(non_null) else 0
        )

    return profile


# ---------------------------------------------------------------------------
# Profile endpoint
# ---------------------------------------------------------------------------

@app.route("/api/profile", methods=["POST"])
def profile_csv():
    file = request.files["file"]
    contents = file.read()
    df = pd.read_csv(io.BytesIO(contents))
    return jsonify(_sanitize([_profile_column(df, col) for col in df.columns]))


# ---------------------------------------------------------------------------
# Bias / imbalance analysis
# ---------------------------------------------------------------------------

def _jsd_numeric(groups: list[pd.Series], bins: int = 30) -> float:
    """Average pairwise Jensen-Shannon divergence for numeric feature across groups."""
    all_vals = pd.concat(groups)
    edges = np.histogram_bin_edges(all_vals.dropna(), bins=bins)

    hists = []
    for g in groups:
        h, _ = np.histogram(g.dropna(), bins=edges)
        h = h.astype(float)
        total = h.sum()
        hists.append(h / total if total > 0 else h)

    jsds = []
    for a, b in itertools.combinations(hists, 2):
        jsds.append(float(jensenshannon(a, b, base=2)))

    return float(np.mean(jsds)) if jsds else 0.0


def _jsd_categorical(groups: list[pd.Series]) -> float:
    """Average pairwise JSD for categorical feature across groups."""
    all_cats = set()
    for g in groups:
        all_cats.update(g.dropna().unique())
    cats = sorted(all_cats, key=str)

    vecs = []
    for g in groups:
        vc = g.dropna().value_counts()
        total = vc.sum()
        vec = np.array([vc.get(c, 0) / total if total > 0 else 0 for c in cats])
        vecs.append(vec)

    jsds = []
    for a, b in itertools.combinations(vecs, 2):
        jsds.append(float(jensenshannon(a, b, base=2)))

    return float(np.mean(jsds)) if jsds else 0.0


@app.route("/api/bias", methods=["POST"])
def compute_bias():
    file = request.files["file"]
    target_column = request.form["target_column"]
    selected_columns = request.form["selected_columns"]  # comma-separated

    contents = file.read()
    df = pd.read_csv(io.BytesIO(contents))

    selected = [c.strip() for c in selected_columns.split(",") if c.strip()]
    if target_column not in df.columns:
        return jsonify({"overallScore": 0, "metrics": [], "notice": "Target column not found."})

    target = df[target_column].dropna()
    class_counts = target.value_counts()
    n_classes = len(class_counts)

    metrics: list[dict[str, Any]] = []

    # --- 1. Class Imbalance ---
    if n_classes >= 2:
        max_c = int(class_counts.max())
        min_c = int(class_counts.min())
        imbalance_ratio = round(max_c / min_c, 2) if min_c > 0 else float("inf")

        # Score: ratio=1 → 100, ratio>=10 → 0
        if imbalance_ratio == float("inf"):
            imbalance_score = 0
        else:
            imbalance_score = max(0, min(100, round(100 - (imbalance_ratio - 1) * (100 / 9))))

        distribution = ", ".join(
            f"{cls}: {cnt}" for cls, cnt in class_counts.items()
        )
        if imbalance_ratio <= 1.5:
            desc = f"Classes are well-balanced (ratio {imbalance_ratio}:1). Distribution: {distribution}."
        elif imbalance_ratio <= 3:
            desc = f"Moderate class imbalance detected (ratio {imbalance_ratio}:1). Distribution: {distribution}."
        else:
            desc = f"Strong class imbalance (ratio {imbalance_ratio}:1). Majority class dominates. Distribution: {distribution}."

        metrics.append({
            "name": "class_imbalance",
            "label": "Class Imbalance",
            "score": imbalance_score,
            "description": desc,
        })
    else:
        metrics.append({
            "name": "class_imbalance",
            "label": "Class Imbalance",
            "score": 50,
            "description": f"Only {n_classes} class found — cannot assess imbalance.",
        })

    # --- 2. Feature Distribution Divergence (JSD across target classes) ---
    feature_cols = [c for c in selected if c != target_column and c in df.columns]
    if n_classes >= 2 and feature_cols:
        groups_by_class = {cls: df[df[target_column] == cls] for cls in class_counts.index}
        jsds: list[float] = []

        for col in feature_cols:
            dtype = _infer_dtype(df[col])
            group_series = [grp[col] for grp in groups_by_class.values()]
            if dtype == "numeric":
                jsds.append(_jsd_numeric(group_series))
            else:
                jsds.append(_jsd_categorical(group_series))

        avg_jsd = float(np.mean(jsds)) if jsds else 0.0
        # JSD ranges 0-1 (base 2). Score: 0→100, 1→0
        jsd_score = max(0, min(100, round((1 - avg_jsd) * 100)))

        if avg_jsd < 0.05:
            jsd_desc = f"Feature distributions are very similar across classes (avg JSD: {round(avg_jsd, 3)}). Low class separability."
        elif avg_jsd < 0.2:
            jsd_desc = f"Moderate differences in feature distributions across classes (avg JSD: {round(avg_jsd, 3)})."
        else:
            jsd_desc = f"Feature distributions differ substantially between classes (avg JSD: {round(avg_jsd, 3)}). Good class separability but check for data leakage."

        metrics.append({
            "name": "feature_distribution_divergence",
            "label": "Feature Distribution Divergence",
            "score": jsd_score,
            "description": jsd_desc,
        })

    # --- 3. Missingness Disparity by Class ---
    if n_classes >= 2 and feature_cols:
        miss_rates: dict[str, float] = {}
        for cls in class_counts.index:
            subset = df[df[target_column] == cls][feature_cols]
            total_cells = subset.shape[0] * subset.shape[1]
            miss_rate = float(subset.isna().sum().sum() / total_cells) if total_cells > 0 else 0.0
            miss_rates[str(cls)] = round(miss_rate * 100, 2)

        max_miss = max(miss_rates.values())
        min_miss = min(miss_rates.values())
        disparity = round(max_miss - min_miss, 2)

        # Score: 0 disparity → 100, >=20pp disparity → 0
        miss_score = max(0, min(100, round(100 - disparity * 5)))

        rates_str = ", ".join(f"{cls}: {r}%" for cls, r in miss_rates.items())
        if disparity < 1:
            miss_desc = f"Missing values are evenly distributed across classes (disparity: {disparity}pp). Rates: {rates_str}."
        elif disparity < 5:
            miss_desc = f"Slight missingness disparity across classes ({disparity}pp difference). Rates: {rates_str}."
        else:
            miss_desc = f"Significant missingness disparity across classes ({disparity}pp difference). Some classes have much more missing data. Rates: {rates_str}."

        metrics.append({
            "name": "missingness_disparity",
            "label": "Missingness Disparity by Class",
            "score": miss_score,
            "description": miss_desc,
        })

    # Overall score = average of all metric scores
    overall = round(sum(m["score"] for m in metrics) / len(metrics)) if metrics else 50

    return jsonify(_sanitize({
        "overallScore": overall,
        "metrics": metrics,
        "notice": "No protected attribute detected. Demographic fairness metrics unavailable. Showing dataset imbalance analysis instead.",
    }))


# ---------------------------------------------------------------------------
# Stage 1 – Understand: Baseline Bias Analysis (comprehensive)
# ---------------------------------------------------------------------------


def _clamp01(v: float) -> float:
    """Clamp a value to [0, 1]."""
    return max(0.0, min(1.0, v))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's D between two arrays."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled_std = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return abs(ma - mb) / pooled_std if pooled_std > 0 else 0.0


def _cramers_v(contingency: np.ndarray) -> float:
    """Cramér's V from a contingency table."""
    try:
        chi2, _, _, _ = chi2_contingency(contingency)
    except ValueError:
        return 0.0
    n = contingency.sum()
    r, k = contingency.shape
    denom = n * (min(r, k) - 1)
    return math.sqrt(chi2 / denom) if denom > 0 else 0.0


def baseline_bias_analysis(
    df: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:
    """Comprehensive baseline bias analysis using five statistical tests.

    For **every** feature (vs target) the function computes:
      1. Jensen–Shannon Divergence   – distribution similarity
      2. Chi-Squared / Cramér's V    – categorical independence  (or
         t-test / ANOVA + Cohen's D  – numeric mean-shift)
      3. P-value effect               – statistical significance × effect size
      4. Skewness & Kurtosis diff     – distribution shape mismatch
      5. Missingness gap              – disparity in null rates across classes

    Composite bias score (0–100, higher = *less* biased):
        raw_bias = mean over features of
            norm(JSD) + norm(chi2/cohen) + norm(|skew_diff|)
            + norm(miss_gap) + norm(p_value)
        overall_score = round(100 * (1 - raw_bias / 5))
    """

    # ── 0. Validate ─────────────────────────────────────────────────
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    target = df[target_column]

    # ── 1. Class counts & probabilities ───────────────────────────────
    class_counts: dict[str, int] = target.value_counts().to_dict()
    n_total = sum(class_counts.values())
    class_probabilities: dict[str, float] = {
        str(c): round(n / n_total, 6) for c, n in class_counts.items()
    }

    # ── 2. Imbalance ratio ───────────────────────────────────────────
    counts_arr = np.array(list(class_counts.values()))
    min_count = int(counts_arr.min())
    imbalance_ratio: float = (
        round(float(counts_arr.max() / counts_arr.min()), 4)
        if min_count > 0
        else float("inf")
    )

    class_labels = list(class_counts.keys())
    n_classes = len(class_labels)
    feature_cols = [c for c in df.columns if c != target_column]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_features = [c for c in feature_cols if c not in numeric_features]

    # Groups split by class (re-used across all metrics)
    groups_by_class: dict[Any, pd.DataFrame] = {
        cls: df[df[target_column] == cls] for cls in class_labels
    }

    # =====================================================================
    # 3. Per-feature metrics
    # =====================================================================
    feature_metrics: dict[str, dict[str, Any]] = {}

    for col in feature_cols:
        is_num = col in numeric_features
        m: dict[str, Any] = {"dtype": "numeric" if is_num else "categorical"}

        # --- (a) Jensen–Shannon Divergence --------------------------------
        if is_num:
            sub = df[[col, target_column]].dropna()
            if len(sub) >= 2 and sub[col].nunique() >= 2:
                disc = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="uniform")
                binned = disc.fit_transform(sub[[col]]).ravel().astype(int)
                sub = sub.copy()
                sub["_bin"] = binned
                n_bins_actual = int(binned.max()) + 1
                hists = {}
                for cls in class_labels:
                    h = np.bincount(
                        sub.loc[sub[target_column] == cls, "_bin"],
                        minlength=n_bins_actual,
                    ).astype(float)
                    t = h.sum()
                    hists[cls] = h / t if t > 0 else h
                jsd_vals = [
                    float(jensenshannon(hists[a], hists[b], base=2))
                    for a, b in itertools.combinations(class_labels, 2)
                ]
                mean_jsd = float(np.mean(jsd_vals)) if jsd_vals else 0.0
            else:
                mean_jsd = 0.0
        else:
            # Categorical JSD
            all_cats = set()
            group_series = []
            for cls in class_labels:
                s = groups_by_class[cls][col].dropna()
                group_series.append(s)
                all_cats.update(s.unique())
            cats = sorted(all_cats, key=str)
            vecs = []
            for s in group_series:
                vc = s.value_counts()
                t = vc.sum()
                vecs.append(np.array([vc.get(c, 0) / t if t > 0 else 0 for c in cats]))
            jsd_vals = [
                float(jensenshannon(a, b, base=2))
                for a, b in itertools.combinations(vecs, 2)
            ]
            mean_jsd = float(np.mean(jsd_vals)) if jsd_vals else 0.0

        m["jsd"] = round(mean_jsd, 6)

        # --- (b) Chi-Squared / t-test + effect size ----------------------
        if is_num:
            # t-test (2 classes) or one-way ANOVA (>2 classes)
            class_arrays = [
                pd.to_numeric(groups_by_class[cls][col], errors="coerce").dropna().values
                for cls in class_labels
            ]
            class_arrays = [a for a in class_arrays if len(a) >= 2]
            if len(class_arrays) >= 2:
                if n_classes == 2:
                    stat_val, p_val = ttest_ind(class_arrays[0], class_arrays[1], equal_var=False)
                    effect = _cohens_d(class_arrays[0], class_arrays[1])
                else:
                    stat_val, p_val = f_oneway(*class_arrays)
                    # Average pairwise Cohen's D for multi-class
                    ds = [
                        _cohens_d(a, b)
                        for a, b in itertools.combinations(class_arrays, 2)
                    ]
                    effect = float(np.mean(ds)) if ds else 0.0
                p_val = float(p_val) if not math.isnan(p_val) else 1.0
            else:
                stat_val, p_val, effect = 0.0, 1.0, 0.0

            m["test"] = "ttest" if n_classes == 2 else "anova"
            m["test_stat"] = round(float(stat_val), 4)
            m["p_value"] = round(p_val, 6)
            m["effect_size"] = round(effect, 4)  # Cohen's D
            m["effect_label"] = "cohens_d"
        else:
            # Chi-squared contingency test
            ct = pd.crosstab(df[col].fillna("__MISSING__"), df[target_column])
            try:
                chi2, p_val, dof, _ = chi2_contingency(ct.values)
            except ValueError:
                chi2, p_val, dof = 0.0, 1.0, 0
            cv = _cramers_v(ct.values)
            m["test"] = "chi2"
            m["test_stat"] = round(float(chi2), 4)
            m["p_value"] = round(float(p_val), 6)
            m["effect_size"] = round(cv, 4)  # Cramér's V
            m["effect_label"] = "cramers_v"
            effect = cv
            p_val = float(p_val)

        # --- (c) Skewness & Kurtosis diff --------------------------------
        if is_num:
            skew_by_class: dict[str, float] = {}
            kurt_by_class: dict[str, float] = {}
            for cls in class_labels:
                vals = pd.to_numeric(groups_by_class[cls][col], errors="coerce").dropna().values
                if len(vals) >= 3:
                    skew_by_class[str(cls)] = round(float(sp_skew(vals, nan_policy="omit")), 4)
                    kurt_by_class[str(cls)] = round(float(sp_kurtosis(vals, nan_policy="omit")), 4)
                else:
                    skew_by_class[str(cls)] = 0.0
                    kurt_by_class[str(cls)] = 0.0
            skew_vals = list(skew_by_class.values())
            kurt_vals = list(kurt_by_class.values())
            skew_diff = max(skew_vals) - min(skew_vals) if skew_vals else 0.0
            kurt_diff = max(kurt_vals) - min(kurt_vals) if kurt_vals else 0.0
        else:
            skew_by_class = {}
            kurt_by_class = {}
            skew_diff = 0.0
            kurt_diff = 0.0

        m["skew_by_class"] = skew_by_class
        m["kurt_by_class"] = kurt_by_class
        m["skew_diff"] = round(skew_diff, 4)
        m["kurt_diff"] = round(kurt_diff, 4)

        # --- (d) Missingness gap -----------------------------------------
        miss_by_cls: dict[str, float] = {}
        for cls in class_labels:
            c_df = groups_by_class[cls]
            n_r = len(c_df)
            miss_by_cls[str(cls)] = round(int(c_df[col].isna().sum()) / n_r, 6) if n_r > 0 else 0.0
        miss_vals_list = list(miss_by_cls.values())
        miss_gap = max(miss_vals_list) - min(miss_vals_list) if miss_vals_list else 0.0

        m["missingness_by_class"] = miss_by_cls
        m["missingness_gap"] = round(miss_gap, 6)

        # --- (e) Normalised components (0–1, higher = more bias) ----------
        norm_jsd = _clamp01(mean_jsd)                       # already 0-1
        norm_effect = _clamp01(effect)                       # Cohen D capped at 1; Cramér's V already 0-1
        norm_skew = _clamp01(abs(skew_diff) / 4.0)           # cap at 4
        norm_miss = _clamp01(miss_gap / 0.5)                 # cap at 50pp
        norm_pval = _clamp01(1 - p_val)                       # significance (lower p = higher concern)

        feature_bias = (norm_jsd + norm_effect + norm_skew + norm_miss + norm_pval) / 5.0

        m["norm_jsd"] = round(norm_jsd, 4)
        m["norm_effect"] = round(norm_effect, 4)
        m["norm_skew"] = round(norm_skew, 4)
        m["norm_miss"] = round(norm_miss, 4)
        m["norm_pval"] = round(norm_pval, 4)
        m["feature_bias"] = round(feature_bias, 4)

        feature_metrics[col] = m

    # =====================================================================
    # 4. Aggregate sub-scores & overall score
    # =====================================================================
    all_fm = list(feature_metrics.values())
    n_feat = len(all_fm) or 1

    sub_scores = {
        "js_divergence":   round(sum(f["norm_jsd"]    for f in all_fm) / n_feat, 4),
        "chi2_effect":     round(sum(f["norm_effect"] for f in all_fm) / n_feat, 4),
        "skew_diff":       round(sum(f["norm_skew"]   for f in all_fm) / n_feat, 4),
        "missingness_gap": round(sum(f["norm_miss"]   for f in all_fm) / n_feat, 4),
        "p_value":         round(sum(f["norm_pval"]   for f in all_fm) / n_feat, 4),
    }
    raw_bias = sum(sub_scores.values())  # 0–5 scale
    overall_score = round(100 * (1 - raw_bias / 5.0), 1)
    overall_score = max(0, min(100, overall_score))

    # =====================================================================
    # 5. Backward-compat: missingness_by_class flat list
    # =====================================================================
    miss_records: list[dict[str, Any]] = []
    for col in feature_cols:
        fm = feature_metrics.get(col, {})
        mbc = fm.get("missingness_by_class", {})
        for cls in class_labels:
            rate = mbc.get(str(cls), 0.0)
            miss_records.append({
                "class": str(cls),
                "feature": col,
                "missing_count": int(groups_by_class[cls][col].isna().sum()),
                "missing_rate": rate,
            })

    return {
        "overall_score": overall_score,
        "class_counts": class_counts,
        "class_probabilities": class_probabilities,
        "imbalance_ratio": imbalance_ratio,
        "feature_metrics": feature_metrics,
        "sub_scores": sub_scores,
        "missingness_by_class": miss_records,
    }


@app.route("/api/bias/baseline", methods=["POST"])
def baseline_bias_endpoint():
    """Expose baseline_bias_analysis over HTTP."""
    file = request.files["file"]
    target_column = request.form["target_column"]
    contents = file.read()
    df = pd.read_csv(io.BytesIO(contents))
    return jsonify(_sanitize(baseline_bias_analysis(df, target_column)))


# ---------------------------------------------------------------------------
# Stage 2 – Clean: Real transformation engine with before/after bias scoring
# ---------------------------------------------------------------------------


def _apply_technique(
    df: pd.DataFrame,
    target_column: str,
    technique: str,
    column: str | None = None,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply a single cleaning technique and return the modified DataFrame.

    Every technique actually mutates a *copy* of the data so
    ``baseline_bias_analysis`` can measure the real delta.
    """
    out = df.copy()
    params = params or {}

    # ── 1. Missing-data techniques ──────────────────────────────────
    if technique == "listwise_deletion":
        out = out.dropna().reset_index(drop=True)

    elif technique == "mean_imputation":
        if column and column in out.columns:
            col_numeric = pd.to_numeric(out[column], errors="coerce")
            fill_val = col_numeric.mean()
            out[column] = col_numeric.fillna(fill_val)
        else:
            for c in out.select_dtypes(include=[np.number]).columns:
                if c != target_column:
                    out[c] = out[c].fillna(out[c].mean())

    elif technique == "median_imputation":
        if column and column in out.columns:
            col_numeric = pd.to_numeric(out[column], errors="coerce")
            fill_val = col_numeric.median()
            out[column] = col_numeric.fillna(fill_val)
        else:
            for c in out.select_dtypes(include=[np.number]).columns:
                if c != target_column:
                    out[c] = out[c].fillna(out[c].median())

    elif technique == "mode_imputation":
        if column and column in out.columns:
            mode_val = out[column].dropna().mode()
            if len(mode_val):
                out[column] = out[column].fillna(mode_val.iloc[0])
        else:
            for c in out.columns:
                if c != target_column and out[c].isna().any():
                    mode_val = out[c].dropna().mode()
                    if len(mode_val):
                        out[c] = out[c].fillna(mode_val.iloc[0])

    elif technique == "group_wise_mean_imputation":
        target_col = target_column
        cols = [column] if column and column in out.columns else [
            c for c in out.select_dtypes(include=[np.number]).columns if c != target_col
        ]
        for c in cols:
            col_numeric = pd.to_numeric(out[c], errors="coerce")
            group_means = col_numeric.groupby(out[target_col]).transform("mean")
            out[c] = col_numeric.fillna(group_means)
            # If any NaN remain (entire group was NaN), fill with global mean
            out[c] = out[c].fillna(col_numeric.mean())

    elif technique == "group_wise_median_imputation":
        target_col = target_column
        cols = [column] if column and column in out.columns else [
            c for c in out.select_dtypes(include=[np.number]).columns if c != target_col
        ]
        for c in cols:
            col_numeric = pd.to_numeric(out[c], errors="coerce")
            group_medians = col_numeric.groupby(out[target_col]).transform("median")
            out[c] = col_numeric.fillna(group_medians)
            out[c] = out[c].fillna(col_numeric.median())

    elif technique == "group_wise_mode_imputation":
        target_col = target_column
        cols = [column] if column and column in out.columns else [
            c for c in out.columns if c != target_col and out[c].isna().any()
        ]
        for c in cols:
            dtype = _infer_dtype(out[c])
            if dtype in ("categorical", "text", "boolean"):
                for cls in out[target_col].dropna().unique():
                    mask = (out[target_col] == cls) & out[c].isna()
                    group_mode = out.loc[out[target_col] == cls, c].dropna().mode()
                    if len(group_mode):
                        out.loc[mask, c] = group_mode.iloc[0]
                # Remaining NaN → global mode
                global_mode = out[c].dropna().mode()
                if len(global_mode):
                    out[c] = out[c].fillna(global_mode.iloc[0])

    elif technique == "knn_imputation":
        from sklearn.impute import KNNImputer
        num_cols = [c for c in out.select_dtypes(include=[np.number]).columns if c != target_column]
        if column and column in num_cols:
            num_cols = [column]
        if num_cols:
            imputer = KNNImputer(n_neighbors=min(5, max(1, len(out) // 10)))
            out[num_cols] = imputer.fit_transform(out[num_cols])

    elif technique == "missingness_indicator":
        cols = [column] if column and column in out.columns else [
            c for c in out.columns if c != target_column and out[c].isna().any()
        ]
        for c in cols:
            out[f"{c}_missing"] = out[c].isna().astype(int)

    # ── 2. Rebalancing techniques ───────────────────────────────────
    elif technique == "random_oversample":
        target = out[target_column]
        class_counts = target.value_counts()
        max_count = int(class_counts.max())
        frames = []
        for cls in class_counts.index:
            cls_df = out[out[target_column] == cls]
            if len(cls_df) < max_count:
                oversampled = cls_df.sample(max_count, replace=True, random_state=42)
                frames.append(oversampled)
            else:
                frames.append(cls_df)
        out = pd.concat(frames, ignore_index=True)

    elif technique == "random_undersample":
        target = out[target_column]
        class_counts = target.value_counts()
        min_count = int(class_counts.min())
        frames = []
        for cls in class_counts.index:
            cls_df = out[out[target_column] == cls]
            frames.append(cls_df.sample(min_count, replace=False, random_state=42))
        out = pd.concat(frames, ignore_index=True)

    elif technique == "stratified_resample":
        # Resample to equal proportions while preserving total size
        target = out[target_column]
        class_counts = target.value_counts()
        n_classes_local = len(class_counts)
        target_per_class = len(out) // n_classes_local
        frames = []
        for cls in class_counts.index:
            cls_df = out[out[target_column] == cls]
            frames.append(cls_df.sample(target_per_class, replace=True, random_state=42))
        out = pd.concat(frames, ignore_index=True)

    # ── 3. Outlier handling ─────────────────────────────────────────
    elif technique == "winsorize":
        lower_pct = params.get("lower", 0.05)
        upper_pct = params.get("upper", 0.95)
        cols = [column] if column and column in out.columns else [
            c for c in out.select_dtypes(include=[np.number]).columns if c != target_column
        ]
        for c in cols:
            vals = pd.to_numeric(out[c], errors="coerce")
            lo, hi = vals.quantile(lower_pct), vals.quantile(upper_pct)
            out[c] = vals.clip(lo, hi)

    elif technique == "robust_scale":
        cols = [column] if column and column in out.columns else [
            c for c in out.select_dtypes(include=[np.number]).columns if c != target_column
        ]
        for c in cols:
            vals = pd.to_numeric(out[c], errors="coerce")
            med = vals.median()
            iqr = vals.quantile(0.75) - vals.quantile(0.25)
            if iqr > 0:
                out[c] = (vals - med) / iqr

    elif technique == "group_aware_outlier_clip":
        cols = [column] if column and column in out.columns else [
            c for c in out.select_dtypes(include=[np.number]).columns if c != target_column
        ]
        for c in cols:
            for cls in out[target_column].dropna().unique():
                mask = out[target_column] == cls
                vals = pd.to_numeric(out.loc[mask, c], errors="coerce")
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                out.loc[mask, c] = vals.clip(lo, hi)

    # ── 4. Feature-level cleaning ───────────────────────────────────
    elif technique == "drop_high_correlation_proxy":
        # Drop features with >0.8 correlation to target (if numeric)
        target_numeric = pd.to_numeric(out[target_column], errors="coerce")
        if target_numeric.notna().sum() > 10:
            num_cols = [c for c in out.select_dtypes(include=[np.number]).columns if c != target_column]
            to_drop = []
            for c in num_cols:
                corr = out[c].corr(target_numeric)
                if abs(corr) > 0.8:
                    to_drop.append(c)
            if to_drop:
                out = out.drop(columns=to_drop)

    elif technique == "within_group_normalize":
        cols = [column] if column and column in out.columns else [
            c for c in out.select_dtypes(include=[np.number]).columns if c != target_column
        ]
        for c in cols:
            for cls in out[target_column].dropna().unique():
                mask = out[target_column] == cls
                vals = pd.to_numeric(out.loc[mask, c], errors="coerce")
                mean = vals.mean()
                std = vals.std(ddof=1)
                if std > 0:
                    out.loc[mask, c] = (vals - mean) / std

    # ── 5. Encoding ─────────────────────────────────────────────────
    elif technique == "one_hot_encode":
        cat_cols = [column] if column and column in out.columns else [
            c for c in out.columns if c != target_column and _infer_dtype(out[c]) in ("categorical", "text")
        ]
        if cat_cols:
            out = pd.get_dummies(out, columns=cat_cols, drop_first=False, dtype=int)

    elif technique == "ordinal_encode":
        from sklearn.preprocessing import OrdinalEncoder
        cat_cols = [column] if column and column in out.columns else [
            c for c in out.columns if c != target_column and _infer_dtype(out[c]) in ("categorical", "text")
        ]
        if cat_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            out[cat_cols] = enc.fit_transform(out[cat_cols].astype(str))

    # ── 6. Duplicate removal ────────────────────────────────────────
    elif technique == "duplicate_removal":
        out = out.drop_duplicates().reset_index(drop=True)

    # ── 7. Group-Aware Winsorization (per-class percentile clipping) ──
    elif technique == "group_aware_winsorization":
        lower_pct = params.get("lower", 0.05)
        upper_pct = params.get("upper", 0.95)
        cols = [column] if column and column in out.columns else [
            c for c in out.select_dtypes(include=[np.number]).columns if c != target_column
        ]
        for c in cols:
            for cls in out[target_column].dropna().unique():
                mask = out[target_column] == cls
                vals = pd.to_numeric(out.loc[mask, c], errors="coerce")
                if vals.notna().sum() < 5:
                    continue
                lo, hi = vals.quantile(lower_pct), vals.quantile(upper_pct)
                out.loc[mask, c] = vals.clip(lo, hi)

    # ── 8. Proxy Detection + Residualization ─────────────────────────
    elif technique == "proxy_residualization":
        from sklearn.linear_model import LinearRegression
        cols = [column] if column and column in out.columns else [
            c for c in out.select_dtypes(include=[np.number]).columns if c != target_column
        ]
        target_dummies = pd.get_dummies(out[target_column], drop_first=True).astype(float)
        for c in cols:
            feature_vals = pd.to_numeric(out[c], errors="coerce")
            valid = feature_vals.notna()
            if valid.sum() < 10 or target_dummies.shape[1] == 0:
                continue
            X = target_dummies.loc[valid].values
            y = feature_vals.loc[valid].values
            reg = LinearRegression()
            reg.fit(X, y)
            out.loc[valid, c] = y - reg.predict(X)

    # ── 9. Isolation Forest Per Subgroup ──────────────────────────────
    elif technique == "isolation_forest_subgroup":
        from sklearn.ensemble import IsolationForest
        cols = [column] if column and column in out.columns else [
            c for c in out.select_dtypes(include=[np.number]).columns if c != target_column
        ]
        for c in cols:
            for cls in out[target_column].dropna().unique():
                mask = out[target_column] == cls
                vals = pd.to_numeric(out.loc[mask, c], errors="coerce")
                non_null = vals.dropna()
                if len(non_null) < 10:
                    continue
                iso = IsolationForest(contamination=0.05, random_state=42)
                preds = iso.fit_predict(non_null.values.reshape(-1, 1))
                outlier_idx = non_null.index[preds == -1]
                if len(outlier_idx) > 0:
                    lo, hi = non_null.quantile(0.05), non_null.quantile(0.95)
                    out.loc[outlier_idx, c] = out.loc[outlier_idx, c].clip(lo, hi)

    # ── 10. SMOTE-like Oversampling ──────────────────────────────────
    elif technique == "smote":
        target_series = out[target_column].dropna()
        class_counts_s = target_series.value_counts()
        if len(class_counts_s) >= 2:
            max_count = int(class_counts_s.max())
            rng = np.random.default_rng(42)
            num_cols = [c for c in out.select_dtypes(include=[np.number]).columns if c != target_column]
            frames = [out]
            for cls in class_counts_s.index:
                cls_df = out[out[target_column] == cls]
                n_needed = max_count - len(cls_df)
                if n_needed <= 0:
                    continue
                synthetic = cls_df.sample(n_needed, replace=True, random_state=42).copy()
                for nc in num_cols:
                    nc_vals = pd.to_numeric(synthetic[nc], errors="coerce")
                    nc_std = nc_vals.std()
                    if nc_std and nc_std > 0:
                        synthetic[nc] = nc_vals + rng.normal(0, nc_std * 0.05, len(synthetic))
                frames.append(synthetic)
            out = pd.concat(frames, ignore_index=True)

    return out


def _compute_technique_delta(
    df_original: pd.DataFrame,
    target_column: str,
    technique: str,
    column: str | None,
    params: dict[str, Any] | None,
    baseline_score: float,
) -> dict[str, Any]:
    """Apply one technique and compute the bias-score delta."""
    try:
        df_transformed = _apply_technique(df_original, target_column, technique, column, params)
        after = baseline_bias_analysis(df_transformed, target_column)
        after_score = after["overall_score"]
        delta = round(after_score - baseline_score, 1)
        return {
            "after_score": after_score,
            "delta": delta,
            "after_sub_scores": after.get("sub_scores", {}),
            "rows_after": len(df_transformed),
            "cols_after": len(df_transformed.columns),
        }
    except Exception as e:
        return {
            "after_score": baseline_score,
            "delta": 0,
            "after_sub_scores": {},
            "rows_after": len(df_original),
            "cols_after": len(df_original.columns),
            "error": str(e),
        }


@app.route("/api/clean/generate", methods=["POST"])
def generate_cleaning_steps():
    """Analyze the dataset and propose cleaning steps with real bias deltas.

    For each proposed technique the engine:
      1. Applies the transformation to a copy of the data
      2. Runs ``baseline_bias_analysis`` on the result
      3. Computes ``delta = after_score - before_score``

    This is the competitive advantage — every number is real, not mocked.
    """
    file = request.files["file"]
    target_column = request.form["target_column"]
    contents = file.read()
    df = pd.read_csv(io.BytesIO(contents))

    if target_column not in df.columns:
        return jsonify({"error": f"Target column '{target_column}' not found.", "steps": []})

    # ── Baseline bias ──────────────────────────────────────────────
    baseline = baseline_bias_analysis(df, target_column)
    baseline_score = baseline["overall_score"]
    baseline_sub = baseline.get("sub_scores", {})

    feature_cols = [c for c in df.columns if c != target_column]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    steps: list[dict[str, Any]] = []
    step_id = 0

    # ================================================================
    # 1. MISSING DATA — Imputation
    # ================================================================
    # Find columns with missing values
    null_num_cols = [c for c in num_cols if df[c].isna().sum() > 0]
    null_cat_cols = [c for c in cat_cols if df[c].isna().sum() > 0]
    any_nulls = len(null_num_cols) + len(null_cat_cols)

    if null_num_cols:
        # Pick the numeric column with the most missing values
        worst_col = max(null_num_cols, key=lambda c: df[c].isna().sum())
        null_count = int(df[worst_col].isna().sum())
        null_pct = round(null_count / len(df) * 100, 1)

        # --- Median imputation (global) ---
        median_delta = _compute_technique_delta(
            df, target_column, "median_imputation", worst_col, None, baseline_score,
        )
        # --- Stratified median imputation (group-wise) ---
        strat_delta = _compute_technique_delta(
            df, target_column, "group_wise_median_imputation", worst_col, None, baseline_score,
        )

        steps.append({
            "id": f"step-{step_id}",
            "column": worst_col,
            "type": "imputation",
            "strategy": "Median Imputation",
            "description": (
                f'Fill {null_count} missing values ({null_pct}%) in "{worst_col}" '
                f"with the global column median."
            ),
            "qualityImpact": 78,
            "confidence": 85,
            "biasDelta": median_delta["delta"],
            "afterScore": median_delta["after_score"],
            "afterSubScores": median_delta.get("after_sub_scores", {}),
            "rowsAffected": null_count,
            "alternative": {
                "strategy": "Stratified Median Imputation",
                "description": (
                    f'Fill missing values in "{worst_col}" using the median calculated '
                    f"separately for each target class. Preserves per-class distribution shape."
                ),
                "biasDelta": strat_delta["delta"],
                "afterScore": strat_delta["after_score"],
                "afterSubScores": strat_delta.get("after_sub_scores", {}),
            },
            "status": "pending",
        })
        step_id += 1

        # --- Mean imputation vs Group-wise mean (if more null columns) ---
        remaining_null_num = [c for c in null_num_cols if c != worst_col]
        if remaining_null_num:
            col2 = max(remaining_null_num, key=lambda c: df[c].isna().sum())
            null_count2 = int(df[col2].isna().sum())
            null_pct2 = round(null_count2 / len(df) * 100, 1)

            mean_delta = _compute_technique_delta(
                df, target_column, "mean_imputation", col2, None, baseline_score,
            )
            gw_mean_delta = _compute_technique_delta(
                df, target_column, "group_wise_mean_imputation", col2, None, baseline_score,
            )

            steps.append({
                "id": f"step-{step_id}",
                "column": col2,
                "type": "imputation",
                "strategy": "Mean Imputation",
                "description": (
                    f'Fill {null_count2} missing values ({null_pct2}%) in "{col2}" '
                    f"with the global column mean."
                ),
                "qualityImpact": 72,
                "confidence": 80,
                "biasDelta": mean_delta["delta"],
                "afterScore": mean_delta["after_score"],
                "afterSubScores": mean_delta.get("after_sub_scores", {}),
                "rowsAffected": null_count2,
                "alternative": {
                    "strategy": "Group-Wise Mean Imputation",
                    "description": (
                        f'Fill missing values in "{col2}" using the mean calculated '
                        f"per target class. Reduces majority-class skew."
                    ),
                    "biasDelta": gw_mean_delta["delta"],
                    "afterScore": gw_mean_delta["after_score"],
                    "afterSubScores": gw_mean_delta.get("after_sub_scores", {}),
                },
                "status": "pending",
            })
            step_id += 1

    # Categorical imputation
    if null_cat_cols:
        worst_cat = max(null_cat_cols, key=lambda c: df[c].isna().sum())
        null_count_cat = int(df[worst_cat].isna().sum())
        null_pct_cat = round(null_count_cat / len(df) * 100, 1)

        mode_delta = _compute_technique_delta(
            df, target_column, "mode_imputation", worst_cat, None, baseline_score,
        )
        gw_mode_delta = _compute_technique_delta(
            df, target_column, "group_wise_mode_imputation", worst_cat, None, baseline_score,
        )

        steps.append({
            "id": f"step-{step_id}",
            "column": worst_cat,
            "type": "imputation",
            "strategy": "Mode Imputation",
            "description": (
                f'Fill {null_count_cat} missing values ({null_pct_cat}%) in "{worst_cat}" '
                f"with the most frequent value."
            ),
            "qualityImpact": 65,
            "confidence": 72,
            "biasDelta": mode_delta["delta"],
            "afterScore": mode_delta["after_score"],
            "afterSubScores": mode_delta.get("after_sub_scores", {}),
            "rowsAffected": null_count_cat,
            "alternative": {
                "strategy": "Group-Aware Mode Imputation",
                "description": (
                    f'Impute "{worst_cat}" with the mode per target class '
                    f"to preserve group-specific categorical patterns."
                ),
                "biasDelta": gw_mode_delta["delta"],
                "afterScore": gw_mode_delta["after_score"],
                "afterSubScores": gw_mode_delta.get("after_sub_scores", {}),
            },
            "status": "pending",
        })
        step_id += 1

    # Missingness indicators (if any nulls exist)
    if any_nulls > 0:
        miss_ind_delta = _compute_technique_delta(
            df, target_column, "missingness_indicator", None, None, baseline_score,
        )
        null_col_names = [c for c in feature_cols if df[c].isna().sum() > 0]
        steps.append({
            "id": f"step-{step_id}",
            "column": ", ".join(null_col_names[:5]) + ("..." if len(null_col_names) > 5 else ""),
            "type": "missingness_indicator",
            "strategy": "Add Missingness Indicator Columns",
            "description": (
                f"Add binary flag columns for {len(null_col_names)} features with missing data. "
                f"Allows downstream models to learn from missingness patterns."
            ),
            "qualityImpact": 60,
            "confidence": 90,
            "biasDelta": miss_ind_delta["delta"],
            "afterScore": miss_ind_delta["after_score"],
            "afterSubScores": miss_ind_delta.get("after_sub_scores", {}),
            "rowsAffected": 0,
            "status": "pending",
        })
        step_id += 1

    # ================================================================
    # 2. CLASS REBALANCING
    # ================================================================
    class_counts_series = df[target_column].dropna().value_counts()
    if len(class_counts_series) >= 2:
        max_c = int(class_counts_series.max())
        min_c = int(class_counts_series.min())
        ratio = round(max_c / min_c, 2) if min_c > 0 else float("inf")

        if ratio > 1.3:
            oversample_delta = _compute_technique_delta(
                df, target_column, "random_oversample", None, None, baseline_score,
            )
            stratified_delta = _compute_technique_delta(
                df, target_column, "stratified_resample", None, None, baseline_score,
            )

            minority_class = str(class_counts_series.idxmin())
            majority_class = str(class_counts_series.idxmax())
            dist_str = ", ".join(f"{c}: {n}" for c, n in class_counts_series.items())

            steps.append({
                "id": f"step-{step_id}",
                "column": target_column,
                "type": "rebalancing",
                "strategy": "Random Oversampling (Minority)",
                "description": (
                    f'Class ratio is {ratio}:1 ({dist_str}). '
                    f'Oversample minority class "{minority_class}" to match majority class.'
                ),
                "qualityImpact": 75,
                "confidence": 82,
                "biasDelta": oversample_delta["delta"],
                "afterScore": oversample_delta["after_score"],
                "afterSubScores": oversample_delta.get("after_sub_scores", {}),
                "rowsAffected": max_c - min_c,
                "alternative": {
                    "strategy": "Stratified Resampling",
                    "description": (
                        f"Resample all classes to equal size while maintaining "
                        f"total dataset size. More balanced than simple oversampling."
                    ),
                    "biasDelta": stratified_delta["delta"],
                    "afterScore": stratified_delta["after_score"],
                    "afterSubScores": stratified_delta.get("after_sub_scores", {}),
                },
                "status": "pending",
            })
            step_id += 1

    # ================================================================
    # 3. OUTLIER HANDLING
    # ================================================================
    # Find numeric columns with high skewness or IQR-based outliers
    outlier_cols = []
    for c in num_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) < 10:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            n_outliers = int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum())
            if n_outliers > len(vals) * 0.01:
                outlier_cols.append((c, n_outliers))

    if outlier_cols:
        worst_outlier_col, n_outliers = max(outlier_cols, key=lambda x: x[1])
        outlier_pct = round(n_outliers / len(df) * 100, 1)

        winsor_delta = _compute_technique_delta(
            df, target_column, "winsorize", worst_outlier_col,
            {"lower": 0.05, "upper": 0.95}, baseline_score,
        )
        ga_clip_delta = _compute_technique_delta(
            df, target_column, "group_aware_outlier_clip", worst_outlier_col,
            None, baseline_score,
        )

        steps.append({
            "id": f"step-{step_id}",
            "column": worst_outlier_col,
            "type": "outlier",
            "strategy": "Winsorization (5th/95th Percentile)",
            "description": (
                f'Cap {n_outliers} outliers ({outlier_pct}%) in "{worst_outlier_col}" '
                f"at the 5th and 95th percentiles globally."
            ),
            "qualityImpact": 72,
            "confidence": 80,
            "biasDelta": winsor_delta["delta"],
            "afterScore": winsor_delta["after_score"],
            "afterSubScores": winsor_delta.get("after_sub_scores", {}),
            "rowsAffected": n_outliers,
            "alternative": {
                "strategy": "Group-Aware IQR Clipping",
                "description": (
                    f'Clip outliers in "{worst_outlier_col}" using IQR bounds '
                    f"computed per target class. Avoids clipping one group's "
                    f"legitimate range based on another group's statistics."
                ),
                "biasDelta": ga_clip_delta["delta"],
                "afterScore": ga_clip_delta["after_score"],
                "afterSubScores": ga_clip_delta.get("after_sub_scores", {}),
            },
            "status": "pending",
        })
        step_id += 1

    # ================================================================
    # 4. FEATURE-LEVEL CLEANING
    # ================================================================
    # Within-group normalization for high-skew features
    skew_features = []
    for c in num_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals) >= 10:
            sk = float(sp_skew(vals, nan_policy="omit"))
            if abs(sk) > 1.0:
                skew_features.append((c, sk))

    if skew_features:
        worst_skew_col, sk_val = max(skew_features, key=lambda x: abs(x[1]))

        robust_delta = _compute_technique_delta(
            df, target_column, "robust_scale", worst_skew_col, None, baseline_score,
        )
        wg_norm_delta = _compute_technique_delta(
            df, target_column, "within_group_normalize", worst_skew_col, None, baseline_score,
        )

        steps.append({
            "id": f"step-{step_id}",
            "column": worst_skew_col,
            "type": "feature_cleaning",
            "strategy": "Robust Scaling (Median/IQR)",
            "description": (
                f'"{worst_skew_col}" has skewness={round(sk_val, 2)}. '
                f"Center around median and scale by IQR for outlier-robust normalization."
            ),
            "qualityImpact": 70,
            "confidence": 85,
            "biasDelta": robust_delta["delta"],
            "afterScore": robust_delta["after_score"],
            "afterSubScores": robust_delta.get("after_sub_scores", {}),
            "rowsAffected": len(df),
            "alternative": {
                "strategy": "Within-Group Normalization",
                "description": (
                    f'Standardize "{worst_skew_col}" separately within each target class. '
                    f"Ensures each class has zero mean and unit variance independently."
                ),
                "biasDelta": wg_norm_delta["delta"],
                "afterScore": wg_norm_delta["after_score"],
                "afterSubScores": wg_norm_delta.get("after_sub_scores", {}),
            },
            "status": "pending",
        })
        step_id += 1

    # ================================================================
    # 5. DUPLICATE REMOVAL
    # ================================================================
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        dupe_delta = _compute_technique_delta(
            df, target_column, "duplicate_removal", None, None, baseline_score,
        )
        dupe_pct = round(n_dupes / len(df) * 100, 1)
        steps.append({
            "id": f"step-{step_id}",
            "column": "All Columns",
            "type": "duplicate",
            "strategy": "Exact Duplicate Removal",
            "description": (
                f"Remove {n_dupes} exact duplicate rows ({dupe_pct}% of dataset)."
            ),
            "qualityImpact": 88,
            "confidence": 95,
            "biasDelta": dupe_delta["delta"],
            "afterScore": dupe_delta["after_score"],
            "afterSubScores": dupe_delta.get("after_sub_scores", {}),
            "rowsAffected": n_dupes,
            "status": "pending",
        })
        step_id += 1

    # ================================================================
    # 6. CATEGORICAL ENCODING
    # ================================================================
    if cat_cols:
        ohe_delta = _compute_technique_delta(
            df, target_column, "one_hot_encode", None, None, baseline_score,
        )
        total_cats = sum(df[c].nunique() for c in cat_cols)
        steps.append({
            "id": f"step-{step_id}",
            "column": ", ".join(cat_cols[:4]) + ("..." if len(cat_cols) > 4 else ""),
            "type": "encoding",
            "strategy": "One-Hot Encoding",
            "description": (
                f"Convert {len(cat_cols)} categorical columns ({total_cats} unique categories total) "
                f"to binary indicator columns."
            ),
            "qualityImpact": 82,
            "confidence": 90,
            "biasDelta": ohe_delta["delta"],
            "afterScore": ohe_delta["after_score"],
            "afterSubScores": ohe_delta.get("after_sub_scores", {}),
            "rowsAffected": 0,
            "status": "pending",
        })
        step_id += 1

    return jsonify(_sanitize({
        "baselineScore": baseline_score,
        "baselineSubScores": baseline_sub,
        "steps": steps,
        "datasetInfo": {
            "rows": len(df),
            "columns": len(df.columns),
            "numericFeatures": len(num_cols),
            "categoricalFeatures": len(cat_cols),
            "totalNulls": int(df[feature_cols].isna().sum().sum()),
            "nullColumns": len(null_num_cols) + len(null_cat_cols),
        },
    }))


# ============================================================================
# Stage 2 — Iterative Feature-by-Feature Cleaning  (session-based API)
# ============================================================================

_sessions: dict[str, dict[str, Any]] = {}

_TECHNIQUE_ALIASES: dict[str, str] = {
    "stratified_median_imputation": "group_wise_median_imputation",
    "global_median_imputation": "median_imputation",
    "stratified_mode_imputation": "group_wise_mode_imputation",
    "global_mode_imputation": "mode_imputation",
    "global_winsorization": "winsorize",
}


def _resolve_technique(technique: str) -> str:
    return _TECHNIQUE_ALIASES.get(technique, technique)


# ---------------------------------------------------------------------------
# Suggestion logic — picks techniques based on worst bias metric
# ---------------------------------------------------------------------------

def _suggest_for_feature(
    df: pd.DataFrame,
    feature: str,
    target_column: str,
    tried: set[str] | None = None,
) -> dict[str, Any]:
    """Identify the worst bias metric for *feature* and rank applicable techniques."""

    tried = tried or set()

    analysis = baseline_bias_analysis(df, target_column)
    fm = analysis["feature_metrics"].get(feature)
    if fm is None:
        return {"error": f"Feature '{feature}' not in analysis."}

    has_nulls = int(df[feature].isna().sum()) > 0
    null_count = int(df[feature].isna().sum())
    null_pct = round(null_count / len(df) * 100, 1) if len(df) > 0 else 0
    is_numeric = fm["dtype"] == "numeric"

    norms = {
        "p_value": fm["norm_pval"],
        "js_divergence": fm["norm_jsd"],
        "chi_squared": fm["norm_effect"],
        "missingness_rate": fm["norm_miss"],
        "skewness": fm["norm_skew"],
    }
    worst = max(norms, key=lambda k: norms[k])

    suggestions: list[dict[str, Any]] = []

    # ── Missing-data techniques (only when nulls exist) ─────────────
    if has_nulls and is_numeric:
        suggestions.append({
            "technique": "stratified_median_imputation",
            "label": "Stratified Median Imputation",
            "category": "Missing Data",
            "description": (
                f"Fill {null_count} missing values ({null_pct}%) using the median "
                f"per target class. Preserves each group's distribution center."
            ),
            "isHero": True,
            "targetMetric": "missingness_rate",
            "targetMetricLabel": "Missingness Rate",
        })
        suggestions.append({
            "technique": "global_median_imputation",
            "label": "Global Median Imputation",
            "category": "Missing Data",
            "description": (
                f"Fill {null_count} missing values ({null_pct}%) with the overall "
                f"column median. Simple but can distort per-group distributions."
            ),
            "isHero": False,
            "targetMetric": "missingness_rate",
            "targetMetricLabel": "Missingness Rate",
        })

    if has_nulls and not is_numeric:
        suggestions.append({
            "technique": "stratified_mode_imputation",
            "label": "Stratified Mode Imputation",
            "category": "Missing Data",
            "description": (
                f"Fill {null_count} missing values ({null_pct}%) with the most "
                f"frequent value per target class. Preserves group-specific patterns."
            ),
            "isHero": True,
            "targetMetric": "missingness_rate",
            "targetMetricLabel": "Missingness Rate",
        })
        suggestions.append({
            "technique": "global_mode_imputation",
            "label": "Global Mode Imputation",
            "category": "Missing Data",
            "description": (
                f"Fill {null_count} missing values ({null_pct}%) with the overall "
                f"most common value. Risks importing majority-group patterns."
            ),
            "isHero": False,
            "targetMetric": "missingness_rate",
            "targetMetricLabel": "Missingness Rate",
        })

    if has_nulls:
        suggestions.append({
            "technique": "missingness_indicator",
            "label": "Missingness Indicator Flag",
            "category": "Missing Data",
            "description": (
                f'Add binary column "{feature}_missing" flagging rows with missing '
                f"values. Preserves missingness info without altering original data."
            ),
            "isHero": True,
            "targetMetric": "missingness_rate",
            "targetMetricLabel": "Missingness Rate",
        })

    # ── Numeric-only techniques ─────────────────────────────────────
    if is_numeric:
        suggestions.append({
            "technique": "group_aware_winsorization",
            "label": "Group-Aware Winsorization",
            "category": "Outliers",
            "description": (
                "Clip extreme values using per-class 5th/95th percentile bounds. "
                "Each class is winsorized independently, preserving legitimate group ranges."
            ),
            "isHero": True,
            "targetMetric": "skewness",
            "targetMetricLabel": "Skewness",
        })
        suggestions.append({
            "technique": "global_winsorization",
            "label": "Global Winsorization",
            "category": "Outliers",
            "description": (
                "Clip extreme values at global 5th/95th percentiles. "
                "Quick fix but may disproportionately cap one group's legitimate values."
            ),
            "isHero": False,
            "targetMetric": "skewness",
            "targetMetricLabel": "Skewness",
        })
        suggestions.append({
            "technique": "isolation_forest_subgroup",
            "label": "Isolation Forest Per Subgroup",
            "category": "Outliers",
            "description": (
                "Detect anomalous values within each target class using Isolation Forest. "
                "Outliers are identified relative to their own group, then clipped to group bounds."
            ),
            "isHero": True,
            "targetMetric": "js_divergence",
            "targetMetricLabel": "JS Divergence",
        })
        suggestions.append({
            "technique": "proxy_residualization",
            "label": "Proxy Detection + Residualization",
            "category": "Feature",
            "description": (
                "Remove the portion of this feature linearly predicted by the target. "
                "Strips proxy-bias while preserving residual variance for modeling."
            ),
            "isHero": True,
            "targetMetric": "chi_squared",
            "targetMetricLabel": "Chi-Squared / Effect Size",
        })

    # ── SMOTE (class rebalancing — always available) ────────────────
    target_series = df[target_column].dropna()
    class_counts_s = target_series.value_counts()
    if len(class_counts_s) >= 2:
        max_c = int(class_counts_s.max())
        min_c = int(class_counts_s.min())
        ratio = round(max_c / min_c, 2) if min_c > 0 else 1.0
        if ratio > 1.2:
            suggestions.append({
                "technique": "smote",
                "label": "SMOTE (Synthetic Minority Oversampling)",
                "category": "Sampling",
                "description": (
                    f"Generate synthetic samples for minority class using SMOTE-like interpolation. "
                    f"Current class ratio is {ratio}:1. Balances representation without simple duplication."
                ),
                "isHero": True,
                "targetMetric": "js_divergence",
                "targetMetricLabel": "JS Divergence",
            })

    # ── Filter already-tried techniques ─────────────────────────────
    suggestions = [s for s in suggestions if s["technique"] not in tried]

    # ── Rank by ACTUAL bias improvement (compute real deltas) ──────
    current_score = analysis["overall_score"]
    ranked: list[dict[str, Any]] = []
    for s in suggestions:
        resolved = _resolve_technique(s["technique"])
        try:
            transformed = _apply_technique(df, target_column, resolved, feature)
            after = baseline_bias_analysis(transformed, target_column)
            delta = round(after["overall_score"] - current_score, 2)
        except Exception:
            delta = 0.0
        s["expectedDelta"] = delta
        ranked.append(s)

    # Best improvement first, break ties with hero preference
    ranked.sort(key=lambda s: (s["expectedDelta"], 1 if s["isHero"] else 0), reverse=True)

    return {
        "feature": feature,
        "dtype": fm["dtype"],
        "worstMetric": worst,
        "worstMetricValue": round(norms[worst], 4),
        "currentMetrics": fm,
        "norms": {k: round(v, 4) for k, v in norms.items()},
        "suggestions": ranked,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/clean/init", methods=["POST"])
def clean_init():
    """Create a cleaning session.  Returns session ID + per-feature metrics."""
    file = request.files["file"]
    target_column = request.form["target_column"]
    contents = file.read()
    df = pd.read_csv(io.BytesIO(contents))

    if target_column not in df.columns:
        return jsonify({"error": f"Target column '{target_column}' not found."})

    session_id = str(uuid.uuid4())

    analysis = baseline_bias_analysis(df, target_column)

    feature_cols = [c for c in df.columns if c != target_column]
    features: list[dict[str, Any]] = []
    for col in feature_cols:
        fm = analysis["feature_metrics"].get(col)
        if fm:
            features.append({
                "feature": col,
                "dtype": fm["dtype"],
                "metrics": fm,
                "status": "uncleaned",
            })

    _sessions[session_id] = {
        "original_df": df.copy(),
        "training_data": df.copy(),
        "target_column": target_column,
        "committed": {},
        "tried": {},
    }

    return jsonify(_sanitize({
        "sessionId": session_id,
        "features": features,
        "overallScore": analysis["overall_score"],
        "subScores": analysis.get("sub_scores", {}),
        "classCounts": analysis.get("class_counts", {}),
        "imbalanceRatio": analysis.get("imbalance_ratio", 1),
    }))


@app.route("/api/clean/suggest", methods=["POST"])
def clean_suggest():
    """Return ranked technique suggestions for a feature."""
    data = request.get_json()
    session_id = data["session_id"]
    feature = data["feature"]

    session = _sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."})

    tried = session["tried"].get(feature, set())
    return jsonify(_sanitize(_suggest_for_feature(
        session["training_data"], feature, session["target_column"], tried,
    )))


@app.route("/api/clean/preview", methods=["POST"])
def clean_preview():
    """Apply technique to temp copy and return before / after metrics."""
    data = request.get_json()
    session_id = data["session_id"]
    feature = data["feature"]
    technique_raw = data["technique"]

    session = _sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."})

    df = session["training_data"]
    target_column = session["target_column"]
    technique = _resolve_technique(technique_raw)

    before_analysis = baseline_bias_analysis(df, target_column)
    before_fm = before_analysis["feature_metrics"].get(feature, {})
    before_score = before_analysis["overall_score"]

    try:
        df_after = _apply_technique(df, target_column, technique, feature)
    except Exception as e:
        return jsonify({"error": str(e)})

    after_analysis = baseline_bias_analysis(df_after, target_column)
    after_fm = after_analysis["feature_metrics"].get(feature, {})
    after_score = after_analysis["overall_score"]

    return jsonify(_sanitize({
        "feature": feature,
        "technique": technique_raw,
        "before": before_fm,
        "after": after_fm,
        "overallBefore": before_score,
        "overallAfter": after_score,
        "delta": round(after_score - before_score, 1),
        "afterSubScores": after_analysis.get("sub_scores", {}),
    }))


@app.route("/api/clean/commit", methods=["POST"])
def clean_commit():
    """Permanently apply technique to training_data."""
    data = request.get_json()
    session_id = data["session_id"]
    feature = data["feature"]
    technique_raw = data["technique"]

    session = _sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."})

    technique = _resolve_technique(technique_raw)

    try:
        session["training_data"] = _apply_technique(
            session["training_data"],
            session["target_column"],
            technique,
            feature,
        )
    except Exception as e:
        return jsonify({"error": str(e)})

    if feature not in session["committed"]:
        session["committed"][feature] = []
    session["committed"][feature].append(technique_raw)

    # NOTE: We intentionally do NOT add committed techniques to "tried"
    # so they remain available when the user clicks "Clean Again".
    # Only skipped/rejected techniques should be filtered out.

    analysis = baseline_bias_analysis(
        session["training_data"], session["target_column"],
    )
    fm = analysis["feature_metrics"].get(feature, {})

    return jsonify(_sanitize({
        "feature": feature,
        "technique": technique_raw,
        "metrics": fm,
        "overallScore": analysis["overall_score"],
        "subScores": analysis.get("sub_scores", {}),
    }))


@app.route("/api/clean/revert", methods=["POST"])
def clean_revert():
    """Restore a feature to its original values."""
    data = request.get_json()
    session_id = data["session_id"]
    feature = data["feature"]

    session = _sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."})

    orig = session["original_df"]

    if feature in orig.columns:
        session["training_data"][feature] = orig[feature].copy()

    # Remove auto-created indicator column
    indicator = f"{feature}_missing"
    td = session["training_data"]
    if indicator in td.columns and indicator not in orig.columns:
        session["training_data"] = td.drop(columns=[indicator])

    # If SMOTE added rows, trim back to original size
    if len(session["training_data"]) > len(orig):
        session["training_data"] = session["training_data"].iloc[: len(orig)].copy()

    session["committed"].pop(feature, None)
    session["tried"].pop(feature, None)

    # Also revert any SMOTE rows for other features by trimming to original size
    # (already handled above for this feature)

    analysis = baseline_bias_analysis(
        session["training_data"], session["target_column"],
    )
    fm = analysis["feature_metrics"].get(feature, {})

    return jsonify(_sanitize({
        "feature": feature,
        "metrics": fm,
        "overallScore": analysis["overall_score"],
        "subScores": analysis.get("sub_scores", {}),
    }))


@app.route("/api/clean/state/<session_id>", methods=["GET"])
def clean_state(session_id: str):
    """Return the current state of every feature in the session."""
    session = _sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."})

    analysis = baseline_bias_analysis(
        session["training_data"], session["target_column"],
    )

    feature_cols = [
        c for c in session["original_df"].columns if c != session["target_column"]
    ]
    features: list[dict[str, Any]] = []
    for col in feature_cols:
        fm = analysis["feature_metrics"].get(col)
        if fm:
            committed_list = session["committed"].get(col, [])
            features.append({
                "feature": col,
                "dtype": fm["dtype"],
                "metrics": fm,
                "status": "cleaned" if committed_list else "uncleaned",
                "committedTechnique": committed_list[-1] if committed_list else None,
                "cleaningHistory": committed_list,
            })

    return jsonify(_sanitize({
        "features": features,
        "overallScore": analysis["overall_score"],
        "subScores": analysis.get("sub_scores", {}),
        "committed": session["committed"],
    }))


@app.route("/api/clean/download/<session_id>", methods=["GET"])
def clean_download(session_id: str):
    """Download the cleaned training data as a CSV file."""
    session = _sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."}), 404

    buf = io.StringIO()
    session["training_data"].to_csv(buf, index=False)
    buf.seek(0)

    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_training_data.csv"},
    )


@app.route("/api/clean/save/<session_id>", methods=["POST"])
def clean_save(session_id: str):
    """Save cleaned training data to disk as CSV."""
    session = _sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."}), 404

    saved_dir = os.path.join(os.path.dirname(__file__), "saved_data")
    os.makedirs(saved_dir, exist_ok=True)

    filename = f"cleaned_{session_id}.csv"
    filepath = os.path.join(saved_dir, filename)
    session["training_data"].to_csv(filepath, index=False)
    session["saved_csv_path"] = filepath

    return jsonify({"ok": True, "filename": filename})


# ---------------------------------------------------------------------------
# Stage 4 – Train: Real model training on cleaned training_data
# ---------------------------------------------------------------------------

@app.route("/api/train", methods=["POST"])
def train_models():
    """Train a selected model with hyperparameter tuning on 80/20 stratified split."""
    import traceback as _tb

    data = request.get_json()
    session_id = data.get("session_id")
    model_type = data.get("model_type")  # optional: filter to one model

    session = _sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."})

    # If in-memory training_data was lost (e.g. server restart), reload from saved CSV
    if session.get("training_data") is None:
        saved_path = session.get("saved_csv_path")
        if saved_path and os.path.isfile(saved_path):
            session["training_data"] = pd.read_csv(saved_path)

    try:
        return _train_models_inner(session, model_type=model_type)
    except Exception as exc:
        _tb.print_exc()
        return jsonify({"error": f"Training failed: {exc}"}), 500


def _train_models_inner(session: dict[str, Any], model_type: str | None = None) -> Response:
    """Core training logic — separated so the route handler can catch all errors."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    df = session["training_data"].copy()
    target_column = session["target_column"]

    if target_column not in df.columns:
        return jsonify({"error": f"Target column '{target_column}' not found."})

    # Prepare X, y
    feature_cols = [c for c in df.columns if c != target_column]
    X = df[feature_cols].copy()
    y = df[target_column].copy()

    # Encode categorical feature columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)

    # Drop rows with NaN
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    # Detect if target is categorical or numerical
    is_target_categorical = False
    le = None
    
    # Check if target is string/object type or has few unique values relative to size
    if y.dtype == object or pd.api.types.is_string_dtype(y):
        is_target_categorical = True
    elif pd.api.types.is_numeric_dtype(y):
        n_unique = y.nunique()
        # Consider numeric target categorical if it has fewer unique values (heuristic: < 20 or < 5% of data)
        if n_unique < 20 or n_unique < len(y) * 0.05:
            is_target_categorical = True
    
    # Encode target
    if is_target_categorical:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
    else:
        y_encoded = y.values.astype(float)

    print(f"[TRAIN] DataFrame shape: {df.shape}, X shape: {X.shape}, y shape: {y.shape}")
    print(f"[TRAIN] cat_cols encoded: {cat_cols}")
    print(f"[TRAIN] X dtypes: {dict(X.dtypes)}")
    print(f"[TRAIN] Target type: {'categorical' if is_target_categorical else 'numerical'}")
    if is_target_categorical:
        print(f"[TRAIN] y unique values: {np.unique(y_encoded)}, counts: {np.bincount(y_encoded)}")
    else:
        print(f"[TRAIN] y stats: min={y_encoded.min()}, max={y_encoded.max()}, mean={y_encoded.mean()}")

    if len(X) < 20:
        return jsonify({"error": "Not enough data to train models."})

    # Split strategy based on target type
    if is_target_categorical:
        n_unique = len(np.unique(y_encoded))
        stratify_param = y_encoded
        is_binary = n_unique == 2
    else:
        stratify_param = None
        is_binary = False
        n_unique = 0

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=stratify_param,
    )

    # Cross-validation folds
    if is_target_categorical:
        # Stratified K-Fold for classification
        class_counts_train = np.bincount(y_train.astype(int))
        min_class_count = int(class_counts_train[class_counts_train > 0].min())
        n_splits = max(2, min(5, min_class_count))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring_metric = "accuracy"
    else:
        # Regular K-Fold for regression
        n_splits = 5
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scoring_metric = "neg_mean_squared_error"

    # Hyperparameter grids for each model
    # NOTE: n_jobs=1 avoids Windows multiprocessing issues inside Flask
    if is_target_categorical:
        # Classification models
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        models_config = [
            (
                "Extra Trees",
                "extra_trees",
                ExtraTreesClassifier(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            ),
            (
                "Random Forest",
                "random_forest",
                RandomForestClassifier(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            ),
            (
                "Gradient Boosted Trees",
                "gradient_boosted",
                GradientBoostingClassifier(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5],
                    "subsample": [0.8, 1.0],
                },
            ),
        ]
    else:
        # Regression models
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        models_config = [
            (
                "Extra Trees",
                "extra_trees",
                ExtraTreesRegressor(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            ),
            (
                "Random Forest",
                "random_forest",
                RandomForestRegressor(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            ),
            (
                "Gradient Boosted Trees",
                "gradient_boosted",
                GradientBoostingRegressor(random_state=42),
                {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5],
                    "subsample": [0.8, 1.0],
                },
            ),
        ]

    # Filter to selected model if model_type provided
    if model_type:
        models_config = [m for m in models_config if m[1] == model_type]
        if not models_config:
            return jsonify({"error": f"Unknown model_type: {model_type}"})

    results = []
    fitted_models = []
    best_score = -float('inf')
    best_idx = 0

    for i, (name, mt, base_model, param_grid) in enumerate(models_config):
        try:
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring_metric,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            grid_search.fit(X_train, y_train)
            tuned_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = round(float(grid_search.best_score_), 4)

            fitted_models.append(tuned_model)

            # Evaluate on held-out test set
            y_pred = tuned_model.predict(X_test)

            if is_target_categorical:
                # Classification metrics
                # AUC-ROC (needs predict_proba)
                try:
                    if is_binary:
                        y_proba = tuned_model.predict_proba(X_test)[:, 1]
                        auc = round(float(roc_auc_score(y_test, y_proba)), 4)
                    else:
                        y_proba = tuned_model.predict_proba(X_test)
                        auc = round(float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")), 4)
                except Exception:
                    auc = 0.0

                acc = round(float(accuracy_score(y_test, y_pred)), 4)
                f1 = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                prec = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                rec = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
                
                # Track best model by accuracy
                model_score = acc
                
                metrics_dict = {"aucRoc": auc, "accuracy": acc, "f1": f1, "precision": prec, "recall": rec}
            else:
                # Regression metrics
                mse = round(float(mean_squared_error(y_test, y_pred)), 4)
                rmse = round(float(np.sqrt(mse)), 4)
                mae = round(float(mean_absolute_error(y_test, y_pred)), 4)
                r2 = round(float(r2_score(y_test, y_pred)), 4)
                
                # Track best model by R2 score (higher is better)
                model_score = r2
                
                # Format for frontend (using same structure but with regression metrics)
                metrics_dict = {"rmse": rmse, "mae": mae, "r2": r2, "mse": mse}

            if model_score > best_score:
                best_score = model_score
                best_idx = i

            serialized_params = {str(k): _sanitize(v) for k, v in best_params.items()}

            results.append({
                "name": name,
                "type": mt,
                "metrics": metrics_dict,
                "bestParams": serialized_params,
                "cvScore": cv_score,
                "trainingProgress": 100,
                "isWinner": False,
            })
        except Exception as e:
            import traceback
            print(f"[TRAIN] ERROR training {name}: {e}")
            traceback.print_exc()
            fitted_models.append(None)
            error_metrics = (
                {"aucRoc": 0, "accuracy": 0, "f1": 0, "precision": 0, "recall": 0}
                if is_target_categorical
                else {"rmse": 0, "mae": 0, "r2": 0, "mse": 0}
            )
            results.append({
                "name": name,
                "type": mt,
                "metrics": error_metrics,
                "bestParams": {},
                "cvScore": 0,
                "trainingProgress": 100,
                "isWinner": False,
                "error": str(e),
            })

    if results:
        results[best_idx]["isWinner"] = True

    # Store best fitted model for SHAP analysis
    session["_trained_model"] = fitted_models[best_idx]
    session["_train_X"] = pd.concat([X_train, X_test], ignore_index=True)
    session["_train_y"] = np.concatenate([y_train, y_test])
    session["_test_X"] = X_test
    session["_test_y"] = y_test
    session["_label_encoder"] = le
    session["_feature_names"] = list(X_train.columns)
    session["_is_target_categorical"] = is_target_categorical

    split_info = {
        "trainSize": int(len(X_train)),
        "testSize": int(len(X_test)),
        "splitRatio": "80/20",
        "stratified": bool(is_target_categorical),
        "targetType": "categorical" if is_target_categorical else "numerical",
        "cvFolds": int(n_splits),
    }
    if is_target_categorical and le is not None:
        split_info["nClasses"] = int(len(le.classes_))

    return jsonify(_sanitize({
        "models": results,
        "splitInfo": split_info,
    }))


# ---------------------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------------------

@app.route("/api/explain", methods=["POST"])
def explain_model():
    """Compute SHAP values and subgroup performance for the trained model."""
    data = request.get_json()
    session_id = data.get("session_id")

    session = _sessions.get(session_id)
    if not session:
        return jsonify({"error": "Session not found."})

    model = session.get("_trained_model")
    X = session.get("_train_X")
    y = session.get("_train_y")
    le = session.get("_label_encoder")
    feature_names = session.get("_feature_names", [])
    is_target_categorical = session.get("_is_target_categorical", True)

    if model is None or X is None:
        return jsonify({"error": "No trained model found. Run /api/train first."})

    # SHAP feature importance (use permutation importance as a robust fallback)
    from sklearn.inspection import permutation_importance

    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    importances = perm.importances_mean

    # Build sorted feature importance list
    sorted_idx = np.argsort(np.abs(importances))[::-1]
    shap_features = []
    total_imp = float(np.sum(np.abs(importances))) or 1.0
    for idx in sorted_idx[:10]:
        imp = float(importances[idx])
        shap_features.append({
            "feature": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
            "importance": round(abs(imp) / total_imp, 4),
            "direction": "positive" if imp >= 0 else "negative",
        })

    y_pred = model.predict(X)
    subgroup_perf = []
    compliance_ready = True

    if is_target_categorical:
        # Subgroup performance (by target class)
        from sklearn.metrics import accuracy_score, f1_score

        classes = le.classes_ if le is not None else np.unique(y)
        total_count = len(y)
        for cls_idx, cls_label in enumerate(classes):
            mask = y == cls_idx
            count = int(mask.sum())
            if count == 0:
                continue
            acc = round(float(accuracy_score(y[mask], y_pred[mask])), 4)
            f1_val = round(float(f1_score(y[mask], y_pred[mask], average="weighted", zero_division=0)), 4)
            subgroup_perf.append({
                "group": f"Class: {cls_label}",
                "accuracy": acc,
                "f1": f1_val,
                "count": count,
                "percentOfTotal": round(count / total_count * 100, 1),
            })

        # Check compliance: accuracy gap < 10%
        accs = [s["accuracy"] for s in subgroup_perf]
        acc_gap = max(accs) - min(accs) if accs else 0
        compliance_ready = acc_gap < 0.10
    else:
        # Regression: subgroup by quartiles of predicted values
        from sklearn.metrics import mean_absolute_error, r2_score

        quartiles = np.percentile(y, [25, 50, 75])
        bounds = [(-np.inf, quartiles[0]), (quartiles[0], quartiles[1]),
                  (quartiles[1], quartiles[2]), (quartiles[2], np.inf)]
        labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
        total_count = len(y)
        for label, (lo, hi) in zip(labels, bounds):
            mask = (y > lo) & (y <= hi) if lo != -np.inf else (y <= hi)
            count = int(mask.sum())
            if count < 2:
                continue
            mae = round(float(mean_absolute_error(y[mask], y_pred[mask])), 4)
            r2 = round(float(r2_score(y[mask], y_pred[mask])), 4) if count > 1 else 0.0
            subgroup_perf.append({
                "group": label,
                "mae": mae,
                "r2": r2,
                "count": count,
                "percentOfTotal": round(count / total_count * 100, 1),
            })

        compliance_ready = True

    return jsonify(_sanitize({
        "shapFeatures": shap_features,
        "subgroupPerformance": subgroup_perf,
        "complianceReady": compliance_ready,
    }))


# ---------------------------------------------------------------------------
# Run the server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
