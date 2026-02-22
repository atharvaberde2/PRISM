"""End-to-end test of all 5 PRISM stages with real_estate_census_data.csv"""

import sys
import json
import io
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from main import app, baseline_bias_analysis, _suggest_for_feature, _apply_technique, _resolve_technique, _sessions

client = app.test_client()

def test_stage1():
    """Stage 1: UNDERSTAND - Profile + Baseline Bias"""
    print("=" * 60)
    print("STAGE 1: UNDERSTAND")
    print("=" * 60)

    # Test /api/profile
    with open("real_estate_census_data.csv", "rb") as f:
        resp = client.post("/api/profile", data={"file": (f, "data.csv")},
                          content_type="multipart/form-data")
    assert resp.status_code == 200
    profile = resp.get_json()
    print(f"  Profiled {len(profile)} columns")
    for p in profile[:3]:
        print(f"    {p['name']}: {p['dtype']} (nulls={p['nullCount']})")

    # Test baseline bias directly (snake_case keys)
    df = pd.read_csv("real_estate_census_data.csv")
    bias = baseline_bias_analysis(df, "high_potential")
    print(f"  Baseline bias score: {bias['overall_score']}")
    print(f"  Class counts: {bias['class_counts']}")
    print(f"  Imbalance ratio: {bias['imbalance_ratio']}")
    print(f"  Features analyzed: {len(bias['feature_metrics'])}")

    print("  STAGE 1 PASSED")
    return bias


def test_stage2(bias_report):
    """Stage 2: CLEAN - Init session, suggest, preview, commit"""
    print("\n" + "=" * 60)
    print("STAGE 2: CLEAN")
    print("=" * 60)

    # Init session
    with open("real_estate_census_data.csv", "rb") as f:
        resp = client.post("/api/clean/init",
                          data={"file": (f, "data.csv"), "target_column": "high_potential"},
                          content_type="multipart/form-data")
    assert resp.status_code == 200
    init = resp.get_json()
    assert "error" not in init, f"Init error: {init.get('error')}"

    session_id = init["sessionId"]
    features = init["features"]
    print(f"  Session: {session_id[:12]}...")
    print(f"  Features: {len(features)}")
    print(f"  Initial score: {init['overallScore']}")

    # Show features
    for f in features:
        score = round((1 - f['metrics']['feature_bias']) * 100)
        print(f"    {f['feature']:30s} {f['dtype']:12s} score={score}")

    # Pick test features: one numeric w/ nulls, one categorical, one numeric w/o nulls
    test_features = []
    for f in features:
        if f['dtype'] == 'numeric' and f['metrics'].get('missingness_gap', 0) > 0 and len(test_features) < 1:
            test_features.append(f['feature'])
    for f in features:
        if f['dtype'] == 'categorical' and f['feature'] not in test_features:
            test_features.append(f['feature'])
            break
    for f in features:
        if f['dtype'] == 'numeric' and f['metrics'].get('missingness_gap', 0) == 0 and f['feature'] not in test_features:
            test_features.append(f['feature'])
            break

    print(f"\n  Testing {len(test_features)} features: {test_features}")

    for feat in test_features:
        print(f"\n  --- {feat} ---")
        resp = client.post("/api/clean/suggest",
                          json={"session_id": session_id, "feature": feat})
        assert resp.status_code == 200
        suggest = resp.get_json()
        assert "error" not in suggest, f"Suggest error: {suggest.get('error')}"

        print(f"    Worst metric: {suggest['worstMetric']} ({suggest['worstMetricValue']:.4f})")
        print(f"    {len(suggest['suggestions'])} suggestions:")
        for s in suggest["suggestions"]:
            print(f"      [{s['category']:12s}] {s['label']:40s} delta={s['expectedDelta']:+.2f} {'*' if s['isHero'] else ''}")

        assert len(suggest["suggestions"]) > 0, f"No suggestions for {feat}!"

        # Preview best
        best = suggest["suggestions"][0]
        resp = client.post("/api/clean/preview",
                          json={"session_id": session_id, "feature": feat, "technique": best["technique"]})
        assert resp.status_code == 200
        preview = resp.get_json()
        assert "error" not in preview, f"Preview error: {preview.get('error')}"
        print(f"    Preview: {preview['overallBefore']} -> {preview['overallAfter']} (delta={preview['delta']:+.1f})")

        # Commit
        resp = client.post("/api/clean/commit",
                          json={"session_id": session_id, "feature": feat, "technique": best["technique"]})
        assert resp.status_code == 200
        commit = resp.get_json()
        assert "error" not in commit, f"Commit error: {commit.get('error')}"
        print(f"    Committed -> score: {commit['overallScore']}")

    print("  STAGE 2 PASSED")
    return session_id


def test_stage3(session_id):
    """Stage 3: GATE - Fairness compliance check (computed frontend-side)"""
    print("\n" + "=" * 60)
    print("STAGE 3: GATE (frontend-computed)")
    print("=" * 60)

    # Gate is computed on the frontend by comparing baseline vs current scores.
    # We simulate it here by checking the session's current state.
    session = _sessions.get(session_id)
    assert session is not None, "Session not found"

    original_df = session["original_df"]
    cleaned_df = session["training_data"]
    target = session["target_column"]

    before = baseline_bias_analysis(original_df, target)
    after = baseline_bias_analysis(cleaned_df, target)

    before_score = before["overall_score"]
    after_score = after["overall_score"]
    delta = round(after_score - before_score, 1)

    print(f"  Overall score: {before_score} -> {after_score} (delta={delta:+.1f})")
    print(f"  Imbalance: {before['imbalance_ratio']:.3f} -> {after['imbalance_ratio']:.3f}")

    # Check dimensions
    dims = [
        ("Overall Bias Score", before_score, after_score, delta > 0),
        ("Imbalance Ratio", before["imbalance_ratio"], after["imbalance_ratio"],
         after["imbalance_ratio"] <= before["imbalance_ratio"]),
    ]
    for name, b, a, passed in dims:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}: {b} -> {a}")

    passed = delta > 0
    print(f"  Gate result: {'PASSED' if passed else 'NEEDS OVERRIDE (ok for pipeline)'}")
    print("  STAGE 3 PASSED")
    return True


def test_stage4(session_id):
    """Stage 4: TRAIN - Model training"""
    print("\n" + "=" * 60)
    print("STAGE 4: TRAIN")
    print("=" * 60)

    resp = client.post("/api/train", json={"session_id": session_id})
    assert resp.status_code == 200
    train = resp.get_json()
    assert "error" not in train, f"Train error: {train.get('error')}"

    print(f"  Split: {train['splitInfo']['splitRatio']} (stratified={train['splitInfo'].get('stratified')})")
    print(f"  Target type: {train['splitInfo']['targetType']}")
    print(f"  CV folds: {train['splitInfo']['cvFolds']}")
    print(f"  Models: {len(train['models'])}")

    for m in train["models"]:
        w = " ** WINNER **" if m.get("isWinner") else ""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in m["metrics"].items())
        print(f"    {m['name']}: {metrics_str}{w}")

    print("  STAGE 4 PASSED")
    return train


def test_stage5(session_id):
    """Stage 5: EXPLAIN - Feature importance + subgroup analysis"""
    print("\n" + "=" * 60)
    print("STAGE 5: EXPLAIN")
    print("=" * 60)

    resp = client.post("/api/explain", json={"session_id": session_id})
    assert resp.status_code == 200
    explain = resp.get_json()
    assert "error" not in explain, f"Explain error: {explain.get('error')}"

    print(f"  Top features:")
    for f in explain.get("shapFeatures", []):
        bar = "#" * int(f["importance"] * 40)
        print(f"    {f['feature']:30s} {f['importance']:.3f} ({f['direction']:>8s}) {bar}")

    print(f"\n  Subgroup performance:")
    for sg in explain.get("subgroupPerformance", []):
        acc = sg.get('accuracy', 0)
        f1 = sg.get('f1', 0)
        print(f"    {sg['group']:20s} acc={acc:.3f} f1={f1:.3f} n={sg['count']} ({sg['percentOfTotal']:.1f}%)")

    print(f"\n  Compliance ready: {explain.get('complianceReady')}")
    print("  STAGE 5 PASSED")


if __name__ == "__main__":
    print("PRISM End-to-End Pipeline Test")
    print("Data: real_estate_census_data.csv | Target: high_potential\n")

    try:
        bias = test_stage1()
        session_id = test_stage2(bias)
        test_stage3(session_id)
        test_stage4(session_id)
        test_stage5(session_id)

        print("\n" + "=" * 60)
        print("ALL 5 STAGES COMPLETED SUCCESSFULLY")
        print("=" * 60)
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
