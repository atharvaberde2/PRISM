import sys, json
import pandas as pd
sys.path.insert(0, ".")
from main import baseline_bias_analysis

df = pd.read_csv(r"C:\Users\athar\Downloads\PRISM\frontend\dist\sample_loan_data.csv")
print("Shape:", df.shape)
print("Target values:", df["approved"].value_counts().to_dict())
print()

result = baseline_bias_analysis(df, "approved")
print("Overall score:", result["overall_score"])
print("Imbalance:", result["imbalance_ratio"])
print("Sub scores:", json.dumps(result["sub_scores"], indent=2))
print()

fm = result["feature_metrics"]
for k, m in fm.items():
    print(f"--- {k} ({m['dtype']}) ---")
    print(f"  jsd={m['jsd']}  test={m['test']}  stat={m['test_stat']}  p={m['p_value']}  effect={m['effect_size']}")
    print(f"  skew_diff={m['skew_diff']}  kurt_diff={m['kurt_diff']}  miss_gap={m['missingness_gap']}")
    print(f"  norms: jsd={m['norm_jsd']} eff={m['norm_effect']} skew={m['norm_skew']} miss={m['norm_miss']} pval={m['norm_pval']}")
    print(f"  feature_bias={m['feature_bias']}")
    print()
