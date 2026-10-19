"""Print the result tables used in REPORT.md from results/*.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = ["NULL_s1", "MAIN_s1", "MAIN_s2", "SUBTLE_s1", "EASY_s1"]
CTRL = ["C-TIME", "C-CTX", "C-ALERT", "C-SCAN", "C-ALERT+CTX"]


def main():
    R = {r: json.loads((HERE / "results" / f"{r}.json").read_text()) for r in RUNS}
    print("| Run | H | Best model PR-AUC | Best control (PR-AUC) | Ratio | Orgs won | ROC-AUC model / best ctrl | "
          "Recall model | Best control recall | Lead (min) | P1 | P2 | P3 | P4 | Pass |")
    print("|" + "---|" * 15)
    for r in RUNS:
        for h, d in R[r]["horizons"].items():
            v, s = d["verdict"], d["results"]
            m, c = v["best_model"], v["best_control"]
            rc = max(CTRL, key=lambda x: s[x]["recall"] or 0)
            yn = lambda b: "yes" if b else "no"
            print(f"| {r} | {h} | {m} {s[m]['pr_auc_median']:.4f} | {c} {s[c]['pr_auc_median']:.4f} | "
                  f"{s[m]['pr_auc_median'] / s[c]['pr_auc_median']:.2f} | {v['P1_detail']['wins']}/6 | "
                  f"{s[m]['roc_auc_median']:.2f} / {s[c]['roc_auc_median']:.2f} | {s[m]['recall']:.2f} | "
                  f"{rc} {s[rc]['recall']:.2f} | {s[m]['median_lead_min']} | {yn(v['P1'])} | {yn(v['P2'])} | "
                  f"{yn(v['P3'])} | {yn(v['P4'])} | **{yn(v['pass'])}** |")
    print()
    print("Recall by chain (best model at 2 h, warned/attacks):")
    for r in RUNS:
        d = R[r]["horizons"]["2h"]
        m = d["verdict"]["best_model"]
        s = d["results"][m]
        bc = ", ".join(f"{k} {x['warned']}/{x['n']}" for k, x in s["by_chain"].items())
        fa, ff = s["false_flags_near_aborted_chain"], s["false_flags"]
        hd = sum(o["rows"] for o in R[r]["orgs"]) / 144
        print(f"- {r} {m}: {bc}; false flags {ff} ({100 * ff / hd:.1f} per 100 host-days), "
              f"{100 * fa / max(ff, 1):.0f}% on hosts with an aborted chain; test flag rate "
              f"{s['test_flag_rate_median']:.4f}; Z recall {d['verdict']['Z_recall']:.3f} (limit {d['verdict']['Z_limit']:.3f})")
    print()
    print("Flag rates of controls at 2 h (MAIN_s1):",
          {c: round(R["MAIN_s1"]["horizons"]["2h"]["results"][c]["test_flag_rate_median"], 4) for c in CTRL})
    lco = R["MAIN_s1"]["horizons"]["2h"]["leave_chain_out"]
    if lco:
        tot = {}
        for v in lco.values():
            for ct, x in v.items():
                a = tot.setdefault(ct, [0, 0]); a[0] += x["warned"]; a[1] += x["n"]
        full = R["MAIN_s1"]["horizons"]["2h"]["results"]["M-GB"]["by_chain"]
        print("Leave-chain-type-out (M-GB, 2 h, MAIN_s1): " + ", ".join(
            f"{ct} {w}/{n} (with that type in training {full[ct]['warned']}/{full[ct]['n']})" for ct, (w, n) in tot.items()))
    print()
    print("NULL leak guard ratios:", {h: round(d["verdict"]["null_ratio"], 2) for h, d in R["NULL_s1"]["horizons"].items()})


if __name__ == "__main__":
    main()
