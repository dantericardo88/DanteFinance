"""
Generate DanteForge compete/matrix.json from sentinel_competitive_matrix_v4.0.json
"""
import json, math, os, sys
from datetime import datetime, timezone

V4_PATH  = os.path.join(os.path.dirname(__file__), "..", "Docs", "sentinel_competitive_matrix_v4.0.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", ".danteforge", "compete", "matrix.json")

CAT_NAMES = {
    1:  "market_data",
    2:  "fundamentals",
    3:  "ownership_sec",
    4:  "fixed_income",
    5:  "macro_economics",
    6:  "ai_nlp",
    7:  "backtesting",
    8:  "screening",
    9:  "portfolio_risk",
    10: "alt_data",
    11: "terminal_ux",
    12: "private_markets",
    13: "esg",
    14: "crypto_defi",
}

CATEGORY_FREQ = {
    "market_data": "high", "fundamentals": "high", "portfolio_risk": "high",
    "screening": "high", "macro_economics": "medium", "ai_nlp": "medium",
    "backtesting": "medium", "ownership_sec": "medium", "fixed_income": "medium",
    "terminal_ux": "medium", "alt_data": "medium", "crypto_defi": "medium",
    "esg": "low", "private_markets": "low",
}

# Higher weight = more impact on composite / priority for sprints
STRATEGIC_WEIGHTS = {
    29: 1.5, 54: 1.5, 59: 1.5, 64: 1.5, 67: 1.5, 76: 1.5,  # unique/leapfrog
     1: 1.4,  2: 1.4, 13: 1.3, 14: 1.3, 15: 1.3, 77: 1.3, 61: 1.3,  # core data/risk
    43: 1.2, 47: 1.2, 79: 1.2, 81: 1.2, 82: 1.2, 83: 1.2, 90: 1.2,  # strong dims
}

# Dims permanently excluded from 9.0 gate (commercial licenses / deliberate non-goals)
PERMANENT_CAPS = {
    10: {"ceiling": 0, "reason": "Requires exchange SIP feed license; not addressable free"},
    12: {"ceiling": 0, "reason": "NYSE/NASDAQ TAQ costs $30K+/yr; not addressable free"},
    18: {"ceiling": 4, "reason": "yfinance proxy only; real IBES requires FactSet license ($2K+/yr)"},
    40: {"ceiling": 0, "reason": "Requires Bloomberg/ICE MBS feed; not addressable free"},
    41: {"ceiling": 0, "reason": "Requires LSTA/Refinitiv LPC commercial license"},
    42: {"ceiling": 0, "reason": "Deliberate non-goal; Bloomberg BVAL moat"},
    58: {"ceiling": 0, "reason": "Deliberate non-goal; Tegus/Mosaic moat; legal/compliance risk"},
    69: {"ceiling": 0, "reason": "Requires co-location + exchange membership; out of scope"},
    87: {"ceiling": 0, "reason": "Maxar/Planet Labs commercial license; not addressable free"},
    88: {"ceiling": 2, "reason": "MarineTraffic free tier too limited for signal generation"},
    94: {"ceiling": 0, "reason": "Not in scope; REST API enables third-party mobile client"},
    99: {"ceiling": 0, "reason": "Deliberate non-goal; PitchBook moat"},
}

CS_COMPETITORS  = ["bloomberg", "capiq", "factset", "lseg", "morningstar", "alphasense"]
OSS_COMPETITORS = ["openbb", "qlib", "vectorbt"]
ALL_COMPETITORS = CS_COMPETITORS + OSS_COMPETITORS

def status_for(score):
    if score >= 9: return "shipped"
    if score >= 5: return "in-progress"
    if score >= 1: return "started"
    return "not-started"

def build_matrix():
    with open(V4_PATH, encoding="utf-8") as f:
        v4 = json.load(f)

    dims_out = []
    score_sum = weight_sum = 0.0

    for d in v4["dimensions"]:
        dim_id   = d["id"]
        cat_id   = d["cat"]
        cat_name = CAT_NAMES[cat_id]
        freq     = CATEGORY_FREQ.get(cat_name, "medium")
        weight   = STRATEGIC_WEIGHTS.get(dim_id, 1.0)

        self_score  = d["sentinel_harsh"]
        target_raw  = d["target"]

        # competitor scores dict
        comp_scores = {c: d.get(c, 0) for c in ALL_COMPETITORS}

        # leader calcs
        all_vals = [(c, comp_scores[c]) for c in ALL_COMPETITORS]
        cs_vals  = [(c, comp_scores[c]) for c in CS_COMPETITORS]
        oss_vals = [(c, comp_scores[c]) for c in OSS_COMPETITORS]

        leader_name, leader_score         = max(all_vals, key=lambda x: x[1])
        cs_leader_name, cs_leader_score   = max(cs_vals,  key=lambda x: x[1])
        oss_leader_name, oss_leader_score = max(oss_vals, key=lambda x: x[1])

        gap_leader  = max(0.0, leader_score   - self_score)
        gap_cs      = max(0.0, cs_leader_score - self_score)
        gap_oss     = max(0.0, oss_leader_score - self_score)

        # cap handling
        cap_info = PERMANENT_CAPS.get(dim_id)
        ceiling  = None
        ceiling_reason = None

        if cap_info:
            ceiling = cap_info["ceiling"]
            ceiling_reason = cap_info["reason"]
            effective_target = ceiling
        else:
            effective_target = target_raw

        # next sprint target: step up by at most 2.0 or reach the effective target
        next_target = min(effective_target, self_score + 2.0)
        next_target = max(next_target, self_score)   # never go backwards

        dim_entry = {
            "id":    f"dim_{dim_id:03d}",
            "label": d["feature"],
            "weight": weight,
            "category": cat_name,
            "frequency": freq,
            "scores": {
                "self": self_score,
                **{c: comp_scores[c] for c in ALL_COMPETITORS},
            },
            "gap_to_leader":                gap_leader,
            "leader":                       leader_name if gap_leader > 0 else "sentinel",
            "gap_to_closed_source_leader":  gap_cs,
            "closed_source_leader":         cs_leader_name if gap_cs > 0 else "sentinel",
            "gap_to_oss_leader":            gap_oss,
            "oss_leader":                   oss_leader_name if gap_oss > 0 else "sentinel",
            "status":                       status_for(self_score),
            "sprint_history":               [],
            "next_sprint_target":           round(next_target, 1),
        }

        if ceiling is not None:
            dim_entry["ceiling"] = ceiling
            dim_entry["ceilingReason"] = ceiling_reason

        score_sum  += self_score * weight
        weight_sum += weight
        dims_out.append(dim_entry)

    overall = round(score_sum / weight_sum, 2) if weight_sum else 0.0

    matrix = {
        "project":               "SENTINEL",
        "description":           "Sovereign institutional-grade financial terminal; target: Bloomberg-parity at $0/yr",
        "competitors":           ALL_COMPETITORS,
        "competitors_closed_source": CS_COMPETITORS,
        "competitors_oss":       OSS_COMPETITORS,
        "lastUpdated":           datetime.now(timezone.utc).isoformat(),
        "overallSelfScore":      overall,
        "harshComposite":        5.0,
        "selfComposite":         7.1,
        "annualCost":            {"sentinel": 0, "bloomberg": 31980, "capiq": 18500,
                                  "factset": 28500, "lseg": 16000, "morningstar": 17500,
                                  "alphasense": 24000, "openbb": 0, "qlib": 0, "vectorbt": 0},
        "prerequisites": [
            "make docker-up",
            "make db-migrate",
            "make check",
            "make bootstrap",
            "make backfill"
        ],
        "permanentExclusions": list(PERMANENT_CAPS.keys()),
        "dimensions":            dims_out,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)

    print(f"[OK] Wrote {len(dims_out)} dimensions to {OUT_PATH}")
    print(f"[OK] Overall weighted self-score: {overall}/10")
    print(f"[OK] Permanent caps: {len(PERMANENT_CAPS)} dims")

    at_target = sum(1 for d in dims_out if d["scores"]["self"] >= 9.0)
    below_target = sum(1 for d in dims_out if d["scores"]["self"] < 9.0 and d.get("ceiling") is None)
    capped = len(PERMANENT_CAPS)
    print(f"[OK] Dims at 9.0+: {at_target}")
    print(f"[OK] Dims below target (buildable): {below_target}")
    print(f"[OK] Dims permanently capped: {capped}")

if __name__ == "__main__":
    build_matrix()
