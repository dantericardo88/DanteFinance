#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys
sys.path.insert(0, '.')

from sentinel.sfe.proxy_intelligence_v3 import (
    _governance_letter_grade,
    _safe_float,
    _safe_int,
    GovernanceScore,
    DirectorRecord,
    BoardComposition,
)

# Test _governance_letter_grade
assert _governance_letter_grade(90.0) == "A"
assert _governance_letter_grade(75.0) == "B"
assert _governance_letter_grade(60.0) == "C"
assert _governance_letter_grade(45.0) == "D"
assert _governance_letter_grade(30.0) == "F"
print("[OK] _governance_letter_grade thresholds correct")

# Test _safe_float
assert _safe_float("1,234.56") == 1234.56
assert _safe_float("bad") is None or _safe_float("bad") == 0.0
print("[OK] _safe_float handles numeric strings")

# Test _safe_int
assert _safe_int("1,000") == 1000
assert _safe_int("abc") is None or _safe_int("abc") == 0
print("[OK] _safe_int handles numeric strings")

# Test DirectorRecord creation (Pydantic model)
dr = DirectorRecord(
    name="Jane Smith",
    age=58,
    tenure_years=5.0,
    independent=True,
    gender="F",
    committees=["Audit", "Compensation"],
    other_boards=2,
    overboarded=False,
)
assert dr.name == "Jane Smith"
assert dr.independent is True
print("[OK] DirectorRecord Pydantic model created successfully")

# Test BoardComposition
bc = BoardComposition(
    ticker="AAPL",
    year=2024,
    board_size=10,
    independent_count=8,
    independent_pct=80.0,
    avg_tenure_years=6.5,
    avg_age=62.0,
    overboarded_count=1,
    directors=[dr],
)
assert bc.board_size == 10
assert bc.independent_pct == 80.0
print("[OK] BoardComposition model created successfully")

# Test GovernanceScore
gs = GovernanceScore(
    ticker="AAPL",
    year=2024,
    total_score=82.0,
    letter_grade="B",
    no_poison_pill=20.0,
    proxy_access=15.0,
)
assert gs.letter_grade == "B"
assert gs.total_score == 82.0
print("[OK] GovernanceScore model created successfully")

print("[PASS]")
PYEOF
