""" Self-verification for SCALE_TEMPLATES / TRADITION_TEMPLATES.
    Run directly: python test_scale_templates.py"""
import sys
from itertools import combinations

from analysis import (
    SCALE_TEMPLATES, TRADITION_TEMPLATES, UNIT_CENTS, _cents_from_steps,
    _build_chroma_template, _cosine_sim, KYRIOS_TO_PLAGAL,
)

OCTAVE_EXEMPT = {"Cypriot Pentachord"}

failures: list[str] = []


def check(condition: bool, message: str):
    """ Verify the scale template table is internally consistent.
    Confirms that the steps of each octave-spanning mode really add up to a
    full octave, that the stored cents match those steps, and that no two
    modes of the same tradition are accidental duplicates."""
    if not condition:
        failures.append(message)


for name, t in SCALE_TEMPLATES.items():
    unit = UNIT_CENTS[t["basis"]]
    total = sum(t["steps"]) * unit
    if name not in OCTAVE_EXEMPT:
        check(abs(total - 1200.0) < 0.01, f"{name}: steps sum to {total}¢, not 1200¢")
    recomputed = _cents_from_steps(t["steps"], t["basis"])
    check(recomputed == t["cents"], f"{name}: stored cents {t['cents']} != recomputed {recomputed}")

byzantine_pool = set(TRADITION_TEMPLATES["byzantine"])
check(len(byzantine_pool) == 8, f"Byzantine pool has {len(byzantine_pool)} modes, expected 8")
check("Mode 8 – Plagios D'" in byzantine_pool, "Mode 8 missing from competition pool")
for old_genus_name in ("Chromatic – Hard", "Chromatic – Soft", "Enharmonic"):
    check(
        old_genus_name not in SCALE_TEMPLATES,
        f"{old_genus_name} should no longer exist as a separate template — "
        f"it's a duplicate of an echos, now expressed via that echos's `genus` field",
    )

all_pooled_names = sorted({n for names in TRADITION_TEMPLATES.values() for n in names})
templates_36 = {n: _build_chroma_template(SCALE_TEMPLATES[n]["cents"]) for n in all_pooled_names}

within_pool_dups = []
for tradition, members in TRADITION_TEMPLATES.items():
    for a, b in combinations(members, 2):
        sim = _cosine_sim(templates_36[a], templates_36[b])
        if sim > 0.999:
            within_pool_dups.append((tradition, a, b, sim))

expected_within_pool = {("byzantine", "Mode 1 – Protos", "Mode 5 – Plagios A'")}
found_within_pool = {(t, a, b) for t, a, b, _ in within_pool_dups}
check(
    found_within_pool == expected_within_pool,
    f"within-pool duplicates {found_within_pool} != expected {expected_within_pool}",
)

cross_pool_dups = []
for a, b in combinations(all_pooled_names, 2):
    same_pool = any(a in m and b in m for m in TRADITION_TEMPLATES.values())
    if same_pool:
        continue
    sim = _cosine_sim(templates_36[a], templates_36[b])
    if sim > 0.999:
        cross_pool_dups.append((a, b, sim))

expected_cross_pool = {
    frozenset({"Mode 3 – Tritos", "Maqam Ajam"}),
    frozenset({"Greek Hijaz", "Maqam Hijaz"}),
    frozenset({"Greek Major (Matzore)", "Maqam Ajam"}),
    frozenset({"Greek Major (Matzore)", "Mode 3 – Tritos"}),
}
found_cross_pool = {frozenset({a, b}) for a, b, _ in cross_pool_dups}
check(
    found_cross_pool == expected_cross_pool,
    f"cross-pool duplicates {found_cross_pool} != expected {expected_cross_pool} "
    f"(these are musicologically real and should stay — a NEW one appearing "
    f"means a template regressed to an artifact-level match)",
)

check(
    KYRIOS_TO_PLAGAL == {"Mode 1 – Protos": "Mode 5 – Plagios A'"},
    f"KYRIOS_TO_PLAGAL has unexpected entries: {KYRIOS_TO_PLAGAL}",
)

if failures:
    print(f"FAILED — {len(failures)} issue(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(
        f"OK — {len(SCALE_TEMPLATES)} templates verified, "
        f"{len(within_pool_dups)} within-pool dup(s), {len(cross_pool_dups)} cross-pool dup(s) "
        f"(all expected)."
    )
