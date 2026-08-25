"""Re-derive state and model prices from a captured tape, using current decoders.

    python3 reprocess_tape.py tapes/<event>.jsonl --me-first

The tape stores each poll's RAW `details` payload, so any decoder fix can be applied
retroactively without re-running the match. That is the whole point of keeping the
raw payload: a live capture cannot be repeated, but it can be re-read.

Emits corrected per-poll rows and reports what changed versus what was recorded live.
"""
import argparse
import json
import sys

import kalshi_tennis as kt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tape")
    ap.add_argument("--out", help="write corrected rows to this .jsonl")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.tape)]
    if not rows:
        sys.exit("empty tape")

    d0 = rows[0]["details"]
    c1, c2 = d0.get("competitor1_id"), d0.get("competitor2_id")
    # "me" is whichever competitor the live capture used; recover it from the state
    s0 = rows[0]["state"]
    me = c1 if s0.get("sets_me") == d0.get("competitor1_overall_score") else c2
    opp = c2 if me == c1 else c1

    fixed = changed = 0
    out = []
    for r in rows:
        det = r["details"]
        st = kt.model_state(det, me, opp)
        if st is None:
            continue
        old = r["state"]
        if (old.get("points_me"), old.get("points_opp")) != (st["points_me"], st["points_opp"]):
            changed += 1
        if old.get("_points_known") is False and st["_points_known"]:
            fixed += 1
        out.append(dict(r, state=st))

    print(f"{len(rows)} polls   points recovered: {fixed}   points changed: {changed}")
    known = sum(1 for r in out if r["state"]["_points_known"])
    print(f"points known after reprocessing: {known}/{len(out)}")

    unknown = [r for r in out if not r["state"]["_points_known"]]
    if unknown:
        from collections import Counter
        c = Counter((r["details"].get("competitor1_current_round_score"),
                     r["details"].get("competitor2_current_round_score"))
                    for r in unknown)
        print("still-unrecognised point codes:")
        for k, v in c.most_common():
            print(f"   raw={k}  n={v}")
    else:
        print("every poll decoded")

    if a.out:
        with open(a.out, "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
