"""One-line justification per ranked candidate, built from structured facts.

No LLM. Each line states experience, current role, the dominant fit signals, and
a counterfactual clause naming the weakest lever. Text pulled from data is
sanitized so no em-dash leaks into the output.
"""

from __future__ import annotations


def _clean(s):
    return s.replace("\u2014", ", ").replace("\u2013", ", ").replace("\n", " ").strip()


def justify(cand, avail, loc) -> str:
    p = cand.profile
    yoe = p.years_of_experience or 0
    role = _clean(f"{p.current_title} at {p.current_company}".strip(" at"))
    bits = [f"{yoe:.0f}y experience", role] if role else [f"{yoe:.0f}y experience"]

    if avail >= 0.5:
        bits.append("active and responsive on platform")
    elif avail <= -0.5:
        bits.append("limited recent platform activity")

    if loc >= 1.0:
        bits.append("based in a target metro")
    elif loc <= -1.0:
        bits.append("located outside India")

    # Counterfactual: name the single weakest lever that would lift the score.
    if loc <= -1.0:
        cf = "would rank higher if based in or open to Pune or Noida"
    elif avail <= -0.5:
        cf = "would rank higher with more recent recruiter engagement"
    elif loc < 1.0:
        cf = "would rank higher if located in a target metro"
    else:
        cf = "strong on both availability and location"

    return _clean(". ".join(bits) + ". " + cf + ".")
