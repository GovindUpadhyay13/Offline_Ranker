"""One-line justification per ranked candidate.

Two paths over the same structured facts (title, years, product-ML evidence, key
system, availability, location, and a counterfactual naming the weakest lever):

  template (default, fallback): assemble the facts into a fixed sentence. No LLM.
  llm: prompt a locally vendored flan-t5-small (offline, CPU, top 100 only) to
       phrase one sentence from those facts. A hallucination guard discards any
       output that introduces a number, percentage, or named entity absent from
       the facts, or that is empty/garbled, falling back to the template line.

Text pulled from data is sanitized so no dash leaks into the output.
"""

from __future__ import annotations

import re

import config
from src import features, scoring

_NUM = re.compile(r"\d+(?:\.\d+)?%?")
_WORD = re.compile(r"[A-Za-z][A-Za-z.&'/+-]*")
_COMMON_CAPS = {"i"}  # pronouns/acronyms allowed to be capitalized without being a fact entity


def _clean(s):
    return s.replace("—", ", ").replace("–", ", ").replace("\n", " ").strip()


def build_facts(cand, avail, loc) -> dict:
    """Structured facts shared by both reasoning paths. Values are None when absent."""
    p = cand.profile
    yoe = p.years_of_experience or 0
    f = {
        "years_short": f"{yoe:.0f}y experience",  # template phrasing
        "years": f"{yoe:.0f} years of experience",  # natural phrasing for the prompt
        "title": _clean(f"{p.current_title} at {p.current_company}".strip(" at")),
    }

    ml_years = scoring.applied_ml_months(cand) / 12.0
    f["product_ml"] = f"{ml_years:.0f} years of applied machine learning work" if ml_years >= 1 else None

    ev = features.evidence_text(cand).lower()
    hits = list(dict.fromkeys(h for h in config.NLP_IR_HINTS if h in ev))[:2]
    f["key_system"] = "hands-on " + " and ".join(hits) if hits else None

    if avail >= 0.5:
        f["availability"] = "active and responsive on platform"
    elif avail <= -0.5:
        f["availability"] = "limited recent platform activity"
    else:
        f["availability"] = None

    if loc >= 1.0:
        f["location"] = "based in a target metro"
    elif loc <= -1.0:
        f["location"] = "located outside India"
    else:
        f["location"] = None

    # Counterfactual: the single weakest lever that would lift the score.
    if loc <= -1.0:
        f["counterfactual"] = "would rank higher if based in or open to Pune or Noida"
    elif avail <= -0.5:
        f["counterfactual"] = "would rank higher with more recent recruiter engagement"
    elif loc < 1.0:
        f["counterfactual"] = "would rank higher if located in a target metro"
    else:
        f["counterfactual"] = "strong on both availability and location"
    return f


def _assemble(f) -> str:
    """Deterministic template line. Also the fallback when the llm output is rejected."""
    bits = [f["years_short"]]
    for k in ("title", "availability", "location"):
        if f[k]:
            bits.append(f[k])
    return _clean(". ".join(bits) + ". " + f["counterfactual"] + ".")


def justify(cand, avail, loc) -> str:
    return _assemble(build_facts(cand, avail, loc))


def generate(mode, top, mods):
    """Return (reasoning by candidate_id, count fallen back to template)."""
    if mode != "llm":
        return {cid: _assemble(build_facts(*mods[cid])) for cid, _ in top}, 0

    model, tok, torch = _load_llm()
    reasons, fell_back = {}, 0
    for cid, _ in top:
        f = build_facts(*mods[cid])
        text = _llm_one(model, tok, torch, f)
        if not _grounded(text, f):
            text = _assemble(f)
            fell_back += 1
        reasons[cid] = text
    return reasons, fell_back


_llm = None


def _load_llm():
    """Load flan-t5-small from local disk only. Cached across the top 100."""
    global _llm
    if _llm is None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(config.FLAN_DIR, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.FLAN_DIR, local_files_only=True)
        model.eval()
        _llm = (model, tok, torch)
    return _llm


def _prompt(f) -> str:
    facts = [f["years"], f["title"], f["product_ml"], f["key_system"],
             f["availability"], f["location"], f["counterfactual"]]
    body = "; ".join(x for x in facts if x)
    return ("Write one sentence recommending this job candidate using only the facts below. "
            "Do not add any company, number, percentage, or detail that is not listed.\n"
            f"Facts: {body}.\nSentence:")


def _llm_one(model, tok, torch, f) -> str:
    ids = tok(_prompt(f), return_tensors="pt", truncation=True, max_length=512).input_ids
    torch.manual_seed(config.SEED)  # reproducible greedy decode
    with torch.no_grad():
        out = model.generate(ids, do_sample=False, num_beams=1,
                             max_new_tokens=config.LLM_MAX_NEW_TOKENS)
    return _clean(tok.decode(out[0], skip_special_tokens=True))


def _grounded(text, f) -> bool:
    """Reject empty/garbled output, any number/percentage not in the facts, and any
    capitalized token (a likely company or named entity) absent from the facts."""
    toks = [w.lower() for w in _WORD.findall(text)]
    if not text or len(text) < 10 or len(toks) < 3:
        return False
    # Repeated trigram: flan-t5 loops on small inputs, the tell for garbled output.
    grams = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    if len(grams) != len(set(grams)):
        return False

    facts_text = " ".join(v for v in f.values() if v)
    fnums = set(_NUM.findall(facts_text))
    if any(n not in fnums for n in _NUM.findall(text)):
        return False

    allowed = {w.lower() for w in _WORD.findall(facts_text)}
    for m in _WORD.finditer(text):
        w = m.group()
        if not w[0].isupper() or w.lower() in allowed or w.lower() in _COMMON_CAPS:
            continue
        prev = text[:m.start()].rstrip()
        if prev and prev[-1] not in ".!?:":  # not sentence-initial, so a real new entity
            return False
    return True
