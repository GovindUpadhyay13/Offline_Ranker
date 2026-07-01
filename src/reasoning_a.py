import re
import numpy as np
import config
from src import features, scoring

# Seeded random number generator for template selection
_RNG = np.random.RandomState(config.SEED)

TEMPLATES = [
    "Candidate features {years} with {current}, possessing {skill_jd} and presenting {notice}; however, {concern}.",
    "Demonstrating {skill_jd} over {current}, this candidate brings {years} and {notice}, though {concern}.",
    "With {years} and {current}, they show strong {skill_jd} and {notice}, but a key concern is that {concern}.",
    "Highlighting {skill_jd} during {current}, they offer {years} and {notice}, with the primary gap being {concern}.",
    "An engineer with {years} (including {current}), they demonstrate {skill_jd} alongside {notice}, but {concern}.",
    "While they present {notice} and {skill_jd} over {current} with {years} total, {concern} remains a notable concern.",
    "Over {current}, they have developed {skill_jd}; they bring {years} and {notice}, although {concern}.",
    "Possessing {years} including {current}, they combine {skill_jd} and {notice}, yet {concern} is a drawback."
]

def _clean(s):
    return s.replace("—", ", ").replace("–", ", ").replace("\n", " ").strip()

def get_grounded_skill(cand):
    skills_lower = {s.name.lower(): s.name for s in cand.skills}
    
    # 1. Evaluation framework skills
    eval_keywords = ["ndcg", "mrr", "map", "a/b testing", "offline evaluation", "online evaluation", "evaluation framework", "correlation"]
    for k in eval_keywords:
        for sk_lower, sk_orig in skills_lower.items():
            if k in sk_lower:
                return sk_orig, f"hands-on design of ranking/retrieval evaluation frameworks — {sk_orig}"
                
    # 2. Vector DB / Embeddings skills
    vector_keywords = ["qdrant", "milvus", "faiss", "pinecone", "chroma", "pgvector", "weaviate", "vector search", "embeddings"]
    for k in vector_keywords:
        for sk_lower, sk_orig in skills_lower.items():
            if k in sk_lower:
                return sk_orig, f"production vector-DB experience — {sk_orig}"
                
    # 3. Search / Recommendation / Ranking skills
    search_keywords = ["elasticsearch", "solr", "opensearch", "search", "ranking", "recsys", "recommendation"]
    for k in search_keywords:
        for sk_lower, sk_orig in skills_lower.items():
            if k in sk_lower:
                return sk_orig, f"end-to-end ranking/search/recommendation production systems — {sk_orig}"
                
    # Fallback to any skill
    if cand.skills:
        s_orig = cand.skills[0].name
        return s_orig, f"applied machine learning capabilities — {s_orig}"
        
    return None, None

def get_genuine_concern(cand, has_eval_exp):
    # 1. Missing evaluation-framework experience
    if not has_eval_exp:
        return "evaluation_framework_absent", "lacks hands-on evaluation framework experience (such as NDCG or A/B testing)"
        
    # 2. Location mismatch
    loc = scoring.location_modifier(cand)
    if loc == -1.0:
        return "location_mismatch", "located outside India which does not match preferred metros"
    elif loc == 0.0:
        return "location_mismatch", "not currently in target metros and unwilling to relocate"
    elif loc == 0.5:
        return "location_mismatch", "requires relocation to preferred target metros"

    # 3. High notice period
    notice = cand.signals.notice_period_days
    if notice is not None and notice > 60:
        return "notice_period", f"has a long notice period of {notice} days"
        
    # 4. Title chaser
    if scoring.is_title_chaser(cand):
        return "title_chaser", "exhibits a rapid title-escalating job-hopping history"

    # 5. Low responsiveness
    resp = cand.signals.recruiter_response_rate
    if resp is not None and resp < 0.5:
        return "responsiveness", "shows lower platform responsiveness"
        
    # Fallbacks
    if notice is not None and notice > 30:
        return "notice_period", f"notice period is {notice} days"
        
    return "none", "no critical concerns identified in current history"

def verify_grounding(text, cand, skill_used) -> bool:
    """Grounding verification: check that proper nouns, skills, numbers in text are true to the candidate."""
    # Find all numbers
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    evidence = features.evidence_text(cand).lower()
    
    # Check numbers
    for num in numbers:
        yoe = cand.profile.years_of_experience
        notice = cand.signals.notice_period_days
        yoe_str = f"{int(yoe)}" if yoe is not None else None
        notice_str = str(notice) if notice is not None else None
        if num != yoe_str and num != notice_str:
            if num not in evidence:
                return False
                
    # Check skill is present verbatim in candidate skills
    if skill_used:
        cand_skills_lower = {s.name.lower() for s in cand.skills}
        if skill_used.lower() not in cand_skills_lower:
            return False
            
    # Check current company
    comp = cand.profile.current_company
    if comp and comp.lower() not in text.lower() and comp.lower() not in evidence:
        return False
        
    return True

def justify_candidate(cand, avail, loc, idx) -> tuple[str, str]:
    """Generates the reasoning for Variant A. Returns (reasoning_string, template_skeleton)."""
    p = cand.profile
    yoe = p.years_of_experience or 0
    years = f"{yoe:.0f} years of experience"
    
    current_company = p.current_company or "current company"
    current_role = next((r for r in cand.career_history if r.is_current), None)
    tenure_months = current_role.duration_months if current_role else 0
    if tenure_months > 0:
        current = f"{tenure_months/12.0:.1f} years at {current_company}"
    else:
        current = f"employment at {current_company}"
        
    notice_days = cand.signals.notice_period_days
    if notice_days is not None and notice_days > 0:
        notice = f"a {notice_days}-day notice period"
    else:
        notice = "immediate availability"
        
    evidence = features.evidence_text(cand).lower()
    has_eval_exp = any(k in evidence for k in ["ndcg", "mrr", "map", "a/b testing", "offline evaluation", "online evaluation", "evaluation framework", "correlation"])
    
    skill_val, skill_jd = get_grounded_skill(cand)
    if not skill_jd:
        skill_jd = "applied machine learning experience"
        
    concern_type, concern = get_genuine_concern(cand, has_eval_exp)
    
    t_idx = idx % len(TEMPLATES)
    template = TEMPLATES[t_idx]
    
    reasoning = template.format(
        years=years,
        current=current,
        skill_jd=skill_jd,
        notice=notice,
        concern=concern
    )
    
    if not verify_grounding(reasoning, cand, skill_val):
        reasoning = f"{years}. {p.current_title} at {current_company}. {notice}. Concern: {concern}."
        template = "Fallback structure"
        
    return _clean(reasoning), template

def generate_variant_a(top, mods) -> tuple[dict[str, str], dict[str, int]]:
    """Return (reasoning dict, template usage count)."""
    reasons = {}
    skeleton_counts = {}
    
    for idx, (cid, _) in enumerate(top):
        cand, avail, loc = mods[cid]
        reason, skeleton = justify_candidate(cand, avail, loc, idx)
        reasons[cid] = reason
        skeleton_counts[skeleton] = skeleton_counts.get(skeleton, 0) + 1
        
    return reasons, skeleton_counts
