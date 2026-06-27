"""Tunable constants for the candidate ranking pipeline.

One place for every knob. Each constant carries the one reason it exists.
Logic elsewhere must read from here, not inline numbers.
"""

from datetime import date

SEED = 42  # single seed for numpy, sklearn, faiss so runs are reproducible
REFERENCE_DATE = date(2026, 6, 24)  # dataset "today" for recency, keeps runs deterministic

# Paths
DATA_PATH = "candidates.jsonl"  # 100k newline-delimited candidate records
JD_PATH = "job_description.txt"  # the Senior AI Engineer JD, embedded once at runtime
MODELS_DIR = "models"  # vendored model weights, gitignored, loaded offline only
EMBED_DIR = "models/embed"  # local copy of the embedding model loaded by rank
CROSS_DIR = "models/cross"  # local copy of the cross-encoder loaded by rank
ARTIFACTS_DIR = "artifacts"  # precomputed indexes and feature caches
EMB_NPY = "artifacts/embeddings.npy"  # persisted candidate evidence vectors
FAISS_PATH = "artifacts/faiss.index"  # persisted dense index over the vectors
BM25_PATH = "artifacts/bm25.pkl"  # persisted lexical index over the same text
IDS_PATH = "artifacts/id_order.json"  # candidate_id order aligning rows to vectors
OUTPUT_CSV = "submission.csv"  # final ranking, must pass validate_submission.py
VALIDATOR = "validate_submission.py"  # run after writing to self-check the output

# Models (downloaded once in precompute, then loaded from MODELS_DIR with no network)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small CPU embedder, no query prefix needed
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # light reranker, shortlist only
EMBED_DIM = 384  # all-MiniLM-L6-v2 output width, fixes the FAISS index dimension
EMBED_BATCH = 256  # encode batch size that fits 100k texts in 16 GB RAM

# Reasoning path (the one-line justification per ranked candidate)
REASONING_MODE = "template"  # "template" (default, deterministic) or "llm" (local flan-t5, offline)
FLAN_MODEL = "google/flan-t5-small"  # tiny seq2seq vendored in precompute, writes the line in llm mode
FLAN_DIR = "models/flan"  # local copy loaded by rank with no network, top 100 only
LLM_MAX_NEW_TOKENS = 48  # one sentence is enough, caps per-candidate generation time
RUNTIME_WARN_S = 240  # warn if total rank.py runtime exceeds about 4 minutes

# Retrieval and fusion
BM25_TOPK = 1500  # lexical candidates pulled before fusion
DENSE_TOPK = 1500  # dense candidates pulled before fusion
RRF_K = 60  # reciprocal rank fusion smoothing constant, the standard default
SHORTLIST = 1500  # survivors handed to the cross-encoder, bounds rerank wall time

# Aspect reranking: short facet queries phrased from what the JD MEANS, not its
# skill list, so a candidate is matched on demonstrated work. Edit freely.
FACETS = (
    "Built and shipped an end to end ranking, search, or recommendation system to real users in production.",
    "Deep hands on experience with text embeddings, retrieval, and vector search at production scale.",
    "Applied machine learning at a product company building its own product, not an IT services or consulting firm and not a pure academic research lab.",
)
FACET_WEIGHTS = (1.0, 1.0, 0.8)  # facets 1 and 2 are the decisive fit signals, 3 supporting
RELEVANCE_TOP_FACETS = 3  # average all three discriminating facets

# Hands-on recency (structural, not a cross-encoder facet): the JD rejects seniors
# who stopped writing code for architecture or management. Reward a candidate
# currently in an individual-contributor engineering role, penalize pure
# management or architecture, fade to neutral for older roles.
RECENCY_WEIGHT = 0.10  # max fractional score adjustment from hands-on recency
RECENT_CODE_MONTHS = 18  # JD's hands-on window; a role ended longer ago fades to neutral
IC_ENGINEERING_HINTS = (
    "engineer", "developer", "scientist", "programmer", "sde", "sre",
)  # titles implying hands-on individual-contributor engineering
MANAGEMENT_HINTS = (
    "manager", "head", "director", "vp", "vice president", "chief",
    "cto", "ceo", "president", "architect",
)  # pure management or architecture, checked first so "Engineering Manager" reads as management

# Integrity gate
ISO_SUBSAMPLE = 20000  # isolation forest training sample, large enough to avoid masking
ISO_CONTAMINATION = 0.05  # assumed suspicious fraction, sets the outlier threshold
ISO_ESTIMATORS = 200  # trees in the forest, trades stability against fit time
# Hard contradictions exclude on their own (no forest corroboration): these are
# within-profile impossibilities, not statistical oddities.
HARD_TENURE_MARGIN_MONTHS = 12  # summed role months over stated experience by a full year cannot be rounding
DURATION_TOLERANCE_MONTHS = 2  # slack between a role duration and its start/end span
COMPANY_PREDATE_MONTHS = 60  # a lone start predating the company's next-earliest by this much is implausible
# Soft anomaly (excluded only with forest corroboration, so genuinely undocumented
# history never floors a real senior on its own): claimed experience far exceeding
# the total documented career. The fake tell missed by the tenure check above,
# which only looked at the opposite direction (sum exceeding experience).
PHANTOM_EXPERIENCE_MARGIN_MONTHS = 36  # years_of_experience beyond documented tenure past this is suspect
MAX_EXCLUSION_FRACTION = 0.15  # if the gate drops more than this share of the pool, stop and surface first

# Modifiers (cut points are pool percentiles, not absolute thresholds)
INACTIVE_PCTL = 0.90  # last-active gap beyond this pool percentile flags staleness
RESPONSE_FLOOR_PCTL = 0.25  # recruiter response below this percentile signals low availability
AVAILABILITY_WEIGHT = 0.15  # max fractional score adjustment from behavioral availability
LOCATION_WEIGHT = 0.10  # max fractional score adjustment from JD location fit

# Applied-ML experience (additive signal): the JD's ideal has 4 to 5 years
# specifically in applied ML/AI roles, distinct from total years_of_experience.
APPLIED_ML_HINTS = (
    "machine learning", "deep learning", "ml engineer", "ai engineer", "ai/ml",
    "applied scientist", "data scientist", "nlp", "natural language",
    "search", "ranking", "learning to rank", "recommendation", "recommender",
    "recsys", "retrieval", "embedding", "information retrieval",
)  # a role whose title or description matches these counts toward applied-ML years
APPLIED_ML_TARGET_YEARS = 4.0  # JD's "4 to 5 years applied ML"; the signal saturates here
APPLIED_ML_WEIGHT = 0.05  # max additive score bonus, small so relevance stays primary

# Impact (additive signal): reward candidates who quantify outcomes over those
# listing generic duties. Read only from career_history descriptions, never the
# skills list. Three components combined to one score in [0, 1], then nudged in.
IMPACT_WEIGHT = 0.03  # max additive bonus, below applied-ML so cross-encoder relevance stays primary
IMPACT_COMPONENT_WEIGHTS = (0.4, 0.3, 0.3)  # metric density, action-verb strength, scale indicators
IMPACT_METRIC_FULL = 8  # distinct quantified metrics (numbers, percents, dollars) earning full density marks
IMPACT_VERB_FULL = 3  # net strong-minus-weak verb hits earning full action-strength marks
IMPACT_SCALE_FULL = 3  # distinct scale mentions (users, throughput, team size, data volume) for full marks
IMPACT_STRONG_VERBS = (
    "architected", "scaled", "shipped", "reduced", "optimized", "launched", "led",
    "built", "designed", "drove", "delivered", "increased", "improved", "automated",
)  # outcome verbs signaling ownership of a result
IMPACT_WEAK_VERBS = (
    "helped", "assisted", "worked on", "responsible for", "involved in",
    "contributed to", "participated in", "supported",
)  # duty-listing phrases signaling no clear owned outcome

# Title chasing (soft penalty, never a floor): the JD rejects candidates who hop
# companies roughly every 1.5 years to chase titles.
TITLE_CHASER_TENURE_YEARS = 1.5  # average tenure per company below this reads as chasing
TITLE_CHASER_MIN_COMPANIES = 3  # need a repeated pattern, not one short stint, before flagging
TITLE_CHASER_PENALTY = 0.10  # fractional score reduction when flagged, soft not vetoing

# Hard disqualifiers (floor the final score), applied only when unambiguous. Each
# carries an escape clause so a mixed or borderline history never triggers.
DISQUALIFIER_FLOOR = 1e-4  # floored score, well below any real candidate so it sinks past the top 100
ACADEMIC_HINTS = (
    "university", "institute", "college", "academia", "research lab",
    "laboratory", "national lab", " labs", "iisc", "iit ",
)  # company-name markers of an academic or research-lab affiliation, not a product company
CV_SPEECH_ROBOTICS_HINTS = (
    "computer vision", "image segmentation", "object detection", "opencv",
    "speech recognition", "text to speech", "robotics", "slam", "lidar",
    "point cloud", "image classification", "pose estimation",
)  # primary-domain markers the JD calls a mismatch when no NLP/IR exposure exists
NLP_IR_HINTS = (
    "nlp", "natural language", "information retrieval", "search", "ranking",
    "recommendation", "recommender", "recsys", "retrieval", "embedding",
)  # NLP/IR/search/recsys evidence anywhere cancels the domain-mismatch flag
CV_DOMAIN_MIN_MENTIONS = 2  # repeated CV/speech/robotics evidence required so one stray word never triggers

# Soft signals (config, never hard string gates)
CONSULTING_FIRMS = (
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
)  # career-long consulting-only is a soft negative, weighed not vetoed
TARGET_LOCATIONS = (
    "noida", "pune", "hyderabad", "mumbai", "delhi", "gurgaon",
    "gurugram", "bengaluru", "bangalore", "ncr",
)  # JD-preferred Indian metros for location fit
RESEARCH_ONLY_HINTS = (
    "phd researcher", "postdoc", "research scientist", "research fellow",
)  # pure-research-without-production is a soft negative per the JD

# Output
TOP_N = 100  # rows in the submission, fixed by the challenge
