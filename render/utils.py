import numpy as np
import re
from scipy.ndimage import gaussian_filter1d

STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at",
             "to", "for", "of", "with", "by", "as", "is", "are",
             "was", "were", "be", "been", "being", "this", "that",
             "these", "those", "it", "its", "which", "who", "whom"}

# Pure contract-skeleton words: structural scaffolding that carries zero
# legal information on its own. These get the harshest suppression because
# they recur on nearly every page of every contract.
CONTRACT_SKELETON = {
    "agreement", "contract", "party", "parties", "section", "sections",
    "clause", "clauses", "exhibit", "schedule", "appendix", "article",
    "subsection", "paragraph", "page", "hereto",
}
# Legal-domain function/connective words that get moderate suppression.
LEGAL_FUNCTION = {
    "hereby", "herein", "thereof", "thereto", "thereunder", "whereas",
    "pursuant", "such", "any", "all", "each", "either", "both", "neither",
    "will", "would", "could", "should", "have", "has", "had",
    "set", "forth",
}
# We do NOT add "shall"/"may"/"not" to STOPWORDS even though they are common,
# because they carry deontic/conditional meaning. They are handled by the
# type-prior boost in token_type_prior() below.

# High-information token regexes
_RE_NUM      = re.compile(r"^[\$£€]?[\d][\d,\.]*[%]?$")
_RE_DATE     = re.compile(
    r"^(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}-\d{2}-\d{2})$|"
    r"^(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)$",
    re.IGNORECASE,
)
_RE_PROPER   = re.compile(r"^[A-Z][a-zA-Z\.\-']+$")
_RE_ALLCAPS  = re.compile(r"^[A-Z]{2,}$")

DEONTIC_TOKENS  = {"shall", "must", "required", "obligated", "prohibited",
                   "permitted", "entitled", "may"}
NEGATION_TOKENS = {"not", "no", "without", "except", "unless", "neither",
                   "nor", "never"}
CONDITIONAL_TOKENS = {"if", "if,", "provided", "subject", "upon", "when",
                      "whenever", "until", "before", "after"}

# ---- Legal-substance lexicons (used unconditionally, regardless of question) ----
# Tokens that carry "legal force" — what makes the contract binding / where it lives
LEGAL_FORCE = {
    "governing", "governed", "jurisdiction", "venue", "arbitration", "arbitrate",
    "binding", "enforceable", "executed", "execute", "valid", "validity",
    "void", "voidable", "effective", "effectiveness", "force", "law", "laws",
    "court", "courts", "tribunal",
}
# Tokens that carry "legal risk" — exposure, liability, dispute, termination
LEGAL_RISK = {
    "liable", "liability", "liabilities", "indemnify", "indemnification",
    "indemnity", "indemnities", "damages", "damage", "breach", "breaches",
    "default", "defaults", "terminate", "termination", "terminated",
    "terminates", "suspend", "suspension", "claim", "claims", "dispute",
    "disputes", "lawsuit", "penalty", "penalties", "forfeiture", "loss",
    "losses", "remedy", "remedies", "injunction", "material", "materially",
}
# Tokens that constitute the "underlying agreement" — rights/obligations/grants
LEGAL_AGREEMENT = {
    "consideration", "warranty", "warranties", "warrant", "represent",
    "representations", "representation", "covenant", "covenants", "obligation",
    "obligations", "right", "rights", "license", "licensed", "licensee",
    "licensor", "grant", "grants", "granted", "granting", "assign",
    "assignment", "assignments", "transfer", "transferred", "sublicense",
    "exclusive", "non-exclusive", "perpetual", "irrevocable", "royalty",
    "royalties", "fee", "fees", "payment", "payments", "compensation",
    "confidential", "confidentiality", "non-compete", "noncompete",
}
# Temporal / conditional triggers — when obligations attach
LEGAL_TEMPORAL = {
    "unless", "provided", "subject", "condition", "conditions", "contingent",
    "prior", "within", "during", "upon", "term", "period", "expiration",
    "expires", "expire", "renewal", "renew", "notice", "written",
    "automatically", "immediately",
}


def compute_legal_lexicon(words):
    """
    lambda(w) — unconditional legal-substance prior. Independent of which
    CUAD question is asked: it measures whether the token *carries legal
    force, risk, or constitutes part of the underlying agreement*.

    This is the signal that should drive saliency for *any* legal review,
    even before the question is consulted.
    """
    lam = np.ones(len(words), dtype=np.float32)
    for i, w in enumerate(words):
        cleaned = re.sub(r"[\W_]+", "", w.lower())
        if not cleaned:
            continue
        stem = simple_stem(cleaned)
        if cleaned in LEGAL_FORCE or stem in LEGAL_FORCE:
            lam[i] = 3.0
        elif cleaned in LEGAL_RISK or stem in LEGAL_RISK:
            lam[i] = 3.0
        elif cleaned in LEGAL_AGREEMENT or stem in LEGAL_AGREEMENT:
            lam[i] = 2.5
        elif cleaned in LEGAL_TEMPORAL or stem in LEGAL_TEMPORAL:
            lam[i] = 1.8
    return lam

# Tunable type-prior table. Order matters: first match wins.
def token_type_prior(word):
    """
    tau(w) — prior that a token TYPE is informative, regardless of context.
    Numbers, currency, dates, deontic/negation/conditional keywords, and
    proper nouns get boosts; legal boilerplate function words get suppressed.
    """
    cleaned = re.sub(r"[\W_]+$|^[\W_]+", "", word)  # strip surrounding punct
    if not cleaned:
        return 0.0
    low = cleaned.lower()

    if _RE_NUM.match(cleaned):                    return 3.5
    if "$" in word or "%" in word:                return 3.5
    if _RE_DATE.match(cleaned):                   return 3.0
    if low in NEGATION_TOKENS:                    return 2.5
    if low in DEONTIC_TOKENS:                     return 2.2
    if low in CONDITIONAL_TOKENS:                 return 2.0
    if _RE_ALLCAPS.match(cleaned) and len(cleaned) >= 3: return 1.8
    if low in CONTRACT_SKELETON:                  return 0.02
    if low in STOPWORDS:                          return 0.05
    if low in LEGAL_FUNCTION:                     return 0.10
    if _RE_PROPER.match(cleaned):                 return 1.6
    return 1.0


def compute_word_idf(words):
    """
    rho(w) — document-local IDF in [0, 1].
    Suppresses tokens that appear all over THIS document ("Agreement",
    "Party", etc.) without needing a global corpus.
    """
    N = len(words)
    if N == 0:
        return np.zeros(0, dtype=np.float32)
    counts = {}
    keys = []
    for w in words:
        k = simple_stem(re.sub(r"[\W_]+", "", w.lower()))
        keys.append(k)
        counts[k] = counts.get(k, 0) + 1
    denom = np.log(N + 1.0) + 1e-8
    out = np.empty(N, dtype=np.float32)
    for i, k in enumerate(keys):
        out[i] = np.log(1.0 + N / counts[k]) / denom
    # Normalise to ~[0,1]
    mx = out.max() if out.max() > 0 else 1.0
    return out / mx


def stitch_word_signal(full_context, chunks_data, key="probs", reducer="max"):
    """
    Generic stitch: char-level pooling -> word-level mean. Returns
    (words_list, word_signal_array). Used for both QA span coverage and
    question-token relevance.
    """
    n = len(full_context)
    if reducer == "max":
        char_buf = np.full(n, -np.inf, dtype=np.float32)
        for chunk in chunks_data:
            for p, (s, e), seq_id in zip(chunk[key], chunk["offsets"], chunk["sequence_ids"]):
                if seq_id == 1 and s < e and s < n:
                    ee = min(e, n)
                    np.maximum(char_buf[s:ee], float(p), out=char_buf[s:ee])
        char_buf[~np.isfinite(char_buf)] = 0.0
    else:  # mean
        char_sum = np.zeros(n, dtype=np.float32)
        char_cnt = np.zeros(n, dtype=np.int32)
        for chunk in chunks_data:
            for p, (s, e), seq_id in zip(chunk[key], chunk["offsets"], chunk["sequence_ids"]):
                if seq_id == 1 and s < e and s < n:
                    ee = min(e, n)
                    char_sum[s:ee] += float(p)
                    char_cnt[s:ee] += 1
        char_buf = np.divide(char_sum, np.maximum(char_cnt, 1), dtype=np.float32)

    words, weights = [], []
    for m in re.finditer(r"\S+", full_context):
        s, e = m.span()
        words.append(m.group())
        weights.append(float(char_buf[s:e].mean()) if e > s else 0.0)
    return words, np.array(weights, dtype=np.float32)


def composite_saliency(words, qa_word, qrel_word,
                       alpha=0.6, beta=0.9, gamma=0.3, delta=0.8, eps=1e-3):
    """
    phi_i = (S_i+eps)^alpha
          * (tau_i * rho_i + eps)^beta
          * (lambda_i + eps)^delta
          * (kappa_i + eps)^gamma

    - S_i      : stitched QA span coverage     (region-of-interest, QA head)
    - tau_i    : token-type prior              (numbers/dates/deontic up;
                                               stopwords/boilerplate down)
    - rho_i    : document IDF                  (kills "Agreement"/"Party"
                                               because they occur 100x)
    - lambda_i : LEGAL-SUBSTANCE lexicon       (unconditional: legal force,
                                               risk, agreement, temporal
                                               triggers — independent of any
                                               question)
    - kappa_i  : question-token cosine         (now a SOFT topical re-ranker
                                               with small gamma=0.3, not the
                                               primary driver)

    Default exponent rationale: lambda gets the largest exponent (0.8) so
    that legal substance dominates; S and tau*rho are co-equal evidence
    sources; kappa is intentionally small so the (often boilerplate) CUAD
    prompt cannot drag saliency to generic terms.
    """
    if len(words) == 0:
        return np.zeros(0, dtype=np.float32)

    def _mm(x):
        lo, hi = float(np.min(x)), float(np.max(x))
        return (x - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(x)

    S   = _mm(np.asarray(qa_word, dtype=np.float32))
    K   = _mm(np.asarray(qrel_word, dtype=np.float32))
    tau = np.array([token_type_prior(w) for w in words], dtype=np.float32)
    rho = compute_word_idf(words)
    lam = compute_legal_lexicon(words)
    info = tau * rho

    phi = ((S   + eps) ** alpha *
           (info + eps) ** beta  *
           (lam  + eps) ** delta *
           (K    + eps) ** gamma)
    return _mm(phi)


def assign_tiers(phi, primary_pct=98.0, secondary_pct=85.0):
    """
    Sparsify phi into 3 rendering tiers via global percentile cuts.
    Returns array of tier ids in {2 (primary), 1 (secondary), 0 (tertiary)}.
    """
    if len(phi) == 0:
        return np.zeros(0, dtype=np.int8)
    p_hi = float(np.percentile(phi, primary_pct))
    p_md = float(np.percentile(phi, secondary_pct))
    tiers = np.zeros(len(phi), dtype=np.int8)
    tiers[phi >= p_md] = 1
    tiers[phi >= p_hi] = 2
    return tiers

def filter_for_barchart(word):
    cleaned = re.sub(r'[\W_]+', '', word.lower())
    if not cleaned:
        return False
    if cleaned in STOPWORDS:
        return False
    return True

def simple_stem(word):
    w = word.lower()
    if len(w) <= 3: return w
    if w.endswith('ies'): return w[:-3] + 'y'
    if w.endswith('es') and w[-3] in 'szxc': return w[:-2]
    if w.endswith('s') and w[-2] not in 'su': return w[:-1]
    if w.endswith('ing'): return w[:-3]
    if w.endswith('ed'): return w[:-2]
    return w

def group_by_stem_and_sort(word_weights):
    stem_map = {}
    for word, weight in word_weights:
        cleaned = re.sub(r'[\W_]+', '', word.lower())
        if not cleaned or cleaned in STOPWORDS:
            continue
        stem = simple_stem(cleaned)
        if stem not in stem_map:
            stem_map[stem] = {'weight': weight, 'orig': word}
        else:
            if weight > stem_map[stem]['weight']:
                stem_map[stem]['weight'] = weight
            if len(word) < len(stem_map[stem]['orig']):
                stem_map[stem]['orig'] = word
    sorted_stems = sorted(stem_map.values(), key=lambda x: x['weight'], reverse=True)
    return [(item['orig'], item['weight']) for item in sorted_stems]

def _sentence_spans(text):
    spans, start = [], 0
    for m in re.finditer(r'[.!?;\n]+', text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans

def stitch_and_pool_saliency_v3(full_context, chunks_data, sigma=0.8,
                                w_word=0.75, w_phrase=0.25, phrase_win=1,
                                clip_lo=5.0, clip_hi=95.0):
    # Initialise to NaN so chars not covered by any chunk stay distinguishable
    char_probs = np.full(len(full_context), np.nan, dtype=np.float32)

    # Max across overlapping chunks: now that per-chunk saliency is on a
    # globally comparable scale (no per-chunk min-max), the *evidence-richest*
    # chunk for each character should win — averaging would let a low-evidence
    # neighbour drag a true peak down by half.
    char_max = np.full(len(full_context), -np.inf, dtype=np.float32)
    for chunk in chunks_data:
        for p, (start, end), seq_id in zip(chunk['probs'], chunk['offsets'], chunk['sequence_ids']):
            if seq_id == 1 and start < end and start < len(full_context):
                e = min(end, len(full_context))
                if float(p) > char_max[start]:
                    np.maximum(char_max[start:e], float(p), out=char_max[start:e])
    valid = np.isfinite(char_max)
    char_probs[valid] = char_max[valid]
    char_probs = np.nan_to_num(char_probs, nan=float(np.nanmin(char_probs[valid])) if valid.any() else 0.0)

    words, word_w = [], []
    for m in re.finditer(r'\S+', full_context):
        s, e = m.span()
        words.append(m.group())
        word_w.append(float(char_probs[s:e].mean()) if e > s else 0.0)
    if not words:
        return []

    word_w = np.array(word_w, dtype=np.float32)

    # Phrase-level: small local window (default ±1) keeps neighbouring words coherent
    # without flattening whole paragraphs
    n = len(word_w)
    phrase_w = np.zeros_like(word_w)
    for i in range(n):
        a, b = max(0, i - phrase_win), min(n, i + phrase_win + 1)
        phrase_w[i] = word_w[a:b].mean()

    combined = w_word * word_w + w_phrase * phrase_w
    smoothed = gaussian_filter1d(combined, sigma=sigma)

    # Percentile clipping suppresses extreme outliers from any single chunk
    lo = float(np.percentile(smoothed, clip_lo))
    hi = float(np.percentile(smoothed, clip_hi))
    if hi > lo:
        norm = np.clip((smoothed - lo) / (hi - lo), 0.0, 1.0)
    else:
        norm = np.zeros_like(smoothed)
    return list(zip(words, norm.astype(np.float32)))

def stitch_and_smooth_saliency(full_context, chunks_data, sigma=2.0):
    char_probs = np.zeros(len(full_context), dtype=np.float32)
    for chunk in chunks_data:
        for p, (start, end), seq_id in zip(chunk['probs'], chunk['offsets'], chunk['sequence_ids']):
            if seq_id == 1 and start < end and start < len(full_context):
                safe_end = min(end, len(full_context))
                if p > np.max(char_probs[start:safe_end]):
                    char_probs[start:safe_end] = p
    words, word_probs = [], []
    for match in re.finditer(r'\S+', full_context):
        s, e = match.span()
        words.append(match.group())
        word_probs.append(np.max(char_probs[s:e]))
    if not word_probs:
        return []
    smoothed = gaussian_filter1d(np.array(word_probs), sigma=sigma)
    lo, hi = np.min(smoothed), np.max(smoothed)
    norm = (smoothed - lo) / (hi - lo) if hi > lo else smoothed
    return list(zip(words, norm))
