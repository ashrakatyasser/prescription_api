import re
import unicodedata
import torch
import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import util
from app.models.config import MEDICINE_CSV


# =========================
# OCR NORMALIZATION 🔥
# =========================
def normalize_ocr(text):
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()

    # OCR fixes
    fixes = [
        ("0", "o"),
        ("1", "l"),
        ("5", "s"),
        ("rn", "m"),
        ("vv", "w"),
        ("|", "i"),
        ("!", "i"),
    ]

    for wrong, right in fixes:
        text = text.replace(wrong, right)

    # remove noise
    text = re.sub(r"[^a-z0-9\s\-]", "", text)

    return text


# =========================
# VALIDATION
# =========================
def is_valid_drug_name(name):
    bad_words = {
        'tablet', 'mg', 'capsule', 'injection', 'syrup',
        'cream', 'eye', 'drops', 'solution', 'ointment'
    }

    tokens = set(re.findall(r'\w+', name.lower()))
    return not any(word in tokens for word in bad_words)


# =========================
# FILTER TOKENS
# =========================
def is_possible_drug(token):
    blacklist = {
        "amount", "receipt", "tax", "inclusive", "total",
        "book", "date", "time", "cash", "invoice",
        "price", "qty", "batch", "exp", "mfg"
    }

    if len(token) < 3 or len(token) > 25:
        return False

    if not re.search(r"[aeiou]", token):
        return False

    if any(b in token for b in blacklist):
        return False

    return True


# =========================
# LOAD + EMBEDDINGS
# =========================
def load_drug_list(model, device):
    df = pd.read_csv(MEDICINE_CSV)
    df.columns = df.columns.str.strip().str.lower()

    if "medicine name" not in df.columns:
        raise ValueError("Column 'medicine name' not found")

    drugs = df["medicine name"].dropna().astype(str).unique().tolist()

    clean_list = [
        d.strip().lower()
        for d in drugs
        if is_valid_drug_name(d)
    ]

    embeddings = model.encode(
        clean_list,
        convert_to_tensor=True
    ).to(device)

    return clean_list, embeddings


# =========================
# CLEAN TOKEN
# =========================
def clean_token(text):
    return normalize_ocr(text).strip()


# =========================
# NGRAMS
# =========================
def generate_ngrams(tokens):
    seen = set()
    result = []

    for i in range(len(tokens)):
        if len(tokens[i]) >= 3 and tokens[i] not in seen:
            result.append(tokens[i])
            seen.add(tokens[i])

        if i < len(tokens) - 1:
            bigram = tokens[i] + " " + tokens[i + 1]
            if bigram not in seen:
                result.append(bigram)
                seen.add(bigram)

    return result


# =========================
# MATCH CORE
# =========================
def find_best_match(candidate, drug_list, drug_embeddings, model, device):
    token = clean_token(candidate)

    if len(token) < 3:
        return None, 0

    candidates = process.extract(
        token,
        drug_list,
        scorer=fuzz.WRatio,
        limit=10
    )

    if not candidates:
        return None, 0

    token_emb = model.encode(
        token,
        convert_to_tensor=True
    ).to(device)

    best_match = None
    best_score = 0

    for cand_name, fuzzy_score, idx in candidates:
        cand_emb = drug_embeddings[idx]

        semantic_sim = util.cos_sim(token_emb, cand_emb).item() * 100

        # 🔥 flexible overlap
        if token[:3] not in cand_name and cand_name[:3] not in token:
            continue

        # 🔥 reject only if both weak
        if fuzzy_score < 40 and semantic_sim < 40:
            continue

        final_score = (fuzzy_score * 0.6) + (semantic_sim * 0.4)

        if final_score > best_score:
            best_score = final_score
            best_match = cand_name

    return best_match, best_score


# =========================
# MAIN MATCH FUNCTION 🚀
# =========================
def match_ocr_text(
    ocr_text,
    drug_list,
    drug_embeddings,
    model,
    device,
    threshold=40
):
    if not ocr_text:
        return None, 0

    # 🔥 Stage 1: Normalize
    normalized = normalize_ocr(ocr_text)

    # 🔥 Stage 2: Tokenize
    raw_tokens = re.split(r'[\s,.\n\-/]+', normalized)
    tokens = [t for t in raw_tokens if len(t) >= 3]

    # 🔥 Stage 3: Ngrams
    tokens = generate_ngrams(tokens)

    # 🔥 Stage 4: Filter
    tokens = [t for t in tokens if is_possible_drug(t)]

    if not tokens:
        return None, 0

    best_match = None
    best_score = 0

    # 🔥 Stage 5: Matching
    for t in tokens:
        match, score = find_best_match(
            t,
            drug_list,
            drug_embeddings,
            model,
            device
        )

        if match and score > best_score:
            best_score = score
            best_match = match

    # 🔥 Stage 6: Confidence Gate
    if best_score < threshold:
        return None, best_score

    return best_match, best_score