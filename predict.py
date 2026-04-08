import joblib
import pandas as pd
import numpy as np
import re
import unicodedata
from scipy.sparse import hstack, csr_matrix

# --- Load artifacts ---
model = joblib.load('best_model/classifier_model.pkl')
tfidf = joblib.load('best_model/tfidf_vectorizer.pkl')
le = joblib.load('best_model/label_encoder.pkl')
scaler = joblib.load('best_model/keyword_scaler.pkl')

def clean_address(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    if re.search(r'^\s*\{.*\}\s*$', text):
        return "garbage_entry"
    text = unicodedata.normalize('NFKD', text)
    replacements = {
        '\u201c': '"', '\u201d': '"',
        '\u2018': "'", '\u2019': "'",
        '\u2013': '-', '\u2014': '-',
        '\ufffd': '',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.encode('ascii', errors='ignore').decode('ascii')
    text = text.lower()
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bna\b', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    abbreviation_map = {
        r'\bs\.?\s*no\.?\b': 'survey_no',
        r'\bsy\.?\s*no\.?\b': 'survey_no',
        r'\bsy\.?\b': 'survey',
        r'\br\.?\s*s\.?\s*no\.?\b': 'rs_no',
        r'\bt\.?\s*s\.?\s*no\.?\b': 'ts_no',
        r'\bf\.?\s*p\.?\b': 'final_plot',
        r'\bt\.?\s*p\.?\b': 'town_plan',
        r'\bc\.?\s*s\.?\s*no\.?\b': 'cs_no',
        r'\bno\.': 'no',
        r'\bflr\b': 'floor',
        r'\bapt\b': 'apartment',
        r'\bofc\b': 'office',
        r'\bsoc\b': 'society',
        r'\bvill\b': 'village',
        r'\bdist\b': 'district',
        r'\bteh\b': 'tehsil',
        r'\btah\b': 'tahsil',
        r'\bnr\b': 'near',
        r'\bopp\b': 'opposite',
        r'\bchsl?\b': 'chs',
        r'\bsq\.\s*ft': 'sqft',
        r'\bsq\.\s*mt': 'sqmt',
        r'\bsq\.\s*yd': 'sqyd',
    }
    for pattern, replacement in abbreviation_map.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'\b\d{6}\b', '', text)
    text = re.sub(r'\b\d{4,}\b', '', text)
    text = re.sub(r'[,;:()\/\\\"\'.]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text.strip()) < 3:
        return "garbage_entry"
    return text


def extract_keyword_features(text):
    t = text.lower()
    features = {
        'has_flat': int(bool(re.search(r'\bflat\b', t))),
        'has_wing': int(bool(re.search(r'\bwing\b', t))),
        'has_apartment': int(bool(re.search(r'\b(apartment|apt)\b', t))),
        'has_chs': int(bool(re.search(r'\b(chs|chsl|society)\b', t))),
        'has_floor': int(bool(re.search(r'\bfloor\b', t))),
        'has_tower': int(bool(re.search(r'\b(tower|block)\b', t))),
        'has_house': int(bool(re.search(r'\b(house|bungalow|duplex|villa|row house|kothi)\b', t))),
        'has_plot': int(bool(re.search(r'\bplot\b', t))),
        'has_scheme': int(bool(re.search(r'\b(scheme|colony|enclave|nagar|vihar|layout)\b', t))),
        'has_shop': int(bool(re.search(r'\b(shop|stall|showroom|godown)\b', t))),
        'has_office': int(bool(re.search(r'\b(office|commercial)\b', t))),
        'has_complex_market': int(bool(re.search(r'\b(complex|market|mall|plaza|arcade)\b', t))),
        'has_khasra': int(bool(re.search(r'\b(khasra|khata|khatian|dag|patta)\b', t))),
        'has_survey': int(bool(re.search(r'\b(survey|survey_no)\b', t))),
        'has_mouza': int(bool(re.search(r'\b(mouza|mauza)\b', t))),
        'has_gat_gut': int(bool(re.search(r'\b(gat|gut)\b', t))),
        'has_land_terms': int(bool(re.search(r'\b(mandal|taluka|tehsil|tahsil|village|gram|panchayat)\b', t))),
        'has_any_structure': int(bool(re.search(r'\b(flat|house|shop|office|apartment|bungalow|duplex|villa|stall|showroom|floor|wing|tower)\b', t))),
        'is_garbage': int(t in ['garbage_entry', '']),
        'text_length': len(t),
        'word_count': len(t.split()),
        'digit_ratio': sum(c.isdigit() for c in t) / max(len(t), 1),
        'has_unit': int(bool(re.search(r'\bunit\b', t))),
    }
    return features


def predict(addresses):
    """Predict categories for a list of raw property address strings."""
    cleaned = [clean_address(addr) for addr in addresses]
    X_tfidf = tfidf.transform(cleaned)
    kw_features = pd.DataFrame([extract_keyword_features(t) for t in cleaned])
    kw_scaled = scaler.transform(kw_features)
    X = hstack([X_tfidf, csr_matrix(kw_scaled)])
    predictions = model.predict(X)
    return le.inverse_transform(predictions)

if __name__ == "__main__":
    test_addresses = [
        "Flat-301, Floor-3, A-Wing, Sarthana Jakatnaka Surat 395006 Gujarat",
        "Plot No. 107, Scheme Jamana Vihar, Jagatpura, Jaipur",
        "Shop No 850, Shradhha Complex, First Floor, Rajkot Gujarat",
        "Sy. No. 1388/1, Mandal Mangalagiri, Near Old Shiv Mandir",
        "Test entry with nothing useful",
    ]
    preds = predict(test_addresses)
    for addr, pred in zip(test_addresses, preds):
        print(f"  {pred:20s} | {addr[:80]}")