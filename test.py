import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

SCAM_WORDS = {
    "urgent","verify","click","win","free","offer","congratulations",
    "account","blocked","suspended","limited","act","now",
    "otp","password","login","bank","kyc","refund","reward",
    "claim","security","alert","update","expire"
}

FEATURE_ORDER = [
    'URL_Extraction',
    'NUM_Extraction',
    'Char_len_ex',
    'Special_Char',
    'Word_len'
]

def handle_msg(txt):
    text = txt.lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'\d+', 'NUM', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def feature_extraction(txt):
    return {
        'URL_Extraction': len(re.findall(r'http\S+|www\S+', txt)),
        'NUM_Extraction': len(re.findall(r'\d+', txt)),
        'Char_len_ex': len(txt),
        'Special_Char': len(re.findall(r'[^a-zA-Z0-9\s]', txt)),
        'Word_len' : len(txt.split())
    }

def show_reasons(txt):
    reasons = []

    if re.search(r'http\S+|www\S+', txt):
        reasons.append("Contains URL")

    if re.search(r'\d+', txt):
        reasons.append("Contains numbers / OTP")

    if len(re.findall(r'[^a-zA-Z0-9\s]', txt)) > 20:
        reasons.append("Too many special characters")

    if len(txt) > 200:
        reasons.append("Unusually long message")

    if any(word in SCAM_WORDS for word in txt.lower().split()):
        reasons.append("Uses scam trigger words")

    return reasons
