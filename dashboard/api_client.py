import requests

API_URL = "https://jwrth-naics-embeddings.hf.space/predict"


def predict_naics(company_description: str) -> dict:
    payload = {"company_description": company_description}
    r = requests.post(API_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()