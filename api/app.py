from flask import Flask, request, jsonify
from predict import predict_one
from load_artifacts import load_artifacts


app = Flask(__name__)

# warm-load everything at startup
load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)

    company_description = payload.get("company_description", "")
    company_description = str(company_description).strip()

    if not company_description:
        return jsonify({"error": "company_description is required"}), 400

    result = predict_one(company_description)
    return jsonify(result)

@app.get("/")
def root():
    return {
        "message": "NAICS hierarchical prediction API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)