import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_pipeline.pkl"

app = FastAPI(title="ASD Screening API")

# ---- load model bundle once at startup ----
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
encoders = bundle["encoders"]
outlier_bounds = bundle["outlier_bounds"]
feature_order = bundle["feature_order"]
threshold = bundle["decision_threshold"]


class ScreeningInput(BaseModel):
    A1_Score: int = Field(..., ge=0, le=1)
    A2_Score: int = Field(..., ge=0, le=1)
    A3_Score: int = Field(..., ge=0, le=1)
    A4_Score: int = Field(..., ge=0, le=1)
    A5_Score: int = Field(..., ge=0, le=1)
    A6_Score: int = Field(..., ge=0, le=1)
    A7_Score: int = Field(..., ge=0, le=1)
    A8_Score: int = Field(..., ge=0, le=1)
    A9_Score: int = Field(..., ge=0, le=1)
    A10_Score: int = Field(..., ge=0, le=1)
    age: int = Field(..., ge=1, le=120)
    gender: str
    ethnicity: str
    jaundice: str
    austim: str
    contry_of_res: str
    used_app_before: str
    result: float
    relation: str


def encode_value(col: str, value: str):
    le = encoders[col]
    classes = set(le.classes_)
    if value in classes:
        return int(le.transform([value])[0])
    return len(le.classes_)  # unseen-category fallback code


def apply_outlier_bounds(col: str, value: float):
    low, high, median = outlier_bounds[col]
    if value < low or value > high:
        return median
    return value


@app.post("/predict")
def predict(payload: ScreeningInput):
    try:
        row = payload.model_dump()

        for col in ["gender", "ethnicity", "jaundice", "austim", "contry_of_res", "used_app_before", "relation"]:
            row[col] = encode_value(col, row[col])

        row["age"] = apply_outlier_bounds("age", row["age"])
        row["result"] = apply_outlier_bounds("result", row["result"])

        X = pd.DataFrame([row])[feature_order]
        proba = float(model.predict_proba(X)[0, 1])
        prediction = int(proba >= threshold)

        return {
            "probability": round(proba, 4),
            "prediction": prediction,
            "label": "Likely ASD traits" if prediction == 1 else "Unlikely ASD traits",
            "threshold_used": round(threshold, 4),
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Unknown category value: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))
