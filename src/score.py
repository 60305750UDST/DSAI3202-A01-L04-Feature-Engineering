import json
import os
import glob
import joblib
import numpy as np

model = None

def init():
    global model
    model_dir = os.environ["AZUREML_MODEL_DIR"]
    model_files = glob.glob(os.path.join(model_dir, "**", "model.pkl"), recursive=True)
    if not model_files:
        raise FileNotFoundError(f"No model.pkl found in {model_dir}")
    model = joblib.load(model_files[0])

def run(raw_data):
    try:
        data = json.loads(raw_data)
        X = np.array(data["data"])
        preds = model.predict(X)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": str(e)}
