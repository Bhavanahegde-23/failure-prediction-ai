from fastapi import FastAPI
import joblib
import pandas as pd
from predAgent import graph

app = FastAPI()

#load the model
model = joblib.load("C:/Users/holla/PycharmProjects/PythonProject1/failure-prediction-ai/model/failure_model.pkl")
col = joblib.load("C:/Users/holla/PycharmProjects/PythonProject1/failure-prediction-ai/model/columns.pkl")

@app.get("/")
def home():
    return {"message":"Failure Prediction API Running"}

@app.post("/predict")
def predict(data:dict):
    try:
        df = pd.DataFrame([data])

        df = pd.get_dummies(df)

        df = df.reindex(columns=col , fill_value=0)

        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return {
            "failure":int(prediction),
            "probability":float(prob)
        }
    except Exception as e:
        return {"error":str(e)}

@app.post("/get_answer")
def get_answer(data: dict):

    # CALL LANGGRAPH
    result = graph.invoke({"input": data})

    # RETURN FULL REPORT
    return {
        "probability": round(result["prob"], 2),
        "risk_level": result["action"],
        "root_cause": result["root_cause"],
        "summary": result["summary"],
        "explanation": result["explain"]
    }