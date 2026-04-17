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

        # clean data
        df.columns = (
            df.columns
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.strip()
            .str.replace(" ", "_")
        )
        #
        # # feature eng
        # df["temp_diff"] = df["Process_temperature_K"] - df["Air_temperature_K"]
        # df["power"] = df["Torque_Nm"] * df["Rotational_speed_rpm"]
        # df["wear_rate"] = df["Tool_wear_min"] / (df["Tool_wear_min"].max() + 1)
        df["stress_index"] = df["Torque_Nm"] * df["Tool_wear_min"]
        print(df)
        df = pd.get_dummies(df)
        print("COLUMNS USED BY MODEL:", col)
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