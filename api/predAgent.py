from langgraph.graph import StateGraph
import requests
import joblib
import pandas as pd
from langchain_ollama.chat_models import ChatOllama
from typing import TypedDict

#url for model prediction
API_URL = "http://127.0.0.1:8000/predict"

#load the model
model = joblib.load("C:/Users/holla/PycharmProjects/PythonProject1/failure-prediction-ai/model/failure_model.pkl")
col = joblib.load("C:/Users/holla/PycharmProjects/PythonProject1/failure-prediction-ai/model/columns.pkl")

#Adding llm
llm = ChatOllama(model="llama3")

#Failure
failure_map = {
    "temp_diff": "Heat Dissipation Issue",
    "power": "Overstrain Failure",
    "wear_rate": "Tool Degradation",
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Issue",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure"
}
#define state
from typing import TypedDict

class State(TypedDict):
    input: dict
    prob: float
    action: str
    root_cause : str
    explain : str
    report : str
    summary:str

#Get prediction
##TODO: add the confidence
def predict_node(state):
    data = state["input"]
    res = requests.post(API_URL , json=data)
    result = res.json()

    state["prob"] = result["probability"]
    print(f"Probability : {state['prob']:.2f}")

    return state

# Decision node
def decision_node(state):
    prob = state["prob"]

    if prob > 0.8:
        state["action"] = "critical"
    elif prob > 0.6:
        state["action"] = "high_risk"
    elif prob > 0.3:
        state["action"] = "moderate"
    else:
        state["action"] = "safe"
    return state

# Action node
def action_node(state):

    print("\n================ FINAL REPORT ================\n")
    print(f"Probability: {round(state['prob'], 2)}")
    print(f"Summary: {state["summary"]}")
    print("\n Root Cause:")
    for rc in state["root_cause"]:
        print(f" - {rc}")
    print("\n Explanation:")
    print(state["explain"])
    print(f"\n Risk Level: {state['action'].upper()}")
    print("\n=============================================\n")
    return state

#explaination node
def diagnose_node(state):
    prob = state["prob"]
    rc = state["root_cause"]
    prompt = f"""
    You are an AI system monitoring machine health.

    Current state:
    - Failure probability: {prob}
    - Root causes: {rc}
    - Input data: {state["input"]}
    - Risk level: {state["action"]}

    Instructions:
    - DO NOT repeat root cause names directly
    - Explain using actual sensor values
    - Be concise (max 120 words)
    - Give ONE actionable recommendation

    Output format:
    - Why risky
    - What could happen
    - Recommendation
    """

    response = llm.invoke(prompt)

    state["report"] = {
    "probability": prob,
    "risk_level": state["action"],
    "root_cause": state["root_cause"],
    "input": state["input"]
     }
    summary = f"{state['action'].upper()} risk due to {', '.join(state['root_cause'][:2])}"
    state["summary"] = summary
    state["explain"] = response.content
    return state

#root cause analysis node
def root_cause_node(state):
    data = state["input"]
    # Convert to DataFrame
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=col, fill_value=0)
    # Get feature importance
    importance = model.feature_importances_
    feature_impact = dict(zip(col, importance))
    # Get top 3 contributing features
    top_features = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)[:3]
    mapped = [failure_map.get(f, f) for f, _ in top_features]
    state["root_cause"] = mapped
    return state

# def root_cause_node(state):
#     data = state["input"]
#
#     # pick top risky features based on value (simple logic)
#     risky = []
#
#     if data["Torque [Nm]"] > 60:
#         risky.append("High Torque")
#
#     if data["Tool wear [min]"] > 200:
#         risky.append("High Tool Wear")
#
#     if data["Rotational speed [rpm]"] > 2500:
#         risky.append("High Speed")
#
#     print(" Root Cause:", risky)
#
#     state["root_cause"] = risky
#     return state

# Build graph
builder = StateGraph(State)
builder.add_node("predict", predict_node)
builder.add_node("root_cause", root_cause_node)
builder.add_node("decision", decision_node)
builder.add_node("explain", diagnose_node)
builder.add_node("action", action_node)
builder.set_entry_point("predict")
builder.add_edge("predict", "root_cause")
builder.add_edge("root_cause", "decision")
builder.add_edge("decision", "explain")
builder.add_edge("explain", "action")
graph = builder.compile()

# Run
if __name__ == "__main__":
    sample = {
        # safe
        # "Type" : "M",
        # "Air temperature [K]": 300,
        # "Process temperature [K]": 310,
        # "Rotational speed [rpm]": 1500,
        # "Torque [Nm]": 40,
        # "Tool wear [min]": 50
        #moderate
        # "Type":"M",
        # "Air temperature [K]": 320,
        # "Process temperature [K]": 330,
        # "Rotational speed [rpm]": 2500,
        # "Torque [Nm]": 70,
        # "Tool wear [min]": 200
        "Type": "M",
        "Air temperature [K]": 330,
        "Process temperature [K]": 340,
        "Rotational speed [rpm]": 3000,
        "Torque [Nm]": 85,
        "Tool wear [min]": 250

    }

    graph.invoke({"input": sample})