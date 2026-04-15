from langgraph.graph import StateGraph
import requests

API_URL = "http://127.0.0.1:8000/predict"

#define state
from typing import TypedDict

class State(TypedDict):
    input: dict
    prob: float
    action: str
#Get prediction
def predict_node(state):
    data = state["input"]

    res = requests.post(API_URL , json=data)
    result = res.json()

    state["prob"] = result["probability"]
    print(f"Probability : {state['prob']:.2f}")

    return state

# Step 2: Decision node
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

# Step 3: Action node
def action_node(state):
    action = state["action"]

    if action == "critical":
        print("🚨 CRITICAL → Restarting system")
    elif action == "warning":
        print("⚠️ Warning → Sending alert")

    elif action == "moderate":
        print("📊 MODERATE RISK → Monitoring closely")
    else:
        print("✅ System stable")

    return state

# from langchain.chat_models import ChatOpenAI
#
# llm = ChatOpenAI()
#
# def diagnose_node(state):
#     prob = state["prob"]
#
#     explanation = llm.invoke(
#         f"System failure probability is {prob}. Explain cause."
#     )
#
#     print("🧠 Diagnosis:", explanation)
#     return state
# def root_cause_node(state):
#     data = state["input"]
#
#     # Convert to DataFrame
#     import pandas as pd
#     df = pd.DataFrame([data])
#     df = pd.get_dummies(df)
#     df = df.reindex(columns=columns, fill_value=0)
#
#     # Get feature importance
#     importance = model.feature_importances_
#
#     feature_impact = dict(zip(columns, importance))
#
#     # Get top 3 contributing features
#     top_features = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)[:3]
#
#     print("🧠 Root Cause Analysis:")
#     for f, val in top_features:
#         print(f" - {f}")
#
#     state["root_cause"] = top_features
#     return state
# builder.add_node("root_cause", root_cause_node)
#
# builder.add_edge("predict", "root_cause")
# builder.add_edge("root_cause", "decision")
# Build graph
builder = StateGraph(State)

builder.add_node("predict", predict_node)
builder.add_node("decision", decision_node)
builder.add_node("action", action_node)

builder.set_entry_point("predict")

builder.add_edge("predict", "decision")
builder.add_edge("decision", "action")

graph = builder.compile()

# Run
if __name__ == "__main__":
    sample = {
        "Air temperature [K]": 330,
        "Process temperature [K]": 340,
        "Rotational speed [rpm]": 3000,
        "Torque [Nm]": 80,
        "Tool wear [min]": 250
    }

    graph.invoke({"input": sample})