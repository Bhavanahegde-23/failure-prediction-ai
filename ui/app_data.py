import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/get_answer"

st.set_page_config(page_title="AI Failure Monitoring System", layout="centered")

st.title("🔧 Self-Healing AIOps System")

# ---------------- INPUT ----------------
st.subheader("📥 Input Machine Data")

air_temp = st.slider("Air Temperature (K)", 290, 350, 320)
process_temp = st.slider("Process Temperature (K)", 300, 360, 330)
rpm = st.slider("Rotational Speed (rpm)", 1000, 3500, 2000)
torque = st.slider("Torque (Nm)", 10, 100, 50)
tool_wear = st.slider("Tool Wear (min)", 0, 300, 100)

input_data = {
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rpm,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}

# ---------------- BUTTON ----------------
if st.button("🔍 Analyze System"):

    with st.spinner("Analyzing system..."):

        try:
            response = requests.post(API_URL, json=input_data)

            result = response.json()

            st.subheader("Analysis Report")

            st.metric("Failure Probability", round(result["probability"], 2))

            st.subheader("Summary")
            st.write(result.get("summary", "N/A"))

            st.subheader("Root Cause")
            for rc in result.get("root_cause", []):
                st.write(f"- {rc}")

            st.subheader("Explanation")
            st.write(result.get("explanation", "No explanation"))

            st.subheader("Risk Level")
            st.write(result.get("risk_level", "").upper())

        except Exception as e:
            st.error(f"Error: {e}")