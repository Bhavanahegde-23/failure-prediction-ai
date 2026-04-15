import requests
import time
import random

def generate_system_data():
    if random.random() < 0.3:  # 30% chance of risky system
        return {
            "Air temperature [K]": random.randint(320, 340),
            "Process temperature [K]": random.randint(330, 350),
            "Rotational speed [rpm]": random.randint(2500, 3200),
            "Torque [Nm]": random.randint(60, 90),
            "Tool wear [min]": random.randint(200, 300)
        }
    else:
        return {
            "Air temperature [K]": random.randint(290, 310),
            "Process temperature [K]": random.randint(300, 320),
            "Rotational speed [rpm]": random.randint(1200, 1800),
            "Torque [Nm]": random.randint(20, 50),
            "Tool wear [min]": random.randint(0, 150)
        }
API_URL = "http://127.0.0.1:8000/predict"

def check_system(system_data):
    try:
        response = requests.post(API_URL, json=system_data)
        result = response.json()

        prob = result["probability"]

        print(f"📊 Failure Probability: {prob:.2f}")

        if prob > 0.7:
            print("⚠️ High risk detected!")
            take_action(system_data, prob)
        else:
            print("✅ System is stable")

    except Exception as e:
        print("❌ Error:", e)


def take_action(data, prob):
    print("🔧 Taking automated actions...")

    # Simulated actions
    restart_service()
    send_alert(prob)
    log_event(data, prob)


def restart_service():
    print("🔄 Restarting service...")


def send_alert(prob):
    print(f"📩 Alert sent! Failure risk = {prob:.2f}")


def log_event(data, prob):
    with open("logs.txt", "a") as f:
        f.write(f"ALERT | Prob: {prob} | Data: {data}\n")


# 🔁 Simulate continuous monitoring
if __name__ == "__main__":
    sample_data = generate_system_data()

    while True:
        print("\n🔍 Checking system...")
        check_system(sample_data)
        time.sleep(5)