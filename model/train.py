import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

#load dataset

df = pd.read_csv("C:/Users/holla/PycharmProjects/PythonProject1/failure-prediction-ai/data/ai4i2020.csv")

#clean data
df.columns = (
    df.columns
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.strip()
    .str.replace(" ", "_")
)



#feature eng
# df["temp_diff"] = df["Process_temperature_K"] - df["Air_temperature_K"]
# df["power"] = df["Torque_Nm"] * df["Rotational_speed_rpm"]/ 100000
df["stress_index"] =  df["Torque_Nm"] *  df["Tool_wear_min"]

df = df.drop(["UDI", "Product_ID"], axis=1)
df = pd.get_dummies(df)
#target column
y = df["Machine_failure"]

#Features
X = df.drop(["Machine_failure", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)

print(X.columns)
from imblearn.over_sampling import SMOTE


#split
X_train ,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)

#Model
# model = RandomForestClassifier(
#     n_estimators=100,
#     random_state=42,
#     class_weight="balanced"
# )
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,  # reduce from 6
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5,
    random_state=42
)
model.fit(X_train,y_train)

import pandas as pd

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print(importance)

#Evaulate
y_prob = model.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, y_prob))
# y_pred = model.predict(X_test)
# print(classification_report(y_test,y_prob))
# accuracy = accuracy_score(y_test, y_prob)
# print("Accuracy:", accuracy)
print(y_train.value_counts())
#save model
joblib.dump(X.columns, "columns.pkl")
joblib.dump(model,"failure_model.pkl")