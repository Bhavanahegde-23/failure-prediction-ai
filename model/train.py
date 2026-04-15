import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import accuracy_score


#load dataset

df = pd.read_csv("C:/Users/holla/PycharmProjects/PythonProject1/failure-prediction-ai/data/ai4i2020.csv")

#target column
y = df["Machine failure"]

#Features
X = df.drop(["Machine failure","UDI","Product ID"] , axis =1)

#convert categorical if needed
X = pd.get_dummies(X,drop_first=True)

#split
X_train ,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

#Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train,y_train)

#Evaulate
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#save model
joblib.dump(X.columns, "columns.pkl")
joblib.dump(model,"failure_model.pkl")