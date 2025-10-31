import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE

# ----------------------------
# 1️⃣ Load dataset
# ----------------------------
df = pd.read_csv("dataset.csv")
df = df.loc[:, ~df.columns.str.contains('Unnamed')]
df.dropna(inplace=True)

# ----------------------------
# 2️⃣ Scale numeric features
# ----------------------------
target_col = 'target'
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove(target_col)

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ----------------------------
# 3️⃣ Split dataset
# ----------------------------
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 4️⃣ Balance data with SMOTE
# ----------------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ----------------------------
# 5️⃣ Train multiple models
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
    "Random Forest": RandomForestClassifier(class_weight="balanced"),
    "AdaBoost": AdaBoostClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, zero_division=1))

# ----------------------------
# 6️⃣ Ensemble Model
# ----------------------------
voting_clf = VotingClassifier(
    estimators=[
        ("LR", LogisticRegression(max_iter=1000)),
        ("DT", DecisionTreeClassifier(class_weight="balanced")),
        ("RF", RandomForestClassifier(class_weight="balanced")),
        ("AB", AdaBoostClassifier())
    ],
    voting="hard"
)
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

ensemble_accuracy = accuracy_score(y_test, y_pred_voting)
results["Voting Classifier"] = ensemble_accuracy
print(f"\nVoting Classifier Accuracy: {ensemble_accuracy:.4f}")
print(classification_report(y_test, y_pred_voting, zero_division=1))

# ----------------------------
# 7️⃣ Save the best model safely
# ----------------------------
best_model_name = max(results, key=results.get)

# ✅ Ensure we save an actual trained model
if best_model_name == "Voting Classifier":
    best_model = voting_clf
else:
    best_model = models[best_model_name]

# Confirm type before saving
print(f"\nSaving model: {best_model_name}, type = {type(best_model)}")

# Save model
joblib.dump(best_model, "heart_disease_model.pkl")

print(f"\n✅ Best model '{best_model_name}' saved successfully as 'heart_disease_model.pkl'")
