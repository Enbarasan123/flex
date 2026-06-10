import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# =========================================
# PATH SETUP
# ── Project structure:
#    new project/
#    └── backend/
#        ├── data/
#        │   ├── calories.csv
#        │   └── exercise.csv
#        ├── model/          ← created automatically
#        ├── app.py
#        └── trainmodel.py   ← this file
# =========================================

# __file__ = .../backend/trainmodel.py
# BASE_DIR = .../backend/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CALORIES_PATH = os.path.join(BASE_DIR, "data", "calories.csv")
EXERCISE_PATH = os.path.join(BASE_DIR, "data", "exercise.csv")
MODEL_DIR     = os.path.join(BASE_DIR, "model")

# ── Quick check before doing anything ──
for path, label in [(CALORIES_PATH, "calories.csv"), (EXERCISE_PATH, "exercise.csv")]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n❌  Cannot find {label}\n"
            f"    Expected at: {path}\n"
            f"    Make sure you run this script from inside the 'backend' folder:\n"
            f"      cd backend\n"
            f"      python trainmodel.py\n"
        )

# =========================================
# LOAD DATASETS
# =========================================

print("Loading datasets...")

calories_df = pd.read_csv(CALORIES_PATH)
exercise_df = pd.read_csv(EXERCISE_PATH)

print("Calories Dataset:")
print(calories_df.head())

print("\nExercise Dataset:")
print(exercise_df.head())

# =========================================
# MERGE ON User_ID
# =========================================

df = exercise_df.merge(calories_df, on="User_ID")

# Drop User_ID — not a feature
df = df.drop(columns=["User_ID"])

# Standardise column names to lowercase
df.columns = df.columns.str.lower().str.strip()

print(f"\nMerged shape : {df.shape}")
print("Columns      :", list(df.columns))

# =========================================
# HANDLE MISSING VALUES
# =========================================

df = df.dropna()
print(f"After dropna : {df.shape}")

# =========================================
# LABEL ENCODING
# =========================================

gender_encoder = LabelEncoder()

if "gender" in df.columns:
    df["gender"] = gender_encoder.fit_transform(df["gender"])
    print("Gender classes:", gender_encoder.classes_)   # ['female' 'male']

# =========================================
# FEATURE / TARGET SPLIT
# =========================================

TARGET = "calories"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Keep only numeric columns (safety net)
X = X.select_dtypes(include=["int64", "float64"])

print("\nTraining features :", list(X.columns))
print(f"Total samples     : {len(X)}")

# =========================================
# TRAIN / TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"Train : {len(X_train)}   Test : {len(X_test)}")

# =========================================
# MODEL TRAINING  (XGBoost — R² ≈ 99.95 %)
# =========================================

print("\nTraining XGBoost model ...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# =========================================
# EVALUATION
# =========================================

predictions = model.predict(X_test)

r2  = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\n==============================")
print(f"R² Score  : {round(r2 * 100, 4)} %")
print(f"MAE       : {round(mae, 4)} kcal")
print("==============================")

# =========================================
# FEATURE IMPORTANCE
# =========================================

importance_df = pd.DataFrame({
    "Feature"   : X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop Important Features:")
print(importance_df.to_string(index=False))

# =========================================
# SAVE MODEL  →  backend/model/
# =========================================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model,          os.path.join(MODEL_DIR, "calorie_model.joblib"))
joblib.dump(gender_encoder, os.path.join(MODEL_DIR, "gender_encoder.joblib"))

print(f"\n✅ Model saved   →  {os.path.join(MODEL_DIR, 'calorie_model.joblib')}")
print(f"✅ Encoder saved →  {os.path.join(MODEL_DIR, 'gender_encoder.joblib')}")

# =========================================
# SANITY CHECK  (one sample prediction)
# =========================================

print("\n--- Sanity Check ---")

sample = pd.DataFrame([{
    "gender"    : gender_encoder.transform(["male"])[0],
    "age"       : 30,
    "height"    : 175.0,
    "weight"    : 75.0,
    "duration"  : 30.0,
    "heart_rate": 110.0,
    "body_temp" : 40.0
}])

predicted = model.predict(sample)[0]
print(f"Input  → Male, 30 yrs, 175 cm, 75 kg, 30 min, HR=110, Temp=40°C")
print(f"Output → {round(predicted, 2)} kcal burned")
print("--------------------\n")