import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)

# ----------------------------
# 1. Find CSV file
# ----------------------------
csv_files = glob.glob("data/*.csv")
if not csv_files:
    raise FileNotFoundError("No CSV file found inside the data/ folder.")

csv_path = csv_files[0]
print(f"Using dataset: {csv_path}")

# ----------------------------
# 2. Load dataset
# ----------------------------
df = pd.read_csv(csv_path)
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)

# ----------------------------
# 3. Clean column names
# ----------------------------
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print("\nCleaned columns:", df.columns.tolist())

# ----------------------------
# 4. Create target column
# ----------------------------
# Delayed = 1 if historical delay > 5 minutes, else 0
df['delayed'] = (df['historical_delay_(min)'] > 5).astype(int)

# Drop the raw delay column — keeping it would let the model "cheat"
df = df.drop(columns=['historical_delay_(min)'])

target_col = 'delayed'
print(f"\nTarget column created: '{target_col}'")
print(f"Delayed (1): {df[target_col].sum()}")
print(f"On-time (0): {(df[target_col] == 0).sum()}")

# ----------------------------
# 5. Split features and target
# ----------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

# ----------------------------
# 6. Identify numeric and categorical columns
# ----------------------------
numeric_features     = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nNumeric features:",     numeric_features)
print("Categorical features:", categorical_features)

# ----------------------------
# 7. Preprocessing pipelines
# ----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,     numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ----------------------------
# 8. Full model pipeline
# ----------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# ----------------------------
# 9. Train / test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ----------------------------
# 10. Train
# ----------------------------
print("\nTraining model...")
model.fit(X_train, y_train)
print("Training complete.")

# ----------------------------
# 11. Predict and evaluate
# ----------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f} ({acc*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------
# 12. Visualisations
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Train Delay Prediction — Results", fontsize=14, fontweight='bold')

# Chart 1 — Confusion Matrix
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["On-Time", "Delayed"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

# Chart 2 — Top 10 Feature Importances
feature_names = (
    numeric_features +
    list(model.named_steps["preprocessor"]
             .named_transformers_["cat"]
             .named_steps["onehot"]
             .get_feature_names_out(categorical_features))
)
importances = model.named_steps["classifier"].feature_importances_
feat_df = (pd.DataFrame({"feature": feature_names, "importance": importances})
             .sort_values("importance", ascending=True)
             .tail(10))

axes[1].barh(feat_df["feature"], feat_df["importance"], color="steelblue")
axes[1].set_title("Top 10 Feature Importances")
axes[1].set_xlabel("Importance Score")

# Chart 3 — Class Distribution
counts = df[target_col].value_counts().sort_index()
axes[2].bar(["On-Time", "Delayed"], counts.values,
            color=["steelblue", "tomato"], edgecolor="white", width=0.5)
axes[2].set_title("Class Distribution")
axes[2].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[2].text(i, v + 10, str(v), ha="center", fontsize=11)

plt.tight_layout()
plt.savefig("train_delay_results.png", dpi=150)
plt.show()
print("\nChart saved as train_delay_results.png")

# ----------------------------
# 13. Business impact summary
# ----------------------------
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

print("\n--- Business Impact Summary ---")
print(f"Of {tp + fn} actual delayed trains in the test set:")
print(f"  Correctly flagged as delayed : {tp} ({recall*100:.1f}% recall)")
print(f"  Missed (false negatives)     : {fn}")
print(f"  False alarms (false pos)     : {fp}")
print(f"\nPrecision: {precision*100:.1f}% — when we flag a delay, we are right {precision*100:.1f}% of the time")
print(f"Recall:    {recall*100:.1f}% — we catch {recall*100:.1f}% of all actual delays")