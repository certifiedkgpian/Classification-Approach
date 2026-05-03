import pandas as pd
import re
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 
# =========================
# 1. LOAD DATA
# =========================
df = pd.read_excel("C:\\Users\\ADMIN\\OneDrive\\Desktop\\FinalSheetSkipper.xlsx")
df.columns = df.columns.str.strip()
print(f"Dataset loaded with {len(df)} rows")
 
# =========================
# 2. PREPROCESSING
# =========================
def parse_dimensions(section_str):
    try:
        parts = re.split('x|X', str(section_str))
        return float(parts[0]), float(parts[1])
    except:
        return np.nan, np.nan
 
def categorize_time(seconds):
    if 55 <= seconds <= 70:
        return '55s-63s'
    elif 70 <= seconds <= 90:
        return '70s-78s'
    elif 90 <= seconds <= 105:
        return '90s-98s'
    elif 105 <= seconds <= 115:
        return '105s-115s'
    else:
        return np.nan
 
df[['L1', 'L2']] = df['SECTION'].apply(parse_dimensions).apply(pd.Series)
 
coating_cols = [c for c in df.columns if 'Avg' in c and 'coating' in c]
df['Coating_Thickness'] = df[coating_cols].mean(axis=1)
 
# =========================
# 3. DOMAIN FILTERING
# =========================
df = df[
    (df['THICKNESS (MM)'] != 4) |
    ((df['THICKNESS (MM)'] == 4) &
     (df['IMMERSION TIME (SEC)'].between(55, 70)))
].copy()
 
df['Target_Class'] = df['IMMERSION TIME (SEC)'].apply(categorize_time)
df.dropna(subset=['Target_Class', 'L1', 'L2', 'Coating_Thickness'], inplace=True)
 
print(f"Rows after cleaning & filtering: {len(df)}")
 
# =========================
# 4. TRAIN / TEST SPLIT
# =========================
features = ['L1', 'L2', 'THICKNESS (MM)', 'JIG WT (KG)', 'Coating_Thickness']
X = df[features]
y = df['Target_Class']
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
 
# =========================
# 5. SCALING (NO LEAKAGE)
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# =========================
# 6. MODEL TRAINING
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
 
model.fit(X_train_scaled, y_train)
 
# =========================
# 7. VALIDATION
# =========================
print("\nModel performance on unseen data:\n")
print(classification_report(y_test, model.predict(X_test_scaled)))
 
# =========================
# 8. SAVE MODEL
# =========================
joblib.dump(
    {
        'model': model,
        'scaler': scaler,
        'features': features
    },
    'zinc_model.joblib'
)
 
print("✅ Updated zinc_model.joblib saved successfully")