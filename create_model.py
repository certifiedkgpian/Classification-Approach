import pandas as pd
import re
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- LOAD YOUR DATA ---
# Ensure 'classification dataset.xlsx' is in the same folder
try:
    df = pd.read_excel("C:\\Users\\ADMIN\\OneDrive\\Desktop\\classification dataset.xlsx")
    df.columns = df.columns.str.strip()
    print("Dataset loaded...")
except FileNotFoundError:
    print("Error: 'classification dataset.xlsx' not found.")
    exit()

# --- PREPROCESSING ---
def parse_dimensions(section_str):
    try:
        parts = re.split('x|X', str(section_str))
        if len(parts) >= 2: return float(parts[0]), float(parts[1])
        return 0.0, 0.0
    except: return 0.0, 0.0

def categorize_time(seconds):
    if seconds <= 70: return '55s-70s'
    elif seconds <= 90: return '70s-90s'
    elif seconds <= 105: return '90s-105s'
    else: return '105s-120s'

df['L1'], df['L2'] = zip(*df['SECTION'].apply(parse_dimensions))
coating_cols = [col for col in df.columns if 'Avg' in col and 'coating' in col]
df['Coating_Thickness'] = df[coating_cols].mean(axis=1)

# Optimization Logic
df_optimized = df[
    (df['THICKNESS (MM)'] != 4) | 
    ((df['THICKNESS (MM)'] == 4) & (df['IMMERSION TIME (SEC)'] >= 55) & (df['IMMERSION TIME (SEC)'] <= 70))
].copy()

df_optimized['Target_Class'] = df_optimized['IMMERSION TIME (SEC)'].apply(categorize_time)

# --- TRAINING ---
features = ['L1', 'L2', 'THICKNESS (MM)', 'JIG WT (KG)', 'Coating_Thickness']
X = df_optimized[features]
y = df_optimized['Target_Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# --- SAVE ---
joblib.dump({'model': model, 'scaler': scaler, 'features': features}, 'zinc_model.joblib')
print("✅ 'zinc_model.joblib' created successfully!")