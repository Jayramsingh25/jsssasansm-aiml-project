import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration
INPUT_FILE = 'aiml traffic data.csv'
TARGET = 'Optimized_Green_Time_Sec'

def load_data():
    try:
        print("Reading file structure...")
        # Reading as Excel to handle potential binary/format mismatches
        data = pd.read_excel(INPUT_FILE, engine='openpyxl')
        return data
    except Exception:
        return pd.read_csv(INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    print(f"File not found: {INPUT_FILE}")
else:
    df = load_data()
    df.columns = df.columns.str.strip()
    
    # Preprocessing
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])
    
    # Encoding categorical variables (Lanes and Weather)
    df_encoded = pd.get_dummies(df, columns=['Lane_ID', 'Weather'])
    
    # Define features and target
    X = df_encoded.drop(TARGET, axis=1)
    y = df_encoded[TARGET]
    
    # --- STEP 1: TRAIN-TEST SPLIT ---
    # We set aside 20% of data to test the model's accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- STEP 2: MODEL TRAINING ---
    print(f"Training on {len(X_train)} samples...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # --- STEP 3: EVALUATION ---
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    print(f"✅ Model Accuracy (R2 Score): {accuracy:.4f}")
    print(f"✅ Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f} seconds")

    # --- STEP 4: VISUALIZATION WITH SEABORN ---
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Plotting Actual vs Predicted Values
    plot = sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.xlabel("Actual Green Time (Seconds)")
    plt.ylabel("Predicted Green Time (Seconds)")
    plt.title("Model Prediction Performance: Actual vs Predicted")
    plt.show()

    # Feature Importance Graph
    plt.figure(figsize=(10, 6))
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=importances, y=importances.index, palette="viridis")
    plt.title("Key Factors Influencing Signal Timing")
    plt.show()

    # --- STEP 5: SAVE ARTIFACTS ---
    joblib.dump(model, 'traffic_model.pkl')
    joblib.dump(X.columns.tolist(), 'model_columns.pkl')
    joblib.dump(df['Lane_ID'].unique().tolist(), 'lanes.pkl')
    joblib.dump(df['Weather'].unique().tolist(), 'weather.pkl')
    
    print("✅ All artifacts saved. Ready for Streamlit/Flask deployment.")