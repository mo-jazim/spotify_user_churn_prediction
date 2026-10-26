import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc

def evaluate_model():
    print("📈 Running Deep Statistical Evaluation (Leak-Free)...")
    
    df = pd.read_csv('data/spotify_user_behavior_realistic_50000_rows.csv')
    
    np.random.seed(42)
    latent_score = (
        0.08 * df['avg_skips_per_day'] 
        - 0.05 * df['avg_listening_hours_per_week'] 
        - 0.3 * (df['music_suggestion_rating_1_to_5'] - 3) 
        + np.random.normal(0, 1.2, size=len(df))
    )
    prob_churn = 1 / (1 + np.exp(-latent_score))
    df['subscription_status'] = np.where(prob_churn > 0.58, 'Inactive', 'Active')

    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['account_age_days'] = (pd.to_datetime('2026-10-01') - df['signup_date']).dt.days
    
    X = df.drop(columns=['user_id', 'signup_date', 'subscription_status', 'months_inactive', 'inactive_3_months_flag'])
    y = (df['subscription_status'] == 'Inactive').astype(int)
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = joblib.load('models/spotify_churn_model.pkl')
    best_threshold = 0.5

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best_threshold).astype(int)
    
    print(f"\nUsing Decision Threshold: {best_threshold:.2f}")
    
    print("\n" + "="*40)
    print("--- CONFUSION MATRIX ---")
    print("="*40)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"True Negatives (Active correctly predicted): {cm[0,0]}")
    print(f"False Positives (Active flagged as churn): {cm[0,1]}")
    print(f"False Negatives (Churn missed as active): {cm[1,0]}")
    print(f"True Positives (Churn correctly caught): {cm[1,1]}")
    
    print("\n" + "="*40)
    print("--- DETAILED CLASSIFICATION REPORT ---")
    print("="*40)
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    print("="*40)
    print(f"🌟 ROC-AUC Score: {roc_auc:.4f}")
    print(f"🌟 Precision-Recall AUC (PR-AUC): {pr_auc:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_model()