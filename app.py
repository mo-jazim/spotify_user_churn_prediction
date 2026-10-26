import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Spotify Churn Intelligence Platform",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL & ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    model_path = 'models/spotify_churn_model.pkl'
    if not os.path.exists(model_path):
        return None, None
    model = joblib.load(model_path)
    features = joblib.load('models/feature_columns.pkl')
    return model, features

model, feature_columns = load_artifacts()

# --- CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1DB954; font-weight: 700; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #666; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<p class="main-header">🎵 Spotify User Churn Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Production-grade Machine Learning application for proactive subscriber retention & behavioral analysis.</p>', unsafe_allow_html=True)

# Check if model exists
if model is None:
    st.warning("⚠️ Trained model not found! Please run `python src/train.py` in your terminal first to generate model artifacts.")
    st.stop()

# --- NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["🔮 Single User Predictor", "📂 Batch CSV Scoring", "📊 Model Performance & Insights"])

# ==========================================
# TAB 1: SINGLE USER PREDICTOR
# ==========================================
with tab1:
    st.sidebar.header("👤 User Profile & Behavior")

    def user_input_features():
        country = st.sidebar.selectbox("Country", ['France', 'Indonesia', 'Italy', 'Australia', 'Brazil', 'USA', 'Germany', 'India', 'UK', 'Canada'])
        age = st.sidebar.slider("Age", 14, 75, 26)
        signup_date = st.sidebar.date_input("Signup Date", datetime.date(2023, 1, 15))
        
        subscription_type = st.sidebar.selectbox("Subscription Type", ['Free', 'Premium Individual', 'Student', 'Premium Family', 'Premium Duo'])
        ad_interaction = st.sidebar.selectbox("Ad Interaction (Free Tier)", ['No', 'Yes'])
        ad_conversion_to_subscription = st.sidebar.selectbox("Converted from Ad Offer", ['No', 'Yes'])
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎧 Listening Engagement")
        avg_listening_hours_per_week = st.sidebar.slider("Avg Listening Hours / Week", 0.0, 50.0, 12.5)
        playlists_created = st.sidebar.slider("Playlists Created", 0, 50, 8)
        avg_skips_per_day = st.sidebar.slider("Avg Skips / Day", 0, 40, 6)
        music_suggestion_rating_1_to_5 = st.sidebar.slider("Music Suggestion Rating (1-5)", 1, 5, 4)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎶 Preferences & Device")
        favorite_genre = st.sidebar.selectbox("Favorite Genre", ['Bollywood', 'Latin', 'Electronic', 'Indie', 'Pop', 'Rock', 'Hip-Hop', 'Classical'])
        most_liked_feature = st.sidebar.selectbox("Most Liked Feature", ['Radio', 'Podcasts', 'Lyrics', 'Playlists', 'Daily Mix'])
        desired_future_feature = st.sidebar.selectbox("Desired Future Feature", ['Concert Alerts', 'Lyrics Translation', 'Better AI Recommendations', 'Social Listening'])
        primary_device = st.sidebar.selectbox("Primary Device", ['Tablet', 'Mobile', 'Desktop', 'Smart Speaker', 'Smartwatch'])
        
        account_age_days = (datetime.date(2026, 10, 1) - signup_date).days
        
        data = {
            'country': country,
            'age': age,
            'subscription_type': subscription_type,
            'ad_interaction': ad_interaction,
            'ad_conversion_to_subscription': ad_conversion_to_subscription,
            'avg_listening_hours_per_week': avg_listening_hours_per_week,
            'playlists_created': playlists_created,
            'avg_skips_per_day': avg_skips_per_day,
            'music_suggestion_rating_1_to_5': music_suggestion_rating_1_to_5,
            'favorite_genre': favorite_genre,
            'most_liked_feature': most_liked_feature,
            'desired_future_feature': desired_future_feature,
            'primary_device': primary_device,
            'account_age_days': account_age_days
        }
        return pd.DataFrame([data])

    input_df = user_input_features()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📋 Configured User Profile")
        st.dataframe(input_df.T.rename(columns={0: 'Parameter Value'}), use_container_width=True, height=500)

    with col2:
        st.subheader("⚡ Churn Assessment & Risk Score")
        
        if st.button("Evaluate Churn Risk", type="primary", use_container_width=True):
            prediction_proba = model.predict_proba(input_df)[0][1]
            risk_percentage = prediction_proba * 100
            
            if prediction_proba > 0.5:
                st.error(f"⚠️ High Churn Risk Detected (Probability: {risk_percentage:.1f}%)")
                st.markdown("""
                * **Behavioral Insights:** This subscriber shows clear friction signals (high skip frequency or lower engagement).
                * **Recommended Action:** Trigger a targeted retention campaign, offer a curated playlist, or provide a discounted bundle upgrade.
                """)
            else:
                st.success(f"✅ Low Churn Risk - Active Subscriber (Probability: {risk_percentage:.1f}%)")
                st.markdown("""
                * **Behavioral Insights:** This user demonstrates healthy listening habits and solid platform engagement.
                * **Recommended Action:** Maintain standard engagement; consider cross-selling family plans or promoting social listening features.
                """)
            
            st.markdown("#### Risk Probability Gauge")
            st.progress(float(prediction_proba))


# ==========================================
# TAB 2: BATCH CSV SCORING
# ==========================================
with tab2:
    st.subheader("📂 Batch Subscriber Churn Evaluation")
    st.markdown("Upload a CSV file containing subscriber records to generate bulk churn risk predictions and download actionable retention lists.")
    
    uploaded_file = st.file_uploader("Upload Subscriber CSV Dataset", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(batch_df):,} records!")
        
        if st.checkbox("Preview Uploaded Data"):
            st.dataframe(batch_df.head(), use_container_width=True)
            
        if st.button("Run Batch Prediction Engine", type="primary"):
            processed_df = batch_df.copy()
            if 'signup_date' in processed_df.columns:
                processed_df['signup_date'] = pd.to_datetime(processed_df['signup_date'])
                processed_df['account_age_days'] = (datetime.date(2026, 10, 1) - processed_df['signup_date']).dt.days
            
            try:
                probs = model.predict_proba(processed_df)[:, 1]
                batch_df['Churn_Probability'] = np.round(probs * 100, 2)
                batch_df['Risk_Status'] = np.where(batch_df['Churn_Probability'] > 50, 'High Risk', 'Low Risk')
                
                st.markdown("### 📊 Batch Results Summary")
                high_risk_count = (batch_df['Risk_Status'] == 'High Risk').sum()
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Total Users Analyzed", f"{len(batch_df):,}")
                col_m2.metric("High-Risk Users Flagged", f"{high_risk_count:,}", delta_color="inverse")
                
                st.dataframe(batch_df.head(20), use_container_width=True)
                
                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Predictions CSV",
                    data=csv_data,
                    file_name="spotify_churn_predictions_output.csv",
                    mime="text/csv",
                    type="primary"
                )
            except Exception as e:
                st.error(f"Error during batch prediction: {e}. Ensure your CSV columns match the training schema.")


# ==========================================
# TAB 3: MODEL PERFORMANCE & INSIGHTS
# ==========================================
with tab3:
    st.subheader("📊 Model Architecture & Validation Metrics")
    st.markdown("This section details the machine learning methodology, model performance metrics, and feature importance drivers.")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Algorithm", "Optimized XGBoost", "Ensemble Gradient Boosting")
    col_p2.metric("Validation ROC-AUC", "~0.66 (Leak-Free)", "Robust Generalization")
    col_p3.metric("Training Dataset", "50,000 Records", "Stratified Split (80/20)")
    
    st.markdown("---")
    st.markdown("### 🔍 Key Feature Drivers of Churn")
    st.markdown("""
    Based on feature importance analysis from our trained XGBoost model, customer churn is primarily driven by:
    1. **Average Skips Per Day (`avg_skips_per_day`):** High skip rates signal content misalignment or poor recommendations.
    2. **Weekly Listening Volume (`avg_listening_hours_per_week`):** A sharp drop in active hours precedes cancellations.
    3. **Recommendation Rating (`music_suggestion_rating_1_to_5`):** Dissatisfaction with AI discovery tools directly correlates with churn.
    4. **Account Tenures & Subscription Tier:** Free tier users show higher baseline churn propensity compared to bundled annual subscribers.
    """)