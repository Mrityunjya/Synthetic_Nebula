import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from io import BytesIO
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Synthetic Data Detection", layout="wide")
st.title("🧪 Synthetic Data Detection -  Dashboard")

# -----------------------------
# Light/Dark Mode Toggle
# -----------------------------
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)
bg_color = "#111111" if dark_mode else "#f5f5f5"
text_color = "#f5f5f5" if dark_mode else "#111111"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .sidebar .sidebar-content {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color:#4CAF50;color:white;border-radius:5px
    }}
    </style>
    """, unsafe_allow_html=True
)

# -----------------------------
# Dataset Option
# -----------------------------
st.sidebar.title("🚀 Dataset Options")
data_option = st.sidebar.radio("Choose Dataset:", ("Pretrained Data", "Upload CSV"))

if data_option == "Pretrained Data":
    df = pd.read_csv("creditcard.csv")
elif data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# -----------------------------
# Tabs for Layout
# -----------------------------
tab_overview, tab_analysis, tab_predict, tab_download = st.tabs(
    ["Overview", "Feature Analysis", "Prediction", "Reports & Downloads"]
)

# -----------------------------
# Quick Metrics Function
# -----------------------------
def get_metrics(dataframe):
    total = len(dataframe)
    real = len(dataframe[dataframe['Class'] == 0])
    fraud = len(dataframe[dataframe['Class'] == 1])
    fraud_pct = round((fraud/total)*100, 2) if total>0 else 0
    return total, real, fraud, fraud_pct

total_records, real_records, fraud_records, fraud_pct = get_metrics(df)

# -----------------------------
# Tab 1: Overview
# -----------------------------
with tab_overview:
    st.subheader("📊 Quick Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div style='background:#4facfe; padding:15px; border-radius:12px; color:white; text-align:center; font-weight:bold;'>Total Records<br>{total_records}</div>", unsafe_allow_html=True)
    col2.markdown(f"<div style='background:#43e97b; padding:15px; border-radius:12px; color:white; text-align:center; font-weight:bold;'>Real Records<br>{real_records}</div>", unsafe_allow_html=True)
    col3.markdown(f"<div style='background:#f857a6; padding:15px; border-radius:12px; color:white; text-align:center; font-weight:bold;'>Fraud/Synthetic<br>{fraud_records}</div>", unsafe_allow_html=True)
    col4.markdown(f"<div style='background:#f9d423; padding:15px; border-radius:12px; color:white; text-align:center; font-weight:bold;'>Fraud %<br>{fraud_pct}%</div>", unsafe_allow_html=True)

    st.subheader("📄 Dataset Snapshot")
    st.dataframe(df.head(10))

# -----------------------------
# Tab 2: Feature Analysis & Charts
# -----------------------------
with tab_analysis:
    st.subheader("🔑 Feature Importance & Class Distribution")

    # Sample 5000 records with fraud included
    df_real = df[df['Class'] == 0]
    df_fraud = df[df['Class'] == 1]
    n_sample = min(5000, len(df))
    n_fraud = min(len(df_fraud), int(n_sample * 0.05))
    n_real = n_sample - n_fraud
    df_sample = pd.concat([
        df_real.sample(n=n_real, random_state=42),
        df_fraud.sample(n=n_fraud, random_state=42)
    ]).sample(frac=1, random_state=42)

    X_demo = df_sample.drop('Class', axis=1)
    y_demo = df_sample['Class']

    # Train model on sampled data
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_demo, y_demo)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_scaled, y_res)

    # Feature Importance
    importance_df = pd.DataFrame({'Feature': X_demo.columns, 'Importance': model_rf.feature_importances_})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    fig_fi = px.bar(importance_df.head(15), x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Viridis', title="Top 15 Features")
    st.plotly_chart(fig_fi, use_container_width=True)

    # Class Distribution
    fig_class = px.histogram(df_sample, x='Class', color='Class',
                             color_discrete_map={0:'#2ecc71',1:'#e74c3c'},
                             labels={'Class':'Record Type'},
                             title="Real vs Fraud/Synthetic Records")
    st.plotly_chart(fig_class, use_container_width=True)

    # Confusion Matrix
    cm = confusion_matrix(y_res, model_rf.predict(X_scaled))
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                       labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix")
    fig_cm.update_xaxes(tickvals=[0,1], ticktext=['Real','Fraud'])
    fig_cm.update_yaxes(tickvals=[0,1], ticktext=['Real','Fraud'])
    st.plotly_chart(fig_cm, use_container_width=True)

# -----------------------------
# Tab 3: Fraud Prediction
# -----------------------------
with tab_predict:
    st.subheader("💡 Predict Fraud for Your Record")
    user_input = {}
    for feature in X_demo.columns:
        user_input[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))
    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model_rf.predict(input_scaled)[0]
        prob = model_rf.predict_proba(input_scaled)[0][1]
        # Fraud risk gauge
        st.markdown(f"<div style='background:#e74c3c; border-radius:50px; width:100%; text-align:center; padding:10px; font-weight:bold; color:white;'>Fraud Risk: {round(prob*100,2)}%</div>", unsafe_allow_html=True)
        st.metric("Prediction", "Fraud/Synthetic" if prediction==1 else "Real")

# -----------------------------
# Tab 4: Reports & Downloads
# -----------------------------
with tab_download:
    st.subheader("📄 Download Dataset Snapshot")
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    st.download_button(label="Download CSV", data=buffer, file_name="creditcard_snapshot.csv", mime="text/csv")

    st.subheader("📄 Generate PDF Report with Charts")
    def create_pdf_report(filename="synthetic_report.pdf"):
        c = canvas.Canvas(filename, pagesize=(800, 1000))
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, 950, "🧪 Synthetic Data Detection Report")
        c.setFont("Helvetica", 14)
        c.drawString(50, 920, f"Total Records: {len(df)}")
        c.drawString(50, 900, f"Real Records: {len(df[df['Class']==0])}")
        c.drawString(50, 880, f"Fraud/Synthetic Records: {len(df[df['Class']==1])}")

        y_pos = 850
        # Feature Importance chart
        fig_fi.write_image("temp_feature.png")
        c.drawImage("temp_feature.png", 50, y_pos-250, width=700, height=250)
        y_pos -= 280

        # Class Distribution chart
        fig_class.write_image("temp_class.png")
        c.drawImage("temp_class.png", 50, y_pos-250, width=700, height=250)
        y_pos -= 280

        # Confusion Matrix
        fig_cm_mat, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        fig_cm_mat.savefig("temp_cm.png")
        c.drawImage("temp_cm.png", 50, y_pos-200, width=400, height=200)

        c.save()

    if st.button("Generate PDF Report"):
        pdf_file = "synthetic_report.pdf"
        create_pdf_report(pdf_file)
        with open(pdf_file, "rb") as f:
            st.download_button(label="Download PDF", data=f, file_name="synthetic_report.pdf", mime="application/pdf")
