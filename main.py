import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Air Quality Index Predictor", layout="wide")
st.title("🌫️ Air Quality Index Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your pollution dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    try:
        # Encode target
        label_encoder = LabelEncoder()
        df['Air Quality'] = label_encoder.fit_transform(df['Air Quality'])

        # Features and target
        X = df.drop('Air Quality', axis=1)
        y = df['Air Quality']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ Accuracy: {accuracy:.4f}")

        # Classification report
        st.subheader("📋 Classification Report")
        target_names = label_encoder.inverse_transform(np.unique(y_test))
        report = classification_report(y_test, y_pred, target_names=target_names.astype(str), output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        # ROC Curve
        st.subheader("📈 Multiclass ROC Curve")
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        ovr_clf = OneVsRestClassifier(model)
        ovr_clf.fit(X_train_scaled, label_binarize(y_train, classes=classes))
        y_score = ovr_clf.predict_proba(X_test_scaled)

        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"{label_encoder.inverse_transform([classes[i]])[0]} (AUC = {roc_auc:.2f})")

        ax_roc.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("👈 Upload a CSV file to begin.")
