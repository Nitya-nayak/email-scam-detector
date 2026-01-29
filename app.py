import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack, csr_matrix
from test import handle_msg, feature_extraction, show_reasons, FEATURE_ORDER
import joblib

# load models

model = joblib.load("model.joblib")
tfidf = joblib.load("tf.joblib")


label_map = {0: "Not Spam", 1: "Spam"}

st.set_page_config(page_title="Email Scam Detector", layout="centered")

st.title("📧 Email / Message Scam Detection")
st.write("Enter a message to check whether it is **Spam or Not**")

user_input = st.text_area("Paste your email or message here", height=180)

if st.button("Analyze Message"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # reasons
        reasons = show_reasons(user_input)

        # ML pipeline
        clean_txt = handle_msg(user_input)
        tfidf_vec = tfidf.transform([clean_txt])

        feat = feature_extraction(user_input)
        feat_array = np.array([feat[f] for f in FEATURE_ORDER]).reshape(1, -1)
        feat_array = csr_matrix(feat_array)

        final_features = hstack([feat_array, tfidf_vec])
        pred = model.predict(final_features)[0]

        st.subheader("🔍 Prediction Result")
        if pred == 1:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")

        st.subheader("🧠 Reasons")
        if reasons:
            for r in reasons:
                st.write("•", r)
        else:
            st.write("• No strong scam indicators found")

        st.markdown("---")
        st.caption("Spam Detection System | Logistic Regression + TF-IDF")
