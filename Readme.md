# 🛡️ Enterprise Email Fraud & Phishing Detection System

A high-performance Machine Learning solution designed to classify inbound emails as **Legitimate (Class 0)** or **Fraudulent/Phishing (Class 1)** using an ensemble framework. This project processes raw text and routing metadata to achieve a robust **97.00% predictive accuracy score**.

🚀 **[Click here to view the Live Streamlit App Dashboard]** *(Replace with your deployment link if hosted online)*

---

## 📊 Project Architecture & Workflow
The data pipeline maps unstructured communications into a high-dimensional mathematical space using a dual-path engineering strategy:

1. **Metadata Engineering**: Extracts numerical structural cues including URL volumetric frequencies, sender-receiver domain verification loops (`is_internal_domain`), distribution list sizes (`receiver_count`), and text length parameters.
2. **Natural Language Processing (NLP)**: Implements a `TfidfVectorizer` to tokenize email body payloads into a sparse matrix consisting of the top 500 vocabulary indicators, filtering out English stop-word noise.
3. **Ensemble Classification**: Combines features via `scipy.sparse.hstack` to train a **Random Forest Classifier** configured with 100 estimators and automated class-balancing penalties.

---

## ⚙️ Repository File Structure
```text
├── app.py                      # Production-grade interactive Streamlit UI
├── email_fraud_notebook.ipynb  # Jupyter Notebook containing exploratory data analysis & training
├── rf_email_fraud_model.pkl    # Serialized Random Forest Classifier binary
├── tfidf_vectorizer.pkl        # Serialized NLP vocabulary matrix mapping
├── requirements.txt            # Operational system dependency package listings
└── README.md                   # System documentation and structural deployment guide
```

---

## 📈 System Metrics & Performance Evaluators
* **Core Framework**: Random Forest Classifier
* **Overall Evaluation Accuracy**: `97.00%`
* **Handling Class Imbalance**: Solved using `class_weight='balanced'` criteria within split logic trees to handle low-volume fraud arrays.
* **Overfitting Safeguards**: Configured with explicit tree pruning parameters (`max_depth=12`) to improve general stability against real-world inputs.

---

## 🚀 Local Deployment Instructions

Follow these steps to spin up the system and run the interactive UI dashboard on your local machine:

### 1. Clone the Workspace Repository
```bash
git clone https://github.com
cd YOUR_REPO_NAME
```

### 2. Configure Environment Dependencies
Ensure you have Python installed, then build your library requirements using the project's dependency script:
```bash
pip install -r requirements.txt
```
*(If you haven't created a requirements.txt file yet, run: `pip install streamlit pandas joblib scikit-learn scipy`)*

### 3. Launch the Interactive UI Application Dashboard
Execute the server script to automatically fire up the framework inside a secure web browser sandbox tab:
```bash
streamlit run app.py
```

---

## 🎓 Academic Presentation Summary
Developed as an end-to-end Capstone Portfolio Framework demonstrating concepts across **Data Cleansing (NaN Imputation)**, **Linguistic Feature Vectorization (TF-IDF)**, **Ensemble Tree Pruning**, and **Interactive Full-Stack ML Deployment**.
