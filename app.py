import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Sivun asetukset
st.set_page_config(page_title="Student Analysis", layout="wide")
st.title("🎓 Student Dropout Analysis & Prediction")

# 2. Datan lataus
@st.cache_data
def load_data():
    return pd.read_csv("student_dropout_dataset.csv")

try:
    df = load_data()

    # --- TEHTÄVÄ 2: Basic Info ---
    st.header("1. Dataset Overview")
    st.write("First 5 rows:", df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
    with col2:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    # --- TEHTÄVÄ 2: Visualisointi ---
    st.header("2. Distribution")
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    selected_viz = st.selectbox("Select column for histogram:", num_cols)
    fig1, ax1 = plt.subplots()
    sns.histplot(df[selected_viz], kde=True, ax=ax1)
    st.pyplot(fig1)

    # --- TEHTÄVÄ 3: Korrelaatio ---
    st.header("3. Correlation Analysis")
    # Yritetään etsiä oikeat sarakkeet vaikka nimet vaihtelisivat
    try:
        c1 = [c for c in df.columns if 'approved' in c.lower() and '1st' in c.lower()][0]
        c2 = [c for c in df.columns if 'grade' in c.lower() and '1st' in c.lower()][0]
        
        corr, _ = stats.pearsonr(df[c1], df[c2])
        st.write(f"Correlation between **{c1}** and **{c2}**: `{corr:.2f}`")
        
        fig2, ax2 = plt.subplots()
        sns.regplot(data=df, x=c1, y=c2, ax=ax2, line_kws={"color": "red"})
        st.pyplot(fig2)
        
        st.info("The positive correlation suggests that students who pass more credits also achieve higher grades.")
    except:
        st.warning("Could not perform correlation. Check column names.")

    # --- TEHTÄVÄ 4: Koneoppiminen ---
    st.header("4. Machine Learning: Prediction")
    try:
        # Etsitään kohdemuuttuja (yleensä 'Target')
        target_col = [c for c in df.columns if 'target' in c.lower() or 'status' in c.lower()][0]
        
        # Valitaan muutama ennustaja
        features = [c1, c2] # Käytetään samoja kuin korrelaatiossa
        X = df[features]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        st.success(f"Model Accuracy: **{acc:.2%}**")
        
        st.write("Interpretation: The model uses early academic performance to predict if a student will graduate or drop out. High accuracy confirms that first-semester results are strong indicators of future success.")
    except:
        st.warning("Could not run the model. Check if 'Target' column exists.")

except Exception as e:
    st.error(f"Error loading application: {e}")
    st.info("Make sure 'student_dropout_dataset.csv' is in your GitHub root folder.")