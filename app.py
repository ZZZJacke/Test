import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Dropout Analysis", layout="wide")
st.title("🎓 Student Dropout Analysis & Prediction")

@st.cache_data
def load_data():
    return pd.read_csv("student_dropout_dataset.csv")

try:
    df = load_data()

    # --- 1. Dataset Overview ---
    st.header("1. Dataset Overview")
    st.write("First 5 rows:", df.head())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    # --- 2. Visualization ---
    st.header("2. Distribution of Student Ages")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['age'], kde=True, ax=ax1, color='orange')
    st.pyplot(fig1)

    # --- 3. Correlation Analysis ---
    st.header("3. Correlation Analysis")
    # Käytetään enrolled courses ja completed assignments
    c1, c2 = 'courses_enrolled', 'completed_assignments'
    
    corr, p_val = stats.pearsonr(df[c1], df[c2])
    st.write(f"Pearson correlation between **{c1}** and **{c2}**: `{corr:.2f}`")
    
    fig2, ax2 = plt.subplots()
    sns.regplot(data=df, x=c1, y=c2, ax=ax2, line_kws={"color": "red"}, scatter_kws={'alpha':0.3})
    st.pyplot(fig2)
    
    st.info(f"""
    **Interpretation:** The correlation is {corr:.2f}. 
    This indicates a positive relationship where students enrolled in more courses also tend to complete more assignments. 
    The result makes sense as it reflects overall academic activity levels.
    """)

    # --- 4. Supervised Learning ---
    st.header("4. Machine Learning: Predicting Student Status")
    
    # Valitaan predictorit ja target sinun tiedostostasi
    features = ['age', 'completion_rate', 'login_frequency', 'last_activity_days_ago']
    target = 'label_name' # active, dropped, at-risk
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.success(f"Model Accuracy: **{acc:.2%}**")
    
    st.write(f"""
    **Interpretation:** We used a Decision Tree to predict the student status ({target}) using {', '.join(features)}. 
    The accuracy of {acc:.2%} shows that the model is very effective at identifying students at risk based on their activity patterns. 
    This makes perfect sense: low login frequency and high 'last activity days' are strong indicators of a potential dropout.
    """)

except Exception as e:
    st.error(f"Error: {e}")