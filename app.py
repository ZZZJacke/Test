import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("🎓 Student Dropout Analysis & Prediction")

# 1. Load the dataset
@st.cache_data
def load_data():
    # CSV file
    df = pd.read_csv("student_dropout_dataset.csv")
    return df

try:
    df = load_data()

    # First header
    st.header("1. Dataset Overview")
    st.subheader("First 5 rows")
    st.write(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Shape")
        st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
    with col2:
        st.subheader("Missing values")
        st.write(f"Total missing: **{df.isnull().sum().sum()}**")

    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    # Second
    st.header("2. Distribution Visualization")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    viz_col = st.selectbox("Select column for histogram:", numeric_cols)
    fig1, ax1 = plt.subplots()
    sns.histplot(df[viz_col], kde=True, ax=ax1, color='skyblue')
    st.pyplot(fig1)

    # Third: Correlation Analysis
    st.header("3. Correlation Analysis")
    var1 = 'Curricular units 1st sem (approved)'
    var2 = 'Curricular units 1st sem (grade)'

    if var1 in df.columns and var2 in df.columns:
        correlation, _ = stats.pearsonr(df[var1], df[var2])
        st.write(f"**Pearson correlation between {var1} and {var2}:** {correlation:.2f}")

        fig2, ax2 = plt.subplots()
        sns.regplot(x=df[var1], y=df[var2], ax=ax2, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        st.pyplot(fig2)

        st.info(f"""
        **Interpretation:** The correlation is {correlation:.2f}, which indicates a strong positive relationship. 
        This means students who pass more units also tend to get higher grades. 
        The result is logical as academic engagement usually reflects in both quantity and quality of credits.
        """)

    # Fourth:
    st.header("4. Supervised Learning: Predicting Dropout")

    if 'Target' in df.columns:
        predictors = ['Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Age at enrollment']
        
        # Cleaning
        available_predictors = [p for p in predictors if p in df.columns]
        
        X = df[available_predictors]
        y = df['Target']

        # 1. Train-test split (70/30)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 2. Train model (Decision Tree)
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X_train, y_train)

        # 3. Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.subheader("Model Results")
        st.write(f"Used predictors: *{', '.join(available_predictors)}*")
        st.write(f"**Model Accuracy:** {acc:.2%}")

        # 4. Interpretation
        st.success(f"""
        **Interpretation:** The Decision Tree model achieved an accuracy of {acc:.2%}. 
        This suggests that the model can predict the student's status relatively well based on their first-semester performance and age. 
        The result makes sense because early academic success is often the strongest indicator of whether a student will graduate or drop out.
        """)
    else:
        st.warning("Target column not found for machine learning.")

# Error message
except FileNotFoundError:
    st.error("Dataset 'student_dropout_dataset.csv' not found. Please check your GitHub repository.")