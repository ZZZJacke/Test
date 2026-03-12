import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sivun otsikko
st.title("🎓 Student Dropout Analysis")
st.markdown("Minimal Exploratory Data Analysis (EDA) on the Student Dropout Dataset.")

# 1. Load the dataset
# Oletetaan, että dataset on samassa kansiossa GitHubissa
@st.cache_data
def load_data():
    df = pd.read_csv("student_dropout_dataset.csv")
    return df

try:
    df = load_data()

    # 2. Show basic info
    st.header("1. Dataset Overview")
    
    st.subheader("First 5 rows")
    st.write(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Shape")
        st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")
    
    with col2:
        st.subheader("Missing values")
        st.write(df.isnull().sum().sum())

    # 3. Compute descriptive statistics
    st.header("2. Descriptive Statistics")
    st.write(df.describe())

    # 4. Create one simple visualization
    st.header("3. Visualization")
    
    # Valitaan numeerinen sarake histogrammia varten (esim. ikä tai arvosanat)
    # Muokkaa 'Age at enrollment' vastaamaan datasetin sarakkeen nimeä jos tarpeen
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    selected_col = st.selectbox("Select a column to visualize:", numeric_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_col], kde=True, ax=ax, color='skyblue')
    ax.set_title(f"Distribution of {selected_col}")
    
    st.pyplot(fig)

except FileNotFoundError:
    st.error("Datasetiä ei löytynyt. Varmista, että 'student_dropout_dataset.csv' on GitHub-reposi juuressa.")