import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

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

    st.header("4. Correlation Analysis")

    # Valitaan kaksi numeerista muuttujaa (voit muuttaa nämä vastaamaan datasetin sarakkeita)
    var1 = 'Curricular units 1st sem (approved)'
    var2 = 'Curricular units 1st sem (grade)'

        if var1 in df.columns and var2 in df.columns:
        # Laske Pearsonin korrelaatio
        correlation, p_value = stats.pearsonr(df[var1], df[var2])
        
        st.subheader(f"Correlation between {var1} and {var2}")
        st.write(f"**Pearson correlation coefficient:** {correlation:.2f}")

        # Luo scatter plot ja regressioviiva
        fig2, ax2 = plt.subplots()
        sns.regplot(x=df[var1], y=df[var2], ax=ax2, 
                    scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        ax2.set_title(f"Scatter plot: {var1} vs {var2}")
        st.pyplot(fig2)

        # Tulkinta (3–5 lausetta)
        st.subheader("Interpretation")
        if correlation > 0.5:
            strength = "strong"
        direction = "positive"
    elif correlation > 0.3:
        strength = "moderate"
        direction = "positive"
    else:
        strength = "weak"
        direction = "positive"

    st.write(f"""
    The analysis shows a **{direction}** and **{strength}** correlation between approved units and grades. 
    This means that as the number of approved units increases, the average grade also tends to rise. 
    The result makes a lot of sense, as successful students usually perform well both in terms of credit accumulation and grade point average. 
    The scatter plot and the red regression line clearly visualize this upward trend.
    """)
else:
    st.warning("Check your column names for correlation analysis!")
    
    st.pyplot(fig)

except FileNotFoundError:
    st.error("Datasetiä ei löytynyt. Varmista, että 'student_dropout_dataset.csv' on GitHub-reposi juuressa.")