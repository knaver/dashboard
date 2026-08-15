import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Page layout configuration
st.set_page_config(page_title="Titanic EDA", layout="wide")

st.title("🚢 Titanic Dataset Exploratory Data Analysis")

# Load data with caching for speed
@st.cache_data
def load_data():
    return sns.load_dataset("titanic")

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Data")

# Pclass Filter
pclasses = sorted(df['pclass'].dropna().unique())
selected_pclass = st.sidebar.multiselect("Passenger Class (Pclass)", pclasses, default=pclasses)

# Sex Filter
sexes = df['sex'].dropna().unique()
selected_sex = st.sidebar.multiselect("Sex", sexes, default=list(sexes))

# Embarked Filter
ports = df['embark_town'].dropna().unique()
selected_ports = st.sidebar.multiselect("Embark Town", ports, default=list(ports))

# Apply Filters
filtered_df = df[
    (df['pclass'].isin(selected_pclass)) &
    (df['sex'].isin(selected_sex)) &
    (df['embark_town'].isin(selected_ports))
].copy()

# Add Age Category column for Tab 4 breakdown
bins = [0, 12, 18, 35, 60, 100]
labels = ['Child (0-12)', 'Teen (13-18)', 'Young Adult (19-35)', 'Adult (36-60)', 'Senior (60+)']
filtered_df['age_group'] = pd.cut(filtered_df['age'], bins=bins, labels=labels)

# --- MAIN DASHBOARD ---

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Passengers", len(filtered_df))
col2.metric("Survivors", int(filtered_df['survived'].sum()))
col3.metric("Overall Survival Rate", f"{(filtered_df['survived'].mean() * 100):.1f}%")
col4.metric("Average Fare", f"${filtered_df['fare'].mean():.2f}")

st.divider()

# Tab Navigation for EDA Sections
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview", 
    "📈 Visual Distributions", 
    "🔗 Feature Correlations", 
    "🎯 Detailed Survival Analysis"
])

with tab1:
    st.subheader("Data Preview")
    st.dataframe(filtered_df, width="stretch")
    
    st.subheader("Summary Statistics")
    st.write(filtered_df.describe())

with tab2:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Survival Rate by Class & Gender")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=filtered_df, x="pclass", y="survived", hue="sex", errorbar=None, palette="Set2", ax=ax)
        ax.set_ylabel("Survival Rate")
        st.pyplot(fig)
        
    with col_b:
        st.subheader("Age Distribution by Survival")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(data=filtered_df, x="age", hue="survived", common_norm=False, palette="Set1", fill=True, ax=ax)
        ax.set_xlabel("Age")
        st.pyplot(fig)

with tab3:
    st.subheader("Numeric Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# --- REVISED TAB 4 (WITH ANNOTATIONS) ---
with tab4:
    st.subheader("Survival Rates Across Demographics")
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        st.markdown("**1. Survival Rate by Sex**")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        sns.barplot(data=filtered_df, x="sex", y="survived", errorbar=None, palette="viridis", ax=ax)
        ax.set_ylabel("Survival Rate")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        st.info("💡 **Insight:** Females achieved a significantly higher survival rate (~74%) compared to males (~19%), reflecting the strict 'women and children first' evacuation protocol.")

        st.markdown("**2. Survival Rate by Embarkation Town**")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        sns.barplot(data=filtered_df, x="embark_town", y="survived", hue="sex", errorbar=None, palette="Blues_d", ax=ax)
        ax.set_ylabel("Survival Rate")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        st.info("💡 **Insight:** Cherbourg passengers had higher survival rates largely due to socioeconomic class distribution—a higher proportion of 1st-class passengers boarded at Cherbourg.")

    with col_y:
        st.markdown("**3. Survival Rate by Age Group**")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        sns.barplot(data=filtered_df, x="age_group", y="survived", errorbar=None, palette="magma", ax=ax)
        ax.set_ylabel("Survival Rate")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        st.info("💡 **Insight:** Children (0–12) had the highest priority and survival outcome (~58%), whereas seniors (60+) faced the lowest odds (~22%).")

        st.markdown("**4. Survival Rate by Age Group and Class**")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        sns.barplot(data=filtered_df, x="age_group", y="survived", hue="pclass", errorbar=None, palette="crest", ax=ax)
        ax.set_ylabel("Survival Rate")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        st.info("💡 **Insight:** Socioeconomic status modified age prioritization: 1st-class passengers across almost all age groups maintained high survival rates compared to 3rd-class peers.")