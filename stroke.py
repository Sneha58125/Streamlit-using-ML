import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.figure_factory as ff

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.drop_duplicates()

for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

if "bmi" in df.columns:
    df = df[df["bmi"] > 0]
df_clean = df.copy()

st.set_page_config(page_title="Healthcare Stroke Dashboard + ML", page_icon="🏥", layout="wide")
st.title("Healthcare Stroke Dataset Dashboard with ML 🏥🤖")
st.markdown("This dashboard shows cleaned healthcare data with charts and a Random Forest ML model.")

st.sidebar.title("Filter your view")
gender_options = df_clean['gender'].unique()
selected_gender = st.sidebar.multiselect("Select Gender(s):", options=gender_options)

work_options = df_clean['work_type'].unique()
selected_work = st.sidebar.multiselect("Select Work Type(s):", options=work_options)

smoke_options = df_clean['smoking_status'].unique()
selected_smoke = st.sidebar.multiselect("Select Smoking Status:", options=smoke_options)

if selected_gender and selected_work and selected_smoke:
    filtered_df = df_clean[
        (df_clean['gender'].isin(selected_gender)) &
        (df_clean['work_type'].isin(selected_work)) &
        (df_clean['smoking_status'].isin(selected_smoke))
    ]
    st.success("Data filtered successfully")

    st.markdown("Choose your desired chart type:")

    with st.expander("Bar chart: Average Age vs Stroke"):
        data = filtered_df.groupby("stroke")['age'].mean().reset_index()
        fig1 = px.bar(
            data, x="stroke", y="age", color="stroke",
            labels={"stroke": "Stroke (0=No, 1=Yes)"},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig1, use_container_width=True)

    with st.expander("Pie chart: Hypertension Distribution"):
        data = filtered_df['hypertension'].value_counts().reset_index()
        data.columns = ['Hypertension', 'Count']
        fig2 = px.pie(
            data, names='Hypertension', values='Count',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Sunburst chart: Stroke by Work Type and Gender"):
        fig3 = px.sunburst(
            filtered_df, path=['work_type', 'gender'], values='stroke',
            color='stroke', color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Clustering chart: Age vs Glucose (KMeans)"):
        from sklearn.cluster import KMeans

        cluster_data = filtered_df[['age', 'avg_glucose_level']].copy()
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_data['Cluster'] = kmeans.fit_predict(cluster_data)

        fig4 = px.scatter(
            cluster_data, x="age", y="avg_glucose_level",
            color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="KMeans Clustering (Age vs Glucose)"
        )
        st.plotly_chart(fig4, use_container_width=True)


    st.markdown("Machine Learning: Predict Stroke with Random Forest")

    if st.button("Train Random Forest Model"):
        ml_df = df_clean.drop(['id'], axis=1)
        encoder = LabelEncoder()
        for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
            ml_df[col] = encoder.fit_transform(ml_df[col])

        X = ml_df.drop('stroke', axis=1)
        y = ml_df['stroke']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: **{acc*100:.2f}%**")

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        z = cm[::-1]
        x = ["No Stroke", "Stroke"]
        y_labels = ["Stroke", "No Stroke"]
        fig_cm = ff.create_annotated_heatmap(
            z, x=x, y=y_labels, colorscale="Blues", showscale=True
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    else:
        st.warning("⚠️ Please select filters from the sidebar to display charts and ML model!")