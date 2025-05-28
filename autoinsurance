import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import altair as alt

# Title
st.set_page_config(page_title="Insurance Claims Clustering", layout="wide")
st.title("Interactive Insurance Claims Clustering Dashboard")

# Load data with caching
def load_data(path):
    @st.cache_data
    def _load():
        df = pd.read_csv(path)
        return df
    return _load()

data = load_data('/mnt/data/insurance_claims.csv')

# Feature selection
numeric_features = [
    'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
    'policy_annual_premium', 'months_as_customer', 'policy_deductable', 'age',
    'incident_hour_of_the_day', 'number_of_vehicles_involved', 'auto_year'
]
cat_features = [
    'incident_severity', 'fraud_reported', 'collision_type'
]
all_features = numeric_features + cat_features

# Sidebar controls
st.sidebar.header("Clustering Controls")
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

axis_x = st.sidebar.selectbox("X-axis Feature", numeric_features, index=0)
axis_y = st.sidebar.selectbox("Y-axis Feature", numeric_features, index=1)

# Data preprocessing
df = data[all_features].dropna().copy()

# Encode categorical features
le = LabelEncoder()
for col in cat_features:
    df[col] = le.fit_transform(df[col])

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(df[all_features])

# Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X)

df['cluster'] = labels.astype(str)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=all_features)
centers_df['cluster'] = centers_df.index.astype(str)

# Main layout
col1, col2 = st.columns((2, 1))

with col1:
    st.subheader("Cluster Scatter Plot")
    chart = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X(axis_x, type='quantitative'),
        y=alt.Y(axis_y, type='quantitative'),
        color='cluster:N',
        tooltip=[axis_x, axis_y, 'cluster']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Fraud Rate by Cluster")
    fraud_df = df.groupby('cluster')['fraud_reported'].mean().reset_index()
    bar = alt.Chart(fraud_df).mark_bar().encode(
        x='cluster:N',
        y='fraud_reported:Q',
        tooltip=['cluster', alt.Tooltip('fraud_reported:Q', format='.2f')]
    )
    st.altair_chart(bar, use_container_width=True)

with col2:
    st.subheader("Cluster Centers")
    st.dataframe(centers_df.set_index('cluster').round(2))

st.markdown("---")
st.subheader("Additional Insights")

# Incident Severity Distribution
dist_df = df.groupby(['cluster', 'incident_severity']).size().reset_index(name='count')
severity_chart = alt.Chart(dist_df).mark_line(point=True).encode(
    x='incident_severity:N',
    y='count:Q',
    color='cluster:N',
    tooltip=['cluster', 'incident_severity', 'count']
).interactive()
st.altair_chart(severity_chart, use_container_width=True)

# Vehicle Age vs Claim Amount
st.subheader("Vehicle Age vs. Vehicle Claim by Cluster")
veh_chart = alt.Chart(df).mark_circle().encode(
    x='auto_year:Q',
    y='vehicle_claim:Q',
    color='cluster:N',
    tooltip=['auto_year', 'vehicle_claim', 'cluster']
).interactive()
st.altair_chart(veh_chart, use_container_width=True)

st.markdown("---")
st.write("**Tip:** Use the controls in the sidebar to adjust the number of clusters and explore different feature combinations.")
