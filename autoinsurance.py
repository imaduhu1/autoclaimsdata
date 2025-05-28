import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# ————————————————————————————————
# 1. Page Configuration & Header
# ————————————————————————————————
st.set_page_config(page_title="Insurance Clustering Explorer", layout="wide")
st.markdown(
    '''
    <style>
    .title {
        font-size: 48px;
        font-weight: 700;
        text-align: center;
        color: #2E86AB !important;
        animation: slideFade 5s ease-in-out infinite;
        margin: 20px 0;
    }
    @keyframes slideFade {
        0% { transform: translateX(-40%); opacity: 0; }
        30% { transform: translateX(0%); opacity: 1; }
        70% { transform: translateX(0%); opacity: 1; }
        100% { transform: translateX(40%); opacity: 0; }
    }
    </style>
    <h1 class="title">Insurance Claims Risk Explorer</h1>
    ''',
    unsafe_allow_html=True
)

# ————————————————————————————————
# 2. Data Loading & Preprocessing
# ————————————————————————————————

@st.cache_data
 def load_data(path):
    df = pd.read_csv(path, parse_dates=['policy_bind_date'])
    return df

# Try default file or upload
DEFAULT_FILE = 'insurance_claims.csv'
if os.path.exists(DEFAULT_FILE):
    df = load_data(DEFAULT_FILE)
else:
    uploaded = st.sidebar.file_uploader('Upload insurance_claims.csv', type='csv')
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=['policy_bind_date'])
    else:
        st.stop()

# Create Age Group
bins = [17, 29, 39, 49, 59, 69, 100]
labels = ['18-29','30-39','40-49','50-59','60-69','70+']
df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels)

# ————————————————————————————————
# 3. Clustering & Risk Labeling
# ————————————————————————————————

features = ['total_claim_amount', 'policy_annual_premium', 'months_as_customer', 'age']
X = df[features].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster_id'] = kmeans.fit_predict(X_scaled)

# Determine risk by average claim amount per cluster
group_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features
)
avg_claim = group_centers['total_claim_amount']
# Sort clusters by avg_claim
sorted_idx = avg_claim.argsort()

risk_labels = ['Low Risk', 'Medium Risk', 'High Risk', 'Extreme Risk']
cluster_to_label = {int(sorted_idx[i]): risk_labels[i] for i in range(len(risk_labels))}
df['RiskLevel'] = df['cluster_id'].map(cluster_to_label)

# Encode for plotting
df['FraudFlag'] = df['fraud_reported'].map({'Y':1, 'N':0})

# ————————————————————————————————
# 4. Sidebar Filters
# ————————————————————————————————
st.sidebar.header('Filters & Options')
selected_risks = st.sidebar.multiselect(
    'Select Risk Levels', risk_labels, default=risk_labels
)
selected_states = st.sidebar.multiselect(
    'Filter by Policy State', sorted(df['policy_state'].unique()), default=[]
)
# Time window filter
date_range = st.sidebar.date_input(
    'Policy Bind Date Range',
    [df['policy_bind_date'].min(), df['policy_bind_date'].max()]
)

# Apply filters
df_filtered = df[df['RiskLevel'].isin(selected_risks)]
if selected_states:
    df_filtered = df_filtered[df_filtered['policy_state'].isin(selected_states)]
start_date, end_date = date_range
mask = (df_filtered['policy_bind_date'] >= pd.to_datetime(start_date)) & \
        (df_filtered['policy_bind_date'] <= pd.to_datetime(end_date))
df_filtered = df_filtered[mask]

# ————————————————————————————————
# 5. KPI Cards
# ————————————————————————————————

st.subheader('Key Metrics')
avg_claim = df_filtered['total_claim_amount'].mean()
median_claim = df_filtered['total_claim_amount'].median()
fraud_rate = df_filtered['FraudFlag'].mean()
count = len(df_filtered)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Avg Claim Amount', f"${avg_claim:,.0f}")
c2.metric('Median Claim', f"${median_claim:,.0f}")
c3.metric('Fraud Rate', f"{fraud_rate:.1%}")
c4.metric('Records', f"{count:,}")

st.markdown('---')

# ————————————————————————————————
# 6. Visualizations
# ————————————————————————————————

# 6a. Bar: Distribution of Risk Levels
dist = df_filtered['RiskLevel'].value_counts().reset_index()
dist.columns = ['RiskLevel','Count']
bar1 = alt.Chart(dist).mark_bar().encode(
    x='RiskLevel:N', y='Count:Q', color='RiskLevel:N',
    tooltip=['RiskLevel','Count']
)
st.altair_chart(bar1, use_container_width=True)

# 6b. Time Series: Avg Claim by Month & Risk

ts = (
    df_filtered
    .set_index('policy_bind_date')
    .resample('M')['total_claim_amount']
    .mean()
    .reset_index()
)
# Merge Risk: compute each risk group separately
lines = []
for risk in selected_risks:
    tmp = (
        df_filtered[df_filtered['RiskLevel']==risk]
        .set_index('policy_bind_date')
        .resample('M')['total_claim_amount']
        .mean()
        .reset_index()
        .assign(RiskLevel=risk)
    )
    lines.append(tmp)
line_df = pd.concat(lines)

line_chart = alt.Chart(line_df).mark_line(point=True).encode(
    x='policy_bind_date:T', y='total_claim_amount:Q', color='RiskLevel:N',
    tooltip=['policy_bind_date','total_claim_amount','RiskLevel']
).properties(width=800)
st.altair_chart(line_chart, use_container_width=True)

# 6c. Scatter: Premium vs Claim colored by Risk
scatter = alt.Chart(df_filtered).mark_circle(size=50, opacity=0.6).encode(
    x='policy_annual_premium:Q', y='total_claim_amount:Q',
    color='RiskLevel:N', tooltip=['policy_annual_premium','total_claim_amount','RiskLevel']
).interactive()
st.altair_chart(scatter, use_container_width=True)

# 6d. Histogram: Tenure by Risk
hist = alt.Chart(df_filtered).mark_bar(opacity=0.7).encode(
    x=alt.X('months_as_customer:Q', bin=alt.Bin(maxbins=30)),
    y='count()', color='RiskLevel:N'
)
st.altair_chart(hist, use_container_width=True)

# 6e. Bar: Fraud Rate by Risk
df_fr = df_filtered.groupby('RiskLevel')['FraudFlag'].mean().reset_index()
b2 = alt.Chart(df_fr).mark_bar().encode(
    x='RiskLevel:N', y='FraudFlag:Q',
    tooltip=['RiskLevel', alt.Tooltip('FraudFlag:Q', format='.2%')]
)
st.altair_chart(b2, use_container_width=True)

st.markdown('---')
st.write('Use the sidebar to refine filters or upload new data.')

