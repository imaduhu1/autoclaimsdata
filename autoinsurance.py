import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans


#1. Page Configuration & Header

st.set_page_config(page_title="Insurance Claims Risk Explorer", layout="wide")
st.markdown(
    '''
    <style>
    .letter {
      display: inline-block;
      color: #00008B;
      font-size: 48px;
      font-weight: 700;
      opacity: 0;
      animation: drop 1s ease-out forwards;
    }
    @keyframes drop {
      0%   { transform: translateY(-200px); opacity: 0; }
      80%  { transform: translateY(20px); opacity: 1; }
      100% { transform: translateY(0); opacity: 1; }
    }
    .final-title {
      text-align: center;
      color: #00008B;
      font-size: 48px;
      font-weight: 700;
      opacity: 0;
      animation: fadeIn 1s ease-in forwards;
      animation-delay: 3.7s;
      margin-top: -48px;
    }
    @keyframes fadeIn {
      from { opacity: 0; }
      to   { opacity: 1; }
    }
    .container { text-align: center; overflow: hidden; white-space: nowrap; }
    </style>
    <div class="container">
      <!-- Falling letters sequence -->
      <span class="letter" style="animation-delay:0s;">I</span>
      <span class="letter" style="animation-delay:0.1s;">n</span>
      <span class="letter" style="animation-delay:0.2s;">s</span>
      <span class="letter" style="animation-delay:0.3s;">u</span>
      <span class="letter" style="animation-delay:0.4s;">r</span>
      <span class="letter" style="animation-delay:0.5s;">a</span>
      <span class="letter" style="animation-delay:0.6s;">n</span>
      <span class="letter" style="animation-delay:0.7s;">c</span>
      <span class="letter" style="animation-delay:0.8s;">e</span>
      <span style="display:inline-block; width:16px;"></span>
      <span class="letter" style="animation-delay:0.9s;">C</span>
      <span class="letter" style="animation-delay:1.0s;">l</span>
      <span class="letter" style="animation-delay:1.1s;">a</span>
      <span class="letter" style="animation-delay:1.2s;">i</span>
      <span class="letter" style="animation-delay:1.3s;">m</span>
      <span class="letter" style="animation-delay:1.4s;">s</span>
      <span style="display:inline-block; width:16px;"></span>
      <span class="letter" style="animation-delay:1.5s;">R</span>
      <span class="letter" style="animation-delay:1.6s;">i</span>
      <span class="letter" style="animation-delay:1.7s;">s</span>
      <span class="letter" style="animation-delay:1.8s;">k</span>
      <span style="display:inline-block; width:16px;"></span>
      <span class="letter" style="animation-delay:1.9s;">E</span>
      <span class="letter" style="animation-delay:2.0s;">x</span>
      <span class="letter" style="animation-delay:2.1s;">p</span>
      <span class="letter" style="animation-delay:2.2s;">l</span>
      <span class="letter" style="animation-delay:2.3s;">o</span>
      <span class="letter" style="animation-delay:2.4s;">r</span>
      <span class="letter" style="animation-delay:2.5s;">e</span>
      <span class="letter" style="animation-delay:2.6s;">r</span>
    </div>
    
    '''
    unsafe_allow_html=True
)

#2. Data Loading & Preprocessing

@st.cache_data
# Cache CSV loading for performance
def load_data(path):
    return pd.read_csv(path, parse_dates=['policy_bind_date'])

# Attempt to load default file, otherwise allow upload
DEFAULT_FILE = 'insurance_claims.csv'
if os.path.exists(DEFAULT_FILE):
    df = load_data(DEFAULT_FILE)
else:
    uploaded = st.sidebar.file_uploader('Upload insurance_claims.csv', type='csv')
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=['policy_bind_date'])
    else:
        st.stop()

# Create AgeGroup bins for segmentation
bins = [17, 29, 39, 49, 59, 69, 100]
labels = ['18-29','30-39','40-49','50-59','60-69','70+']
df['AgeGroup'] = pd.cut(df['age'], bins=bins, labels=labels)


#3. Clustering & Risk Labeling

features = ['total_claim_amount', 'policy_annual_premium', 'months_as_customer', 'age']
X = df[features].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster_id'] = kmeans.fit_predict(X_scaled)

# Map clusters to risk levels based on sorted avg claim
centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features
)
avg_claim = centers['total_claim_amount']
sorted_idx = avg_claim.argsort()
risk_labels = ['Low Risk', 'Medium Risk', 'High Risk', 'Extreme Risk']
cluster_to_label = {int(idx): label for idx, label in zip(sorted_idx, risk_labels)}
df['RiskLevel'] = df['cluster_id'].map(cluster_to_label)

# Encode fraud for quantitative charts
df['FraudFlag'] = df['fraud_reported'].map({'Y': 1, 'N': 0})


# 4. Sidebar Filters

st.sidebar.header('Filters & Options')
selected_risks = st.sidebar.multiselect('Select Risk Levels', risk_labels, default=risk_labels)
selected_states = st.sidebar.multiselect('Filter by Policy State', sorted(df['policy_state'].unique()), default=[])
date_range = st.sidebar.date_input('Policy Bind Date Range', [df['policy_bind_date'].min(), df['policy_bind_date'].max()])

# Apply filters
df_filtered = df[df['RiskLevel'].isin(selected_risks)]
if selected_states:
    df_filtered = df_filtered[df_filtered['policy_state'].isin(selected_states)]
start_date, end_date = date_range
mask = (df_filtered['policy_bind_date'] >= pd.to_datetime(start_date)) & (df_filtered['policy_bind_date'] <= pd.to_datetime(end_date))
df_filtered = df_filtered[mask]


# 5. KPI Cards

st.subheader('Key Metrics')
avg_claim = df_filtered['total_claim_amount'].mean()
median_claim = df_filtered['total_claim_amount'].median()
fraud_rate = df_filtered['FraudFlag'].mean()
count = df_filtered.shape[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric('Avg Claim Amount', f"${avg_claim:,.0f}")
c2.metric('Median Claim', f"${median_claim:,.0f}")
c3.metric('Fraud Rate', f"{fraud_rate:.1%}")
c4.metric('Records', f"{count:,}")

st.markdown('---')


# 6. Visualizations

# 6a. Distribution of Risk Levels
risk_dist = df_filtered['RiskLevel'].value_counts().reset_index()
risk_dist.columns = ['RiskLevel', 'Count']
st.altair_chart(
    alt.Chart(risk_dist).mark_bar().encode(
        x='RiskLevel:N', y='Count:Q', color='RiskLevel:N', tooltip=['RiskLevel','Count']
    ), use_container_width=True
)

# 6b. Time Series: Avg Claim by Month & Risk
line_frames = []
for risk in selected_risks:
    monthly = (
        df_filtered[df_filtered['RiskLevel'] == risk]
            .set_index('policy_bind_date')['total_claim_amount']
            .resample('M').mean()
            .reset_index()
            .assign(RiskLevel=risk)
    )
    line_frames.append(monthly)
line_df = pd.concat(line_frames)
st.altair_chart(
    alt.Chart(line_df).mark_line(point=True).encode(
        x='policy_bind_date:T', y='total_claim_amount:Q', color='RiskLevel:N',
        tooltip=['policy_bind_date','total_claim_amount','RiskLevel']
    ).properties(width=800), use_container_width=True
)

# 6c. Scatter: Premium vs Claim
st.altair_chart(
    alt.Chart(df_filtered).mark_circle(size=50, opacity=0.6).encode(
        x='policy_annual_premium:Q', y='total_claim_amount:Q',
        color='RiskLevel:N', tooltip=['policy_annual_premium','total_claim_amount','RiskLevel']
    ).interactive(), use_container_width=True
)

# 6d. Tenure Histogram by Risk
st.altair_chart(
    alt.Chart(df_filtered).mark_bar(opacity=0.7).encode(
        x=alt.X('months_as_customer:Q', bin=alt.Bin(maxbins=30)),
        y='count()', color='RiskLevel:N'
    ), use_container_width=True
)

# 6e. Fraud Rate by Risk
fraud_summary = df_filtered.groupby('RiskLevel')['FraudFlag'].mean().reset_index()
st.altair_chart(
    alt.Chart(fraud_summary).mark_bar().encode(
        x='RiskLevel:N', y=alt.Y('FraudFlag:Q', title='Avg Fraud Rate'),
        tooltip=['RiskLevel', alt.Tooltip('FraudFlag:Q', format='.2%')]
    ), use_container_width=True
)

st.markdown('---')
st.write('Use the sidebar to refine filters or upload new data.')


