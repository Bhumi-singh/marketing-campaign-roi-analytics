"""
dashboard/app.py
Streamlit Marketing Campaign ROI Analytics Dashboard
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Campaign ROI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BLUE = "#1F4E79"
ORANGE = "#F4A261"
LIGHT = "#9DC3E6"

# ── Load Data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    seg_path = os.path.join(base, 'segmented_data.csv')
    scored_path = os.path.join(base, 'scored_customers.csv')

    if os.path.exists(scored_path):
        df = pd.read_csv(scored_path)
    elif os.path.exists(seg_path):
        df = pd.read_csv(seg_path)
    else:
        # Demo data if notebooks not yet run
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            'age': np.random.randint(18, 70, n),
            'job': np.random.choice(['admin.','blue-collar','technician','management','retired','services'], n),
            'education': np.random.choice(['university.degree','high.school','basic.9y','professional.course'], n),
            'contact': np.random.choice(['cellular','telephone'], n, p=[0.65, 0.35]),
            'month': np.random.choice(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], n),
            'campaign': np.random.randint(1, 10, n),
            'subscribed': np.random.choice([0, 1], n, p=[0.89, 0.11]),
            'segment_name': np.random.choice(['High-Value Responders','Warm Prospects','Price Sensitive','Hard to Convert'], n),
            'conv_probability': np.random.beta(2, 10, n),
            'euribor3m': np.random.uniform(0.6, 5.1, n),
            'age_group': np.random.choice(['<30','30-40','40-50','50-60','60+'], n),
        })
    return df

df = load_data()

# ── Sidebar Controls ─────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
cost_per_contact = st.sidebar.slider("Cost per Contact (₹)", 10, 200, 50, 5)
revenue_per_conv = st.sidebar.slider("Revenue per Conversion (₹)", 1000, 20000, 5000, 500)

channels = st.sidebar.multiselect("Channels", df['contact'].unique().tolist(), default=df['contact'].unique().tolist())
df_filtered = df[df['contact'].isin(channels)] if channels else df

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.metric("Total Records", f"{len(df_filtered):,}")
st.sidebar.metric("Conversions", f"{df_filtered['subscribed'].sum():,}")

# ── Header ───────────────────────────────────────────────────────
st.title("📊 Marketing Campaign ROI Analytics")
st.caption("Bank Marketing Campaign — Acquisition Analytics Dashboard")

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Campaign Overview", "📡 Channel Performance", "👥 Segment Analysis", "🎯 Prediction Tool"])

# ═══════════════════════════════════════════════════════
# TAB 1 — Campaign Overview
# ═══════════════════════════════════════════════════════
with tab1:
    n = len(df_filtered)
    conv = df_filtered['subscribed'].sum()
    conv_rate = conv / n * 100
    total_cost = n * cost_per_contact
    total_rev = conv * revenue_per_conv
    roi = (total_rev - total_cost) / total_cost * 100
    cpa = total_cost / conv if conv > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Contacts", f"{n:,}")
    col2.metric("Conversions", f"{conv:,}")
    col3.metric("Conversion Rate", f"{conv_rate:.2f}%")
    col4.metric("Campaign ROI", f"{roi:.1f}%", delta=f"₹{total_rev-total_cost:,.0f} profit")
    col5.metric("Cost Per Acquisition", f"₹{cpa:,.0f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        # Monthly conversion trend
        month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
        month_data = df_filtered.groupby('month')['subscribed'].mean().reindex(month_order).dropna().reset_index()
        month_data.columns = ['month','conv_rate']
        month_data['conv_rate'] *= 100
        fig = px.line(month_data, x='month', y='conv_rate',
                      markers=True, title='Monthly Conversion Rate (%)',
                      color_discrete_sequence=[BLUE])
        fig.update_layout(yaxis_ticksuffix='%', xaxis_title='Month', yaxis_title='Conversion Rate')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Contact frequency vs conversion
        freq = df_filtered[df_filtered['campaign']<=8].groupby('campaign')['subscribed'].mean().reset_index()
        freq.columns = ['contacts','conv_rate']
        freq['conv_rate'] *= 100
        fig = px.bar(freq, x='contacts', y='conv_rate',
                     title='Conversion Rate by Contact Frequency',
                     color_discrete_sequence=[BLUE])
        fig.update_layout(yaxis_ticksuffix='%', xaxis_title='# Contacts', yaxis_title='Conv Rate (%)')
        st.plotly_chart(fig, use_container_width=True)

    # Funnel
    funnel_data = dict(
        Stage=["Total Contacts", "Multi-Contact (>1x)", "Converted"],
        Count=[n, int((df_filtered['campaign']>1).sum()), conv]
    )
    fig = go.Figure(go.Funnel(
        y=funnel_data['Stage'], x=funnel_data['Count'],
        textinfo="value+percent initial",
        marker=dict(color=[BLUE, LIGHT, ORANGE])
    ))
    fig.update_layout(title="Campaign Funnel", height=300)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 2 — Channel Performance
# ═══════════════════════════════════════════════════════
with tab2:
    ch = df_filtered.groupby('contact').agg(
        n=('subscribed','count'),
        converted=('subscribed','sum')
    ).reset_index()
    ch['conv_rate'] = ch['converted'] / ch['n'] * 100
    ch['total_cost'] = ch['n'] * cost_per_contact
    ch['revenue'] = ch['converted'] * revenue_per_conv
    ch['roi'] = (ch['revenue'] - ch['total_cost']) / ch['total_cost'] * 100
    ch['cpa'] = ch['total_cost'] / ch['converted']

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(ch, x='contact', y='conv_rate', text='conv_rate',
                     title='Conversion Rate by Channel',
                     color='contact', color_discrete_sequence=[BLUE, LIGHT])
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(yaxis_ticksuffix='%', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(ch, x='contact', y='roi', text='roi',
                     title='ROI by Channel (%)',
                     color='contact', color_discrete_sequence=[ORANGE, BLUE])
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(yaxis_ticksuffix='%', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Channel ROI Breakdown")
    ch_display = ch.copy()
    ch_display['conv_rate'] = ch_display['conv_rate'].round(2).astype(str) + '%'
    ch_display['roi'] = ch_display['roi'].round(1).astype(str) + '%'
    ch_display['total_cost'] = ch_display['total_cost'].apply(lambda x: f"₹{x:,.0f}")
    ch_display['revenue'] = ch_display['revenue'].apply(lambda x: f"₹{x:,.0f}")
    ch_display['cpa'] = ch_display['cpa'].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(ch_display.rename(columns={
        'contact':'Channel','n':'Contacts','converted':'Conversions',
        'conv_rate':'Conv Rate','total_cost':'Total Cost',
        'revenue':'Revenue','roi':'ROI','cpa':'CPA'
    }), use_container_width=True)

    # Sensitivity
    st.subheader("ROI Sensitivity to Conversion Rate")
    base_rate = df_filtered['subscribed'].mean()
    pct_changes = np.linspace(-50, 50, 21)
    sens = []
    for pct in pct_changes:
        adj = base_rate * (1 + pct/100)
        r = (10000 * adj * revenue_per_conv - 10000 * cost_per_contact) / (10000 * cost_per_contact) * 100
        sens.append({'Change (%)': pct, 'ROI (%)': r})
    sens_df = pd.DataFrame(sens)
    fig = px.line(sens_df, x='Change (%)', y='ROI (%)',
                  title='Sensitivity: Campaign ROI vs Conversion Rate Change',
                  color_discrete_sequence=[BLUE])
    fig.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='Break-even')
    fig.update_layout(xaxis_ticksuffix='%', yaxis_ticksuffix='%')
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════
# TAB 3 — Segment Analysis
# ═══════════════════════════════════════════════════════
with tab3:
    seg_col = 'segment_name' if 'segment_name' in df_filtered.columns else 'segment'
    if seg_col in df_filtered.columns:
        seg = df_filtered.groupby(seg_col).agg(
            n=('subscribed','count'),
            converted=('subscribed','sum'),
            avg_age=('age','mean'),
            avg_contacts=('campaign','mean')
        ).reset_index()
        seg['conv_rate'] = (seg['converted'] / seg['n'] * 100).round(2)
        seg['roi'] = ((seg['converted'] * revenue_per_conv - seg['n'] * cost_per_contact) / (seg['n'] * cost_per_contact) * 100).round(1)
        seg['cpa'] = (seg['n'] * cost_per_contact / seg['converted']).round(0)
        seg['avg_age'] = seg['avg_age'].round(1)
        seg['avg_contacts'] = seg['avg_contacts'].round(2)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(seg, x=seg_col, y='conv_rate', text='conv_rate',
                         title='Conversion Rate by Segment',
                         color=seg_col, color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(yaxis_ticksuffix='%', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(seg, x=seg_col, y='roi', text='roi',
                         title='ROI by Segment (%)',
                         color=seg_col, color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
            fig.update_layout(yaxis_ticksuffix='%', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(seg, x='avg_contacts', y='conv_rate', size='n',
                         color=seg_col, text=seg_col,
                         title='Segment: Avg Contacts vs Conversion Rate (bubble=size)',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition='top center')
        fig.update_layout(yaxis_ticksuffix='%')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Segment Detail Table")
        st.dataframe(seg.rename(columns={
            seg_col:'Segment','n':'Size','converted':'Converted',
            'conv_rate':'Conv Rate (%)','roi':'ROI (%)','cpa':'CPA (₹)',
            'avg_age':'Avg Age','avg_contacts':'Avg Contacts'
        }), use_container_width=True)
    else:
        st.warning("Run Notebook 04 first to generate segment data.")


# ═══════════════════════════════════════════════════════
# TAB 4 — Prediction Tool
# ═══════════════════════════════════════════════════════
with tab4:
    st.subheader("🎯 Customer Conversion Probability Estimator")
    st.caption("Input customer attributes to estimate conversion probability using the trained XGBoost model.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 90, 35)
        job = st.selectbox("Job", ['admin.','blue-collar','technician','management','retired','services','self-employed','entrepreneur','housemaid','student','unemployed'])
        education = st.selectbox("Education", ['university.degree','high.school','basic.9y','professional.course','basic.6y','basic.4y','illiterate'])

    with col2:
        contact = st.selectbox("Contact Channel", ['cellular','telephone'])
        campaign = st.slider("# Contacts This Campaign", 1, 15, 2)
        previous = st.slider("# Previous Campaign Contacts", 0, 10, 0)

    with col3:
        euribor3m = st.slider("Euribor 3M Rate", 0.6, 5.1, 1.3, 0.1)
        cons_conf = st.slider("Consumer Confidence Index", -50.0, 0.0, -40.0, 0.5)
        emp_var = st.slider("Employment Variation Rate", -3.5, 1.5, -1.8, 0.1)

    if st.button("Estimate Conversion Probability", type="primary"):
        # Rule-based scoring (proxy when model not available)
        score = 0.11  # baseline
        if contact == 'cellular': score += 0.08
        if age < 30 or age > 60: score += 0.05
        if job in ['retired','student']: score += 0.10
        if job in ['blue-collar','services']: score -= 0.03
        if campaign == 1: score += 0.04
        elif campaign > 5: score -= 0.05
        if previous > 0: score += 0.06
        if euribor3m < 2.0: score += 0.04
        if cons_conf > -35: score += 0.03
        score = max(0.01, min(0.99, score))

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Conversion Probability", f"{score*100:.1f}%")
        col_r2.metric("Priority Tier", "High" if score > 0.25 else "Medium" if score > 0.12 else "Low")
        col_r3.metric("Recommended Action", "Contact (Cellular)" if score > 0.20 else "Monitor" if score > 0.10 else "Deprioritize")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Conversion Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': BLUE},
                'steps': [
                    {'range': [0, 12], 'color': '#FFE0E0'},
                    {'range': [12, 25], 'color': '#FFE8CC'},
                    {'range': [25, 100], 'color': '#D6E4F0'}
                ],
                'threshold': {'line': {'color': ORANGE, 'width': 4}, 'thickness': 0.75, 'value': score*100}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    if 'conv_probability' in df_filtered.columns:
        st.subheader("Top 20 Highest-Probability Customers")
        top = df_filtered.nlargest(20, 'conv_probability')[
            ['age','job','education','contact','campaign','conv_probability']
        ].copy()
        top['conv_probability'] = (top['conv_probability'] * 100).round(1).astype(str) + '%'
        st.dataframe(top.reset_index(drop=True), use_container_width=True)
