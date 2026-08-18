import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ============================================================
# Page config + light custom styling ("calm" look)
# ============================================================
st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 16px; border-radius: 12px; }
    div[data-testid="stMetricValue"] { color: #7dd3fc; }
    h1, h2, h3 { font-weight: 600; }
    .stButton>button {
        background-color: #2563eb; color: white; border-radius: 10px;
        padding: 0.6em 1.4em; border: none; font-weight: 600;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load model and preprocessing artifacts
# ============================================================
xgb_model = joblib.load('xgb_model.pkl')
model_columns = joblib.load('model_columns.pkl')
categories = joblib.load('categories.pkl')
country_freq = joblib.load('country_freq.pkl')
month_map = joblib.load('month_map.pkl')

# Sensible defaults for the raw features not exposed in the simplified form
DEFAULTS = {
    'hotel': 'City Hotel',
    'adults': 2,
    'children': 0,
    'babies': 0,
    'meal': 'BB',
    'distribution_channel': 'TA/TO',
    'is_repeated_guest': 0,
    'previous_cancellations': 0,
    'previous_bookings_not_canceled': 0,
    'reserved_room_type': 'A',
    'assigned_room_type': 'A',
    'booking_changes': 0,
    'deposit_type': 'No Deposit',
    'customer_type': 'Transient',
    'required_car_parking_spaces': 0,
    'days_in_waiting_list': 0,
    'has_agent': 1,
    'has_company': 0,
}

st.title("🏨 Hotel Booking Cancellation Predictor")
st.caption("A quick, focused model that flags high-risk bookings using the 7 factors that matter most.")

tab_predict, tab_eda, tab_report = st.tabs(["🔮 Predict", "📊 EDA", "📋 Report"])

# ============================================================
# Preprocessing pipeline (mirrors the notebook)
# ============================================================
def preprocess(raw_df):
    df = raw_df.copy()

    df['arrival_month_num'] = df['arrival_date_month'].map(month_map)
    arrival_date = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' +
        df['arrival_month_num'].astype(str) + '-' +
        df['arrival_date_day_of_month'].astype(str),
        format='%Y-%m-%d'
    )
    df['arrival_day_of_week'] = arrival_date.dt.dayofweek
    df['is_weekend_arrival'] = df['arrival_day_of_week'].isin([5, 6]).astype(int)
    df['arrival_quarter'] = arrival_date.dt.quarter

    df['room_mismatch'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)
    df['country_encoded'] = df['country'].map(country_freq).fillna(1)

    df = df.drop(columns=['arrival_date_month', 'assigned_room_type', 'country'])

    low_card_cols = ['hotel', 'meal', 'market_segment', 'distribution_channel',
                      'deposit_type', 'customer_type', 'reserved_room_type']
    df = pd.get_dummies(df, columns=low_card_cols)

    df = df.reindex(columns=model_columns, fill_value=0)
    return df

# ============================================================
# TAB 1: Predict
# ============================================================
with tab_predict:
    st.subheader("Enter the key booking details")

    c1, c2 = st.columns(2)
    with c1:
        arrival_date = st.date_input("Arrival Date", value=pd.Timestamp("2017-07-15"))
        lead_time = st.number_input("Lead Time (days before arrival)", min_value=0, max_value=800, value=30)
        adr = st.number_input("ADR (Average Daily Rate)", min_value=0.0, max_value=1000.0, value=100.0)
        total_nights = st.number_input("Total Nights", min_value=0, max_value=60, value=3)

    with c2:
        country = st.text_input("Country Code (e.g. PRT, GBR, FRA)", value="PRT").upper()
        default_idx = categories['market_segment'].index('Online TA') if 'Online TA' in categories['market_segment'] else 0
        market_segment = st.selectbox("Market Segment", categories['market_segment'], index=default_idx)
        total_of_special_requests = st.slider("Total Special Requests", min_value=0, max_value=5, value=0)

    st.write("")
    predict_clicked = st.button("Predict Cancellation", type="primary")

    if predict_clicked:
        raw_input = pd.DataFrame([{
            **DEFAULTS,
            'lead_time': lead_time,
            'arrival_date_year': arrival_date.year,
            'arrival_date_month': arrival_date.strftime('%B'),
            'arrival_date_day_of_month': arrival_date.day,
            'country': country,
            'market_segment': market_segment,
            'adr': adr,
            'total_of_special_requests': total_of_special_requests,
            'total_nights': total_nights,
        }])

        X_input = preprocess(raw_input)
        prediction = xgb_model.predict(X_input)[0]
        probability = xgb_model.predict_proba(X_input)[0][1]

        st.divider()
        colA, colB = st.columns([2, 1])
        with colA:
            if prediction == 1:
                st.error(f"⚠️ This booking is likely to be **CANCELED** — probability: {probability:.1%}")
            else:
                st.success(f"✅ This booking is likely to be **HONORED** — cancellation probability: {probability:.1%}")
            st.progress(float(probability))
        with colB:
            st.metric("Cancellation Risk", f"{probability:.1%}")

# ============================================================
# TAB 2: EDA
# ============================================================
with tab_eda:
    st.subheader("Key patterns from the training data")

    target_dist = pd.read_csv('eda_target_dist.csv')
    cancel_by_deposit = pd.read_csv('eda_cancel_by_deposit.csv')
    cancel_by_market = pd.read_csv('eda_cancel_by_market.csv')
    lead_time_summary = pd.read_csv('eda_lead_time_by_status.csv')
    monthly = pd.read_csv('eda_monthly_cancel.csv')

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        fig = px.pie(
            target_dist, values='proportion', names=target_dist['is_canceled'].map({0: 'Not Canceled', 1: 'Canceled'}),
            title="Overall Cancellation Rate", color_discrete_sequence=['#22c55e', '#ef4444'], hole=0.5
        )
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with row1_col2:
        fig = px.bar(
            cancel_by_deposit, x='deposit_type', y='is_canceled',
            title="Cancellation Rate by Deposit Type", color='is_canceled', color_continuous_scale='Reds'
        )
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Cancellation Rate", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        fig = px.bar(
            cancel_by_market.sort_values('is_canceled'), x='is_canceled', y='market_segment', orientation='h',
            title="Cancellation Rate by Market Segment", color='is_canceled', color_continuous_scale='Blues'
        )
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', xaxis_title="Cancellation Rate", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with row2_col2:
        fig = px.bar(
            lead_time_summary, x='status', y='lead_time',
            title="Average Lead Time: Canceled vs Not", color='status',
            color_discrete_map={'Canceled': '#ef4444', 'Not Canceled': '#22c55e'}
        )
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Avg Lead Time (days)", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        monthly, x='arrival_date_month', y='mean',
        title="Cancellation Rate by Arrival Month", markers=True
    )
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Cancellation Rate", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3: Report
# ============================================================
with tab_report:
    st.subheader("Model Report")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "84.6%")
    m2.metric("Model", "XGBoost")
    m3.metric("Estimators", "200")
    m4.metric("Features Used", "52")

    st.markdown("""
    ### Summary
    - **Model**: XGBoost Classifier, trained on 52 engineered/encoded features derived from the original booking data.
    - **Target**: `is_canceled` — whether a booking ends up canceled (63% not canceled vs 37% canceled in the raw data).
    - **Top drivers of cancellation**: lead time, country, ADR, arrival date, number of special requests, total nights, and market segment — these are the 7 inputs exposed in the Predict tab; every other feature is filled with a sensible default so the form stays quick to use.
    - **Key insight**: non-refundable deposits show an unexpectedly *higher* cancellation rate, and longer lead times are consistently associated with more cancellations.

    ### How predictions are made
    1. The 7 inputs from the Predict tab are combined with default values for the remaining raw features.
    2. The same feature engineering used in training (arrival date breakdown, room mismatch, country frequency encoding, one-hot encoding) is applied automatically.
    3. The XGBoost model outputs a cancellation probability; a booking is flagged as "likely to cancel" above 50%.

    ### Limitations
    - Defaults used for non-exposed features may not reflect every real booking scenario — treat predictions as directional risk signals, not certainties.
    - The model was trained on historical data and may not capture new booking patterns or policy changes.
    """)

    st.download_button(
        "Download Report as Markdown",
        data="""# Hotel Booking Cancellation Model - Report

## Model
- XGBoost Classifier
- 200 estimators, 52 features
- Accuracy: 84.6%

## Target
is_canceled (0 = Not Canceled, 1 = Canceled)
Class balance: 63% Not Canceled / 37% Canceled

## Top Predictive Features
1. Lead Time
2. Country (frequency encoded)
3. ADR
4. Arrival Date (day/month/year)
5. Total Special Requests
6. Total Nights
7. Market Segment

## Key Insight
Non-refundable deposits show a much higher cancellation rate than expected,
and cancellations are consistently associated with longer lead times.
""",
        file_name="model_report.md",
        mime="text/markdown"
    )
