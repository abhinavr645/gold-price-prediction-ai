import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Gold Price Prediction Dashboard",
    layout="wide"
)

# -----------------------------------
# TITLE
# -----------------------------------
st.title("Gold Price Prediction Dashboard")
st.markdown("### AI-Powered Gold Market Analysis")

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("XAU_1h_data.csv", sep=";")
    return df

df = load_data()

# -----------------------------------
# DATE CONVERSION
# -----------------------------------
df["Date"] = pd.to_datetime(df["Date"])

# -----------------------------------
# SMALLER DATASET FOR FAST DEPLOYMENT
# -----------------------------------
df_small = df.sample(10000, random_state=42)

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.header("Dashboard Controls")

numeric_cols = ["Open", "High", "Low", "Close", "Volume"]

feature = st.sidebar.selectbox(
    "Select Feature Column",
    numeric_cols,
    index=0
)

target = st.sidebar.selectbox(
    "Select Target Column",
    numeric_cols,
    index=3
)

# -----------------------------------
# DATA PREVIEW
# -----------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------------
# MARKET METRICS
# -----------------------------------
st.subheader("Market Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Current Gold Price",
    f"${df['Close'].iloc[-1]:.2f}"
)

col2.metric(
    "Highest Price",
    f"${df['High'].max():.2f}"
)

col3.metric(
    "Lowest Price",
    f"${df['Low'].min():.2f}"
)

col4.metric(
    "Average Volume",
    f"{df['Volume'].mean():.0f}"
)

# -----------------------------------
# MACHINE LEARNING DATA
# -----------------------------------
X = df_small[[feature]]
y = df_small[target]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# MODEL TRAINING
# -----------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------------
# PREDICTIONS
# -----------------------------------
predictions = model.predict(X_test)

# -----------------------------------
# MODEL PERFORMANCE
# -----------------------------------
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.subheader("Model Performance")

m1, m2 = st.columns(2)

m1.metric(
    "Mean Absolute Error",
    f"{mae:.2f}"
)

m2.metric(
    "R² Score",
    f"{r2:.4f}"
)

# -----------------------------------
# ACTUAL VS PREDICTED GRAPH
# -----------------------------------
st.subheader("Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    y_test.values[:50],
    label="Actual",
    linewidth=3
)

ax.plot(
    predictions[:50],
    label="Predicted",
    linewidth=3
)

ax.set_title("Gold Price Prediction")
ax.set_xlabel("Samples")
ax.set_ylabel("Price")

ax.legend()
ax.grid(True)

st.pyplot(fig)

# -----------------------------------
# HISTORICAL TREND
# -----------------------------------
st.subheader("Historical Gold Price Trend")

fig2, ax2 = plt.subplots(figsize=(14, 5))

ax2.plot(df["Close"][:500])

ax2.set_title("Gold Price Trend")
ax2.set_xlabel("Time")
ax2.set_ylabel("Close Price")

st.pyplot(fig2)

# -----------------------------------
# MOVING AVERAGE
# -----------------------------------
st.subheader("Moving Average Analysis")

df["MA50"] = df["Close"].rolling(50).mean()

fig3, ax3 = plt.subplots(figsize=(14, 5))

ax3.plot(
    df["Close"][:500],
    label="Close Price"
)

ax3.plot(
    df["MA50"][:500],
    label="50 Moving Average"
)

ax3.legend()
ax3.set_title("Moving Average Analysis")

st.pyplot(fig3)

# -----------------------------------
# CUSTOM PREDICTION
# -----------------------------------
st.subheader("Predict Future Gold Price")

user_input = st.number_input(
    f"Enter {feature} value",
    value=float(df[feature].mean())
)

if st.button("Predict Price"):

    prediction = model.predict([[user_input]])

    st.success(
        f"Predicted {target} Price: ${prediction[0]:.2f}"
    )

# -----------------------------------
# FUTURE FORECAST
# -----------------------------------
st.subheader("Future Gold Price Forecast")

future_inputs = np.array(
    df_small[feature].tail(10)
).reshape(-1, 1)

future_predictions = model.predict(future_inputs)

forecast_df = pd.DataFrame({
    "Input Value": future_inputs.flatten(),
    "Predicted Price": future_predictions
})

st.write("Next Predicted Gold Prices")

st.dataframe(forecast_df)

# -----------------------------------
# FORECAST GRAPH
# -----------------------------------
fig_forecast, ax_forecast = plt.subplots(figsize=(12, 5))

ax_forecast.plot(
    future_predictions,
    marker="o",
    linewidth=3
)

ax_forecast.set_title("Future Gold Price Forecast")
ax_forecast.set_xlabel("Future Steps")
ax_forecast.set_ylabel("Predicted Price")

ax_forecast.grid(True)

st.pyplot(fig_forecast)

# -----------------------------------
# DATASET STATISTICS
# -----------------------------------
st.subheader("Dataset Statistics")

st.dataframe(df.describe())

# -----------------------------------
# CORRELATION MATRIX
# -----------------------------------
st.subheader("Correlation Matrix")

corr = df[numeric_cols].corr()

fig4, ax4 = plt.subplots(figsize=(8, 6))

cax = ax4.matshow(corr)

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

fig4.colorbar(cax)

st.pyplot(fig4)

# -----------------------------------
# CANDLESTICK CHART
# -----------------------------------
st.subheader("Live Candlestick Chart")

candlestick_df = df.head(300)

fig5 = go.Figure(data=[go.Candlestick(
    x=candlestick_df["Date"],
    open=candlestick_df["Open"],
    high=candlestick_df["High"],
    low=candlestick_df["Low"],
    close=candlestick_df["Close"]
)])

fig5.update_layout(
    title="Gold Price Candlestick Chart",
    xaxis_title="Date",
    yaxis_title="Gold Price",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

# -----------------------------------
# FOOTER
# -----------------------------------
st.success("Gold Price Prediction Dashboard Running Successfully")
