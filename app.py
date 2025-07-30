import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

st.title("📈 Sales Prediction and Visualization Tool")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.write(df.head())

    columns = df.columns.tolist()

    # Allow user to select date and sales columns if not standard
    st.markdown("### 🛠️ Select Columns for Analysis")
    date_col = st.selectbox("Select the Date column", columns)
    sales_col = st.selectbox("Select the Sales column", columns)

    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, sales_col])
        df.sort_values(date_col, inplace=True)

        st.subheader("📊 Sales Over Time")
        fig = px.line(df, x=date_col, y=sales_col, title="Sales Over Time")
        st.plotly_chart(fig)

        st.subheader("🤖 Train Sales Prediction Model")
        df["Day"] = (df[date_col] - df[date_col].min()).dt.days
        X = df[["Day"]]
        y = df[sales_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        st.write(f"Model MSE: {mse:.2f}")

        future_days = st.slider("Days to Predict", 1, 60, 30)
        last_day = df["Day"].max()
        future_X = pd.DataFrame({"Day": np.arange(last_day + 1, last_day + future_days + 1)})
        future_preds = model.predict(future_X)

        future_dates = df[date_col].max() + pd.to_timedelta(future_X["Day"] - last_day, unit="D")
        future_df = pd.DataFrame({"Date": future_dates, "Predicted Sales": future_preds})

        st.subheader("📈 Future Sales Predictions")
        st.write(future_df)

        fig_future = px.line(future_df, x="Date", y="Predicted Sales", title="Future Sales Prediction")
        st.plotly_chart(fig_future)

        csv = future_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name="future_sales.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
