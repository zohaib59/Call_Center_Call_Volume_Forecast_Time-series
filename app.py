!pip install -q prophet holidays xgboost statsmodels scikit-learn tensorflow openpyxl
# IMPORT LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from prophet import Prophet
import holidays
import xgboost as xgb

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error
)

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from google.colab import files
import io

# STEP 1: UPLOAD FILE
# =========================================================
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# =========================================================
# AUTO READ FILE
# =========================================================
if filename.endswith(".csv"):
    data_raw = pd.read_csv(
        io.BytesIO(uploaded[filename])
    )

elif filename.endswith(".xlsx"):
    data_raw = pd.read_excel(
        io.BytesIO(uploaded[filename])
    )

else:
    raise Exception("❌ Unsupported file format")

print("\n✅ FILE LOADED SUCCESSFULLY")
print("Shape:", data_raw.shape)

# STEP 2: AUTO DETECT DATE COLUMN
# =========================================================
def detect_date_column(df):

    possible_dates = []

    for col in df.columns:

        try:
            temp = pd.to_datetime(
                df[col],
                dayfirst=True,
                errors='coerce'
            )

            valid_ratio = temp.notna().mean()

            if valid_ratio > 0.6:
                possible_dates.append(col)

        except:
            pass

    if len(possible_dates) == 0:
        raise Exception(
            "❌ No valid date column detected"
        )

    return possible_dates[0]

date_col = detect_date_column(data_raw)

print(f"\n✅ Detected Date Column: {date_col}")

# STEP 3: AUTO DETECT TARGET COLUMN
# =========================================================
def detect_target_column(df, date_col):

    numeric_cols = []

    for col in df.columns:

        if col == date_col:
            continue

        try:
            temp = pd.to_numeric(
                df[col],
                errors='coerce'
            )

            valid_ratio = temp.notna().mean()

            if valid_ratio > 0.7:
                numeric_cols.append(col)

        except:
            pass

    if len(numeric_cols) == 0:
        raise Exception(
            "❌ No numeric target column found"
        )

    # Choose highest variance column
    variances = {}

    for col in numeric_cols:

        variances[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        ).var()

    target_col = max(
        variances,
        key=variances.get
    )

    return target_col

target_col = detect_target_column(
    data_raw,
    date_col
)

print(f"✅ Detected Target Column: {target_col}")

# STEP 4: CLEAN DATA
# =========================================================
data = data_raw[[date_col, target_col]].copy()

data.columns = ['Date', 'Target']

# Convert
data['Date'] = pd.to_datetime(
    data['Date'],
    dayfirst=True,
    errors='coerce'
)

data['Target'] = pd.to_numeric(
    data['Target'],
    errors='coerce'
)

# Remove invalid
data.dropna(inplace=True)

# Remove duplicates
data = data.drop_duplicates(
    subset=['Date']
)

# Sort
data = data.sort_values('Date')

print("\n✅ CLEANED DATA")
print(data.head())

# STEP 5: OUTLIER CAPPING
# =========================================================
q1, q99 = data['Target'].quantile(
    [0.01, 0.99]
)

data['Target'] = data['Target'].clip(
    lower=q1,
    upper=q99
)
# STEP 6: AUTO DETECT FREQUENCY
# =========================================================
freq = pd.infer_freq(data['Date'])

if freq is None:
    freq = 'D'

print(f"\n✅ Detected Frequency: {freq}")

# STEP 7: HOLIDAY CALENDAR
# =========================================================
years = pd.DatetimeIndex(
    data["Date"]
).year.unique()

ind_holidays = holidays.India(
    years=years
)

holiday_df = pd.DataFrame({
    "ds": pd.to_datetime(
        list(ind_holidays.keys())
    ),
    "holiday": "india_national"
})

# STEP 8: FEATURE ENGINEERING
# =========================================================
def create_features(
    df,
    log_transform=False
):

    df = df.copy()

    # Rename for Prophet compatibility
    if 'Date' in df.columns:
        df.rename(columns={
            'Date': 'ds',
            'Target': 'y'
        }, inplace=True)

    # Log Transform
    if log_transform:

        # Avoid log(0)
        df['y'] = np.log1p(
            df['y']
        )

    # Date Features
    df['day'] = df['ds'].dt.day
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter

    # Lag Features
    df['lag_1'] = df['y'].shift(1)
    df['lag_7'] = df['y'].shift(7)

    # Rolling Features
    df['rolling_mean_7'] = (
        df['y']
        .rolling(7)
        .mean()
    )

    df['rolling_std_7'] = (
        df['y']
        .rolling(7)
        .std()
    )

    # Returns
    df['pct_change'] = (
        df['y']
        .pct_change()
    )

    # Clean
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.dropna(inplace=True)

    return df

# CREATE DATASETS
# =========================================================
df_raw = create_features(data)

df_log = create_features(
    data,
    log_transform=True
)

# =========================================================
# FEATURES
# =========================================================
FEATURES = [
    'day',
    'month',
    'year',
    'dayofweek',
    'quarter',
    'lag_1',
    'lag_7',
    'rolling_mean_7',
    'rolling_std_7',
    'pct_change'
]

# STEP 9: PROPHET
# =========================================================
def train_prophet(
    df,
    holiday_df,
    log=False
):

    model = Prophet(
        holidays=holiday_df
    )

    for reg in FEATURES:
        model.add_regressor(reg)

    model.fit(df)

    future = model.make_future_dataframe(
        periods=30,
        freq=freq
    )

    feature_df = df[
        ['ds'] + FEATURES
    ]

    future = future.merge(
        feature_df,
        on='ds',
        how='left'
    )

    future.fillna(
        method='ffill',
        inplace=True
    )

    forecast = model.predict(future)

    if log:
        forecast['yhat'] = np.expm1(
            forecast['yhat']
        )

    eval_df = forecast.merge(
        df[['ds', 'y']],
        on='ds'
    )

    mape = (
        mean_absolute_percentage_error(
            eval_df['y'],
            eval_df['yhat']
        ) * 100
    )

    rmse = np.sqrt(
        mean_squared_error(
            eval_df['y'],
            eval_df['yhat']
        )
    )

    return mape, rmse

# STEP 10: XGBOOST
# =========================================================
def train_xgb(df):

    X = df[FEATURES]
    y = df['y']

    split = int(
        len(df) * 0.8
    )

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    mape = (
        mean_absolute_percentage_error(
            y_test,
            y_pred
        ) * 100
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    return mape, rmse, model

# STEP 11: ARIMA
# =========================================================
def train_arima(df):

    series = df['y']

    split = int(
        len(series) * 0.8
    )

    train = series[:split]
    test = series[split:]

    model = ARIMA(
        train,
        order=(5,1,0)
    ).fit()

    forecast = model.forecast(
        len(test)
    )

    mape = (
        mean_absolute_percentage_error(
            test,
            forecast
        ) * 100
    )

    rmse = np.sqrt(
        mean_squared_error(
            test,
            forecast
        )
    )

    return mape, rmse

# STEP 12: LSTM
# =========================================================
def train_lstm(
    df,
    look_back=30
):

    series = (
        df['y']
        .values
        .reshape(-1,1)
    )

    split = int(
        len(series) * 0.8
    )

    train = series[:split]
    test = series[split:]

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train
    )

    test_scaled = scaler.transform(
        test
    )

    def create_sequences(data):

        X = []
        y = []

        for i in range(
            len(data)-look_back
        ):

            X.append(
                data[i:i+look_back]
            )

            y.append(
                data[i+look_back]
            )

        return (
            np.array(X),
            np.array(y)
        )

    X_train, y_train = (
        create_sequences(
            train_scaled
        )
    )

    X_test, y_test = (
        create_sequences(
            test_scaled
        )
    )

    model = Sequential([
        LSTM(
            64,
            return_sequences=True,
            input_shape=(look_back,1)
        ),

        Dropout(0.2),

        LSTM(32),

        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        verbose=0
    )

    y_pred = model.predict(
        X_test,
        verbose=0
    )

    y_pred = scaler.inverse_transform(
        y_pred
    )

    y_test = scaler.inverse_transform(
        y_test
    )

    mape = (
        mean_absolute_percentage_error(
            y_test,
            y_pred
        ) * 100
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    return mape, rmse

# STEP 13: TRAIN MODELS
# =========================================================
print("\n🚀 TRAINING MODELS...\n")

# Prophet
mape_prophet_raw, rmse_prophet_raw = (
    train_prophet(
        df_raw,
        holiday_df
    )
)

mape_prophet_log, rmse_prophet_log = (
    train_prophet(
        df_log,
        holiday_df,
        log=True
    )
)

# XGBoost
mape_xgb, rmse_xgb, xgb_model = (
    train_xgb(df_raw)
)

mape_xgb_log, rmse_xgb_log, _ = (
    train_xgb(df_log)
)

# ARIMA
mape_arima, rmse_arima = (
    train_arima(df_raw)
)

mape_arima_log, rmse_arima_log = (
    train_arima(df_log)
)

# LSTM
mape_lstm, rmse_lstm = (
    train_lstm(df_raw)
)

mape_lstm_log, rmse_lstm_log = (
    train_lstm(df_log)
)

# STEP 14: FINAL COMPARISON
# =========================================================
comparison = pd.DataFrame({

    "Model": [
        "Prophet",
        "ARIMA",
        "XGBoost",
        "LSTM"
    ],

    "MAPE Raw": [
        mape_prophet_raw,
        mape_arima,
        mape_xgb,
        mape_lstm
    ],

    "MAPE Log": [
        mape_prophet_log,
        mape_arima_log,
        mape_xgb_log,
        mape_lstm_log
    ],

    "RMSE Raw": [
        rmse_prophet_raw,
        rmse_arima,
        rmse_xgb,
        rmse_lstm
    ],

    "RMSE Log": [
        rmse_prophet_log,
        rmse_arima_log,
        rmse_xgb_log,
        rmse_lstm_log
    ]
})

print("\n📊 FINAL MODEL COMPARISON\n")

print(
    comparison.round(2)
)

# STEP 15: FUTURE FORECAST
# =========================================================
def future_forecast_xgb(
    df,
    model,
    steps=30
):

    future_df = df.copy()

    forecasts = []

    for i in range(steps):

        # Latest row
        last_row = (
            future_df
            .iloc[-1:]
            .copy()
        )

        # Predict
        X = last_row[FEATURES]

        pred = model.predict(X)[0]

        # FIXED DATE LOGIC
        last_date = pd.to_datetime(
            last_row['ds'].iloc[0]
        )

        next_date = pd.date_range(
            start=last_date,
            periods=2,
            freq=freq
        )[1]

        # New row
        next_row = pd.DataFrame({
            'ds': [next_date],
            'y': [pred]
        })

        # Append
        temp = pd.concat(
            [
                future_df[['ds','y']],
                next_row
            ],
            ignore_index=True
        )

        # Recreate features
        temp.rename(columns={
            'ds': 'Date',
            'y': 'Target'
        }, inplace=True)

        temp_features = create_features(
            temp
        )

        future_df = temp_features.copy()

        forecasts.append({
            "Date": next_date.strftime("%d-%m-%Y"),
            "Forecast": round(
                float(pred),
                2
            )
        })

    return pd.DataFrame(
        forecasts
    )

# STEP 16: GENERATE FORECAST
# =========================================================
forecast_df = future_forecast_xgb(
    df_raw,
    xgb_model,
    steps=30
)

print("\n📈 30-DAY FUTURE FORECAST\n")


# =========================================================
# ADVANCED FORECAST VISUALIZATION
# =========================================================

import matplotlib.pyplot as plt

# Historical
historical_data = data.tail(120).copy()

# Forecast
forecast_plot = forecast_df.copy()

forecast_plot['Date'] = pd.to_datetime(
    forecast_plot['Date'],
    dayfirst=True
)

# =========================================================
# COMBINE FOR CONTINUOUS TREND
# =========================================================

combined_dates = list(historical_data['Date']) + \
                 list(forecast_plot['Date'])

combined_values = list(historical_data['Target']) + \
                  list(forecast_plot['Forecast'])

split_point = len(historical_data)

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(18,7))

# Historical trend
plt.plot(
    combined_dates[:split_point],
    combined_values[:split_point],
    linewidth=2,
    label='Historical Data'
)

# Forecast trend
plt.plot(
    combined_dates[split_point-1:],
    combined_values[split_point-1:],
    linewidth=3,
    linestyle='--',
    label='Forecast'
)

# Forecast start marker
plt.axvline(
    historical_data['Date'].max(),
    linestyle='--'
)

# Labels
plt.title(
    'Forecast Trend for Next 30 Days',
    fontsize=18
)

plt.xlabel('Date')
plt.ylabel('Forecast Value')

plt.legend()

plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()  



print(forecast_df)
