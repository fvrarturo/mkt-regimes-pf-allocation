from midas_preprocessing import load_macro_data, load_daily_oil_data, prepare_midas_forecast_data

macro_df = load_macro_data()
oil_prices = load_daily_oil_data()
train_data, test_data, full_data, test_dates = prepare_midas_forecast_data(
    macro_df, oil_prices, train_split=0.65, horizons=[1,3,6], theta=0.03, K=60
)
print("Full data length:", len(full_data))
print("Test data length:", len(test_data))
print("Test dates length:", len(test_dates))
print("Max horizon excluded:", len(test_data) - len(test_dates), "dates")
print("Full data shape:", full_data.shape)
print("Oil midas NaN count:", full_data['oil_midas'].isna().sum())

