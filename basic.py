import pmdarima as pm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape

msft = pm.datasets.stocks.load_msft()

simulation = 30 # how many days to run the simulation
horizon = 3 # how far in the future the model forecasts

prices = msft['Close']

# NOTE: drops the first horizon prices to align with exogenous modelling challenge
prices = prices[horizon:].reset_index(drop=True)

train_size = int(len(prices) * 0.9)
train, test = prices[:train_size], prices[train_size:]

# NOTE: you can defer to auto arima to find the "d" parameter via kpss method only.
# Using the maximum recommended differences increases the likelihood of stationarity though
# at a very small extra cost of about 1/3rd of a second.
kpss_diffs = pm.arima.ndiffs(train, alpha=0.05, test='kpss', max_d=6)
adf_diffs = pm.arima.ndiffs(train, alpha=0.05, test='adf', max_d=6)
pp_diffs = pm.arima.ndiffs(train, alpha=0.05, test='pp', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs, pp_diffs)

model = pm.auto_arima(train, d=n_diffs,
                      stepwise=False, n_jobs=-1, # NOTE: 30s -> 4s parallelizes
                      trace=True)

print(model.summary())

forecast_predictions = []
forecast_predictions_ci = []
actuals = []

for t in range(simulation - horizon):
    model.update(test[t:t+1])
    fc, ci = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.1)
    fc, ci = fc[-1], ci[-1] # WARN: type conversion

    actual = test[train_size + t + horizon]

    forecast_predictions.append(fc)
    forecast_predictions_ci.append(ci)
    actuals.append(actual)

print(f"Mean squared error: {mean_squared_error(actuals, forecast_predictions)}")
print(f"SMAPE: {smape(actuals, forecast_predictions)}")

plt.figure(figsize=(10, 6))

dates = msft['Date'][train_size + horizon:train_size + simulation]

plt.plot(dates, forecast_predictions, label=f'Predicted Day {horizon}', color='red')
plt.plot(dates, actuals, label=f'Actual Day {horizon}', color='black')

lower_bounds = [ci[0] for ci in forecast_predictions_ci]
upper_bounds = [ci[1] for ci in forecast_predictions_ci]

plt.fill_between(dates, lower_bounds, upper_bounds, color='lightcoral', alpha=0.5)

plt.title(f'{horizon}-Day Out Predictions vs. Actuals Over {simulation}-Day Period')
plt.xticks(dates[::7], rotation=45)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()
