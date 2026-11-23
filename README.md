# Stock Price Prediction — Hybrid SARIMA + LSTM

Forecasting daily closing prices of **NIFTY 50 (`^NSEI`)** and **TCS (`TCS.NS`)** using a
statistical base model whose *residuals* are corrected by a neural network.

A classical SARIMA model captures the linear, trend and seasonal structure of the series.
Whatever it gets wrong — the residuals — still contains non-linear structure, so an LSTM is
trained on those residuals and its output is added back to the SARIMA forecast.

```
final_prediction = SARIMA_forecast + LSTM(recent_residuals)
```
