# Stock Price Prediction — Hybrid SARIMA + LSTM

Forecasting daily closing prices of **NIFTY 50 (`^NSEI`)** and **TCS (`TCS.NS`)** using a
statistical base model whose *residuals* are corrected by a neural network.

A classical SARIMA model captures the linear, trend and seasonal structure of the series.
Whatever it gets wrong — the residuals — still contains non-linear structure, so an LSTM is
trained on those residuals and its output is added back to the SARIMA forecast.

```
final_prediction = SARIMA_forecast + LSTM(recent_residuals)
```

On TCS this cut test RMSE from **230.22 → 53.57** and MAPE from **5.61% → 1.03%**.

---

## Results

Trained on the first 80% of 10 years of daily data, evaluated on the held-out final 20%.

### TCS.NS

| Metric | SARIMA baseline | Hybrid (SARIMA + LSTM) | Improvement |
|---|---|---|---|
| RMSE | 230.22 | **53.57** | −76.7% |
| MAPE | 5.61% | **1.03%** | −4.58 pp |
| MASE | 82.15 | **1.08** | — |

### NIFTY 50 (^NSEI)

| Metric | SARIMA baseline | Hybrid (SARIMA + LSTM) | Improvement |
|---|---|---|---|
| RMSE | 1854.59 | **245.12** | −86.8% |
| MAPE | 6.43% | **0.92%** | −5.51 pp |
| MASE | 130.19 | **2.37** | — |

---

## Pipeline

**1. Data**
10 years of daily OHLCV data pulled with `yfinance` (`period='10y'`, data through 2025-01-29).
Chronological 80/20 train–test split — no shuffling, so the test set is strictly future data.

**2. Stationarity**
Augmented Dickey–Fuller test on the raw closing prices:

| Series | ADF statistic | p-value | Verdict |
|---|---|---|---|
| NIFTY 50 | −0.29 | 0.927 | Not stationary |
| TCS | −0.34 | 0.919 | Not stationary |

After second-order differencing (`.diff(2)`):

| Series | ADF statistic | p-value | Verdict |
|---|---|---|---|
| NIFTY 50 | −7.36 | 9.5e−11 | Stationary |
| TCS | −9.36 | 8.1e−16 | Stationary |

**3. Order selection**
ACF and PACF plots of the differenced series were used to read off the AR and MA orders,
giving `order = (p=4, d=2, q=3)` for both series. A seasonal component
`seasonal_order = (1, 1, 1, s)` was added, with `s = 10` for NIFTY 50 and `s = 7` for TCS,
fit via `statsmodels` `SARIMAX`.

**4. Residual model**
Residuals `y_true − y_SARIMA` are min–max normalised to [0, 1] and turned into supervised
sequences with a **10-step lag window** (predict residual `t` from residuals `t−10 … t−1`).
A single-layer PyTorch LSTM (`hidden_dim=15`) with a linear head is trained for 20 epochs
with Adam (`lr=0.01`) and MSE loss.

**5. Recombination**
At inference the LSTM's predicted residual is denormalised and added to the SARIMA
prediction. The first 10 test points have no lag window available, so they fall back to the
plain SARIMA output. The same rolling procedure extends the forecast to the end of 2025.

---

## Repo contents

```
Stock_price_prediction.ipynb   # full pipeline: data → SARIMA → residual LSTM → evaluation
README.md
```

## Running it

```bash
pip install numpy pandas yfinance matplotlib statsmodels torch
jupyter notebook Stock_price_prediction.ipynb
```
