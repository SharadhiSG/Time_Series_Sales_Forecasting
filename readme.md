# 📈 Time Series Sales Forecasting using Prophet

##  Project Overview

This project focuses on building a **robust time series forecasting model** to predict **monthly product unit sales** using historical sales data.
The objective is to evaluate forecasting performance using **strict rolling 6-month validation** and **MAPE-based accuracy thresholds**, while accounting for **trend, seasonality, and external business factors** such as discounts and pricing.

---

##  Problem Statement

The goal is to develop a forecasting model that:

* Predicts **product unit sales** over time
* Uses **training, testing, and unseen validation splits**
* Evaluates accuracy using:

  * **Month-on-Month (MoM) MAPE ≤ 15%**
  * **Overall 6-Month MAPE ≤ 15%**
* Is resilient to:

  * Seasonal fluctuations
  * Trend changes
  * External business drivers (promotions, pricing)

---

##  What is Time Series Data?

A **time series** is a sequence of observations collected at **regular time intervals** (daily, monthly, yearly).

Examples:

* Monthly sales
* Daily stock prices
* Hourly temperature readings

In this project:

* Each observation represents **monthly shirt sales**

---

##  Components of a Time Series

A time series typically consists of the following components:

### 1️ Trend

* Long-term increase or decrease in data
* Example: Gradual decline in sales after 2019

### 2️ Seasonality

* Repeating patterns at fixed intervals
* Example: Higher sales during festive months

### 3️ Cyclical Effects

* Irregular fluctuations caused by external events
* Example: Economic slowdowns or pandemics

### 4️ Noise (Residuals)

* Random variation not explained by trend or seasonality

Understanding these components is essential for choosing the right forecasting model.

---

##  Tools & Technologies Used

* **Python**
* **Pandas & NumPy** – data manipulation
* **Matplotlib** – exploratory visualizations
* **Prophet (fbprophet)** – time series forecasting
* **Scikit-learn** – evaluation metrics (MAPE)
* **Google Colab** – execution environment

---

##  Dataset Description

The dataset contains historical sales records with the following fields:

| Column Name          | Description                          |
| -------------------- | ------------------------------------ |
| `sale_date`          | Date of sale                         |
| `product_unit_sales` | Units sold                           |
| `discount_amount`    | Total discount applied               |
| `mrp_amount`         | Maximum Retail Price                 |
| `product`            | Product name (single product: shirt) |

---

##  Data Preprocessing Steps

1. **Load dataset**
2. **Convert date column to datetime**
3. **Sort data chronologically**
4. **Aggregate transaction-level data to monthly level**
5. **Handle single-product scope**
6. **Rename columns for Prophet compatibility**
7. **Apply log transformation to stabilize variance**

---

##  Exploratory Data Analysis (EDA)

EDA was performed to understand the data before modeling:

### ✔ Sales Trend Visualization

* Revealed high volatility and a **structural break around 2020**
* Justified the need for **changepoints**

### ✔ Seasonality Analysis

* Monthly averages highlighted **recurring seasonal patterns**
* Justified enabling yearly and quarterly seasonality in Prophet

### ✔ Business Insight

* Sales are influenced by promotions and pricing
* Motivated inclusion of external regressors

---

##  Why Prophet?

Prophet was chosen because it:

* Automatically models **trend and seasonality**
* Handles **missing data and outliers**
* Supports **external regressors**
* Is interpretable and industry-accepted

---

##  Prophet Model Configuration

Key settings used:

* **Yearly seasonality** enabled
* **Weekly & daily seasonality** disabled (monthly data)
* **Changepoints** to capture trend shifts
* **External regressors**:

  * Discount amount
  * MRP amount

---

##  Rolling 6-Month Validation Strategy

Instead of a single train-test split, the project uses:

* **Rolling-window validation**
* Each iteration:

  * Trains on historical data
  * Validates on the **next unseen 6 months**

This ensures:

* No data leakage
* Realistic evaluation
* Robust performance assessment

---

## 📐 Evaluation Metrics

### 🔹 Month-on-Month (MoM) MAPE

Measures accuracy for each individual month.

### 🔹 Overall 6-Month MAPE

Measures cumulative forecasting accuracy across the validation window.

MAPE Formula:

```
MAPE = mean(|Actual − Predicted| / Actual) × 100
```

---

##  Model Performance Summary

* **MoM MAPE ≤ 15%** achieved in ~15% of windows
* **Overall 6-Month MAPE ≤ 15%** achieved in ~22% of windows

---

## Interpretation of Results

* The model performs **better at medium-term forecasting**
* Short-term accuracy is affected by:

  * Promotions
  * Sudden demand changes
  * Structural breaks
* Prophet captures **trend and seasonality well**, but smooths sharp spikes

---

##  Key Insight

Strict MoM accuracy thresholds are **challenging for retail sales data** due to inherent volatility.
This reflects a **real-world limitation**, not a modeling error.

---

##  Final Forecast Visualization

The final model was trained on the full dataset and used to forecast future sales.
Forecast plots include:

* Actual vs predicted values
* Confidence intervals
* Trend and seasonality components

---

##  Conclusion

* Prophet is effective for **strategic and long-term sales forecasting**
* Rolling validation provides a realistic measure of performance
* External regressors improve interpretability
* Short-term promotional forecasting requires hybrid or ML-based extensions

---

## 🚀 Future Enhancements

* Hybrid models (Prophet + ML residual correction)
* Promotion flags and event-based regressors
* Lag-based machine learning models (XGBoost, LSTM)
* Separate modeling for promotional vs non-promotional periods

---
