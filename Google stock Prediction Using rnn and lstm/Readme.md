# 📈 Google Stock Price Prediction using RNN & LSTM

## 📌 Overview

This project implements a **Recurrent Neural Network (RNN)** and **Long Short-Term Memory (LSTM)** model to predict future stock prices of **Google (GOOG)** based on historical stock market data. The model is trained on past closing prices and aims to forecast future trends with a focus on sequential dependencies in time series data.

---

## 🗃️ Dataset

* **Source**: Yahoo Finance / Kaggle
* **Content**: Daily stock prices of Google (GOOG)
* **Features**:

  * Date
  * Open
  * High
  * Low
  * Close
  * Volume

---

## 🛠️ Technologies & Tools

* **Python** – Programming language
* **Pandas, NumPy** – Data manipulation and preprocessing
* **Matplotlib, Seaborn** – Data visualization
* **TensorFlow / Keras** – Deep learning framework to build and train RNN/LSTM models
* **Scikit-learn** – Data scaling and preprocessing
* **Google Colab / Jupyter Notebook** – Development and training environment

---

## 🧠 Model Architecture

* **Recurrent Neural Network (RNN)**
* **Long Short-Term Memory (LSTM) layers**
* Dropout layers to prevent overfitting
* Dense layer for final price prediction

---

## 📊 Evaluation Metrics

* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **Visualization of Actual vs Predicted Prices**

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/Simra-Kazi/Deep-Learning/edit/main/Google%20stock%20Prediction%20Using%20rnn%20and%20lstm
cd Google stock Prediction Using rnn and lstm
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook Google_Stock_Prediction_LSTM.ipynb
```

Or open in **Google Colab**.

---

## 📈 Results

* Achieved low RMSE on test set.
* Accurately captured the trend of stock price movement.
* Visual plots comparing actual and predicted prices show strong alignment.

---

## 🔮 Future Improvements

* Integrate real-time stock data for live predictions.
* Combine technical indicators (RSI, MACD) as additional features.
* Deploy the model using a web interface (Streamlit/Flask).

---

## 📄 License

This project is licensed under the **MIT License**.

---



