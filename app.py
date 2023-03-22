from flask import Flask, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeRegressor
import plotly.graph_objs as go
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def home():

	end_date = datetime.now() - timedelta(days=1)
	start_date = end_date - timedelta(days=365*7) 

	df = yf.download("AAPL", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

	if df.isnull().values.any():
	    df = df.dropna()

	X = df["Close"].values[:-1].reshape(-1, 1)
	y = df["Close"].values[1:].reshape(-1, 1)

	train_size = int(len(X) * 0.8)
	X_train, X_test = X[:train_size], X[train_size:]
	y_train, y_test = y[:train_size], y[train_size:]

	regressor = DecisionTreeRegressor(random_state=0)
	regressor.fit(X_train, y_train)

	prediction = regressor.predict(X_test[-1].reshape(1, -1))

	index = pd.date_range(df.index[-1], periods=2, freq="D")[1:]
	predictions_df = pd.DataFrame({"Close": prediction}, index=index)

	combined_df = pd.concat([df, predictions_df])

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df["Close"], name="Actual"))
	fig.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df["Close"], name="Predicted"))

	fig.update_layout(
	    title="Actual vs Predicted Closing Price",
	    xaxis_title="Date",
	    yaxis_title="Price",
	    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
	    margin=dict(l=20, r=20, t=60, b=20),
	)

	return jsonify(fig.show())


if __name__ == '__main__':
    app.run()













	# %tb
# from flask import Flask, render_template, jsonify
# import pandas as pd
# import pickle
# import yfinance as yf
# from sklearn.tree import DecisionTreeRegressor
# from datetime import datetime, timedelta

# app = Flask(__name__)

# # Define the route to serve the live dashboard
# @app.route('/')
# def index():
#     return render_template('dashboard.html')

# # Define a function to load the trained model and make predictions
# def predict_closing_price(ticker):
#     # Set the date range
#     end_date = datetime.now() - timedelta(days=1)
#     start_date = end_date - timedelta(days=365*7) # 7 years of data

#     # Load the data from Yahoo Finance
#     df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

#     # Check for missing values and drop them
#     if df.isnull().values.any():
#         df = df.dropna()

#     # Define the training data and the number of steps to forecast
#     X = df["Close"].values[:-1].reshape(-1, 1)
#     y = df["Close"].values[1:].reshape(-1, 1)

#     # Split the data into training and testing sets
#     train_size = int(len(X) * 0.8)
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]

#     # Train the decision tree regressor on the training data
#     regressor = DecisionTreeRegressor(random_state=0)
#     regressor.fit(X_train, y_train)

#     # Generate predictions for the next day's closing price
#     prediction = regressor.predict(X_test[-1].reshape(1, -1))

#     # Convert the prediction to a pandas DataFrame
#     index = pd.date_range(df.index[-1], periods=2, freq="D")[1:]
#     predictions_df = pd.DataFrame({"Close": prediction}, index=index)

#     # Concatenate the original data and the predictions
#     combined_df = pd.concat([df, predictions_df])

#     # Return the most recent data and the predicted closing price
#     return combined_df[-1:], prediction[0]

# # Define a route to handle AJAX requests and return the predicted closing price
# @app.route('/predict/<ticker>')
# def predict(ticker):
#      def show_graph():
#              return jsonify(fig.to_json())

# if __name__ == '__main__':
#     app.run(debug=False)
