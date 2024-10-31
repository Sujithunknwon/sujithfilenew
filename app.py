from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import os

app = Flask(_name_)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return "No file provided"

    file = request.files['file']
    n_parts_per_day = int(request.form['num_parts'])

    # Read the file into a DataFrame
    df = pd.read_excel(file)
    
    # Preprocess Data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Squareness'] = df['Values'].rolling(window=3).mean() - df['Values'].shift(1)
    df['Rolling_Mean'] = df['Values'].rolling(window=3).mean()
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    df['Lag_1'] = df['Values'].shift(1)
    df['Lag_2'] = df['Values'].shift(2)
    df.dropna(inplace=True)
    
    # Prepare data for prediction
    X = df[['Rolling_Mean', 'Month', 'Day', 'DayOfWeek', 'Lag_1', 'Lag_2', 'Squareness']]
    y = df['Values']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    # Predict future deviation
    future_date = df.index[-1]
    exceeded_part = None
    prediction_result = f"Model RMSE: {rmse:.4f}\n"
    
    for part in range(1, n_parts_per_day + 1):
        future_features = pd.DataFrame({
            'Rolling_Mean': [df['Rolling_Mean'].iloc[-3:].mean()],
            'Month': [future_date.month],
            'Day': [future_date.day],
            'DayOfWeek': [future_date.dayofweek],
            'Lag_1': [df['Values'].iloc[-1]],
            'Lag_2': [df['Values'].iloc[-2]],
            'Squareness': [df['Squareness'].iloc[-1]]
        }).ffill().bfill()
        
        future_pred = best_model.predict(future_features)[0]
        future_date += pd.Timedelta(days=1 / n_parts_per_day)
        prediction_result += f"Part {part}: {future_pred:.4f}\n"
        
        if future_pred > 0.066 and exceeded_part is None:
            exceeded_part = part
    
    if exceeded_part is not None:
        deviation_date = future_date
        prediction_result += f"Deviation exceeds 0.066 on {deviation_date.strftime('%Y-%m-%d')} (Day: {deviation_date.strftime('%A')}) with predicted value: {future_pred:.4f}"
    else:
        prediction_result += "Deviation did not exceed 0.066 in the specified parts."
    
    return render_template('result.html', prediction=prediction_result)

if _name_ == "_main_":
    app.run(debug=True)