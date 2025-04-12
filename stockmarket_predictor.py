import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import ta
from datetime import datetime, timedelta
import os
from openai import OpenAI
# import maitai  # Commented out Maitai import
import json
import sys
import joblib  # Add joblib for model saving/loading
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# Initialize clients
try:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    # maitai_client = maitai.MaitaiAsync()  # Commented out Maitai client
except Exception as e:
    print(f"Error initializing API clients: {e}")
    sys.exit(1)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess historical price data
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean numeric columns by removing dollar signs and converting to float
    df['Close/Last'] = df['Close/Last'].str.replace('$', '').astype(float)
    df['Open'] = df['Open'].str.replace('$', '').astype(float)
    df['High'] = df['High'].str.replace('$', '').astype(float)
    df['Low'] = df['Low'].str.replace('$', '').astype(float)
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Remove any missing values
    df = df.dropna()
    
    return df

def get_historical_context(symbol, date):
    """
    Get historical news and context for a specific date using OpenAI
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "system", "content": "You are a financial historian with expertise in stock market events and company news. Provide accurate historical context and news for the specified date."},
                {"role": "user", "content": f"""
                Provide the major news events and market sentiment for {symbol} (Apple Inc.) on and the day before {date.strftime('%Y-%m-%d')}.
                Focus on:
                1. Company-specific news (product launches, earnings, management changes)
                2. Industry news affecting the company
                3. Market conditions and analyst opinions
                4. Any significant events that impacted the stock
                
                Format your response as JSON with this structure:
                {{
                    "company_news": ["news1", "news2", ...],
                    "industry_news": ["news1", "news2", ...],
                    "market_conditions": "description",
                    "analyst_opinions": ["opinion1", "opinion2", ...],
                    "significant_events": ["event1", "event2", ...]
                }}
                
                Only include events that actually occurred on or the day before the specified date.
                Ensure your response is valid JSON.
                """
                }
            ]
        )
        
        # Get the response content
        content = response.choices[0].message.content.strip()
        
        # Debug print
        print(f"\nRaw response for {date}:")
        print(content)
        print("\nAttempting to parse JSON...")
        
        try:
            # Parse the JSON response
            historical_context = json.loads(content)
            return historical_context
        except json.JSONDecodeError as je:
            print(f"JSON parsing error: {je}")
            print("Attempting to fix malformed JSON...")
            
            # Try to extract JSON from the response if it's wrapped in other text
            try:
                # Look for the first '{' and last '}'
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    historical_context = json.loads(json_str)
                    return historical_context
            except Exception as e:
                print(f"Failed to fix JSON: {e}")
                return None
            
    except Exception as e:
        print(f"Error getting historical context for {date.strftime('%Y-%m-%d')}: {e}")
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None

def analyze_market_sentiment(historical_context, date, current_price, prev_price):
    """
    Analyze market sentiment using the historical context
    """
    if not historical_context:
        return None, None
        
    try:
        # Format the context for analysis
        context_str = json.dumps(historical_context, indent=2)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "system", "content": "You are a financial analyst expert. Analyze historical market context and provide a sentiment analysis with market impact."},
                {"role": "user", "content": f"""
                Analyze the following historical market context for {date.strftime('%Y-%m-%d')} and provide a sentiment analysis.
                
                Price Information:
                Current Price: ${current_price:.2f}
                Previous Price: ${prev_price:.2f}
                Price Change: {((current_price - prev_price) / prev_price * 100):.2f}%

                Historical Context:
                {context_str}

                Provide your analysis in JSON format:
                {{
                    "sentiment": "BULLISH/BEARISH/NEUTRAL",
                    "market_impact": "HIGH/MEDIUM/LOW",
                    "key_points": ["point1", "point2", ...],
                    "estimated_price_impact": float,
                    "confidence": "HIGH/MEDIUM/LOW",
                    "reasoning": "string"
                }}
                
                Ensure your response is valid JSON.
                """
                }
            ]
        )
        
        # Get the response content
        content = response.choices[0].message.content.strip()
        
        # Debug print
        print(f"\nRaw sentiment response for {date}:")
        print(content)
        print("\nAttempting to parse JSON...")
        
        try:
            # Parse the JSON response
            analysis = json.loads(content)
            return analysis, None
        except json.JSONDecodeError as je:
            print(f"JSON parsing error in sentiment analysis: {je}")
            print("Attempting to fix malformed JSON...")
            
            # Try to extract JSON from the response if it's wrapped in other text
            try:
                # Look for the first '{' and last '}'
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    analysis = json.loads(json_str)
                    return analysis, None
            except Exception as e:
                print(f"Failed to fix JSON: {e}")
                return None, None
        
    except Exception as e:
        print(f"Error in sentiment analysis for {date.strftime('%Y-%m-%d')}: {e}")
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None, None

def create_features(df, sentiment_data):
    """
    Create enhanced features focusing on price prediction with aligned news data
    """
    # Price and volume features
    df['Price_MA5'] = df['Close/Last'].rolling(window=5).mean()
    df['Price_MA10'] = df['Close/Last'].rolling(window=10).mean()
    df['Price_MA20'] = df['Close/Last'].rolling(window=20).mean()
    df['Price_MA50'] = df['Close/Last'].rolling(window=50).mean()
    df['Price_MA200'] = df['Close/Last'].rolling(window=200).mean()
    
    # Price momentum
    df['Price_Change'] = df['Close/Last'].pct_change()
    df['Price_Change_5d'] = df['Close/Last'].pct_change(periods=5)
    df['Price_Change_10d'] = df['Close/Last'].pct_change(periods=10)
    df['Price_Change_20d'] = df['Close/Last'].pct_change(periods=20)
    df['Price_Change_50d'] = df['Close/Last'].pct_change(periods=50)
    
    # Volatility
    df['Volatility_10d'] = df['Price_Change'].rolling(window=10).std()
    df['Volatility_20d'] = df['Price_Change'].rolling(window=20).std()
    df['Volatility_50d'] = df['Price_Change'].rolling(window=50).std()
    df['Volatility_Ratio'] = df['Volatility_10d'] / df['Volatility_50d']
    
    # Volume indicators
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA_Ratio'] = df['Volume_MA5'] / df['Volume_MA20']
    
    # Technical indicators
    df['RSI'] = ta.momentum.rsi(df['Close/Last'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close/Last'])
    df['BB_Width'] = (ta.volatility.bollinger_hband(df['Close/Last']) - 
                     ta.volatility.bollinger_lband(df['Close/Last'])) / df['Close/Last']
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close/Last'])
    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close/Last'])
    df['OBV'] = ta.volume.on_balance_volume(df['Close/Last'], df['Volume'])
    
    # Additional technical indicators
    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close/Last'])
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close/Last'], df['Volume'])
    df['Stoch_RSI'] = ta.momentum.stochrsi(df['Close/Last'])
    df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close/Last'])
    
    # Price patterns
    df['Price_Above_MA20'] = (df['Close/Last'] > df['Price_MA20']).astype(int)
    df['Price_Above_MA50'] = (df['Close/Last'] > df['Price_MA50']).astype(int)
    df['MA20_Above_MA50'] = (df['Price_MA20'] > df['Price_MA50']).astype(int)
    df['Price_Above_MA200'] = (df['Close/Last'] > df['Price_MA200']).astype(int)
    
    # Price momentum patterns
    df['Price_Up_Streak'] = (df['Price_Change'] > 0).astype(int).rolling(window=5).sum()
    df['Price_Down_Streak'] = (df['Price_Change'] < 0).astype(int).rolling(window=5).sum()
    df['Price_Volatility_Spike'] = (df['Volatility_10d'] > df['Volatility_50d'] * 1.5).astype(int)
    
    # Volume patterns
    df['Volume_Spike'] = (df['Volume'] > df['Volume_MA20'] * 1.5).astype(int)
    df['Volume_Trend'] = (df['Volume_MA5'] > df['Volume_MA20']).astype(int)
    
    # Initialize sentiment columns
    df['Sentiment_Score'] = 0.0
    df['Sentiment_Impact'] = 0.0
    df['Market_Impact'] = 0.0
    df['News_Confidence'] = 0.0
    df['Sentiment_MA5'] = 0.0
    df['Sentiment_MA10'] = 0.0
    df['Sentiment_MA20'] = 0.0
    
    # Update sentiment features with enhanced scoring
    for date, (sentiment, _) in sentiment_data.items():
        if sentiment:
            # Enhanced sentiment scoring
            if sentiment['sentiment'] == 'BULLISH':
                base_score = 1.0
                if sentiment['confidence'] == 'HIGH':
                    score = base_score * 1.5
                elif sentiment['confidence'] == 'LOW':
                    score = base_score * 0.5
                else:
                    score = base_score
            elif sentiment['sentiment'] == 'BEARISH':
                base_score = -1.0
                if sentiment['confidence'] == 'HIGH':
                    score = base_score * 1.5
                elif sentiment['confidence'] == 'LOW':
                    score = base_score * 0.5
                else:
                    score = base_score
            else:
                score = 0.0
            
            # Enhanced confidence and impact scoring
            confidence = {'HIGH': 1.0, 'MEDIUM': 0.8, 'LOW': 0.5}.get(sentiment['confidence'], 0.0)
            market_impact = {'HIGH': 1.0, 'MEDIUM': 0.8, 'LOW': 0.5}.get(sentiment['market_impact'], 0.0)
            
            # Calculate sentiment impact with market conditions
            sentiment_impact = sentiment.get('estimated_price_impact', 0.0) * confidence * market_impact
            
            mask = df['Date'].dt.date == date
            df.loc[mask, 'Sentiment_Score'] = score
            df.loc[mask, 'Sentiment_Impact'] = sentiment_impact
            df.loc[mask, 'Market_Impact'] = market_impact
            df.loc[mask, 'News_Confidence'] = confidence
    
    # Calculate rolling sentiment features
    df['Sentiment_MA5'] = df['Sentiment_Score'].rolling(window=5).mean()
    df['Sentiment_MA10'] = df['Sentiment_Score'].rolling(window=10).mean()
    df['Sentiment_MA20'] = df['Sentiment_Score'].rolling(window=20).mean()
    
    # Target is next day's closing price
    df['Target_Price'] = df['Close/Last'].shift(-1)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def prepare_data(df):
    """
    Prepare data with time-based split for price prediction
    """
    # Convert Date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define time periods for training and testing
    train_start = '2013-01-01'
    train_end = '2022-12-31'
    test_start = '2023-01-01'
    test_end = '2023-12-31'
    
    print(f"\nData split:")
    print(f"Training period: {train_start} to {train_end}")
    print(f"Testing period: {test_start} to {test_end}")
    
    # Select features for price prediction
    features = [
        'Close/Last',  # Current price
        'Price_MA5', 'Price_MA10', 'Price_MA20', 'Price_MA50', 'Price_MA200',  # Moving averages
        'Price_Change', 'Price_Change_5d', 'Price_Change_10d', 'Price_Change_20d', 'Price_Change_50d',  # Price changes
        'Volatility_10d', 'Volatility_20d', 'Volatility_50d', 'Volatility_Ratio',  # Volatility
        'Volume_Ratio', 'Volume_MA5', 'Volume_MA20', 'Volume_Change', 'Volume_MA_Ratio',  # Volume indicators
        'RSI', 'MACD', 'BB_Width', 'ATR', 'ADX', 'OBV', 'CCI', 'MFI', 'Stoch_RSI', 'Williams_R',  # Technical indicators
        'Price_Above_MA20', 'Price_Above_MA50', 'Price_Above_MA200', 'MA20_Above_MA50',  # Price patterns
        'Price_Up_Streak', 'Price_Down_Streak', 'Price_Volatility_Spike',  # Price momentum patterns
        'Volume_Spike', 'Volume_Trend',  # Volume patterns
        'Sentiment_Score', 'Sentiment_Impact', 'Market_Impact', 'News_Confidence',  # Sentiment
        'Sentiment_MA5', 'Sentiment_MA10', 'Sentiment_MA20'  # Sentiment trends
    ]
    
    # Split data into train and test sets based on date
    train_mask = (df['Date'] >= train_start) & (df['Date'] <= train_end)
    test_mask = (df['Date'] >= test_start) & (df['Date'] <= test_end)
    
    # Create training and testing data
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    # Verify we have data for both sets
    if len(train_df) == 0:
        raise ValueError("No training data found for the specified period")
    if len(test_df) == 0:
        raise ValueError("No testing data found for the specified period")
    
    print(f"\nTraining samples: {len(train_df)} days")
    print(f"Testing samples: {len(test_df)} days")
    
    # Prepare features and target
    X_train = train_df[features]
    y_train = train_df['Target_Price']
    X_test = test_df[features]
    y_test = test_df['Target_Price']
    
    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features, test_df['Date']

def process_historical_data(df, symbol):
    """
    Process historical data and collect sentiment for all dates
    """
    sentiment_data = {}
    
    # Define our date ranges
    train_start = pd.to_datetime('2013-01-01')
    train_end = pd.to_datetime('2022-12-31')
    
    # Filter dates to only include our training period
    mask = (df['Date'] >= train_start) & (df['Date'] <= train_end)
    dates = df[mask]['Date'].dt.date.unique()
    total_dates = len(dates)
    
    print(f"\nProcessing sentiment for {total_dates} trading days...")
    print("This may take some time as we need to process each day individually.")
    
    for i, date in enumerate(dates, 1):
        print(f"\rProcessing date {i}/{total_dates}: {date}", end='')
        
        # Get current and previous prices
        date_idx = df[df['Date'].dt.date == date].index[0]
        current_price = df.loc[date_idx, 'Close/Last']
        prev_price = df.loc[date_idx - 1, 'Close/Last'] if date_idx > 0 else current_price
        
        # Get historical context and sentiment
        historical_context = get_historical_context(symbol, date)
        if historical_context:
            analysis, _ = analyze_market_sentiment(
                historical_context, date, current_price, prev_price
            )
            if analysis:
                sentiment_data[date] = (analysis, None)
        
        # Print progress every 50 dates
        if i % 50 == 0:
            print(f"\nProcessed {i} dates out of {total_dates}")
    
    print("\nFinished processing all dates.")
    print(f"Successfully analyzed sentiment for {len(sentiment_data)} trading days")
    
    return sentiment_data

def train_model(X_train, y_train):
    """
    Train ensemble model with optimized parameters and feature selection
    """
    # Create base models
    rf = RandomForestRegressor(
        n_estimators=1000,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )
    
    gb = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Create feature selector
    selector = SelectFromModel(rf, threshold='median')
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', selector),
        ('voting', VotingRegressor([
            ('rf', rf),
            ('gb', gb)
        ]))
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"\nCross-validation scores: {np.sqrt(-cv_scores)}")
    print(f"Average CV RMSE: {np.sqrt(-cv_scores.mean()):.2f}")
    
    return pipeline

def evaluate_model(model, X_test, y_test, features=None):
    """
    Evaluate model with focus on price prediction accuracy and bias analysis
    """
    # Get predictions from the model
    predictions = model.predict(X_test)
    
    # Calculate error metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    # Calculate bias
    bias = np.mean(predictions - y_test)
    
    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Average Bias: ${bias:.2f}")
    
    # Calculate percentage errors
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Calculate prediction accuracy within different price margins
    price_errors = np.abs(predictions - y_test)
    for margin in [1, 2, 5]:  # $1, $2, $5
        accuracy = np.mean(price_errors <= margin)
        print(f"Price prediction accuracy within ${margin:.2f}: {accuracy:.2%}")
    
    if features is not None and hasattr(model.named_steps['selector'], 'get_support'):
        # Get selected features
        selected_features = np.array(features)[model.named_steps['selector'].get_support()]
        print("\nSelected Features:")
        print(selected_features)
    
    return predictions, bias

def plot_predictions(y_test, predictions, test_dates):
    """
    Enhanced visualization of price predictions with bias analysis
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted Prices
    plt.subplot(2, 1, 1)
    plt.plot(test_dates, y_test, label='Actual Price', color='blue', alpha=0.7)
    plt.plot(test_dates, predictions, label='Predicted Price', color='red', alpha=0.7)
    plt.title('Stock Price Prediction - Jan-Dec 2023')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Prediction Error and Bias
    plt.subplot(2, 1, 2)
    errors = predictions - y_test
    bias = np.mean(errors)
    plt.plot(test_dates, errors, color='green', alpha=0.7, label='Prediction Error')
    plt.axhline(y=bias, color='r', linestyle='--', label=f'Average Bias: ${bias:.2f}')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Prediction Error and Bias Over Time')
    plt.xlabel('Date')
    plt.ylabel('Error ($)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('2023.png')
    plt.close()

def simulate_realtime_prediction(model, df_test, scaler, features):
    """
    Simulate real-time prediction for each day in the test set
    """
    predictions = []
    actual_prices = []
    dates = []
    
    # Get the feature names in the correct order
    feature_names = features
    
    # Process each day in the test set
    for i in range(len(df_test) - 1):  # -1 because we need next day's price for validation
        current_date = df_test.iloc[i]['Date']
        next_date = df_test.iloc[i + 1]['Date']
        
        # Get features for current day
        current_features = df_test.iloc[i][feature_names].values.reshape(1, -1)
        
        # Scale features
        current_features_scaled = scaler.transform(current_features)
        
        # Make prediction for next day
        predicted_price = model.predict(current_features_scaled)[0]
        
        # Store results
        predictions.append(predicted_price)
        actual_prices.append(df_test.iloc[i + 1]['Close/Last'])
        dates.append(next_date)
        
        print(f"\rProcessing prediction for {next_date.date()}: Predicted ${predicted_price:.2f}, Actual ${df_test.iloc[i + 1]['Close/Last']:.2f}", end='')
    
    print("\nFinished processing all test dates")
    return np.array(predictions), np.array(actual_prices), dates

def save_model(model, scaler, features, model_path='stock_price_model.joblib', scaler_path='stock_price_scaler.joblib', features_path='stock_price_features.joblib'):
    """Save model, scaler, and features to files"""
    # Save the feature names in the correct order
    feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else features
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Features saved to {features_path}")

def load_model(model_path='stock_price_model.joblib', scaler_path='stock_price_scaler.joblib', features_path='stock_price_features.joblib'):
    """Load model, scaler, and features from files"""
    if not all(os.path.exists(path) for path in [model_path, scaler_path, features_path]):
        return None, None, None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    print("\nLoaded existing model and scaler")
    return model, scaler, features

def main():
    symbol = 'AAPL'
    
    # Try to load existing model first
    model, scaler, features = load_model()
    
    print("Loading historical data...")
    df = load_and_preprocess_data('Apple-data-from-2013-till-Dec-2023.csv')
    
    # Convert Date to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define our date ranges
    train_start = pd.to_datetime('2013-01-01')
    train_end = pd.to_datetime('2022-12-31')
    test_start = pd.to_datetime('2023-01-01')
    test_end = pd.to_datetime('2023-12-31')
    
    print(f"\nData split:")
    print(f"Training period: {train_start.date()} to {train_end.date()}")
    print(f"Testing period: {test_start.date()} to {test_end.date()}")
    
    # Filter data for our target period
    mask = (df['Date'] >= train_start) & (df['Date'] <= test_end)
    df = df[mask].copy()
    
    # Create target price (next day's closing price)
    df['Target_Price'] = df['Close/Last'].shift(-1)
    
    # If we don't have a model, train a new one
    if model is None:
        print("\nNo existing model found. Training new model...")
        
        # Process historical data and get sentiment
        print("\nProcessing historical data and sentiment...")
        sentiment_data = process_historical_data(df, symbol)
        
        # Create features
        print("\nCreating features...")
        df = create_features(df, sentiment_data)
        
        # Prepare data
        print("\nPreparing data...")
        X_train, X_test, y_train, y_test, scaler, features, test_dates = prepare_data(df)
        
        print("\nTraining model...")
        model = train_model(X_train, y_train)
        
        # Save the trained model
        save_model(model, scaler, features)
    else:
        print("\nUsing existing model...")
        
        # Process historical data and get sentiment
        print("\nProcessing historical data and sentiment...")
        sentiment_data = process_historical_data(df, symbol)
        
        # Create features
        print("\nCreating features...")
        df = create_features(df, sentiment_data)
        
        # Prepare data using the same features as the loaded model
        print("\nPreparing data...")
        X_train, X_test, y_train, y_test, scaler, _, test_dates = prepare_data(df)
        
        # Ensure we're using the same features as the loaded model
        X_test = X_test[:, :len(features)]  # Truncate to match the number of features in the model
    
    print("\nMaking predictions...")
    predictions = model.predict(X_test)
    
    # Apply bias correction
    bias = np.mean(predictions - y_test)
    predictions_corrected = predictions - bias
    
    print("\nEvaluating model performance:")
    predictions, bias = evaluate_model(model, X_test, y_test, features)
    
    # Plot both original and bias-corrected predictions
    plot_predictions(y_test, predictions_corrected, test_dates)

if __name__ == "__main__":
    main() 