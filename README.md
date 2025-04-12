# Stock Market Price Prediction System

A sophisticated stock market price prediction system that combines technical analysis, sentiment analysis, and machine learning to forecast stock prices. This system uses historical price data, technical indicators, and market sentiment to make predictions.

## Features

- **Technical Analysis**: Utilizes multiple technical indicators including:
  - Moving Averages (5, 10, 20, 50, 200-day)
  - RSI, MACD, Bollinger Bands
  - Volume indicators
  - Price momentum patterns
  - Volatility measures

- **Sentiment Analysis**: Integrates market sentiment through:
  - Historical news analysis
  - Market impact assessment
  - Confidence scoring
  - Sentiment trends

- **Machine Learning**: Implements an ensemble model combining:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Feature selection
  - Cross-validation

## Requirements

- Python 3.8+
- Required packages:
  ```
  pandas
  numpy
  scikit-learn
  matplotlib
  ta (Technical Analysis library)
  openai
  joblib
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd stock-market
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

## Usage

1. Prepare your data:
   - Place your historical stock data in CSV format in the project directory
   - Ensure the CSV contains columns: Date, Open, High, Low, Close/Last, Volume

2. Run the prediction system:
   ```bash
   python stock_predictor.py
   ```

3. The system will:
   - Process historical data
   - Generate technical indicators
   - Analyze market sentiment
   - Train the prediction model
   - Generate predictions and visualizations

## Output

The system generates:
- Model performance metrics (MSE, RMSE, R2 Score)
- Price prediction accuracy within different margins
- Feature importance analysis
- Prediction vs. actual price plots
- Bias analysis

## Model Performance

The model provides:
- Price predictions for the next trading day
- Confidence intervals for predictions
- Sentiment impact analysis
- Technical indicator-based signals

## File Structure

- `stock_predictor.py`: Main prediction system
- `requirements.txt`: Project dependencies
- `*.joblib`: Saved model files
- `*.png`: Generated prediction plots

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the GPT API
- Technical Analysis library for indicators
- Scikit-learn for machine learning implementation
