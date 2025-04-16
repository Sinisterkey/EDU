import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataProcessor:
    """Class to process market data for the trading agent"""
    
    def __init__(self):
        self.latest_data = None
        self.latest_update = None
        
    def preprocess_data(self, df):
        """Preprocess raw data from MT5"""
        if df is None or len(df) == 0:
            logger.warning("Empty dataframe provided")
            return None
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure dataframe has required columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Required column {col} not found in dataframe")
                return None
                
        # Convert time to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
            
        # Sort by time
        df = df.sort_values('time')
        
        # Calculate basic features
        df['returns'] = df['close'].pct_change()
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / df['range']
        
        # Identify potential turning points
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        # Volume analysis
        df['volume_ma5'] = df['tick_volume'].rolling(5).mean()
        df['volume_ratio'] = df['tick_volume'] / df['volume_ma5']
        df['high_volume'] = df['volume_ratio'] > 1.5
        
        # Mark potential breakout candles
        df['breakout_up'] = (df['close'] > df['open']) & (df['body_pct'] > 0.6) & df['high_volume']
        df['breakout_down'] = (df['close'] < df['open']) & (df['body_pct'] > 0.6) & df['high_volume']
        
        # Fill NaN values from calculations
        df = df.fillna(0)
        
        self.latest_data = df
        self.latest_update = datetime.now()
        
        logger.debug(f"Preprocessed dataframe with {len(df)} rows")
        return df

    def get_feature_vector(self, df=None, lookback=10):
        """Generate feature vector for the ML model from processed data"""
        if df is None:
            df = self.latest_data
            
        if df is None or len(df) < lookback:
            logger.warning("Not enough data to create feature vector")
            return None
            
        # Get the most recent data
        recent_data = df.iloc[-lookback:].copy()
        
        # Normalize price data to recent range
        min_price = recent_data['low'].min()
        max_price = recent_data['high'].max()
        price_range = max_price - min_price
        
        if price_range == 0:
            logger.warning("Price range is zero, cannot normalize")
            return None
            
        # Create normalized OHLC
        for col in ['open', 'high', 'low', 'close']:
            recent_data[f'norm_{col}'] = (recent_data[col] - min_price) / price_range
            
        # Normalize volume
        vol_max = recent_data['tick_volume'].max()
        if vol_max > 0:
            recent_data['norm_volume'] = recent_data['tick_volume'] / vol_max
            
        # Extract features for the model
        features = []
        
        # Price action features
        for i in range(lookback):
            if i < len(recent_data):
                row = recent_data.iloc[i]
                features.extend([
                    row['norm_open'],
                    row['norm_high'],
                    row['norm_low'],
                    row['norm_close'],
                    row['norm_volume'],
                    row['body_pct'],
                    row['breakout_up'],
                    row['breakout_down'],
                    row['higher_high'],
                    row['lower_low']
                ])
            else:
                # Padding if not enough data
                features.extend([0] * 10)
                
        return np.array(features)
        
    def identify_turning_points(self, df=None, window=10):
        """Identify potential turning points in the price series"""
        if df is None:
            df = self.latest_data
            
        if df is None or len(df) < window * 2:
            logger.warning("Not enough data to identify turning points")
            return []
            
        turning_points = []
        
        # Find local maxima (resistance points)
        for i in range(window, len(df) - window):
            # Check if this point is a local maximum
            if all(df.iloc[i]['high'] >= df.iloc[i-j]['high'] for j in range(1, window+1)) and \
               all(df.iloc[i]['high'] >= df.iloc[i+j]['high'] for j in range(1, window+1)):
                turning_points.append({
                    'type': 'resistance',
                    'price': df.iloc[i]['high'],
                    'time': df.iloc[i]['time'],
                    'strength': sum(df.iloc[i]['high'] - df.iloc[i-j]['high'] for j in range(1, window+1)) +
                                sum(df.iloc[i]['high'] - df.iloc[i+j]['high'] for j in range(1, window+1))
                })
                
        # Find local minima (support points)
        for i in range(window, len(df) - window):
            # Check if this point is a local minimum
            if all(df.iloc[i]['low'] <= df.iloc[i-j]['low'] for j in range(1, window+1)) and \
               all(df.iloc[i]['low'] <= df.iloc[i+j]['low'] for j in range(1, window+1)):
                turning_points.append({
                    'type': 'support',
                    'price': df.iloc[i]['low'],
                    'time': df.iloc[i]['time'],
                    'strength': sum(df.iloc[i-j]['low'] - df.iloc[i]['low'] for j in range(1, window+1)) +
                                sum(df.iloc[i+j]['low'] - df.iloc[i]['low'] for j in range(1, window+1))
                })
                
        # Sort by strength
        turning_points.sort(key=lambda x: x['strength'], reverse=True)
        
        return turning_points
