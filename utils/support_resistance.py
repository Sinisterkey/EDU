import logging
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class SupportResistanceDetector:
    """Class to detect support and resistance zones using DBSCAN clustering"""
    
    def __init__(self, eps=0.01, min_samples=2):
        """
        Initialize the detector with DBSCAN parameters
        
        Parameters:
        -----------
        eps : float
            The maximum distance between two samples for one to be considered 
            as in the neighborhood of the other.
        min_samples : int
            The number of samples in a neighborhood for a point to be considered as a core point.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.clusters = None
        self.price_levels = None
        self.zones = []
        
    def detect_zones(self, df, n_candles=50):
        """
        Detect support and resistance zones using DBSCAN clustering
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data with 'high' and 'low' columns
        n_candles : int
            Number of recent candles to consider
            
        Returns:
        --------
        list : List of dictionaries with zone information
        """
        if df is None or len(df) == 0:
            logger.warning("Empty dataframe provided")
            return []
            
        # Use the most recent n_candles
        if len(df) > n_candles:
            df = df.iloc[-n_candles:]
            
        # Extract turning points from the data
        turning_points = []
        
        # Find potential resistance points (local highs)
        for i in range(1, len(df)-1):
            if df.iloc[i]['high'] > df.iloc[i-1]['high'] and df.iloc[i]['high'] > df.iloc[i+1]['high']:
                turning_points.append({
                    'price': df.iloc[i]['high'],
                    'type': 'resistance',
                    'time': df.iloc[i]['time']
                })
                
        # Find potential support points (local lows)
        for i in range(1, len(df)-1):
            if df.iloc[i]['low'] < df.iloc[i-1]['low'] and df.iloc[i]['low'] < df.iloc[i+1]['low']:
                turning_points.append({
                    'price': df.iloc[i]['low'],
                    'type': 'support',
                    'time': df.iloc[i]['time']
                })
                
        if len(turning_points) < self.min_samples:
            logger.warning(f"Not enough turning points ({len(turning_points)}) for clustering")
            return []
            
        # Extract prices for clustering
        prices = np.array([tp['price'] for tp in turning_points]).reshape(-1, 1)
        
        # Normalize the data
        scaler = StandardScaler()
        prices_scaled = scaler.fit_transform(prices)
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(prices_scaled)
        
        # Store the clustering results
        self.clusters = cluster_labels
        
        # Group turning points by cluster
        cluster_groups = {}
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Skip noise points
                if label not in cluster_groups:
                    cluster_groups[label] = []
                cluster_groups[label].append(turning_points[i])
                
        # Calculate zone information for each cluster
        zones = []
        for label, points in cluster_groups.items():
            # Calculate the average price as the zone level
            prices = [p['price'] for p in points]
            avg_price = sum(prices) / len(prices)
            
            # Determine zone type (majority vote)
            support_count = sum(1 for p in points if p['type'] == 'support')
            resistance_count = sum(1 for p in points if p['type'] == 'resistance')
            zone_type = 'support' if support_count > resistance_count else 'resistance'
            
            # Calculate strength based on number of points and recency
            strength = len(points) * 0.6
            
            # Add bonus for recent points
            recent_points = sum(1 for p in points if (pd.Timestamp.now() - p['time']).total_seconds() < 86400)
            strength += recent_points * 0.4
            
            # Create zone entry
            zone = {
                'price_level': avg_price,
                'zone_type': zone_type,
                'strength': strength,
                'num_points': len(points),
                'points': points
            }
            
            zones.append(zone)
            
        # Sort zones by strength
        zones.sort(key=lambda x: x['strength'], reverse=True)
        
        self.zones = zones
        return zones
        
    def find_nearest_zones(self, price, n_zones=3):
        """
        Find the nearest zones to a given price
        
        Parameters:
        -----------
        price : float
            The price to find nearest zones to
        n_zones : int
            Number of zones to return
            
        Returns:
        --------
        list : List of nearest zones
        """
        if not self.zones:
            logger.warning("No zones detected yet")
            return []
            
        # Calculate distance from price to each zone
        for zone in self.zones:
            zone['distance'] = abs(zone['price_level'] - price)
            
        # Sort by distance and return top n_zones
        nearest = sorted(self.zones, key=lambda x: x['distance'])[:n_zones]
        
        return nearest
        
    def is_near_zone(self, price, threshold_pct=0.001):
        """
        Check if price is near any zone
        
        Parameters:
        -----------
        price : float
            The price to check
        threshold_pct : float
            Threshold as percentage of price to consider "near"
            
        Returns:
        --------
        tuple : (bool, zone) if near a zone, else (False, None)
        """
        if not self.zones:
            return False, None
            
        threshold = price * threshold_pct
        
        for zone in self.zones:
            if abs(zone['price_level'] - price) <= threshold:
                return True, zone
                
        return False, None
        
    def check_breakout(self, current_price, previous_price):
        """
        Check if there's a breakout of any zone
        
        Parameters:
        -----------
        current_price : float
            Current price
        previous_price : float
            Previous price
            
        Returns:
        --------
        tuple : (bool, zone, direction) if breakout detected, else (False, None, None)
        """
        if not self.zones:
            return False, None, None
            
        for zone in self.zones:
            price_level = zone['price_level']
            zone_type = zone['zone_type']
            
            # Check for resistance breakout
            if zone_type == 'resistance' and previous_price < price_level and current_price > price_level:
                return True, zone, 'up'
                
            # Check for support breakout
            if zone_type == 'support' and previous_price > price_level and current_price < price_level:
                return True, zone, 'down'
                
        return False, None, None
