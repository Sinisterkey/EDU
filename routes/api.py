import logging
import json
import pandas as pd
import numpy as np
from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from models import TradeHistory, AgentSettings, PerformanceMetrics, SupportResistanceZone, PriceData, TradeStatus
from app import db
from utils.mt5_connector import MT5Connector
from utils.data_processor import DataProcessor
from utils.support_resistance import SupportResistanceDetector
from utils.rl_agent import RLAgent
from utils.trade_executor import TradeExecutor

logger = logging.getLogger(__name__)

def generate_mock_price_data(n_candles=50):
    """
    Generate mock price data for development/testing purposes
    
    Parameters:
    -----------
    n_candles : int
        Number of candles to generate
        
    Returns:
    --------
    pandas.DataFrame : DataFrame with OHLCV data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Starting price (XAUUSD is typically around 1900)
    base_price = 1900.0
    
    # Generate timestamps, most recent first
    end_time = datetime.now()
    times = [end_time - timedelta(minutes=30*i) for i in range(n_candles)]
    times.reverse()
    
    # Generate prices with random walk and volatility appropriate for gold
    volatility = 1.5  # Gold can move $1-3 in 30 minutes
    
    # Initialize price arrays
    close_prices = []
    high_prices = []
    low_prices = []
    open_prices = []
    volumes = []
    
    # Generate the first open price
    current_price = base_price
    open_prices.append(current_price)
    
    # Generate OHLCV data
    for i in range(n_candles):
        # Random price change
        price_change = np.random.normal(0, volatility)
        
        # Determine open price (first one already set)
        if i > 0:
            open_prices.append(close_prices[i-1])
        
        # Determine close price
        close_price = open_prices[i] + price_change
        close_prices.append(close_price)
        
        # Determine high and low with random intra-candle volatility
        intra_high = np.random.uniform(0.2, 1.0) * volatility
        intra_low = np.random.uniform(0.2, 1.0) * volatility
        
        high_price = max(open_prices[i], close_price) + intra_high
        low_price = min(open_prices[i], close_price) - intra_low
        
        high_prices.append(high_price)
        low_prices.append(low_price)
        
        # Generate random volume
        volume = np.random.normal(1000, 200)  # Average 1000 contracts
        volumes.append(max(100, volume))  # Ensure positive volume
    
    # Create DataFrame
    data = {
        'time': times,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }
    
    return pd.DataFrame(data)

# Initialize components
mt5 = MT5Connector()
data_processor = DataProcessor()
sr_detector = SupportResistanceDetector()
rl_agent = RLAgent(data_processor, sr_detector)
trade_executor = TradeExecutor(mt5)

# Create blueprint
api_bp = Blueprint('api', __name__)

@api_bp.route('/status', methods=['GET'])
def get_status():
    """Get the current status of the trading agent"""
    try:
        # Get agent settings
        agent_settings = AgentSettings.query.first()
        
        # Get account info if MT5 is connected
        account_info = None
        if mt5.initialized:
            account_info = mt5.get_account_info()
            
        # Get open trades
        open_trades = TradeHistory.query.filter_by(status=TradeStatus.OPEN).all()
        open_trades_count = len(open_trades)
        
        # Get today's performance
        today = datetime.now().date()
        today_metrics = PerformanceMetrics.query.filter_by(date=today).first()
        
        # Get latest trade
        latest_trade = TradeHistory.query.order_by(TradeHistory.entry_time.desc()).first()
        
        # Compile status info
        status = {
            "agent": {
                "status": agent_settings.agent_status if agent_settings else "Unknown",
                "is_active": agent_settings.is_active if agent_settings else False,
                "risk_per_trade": agent_settings.risk_per_trade if agent_settings else 0,
                "lot_size": agent_settings.lot_size if agent_settings else 0,
                "training_in_progress": getattr(rl_agent, 'training_in_progress', False)
            },
            "account": account_info,
            "trades": {
                "open_count": open_trades_count,
                "total_count": TradeHistory.query.count(),
                "latest_trade": {
                    "id": latest_trade.id if latest_trade else None,
                    "type": latest_trade.order_type.value if latest_trade else None,
                    "entry_time": latest_trade.entry_time.isoformat() if latest_trade else None,
                    "status": latest_trade.status.value if latest_trade else None
                } if latest_trade else None
            },
            "performance": {
                "today": {
                    "win_rate": today_metrics.win_rate if today_metrics else 0,
                    "profit_loss": today_metrics.profit_loss if today_metrics else 0,
                    "trade_count": today_metrics.win_count + today_metrics.loss_count if today_metrics else 0
                } if today_metrics else None
            },
            "mt5_connected": mt5.initialized,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/connect', methods=['POST'])
def connect_mt5():
    """Connect to MetaTrader 5"""
    try:
        # Use the provided credentials if available, otherwise use the ones from the request
        path = request.json.get('path')
        login = request.json.get('login', '91873732')  # Default to the user's provided credentials
        password = request.json.get('password', 'Ds!j4wUh')  # Default to the user's provided credentials
        server = request.json.get('server', 'MetaQuotes-Demo')  # Default server
        
        logger.info(f"Connecting to MT5 with login: {login}")
        result = mt5.connect(path, login, password, server)
        
        if result:
            return jsonify({"success": True, "message": f"Connected to MT5 successfully (Account: {login})"})
        else:
            return jsonify({"success": False, "message": "Failed to connect to MT5"}), 400
            
    except Exception as e:
        logger.error(f"Error connecting to MT5: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route('/disconnect', methods=['POST'])
def disconnect_mt5():
    """Disconnect from MetaTrader 5"""
    try:
        result = mt5.disconnect()
        
        if result:
            return jsonify({"success": True, "message": "Disconnected from MT5 successfully"})
        else:
            return jsonify({"success": False, "message": "MT5 was not connected"}), 400
            
    except Exception as e:
        logger.error(f"Error disconnecting from MT5: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route('/historical_data', methods=['GET'])
def get_historical_data():
    """Get historical price data"""
    try:
        bars = request.args.get('bars', 50, type=int)
        
        # Try to get from MT5 first
        if mt5.initialized:
            data = mt5.get_historical_data(bars=bars)
            
            if data is not None:
                # Convert to dict for JSON response
                data_dict = data.to_dict(orient='records')
                return jsonify({"success": True, "data": data_dict})
                
        # Fallback to database
        price_data = PriceData.query.order_by(PriceData.timestamp.desc()).limit(bars).all()
        
        # If no database data either, generate mock data for development
        if not price_data:
            mock_data = generate_mock_price_data(bars)
            data_dict = mock_data.to_dict(orient='records')
            return jsonify({"success": True, "data": data_dict})
        
        # Use database data
        data_dict = [{
            "time": p.timestamp.isoformat(),
            "open": p.open,
            "high": p.high,
            "low": p.low,
            "close": p.close,
            "volume": p.volume
        } for p in price_data]
        
        return jsonify({"success": True, "data": data_dict})
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route('/detect_zones', methods=['POST'])
def detect_zones():
    """Detect support and resistance zones"""
    try:
        data = request.json.get('data')
        n_candles = request.json.get('n_candles', 50)
        
        if data is None:
            # Get data from MT5 if not provided
            if mt5.initialized:
                df = mt5.get_historical_data(bars=n_candles)
            else:
                # When not connected, generate mock data for development purposes
                df = generate_mock_price_data(n_candles)
        else:
            # Convert provided data to DataFrame
            df = pd.DataFrame(data)
            
        # Process data and detect zones
        processed_data = data_processor.preprocess_data(df)
        zones = sr_detector.detect_zones(processed_data, n_candles)
        
        # Store zones in database
        SupportResistanceZone.query.filter_by(is_active=True).update({"is_active": False})
        db.session.commit()
        
        for zone in zones:
            # Convert numpy float values to Python floats to avoid database errors
            price_level = float(zone['price_level']) if 'price_level' in zone else 0.0
            strength = float(zone['strength']) if 'strength' in zone else 1.0
            
            db_zone = SupportResistanceZone(
                zone_type=zone['zone_type'],
                price_level=price_level,
                strength=strength,
                is_active=True
            )
            db.session.add(db_zone)
            
        db.session.commit()
        
        return jsonify({"success": True, "zones": zones})
        
    except Exception as e:
        logger.error(f"Error detecting zones: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route('/train_agent', methods=['POST'])
def train_agent():
    """Train the reinforcement learning agent"""
    try:
        # Get training parameters
        timesteps = request.json.get('timesteps', 10000)
        
        # Get historical data for training
        if mt5.initialized:
            # Use more data for training
            df = mt5.get_historical_data(bars=500)
            
            if df is None or len(df) < 100:
                return jsonify({
                    "success": False, 
                    "message": "Not enough historical data for training"
                }), 400
                
            # Start training in a non-blocking way (in a real app, this would be in a separate thread)
            # For this demo, we'll do it synchronously
            result = rl_agent.train(df, total_timesteps=timesteps)
            
            if result:
                # Update agent status
                agent_settings = AgentSettings.query.first()
                if agent_settings:
                    agent_settings.agent_status = "Trained"
                    db.session.commit()
                
                # Return training metrics
                metrics = rl_agent.get_performance_metrics()
                
                return jsonify({
                    "success": True, 
                    "message": f"Agent trained successfully with {timesteps} timesteps",
                    "metrics": metrics
                })
            else:
                return jsonify({
                    "success": False, 
                    "message": "Training failed"
                }), 500
        else:
            return jsonify({
                "success": False, 
                "message": "MT5 not connected, cannot get training data"
            }), 400
            
    except Exception as e:
        logger.error(f"Error training agent: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route('/training_history', methods=['GET'])
def get_training_history():
    """Get the training history for visualization"""
    try:
        history = rl_agent.get_training_history()
        metrics = rl_agent.get_performance_metrics()
        
        return jsonify({
            "success": True,
            "history": history,
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Error getting training history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route('/trade_decision', methods=['GET'])
def get_trade_decision():
    """Get the current trade decision from the agent"""
    try:
        try:
            # Get latest data
            if mt5.initialized:
                df = mt5.get_historical_data(bars=50)
            else:
                # When not connected, use mock data for development
                df = generate_mock_price_data(50)
                
            # Process data
            processed_data = data_processor.preprocess_data(df)
            
            # Detect zones
            zones = sr_detector.detect_zones(processed_data)
            
            # Get trading decision
            decision = rl_agent.decide_trade(processed_data, zones)
            
            return jsonify({"success": True, "decision": decision})
        except Exception as e:
            logger.error(f"Error in trade decision: {str(e)}")
            return jsonify({
                "success": False, 
                "message": f"Error getting trade decision: {str(e)}"
            }), 400
            
    except Exception as e:
        logger.error(f"Error getting trade decision: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route('/execute_trade', methods=['POST'])
def execute_trade():
    """Execute a trade based on the current decision"""
    try:
        # Check if agent is active
        agent_settings = AgentSettings.query.first()
        if not agent_settings or not agent_settings.is_active:
            return jsonify({
                "success": False, 
                "message": "Agent is not active, enable it in settings first"
            }), 400
            
        # Check if MT5 is connected
        if not mt5.initialized:
            return jsonify({
                "success": False, 
                "message": "MT5 not connected"
            }), 400
            
        # Get latest decision
        df = mt5.get_historical_data(bars=50)
        processed_data = data_processor.preprocess_data(df)
        zones = sr_detector.detect_zones(processed_data)
        decision = rl_agent.decide_trade(processed_data, zones)
        
        # Check if decision confidence is high enough
        if decision["action"] == "HOLD" or float(decision["confidence"]) < 0.6:
            return jsonify({
                "success": True, 
                "message": f"No trade executed: {decision['action']} with confidence {decision['confidence']}"
            })
            
        # Execute the trade
        trade_result = trade_executor.execute_trade(decision)
        
        if trade_result["success"]:
            return jsonify({
                "success": True, 
                "trade": trade_result,
                "message": f"Trade executed: {decision['action']} @ {decision['price']}"
            })
        else:
            return jsonify({
                "success": False, 
                "error": trade_result["error"]
            }), 400
            
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route('/open_trades', methods=['GET'])
def get_open_trades():
    """Get all open trades"""
    try:
        # Update open trades from MT5 if connected
        if mt5.initialized:
            trade_executor.update_open_trades()
            
        # Get open trades from database
        open_trades = TradeHistory.query.filter_by(status=TradeStatus.OPEN).all()
        
        trades_list = [{
            "id": trade.id,
            "order_type": trade.order_type.value,
            "entry_price": trade.entry_price,
            "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit,
            "lot_size": trade.lot_size,
            "entry_time": trade.entry_time.isoformat(),
            "mt5_ticket": trade.mt5_ticket
        } for trade in open_trades]
        
        return jsonify({"success": True, "trades": trades_list})
        
    except Exception as e:
        logger.error(f"Error getting open trades: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route('/close_trade/<int:trade_id>', methods=['POST'])
def api_close_trade(trade_id):
    """Close a specific trade"""
    try:
        # Check if MT5 is connected
        if not mt5.initialized:
            return jsonify({
                "success": False, 
                "message": "MT5 not connected"
            }), 400
            
        # Close the trade
        result = trade_executor.close_trade(trade_id)
        
        if result:
            return jsonify({
                "success": True, 
                "message": f"Trade {trade_id} closed successfully"
            })
        else:
            return jsonify({
                "success": False, 
                "message": f"Failed to close trade {trade_id}"
            }), 400
            
    except Exception as e:
        logger.error(f"Error closing trade {trade_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route('/performance', methods=['GET'])
def get_performance():
    """Get performance metrics"""
    try:
        # Get time range
        days = request.args.get('days', 30, type=int)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Get metrics for the range
        metrics = PerformanceMetrics.query.filter(
            PerformanceMetrics.date >= start_date,
            PerformanceMetrics.date <= end_date
        ).order_by(PerformanceMetrics.date).all()
        
        # Calculate overall metrics
        total_profit = sum(m.profit_loss for m in metrics)
        total_trades = sum(m.win_count + m.loss_count for m in metrics)
        win_count = sum(m.win_count for m in metrics)
        
        if total_trades > 0:
            overall_win_rate = (win_count / total_trades) * 100
        else:
            overall_win_rate = 0
            
        # Format daily metrics
        daily_metrics = [{
            "date": m.date.isoformat(),
            "profit_loss": m.profit_loss,
            "win_rate": m.win_rate,
            "trade_count": m.win_count + m.loss_count,
            "avg_profit": m.avg_profit,
            "avg_loss": m.avg_loss,
            "max_drawdown": m.max_drawdown
        } for m in metrics]
        
        return jsonify({
            "success": True,
            "overall": {
                "total_profit": total_profit,
                "total_trades": total_trades,
                "win_rate": overall_win_rate,
                "period_days": days
            },
            "daily": daily_metrics
        })
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
