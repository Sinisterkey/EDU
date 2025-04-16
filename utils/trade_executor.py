import logging
import time
from datetime import datetime
from models import TradeHistory, TradeStatus, OrderType
from app import db

logger = logging.getLogger(__name__)

class TradeExecutor:
    """Class to handle trade execution and management"""
    
    def __init__(self, mt5_connector, risk_per_trade=2.0, lot_size=0.02, risk_reward_ratio=3.0):
        self.mt5 = mt5_connector
        self.risk_per_trade = risk_per_trade  # Percentage of account to risk per trade
        self.lot_size = lot_size  # Default lot size
        self.risk_reward_ratio = risk_reward_ratio  # Risk/reward ratio (1:3 by default)
        self.open_trades = []
        
    def calculate_position_size(self, price, stop_loss):
        """
        Calculate position size based on risk parameters
        
        Parameters:
        -----------
        price : float
            Current price
        stop_loss : float
            Stop loss price
            
        Returns:
        --------
        float : Position size in lots
        """
        if self.mt5 is None or not self.mt5.initialized:
            logger.error("MT5 connector not initialized")
            return self.lot_size
            
        # Get account info
        account_info = self.mt5.get_account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return self.lot_size
            
        # Calculate risk amount
        account_balance = account_info['balance']
        risk_amount = account_balance * (self.risk_per_trade / 100)
        
        # Calculate pip value and risk in pips
        pip_value = 0.1  # For XAUUSD, 1 pip is 0.1 USD per 0.01 lot
        risk_pips = abs(price - stop_loss) * 10  # Convert price difference to pips
        
        # Calculate lot size
        if risk_pips > 0:
            calculated_lot_size = risk_amount / (risk_pips * pip_value)
            # Round to 2 decimal places (0.01 lot precision)
            calculated_lot_size = round(calculated_lot_size * 100) / 100
            
            # Cap at maximum 0.1 lots for safety
            return min(calculated_lot_size, 0.1)
        else:
            logger.warning("Risk in pips is zero or negative, using default lot size")
            return self.lot_size
            
    def calculate_sl_tp(self, order_type, entry_price):
        """
        Calculate stop loss and take profit levels
        
        Parameters:
        -----------
        order_type : str
            "BUY" or "SELL"
        entry_price : float
            Entry price
            
        Returns:
        --------
        tuple : (stop_loss, take_profit)
        """
        # For XAUUSD, use $20 as default SL distance
        sl_distance = 20.0
        tp_distance = sl_distance * self.risk_reward_ratio
        
        if order_type.upper() == "BUY":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SELL
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
            
        return round(stop_loss, 2), round(take_profit, 2)
        
    def execute_trade(self, trade_decision):
        """
        Execute a trade based on the decision
        
        Parameters:
        -----------
        trade_decision : dict
            Trading decision including action, price, etc.
            
        Returns:
        --------
        dict : Execution result
        """
        if self.mt5 is None or not self.mt5.initialized:
            logger.error("MT5 connector not initialized")
            return {"success": False, "error": "MT5 not initialized"}
            
        action = trade_decision.get("action", "HOLD")
        confidence = trade_decision.get("confidence", 0)
        
        # Only execute if action is BUY or SELL and confidence is above threshold
        if action == "HOLD" or confidence < 0.6:
            return {"success": False, "error": "Action is HOLD or confidence too low"}
            
        # Get current price
        current_price = trade_decision.get("price")
        if current_price is None:
            logger.error("No price provided in trade decision")
            return {"success": False, "error": "No price provided"}
            
        # Calculate stop loss and take profit
        stop_loss, take_profit = self.calculate_sl_tp(action, current_price)
        
        # Calculate lot size
        lot_size = self.calculate_position_size(current_price, stop_loss)
        
        # Create a trade record in database
        trade = TradeHistory(
            symbol="XAUUSD",
            order_type=OrderType.BUY if action == "BUY" else OrderType.SELL,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            status=TradeStatus.PENDING,
            notes=trade_decision.get("reason", "")
        )
        
        try:
            db.session.add(trade)
            db.session.commit()
            logger.info(f"Created trade record ID: {trade.id}")
        except Exception as e:
            logger.error(f"Error creating trade record: {e}")
            db.session.rollback()
            return {"success": False, "error": f"Database error: {str(e)}"}
            
        # Execute the trade on MT5
        try:
            trade_result = self.mt5.execute_trade(
                order_type=action,
                lot_size=lot_size,
                sl=stop_loss,
                tp=take_profit
            )
            
            if trade_result is None:
                logger.error("Trade execution failed")
                trade.status = TradeStatus.FAILED
                trade.notes += " | Execution failed"
                db.session.commit()
                return {"success": False, "error": "Trade execution failed"}
                
            # Update trade record with MT5 info
            trade.mt5_ticket = trade_result.get("order_id")
            trade.status = TradeStatus.OPEN
            trade.notes += f" | MT5 Order: {trade_result.get('order_id')}"
            db.session.commit()
            
            # Add to open trades list
            self.open_trades.append({
                "id": trade.id,
                "mt5_ticket": trade.mt5_ticket,
                "order_type": action,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "lot_size": lot_size,
                "entry_time": datetime.now()
            })
            
            logger.info(f"Trade executed: {action} XAUUSD {lot_size} lots @ {current_price}")
            
            return {
                "success": True,
                "trade_id": trade.id,
                "mt5_ticket": trade.mt5_ticket,
                "action": action,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "lot_size": lot_size
            }
            
        except Exception as e:
            logger.error(f"Error during trade execution: {e}")
            trade.status = TradeStatus.FAILED
            trade.notes += f" | Error: {str(e)}"
            db.session.commit()
            return {"success": False, "error": f"Execution error: {str(e)}"}
            
    def close_trade(self, trade_id):
        """
        Close a specific trade
        
        Parameters:
        -----------
        trade_id : int
            ID of the trade to close
            
        Returns:
        --------
        bool : Success status
        """
        if self.mt5 is None or not self.mt5.initialized:
            logger.error("MT5 connector not initialized")
            return False
            
        # Find the trade in the database
        trade = TradeHistory.query.get(trade_id)
        if trade is None:
            logger.error(f"Trade with ID {trade_id} not found")
            return False
            
        # Only close open trades
        if trade.status != TradeStatus.OPEN:
            logger.warning(f"Trade {trade_id} is not open (status: {trade.status})")
            return False
            
        # Close the trade on MT5
        if trade.mt5_ticket:
            try:
                close_result = self.mt5.close_trade(trade.mt5_ticket)
                
                if not close_result:
                    logger.error(f"Failed to close trade {trade_id} on MT5")
                    return False
                    
                # Update trade in database
                current_price = self.mt5.get_historical_data().iloc[-1]['close']
                trade.exit_price = current_price
                trade.exit_time = datetime.now()
                trade.status = TradeStatus.CLOSED
                
                # Calculate profit/loss
                if trade.order_type == OrderType.BUY:
                    trade.profit_loss = (current_price - trade.entry_price) * trade.lot_size * 1000
                else:  # SELL
                    trade.profit_loss = (trade.entry_price - current_price) * trade.lot_size * 1000
                    
                trade.is_success = trade.profit_loss > 0
                trade.notes += f" | Closed manually at {current_price}"
                
                db.session.commit()
                logger.info(f"Trade {trade_id} closed successfully")
                
                # Remove from open trades list
                self.open_trades = [t for t in self.open_trades if t["id"] != trade_id]
                
                return True
                
            except Exception as e:
                logger.error(f"Error closing trade {trade_id}: {e}")
                return False
        else:
            logger.error(f"Trade {trade_id} has no MT5 ticket")
            return False
            
    def update_open_trades(self):
        """
        Update the status of all open trades
        
        Returns:
        --------
        list : Updated open trades
        """
        if self.mt5 is None or not self.mt5.initialized:
            logger.error("MT5 connector not initialized")
            return self.open_trades
            
        # Get latest price
        try:
            latest_data = self.mt5.get_historical_data(bars=1)
            if latest_data is None or len(latest_data) == 0:
                logger.error("Failed to get latest price data")
                return self.open_trades
                
            current_price = latest_data.iloc[-1]['close']
            
            # Query all open trades from database
            open_trades = TradeHistory.query.filter_by(status=TradeStatus.OPEN).all()
            
            # Update each trade
            for trade in open_trades:
                # Check if trade reached TP or SL
                if trade.order_type == OrderType.BUY:
                    if current_price >= trade.take_profit:
                        self.close_trade(trade.id)
                        logger.info(f"Trade {trade.id} reached take profit")
                    elif current_price <= trade.stop_loss:
                        self.close_trade(trade.id)
                        logger.info(f"Trade {trade.id} reached stop loss")
                else:  # SELL
                    if current_price <= trade.take_profit:
                        self.close_trade(trade.id)
                        logger.info(f"Trade {trade.id} reached take profit")
                    elif current_price >= trade.stop_loss:
                        self.close_trade(trade.id)
                        logger.info(f"Trade {trade.id} reached stop loss")
                        
            # Refresh open trades list
            self.open_trades = []
            for trade in open_trades:
                if trade.status == TradeStatus.OPEN:
                    # Calculate current P/L
                    if trade.order_type == OrderType.BUY:
                        unrealized_pl = (current_price - trade.entry_price) * trade.lot_size * 1000
                    else:  # SELL
                        unrealized_pl = (trade.entry_price - current_price) * trade.lot_size * 1000
                        
                    self.open_trades.append({
                        "id": trade.id,
                        "mt5_ticket": trade.mt5_ticket,
                        "order_type": trade.order_type.value,
                        "entry_price": trade.entry_price,
                        "current_price": current_price,
                        "stop_loss": trade.stop_loss,
                        "take_profit": trade.take_profit,
                        "lot_size": trade.lot_size,
                        "entry_time": trade.entry_time,
                        "unrealized_pl": unrealized_pl
                    })
            
            return self.open_trades
            
        except Exception as e:
            logger.error(f"Error updating open trades: {e}")
            return self.open_trades
