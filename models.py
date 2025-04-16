from datetime import datetime
from app import db
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Enum
import enum

class TradeStatus(enum.Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class OrderType(enum.Enum):
    BUY = "buy"
    SELL = "sell"

class TradeHistory(db.Model):
    """Model for storing trade history data"""
    __tablename__ = 'trade_history'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, default="XAUUSD")
    order_type = db.Column(db.Enum(OrderType), nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    lot_size = db.Column(db.Float, nullable=False)
    stop_loss = db.Column(db.Float, nullable=False)
    take_profit = db.Column(db.Float, nullable=False)
    entry_time = db.Column(db.DateTime, default=datetime.utcnow)
    exit_time = db.Column(db.DateTime)
    profit_loss = db.Column(db.Float)
    status = db.Column(db.Enum(TradeStatus), default=TradeStatus.PENDING)
    is_success = db.Column(db.Boolean)
    notes = db.Column(db.Text)
    mt5_ticket = db.Column(db.Integer)
    
    def __repr__(self):
        return f"<Trade {self.id}: {self.order_type.value} {self.symbol} @ {self.entry_price}>"

class SupportResistanceZone(db.Model):
    """Model for storing support and resistance zones"""
    __tablename__ = 'support_resistance_zones'
    
    id = db.Column(db.Integer, primary_key=True)
    zone_type = db.Column(db.String(20), nullable=False)  # "support" or "resistance"
    price_level = db.Column(db.Float, nullable=False)
    strength = db.Column(db.Float, nullable=False)  # 1-10 scale based on touches
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f"<{self.zone_type.capitalize()} Zone @ {self.price_level} Strength: {self.strength}>"

class AgentSettings(db.Model):
    """Model for storing agent settings"""
    __tablename__ = 'agent_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    risk_per_trade = db.Column(db.Float, nullable=False, default=2.0)  # percentage of account
    lot_size = db.Column(db.Float, nullable=False, default=0.02)
    risk_reward_ratio = db.Column(db.Float, nullable=False, default=3.0)
    max_daily_drawdown = db.Column(db.Float, nullable=False, default=5.0)  # percentage
    max_consecutive_losses = db.Column(db.Integer, nullable=False, default=3)
    is_active = db.Column(db.Boolean, default=True)
    agent_status = db.Column(db.String(20), default="Idle")  # Active, Idle, Retraining, Error
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Agent Settings: {self.risk_per_trade}% risk, {self.lot_size} lots>"

class PerformanceMetrics(db.Model):
    """Model for storing daily performance metrics"""
    __tablename__ = 'performance_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    win_count = db.Column(db.Integer, default=0)
    loss_count = db.Column(db.Integer, default=0)
    win_rate = db.Column(db.Float, default=0.0)
    profit_loss = db.Column(db.Float, default=0.0)
    avg_profit = db.Column(db.Float, default=0.0)
    avg_loss = db.Column(db.Float, default=0.0)
    max_drawdown = db.Column(db.Float, default=0.0)
    sharpe_ratio = db.Column(db.Float, default=0.0)
    
    def __repr__(self):
        return f"<Metrics for {self.date}: Win Rate: {self.win_rate}%, P/L: {self.profit_loss}>"

class PriceData(db.Model):
    """Model for storing historical price data"""
    __tablename__ = 'price_data'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, default="XAUUSD")
    timestamp = db.Column(db.DateTime, nullable=False)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=False)
    
    def __repr__(self):
        return f"<Price {self.symbol} @ {self.timestamp}: O:{self.open} H:{self.high} L:{self.low} C:{self.close}>"
