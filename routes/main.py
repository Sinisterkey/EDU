import logging
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from models import TradeHistory, AgentSettings, PerformanceMetrics, SupportResistanceZone
from app import db
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Render the homepage"""
    return render_template('index.html')

@main_bp.route('/dashboard')
def dashboard():
    """Render the trading dashboard"""
    # Get agent settings
    agent_settings = AgentSettings.query.first()
    if agent_settings is None:
        # Create default settings if none exist
        agent_settings = AgentSettings(
            risk_per_trade=2.0,
            lot_size=0.02,
            risk_reward_ratio=3.0,
            max_daily_drawdown=5.0,
            max_consecutive_losses=3,
            is_active=False,
            agent_status="Idle"
        )
        db.session.add(agent_settings)
        db.session.commit()
        
    # Get recent trade history
    recent_trades = TradeHistory.query.order_by(TradeHistory.entry_time.desc()).limit(10).all()
    
    # Get today's performance
    today = datetime.now().date()
    today_metrics = PerformanceMetrics.query.filter_by(date=today).first()
    
    if today_metrics is None:
        # Create empty metrics for today
        today_metrics = PerformanceMetrics(date=today)
        db.session.add(today_metrics)
        db.session.commit()
        
    # Get active support/resistance zones
    active_zones = SupportResistanceZone.query.filter_by(is_active=True).order_by(SupportResistanceZone.strength.desc()).all()
    
    return render_template(
        'dashboard.html',
        agent_settings=agent_settings,
        recent_trades=recent_trades,
        today_metrics=today_metrics,
        active_zones=active_zones
    )

@main_bp.route('/settings', methods=['POST'])
def update_settings():
    """Update agent settings"""
    try:
        # Get form data
        risk_per_trade = float(request.form.get('risk_per_trade', 2.0))
        lot_size = float(request.form.get('lot_size', 0.02))
        risk_reward_ratio = float(request.form.get('risk_reward_ratio', 3.0))
        max_daily_drawdown = float(request.form.get('max_daily_drawdown', 5.0))
        max_consecutive_losses = int(request.form.get('max_consecutive_losses', 3))
        is_active = 'is_active' in request.form
        
        # Validate inputs
        if risk_per_trade < 0.1 or risk_per_trade > 10:
            flash('Risk per trade must be between 0.1% and 10%', 'danger')
            return redirect(url_for('main.dashboard'))
            
        if lot_size < 0.01 or lot_size > 0.5:
            flash('Lot size must be between 0.01 and 0.5', 'danger')
            return redirect(url_for('main.dashboard'))
            
        if risk_reward_ratio < 1 or risk_reward_ratio > 5:
            flash('Risk/reward ratio must be between 1 and 5', 'danger')
            return redirect(url_for('main.dashboard'))
            
        # Update settings
        agent_settings = AgentSettings.query.first()
        if agent_settings is None:
            agent_settings = AgentSettings()
            db.session.add(agent_settings)
            
        agent_settings.risk_per_trade = risk_per_trade
        agent_settings.lot_size = lot_size
        agent_settings.risk_reward_ratio = risk_reward_ratio
        agent_settings.max_daily_drawdown = max_daily_drawdown
        agent_settings.max_consecutive_losses = max_consecutive_losses
        agent_settings.is_active = is_active
        
        if is_active and agent_settings.agent_status == "Idle":
            agent_settings.agent_status = "Active"
        elif not is_active and agent_settings.agent_status == "Active":
            agent_settings.agent_status = "Idle"
            
        db.session.commit()
        
        flash('Settings updated successfully', 'success')
        logger.info(f"Agent settings updated: risk={risk_per_trade}%, lot_size={lot_size}, active={is_active}")
        
    except Exception as e:
        flash(f'Error updating settings: {str(e)}', 'danger')
        logger.error(f"Error updating settings: {e}")
        
    return redirect(url_for('main.dashboard'))

@main_bp.route('/trades')
def trade_history():
    """Get trade history"""
    # Get query parameters
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    # Query trades with pagination
    trades = TradeHistory.query.order_by(TradeHistory.entry_time.desc()).limit(limit).offset(offset).all()
    
    # Count total trades
    total_trades = TradeHistory.query.count()
    
    # Get trade statistics
    total_profit = db.session.query(db.func.sum(TradeHistory.profit_loss)).scalar() or 0
    win_count = TradeHistory.query.filter_by(is_success=True).count()
    loss_count = TradeHistory.query.filter_by(is_success=False).count()
    
    if win_count + loss_count > 0:
        win_rate = (win_count / (win_count + loss_count)) * 100
    else:
        win_rate = 0
        
    # Format for template
    return render_template(
        'trades.html',
        trades=trades,
        total_trades=total_trades,
        total_profit=total_profit,
        win_count=win_count,
        loss_count=loss_count,
        win_rate=win_rate,
        limit=limit,
        offset=offset
    )

@main_bp.route('/close_trade/<int:trade_id>', methods=['POST'])
def close_trade(trade_id):
    """Close a specific trade"""
    # This would normally call the trade executor, for now just update the DB
    try:
        trade = TradeHistory.query.get_or_404(trade_id)
        
        if trade.status != 'open':
            flash('Trade is not open and cannot be closed', 'warning')
        else:
            trade.status = 'closed'
            trade.exit_time = datetime.now()
            trade.notes += " | Closed manually via dashboard"
            db.session.commit()
            flash(f'Trade {trade_id} closed successfully', 'success')
            logger.info(f"Trade {trade_id} closed manually")
            
    except Exception as e:
        flash(f'Error closing trade: {str(e)}', 'danger')
        logger.error(f"Error closing trade {trade_id}: {e}")
        
    return redirect(url_for('main.dashboard'))

@main_bp.route('/emergency_stop', methods=['POST'])
def emergency_stop():
    """Emergency stop - close all trades and deactivate agent"""
    try:
        # Update agent settings
        agent_settings = AgentSettings.query.first()
        if agent_settings:
            agent_settings.is_active = False
            agent_settings.agent_status = "Idle"
            
        # Mark all open trades as requiring closure
        open_trades = TradeHistory.query.filter_by(status='open').all()
        for trade in open_trades:
            trade.notes += " | Marked for closure by emergency stop"
            
        db.session.commit()
        
        flash('Emergency stop activated. All trades will be closed and agent stopped.', 'warning')
        logger.warning("Emergency stop activated by user")
        
    except Exception as e:
        flash(f'Error during emergency stop: {str(e)}', 'danger')
        logger.error(f"Error during emergency stop: {e}")
        
    return redirect(url_for('main.dashboard'))
