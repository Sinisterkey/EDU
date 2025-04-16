import logging
import numpy as np
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import Stable Baselines3, use a simplified model if not available
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.env_checker import check_env
    SB3_AVAILABLE = True
    logger.info("Using Stable Baselines3 for RL agent")
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("Stable Baselines3 not available, using simplified agent model")

class SimplifiedRLModel:
    """A simplified RL model for when Stable Baselines3 is not available"""
    
    def __init__(self):
        self.action_mapping = {
            0: "HOLD",
            1: "BUY",
            2: "SELL"
        }
        
    def predict(self, observation, deterministic=True):
        """
        Make a prediction based on the observation
        
        Parameters:
        -----------
        observation : numpy.ndarray
            The observation from the environment
        deterministic : bool
            Whether to use deterministic action selection
            
        Returns:
        --------
        tuple : (action, None)
        """
        # Simple logic based on observation patterns
        # This is a very basic heuristic for demonstration purposes
        
        # Assuming observation has price features
        if observation is None or len(observation) < 10:
            return 0, None  # HOLD
            
        # Extract recent price trend from observation (simplified)
        # In a real implementation, this would be based on the actual observation structure
        recent_trend = sum(observation[-10:])
        
        if recent_trend > 0:
            return 1, None  # BUY
        elif recent_trend < 0:
            return 2, None  # SELL
        else:
            return 0, None  # HOLD
            
    def save(self, path):
        """Mock save method"""
        logger.info(f"Saving simplified model to {path} (mock)")
        return True
        
    def learn(self, total_timesteps):
        """Mock learn method"""
        logger.info(f"Learning for {total_timesteps} steps (mock)")
        return self

class TradingEnv:
    """
    A trading environment that implements a gym-like interface
    for reinforcement learning with enhanced reward functions
    """
    
    def __init__(self, data_processor, support_resistance_detector):
        self.data_processor = data_processor
        self.sr_detector = support_resistance_detector
        self.current_step = 0
        self.data = None
        self.history = []
        self.current_position = None
        self.account_balance = 10000.0
        self.initial_balance = 10000.0
        self.max_balance = 10000.0
        self.min_balance = 10000.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        self.position_hold_time = 0
        
        # Action and observation spaces
        self.action_space = 3  # 0: HOLD, 1: BUY, 2: SELL
        self.observation_space_dim = 100  # 10 features per candle * 10 candles
        
        # Transaction costs
        self.transaction_cost = 0.0001  # 0.01% per trade
        
        # Reward shaping parameters
        self.reward_scale = 100.0  # Scale rewards to be more pronounced
        self.profit_reward_weight = 1.0
        self.risk_reward_weight = 0.3
        self.holding_penalty = -0.01  # Small penalty for holding positions too long
        self.entry_exit_zone_reward = 0.5  # Reward for entering/exiting at good zones
        self.overtrading_penalty = -0.2  # Penalty for excessive trading
        self.trend_alignment_reward = 0.3  # Reward for trading in direction of trend
        
    def reset(self, data=None):
        """Reset the environment with new data"""
        if data is not None:
            self.data = data
        
        self.current_step = 0
        self.history = []
        self.current_position = None
        self.account_balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.min_balance = self.initial_balance
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_losses = 0
        self.max_drawdown = 0.0
        self.position_hold_time = 0
        self.trades_count = 0
        
        return self._get_observation()
        
    def step(self, action):
        """
        Take a step in the environment
        
        Parameters:
        -----------
        action : int
            0: HOLD, 1: BUY, 2: SELL
            
        Returns:
        --------
        tuple : (observation, reward, done, info)
        """
        if self.data is None or self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
            
        # Get current price data
        current_price = self.data.iloc[self.current_step]['close']
        current_high = self.data.iloc[self.current_step]['high']
        current_low = self.data.iloc[self.current_step]['low']
        
        # Detect support and resistance zones for the current step
        current_candles = self.data.iloc[max(0, self.current_step-20):self.current_step+1]
        zones = self.sr_detector.detect_zones(current_candles)
        
        # Take the action
        reward = 0
        reward_components = {}
        info = {"action": action}
        
        # Calculate trend direction (simple moving average direction)
        trend_window = 10
        trend_start = max(0, self.current_step - trend_window)
        if self.current_step > trend_window:
            recent_prices = self.data.iloc[trend_start:self.current_step+1]['close'].values
            trend_direction = 1 if recent_prices[-1] > recent_prices[0] else -1
        else:
            trend_direction = 0  # No clear trend with limited data
        
        # Process the action
        if action == 1:  # BUY
            if self.current_position is None:
                # Open a new long position
                self.current_position = {
                    "type": "BUY",
                    "entry_price": current_price,
                    "entry_step": self.current_step,
                    "size": 0.02,  # Fixed lot size
                    "zones_at_entry": zones
                }
                
                # Check if entry is at a good zone (support)
                is_near_zone, nearest_zone = self.sr_detector.is_near_zone(current_price)
                zone_reward = 0
                if is_near_zone and nearest_zone and nearest_zone['zone_type'] == 'support':
                    zone_reward = self.entry_exit_zone_reward
                    reward_components["support_zone_entry"] = zone_reward
                
                # Check if trade aligns with trend
                trend_reward = 0
                if trend_direction > 0:  # Bullish trend
                    trend_reward = self.trend_alignment_reward
                    reward_components["trend_alignment"] = trend_reward
                
                reward = zone_reward + trend_reward
                
                # Overtrading penalty (if too many trades in a short window)
                if self.trades_count > 5:  # More than 5 trades is considered overtrading
                    over_trading_reward = self.overtrading_penalty
                    reward += over_trading_reward
                    reward_components["overtrading"] = over_trading_reward
                
                self.trades_count += 1
                info["position_opened"] = "BUY"
                
            elif self.current_position["type"] == "SELL":
                # Close existing short position
                profit = self.current_position["entry_price"] - current_price
                profit_pct = profit / self.current_position["entry_price"]
                profit_amount = profit_pct * self.account_balance * 50  # 50x leverage assumed
                
                # Apply transaction cost
                profit_amount = profit_amount - (self.account_balance * self.transaction_cost)
                
                # Update account balance
                self.account_balance += profit_amount
                
                # Update max and min balances
                self.max_balance = max(self.max_balance, self.account_balance)
                self.min_balance = min(self.min_balance, self.account_balance)
                
                # Calculate drawdown
                current_drawdown = (self.max_balance - self.account_balance) / self.max_balance
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
                
                # Base reward on profit
                profit_reward = profit_amount * self.profit_reward_weight
                reward_components["profit"] = profit_reward
                
                # Trade win/loss tracking
                if profit_amount > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                    win_streak_bonus = min(self.winning_trades * 0.1, 0.5)  # Bonus for consecutive wins
                    reward_components["win_streak"] = win_streak_bonus
                    profit_reward += win_streak_bonus
                else:
                    self.losing_trades += 1
                    self.consecutive_losses += 1
                    # Increased penalty for consecutive losses
                    consecutive_loss_penalty = -0.2 * self.consecutive_losses
                    reward_components["consecutive_losses"] = consecutive_loss_penalty
                    profit_reward += consecutive_loss_penalty
                
                # Risk-adjusted reward
                sharpe_like = 0
                if self.winning_trades + self.losing_trades > 0:
                    win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
                    risk_reward = (win_rate / (1 - win_rate if win_rate < 1 else 0.99)) * (1 - self.max_drawdown)
                    sharpe_like = risk_reward * self.risk_reward_weight
                    reward_components["risk_adjusted"] = sharpe_like
                
                # Position hold time reward/penalty
                hold_time = self.current_step - self.current_position["entry_step"]
                hold_reward = 0
                if hold_time > 20:  # Penalty for very long holds
                    hold_reward = self.holding_penalty * (hold_time - 20)
                    reward_components["holding_time"] = hold_reward
                
                # Final reward calculation
                reward = profit_reward + sharpe_like + hold_reward
                
                info["position_closed"] = "SELL"
                info["profit"] = profit_amount
                info["hold_time"] = hold_time
                
                self.current_position = None
                
        elif action == 2:  # SELL
            if self.current_position is None:
                # Open a new short position
                self.current_position = {
                    "type": "SELL",
                    "entry_price": current_price,
                    "entry_step": self.current_step,
                    "size": 0.02,  # Fixed lot size
                    "zones_at_entry": zones
                }
                
                # Check if entry is at a good zone (resistance)
                is_near_zone, nearest_zone = self.sr_detector.is_near_zone(current_price)
                zone_reward = 0
                if is_near_zone and nearest_zone and nearest_zone['zone_type'] == 'resistance':
                    zone_reward = self.entry_exit_zone_reward
                    reward_components["resistance_zone_entry"] = zone_reward
                
                # Check if trade aligns with trend
                trend_reward = 0
                if trend_direction < 0:  # Bearish trend
                    trend_reward = self.trend_alignment_reward
                    reward_components["trend_alignment"] = trend_reward
                
                reward = zone_reward + trend_reward
                
                # Overtrading penalty
                if self.trades_count > 5:
                    over_trading_reward = self.overtrading_penalty
                    reward += over_trading_reward
                    reward_components["overtrading"] = over_trading_reward
                
                self.trades_count += 1
                info["position_opened"] = "SELL"
                
            elif self.current_position["type"] == "BUY":
                # Close existing long position
                profit = current_price - self.current_position["entry_price"]
                profit_pct = profit / self.current_position["entry_price"]
                profit_amount = profit_pct * self.account_balance * 50  # 50x leverage assumed
                
                # Apply transaction cost
                profit_amount = profit_amount - (self.account_balance * self.transaction_cost)
                
                # Update account balance
                self.account_balance += profit_amount
                
                # Update max and min balances
                self.max_balance = max(self.max_balance, self.account_balance)
                self.min_balance = min(self.min_balance, self.account_balance)
                
                # Calculate drawdown
                current_drawdown = (self.max_balance - self.account_balance) / self.max_balance
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
                
                # Base reward on profit
                profit_reward = profit_amount * self.profit_reward_weight
                reward_components["profit"] = profit_reward
                
                # Trade win/loss tracking
                if profit_amount > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                    win_streak_bonus = min(self.winning_trades * 0.1, 0.5)  # Bonus for consecutive wins
                    reward_components["win_streak"] = win_streak_bonus
                    profit_reward += win_streak_bonus
                else:
                    self.losing_trades += 1
                    self.consecutive_losses += 1
                    # Increased penalty for consecutive losses
                    consecutive_loss_penalty = -0.2 * self.consecutive_losses
                    reward_components["consecutive_losses"] = consecutive_loss_penalty
                    profit_reward += consecutive_loss_penalty
                
                # Risk-adjusted reward
                sharpe_like = 0
                if self.winning_trades + self.losing_trades > 0:
                    win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
                    risk_reward = (win_rate / (1 - win_rate if win_rate < 1 else 0.99)) * (1 - self.max_drawdown)
                    sharpe_like = risk_reward * self.risk_reward_weight
                    reward_components["risk_adjusted"] = sharpe_like
                
                # Position hold time reward/penalty
                hold_time = self.current_step - self.current_position["entry_step"]
                hold_reward = 0
                if hold_time > 20:  # Penalty for very long holds
                    hold_reward = self.holding_penalty * (hold_time - 20)
                    reward_components["holding_time"] = hold_reward
                
                # Final reward calculation
                reward = profit_reward + sharpe_like + hold_reward
                
                info["position_closed"] = "BUY"
                info["profit"] = profit_amount
                info["hold_time"] = hold_time
                
                self.current_position = None
                
        # Track position hold time
        if self.current_position is not None:
            self.position_hold_time = self.current_step - self.current_position["entry_step"]
            
        # Move to next step
        self.current_step += 1
        
        # Check if we have an open position and update its P&L
        if self.current_position is not None:
            next_price = self.data.iloc[self.current_step]['close']
            
            if self.current_position["type"] == "BUY":
                profit = next_price - self.current_position["entry_price"]
            else:  # SELL
                profit = self.current_position["entry_price"] - next_price
                
            profit_pct = profit / self.current_position["entry_price"]
            unrealized_pnl = profit_pct * self.account_balance * 50  # 50x leverage
            
            info["unrealized_pnl"] = unrealized_pnl
            
            # Small reward/penalty for unrealized P&L (encourages cutting losses)
            if unrealized_pnl < -200:  # Big loss, negative reinforcement to avoid holding losing positions
                reward += -0.1
                reward_components["unrealized_loss_penalty"] = -0.1
            
        # Scale the final reward
        reward = reward * self.reward_scale
        
        # Store step in history
        self.history.append({
            "step": self.current_step,
            "action": action,
            "reward": reward,
            "balance": self.account_balance,
            "info": info,
            "reward_components": reward_components
        })
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Add reward components to info
        info["reward_components"] = reward_components
        info["winning_trades"] = self.winning_trades
        info["losing_trades"] = self.losing_trades
        info["max_drawdown"] = self.max_drawdown
        
        return self._get_observation(), reward, done, info
        
    def _get_observation(self):
        """Get the current observation"""
        if self.data is None or self.current_step >= len(self.data):
            return np.zeros(self.observation_space_dim)
            
        # Use data processor to get feature vector
        window_data = self.data.iloc[max(0, self.current_step-9):self.current_step+1]
        features = self.data_processor.get_feature_vector(window_data)
        
        if features is None or len(features) != self.observation_space_dim:
            # If not enough data, pad with zeros
            logger.warning("Feature vector is None or incorrect length, using zero array")
            return np.zeros(self.observation_space_dim)
            
        return features

class RLAgent:
    """
    Reinforcement learning agent for trading with enhanced reward functions
    """
    
    def __init__(self, data_processor, support_resistance_detector):
        self.data_processor = data_processor
        self.sr_detector = support_resistance_detector
        self.env = TradingEnv(data_processor, support_resistance_detector)
        
        # Adjust hyperparameters for better learning
        if SB3_AVAILABLE:
            self.model = PPO('MlpPolicy', DummyVecEnv([lambda: self.env]), 
                            verbose=1, 
                            learning_rate=0.0001,  # Lower learning rate for better stability
                            n_steps=2048,
                            batch_size=64,
                            n_epochs=10,
                            gamma=0.99,
                            ent_coef=0.005,  # Reduced exploration to be more conservative
                            clip_range=0.2,
                            gae_lambda=0.95,  # Generalized Advantage Estimation
                            max_grad_norm=0.5,  # Clip gradients to stabilize training
                            vf_coef=0.5  # Value function coefficient
                            )
        else:
            self.model = SimplifiedRLModel()
            
        self.action_mapping = {
            0: "HOLD",
            1: "BUY",
            2: "SELL"
        }
        
        self.latest_observation = None
        self.latest_action = None
        self.training_in_progress = False
        self.training_history = []
        self.performance_metrics = {
            "win_rate": 0,
            "avg_reward": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "total_trades": 0
        }
        
    def train(self, data, total_timesteps=10000):
        """Train the agent on historical data with enhanced monitoring"""
        if data is None or len(data) == 0:
            logger.error("No data provided for training")
            return False
            
        # Preprocess the data
        processed_data = self.data_processor.preprocess_data(data)
        if processed_data is None:
            logger.error("Failed to preprocess data")
            return False
            
        # Split data into training and validation sets
        split_idx = int(len(processed_data) * 0.8)
        train_data = processed_data.iloc[:split_idx]
        val_data = processed_data.iloc[split_idx:]
        
        logger.info(f"Training data: {len(train_data)} candles, Validation data: {len(val_data)} candles")
        
        # Reset the environment with the training data
        self.env.reset(train_data)
        
        logger.info(f"Starting training for {total_timesteps} timesteps")
        self.training_in_progress = True
        self.training_history = []
        
        try:
            # Train the model with callbacks for logging
            time_start = datetime.now()
            
            # Calculate initial performance on validation set for comparison
            initial_performance = self._evaluate_on_validation(val_data)
            logger.info(f"Initial performance: {initial_performance}")
            
            # Record training progress
            episode_rewards = []
            episode_lengths = []
            
            # For simplified model, we'll use our own training loop
            if not SB3_AVAILABLE:
                # Simple training loop for the mock model
                for step in range(total_timesteps):
                    if step % 1000 == 0:
                        logger.info(f"Training step {step}/{total_timesteps}")
                        
                    # Reset environment every 200 steps to simulate episodes
                    if step % 200 == 0:
                        obs = self.env.reset()
                        done = False
                        episode_reward = 0
                        episode_length = 0
                        
                    # Get action from model
                    action = self.predict(obs)
                    
                    # Take action in environment
                    obs, reward, done, info = self.env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    if done:
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(episode_length)
                        
                        if len(episode_rewards) % 10 == 0:
                            # Periodically log progress
                            avg_reward = sum(episode_rewards[-10:]) / 10
                            avg_length = sum(episode_lengths[-10:]) / 10
                            logger.info(f"Episodes: {len(episode_rewards)}, Avg reward: {avg_reward:.2f}, Avg length: {avg_length:.1f}")
                            
                            self.training_history.append({
                                "step": step,
                                "avg_reward": avg_reward,
                                "avg_length": avg_length
                            })
                            
                            # Periodically evaluate on validation set
                            if len(episode_rewards) % 50 == 0:
                                val_performance = self._evaluate_on_validation(val_data)
                                logger.info(f"Validation performance: {val_performance}")
                                
                                self.training_history[-1].update({
                                    "val_performance": val_performance
                                })
                                
                # The Stable Baselines3 learn method with callbacks
            else:
                self.model.learn(
                    total_timesteps=total_timesteps,
                    log_interval=10,
                    reset_num_timesteps=True,
                    progress_bar=True
                )
                
                # Extract training metrics from Stable Baselines3
                self.training_history = [{
                    "step": step,
                    "reward": reward
                } for step, reward in self.model.ep_info_buffer]
                
            # Final evaluation on validation set
            final_performance = self._evaluate_on_validation(val_data)
            logger.info(f"Final performance: {final_performance}")
            
            # Update performance metrics
            self.performance_metrics = final_performance
            
            training_time = (datetime.now() - time_start).total_seconds()
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save the trained model
            model_dir = os.path.join("models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"trading_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.model.save(model_path)
            
            # Save training history
            history_path = model_path + "_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f)
            
            logger.info(f"Training completed, model saved to {model_path}")
            self.training_in_progress = False
            return True
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self.training_in_progress = False
            return False
    
    def _evaluate_on_validation(self, val_data, n_episodes=5):
        """Evaluate model performance on validation data"""
        if val_data is None or len(val_data) < 50:
            logger.warning("Validation data too small, skipping evaluation")
            return {
                "win_rate": 0,
                "avg_reward": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "total_trades": 0
            }
            
        # Create a new environment for evaluation to avoid affecting training
        eval_env = TradingEnv(self.data_processor, self.sr_detector)
        eval_env.reset(val_data)
        
        # Track metrics
        total_rewards = []
        episode_rewards = []
        episode_lengths = []
        trades = []
        wins = 0
        losses = 0
        
        # Run multiple episodes for more robust evaluation
        for episode in range(n_episodes):
            # Reset environment with different starting points in the validation data
            start_idx = (episode * len(val_data) // n_episodes) % max(1, len(val_data) - 50)
            episode_data = val_data.iloc[start_idx:start_idx + min(200, len(val_data) - start_idx)]
            
            obs = eval_env.reset(episode_data)
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Get action from model
                action = self.predict(obs)
                
                # Take action in environment
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                total_rewards.append(reward)
                
                # Record trades
                if "position_opened" in info:
                    trades.append({
                        "type": info["position_opened"],
                        "step": episode_length
                    })
                elif "position_closed" in info:
                    trades[-1]["profit"] = info.get("profit", 0)
                    if info.get("profit", 0) > 0:
                        wins += 1
                    else:
                        losses += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate performance metrics
        avg_reward = sum(episode_rewards) / max(1, len(episode_rewards))
        avg_length = sum(episode_lengths) / max(1, len(episode_lengths))
        win_rate = wins / max(1, wins + losses) * 100
        
        # Calculate Sharpe ratio (risk-adjusted return)
        if len(total_rewards) > 1:
            returns = np.array(total_rewards)
            sharpe_ratio = returns.mean() / (returns.std() + 1e-6) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
            
        # Maximum drawdown
        max_drawdown = eval_env.max_drawdown
        
        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_episode_length": avg_length,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": wins + losses
        }
            
    def predict(self, observation):
        """
        Make a prediction based on the current observation
        
        Parameters:
        -----------
        observation : numpy.ndarray
            The current observation
            
        Returns:
        --------
        int : The action to take (0: HOLD, 1: BUY, 2: SELL)
        """
        if observation is None:
            logger.warning("Observation is None, returning HOLD action")
            return 0
            
        self.latest_observation = observation
        
        try:
            action, _ = self.model.predict(observation, deterministic=True)
            self.latest_action = action
            return action
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return 0  # Default to HOLD on error
            
    def get_action_name(self, action):
        """Get the string representation of an action"""
        return self.action_mapping.get(action, "UNKNOWN")
        
    def decide_trade(self, current_data, current_zones):
        """
        Decide whether to trade based on current data and detected zones
        
        Parameters:
        -----------
        current_data : pandas.DataFrame
            Current market data
        current_zones : list
            List of detected support/resistance zones
            
        Returns:
        --------
        dict : Trading decision
        """
        if current_data is None or len(current_data) < 10:
            logger.warning("Not enough data for trade decision")
            return {"action": "HOLD", "reason": "Insufficient data"}
            
        # Preprocess the data
        processed_data = self.data_processor.preprocess_data(current_data)
        if processed_data is None:
            logger.error("Failed to preprocess data")
            return {"action": "HOLD", "reason": "Processing error"}
            
        # Get the latest price
        latest_price = processed_data.iloc[-1]['close']
        latest_high = processed_data.iloc[-1]['high']
        latest_low = processed_data.iloc[-1]['low']
        
        # Calculate volatility for dynamic confidence adjustment
        if len(processed_data) > 20:
            volatility = processed_data['close'].rolling(window=20).std().iloc[-1] / latest_price * 100
        else:
            volatility = 0.5  # Default volatility if not enough data
        
        # Check if price is near any zone
        is_near_zone, nearest_zone = self.sr_detector.is_near_zone(latest_price)
        
        # Check for multiple zones (confluence)
        nearby_zones = [zone for zone in current_zones 
                      if abs(zone['price_level'] - latest_price) / latest_price < 0.005]
        zone_confluence = len(nearby_zones) > 1
        
        # Generate observation for model
        observation = self.data_processor.get_feature_vector(processed_data)
        
        # Get model prediction
        action = self.predict(observation)
        action_name = self.get_action_name(action)
        
        # Calculate trend metrics
        sma_20 = processed_data['close'].rolling(window=20).mean().iloc[-1] if len(processed_data) >= 20 else latest_price
        sma_50 = processed_data['close'].rolling(window=50).mean().iloc[-1] if len(processed_data) >= 50 else latest_price
        
        trend_strength = 0
        if sma_20 > sma_50:
            trend_strength = (sma_20 / sma_50 - 1) * 100  # Uptrend strength as percentage
            trend_direction = "uptrend"
        else:
            trend_strength = (sma_50 / sma_20 - 1) * 100  # Downtrend strength as percentage
            trend_direction = "downtrend"
        
        # Decision logic combining model prediction and enhanced analysis
        decision = {
            "action": action_name,
            "confidence": 0.5,  # Default confidence
            "reason": "Model prediction",
            "price": latest_price,
            "timestamp": datetime.now().isoformat(),
            "volatility": volatility,
            "trend": {
                "direction": trend_direction,
                "strength": trend_strength
            }
        }
        
        # If near a zone, adjust decision based on zone type and strength
        if is_near_zone and nearest_zone:
            zone_type = nearest_zone['zone_type']
            zone_price = nearest_zone['price_level']
            zone_strength = nearest_zone['strength']
            
            decision["zone_nearby"] = True
            decision["zone_type"] = zone_type
            decision["zone_price"] = zone_price
            decision["zone_strength"] = zone_strength
            decision["zone_confluence"] = zone_confluence
            
            # Adjust confidence based on zone strength, volatility, and confluence
            base_confidence = min(0.5 + (zone_strength / 20), 0.9)
            
            # Adjust for confluence (multiple zones close together)
            if zone_confluence:
                base_confidence = min(base_confidence + 0.1, 0.95)
                
            # Adjust for volatility (reduce confidence in high volatility)
            vol_adjustment = max(-0.2, min(0.1, -0.1 * (volatility - 1)))
            base_confidence += vol_adjustment
            
            if zone_type == "resistance" and action_name == "SELL":
                decision["confidence"] = base_confidence
                decision["reason"] = "Price near resistance zone, model predicts SELL"
                if zone_confluence:
                    decision["reason"] += " with zone confluence"
            elif zone_type == "support" and action_name == "BUY":
                decision["confidence"] = base_confidence
                decision["reason"] = "Price near support zone, model predicts BUY"
                if zone_confluence:
                    decision["reason"] += " with zone confluence"
            elif zone_type == "resistance" and action_name == "BUY":
                # Model wants to buy at resistance - reduce confidence
                decision["confidence"] = max(0.3, base_confidence - 0.3)
                decision["reason"] = "Price near resistance zone but model predicts BUY (conflicting)"
            elif zone_type == "support" and action_name == "SELL":
                # Model wants to sell at support - reduce confidence
                decision["confidence"] = max(0.3, base_confidence - 0.3)
                decision["reason"] = "Price near support zone but model predicts SELL (conflicting)"
        
        # Check for breakout with improved detection
        if len(processed_data) > 1:
            previous_price = processed_data.iloc[-2]['close']
            previous_high = processed_data.iloc[-2]['high']
            previous_low = processed_data.iloc[-2]['low']
            
            # Check for strong candle breakouts
            strong_candle = (latest_high - latest_low) > 2 * (previous_high - previous_low)
            
            is_breakout, breakout_zone, breakout_direction = self.sr_detector.check_breakout(
                latest_price, previous_price)
                
            if is_breakout:
                decision["breakout_detected"] = True
                decision["breakout_direction"] = breakout_direction
                decision["breakout_zone_type"] = breakout_zone['zone_type']
                decision["strong_candle"] = strong_candle
                
                # Adjust confidence and decision for breakout
                breakout_confidence = 0.8
                if strong_candle:
                    breakout_confidence = 0.9  # Higher confidence for strong candle breakouts
                
                if breakout_direction == "up" and action_name == "BUY":
                    decision["confidence"] = breakout_confidence
                    decision["reason"] = f"Upward breakout of resistance, model confirms BUY{' with strong candle' if strong_candle else ''}"
                elif breakout_direction == "down" and action_name == "SELL":
                    decision["confidence"] = breakout_confidence
                    decision["reason"] = f"Downward breakout of support, model confirms SELL{' with strong candle' if strong_candle else ''}"
                elif breakout_direction == "up" and action_name == "SELL":
                    # Conflicting signals
                    decision["confidence"] = 0.3
                    decision["reason"] = "Upward breakout of resistance, but model predicts SELL (conflicting)"
                elif breakout_direction == "down" and action_name == "BUY":
                    # Conflicting signals
                    decision["confidence"] = 0.3
                    decision["reason"] = "Downward breakout of support, but model predicts BUY (conflicting)"
        
        # Add trend-based adjustments
        if action_name == "BUY" and trend_direction == "uptrend" and trend_strength > 1.0:
            # Buying in an uptrend - increase confidence
            decision["confidence"] = min(decision["confidence"] + 0.1, 0.95)
            decision["reason"] += f" aligned with strong uptrend ({trend_strength:.1f}%)"
        elif action_name == "SELL" and trend_direction == "downtrend" and trend_strength > 1.0:
            # Selling in a downtrend - increase confidence
            decision["confidence"] = min(decision["confidence"] + 0.1, 0.95)
            decision["reason"] += f" aligned with strong downtrend ({trend_strength:.1f}%)"
        elif action_name == "BUY" and trend_direction == "downtrend" and trend_strength > 2.0:
            # Buying against a strong downtrend - decrease confidence
            decision["confidence"] = max(decision["confidence"] - 0.15, 0.2)
            decision["reason"] += f" against strong downtrend ({trend_strength:.1f}%)"
        elif action_name == "SELL" and trend_direction == "uptrend" and trend_strength > 2.0:
            # Selling against a strong uptrend - decrease confidence
            decision["confidence"] = max(decision["confidence"] - 0.15, 0.2)
            decision["reason"] += f" against strong uptrend ({trend_strength:.1f}%)"
        
        # Add performance metrics to the decision for monitoring
        decision["metrics"] = {
            "win_rate": self.performance_metrics.get("win_rate", 0),
            "sharpe_ratio": self.performance_metrics.get("sharpe_ratio", 0),
            "max_drawdown": self.performance_metrics.get("max_drawdown", 0),
        }
            
        return decision
        
    def get_training_history(self):
        """Get the training history for visualization"""
        return self.training_history
        
    def get_performance_metrics(self):
        """Get the current performance metrics"""
        return self.performance_metrics
