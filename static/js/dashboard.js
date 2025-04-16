// Dashboard.js - Main dashboard functionality for the AI Trading Agent

// State variables
let agentStatus = 'Idle';
let mt5Connected = false;
let currentDecision = null;
let openTrades = [];
let supportResistanceZones = [];

// DOM ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the dashboard
    initDashboard();
    
    // Setup event listeners
    setupEventListeners();
    
    // Start the refresh cycle
    startRefreshCycle();
});

// Initialize dashboard components
function initDashboard() {
    // Get initial status
    updateAgentStatus();
    
    // Automatically connect to MT5 with the provided credentials
    setTimeout(function() {
        // Only connect if not already connected
        if (!mt5Connected) {
            console.log("Auto-connecting to MT5...");
            connectToMT5();
        }
    }, 2000); // Wait 2 seconds after page load
    
    // Load open trades
    loadOpenTrades();
    
    // Initial zone detection
    updateSupportResistanceZones();
}

// Setup event listeners for interactive elements
function setupEventListeners() {
    // Connect to MT5 button
    const connectMT5Button = document.getElementById('connect-mt5');
    if (connectMT5Button) {
        connectMT5Button.addEventListener('click', connectToMT5);
    }
    
    // Train agent button
    const trainAgentButton = document.getElementById('train-agent');
    if (trainAgentButton) {
        trainAgentButton.addEventListener('click', trainAgent);
    }
    
    // Get latest decision button
    const getDecisionButton = document.getElementById('get-decision');
    if (getDecisionButton) {
        getDecisionButton.addEventListener('click', getTradeDecision);
    }
    
    // Detect zones button
    const detectZonesButton = document.getElementById('detect-zones');
    if (detectZonesButton) {
        detectZonesButton.addEventListener('click', detectZones);
    }
    
    // Agent active toggle
    const agentActiveToggle = document.getElementById('agent-active');
    if (agentActiveToggle) {
        agentActiveToggle.addEventListener('change', function() {
            const agentStatus = document.getElementById('agent-status-text');
            const agentStatusIcon = document.getElementById('agent-status-icon').querySelector('i');
            
            if (this.checked) {
                agentStatus.textContent = 'Active';
                agentStatus.classList.add('text-success');
                agentStatus.classList.remove('text-secondary');
                agentStatusIcon.classList.add('text-success');
                agentStatusIcon.classList.remove('text-secondary');
                document.getElementById('agent-mode').textContent = 'Trading Mode';
            } else {
                agentStatus.textContent = 'Idle';
                agentStatus.classList.add('text-secondary');
                agentStatus.classList.remove('text-success');
                agentStatusIcon.classList.add('text-secondary');
                agentStatusIcon.classList.remove('text-success');
                document.getElementById('agent-mode').textContent = 'Monitoring Mode';
            }
        });
    }
}

// Start background refresh cycle
function startRefreshCycle() {
    // Update status every 10 seconds
    setInterval(updateAgentStatus, 10000);
    
    // Update open trades every 15 seconds
    setInterval(loadOpenTrades, 15000);
    
    // Update trade decision every 30 seconds
    setInterval(getTradeDecision, 30000);
    
    // Update S/R zones every 5 minutes
    setInterval(updateSupportResistanceZones, 300000);
}

// Update agent status from backend
function updateAgentStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            if (data.success === false) {
                console.error('Error fetching status:', data.error);
                return;
            }
            
            // Update connection status
            mt5Connected = data.mt5_connected;
            updateConnectionStatus(mt5Connected);
            
            // Update agent status
            if (data.agent) {
                agentStatus = data.agent.status;
                updateAgentStatusDisplay(data.agent);
            }
            
            // Update account info
            if (data.account) {
                updateAccountInfo(data.account);
            }
            
            // Update performance metrics
            if (data.performance && data.performance.today) {
                updatePerformanceMetrics(data.performance.today);
            }
        })
        .catch(error => {
            console.error('Error fetching status:', error);
        });
}

// Update connection status display
function updateConnectionStatus(isConnected) {
    const connectionStatus = document.getElementById('connection-status');
    if (!connectionStatus) return;
    
    const icon = connectionStatus.querySelector('i') || document.createElement('i');
    
    if (isConnected) {
        connectionStatus.innerHTML = '';
        icon.className = 'fas fa-plug me-1 text-success';
        connectionStatus.appendChild(icon);
        connectionStatus.appendChild(document.createTextNode('Connected'));
    } else {
        connectionStatus.innerHTML = '';
        icon.className = 'fas fa-plug me-1 text-danger';
        connectionStatus.appendChild(icon);
        connectionStatus.appendChild(document.createTextNode('Disconnected'));
    }
}

// Update agent status display
function updateAgentStatusDisplay(agentData) {
    const agentStatusElement = document.getElementById('agent-status');
    const agentStatusText = document.getElementById('agent-status-text');
    const agentStatusIcon = document.getElementById('agent-status-icon').querySelector('i');
    const agentMode = document.getElementById('agent-mode');
    const agentActiveToggle = document.getElementById('agent-active');
    
    if (!agentStatusElement || !agentStatusText || !agentStatusIcon || !agentMode) return;
    
    // Update active toggle
    if (agentActiveToggle) {
        agentActiveToggle.checked = agentData.is_active;
    }
    
    // Update status text and styling
    agentStatusText.textContent = agentData.status;
    
    // Remove all status classes
    agentStatusIcon.classList.remove('text-success', 'text-danger', 'text-info', 'text-secondary');
    
    // Add appropriate status class
    switch (agentData.status.toLowerCase()) {
        case 'active':
            agentStatusIcon.classList.add('text-success');
            agentMode.textContent = 'Trading Mode';
            break;
        case 'idle':
            agentStatusIcon.classList.add('text-secondary');
            agentMode.textContent = 'Monitoring Mode';
            break;
        case 'retraining':
        case 'training':
            agentStatusIcon.classList.add('text-info');
            agentMode.textContent = 'Training Mode';
            break;
        case 'error':
            agentStatusIcon.classList.add('text-danger');
            agentMode.textContent = 'Error State';
            break;
        default:
            agentStatusIcon.classList.add('text-secondary');
            agentMode.textContent = 'Unknown State';
    }
    
    // Update nav status
    const navStatus = document.getElementById('agent-status');
    if (navStatus) {
        const navIcon = navStatus.querySelector('i') || document.createElement('i');
        navIcon.className = `fas fa-circle me-1 ${agentStatusIcon.className.split(' ').find(c => c.startsWith('text-'))}`;
        navStatus.innerHTML = '';
        navStatus.appendChild(navIcon);
        navStatus.appendChild(document.createTextNode(agentData.status));
    }
}

// Update account information
function updateAccountInfo(accountData) {
    const accountBalance = document.getElementById('account-balance');
    const accountEquity = document.getElementById('account-equity');
    
    if (!accountBalance || !accountEquity) return;
    
    if (accountData) {
        accountBalance.textContent = `$${accountData.balance.toFixed(2)}`;
        accountEquity.textContent = `Equity: $${accountData.equity.toFixed(2)}`;
    } else {
        accountBalance.textContent = '$0.00';
        accountEquity.textContent = 'Equity: $0.00';
    }
}

// Update performance metrics
function updatePerformanceMetrics(metrics) {
    const dailyPL = document.getElementById('daily-pl');
    const dailyTrades = document.getElementById('daily-trades');
    const winRate = document.getElementById('win-rate');
    const winLossRatio = document.getElementById('win-loss-ratio');
    
    if (!dailyPL || !dailyTrades || !winRate || !winLossRatio) return;
    
    if (metrics) {
        // Update P/L display
        const plValue = metrics.profit_loss || 0;
        dailyPL.textContent = `$${plValue.toFixed(2)}`;
        dailyPL.className = plValue >= 0 ? 'mb-0 text-success' : 'mb-0 text-danger';
        
        // Daily trades count
        const tradeCount = metrics.trade_count || 0;
        dailyTrades.textContent = `${tradeCount} Trade${tradeCount !== 1 ? 's' : ''}`;
        
        // Win rate
        const winRateValue = metrics.win_rate || 0;
        winRate.textContent = `${winRateValue.toFixed(1)}%`;
        
        // Win/Loss ratio
        const wins = metrics.win_count || 0;
        const losses = metrics.loss_count || 0;
        winLossRatio.textContent = `${wins}W / ${losses}L`;
    } else {
        dailyPL.textContent = '$0.00';
        dailyTrades.textContent = '0 Trades';
        winRate.textContent = '0%';
        winLossRatio.textContent = '0W / 0L';
    }
}

// Connect to MT5
function connectToMT5() {
    // Show loading state
    const button = document.getElementById('connect-mt5');
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Connecting...';
    button.disabled = true;
    
    // If already connected, disconnect
    if (mt5Connected) {
        fetch('/api/disconnect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                mt5Connected = false;
                updateConnectionStatus(false);
                button.innerHTML = '<i class="fas fa-plug me-2"></i>Connect MT5';
                button.disabled = false;
                showAlert('Disconnected from MT5 successfully', 'success');
            } else {
                button.innerHTML = originalText;
                button.disabled = false;
                showAlert('Failed to disconnect: ' + (data.message || data.error), 'danger');
            }
        })
        .catch(error => {
            button.innerHTML = originalText;
            button.disabled = false;
            showAlert('Error: ' + error.message, 'danger');
        });
        return;
    }
    
    // Connect to MT5 with user's credentials
    fetch('/api/connect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            path: null,
            login: '91873732',
            password: 'Ds!j4wUh',
            server: 'MetaQuotes-Demo'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            mt5Connected = true;
            updateConnectionStatus(true);
            button.innerHTML = '<i class="fas fa-times me-2"></i>Disconnect MT5';
            button.disabled = false;
            showAlert('Connected to MT5 successfully', 'success');
            
            // Update chart data
            updateChartData();
        } else {
            button.innerHTML = originalText;
            button.disabled = false;
            showAlert('Failed to connect: ' + (data.message || data.error), 'danger');
        }
    })
    .catch(error => {
        button.innerHTML = originalText;
        button.disabled = false;
        showAlert('Error: ' + error.message, 'danger');
    });
}

// Train the agent
function trainAgent() {
    // Check if MT5 is connected
    if (!mt5Connected) {
        showAlert('Please connect to MT5 first', 'warning');
        return;
    }
    
    // Show loading state
    const button = document.getElementById('train-agent');
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Training...';
    button.disabled = true;
    
    // Start training
    fetch('/api/train_agent', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            timesteps: 10000
        })
    })
    .then(response => response.json())
    .then(data => {
        button.innerHTML = originalText;
        button.disabled = false;
        
        if (data.success) {
            showAlert(data.message, 'success');
            updateAgentStatus();
        } else {
            showAlert(data.message || data.error, 'danger');
        }
    })
    .catch(error => {
        button.innerHTML = originalText;
        button.disabled = false;
        showAlert('Error: ' + error.message, 'danger');
    });
}

// Get the latest trade decision
function getTradeDecision() {
    // Check if MT5 is connected
    if (!mt5Connected) {
        return;
    }
    
    const decisionContainer = document.getElementById('latest-decision');
    if (!decisionContainer) return;
    
    // Fetch latest decision
    fetch('/api/trade_decision')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.decision) {
                currentDecision = data.decision;
                
                // Create decision card
                const action = data.decision.action;
                const confidence = data.decision.confidence;
                const reason = data.decision.reason;
                const price = data.decision.price;
                
                let cardClass = 'alert ';
                let actionIcon = '';
                
                switch (action.toUpperCase()) {
                    case 'BUY':
                        cardClass += 'alert-success';
                        actionIcon = '<i class="fas fa-arrow-up me-2"></i>';
                        break;
                    case 'SELL':
                        cardClass += 'alert-danger';
                        actionIcon = '<i class="fas fa-arrow-down me-2"></i>';
                        break;
                    default:
                        cardClass += 'alert-warning';
                        actionIcon = '<i class="fas fa-minus me-2"></i>';
                }
                
                // Build the decision HTML
                let decisionHTML = `
                    <div class="${cardClass}">
                        <h5 class="alert-heading">${actionIcon}${action.toUpperCase()}</h5>
                        <p><strong>Confidence:</strong> ${(confidence * 100).toFixed(1)}%</p>
                        <p><strong>Price:</strong> $${price.toFixed(2)}</p>
                        <p><strong>Reasoning:</strong> ${reason}</p>
                        <hr>
                        <p class="mb-0 text-muted"><small>Updated at ${new Date().toLocaleTimeString()}</small></p>
                    </div>
                `;
                
                // Add Execute button if agent is active and confidence is high enough
                if (action !== 'HOLD' && confidence >= 0.6 && agentStatus === 'Active') {
                    decisionHTML += `
                        <div class="d-grid">
                            <button id="execute-trade" class="btn btn-sm btn-primary mb-2">
                                <i class="fas fa-check-circle me-2"></i>Execute ${action.toUpperCase()} Trade
                            </button>
                        </div>
                    `;
                }
                
                decisionContainer.innerHTML = decisionHTML;
                
                // Add event listener to the execute button
                const executeButton = document.getElementById('execute-trade');
                if (executeButton) {
                    executeButton.addEventListener('click', executeTrade);
                }
            } else {
                decisionContainer.innerHTML = `
                    <div class="alert alert-secondary">
                        ${data.message || 'No trade decision available. MT5 might be disconnected.'}
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error fetching trade decision:', error);
        });
}

// Execute a trade based on the current decision
function executeTrade() {
    // Check if MT5 is connected
    if (!mt5Connected) {
        showAlert('Please connect to MT5 first', 'warning');
        return;
    }
    
    // Show loading state
    const button = document.getElementById('execute-trade');
    const originalText = button.innerHTML;
    button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Executing...';
    button.disabled = true;
    
    // Execute the trade
    fetch('/api/execute_trade', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert(data.message, 'success');
            
            // Refresh open trades
            loadOpenTrades();
            
            // Get new decision
            setTimeout(getTradeDecision, 2000);
        } else {
            button.innerHTML = originalText;
            button.disabled = false;
            showAlert(data.message || data.error, 'danger');
        }
    })
    .catch(error => {
        button.innerHTML = originalText;
        button.disabled = false;
        showAlert('Error: ' + error.message, 'danger');
    });
}

// Load open trades
function loadOpenTrades() {
    fetch('/api/open_trades')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                openTrades = data.trades || [];
                updateOpenTradesTable(openTrades);
            } else {
                console.error('Error loading open trades:', data.error);
            }
        })
        .catch(error => {
            console.error('Error fetching open trades:', error);
        });
}

// Update the open trades table
function updateOpenTradesTable(trades) {
    const table = document.getElementById('open-trades-table');
    if (!table) return;
    
    const tbody = table.querySelector('tbody');
    
    // Clear existing rows
    tbody.innerHTML = '';
    
    if (trades.length === 0) {
        tbody.innerHTML = `
            <tr id="no-trades-row">
                <td colspan="9" class="text-center">No open trades</td>
            </tr>
        `;
        return;
    }
    
    // Add trade rows
    trades.forEach(trade => {
        const row = document.createElement('tr');
        
        // Calculate P/L
        let pl = trade.unrealized_pl || 0;
        let plClass = pl >= 0 ? 'text-success' : 'text-danger';
        
        row.innerHTML = `
            <td>${trade.id}</td>
            <td class="${trade.order_type === 'buy' ? 'text-success' : 'text-danger'}">${trade.order_type.toUpperCase()}</td>
            <td>$${trade.entry_price.toFixed(2)}</td>
            <td>$${(trade.current_price || trade.entry_price).toFixed(2)}</td>
            <td>$${trade.stop_loss.toFixed(2)}</td>
            <td>$${trade.take_profit.toFixed(2)}</td>
            <td>${trade.lot_size}</td>
            <td class="${plClass}">$${Math.abs(pl).toFixed(2)}</td>
            <td>
                <button class="btn btn-sm btn-outline-danger close-trade-btn" data-trade-id="${trade.id}">
                    <i class="fas fa-times"></i> Close
                </button>
            </td>
        `;
        
        tbody.appendChild(row);
    });
    
    // Add event listeners to close buttons
    const closeButtons = document.querySelectorAll('.close-trade-btn');
    closeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tradeId = this.getAttribute('data-trade-id');
            closeTrade(tradeId);
        });
    });
}

// Close a specific trade
function closeTrade(tradeId) {
    // Show confirmation
    if (!confirm('Are you sure you want to close this trade?')) {
        return;
    }
    
    fetch(`/api/close_trade/${tradeId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showAlert(data.message, 'success');
            loadOpenTrades();
        } else {
            showAlert(data.message || data.error, 'danger');
        }
    })
    .catch(error => {
        showAlert('Error: ' + error.message, 'danger');
    });
}

// Update support and resistance zones
function updateSupportResistanceZones() {
    // If MT5 is not connected, use zones from the template
    if (!mt5Connected) return;
    
    // Detect zones
    detectZones();
}

// Detect support and resistance zones
function detectZones() {
    fetch('/api/detect_zones', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            n_candles: 50
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            supportResistanceZones = data.zones || [];
            updateZonesList(supportResistanceZones);
            
            // If chart is initialized, update zones on chart
            if (typeof updateChartZones === 'function') {
                updateChartZones(supportResistanceZones);
            }
        } else {
            console.error('Error detecting zones:', data.error);
        }
    })
    .catch(error => {
        console.error('Error detecting zones:', error);
    });
}

// Update zones list in the UI
function updateZonesList(zones) {
    const zonesList = document.getElementById('zones-list');
    if (!zonesList) return;
    
    // Clear existing zones
    zonesList.innerHTML = '';
    
    if (zones.length === 0) {
        zonesList.innerHTML = '<li class="list-group-item text-center">No active zones detected</li>';
        return;
    }
    
    // Add zone items
    zones.forEach(zone => {
        const listItem = document.createElement('li');
        listItem.className = `list-group-item d-flex justify-content-between align-items-center ${zone.zone_type === 'resistance' ? 'list-group-item-danger' : 'list-group-item-success'}`;
        
        listItem.innerHTML = `
            <div>
                <strong>${zone.zone_type.charAt(0).toUpperCase() + zone.zone_type.slice(1)}</strong>
                <span class="badge bg-dark ms-2">Strength: ${zone.strength.toFixed(1)}</span>
            </div>
            <div class="fw-bold">$${zone.price_level.toFixed(2)}</div>
        `;
        
        zonesList.appendChild(listItem);
    });
}

// Show an alert message
function showAlert(message, type = 'info') {
    // Create alert container if it doesn't exist
    let alertContainer = document.querySelector('.alert-container');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.className = 'alert-container position-fixed top-0 end-0 p-3';
        alertContainer.style.zIndex = 1050;
        document.body.appendChild(alertContainer);
    }
    
    // Create alert
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type} alert-dismissible fade show`;
    alertElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to container
    alertContainer.appendChild(alertElement);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertElement.parentNode) {
            alertElement.classList.remove('show');
            setTimeout(() => {
                if (alertElement.parentNode) {
                    alertElement.parentNode.removeChild(alertElement);
                }
            }, 150);
        }
    }, 5000);
}
