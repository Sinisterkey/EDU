// Training visualization functions for the RL agent

// Initialize the training metrics visualization
function initTrainingViz() {
    console.log('Initializing training visualization');
    
    // Setup event listeners
    document.getElementById('view-training-metrics').addEventListener('click', loadTrainingHistory);
}

// Load training history data from the API
function loadTrainingHistory() {
    fetch('/api/training_history')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayTrainingMetrics(data.history, data.metrics);
            } else {
                console.error('Error fetching training history:', data.error);
                showTrainingError('Failed to load training metrics. Please try again.');
            }
        })
        .catch(error => {
            console.error('Error fetching training history:', error);
            showTrainingError('Failed to load training metrics. Please check the console for details.');
        });
}

// Display training metrics in the UI
function displayTrainingMetrics(history, metrics) {
    const trainingModal = document.getElementById('training-metrics-modal');
    const modalBody = trainingModal.querySelector('.modal-body');
    
    // Clear previous content
    modalBody.innerHTML = '';
    
    if (!history || history.length === 0) {
        modalBody.innerHTML = '<div class="alert alert-info">No training history available. Train the agent first.</div>';
        // Show the modal
        const bsModal = new bootstrap.Modal(trainingModal);
        bsModal.show();
        return;
    }
    
    // Create metrics overview section
    const metricsOverview = document.createElement('div');
    metricsOverview.className = 'mb-4 p-3 border rounded bg-dark';
    metricsOverview.innerHTML = `
        <h5>Performance Metrics</h5>
        <div class="row">
            <div class="col-md-4">
                <div class="card bg-secondary mb-2">
                    <div class="card-body text-center">
                        <h3>${metrics.win_rate ? metrics.win_rate.toFixed(1) : 0}%</h3>
                        <p class="mb-0">Win Rate</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-secondary mb-2">
                    <div class="card-body text-center">
                        <h3>${metrics.sharpe_ratio ? metrics.sharpe_ratio.toFixed(2) : 0}</h3>
                        <p class="mb-0">Sharpe Ratio</p>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-secondary mb-2">
                    <div class="card-body text-center">
                        <h3>${metrics.max_drawdown ? (metrics.max_drawdown * 100).toFixed(1) : 0}%</h3>
                        <p class="mb-0">Max Drawdown</p>
                    </div>
                </div>
            </div>
        </div>
        <div class="row mt-2">
            <div class="col-md-6">
                <div class="card bg-secondary">
                    <div class="card-body text-center">
                        <h3>${metrics.total_trades || 0}</h3>
                        <p class="mb-0">Total Trades</p>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card bg-secondary">
                    <div class="card-body text-center">
                        <h3>${metrics.avg_reward ? metrics.avg_reward.toFixed(2) : 0}</h3>
                        <p class="mb-0">Avg Reward</p>
                    </div>
                </div>
            </div>
        </div>
    `;
    modalBody.appendChild(metricsOverview);
    
    // Create container for reward chart
    const chartContainer = document.createElement('div');
    chartContainer.className = 'mt-4';
    chartContainer.innerHTML = `
        <h5>Training Progress</h5>
        <div class="chart-container" style="position: relative; height: 300px;">
            <canvas id="training-reward-chart"></canvas>
        </div>
    `;
    modalBody.appendChild(chartContainer);
    
    // Create reward components section if available
    if (history.length > 0 && history[0].reward_components) {
        const componentsContainer = document.createElement('div');
        componentsContainer.className = 'mt-4';
        componentsContainer.innerHTML = `
            <h5>Reward Components</h5>
            <div class="chart-container" style="position: relative; height: 250px;">
                <canvas id="reward-components-chart"></canvas>
            </div>
            <div class="mt-3 small text-muted">
                <p>Components explanation:</p>
                <ul>
                    <li><strong>Profit:</strong> Direct profit from trades</li>
                    <li><strong>Risk-adjusted:</strong> Reward based on risk/return profile</li>
                    <li><strong>Zone entry:</strong> Reward for entering at support/resistance</li>
                    <li><strong>Trend alignment:</strong> Reward for trading with the trend</li>
                    <li><strong>Holding time:</strong> Penalty for holding positions too long</li>
                    <li><strong>Overtrading:</strong> Penalty for excessive trading</li>
                </ul>
            </div>
        `;
        modalBody.appendChild(componentsContainer);
    }
    
    // Show the modal
    const bsModal = new bootstrap.Modal(trainingModal);
    bsModal.show();
    
    // Render charts after modal is shown (to ensure proper sizing)
    setTimeout(() => {
        renderTrainingCharts(history);
    }, 300);
}

// Render training progress charts
function renderTrainingCharts(history) {
    // Training reward chart
    renderRewardChart(history);
    
    // Render reward components if available
    if (history.length > 0 && history[0].reward_components) {
        renderRewardComponentsChart(history);
    }
}

// Render main reward chart
function renderRewardChart(history) {
    const ctx = document.getElementById('training-reward-chart').getContext('2d');
    
    // Extract data points
    const steps = history.map(h => h.step || 0);
    const rewards = history.map(h => h.avg_reward || h.reward || 0);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: steps,
            datasets: [{
                label: 'Average Reward',
                data: rewards,
                fill: false,
                borderColor: '#4bc0c0',
                tension: 0.1,
                pointRadius: 1,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Training Step'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Reward'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            }
        }
    });
}

// Render reward components chart (stacked bar chart)
function renderRewardComponentsChart(history) {
    // This is a placeholder - in a real implementation, we would extract
    // and visualize the actual reward components
    const ctx = document.getElementById('reward-components-chart').getContext('2d');
    
    // Create mock data for visualization - in a real app, extract from history
    const data = {
        labels: ['Trade 1', 'Trade 2', 'Trade 3', 'Trade 4', 'Trade 5'],
        datasets: [
            {
                label: 'Profit',
                data: [120, -50, 80, 90, -30],
                backgroundColor: '#4bc0c0',
            },
            {
                label: 'Risk Adjusted',
                data: [20, 10, 15, 25, 5],
                backgroundColor: '#36a2eb',
            },
            {
                label: 'Zone Entry',
                data: [30, 0, 40, 0, 25],
                backgroundColor: '#ffcd56',
            },
            {
                label: 'Trend Alignment',
                data: [15, 10, 20, 25, 0],
                backgroundColor: '#ff9f40',
            },
            {
                label: 'Holding Time',
                data: [-10, -5, -20, -15, -5],
                backgroundColor: '#ff6384',
            },
            {
                label: 'Overtrading',
                data: [0, -15, 0, -10, -5],
                backgroundColor: '#9966ff',
            }
        ]
    };

    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    stacked: true,
                },
                y: {
                    stacked: true,
                    title: {
                        display: true,
                        text: 'Reward Component Value'
                    }
                }
            },
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            }
        }
    });
}

// Display error message
function showTrainingError(message) {
    const trainingModal = document.getElementById('training-metrics-modal');
    const modalBody = trainingModal.querySelector('.modal-body');
    
    modalBody.innerHTML = `<div class="alert alert-danger">${message}</div>`;
    
    const bsModal = new bootstrap.Modal(trainingModal);
    bsModal.show();
}

// Initialize when the document is ready
document.addEventListener('DOMContentLoaded', function() {
    initTrainingViz();
});