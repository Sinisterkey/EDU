// Chart.js - Handles the trading chart visualization

// Chart state
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let supportZones = [];
let resistanceZones = [];
let showZones = true;
let tradeMarkers = [];

// Initialize chart when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initChart();
    
    // Add event listeners
    setupChartEventListeners();
});

// Initialize the trading chart
function initChart() {
    const chartContainer = document.getElementById('chart-container');
    if (!chartContainer) return;
    
    // Create the chart
    chart = LightweightCharts.createChart(chartContainer, {
        width: chartContainer.clientWidth,
        height: 400,
        layout: {
            backgroundColor: '#131722',
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: {
                color: 'rgba(42, 46, 57, 0.5)',
            },
            horzLines: {
                color: 'rgba(42, 46, 57, 0.5)',
            },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: 'rgba(197, 203, 206, 0.8)',
            scaleMargins: {
                top: 0.1,
                bottom: 0.25,
            },
        },
        timeScale: {
            borderColor: 'rgba(197, 203, 206, 0.8)',
            timeVisible: true,
            secondsVisible: false,
        },
    });
    
    // Add a candlestick series
    candleSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderDownColor: '#ef5350',
        borderUpColor: '#26a69a',
        wickDownColor: '#ef5350',
        wickUpColor: '#26a69a',
    });
    
    // Add a volume series
    volumeSeries = chart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: '',
        scaleMargins: {
            top: 0.8,
            bottom: 0,
        },
    });
    
    // Load initial data
    loadChartData();
    
    // Load initial trade markers
    fetchOpenTrades();
    
    // Handle resize
    window.addEventListener('resize', function() {
        if (chart) {
            chart.resize(chartContainer.clientWidth, 400);
        }
    });
}

// Setup chart event listeners
function setupChartEventListeners() {
    // Refresh chart button
    const refreshButton = document.getElementById('refresh-chart');
    if (refreshButton) {
        refreshButton.addEventListener('click', function() {
            loadChartData();
        });
    }
    
    // Toggle zones button
    const toggleZonesButton = document.getElementById('toggle-zones');
    if (toggleZonesButton) {
        toggleZonesButton.addEventListener('click', function() {
            showZones = !showZones;
            this.querySelector('i').className = showZones 
                ? 'fas fa-layer-group' 
                : 'fas fa-layer-group text-secondary';
            
            // Update zones visibility
            if (showZones) {
                drawSupportResistanceZones();
            } else {
                clearSupportResistanceZones();
            }
        });
    }
}

// Load chart data from the API
function loadChartData() {
    // Show loading state
    const chartContainer = document.getElementById('chart-container');
    if (chartContainer) {
        chartContainer.style.opacity = 0.5;
    }
    
    fetch('/api/historical_data?bars=200')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.data) {
                updateChartWithData(data.data);
            } else {
                console.error('Error loading chart data:', data.error || 'Unknown error');
            }
            
            // Remove loading state
            if (chartContainer) {
                chartContainer.style.opacity = 1;
            }
        })
        .catch(error => {
            console.error('Error fetching chart data:', error);
            
            // Remove loading state
            if (chartContainer) {
                chartContainer.style.opacity = 1;
            }
        });
}

// Update chart with new data
function updateChartWithData(data) {
    if (!chart || !candleSeries || !volumeSeries) return;
    
    // Format data for chart
    const candleData = [];
    const volumeData = [];
    
    data.forEach(item => {
        const time = typeof item.time === 'string' 
            ? new Date(item.time).getTime() / 1000 
            : item.time;
            
        candleData.push({
            time: time,
            open: item.open,
            high: item.high,
            low: item.low,
            close: item.close
        });
        
        volumeData.push({
            time: time,
            value: item.volume,
            color: item.close >= item.open ? 'rgba(38, 166, 154, 0.5)' : 'rgba(239, 83, 80, 0.5)'
        });
    });
    
    // Set the new data
    candleSeries.setData(candleData);
    volumeSeries.setData(volumeData);
    
    // Fit content
    chart.timeScale().fitContent();
    
    // Fetch and draw support/resistance zones
    fetchSupportResistanceZones();
}

// Fetch support and resistance zones
function fetchSupportResistanceZones() {
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
            // Update zones globally
            updateChartZones(data.zones || []);
        } else {
            console.error('Error detecting zones:', data.error);
        }
    })
    .catch(error => {
        console.error('Error detecting zones:', error);
    });
}

// Update chart with support/resistance zones
function updateChartZones(zones) {
    // Split zones by type
    const support = zones.filter(zone => zone.zone_type === 'support');
    const resistance = zones.filter(zone => zone.zone_type === 'resistance');
    
    // Store zones
    supportZones = support;
    resistanceZones = resistance;
    
    // Draw zones if enabled
    if (showZones) {
        drawSupportResistanceZones();
    }
    
    // Update zones list
    updateZonesList(zones);
}

// Draw support and resistance zones on the chart
function drawSupportResistanceZones() {
    if (!chart || !candleSeries) return;
    
    // Clear existing zones
    clearSupportResistanceZones();
    
    // Get price range to calculate zone height
    const visibleRange = candleSeries.priceScale().getVisibleRange();
    if (!visibleRange) return;
    
    const priceRange = visibleRange.maxValue - visibleRange.minValue;
    const zoneHeight = priceRange * 0.005; // 0.5% of visible price range
    
    // Draw support zones
    supportZones.forEach(zone => {
        const line = {
            price: zone.price_level,
            color: 'rgba(38, 166, 154, 0.6)',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            axisLabelVisible: true,
            title: `S (${zone.strength.toFixed(1)})`
        };
        
        candleSeries.createPriceLine(line);
        
        // Add zone rectangle
        const rect = {
            price1: zone.price_level - zoneHeight,
            price2: zone.price_level + zoneHeight,
            color: 'rgba(38, 166, 154, 0.2)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Solid,
            lineVisible: false,
            axisLabelVisible: false
        };
        
        candleSeries.createPriceLine(rect);
    });
    
    // Draw resistance zones
    resistanceZones.forEach(zone => {
        const line = {
            price: zone.price_level,
            color: 'rgba(239, 83, 80, 0.6)',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            axisLabelVisible: true,
            title: `R (${zone.strength.toFixed(1)})`
        };
        
        candleSeries.createPriceLine(line);
        
        // Add zone rectangle
        const rect = {
            price1: zone.price_level - zoneHeight,
            price2: zone.price_level + zoneHeight,
            color: 'rgba(239, 83, 80, 0.2)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Solid,
            lineVisible: false,
            axisLabelVisible: false
        };
        
        candleSeries.createPriceLine(rect);
    });
}

// Clear all support and resistance zones from chart
function clearSupportResistanceZones() {
    if (!candleSeries) return;
    
    try {
        // Check if the method exists before calling it
        if (typeof candleSeries.removeAllPriceLines === 'function') {
            // Remove all price lines
            candleSeries.removeAllPriceLines();
        } else {
            console.warn('removeAllPriceLines method not available, trying to manually clear all price lines');
            // Alternative way to clear price lines
            supportZones.forEach(zone => {
                try {
                    if (zone.line && candleSeries.removePriceLine) {
                        candleSeries.removePriceLine(zone.line);
                    }
                } catch (e) {
                    console.error('Failed to remove price line:', e);
                }
            });
            
            resistanceZones.forEach(zone => {
                try {
                    if (zone.line && candleSeries.removePriceLine) {
                        candleSeries.removePriceLine(zone.line);
                    }
                } catch (e) {
                    console.error('Failed to remove price line:', e);
                }
            });
        }
    } catch (e) {
        console.error('Error clearing price lines:', e);
    }
    
    // Re-add trade markers if they exist
    if (tradeMarkers.length > 0) {
        addTradeMarkers(tradeMarkers);
    }
}

// Update chart data (can be called from dashboard.js)
function updateChartData() {
    loadChartData();
    // Also update trades
    fetchOpenTrades();
}

// Add trade markers to the chart
function addTradeMarkers(trades) {
    if (!candleSeries) return;
    
    // Prepare markers
    const markers = [];
    
    trades.forEach(trade => {
        // Create marker object for each trade
        const isBuy = trade.order_type === 'buy';
        const markerColor = isBuy ? '#26a69a' : '#ef5350'; // Green for buy, red for sell
        const markerShape = isBuy ? 'arrowUp' : 'arrowDown';
        const markerText = `${isBuy ? 'BUY' : 'SELL'} ${trade.lot_size}`;
        
        // Convert time to seconds
        const time = typeof trade.entry_time === 'string' 
            ? new Date(trade.entry_time).getTime() / 1000 
            : trade.entry_time;
        
        // Create marker
        const marker = {
            time: time,
            position: isBuy ? 'belowBar' : 'aboveBar',
            color: markerColor,
            shape: markerShape,
            text: markerText,
            id: `trade-${trade.id}`
        };
        
        markers.push(marker);
    });
    
    // Add markers to chart
    candleSeries.setMarkers(markers);
    
    // Store markers for later reference
    tradeMarkers = trades;
}

// Fetch open trades to display on chart
function fetchOpenTrades() {
    fetch('/api/open_trades')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.trades) {
                // Add trade markers to chart
                addTradeMarkers(data.trades);
            } else {
                console.error('Error fetching open trades:', data.error);
            }
        })
        .catch(error => {
            console.error('Error fetching open trades:', error);
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
