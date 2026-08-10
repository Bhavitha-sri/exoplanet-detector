/**
 * Dynamic Light Curve Chart Builder using Chart.js.
 * Renders stellar flux F(t) over time with transit dip highlights.
 */

let lightCurveChart = null;

function renderLightCurveChart(canvasId, timeArray, fluxArray, sampleName = "Stellar Light Curve") {
    const ctx = document.getElementById(canvasId).getContext('2d');

    // Create background gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(139, 92, 246, 0.35)');
    gradient.addColorStop(1, 'rgba(139, 92, 246, 0.01)');

    // Compute min flux for highlighting dip
    const minFlux = Math.min(...fluxArray);
    const minIndices = fluxArray.map((f, i) => f === minFlux ? i : -1).filter(i => i !== -1);

    const dataPoints = timeArray.map((t, idx) => ({
        x: t,
        y: fluxArray[idx]
    }));

    if (lightCurveChart) {
        lightCurveChart.destroy();
    }

    lightCurveChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Relative Flux F(t)',
                data: dataPoints,
                borderColor: '#8b5cf6',
                borderWidth: 1.8,
                backgroundColor: gradient,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                pointHoverRadius: 4,
                pointHoverBackgroundColor: '#06b6d4'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 600,
                easing: 'easeOutQuart'
            },
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Time (Hours)',
                        color: '#94a3b8',
                        font: { size: 12, family: 'Inter' }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.04)'
                    },
                    ticks: {
                        color: '#64748b'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Normalized Flux',
                        color: '#94a3b8',
                        font: { size: 12, family: 'Inter' }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.04)'
                    },
                    ticks: {
                        color: '#64748b',
                        callback: function(val) {
                            return val.toFixed(4);
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#f8fafc',
                    bodyColor: '#c4b5fd',
                    borderColor: 'rgba(139, 92, 246, 0.3)',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: false,
                    callbacks: {
                        title: function(items) {
                            return `Time: ${parseFloat(items[0].parsed.x).toFixed(2)} hrs`;
                        },
                        label: function(item) {
                            return `Flux: ${parseFloat(item.parsed.y).toFixed(6)}`;
                        }
                    }
                }
            }
        }
    });
}
