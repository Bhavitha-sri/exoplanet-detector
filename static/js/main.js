/**
 * Main Application Logic for AI-Powered Exoplanet Detector.
 * Manages tab switching, sample target fetching, file uploads, slider input, and metrics loading.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Load default preset on initialization
    loadSamplePreset('kepler_22b_candidate.csv');
    // Fetch empirical model benchmark results
    fetchBenchmarkMetrics();
});

/** Tab Switcher */
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');

    document.getElementById('view-presets').style.display = tabName === 'presets' ? 'flex' : 'none';
    document.getElementById('view-upload').style.display = tabName === 'upload' ? 'block' : 'none';
    document.getElementById('view-catalog').style.display = tabName === 'catalog' ? 'block' : 'none';

    if (tabName === 'catalog') {
        updateCatalogPrediction();
    }
}

/** Load Selected Kepler Preset Target */
function loadSamplePreset(filename, cardElem = null) {
    if (cardElem) {
        document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('selected'));
        cardElem.classList.add('selected');
    }

    document.getElementById('target-filename').innerText = filename;

    fetch(`/api/sample/${filename}`)
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                updateUIWithResult(data);
            } else {
                alert(`Error loading sample: ${data.message}`);
            }
        })
        .catch(err => {
            console.error('Error fetching sample:', err);
        });
}

/** Handle Custom CSV File Upload */
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    document.getElementById('target-filename').innerText = file.name;

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            updateUIWithResult(data);
        } else {
            alert(`Upload Error: ${data.message}`);
        }
    })
    .catch(err => {
        console.error('Error uploading file:', err);
        alert('Failed to upload and process CSV file.');
    });
}

/** Update KOI Catalog Slider Predictions */
function updateCatalogPrediction() {
    const depth = parseFloat(document.getElementById('slider-depth').value);
    const period = parseFloat(document.getElementById('slider-period').value);
    const prad = parseFloat(document.getElementById('slider-prad').value);
    const snr = parseFloat(document.getElementById('slider-snr').value);

    document.getElementById('lbl-depth').innerText = `${depth} ppm`;
    document.getElementById('lbl-period').innerText = `${period} days`;
    document.getElementById('lbl-prad').innerText = `${prad} R⊕`;
    document.getElementById('lbl-snr').innerText = `${snr}`;

    fetch('/api/predict_catalog', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            koi_depth: depth,
            koi_period: period,
            koi_prad: prad,
            koi_model_snr: snr
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            const banner = document.getElementById('result-banner');
            const badge = document.getElementById('result-badge');
            const text = document.getElementById('result-text');
            const conf = document.getElementById('result-confidence');

            if (data.is_exoplanet_candidate) {
                banner.className = 'result-banner banner-candidate';
                badge.className = 'result-badge badge-candidate';
                text.innerText = 'EXOPLANET CANDIDATE';
            } else {
                banner.className = 'result-banner banner-non-candidate';
                badge.className = 'result-badge badge-non-candidate';
                text.innerText = 'NON-EXOPLANET';
            }
            conf.innerText = `${(data.score * 100).toFixed(1)}%`;
            document.getElementById('explanation-box').innerText = 
                `KOI Catalog Parameters: Transit Depth=${depth} ppm, Radius=${prad} R⊕, Period=${period}d, SNR=${snr}. Score=${data.score.toFixed(2)}.`;
        }
    });
}

/** Update Dashboard UI with Prediction Outcome */
function updateUIWithResult(data) {
    // 1. Render Interactive Light Curve Plot
    if (data.time && data.flux) {
        renderLightCurveChart('lightcurve-chart', data.time, data.flux, data.filename || 'Kepler Target');
    }

    // 2. Update Classification Outcome Banner
    const banner = document.getElementById('result-banner');
    const badge = document.getElementById('result-badge');
    const text = document.getElementById('result-text');
    const conf = document.getElementById('result-confidence');

    if (data.is_exoplanet_candidate) {
        banner.className = 'result-banner banner-candidate';
        badge.className = 'result-badge badge-candidate';
        text.innerText = 'EXOPLANET CANDIDATE';
    } else {
        banner.className = 'result-banner banner-non-candidate';
        badge.className = 'result-badge badge-non-candidate';
        text.innerText = 'NON-EXOPLANET';
    }

    conf.innerText = `${(data.exoplanet_probability * 100).toFixed(1)}%`;

    // 3. Update Transit Key Feature Cards
    if (data.extracted_features) {
        const feat = data.extracted_features;
        document.getElementById('metric-snr').innerText = feat.dip_snr ? feat.dip_snr.toFixed(2) : 'N/A';
        document.getElementById('metric-depth').innerText = feat.max_dip_depth_ppm ? Math.round(feat.max_dip_depth_ppm).toLocaleString() : 'N/A';
        document.getElementById('metric-skew').innerText = feat.flux_skew ? feat.flux_skew.toFixed(2) : 'N/A';
        document.getElementById('metric-power').innerText = feat.lomb_scargle_max_power ? feat.lomb_scargle_max_power.toFixed(3) : 'N/A';
    }

    // 4. Update Astronomical Physical Explanation
    if (data.explanation) {
        document.getElementById('explanation-box').innerText = data.explanation;
    }
}

/** Fetch & Render Genuine Empirical Model Metrics Table */
function fetchBenchmarkMetrics() {
    fetch('/api/metrics')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                renderBenchmarkTable(data);
            } else {
                document.getElementById('benchmark-table-body').innerHTML = 
                    `<tr><td colspan="8" style="text-align:center; color:var(--status-warning);">Training metrics pending. Execute 'python train.py' to generate results.</td></tr>`;
            }
        })
        .catch(err => {
            console.error('Error fetching metrics:', err);
        });
}

/** Render Model Comparison Table */
function renderBenchmarkTable(metricsData) {
    const tbody = document.getElementById('benchmark-table-body');
    const bestModelName = metricsData.best_model_name;
    const testMetrics = metricsData.test_metrics;
    const comparisons = metricsData.model_comparison || {};

    let html = '';

    // Render best selected model first
    html += `
        <tr class="best-row">
            <td class="best-tag">★ ${bestModelName} (Selected Best)</td>
            <td>${(testMetrics.accuracy * 100).toFixed(2)}%</td>
            <td>${(testMetrics.precision * 100).toFixed(2)}%</td>
            <td>${(testMetrics.recall * 100).toFixed(2)}%</td>
            <td>${(testMetrics.f1_score * 100).toFixed(2)}%</td>
            <td>${(testMetrics.pr_auc * 100).toFixed(2)}%</td>
            <td>${(testMetrics.roc_auc * 100).toFixed(2)}%</td>
            <td><span style="color: var(--accent-cyan); font-weight:600;">Optimal</span></td>
        </tr>
    `;

    // Render comparison models (skipping bestModelName to avoid duplicate entry)
    for (const [modelName, m] of Object.entries(comparisons)) {
        if (modelName === bestModelName) continue;
        html += `
            <tr>
                <td style="color: var(--text-primary);">${modelName}</td>
                <td>${(m.accuracy * 100).toFixed(2)}%</td>
                <td>${(m.precision * 100).toFixed(2)}%</td>
                <td>${(m.recall * 100).toFixed(2)}%</td>
                <td>${(m.f1_score * 100).toFixed(2)}%</td>
                <td>${(m.pr_auc * 100).toFixed(2)}%</td>
                <td>${(m.roc_auc * 100).toFixed(2)}%</td>
                <td style="color: var(--text-muted);">Evaluated</td>
            </tr>
        `;
    }

    tbody.innerHTML = html;
}
