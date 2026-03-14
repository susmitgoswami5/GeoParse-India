/**
 * GeoParse-India — Frontend Application
 *
 * Handles API calls, entity rendering, Leaflet map integration,
 * and micro-animations for the address parsing UI.
 */

// ---- Configuration ----
const API_BASE = '';

// Sample addresses for demo
const SAMPLE_ADDRESSES = [
    "Behind the big banyan tree, near Sharma Medicals, 2nd cross, Whitefield, Bangalore 560066",
    "Flat 302, Sunshine Apts, opp SBI Bank, Koramangala, Bengaluru",
    "H.No. 45, near Hanuman Temple, MG Road, Connaught Place, Delhi 110001",
    "Plot 12, DLF Phase 3, Golf Course Road, Gurugram 122002",
    "D.No. 8-2-120, Banjara Hills, opp HDFC Bank ATM, Hyderabad 500034",
    "42/B Royal Towers, nr Metro Station, Andheri, Mumbai 400050",
    "#18 3rd Cross, behind Dominos Pizza, HSR Layout, Bnglr",
    "Flat 5A Ganesh Tower beside Apollo Pharmacy Velachery Chennai",
];

// Entity color map for visualization
const ENTITY_COLORS = {
    'HOUSE_NO': { bg: 'rgba(59, 130, 246, 0.2)', text: '#93c5fd', border: 'rgba(59, 130, 246, 0.5)', label: '🏠 House No' },
    'BUILDING': { bg: 'rgba(139, 92, 246, 0.2)', text: '#c4b5fd', border: 'rgba(139, 92, 246, 0.5)', label: '🏢 Building' },
    'STREET':   { bg: 'rgba(236, 72, 153, 0.2)', text: '#f9a8d4', border: 'rgba(236, 72, 153, 0.5)', label: '🛣️ Street' },
    'LANDMARK': { bg: 'rgba(245, 158, 11, 0.2)', text: '#fcd34d', border: 'rgba(245, 158, 11, 0.5)', label: '📍 Landmark' },
    'LOCALITY': { bg: 'rgba(16, 185, 129, 0.2)', text: '#6ee7b7', border: 'rgba(16, 185, 129, 0.5)', label: '📌 Locality' },
    'CITY':     { bg: 'rgba(99, 102, 241, 0.2)', text: '#a5b4fc', border: 'rgba(99, 102, 241, 0.5)', label: '🏙️ City' },
    'STATE':    { bg: 'rgba(168, 85, 247, 0.2)', text: '#d8b4fe', border: 'rgba(168, 85, 247, 0.5)', label: '🗺️ State' },
    'PINCODE':  { bg: 'rgba(244, 63, 94, 0.2)', text: '#fda4af', border: 'rgba(244, 63, 94, 0.5)', label: '📮 Pincode' },
};

// ---- State ----
let map = null;
let mapMarker = null;
let mapCircle = null;

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    initSamples();
    checkHealth();

    // Enter key to submit
    document.getElementById('address-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            geocodeAddress();
        }
    });
});

function initSamples() {
    const bar = document.getElementById('samples-bar');
    SAMPLE_ADDRESSES.forEach((addr, i) => {
        const chip = document.createElement('span');
        chip.className = 'sample-chip';
        // Show truncated version
        chip.textContent = addr.length > 50 ? addr.substring(0, 47) + '...' : addr;
        chip.title = addr;
        chip.onclick = () => {
            document.getElementById('address-input').value = addr;
            geocodeAddress();
        };
        bar.appendChild(chip);
    });
}

async function checkHealth() {
    try {
        const resp = await fetch(`${API_BASE}/api/health`);
        const data = await resp.json();
        const statusEl = document.getElementById('system-status');
        if (data.model_loaded) {
            statusEl.textContent = 'NER Model Active • Transformer Pipeline';
        } else {
            statusEl.textContent = 'Rule-Based Mode • Ready';
        }
    } catch (e) {
        document.getElementById('system-status').textContent = 'API Offline';
    }
}

// ---- Main Actions ----

async function geocodeAddress() {
    const input = document.getElementById('address-input').value.trim();
    if (!input) return;

    const btn = document.getElementById('btn-geocode');
    btn.classList.add('btn--loading');

    try {
        const resp = await fetch(`${API_BASE}/api/geocode`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ address: input }),
        });

        if (!resp.ok) {
            throw new Error(`API error: ${resp.status}`);
        }

        const data = await resp.json();
        renderResults(data);
    } catch (err) {
        console.error('Geocode error:', err);
        showError('Failed to process address. Please check that the server is running.');
    } finally {
        btn.classList.remove('btn--loading');
    }
}

function clearAll() {
    document.getElementById('address-input').value = '';
    document.getElementById('results-section').style.display = 'none';
    if (map) {
        map.remove();
        map = null;
    }
}

// ---- Rendering ----

function renderResults(data) {
    const section = document.getElementById('results-section');
    section.style.display = 'grid';

    renderTokens(data.token_predictions || []);
    renderEntities(data.entities || {});
    renderQuery(data.query || {});
    renderMap(data);
    renderGeocodeInfo(data);
    renderConfidence(data.confidence);
    renderReviewAlert(data);

    // Smooth scroll to results
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderTokens(predictions) {
    const flow = document.getElementById('token-flow');
    flow.innerHTML = '';

    if (predictions.length === 0) {
        flow.innerHTML = '<span style="color: var(--text-muted); font-size: 0.85rem;">No token predictions (rule-based mode)</span>';
        return;
    }

    const seenLabels = new Set();

    predictions.forEach((pred, i) => {
        const chip = document.createElement('span');
        const labelType = pred.label.replace(/^[BI]-/, '');
        chip.className = `token-chip label-${labelType}`;
        chip.textContent = pred.token;
        chip.title = `${pred.label} (${(pred.confidence * 100).toFixed(1)}%)`;
        chip.style.animationDelay = `${i * 30}ms`;
        flow.appendChild(chip);
        if (labelType !== 'O') seenLabels.add(labelType);
    });

    // Render legend
    const legend = document.getElementById('token-legend');
    legend.innerHTML = '';
    seenLabels.forEach(label => {
        const colors = ENTITY_COLORS[label];
        if (!colors) return;
        const item = document.createElement('span');
        item.className = 'token-legend__item';
        item.innerHTML = `<span class="token-legend__color" style="background:${colors.border}"></span>${colors.label}`;
        legend.appendChild(item);
    });
}

function renderEntities(entities) {
    const grid = document.getElementById('entity-grid');
    grid.innerHTML = '';

    const entityTypes = Object.keys(entities);
    if (entityTypes.length === 0) {
        grid.innerHTML = '<div class="empty-state"><span class="empty-state__icon">🔍</span><span class="empty-state__text">No entities extracted</span></div>';
        return;
    }

    // Render in priority order
    const order = ['HOUSE_NO', 'BUILDING', 'STREET', 'LANDMARK', 'LOCALITY', 'CITY', 'STATE', 'PINCODE'];

    order.forEach(type => {
        if (!(type in entities)) return;
        const entity = entities[type];
        const colors = ENTITY_COLORS[type] || {};
        const conf = entity.confidence || entity.correction_confidence || 0;

        const tag = document.createElement('div');
        tag.className = 'entity-tag';
        tag.style.borderLeftColor = colors.border || 'var(--accent-primary)';
        tag.style.animationDelay = `${order.indexOf(type) * 60}ms`;

        let correctedBadge = '';
        if (entity.was_corrected && entity.original) {
            correctedBadge = `<span class="entity-tag__corrected" title="Corrected from: ${entity.original}">← ${entity.original}</span>`;
        }

        const confClass = conf >= 0.8 ? 'conf-high' : conf >= 0.5 ? 'conf-medium' : 'conf-low';

        tag.innerHTML = `
            <span class="entity-tag__type" style="color: ${colors.text || 'var(--accent-tertiary)'}">${type}</span>
            <span class="entity-tag__value">${entity.text}${correctedBadge}</span>
            <span class="entity-tag__confidence ${confClass}">${(conf * 100).toFixed(0)}%</span>
        `;

        grid.appendChild(tag);
    });
}

function renderQuery(query) {
    const display = document.getElementById('query-display');
    if (query.query_string) {
        display.textContent = query.query_string;
        document.getElementById('query-section').style.display = 'block';
    } else {
        document.getElementById('query-section').style.display = 'none';
    }
}

function renderMap(data) {
    const container = document.getElementById('map-container');
    const placeholder = document.getElementById('map-placeholder');

    if (!data.lat || !data.lng) {
        if (map) { map.remove(); map = null; }
        placeholder.style.display = 'flex';
        return;
    }

    placeholder.style.display = 'none';

    // Initialize or update map
    if (map) {
        map.remove();
    }

    map = L.map('map-container', {
        zoomControl: true,
        attributionControl: true,
    }).setView([data.lat, data.lng], getZoomLevel(data.radius_m));

    // Dark tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        maxZoom: 19,
    }).addTo(map);

    // Confidence circle
    if (data.radius_m) {
        mapCircle = L.circle([data.lat, data.lng], {
            radius: data.radius_m,
            color: '#6366f1',
            fillColor: '#6366f1',
            fillOpacity: 0.15,
            weight: 2,
            dashArray: '6, 4',
        }).addTo(map);
    }

    // Marker
    const markerIcon = L.divIcon({
        className: '',
        html: `<div style="
            width: 20px; height: 20px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border: 3px solid white;
            border-radius: 50%;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.5);
        "></div>`,
        iconSize: [20, 20],
        iconAnchor: [10, 10],
    });

    mapMarker = L.marker([data.lat, data.lng], { icon: markerIcon })
        .bindPopup(`
            <div style="font-family: 'Inter', sans-serif; font-size: 13px; line-height: 1.5;">
                <strong>${data.entities?.LOCALITY?.text || data.entities?.CITY?.text || 'Location'}</strong><br>
                <span style="color: #666;">${data.lat.toFixed(4)}, ${data.lng.toFixed(4)}</span><br>
                <span style="color: #666;">Radius: ${data.radius_m}m</span>
            </div>
        `)
        .addTo(map);

    // Fit bounds if circle exists
    if (mapCircle) {
        map.fitBounds(mapCircle.getBounds().pad(0.3));
    }
}

function getZoomLevel(radius) {
    if (!radius) return 14;
    if (radius < 500) return 16;
    if (radius < 1000) return 15;
    if (radius < 2000) return 14;
    if (radius < 5000) return 13;
    return 11;
}

function renderGeocodeInfo(data) {
    const info = document.getElementById('geocode-info');
    info.innerHTML = '';

    const items = [
        { label: 'Latitude', value: data.lat ? data.lat.toFixed(6) : '—' },
        { label: 'Longitude', value: data.lng ? data.lng.toFixed(6) : '—' },
        { label: 'Radius', value: data.radius_m ? `${data.radius_m}m` : '—' },
        { label: 'Resolution', value: data.resolution || '—' },
    ];

    if (data.h3_cell) {
        items.push({ label: 'H3 Cell', value: data.h3_cell.substring(0, 12) + '...' });
    }

    items.push({
        label: 'Entities Found',
        value: data.query?.num_entities || Object.keys(data.entities || {}).length
    });

    items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'geocode-info__item';
        div.innerHTML = `
            <div class="geocode-info__label">${item.label}</div>
            <div class="geocode-info__value">${item.value}</div>
        `;
        info.appendChild(div);
    });
}

function renderConfidence(confidence) {
    const meter = document.getElementById('confidence-meter');
    const fill = document.getElementById('confidence-fill');
    const value = document.getElementById('confidence-value');

    if (confidence === undefined || confidence === null) {
        meter.style.display = 'none';
        return;
    }

    meter.style.display = 'block';
    const pct = Math.round(confidence * 100);
    value.textContent = `${pct}%`;

    // Animate the fill
    setTimeout(() => {
        fill.style.width = `${pct}%`;
        fill.className = 'confidence-meter__fill';
        if (pct >= 70) fill.classList.add('high');
        else if (pct >= 40) fill.classList.add('medium');
        else fill.classList.add('low');
    }, 100);
}

function renderReviewAlert(data) {
    const alert = document.getElementById('review-alert');

    if (data.needs_review) {
        alert.style.display = 'flex';
        alert.className = 'review-alert';
        alert.innerHTML = `
            <span class="review-alert__icon">⚠️</span>
            <div>
                <strong>Manual Review Recommended</strong><br>
                <span style="font-size: 0.8rem; opacity: 0.85;">${data.review_reason || 'Address may be ambiguous'}</span>
            </div>
        `;
    } else {
        alert.style.display = 'none';
    }
}

function showError(message) {
    const section = document.getElementById('results-section');
    section.style.display = 'grid';
    section.innerHTML = `
        <div class="glass-card" style="grid-column: 1 / -1;">
            <div class="review-alert" style="background: var(--error-bg); border-color: rgba(239, 68, 68, 0.3); color: var(--error);">
                <span class="review-alert__icon">❌</span>
                <div>${message}</div>
            </div>
        </div>
    `;
}
