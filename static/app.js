/* ══════════════════════════════════════════════════════════════
   RL Agentic RAG — Frontend Application Logic
   ══════════════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;

// ── State ────────────────────────────────────────────────────
const state = {
    sessionId: 'default',
    sessions: {},
    isLoading: false,
    metricsVisible: false,
    lastMetrics: null,
};

// ── DOM Elements ─────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
    queryInput: $('#queryInput'),
    btnSend: $('#btnSend'),
    chatArea: $('#chatArea'),
    welcomeScreen: $('#welcomeScreen'),
    messagesContainer: $('#messagesContainer'),
    sessionList: $('#sessionList'),
    sessionBadge: $('#sessionBadge'),
    statusDot: $('#statusDot'),
    statusText: $('#statusText'),
    statusModel: $('#statusModel'),
    uploadZone: $('#uploadZone'),
    fileInput: $('#fileInput'),
    uploadStatus: $('#uploadStatus'),
    metricsPanel: $('#metricsPanel'),
    btnCloseMetrics: $('#btnCloseMetrics'),
    headerTitle: $('#headerTitle'),
    headerMetrics: $('#headerMetrics'),
    sidebar: $('#sidebar'),
    mobileMenuBtn: $('#mobileMenuBtn'),
    btnNewChat: $('#btnNewChat'),
    pipelineSteps: $('#pipelineSteps'),
    // Metrics
    gaugeConfidence: $('#gaugeConfidence'),
    gaugeValue: $('#gaugeValue'),
    rewardBar: $('#rewardBar'),
    rewardValue: $('#rewardValue'),
    retryDots: $('#retryDots'),
    retryValue: $('#retryValue'),
    citationList: $('#citationList'),
    routeBadge: $('#routeBadge'),
};

// ── Initialize ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
    loadSessions();
    setInterval(checkHealth, 30000); // Check health every 30s
});

// ── Health Check ─────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();
        els.statusDot.className = 'status-dot online';
        els.statusText.textContent = 'Online';
        els.statusModel.textContent = data.model || 'Unknown';
    } catch {
        els.statusDot.className = 'status-dot offline';
        els.statusText.textContent = 'Offline';
        els.statusModel.textContent = 'Server not reachable';
    }
}

// ── Event Listeners ──────────────────────────────────────────
function setupEventListeners() {
    // Send query
    els.btnSend.addEventListener('click', sendQuery);
    els.queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });

    // Enable/disable send button
    els.queryInput.addEventListener('input', () => {
        els.btnSend.disabled = !els.queryInput.value.trim();
        autoResizeTextarea();
    });

    // Suggestion chips
    $$('.suggestion-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            els.queryInput.value = chip.dataset.query;
            els.btnSend.disabled = false;
            sendQuery();
        });
    });

    // File upload
    els.uploadZone.addEventListener('click', () => els.fileInput.click());
    els.fileInput.addEventListener('change', handleFileUpload);

    // Drag and drop
    els.uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        els.uploadZone.classList.add('drag-over');
    });
    els.uploadZone.addEventListener('dragleave', () => {
        els.uploadZone.classList.remove('drag-over');
    });
    els.uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        els.uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            els.fileInput.files = e.dataTransfer.files;
            handleFileUpload();
        }
    });

    // Metrics panel
    els.btnCloseMetrics.addEventListener('click', toggleMetrics);

    // New chat
    els.btnNewChat.addEventListener('click', createNewSession);

    // Mobile menu
    els.mobileMenuBtn.addEventListener('click', () => {
        els.sidebar.classList.toggle('open');
        toggleOverlay();
    });
}

// ── Auto-resize textarea ─────────────────────────────────────
function autoResizeTextarea() {
    els.queryInput.style.height = 'auto';
    els.queryInput.style.height = Math.min(els.queryInput.scrollHeight, 120) + 'px';
}

// ── Send Query ───────────────────────────────────────────────
async function sendQuery() {
    const query = els.queryInput.value.trim();
    if (!query || state.isLoading) return;

    state.isLoading = true;
    els.btnSend.classList.add('loading');
    els.btnSend.disabled = true;

    // Hide welcome, show messages
    els.welcomeScreen.style.display = 'none';
    els.messagesContainer.style.display = 'flex';

    // Add user message
    addMessage('user', query);
    els.queryInput.value = '';
    els.queryInput.style.height = 'auto';

    // Show typing indicator
    const typingEl = addTypingIndicator();

    // Animate pipeline
    animatePipeline();

    try {
        const res = await fetch(`${API_BASE}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, session_id: state.sessionId }),
        });

        const data = await res.json();

        // Remove typing indicator
        typingEl.remove();

        // Add assistant message
        addMessage('assistant', data.answer, {
            citations: data.citations,
            confidence: data.confidence,
            reward: data.reward,
            retry_count: data.retry_count,
        });

        // Update metrics
        updateMetrics(data);

        // Save to session
        saveToSession(query, data);

        // Update header
        updateHeader(data);

        // Complete pipeline animation
        completePipeline();

    } catch (err) {
        typingEl.remove();
        addMessage('assistant', `⚠️ Error: ${err.message}. Is the server running?`);
        resetPipeline();
    } finally {
        state.isLoading = false;
        els.btnSend.classList.remove('loading');
        els.btnSend.disabled = !els.queryInput.value.trim();
    }
}

// ── Add Message ──────────────────────────────────────────────
function addMessage(role, content, meta = null) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const avatar = role === 'user' ? 'U' : 'AI';
    const author = role === 'user' ? 'You' : 'Agentic RAG';

    let citationsHtml = '';
    if (meta?.citations?.length) {
        citationsHtml = `
            <div class="message-citations">
                ${meta.citations.map(c => `
                    <span class="citation-tag">
                        <svg viewBox="0 0 12 12" fill="none"><path d="M4 6h4M6 4v4" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>
                        ${escapeHtml(c)}
                    </span>
                `).join('')}
            </div>
        `;
    }

    let metricsHtml = '';
    if (meta) {
        metricsHtml = `
            <div class="message-metrics-bar">
                <span class="msg-metric">Confidence: <span class="val">${(meta.confidence * 100).toFixed(1)}%</span></span>
                <span class="msg-metric">Reward: <span class="val">${meta.reward.toFixed(3)}</span></span>
                ${meta.retry_count > 0 ? `<span class="msg-metric">Retries: <span class="val">${meta.retry_count}</span></span>` : ''}
            </div>
        `;
    }

    // Convert answer text to paragraphs
    const formattedContent = formatAnswer(content);

    msg.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-header">
                <span class="message-author">${author}</span>
                <span class="message-time">${time}</span>
            </div>
            <div class="message-body">
                ${formattedContent}
                ${citationsHtml}
                ${metricsHtml}
            </div>
        </div>
    `;

    els.messagesContainer.appendChild(msg);
    scrollToBottom();
    return msg;
}

// ── Format Answer ────────────────────────────────────────────
function formatAnswer(text) {
    // Convert markdown-style formatting to HTML
    let html = escapeHtml(text);
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Inline code
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');
    // Citations [source]
    html = html.replace(/\[([^\]]+)\]/g, '<strong>[$1]</strong>');
    // Paragraphs
    html = html.split('\n\n').map(p => `<p>${p}</p>`).join('');
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    return html;
}

// ── Add Typing Indicator ─────────────────────────────────────
function addTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="message-header">
                <span class="message-author">Agentic RAG</span>
                <span class="message-time">thinking...</span>
            </div>
            <div class="message-body">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;
    els.messagesContainer.appendChild(el);
    scrollToBottom();
    return el;
}

// ── Pipeline Animation ───────────────────────────────────────
function animatePipeline() {
    const steps = els.pipelineSteps.querySelectorAll('.pipeline-step');
    const connectors = els.pipelineSteps.querySelectorAll('.pipeline-connector');

    // Reset
    steps.forEach(s => s.classList.remove('active', 'done'));
    connectors.forEach(c => c.classList.remove('active', 'done'));

    // Animate through steps
    const stepNames = ['decompose', 'retrieve', 'rerank', 'generate', 'verify'];
    let delay = 0;
    stepNames.forEach((name, i) => {
        setTimeout(() => {
            // Mark previous as done
            if (i > 0) {
                steps[i - 1].classList.remove('active');
                steps[i - 1].classList.add('done');
                if (connectors[i - 1]) {
                    connectors[i - 1].classList.remove('active');
                    connectors[i - 1].classList.add('done');
                }
            }
            steps[i].classList.add('active');
            if (connectors[i]) connectors[i].classList.add('active');
        }, delay);
        delay += 800;
    });
}

function completePipeline() {
    const steps = els.pipelineSteps.querySelectorAll('.pipeline-step');
    const connectors = els.pipelineSteps.querySelectorAll('.pipeline-connector');
    steps.forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
    connectors.forEach(c => { c.classList.remove('active'); c.classList.add('done'); });
}

function resetPipeline() {
    const steps = els.pipelineSteps.querySelectorAll('.pipeline-step');
    const connectors = els.pipelineSteps.querySelectorAll('.pipeline-connector');
    steps.forEach(s => s.classList.remove('active', 'done'));
    connectors.forEach(c => c.classList.remove('active', 'done'));
}

// ── Update Metrics ───────────────────────────────────────────
function updateMetrics(data) {
    state.lastMetrics = data;

    // Show metrics panel
    if (!state.metricsVisible) toggleMetrics();

    // Confidence gauge
    const conf = data.confidence || 0;
    const circumference = 326.7; // 2 * PI * 52
    const offset = circumference - (conf * circumference);
    els.gaugeConfidence.style.strokeDashoffset = offset;
    els.gaugeValue.textContent = `${(conf * 100).toFixed(0)}%`;

    // Add gradient def to gauge SVG if not exists
    const gaugeSvg = els.gaugeConfidence.closest('svg');
    if (!gaugeSvg.querySelector('#gaugeGradient')) {
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        defs.innerHTML = `
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#818cf8"/>
                <stop offset="100%" stop-color="#c084fc"/>
            </linearGradient>
        `;
        gaugeSvg.prepend(defs);
        els.gaugeConfidence.style.stroke = 'url(#gaugeGradient)';
    }

    // Reward bar
    const reward = data.reward || 0;
    els.rewardBar.style.width = `${reward * 100}%`;
    els.rewardValue.textContent = reward.toFixed(3);

    // Retry dots
    const retryCount = data.retry_count || 0;
    const dots = els.retryDots.querySelectorAll('.retry-dot');
    dots.forEach((dot, i) => {
        dot.classList.toggle('used', i < retryCount);
    });
    els.retryValue.textContent = `${retryCount} / 3`;

    // Citations
    const citations = data.citations || [];
    if (citations.length) {
        els.citationList.innerHTML = citations.map(c => `
            <div class="citation-item">
                <svg viewBox="0 0 12 12" fill="none"><path d="M2 6h8M6 2v8" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>
                ${escapeHtml(c)}
            </div>
        `).join('');
    } else {
        els.citationList.innerHTML = '<span class="no-data">No citations</span>';
    }

    // Route badge
    els.routeBadge.textContent = data.route || 'rag';
    els.routeBadge.className = `route-badge ${data.route || 'rag'}`;
}

// ── Update Header ────────────────────────────────────────────
function updateHeader(data) {
    els.headerMetrics.innerHTML = `
        <span class="header-metric">Confidence: <span class="value">${((data.confidence || 0) * 100).toFixed(1)}%</span></span>
        <span class="header-metric">Reward: <span class="value">${(data.reward || 0).toFixed(3)}</span></span>
    `;
}

// ── Toggle Metrics Panel ─────────────────────────────────────
function toggleMetrics() {
    state.metricsVisible = !state.metricsVisible;
    els.metricsPanel.classList.toggle('visible', state.metricsVisible);
}

// ── File Upload ──────────────────────────────────────────────
async function handleFileUpload() {
    const files = els.fileInput.files;
    if (!files.length) return;

    els.uploadStatus.textContent = `Uploading ${files.length} file(s)...`;
    els.uploadStatus.style.color = 'var(--accent-primary)';

    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }

    try {
        const res = await fetch(`${API_BASE}/ingest`, {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        els.uploadStatus.textContent = `✓ ${data.files_processed} files, ${data.chunks_created} chunks`;
        els.uploadStatus.style.color = 'var(--success)';
    } catch (err) {
        els.uploadStatus.textContent = `✗ Upload failed: ${err.message}`;
        els.uploadStatus.style.color = 'var(--error)';
    }

    // Reset file input
    els.fileInput.value = '';
}

// ── Session Management ───────────────────────────────────────
function createNewSession() {
    const id = `session_${Date.now()}`;
    state.sessionId = id;
    state.sessions[id] = { name: `Chat ${Object.keys(state.sessions).length + 1}`, messages: [] };
    
    // Clear chat
    els.messagesContainer.innerHTML = '';
    els.welcomeScreen.style.display = 'flex';
    els.messagesContainer.style.display = 'none';
    els.headerTitle.textContent = state.sessions[id].name;
    els.sessionBadge.textContent = `Session: ${id.slice(0, 12)}...`;
    els.headerMetrics.innerHTML = '';
    
    // Reset metrics
    resetMetrics();
    
    renderSessions();
    saveSessions();

    // Close mobile menu
    els.sidebar.classList.remove('open');
    removeOverlay();
}

function switchSession(id) {
    state.sessionId = id;
    const session = state.sessions[id];
    if (!session) return;

    // Clear and replay messages
    els.messagesContainer.innerHTML = '';

    if (session.messages.length === 0) {
        els.welcomeScreen.style.display = 'flex';
        els.messagesContainer.style.display = 'none';
    } else {
        els.welcomeScreen.style.display = 'none';
        els.messagesContainer.style.display = 'flex';
        session.messages.forEach(m => {
            addMessage(m.role, m.content, m.meta);
        });
    }

    els.headerTitle.textContent = session.name;
    els.sessionBadge.textContent = `Session: ${id.slice(0, 12)}...`;
    renderSessions();

    // Close mobile menu
    els.sidebar.classList.remove('open');
    removeOverlay();
}

function saveToSession(query, data) {
    if (!state.sessions[state.sessionId]) {
        state.sessions[state.sessionId] = { name: query.slice(0, 30) + '...', messages: [] };
    }

    const session = state.sessions[state.sessionId];
    
    // Update name from first query
    if (session.messages.length === 0) {
        session.name = query.slice(0, 35) + (query.length > 35 ? '...' : '');
        els.headerTitle.textContent = session.name;
    }

    session.messages.push({ role: 'user', content: query });
    session.messages.push({
        role: 'assistant',
        content: data.answer,
        meta: {
            citations: data.citations,
            confidence: data.confidence,
            reward: data.reward,
            retry_count: data.retry_count,
        }
    });

    renderSessions();
    saveSessions();
}

function renderSessions() {
    const ids = Object.keys(state.sessions);
    if (ids.length === 0) {
        els.sessionList.innerHTML = '<div class="no-data" style="padding: 8px;">No sessions yet</div>';
        return;
    }

    els.sessionList.innerHTML = ids.map(id => {
        const s = state.sessions[id];
        const isActive = id === state.sessionId;
        const count = Math.floor((s.messages?.length || 0) / 2);
        return `
            <div class="session-item ${isActive ? 'active' : ''}" onclick="switchSession('${id}')">
                <span class="session-icon">💬</span>
                <span class="session-name">${escapeHtml(s.name)}</span>
                <span class="session-count">${count}</span>
            </div>
        `;
    }).join('');
}

function saveSessions() {
    try {
        localStorage.setItem('rag_sessions', JSON.stringify(state.sessions));
        localStorage.setItem('rag_current_session', state.sessionId);
    } catch { /* localStorage full or unavailable */ }
}

function loadSessions() {
    try {
        const saved = localStorage.getItem('rag_sessions');
        const currentId = localStorage.getItem('rag_current_session');
        if (saved) {
            state.sessions = JSON.parse(saved);
            if (currentId && state.sessions[currentId]) {
                switchSession(currentId);
            }
        }
    } catch { /* ignore */ }
    renderSessions();
}

// ── Reset Metrics ────────────────────────────────────────────
function resetMetrics() {
    els.gaugeConfidence.style.strokeDashoffset = 326.7;
    els.gaugeValue.textContent = '—';
    els.rewardBar.style.width = '0%';
    els.rewardValue.textContent = '—';
    els.retryDots.querySelectorAll('.retry-dot').forEach(d => d.classList.remove('used'));
    els.retryValue.textContent = '0 / 3';
    els.citationList.innerHTML = '<span class="no-data">No citations yet</span>';
    els.routeBadge.textContent = '—';
    els.routeBadge.className = 'route-badge';

    if (state.metricsVisible) toggleMetrics();
}

// ── Utilities ────────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    setTimeout(() => {
        els.chatArea.scrollTop = els.chatArea.scrollHeight;
    }, 50);
}

// ── Mobile overlay ───────────────────────────────────────────
function toggleOverlay() {
    let overlay = document.querySelector('.sidebar-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';
        overlay.addEventListener('click', () => {
            els.sidebar.classList.remove('open');
            removeOverlay();
        });
        document.body.appendChild(overlay);
    }
    overlay.classList.toggle('visible', els.sidebar.classList.contains('open'));
}

function removeOverlay() {
    const overlay = document.querySelector('.sidebar-overlay');
    if (overlay) overlay.classList.remove('visible');
}

// ── Expose to global scope for inline handlers ───────────────
window.switchSession = switchSession;
