/**
 * Nhatrovn Assistant Frontend
 * WebSocket + SQLite Chat History
 */

let ws = null;
let wsReconnectAttempt = 0;
const MAX_RECONNECT = 5;

let isProcessing = false;
let chatHistory = [];
let currentSessionId = null;
let currentStreamingEl = null;
let currentStreamingText = '';

document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    loadSessions();

    const input = document.getElementById('queryInput');
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    });
});

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/chat`;
    try { ws = new WebSocket(wsUrl); } catch { return; }

    ws.onopen = () => {
        wsReconnectAttempt = 0;
        updateStatus('connected');
    };

    ws.onmessage = (e) => {
        try { handleWsMessage(JSON.parse(e.data)); } catch {}
    };

    ws.onclose = () => {
        updateStatus('disconnected');
        if (wsReconnectAttempt < MAX_RECONNECT) {
            const delay = Math.min(1000 * 2 ** wsReconnectAttempt, 30000);
            wsReconnectAttempt++;
            setTimeout(connectWebSocket, delay);
        }
    };

    ws.onerror = () => {};
}

function updateStatus(status) {
    ['topNavStatus', 'topNavStatusMobile'].forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        const dot = el.querySelector('span:first-child');
        const txt = el.querySelector('span:last-child');
        if (status === 'connected') {
            if (dot) dot.className = 'w-2 h-2 rounded-full bg-tertiary animate-pulse';
            if (txt) txt.textContent = 'Nhatrovn Online';
        } else {
            if (dot) dot.className = 'w-2 h-2 rounded-full bg-amber-400 animate-pulse';
            if (txt) txt.textContent = 'Reconnecting...';
        }
    });
}

function handleWsMessage(data) {
    if (data.type === 'status') {
        updateTypingText(data.message || 'Đang xử lý...');
    } else if (data.type === 'stream') {
        removeTyping();
        appendStreamToken(data.token || '');
    } else if (data.type === 'answer') {
        removeTyping();
        if (data.session_id) currentSessionId = data.session_id;
        finalizeStream(data.answer, data.sources || [], data.agent_trace, data.processing_time_ms || 0, data);
        isProcessing = false;
        setSubmitEnabled(true);
        chatHistory.push({ role: 'assistant', content: data.answer });
        loadSessions();
    } else if (data.type === 'error') {
        removeTyping();
        clearStream();
        isProcessing = false;
        setSubmitEnabled(true);
        addAIMessage(data.message || 'Lỗi không xác định', [], null, 0);
    }
}

function appendStreamToken(token) {
    if (!currentStreamingEl) {
        const area = document.getElementById('messagesArea');
        const div = document.createElement('div');
        div.id = 'streaming-msg';
        div.className = 'flex flex-col items-start group';
        div.innerHTML = `
            <div class="flex items-center gap-3 mb-2">
                <span class="text-xs font-semibold text-primary">Nhatrovn Assistant</span>
            </div>
            <div class="bg-surface-container-low p-6 rounded-2xl rounded-tl-none border border-primary/5 text-on-surface max-w-[90%] shadow-2xl relative overflow-hidden">
                <div class="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-tertiary to-transparent opacity-40"></div>
                <div class="streaming-text prose text-base leading-relaxed"></div>
                <span class="inline-block w-2 h-4 bg-primary animate-pulse ml-1"></span>
            </div>`;
        area.appendChild(div);
        currentStreamingEl = div.querySelector('.streaming-text');
        currentStreamingText = '';
        scrollToBottom();
    }
    currentStreamingText += token;
    currentStreamingEl.innerHTML = formatMarkdown(currentStreamingText);
    scrollToBottom();
}

function finalizeStream(answer, sources, trace, timeMs, payload) {
    const el = document.getElementById('streaming-msg');
    if (el) el.remove();
    clearStream();
    addAIMessage(answer, sources, trace, timeMs, 'Nhatrovn Assistant', payload || {});
}

function clearStream() {
    currentStreamingEl = null;
    currentStreamingText = '';
}

async function handleSubmit(e) {
    if (e && e.preventDefault) e.preventDefault();
    if (isProcessing) return;

    const input = document.getElementById('queryInput');
    const q = input.value.trim();
    if (!q) return;

    showChatView();
    addUserMessage(q);
    chatHistory.push({ role: 'user', content: q });
    input.value = '';
    submitAssistant(q);
}

function submitAssistant(q) {
    showTyping();
    isProcessing = true;
    setSubmitEnabled(false);
    clearStream();

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            question: q,
            session_id: currentSessionId,
            top_k: 5,
            history: chatHistory.slice(-8)
        }));
    } else {
        removeTyping();
        isProcessing = false;
        setSubmitEnabled(true);
        addAIMessage('Không thể kết nối server. Kiểm tra Docker, Rust gateway và Python workers.', [], null, 0);
    }
}

function fillExample(text) {
    document.getElementById('queryInput').value = text;
    handleSubmit();
}

function showChatView() {
    document.getElementById('welcomeView').classList.add('hidden');
    document.getElementById('messagesArea').classList.remove('hidden');
}

function showWelcomeView() {
    document.getElementById('welcomeView').classList.remove('hidden');
    const area = document.getElementById('messagesArea');
    area.classList.add('hidden');
    area.innerHTML = '';
    chatHistory = [];
    currentSessionId = null;
}

function newResearch() {
    showWelcomeView();
}

async function loadSessions() {
    try {
        const res = await fetch('/api/sessions');
        const data = await res.json();
        renderSessionList(data.sessions || []);
    } catch {}
}

function renderSessionList(sessions) {
    const container = document.getElementById('sessionList');
    container.innerHTML = '<p class="px-4 text-[11px] font-bold text-on-surface-variant/50 tracking-tighter uppercase mb-2">Recent Sessions</p>';

    sessions.forEach(s => {
        const div = document.createElement('div');
        div.className = 'session-item flex items-center justify-between px-4 py-2 rounded-lg hover:bg-surface-variant cursor-pointer transition-all';
        div.innerHTML = `
            <span class="text-xs text-on-surface-variant hover:text-primary transition-colors truncate flex-1" title="${esc(s.title)}">${esc(s.title || 'Untitled')}</span>
            <button onclick="event.stopPropagation(); deleteSession('${s.id}')" class="session-delete p-1 text-on-surface-variant hover:text-error transition-colors rounded">
                <span class="material-symbols-outlined text-[14px]">delete</span>
            </button>`;
        div.addEventListener('click', () => loadSession(s.id));
        container.appendChild(div);
    });
}

async function loadSession(sessionId) {
    try {
        const res = await fetch(`/api/sessions/${sessionId}/messages`);
        const data = await res.json();

        currentSessionId = sessionId;
        chatHistory = [];
        showChatView();
        document.getElementById('messagesArea').innerHTML = '';

        (data.messages || []).forEach(msg => {
            chatHistory.push({ role: msg.role, content: msg.content });
            if (msg.role === 'user') addUserMessage(msg.content);
            else addAIMessage(msg.content, [], null, 0);
        });
    } catch (e) {
        console.error('Load session failed:', e);
    }
}

async function deleteSession(sessionId) {
    try {
        await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        loadSessions();
        if (currentSessionId === sessionId) showWelcomeView();
    } catch {}
}

function toggleHistoryPanel() {
    loadSessions();
}

function addUserMessage(text) {
    const area = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.className = 'flex flex-col items-end group';
    div.innerHTML = `
        <div class="flex items-center gap-3 mb-2">
            <span class="text-xs font-semibold text-secondary">Bạn</span>
        </div>
        <div class="glass p-5 rounded-2xl rounded-tr-none border border-outline-variant/10 text-on-surface max-w-[80%] shadow-xl">
            <p class="text-base leading-relaxed">${esc(text)}</p>
        </div>`;
    area.appendChild(div);
    scrollToBottom();
}

function addAIMessage(answer, sources, trace, timeMs, label, payload) {
    const area = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.className = 'flex flex-col items-start group';

    const data = payload || {};
    const senderName = label || 'Nhatrovn Assistant';
    const traceId = 'trace-' + Date.now();
    const listingsHtml = renderListings(data.listings || []);
    const sourcesHtml = renderSources(sources || []);
    const suggestionsHtml = renderSuggestions(data.suggested_questions || []);

    const badgesHtml = `
        <div class="flex flex-wrap items-center gap-3 pt-3">
            <div class="px-3 py-1.5 rounded-full bg-secondary-container/30 border border-secondary/20 flex items-center gap-2">
                <span class="material-symbols-outlined text-[14px] text-secondary">travel_explore</span>
                <span class="text-[10px] font-bold text-secondary uppercase tracking-tight">${esc(data.intent || 'READ_ONLY')}</span>
            </div>
            ${timeMs > 0 ? `
                <div class="px-3 py-1.5 rounded-full bg-surface-variant flex items-center gap-2">
                    <span class="material-symbols-outlined text-[14px] text-tertiary">timer</span>
                    <span class="text-[10px] font-medium text-on-surface-variant">${timeMs}ms</span>
                </div>` : ''}
            ${trace ? `
                <button onclick="showTrace('${traceId}')" class="px-3 py-1.5 rounded-full bg-primary/10 hover:bg-primary/20 border border-primary/20 text-primary transition-all flex items-center gap-2 group/btn">
                    <span class="material-symbols-outlined text-[14px]">account_tree</span>
                    <span class="text-[10px] font-bold uppercase tracking-tight">Trace</span>
                </button>` : ''}
        </div>`;

    div.innerHTML = `
        <div class="flex items-center gap-3 mb-2">
            <span class="text-xs font-semibold text-primary">${esc(senderName)}</span>
        </div>
        <div class="bg-surface-container-low p-6 rounded-2xl rounded-tl-none border border-primary/5 text-on-surface max-w-[90%] shadow-2xl relative overflow-hidden">
            <div class="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-tertiary to-transparent opacity-40"></div>
            <div class="text-base leading-relaxed text-on-surface/90">${formatMarkdown(answer)}</div>
            ${listingsHtml}
            ${suggestionsHtml}
            ${badgesHtml}
        </div>
        ${sourcesHtml}
        ${trace ? `<div id="${traceId}" class="hidden">${esc(JSON.stringify(trace))}</div>` : ''}`;

    area.appendChild(div);
    scrollToBottom();
}

function renderListings(listings) {
    if (!listings.length) return '';
    return `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 mt-5">
            ${listings.slice(0, 4).map(l => `
                <div class="rounded-lg border border-outline-variant/20 bg-white/70 p-4">
                    <div class="text-sm font-semibold text-on-surface">${esc(l.title || l.listing_id || 'Listing')}</div>
                    <div class="mt-2 text-xs text-on-surface-variant">${esc(l.district || l.address || 'Khu vực chưa rõ')}</div>
                    <div class="mt-3 flex items-center justify-between text-xs">
                        <span class="font-mono text-primary">#${esc(l.listing_id || '')}</span>
                        <span class="font-semibold">${formatVnd(l.rent_price)}</span>
                    </div>
                </div>`).join('')}
        </div>`;
}

function renderSources(sources) {
    if (!sources.length) return '';
    return `
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">
            ${sources.slice(0, 3).map(s => `
                <div class="glass p-4 rounded-xl border border-outline-variant/10 hover:border-primary/20 transition-all">
                    <div class="text-[10px] font-bold text-primary uppercase mb-1">Nguồn listing</div>
                    <div class="text-xs font-semibold text-on-surface leading-tight">${esc(s.title || s.listing_id || 'Listing')}</div>
                    ${s.source_version !== undefined ? `<div class="text-[10px] text-on-surface-variant mt-1 font-mono">v${esc(String(s.source_version))}</div>` : ''}
                </div>`).join('')}
        </div>`;
}

function renderSuggestions(questions) {
    if (!questions.length) return '';
    return `
        <div class="flex flex-wrap gap-2 mt-5">
            ${questions.slice(0, 3).map(q => `
                <button type="button" onclick="fillExample('${escAttr(q)}')" class="px-3 py-1.5 rounded-full bg-surface-variant text-xs text-on-surface-variant hover:text-primary hover:bg-primary-container/20 transition-colors">${esc(q)}</button>
            `).join('')}
        </div>`;
}

function showTyping() {
    removeTyping();
    const area = document.getElementById('messagesArea');
    const div = document.createElement('div');
    div.id = 'typing-indicator';
    div.className = 'flex flex-col items-start';
    div.innerHTML = `
        <div class="flex items-center gap-3 mb-2">
            <span class="text-xs font-semibold text-primary">Nhatrovn Assistant</span>
        </div>
        <div class="bg-surface-container-low p-6 rounded-2xl rounded-tl-none border border-primary/5 max-w-[60%] shadow-xl">
            <div class="flex items-center gap-3">
                <div class="flex gap-1">
                    <span class="w-2.5 h-2.5 rounded-full bg-primary/50 animate-bounce" style="animation-delay:0ms"></span>
                    <span class="w-2.5 h-2.5 rounded-full bg-primary/50 animate-bounce" style="animation-delay:150ms"></span>
                    <span class="w-2.5 h-2.5 rounded-full bg-primary/50 animate-bounce" style="animation-delay:300ms"></span>
                </div>
                <span id="typing-text" class="text-xs text-on-surface-variant">Đang phân tích nhu cầu...</span>
            </div>
        </div>`;
    area.appendChild(div);
    scrollToBottom();
}

function updateTypingText(msg) {
    const el = document.getElementById('typing-text');
    if (el) el.textContent = msg;
}

function removeTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

function showTrace(dataId) {
    const dataEl = document.getElementById(dataId);
    if (!dataEl) return;
    const trace = JSON.parse(dataEl.textContent);
    const modal = document.getElementById('traceModal');
    const content = document.getElementById('traceContent');

    const steps = [
        { icon: 'alt_route', title: 'Intent', body: `<strong>${esc(trace.intent || 'N/A')}</strong>` },
        { icon: 'tune', title: 'State Patch', body: `<span class="text-xs">${esc(JSON.stringify(trace.applied_operations || []))}</span>` },
        { icon: 'search', title: 'Read-only Tools', body: `Read calls: <strong>${trace.read_tool_calls || 0}</strong>, Write calls: <strong>${trace.write_tool_calls || 0}</strong>` },
        { icon: 'verified', title: 'Grounding', body: `${esc(trace.grounding_result || 'N/A')}` },
    ];

    content.innerHTML = steps.map((s, i) => `
        <div class="p-4 rounded-xl bg-surface-container border border-outline-variant/10">
            <div class="flex items-center gap-2 mb-2">
                <span class="material-symbols-outlined text-primary text-lg">${s.icon}</span>
                <span class="text-sm font-bold text-on-surface">${i + 1}. ${s.title}</span>
            </div>
            <div class="text-sm text-on-surface-variant">${s.body}</div>
        </div>`).join('');

    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

function closeTrace(e) {
    const modal = document.getElementById('traceModal');
    if (e && e.target !== modal) return;
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

function esc(text) {
    const d = document.createElement('div');
    d.textContent = text || '';
    return d.innerHTML;
}

function escAttr(text) {
    return String(text || '').replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, ' ');
}

function formatMarkdown(text) {
    if (!text) return '';
    return esc(text)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 bg-surface-variant rounded text-xs">$1</code>')
        .replace(/\n\n/g, '</p><p class="mt-2">')
        .replace(/\n/g, '<br>')
        .replace(/^(.*)$/, '<p>$1</p>');
}

function formatVnd(value) {
    if (value === null || value === undefined || value === '') return 'Chưa rõ giá';
    const n = Number(value);
    if (Number.isNaN(n)) return String(value);
    return `${n.toLocaleString('vi-VN')} VND`;
}

function scrollToBottom() {
    const c = document.getElementById('chatCanvas');
    setTimeout(() => c.scrollTop = c.scrollHeight, 50);
}

function setSubmitEnabled(enabled) {
    document.getElementById('submitBtn').disabled = !enabled;
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeTrace();
});
