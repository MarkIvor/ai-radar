/* AI Radar — фронтенд-логика SPA */

const API_BASE = '';

// ------------------ STATE ------------------

let accessToken = localStorage.getItem('air_access_token') || null;
let refreshToken = localStorage.getItem('air_refresh_token') || null;

let activeReportData = null;
let userRole = null;

// ------------------ METRIC HINTS ------------------

const METRIC_HINTS = {
    burstiness: 'SD/Mean длин предложений. ИИ → 0.1–0.4, человек → 0.6–1.2.',
    lexical_diversity_ttr: 'Уникальные леммы / всего слов.',
    sentence_length_variance: 'Дисперсия длин предложений.',
    perplexity_proxy: 'Shannon-энтропия биграмм — прокси перплексии.',
    synthetic_cliche_density: 'Плотность синтетических клише на 1000 слов.',
    punctuation_entropy: 'Разнообразие пунктуации.',
    sentence_starter_entropy: 'Энтропия первых слов предложений.',
    yule_k: "Yule's K — характеристика богатства словаря.",
    honore_r: "Honore's R — лексическое богатство.",
    sichel_s: "Sichel's S — доля hapax legomena.",
    hapax_ratio: 'Слова, встретившиеся 1 раз / всего слов.',
    connective_density: 'Логические маркеры / 1000 слов.',
    ngram_repetition_rate: 'Доля повторяющихся 3-грамм.',
    sentence_rhythm_entropy: 'Энтропия распределения длин предложений.',
    comma_per_sentence: 'Среднее число запятых на предложение.',
    avg_syllables_per_word: 'Среднее число слогов в слове.',
    function_word_ratio: 'Доля служебных слов.',
    lexical_density: 'Content-words / всего слов.',
    ttr_windowed: 'Скользящий TTR по окну 500 слов (MATTR-подобный).',
    determiner_density: 'Определители / 1000 слов.',
    num_sentences: 'Число предложений в тексте.',
    num_words: 'Число слов в тексте.',
    num_chars: 'Число символов в тексте.',
};

const METRIC_LABELS = {
    burstiness: 'Burstiness (Всплесковость)',
    lexical_diversity_ttr: 'Lexical Diversity (TTR)',
    sentence_length_variance: 'Sentence Length Variance',
    perplexity_proxy: 'Perplexity Proxy',
    synthetic_cliche_density: 'Cliche Density',
    punctuation_entropy: 'Punctuation Entropy',
    sentence_starter_entropy: 'Starter Entropy',
    yule_k: "Yule's K",
    honore_r: "Honore's R",
    sichel_s: "Sichel's S",
    hapax_ratio: 'Hapax Ratio',
    connective_density: 'Connective Density',
    ngram_repetition_rate: 'N-gram Repetition',
    sentence_rhythm_entropy: 'Rhythm Entropy',
    comma_per_sentence: 'Comma / Sentence',
    avg_syllables_per_word: 'Avg Syllables / Word',
    function_word_ratio: 'Function Word Ratio',
    lexical_density: 'Lexical Density',
    ttr_windowed: 'TTR windowed',
    determiner_density: 'Determiner Density',
    num_sentences: 'Предложений',
    num_words: 'Слов',
    num_chars: 'Символов',
};

// ------------------ INIT ------------------

document.addEventListener('DOMContentLoaded', () => {
    if (window.lucide) lucide.createIcons();
    userRole = localStorage.getItem('air_role') || null;
    if (accessToken) {
        showApp();
        bootstrap();
    }
});

async function bootstrap() {
    refreshJudgeStatus();
    renderRecentScans();
    renderFolderTree();
    loadAdminSettings();
    renderLlmProviders();
    renderApiKeys();
    setInterval(refreshJudgeStatus, 30000);
}

function showApp() {
    document.getElementById('login-screen').style.display = 'none';
    document.getElementById('app-container').style.display = 'flex';
    // Скрыть админ-вкладку для teacher
    const adminNav = document.querySelector('.nav-item[data-tab="admin"]');
    if (adminNav) {
        adminNav.style.display = (userRole === 'admin') ? 'flex' : 'none';
    }
    if (window.lucide) lucide.createIcons();
}

// ------------------ HTTP ------------------

async function api(path, opts = {}) {
    const headers = Object.assign({}, opts.headers || {});
    if (accessToken) headers['Authorization'] = 'Bearer ' + accessToken;
    if (opts.body && !(opts.body instanceof FormData) && !headers['Content-Type']) {
        headers['Content-Type'] = 'application/json';
    }
    const resp = await fetch(API_BASE + path, {
        method: opts.method || 'GET',
        headers,
        body: opts.body,
    });
    if (resp.status === 401 && accessToken) {
        const refreshed = await tryRefresh();
        if (refreshed) return api(path, opts);
        logout();
        throw new Error('Не авторизован');
    }
    if (!resp.ok) {
        let detail = resp.statusText;
        try {
            const j = await resp.json();
            detail = j.detail || JSON.stringify(j);
        } catch (e) {}
        throw new Error(detail);
    }
    if (resp.status === 204) return null;
    return resp.json();
}

async function tryRefresh() {
    if (!refreshToken) return false;
    try {
        const resp = await fetch(API_BASE + '/api/auth/refresh', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({refresh_token: refreshToken}),
        });
        if (!resp.ok) return false;
        const data = await resp.json();
        accessToken = data.access_token;
        refreshToken = data.refresh_token;
        localStorage.setItem('air_access_token', accessToken);
        localStorage.setItem('air_refresh_token', refreshToken);
        return true;
    } catch (e) { return false; }
}

// ------------------ LOGIN ------------------

async function handleLogin(e) {
    e.preventDefault();
    const login = document.getElementById('auth-login').value.trim();
    const password = document.getElementById('auth-password').value;
    const errBox = document.getElementById('login-error');
    errBox.style.display = 'none';
    try {
        const resp = await fetch(API_BASE + '/api/auth/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({login, password}),
        });
        if (!resp.ok) {
            errBox.style.display = 'block';
            return;
        }
        const data = await resp.json();
        accessToken = data.access_token;
        refreshToken = data.refresh_token;
        userRole = data.role || 'admin';
        localStorage.setItem('air_access_token', accessToken);
        localStorage.setItem('air_refresh_token', refreshToken);
        localStorage.setItem('air_role', userRole);
        document.getElementById('auth-password').value = '';
        document.getElementById('auth-login').value = '';
        showApp();
        bootstrap();
    } catch (e) {
        errBox.textContent = 'Сеть: ' + e.message;
        errBox.style.display = 'block';
    }
}

function logout() {
    accessToken = null;
    refreshToken = null;
    userRole = null;
    localStorage.removeItem('air_access_token');
    localStorage.removeItem('air_refresh_token');
    localStorage.removeItem('air_role');
    document.getElementById('login-screen').style.display = 'flex';
    document.getElementById('app-container').style.display = 'none';
}

// ------------------ NAVIGATION ------------------

function switchTab(tabName) {
    // Teacher не может открыть админ-панель
    if (tabName === 'admin' && userRole !== 'admin') {
        alert('Доступ к настройкам системы только для администратора.');
        return;
    }
    document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
    document.querySelectorAll('.view-panel').forEach(panel => panel.style.display = 'none');
    if (tabName === 'dashboard') {
        document.querySelector('.nav-item[data-tab="dashboard"]').classList.add('active');
        document.getElementById('view-dashboard').style.display = 'flex';
        renderRecentScans();
    } else if (tabName === 'folders') {
        document.querySelector('.nav-item[data-tab="folders"]').classList.add('active');
        document.getElementById('view-folders').style.display = 'flex';
        renderFolderTree();
    } else if (tabName === 'admin') {
        document.querySelector('.nav-item[data-tab="admin"]').classList.add('active');
        document.getElementById('view-admin').style.display = 'flex';
        loadAdminSettings();
        renderLlmProviders();
        renderApiKeys();
        renderUsers();
    }
    if (window.lucide) lucide.createIcons();
}

function triggerQuickScanFocus() {
    switchTab('dashboard');
    document.getElementById('quick-text-input').focus();
}

// ------------------ JUDGE STATUS ------------------

async function refreshJudgeStatus() {
    const badge = document.getElementById('judge-status-badge');
    const text = document.getElementById('judge-status-text');
    try {
        const data = await api('/api/admin/judge-status');
        badge.classList.remove('offline');
        if (data.status === 'OFFLINE') {
            badge.classList.add('offline');
        }
        text.textContent = data.label || data.status;
    } catch (e) {
        badge.classList.add('offline');
        text.textContent = 'JUDGE OFFLINE';
    }
}

// ------------------ QUICK SCAN ------------------

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    if (file.size > 50 * 1024 * 1024) {
        alert('Файл слишком большой (>50 МБ)');
        return;
    }
    document.getElementById('file-selected-name').textContent = file.name;
    document.getElementById('file-selected-banner').style.display = 'flex';
}

function clearQuickScan() {
    const ta = document.getElementById('quick-text-input');
    ta.value = '';
    delete ta.dataset.fileName;
    document.getElementById('quick-file-input').value = '';
    document.getElementById('file-selected-banner').style.display = 'none';
}

async function runQuickScan() {
    const ta = document.getElementById('quick-text-input');
    const fileInput = document.getElementById('quick-file-input');
    const file = fileInput.files[0];
    const text = ta.value.trim();

    if (!file && !text) {
        alert('Пожалуйста, введите текст или загрузите файл!');
        return;
    }

    const btn = document.getElementById('btn-run-quick');
    btn.disabled = true;

    try {
        let taskData;
        if (file) {
            const fd = new FormData();
            fd.append('file', file);
            const resp = await fetch(API_BASE + '/api/scan/tasks/upload', {
                method: 'POST',
                headers: accessToken ? { 'Authorization': 'Bearer ' + accessToken } : {},
                body: fd,
            });
            if (!resp.ok) throw new Error(await resp.text());
            taskData = await resp.json();
        } else {
            taskData = await api('/api/scan/tasks', {
                method: 'POST',
                body: JSON.stringify({ text, title: 'Проверка текста' }),
            });
        }
        await trackTaskProgress(taskData.id);
        clearQuickScan();
    } catch (e) {
        alert('Ошибка запуска проверки: ' + e.message);
    } finally {
        btn.disabled = false;
    }
}

async function trackTaskProgress(taskId) {
    const progressBox = document.getElementById('scan-progress');
    const progressSteps = document.getElementById('progress-steps');
    const progressBar = document.getElementById('progress-bar-fill');
    const progressTitle = document.getElementById('progress-title');
    const progressEta = document.getElementById('progress-eta');
    progressBox.style.display = 'block';
    progressSteps.innerHTML = '';
    progressBar.style.width = '0%';
    progressTitle.textContent = 'Запуск...';
    progressEta.textContent = '— сек';
    const startTime = Date.now();
    renderActiveTasks();

    try {
        const resp = await fetch(API_BASE + `/api/scan/tasks/${taskId}/stream`, {
            headers: accessToken ? { 'Authorization': 'Bearer ' + accessToken } : {},
        });
        if (!resp.ok) throw new Error('Не удалось подключиться к потоку прогресса');
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalReportId = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.slice(6));
                    if (data.progress !== undefined) {
                        progressBar.style.width = data.progress + '%';
                    }
                    if (data.message) {
                        progressTitle.textContent = data.message;
                    }
                    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                    progressEta.textContent = elapsed + ' сек';
                    if (data.steps && data.steps.length) {
                        progressSteps.innerHTML = data.steps.slice(-20).map(s => {
                            const icon = (s.event === 'final' || s.event === 'done' || s.event === 'finalizing') ? '✓' : '●';
                            return `<div class="progress-step done"><span class="step-icon">${icon}</span> ${escapeHtml(s.message || s.event)}</div>`;
                        }).join('');
                    }
                    if (data.event === 'final') {
                        finalReportId = data.report_id;
                        progressBar.style.width = '100%';
                        progressTitle.textContent = 'Готово!';
                    } else if (data.event === 'error') {
                        throw new Error(data.message || 'Ошибка проверки');
                    }
                } catch (parseErr) { /* skip */ }
            }
        }
        if (finalReportId) {
            window.open(`/report/${finalReportId}`, '_blank');
        }
    } catch (e) {
        progressTitle.textContent = 'Ошибка: ' + e.message;
    } finally {
        renderRecentScans();
        renderActiveTasks();
        setTimeout(() => { progressBox.style.display = 'none'; }, 5000);
    }
}

async function renderActiveTasks() {
    const card = document.getElementById('active-tasks-card');
    const list = document.getElementById('active-tasks-list');
    if (!card || !list) return;
    try {
        const tasks = await api('/api/scan/tasks?limit=10');
        const active = tasks.filter(t => t.status === 'running' || t.status === 'pending');
        if (active.length === 0) { card.style.display = 'none'; return; }
        card.style.display = 'block';
        list.innerHTML = active.map(t => `
            <div class="task-row" data-task-id="${t.id}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-weight:600; font-size:13px;">${escapeHtml(t.title)}</span>
                    <span style="font-size:11px; color: var(--text-muted);">${t.progress || 0}%</span>
                </div>
                <div class="progress-bar-track" style="height:6px;">
                    <div class="progress-bar-fill" style="width:${t.progress || 0}%;"></div>
                </div>
                <div style="font-size:11px; color: var(--text-muted); margin-top:4px;">${escapeHtml(t.progress_msg || '')}</div>
            </div>
        `).join('');
        if (active.length > 0) setTimeout(renderActiveTasks, 2000);
    } catch (e) {
        card.style.display = 'none';
    }
}

// ------------------ WIDGETS ------------------

function updateGaugeWidget(score, veto, judgeStatus) {
    document.getElementById('gauge-score-val').innerText = (score ?? 0) + '%';
    const circle = document.getElementById('gauge-circle-path');
    circle.setAttribute('stroke-dasharray', `${score || 0}, 100`);
    circle.classList.remove('danger', 'warning', 'success');
    if (score >= 75) circle.classList.add('danger');
    else if (score >= 50) circle.classList.add('warning');
    else circle.classList.add('success');

    const pill = document.getElementById('veto-status-pill');
    if (veto) {
        pill.className = 'veto-badge veto-triggered';
        pill.innerHTML = '<i data-lucide="shield-alert" style="width:16px;"></i> ВЕТО: АКТИВИРОВАНО';
    } else {
        pill.className = 'veto-badge veto-passed';
        pill.innerHTML = '<i data-lucide="shield-check" style="width:16px;"></i> ВЕТО: НЕ АКТИВИРОВАНО';
    }
    if (window.lucide) lucide.createIcons();
}

function renderMetrics(m) {
    const container = document.getElementById('metrics-list');
    if (!m) {
        container.innerHTML = '<p class="metric-empty">Метрики недоступны</p>';
        return;
    }
    const entries = [
        ['burstiness', m.burstiness],
        ['lexical_diversity_ttr', m.lexical_diversity_ttr],
        ['sentence_length_variance', m.sentence_length_variance],
        ['perplexity_proxy', m.perplexity_proxy],
        ['synthetic_cliche_density', m.synthetic_cliche_density],
        ['punctuation_entropy', m.punctuation_entropy],
        ['sentence_starter_entropy', m.sentence_starter_entropy],
        ['yule_k', m.yule_k],
        ['honore_r', m.honore_r],
        ['sichel_s', m.sichel_s],
        ['hapax_ratio', m.hapax_ratio],
        ['connective_density', m.connective_density],
        ['ngram_repetition_rate', m.ngram_repetition_rate],
        ['sentence_rhythm_entropy', m.sentence_rhythm_entropy],
        ['comma_per_sentence', m.comma_per_sentence],
        ['avg_syllables_per_word', m.avg_syllables_per_word],
        ['function_word_ratio', m.function_word_ratio],
        ['lexical_density', m.lexical_density],
        ['ttr_windowed', m.ttr_windowed],
        ['determiner_density', m.determiner_density],
        ['num_sentences', `${m.num_sentences} / ${m.num_words} / ${m.num_chars}`],
    ];
    container.innerHTML = entries.map(([k, v]) => {
        const label = METRIC_LABELS[k] || k;
        const hint = METRIC_HINTS[k] || '';
        const valStr = (typeof v === 'number') ? v.toFixed(3) : v;
        return `<div class="metric-row" title="${escapeHtml(hint)}" style="cursor: help;">
            <span style="color: var(--text-muted);">${escapeHtml(label)}
                ${hint ? '<i data-lucide="help-circle" style="width:11px; vertical-align:middle; opacity:0.5;"></i>' : ''}
            </span>
            <span class="metric-val">${valStr}</span>
        </div>`;
    }).join('');
    if (window.lucide) lucide.createIcons();
}

// ------------------ RECENT SCANS ------------------

async function renderRecentScans() {
    const container = document.getElementById('recent-scans-list');
    try {
        const items = await api('/api/scan/reports/recent?limit=10');
        if (!items.length) {
            container.innerHTML = '<p style="font-size:13px; color: var(--text-muted);">Проверок пока нет.</p>';
            return;
        }
        container.innerHTML = items.map(item => `
            <div class="folder-item" onclick="openReportPage(${item.id})" style="cursor:pointer;">
                <div class="folder-info">
                    <i data-lucide="file-text" style="color: var(--accent-dark);"></i>
                    <div>
                        <div>${escapeHtml(item.title)}</div>
                        <div style="font-size: 11px; color: var(--text-muted);">${item.created_at}</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-weight: 700; font-size: 14px; color: ${item.ai_score > 50 ? 'var(--accent-danger)' : 'var(--accent-success)'}">${item.ai_score}% AI</span>
                    <span class="veto-badge ${item.veto ? 'veto-triggered' : 'veto-passed'}" style="font-size: 10px; padding: 4px 10px;">
                        ${item.veto ? 'VETO' : 'PASS'}
                    </span>
                </div>
            </div>
        `).join('');
        if (window.lucide) lucide.createIcons();
    } catch (e) {
        container.innerHTML = `<p style="font-size:13px; color: var(--accent-danger);">Ошибка: ${escapeHtml(e.message)}</p>`;
    }
}

function openReportPage(id) {
    window.open(`/report/${id}`, '_blank');
}

// ------------------ FOLDER TREE ------------------

async function renderFolderTree() {
    const root = document.getElementById('folder-tree-root');
    try {
        const folders = await api('/api/folders');
        if (!folders.length) {
            root.innerHTML = '<p style="font-size:13px; color: var(--text-muted);">Папок пока нет. Создайте первую.</p>';
            return;
        }
        root.innerHTML = folders.map(f => `
            <div class="folder-item">
                <div class="folder-info" onclick="toggleFolderFiles(${f.id}, this)">
                    <i data-lucide="folder" style="color: var(--accent-warning);"></i>
                    <span>${escapeHtml(f.name)}</span>
                </div>
                <div style="display:flex; gap: 8px;">
                    <button class="btn-secondary btn-sm" onclick="event.stopPropagation(); uploadFilePrompt(${f.id})">+ Файл</button>
                    <button class="btn-secondary btn-sm" onclick="event.stopPropagation(); createSubfolder(${f.id})">+ Подпапка</button>
                    <button class="btn-danger" onclick="event.stopPropagation(); deleteFolder(${f.id})">✕</button>
                </div>
            </div>
            <div id="folder-${f.id}-files" class="subfolder-tree" style="display:none;"></div>
        `).join('');
        if (window.lucide) lucide.createIcons();
    } catch (e) {
        root.innerHTML = `<p style="font-size:13px; color: var(--accent-danger);">Ошибка: ${escapeHtml(e.message)}</p>`;
    }
}

async function toggleSubfolderFiles(folderId) {
    const container = document.getElementById(`subfolder-${folderId}-files`);
    if (container.style.display === 'none') {
        container.style.display = 'flex';
        try {
            const [subfolders, files] = await Promise.all([
                api(`/api/folders?parent_id=${folderId}`),
                api(`/api/folders/${folderId}/files`),
            ]);
            let html = '';
            if (subfolders.length) {
                html += subfolders.map(sf => `
                    <div class="folder-item" style="padding: 8px 12px; background: #f8fafc;">
                        <div class="folder-info" style="font-size: 13px; cursor:pointer;" onclick="toggleSubfolderFiles(${sf.id})">
                            <i data-lucide="folder-open" style="color: var(--text-muted); width: 14px;"></i>
                            <span>${escapeHtml(sf.name)}</span>
                        </div>
                        <div style="display:flex; gap: 6px;">
                            <button class="btn-secondary btn-sm" onclick="event.stopPropagation(); uploadFilePrompt(${sf.id})">+ Файл</button>
                            <button class="btn-danger" onclick="deleteFolder(${sf.id})">✕</button>
                        </div>
                    </div>
                    <div id="subfolder-${sf.id}-files" class="subfolder-tree" style="display:none;"></div>
                `).join('');
            }
            if (files.length) {
                html += files.map(f => `
                    <div class="folder-item" style="padding: 8px 12px; background: #fff;">
                        <div class="folder-info" style="font-size: 13px;">
                            <i data-lucide="file-text" style="color: var(--text-muted); width: 14px;"></i>
                            <span>${escapeHtml(f.name)}</span>
                        </div>
                        <div style="display:flex; gap: 8px;">
                            <button class="btn-secondary btn-sm" onclick="deepScanFile(${f.id})">Проверить</button>
                            <button class="btn-danger" onclick="deleteFile(${f.id})">✕</button>
                        </div>
                    </div>
                `).join('');
            }
            if (!html) html = '<p style="font-size:12px; color: var(--text-muted); padding: 4px 0;">Папка пуста</p>';
            container.innerHTML = html;
            if (window.lucide) lucide.createIcons();
        } catch (e) {
            container.innerHTML = `<p style="font-size:12px; color: var(--accent-danger);">Ошибка: ${escapeHtml(e.message)}</p>`;
        }
    } else {
        container.style.display = 'none';
    }
}

async function toggleFolderFiles(folderId, el) {
    const container = document.getElementById(`folder-${folderId}-files`);
    if (container.style.display === 'none') {
        container.style.display = 'flex';
        try {
            // Загружаем и подпапки, и файлы
            const [subfolders, files] = await Promise.all([
                api(`/api/folders?parent_id=${folderId}`),
                api(`/api/folders/${folderId}/files`),
            ]);
            let html = '';
            if (subfolders.length) {
                html += subfolders.map(sf => `
                    <div class="folder-item" style="padding: 8px 12px; background: #f8fafc; flex-wrap: wrap;">
                        <div class="folder-info" style="font-size: 13px; cursor:pointer;" onclick="toggleSubfolderFiles(${sf.id})">
                            <i data-lucide="folder-open" style="color: var(--text-muted); width: 14px;"></i>
                            <span>${escapeHtml(sf.name)}</span>
                        </div>
                        <div style="display:flex; gap: 6px;">
                            <button class="btn-secondary btn-sm" onclick="event.stopPropagation(); uploadFilePrompt(${sf.id})">+ Файл</button>
                            <button class="btn-danger" onclick="deleteFolder(${sf.id})">✕</button>
                        </div>
                    </div>
                    <div id="subfolder-${sf.id}-files" class="subfolder-tree" style="display:none;"></div>
                `).join('');
            }
            if (files.length) {
                html += files.map(f => `
                    <div class="folder-item" style="padding: 8px 12px; background: #fff;">
                        <div class="folder-info" style="font-size: 13px;">
                            <i data-lucide="file-text" style="color: var(--text-muted); width: 14px;"></i>
                            <span>${escapeHtml(f.name)}</span>
                        </div>
                        <div style="display:flex; gap: 8px;">
                            <button class="btn-secondary btn-sm" onclick="deepScanFile(${f.id})">Проверить</button>
                            <button class="btn-danger" onclick="deleteFile(${f.id})">✕</button>
                        </div>
                    </div>
                `).join('');
            }
            if (!html) html = '<p style="font-size:12px; color: var(--text-muted); padding: 4px 0;">Папка пуста</p>';
            container.innerHTML = html;
            if (window.lucide) lucide.createIcons();
        } catch (e) {
            container.innerHTML = `<p style="font-size:12px; color: var(--accent-danger);">Ошибка: ${escapeHtml(e.message)}</p>`;
        }
    } else {
        container.style.display = 'none';
    }
}

async function createNewFolder() {
    const name = prompt('Название новой папки:');
    if (!name) return;
    try {
        await api('/api/folders', {method: 'POST', body: JSON.stringify({name})});
        renderFolderTree();
    } catch (e) { alert('Ошибка: ' + e.message); }
}

async function createSubfolder(parentId) {
    const name = prompt('Название подпапки:');
    if (!name) return;
    try {
        await api('/api/folders', {method: 'POST', body: JSON.stringify({name, parent_id: parentId})});
        // Обновить развёрнутый вид
        const container = document.getElementById(`folder-${parentId}-files`);
        if (container && container.style.display !== 'none') {
            toggleFolderFiles(parentId, null);
            setTimeout(() => toggleFolderFiles(parentId, null), 100);
        } else {
            renderFolderTree();
        }
    } catch (e) { alert('Ошибка: ' + e.message); }
}

async function deleteFolder(id) {
    if (!confirm('Удалить папку со всем содержимым?')) return;
    try {
        await api(`/api/folders/${id}`, {method: 'DELETE'});
        renderFolderTree();
    } catch (e) { alert('Ошибка: ' + e.message); }
}

function uploadFilePrompt(folderId) {
    const inp = document.createElement('input');
    inp.type = 'file';
    inp.accept = '.txt,.docx,.pdf';
    inp.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const fd = new FormData();
        fd.append('file', file);
        try {
            await api(`/api/folders/${folderId}/files`, {method: 'POST', body: fd});
            // Обновить развёрнутый вид (проверить оба возможных контейнера)
            const topContainer = document.getElementById(`folder-${folderId}-files`);
            const subContainer = document.getElementById(`subfolder-${folderId}-files`);
            const container = (topContainer && topContainer.style.display !== 'none') ? topContainer :
                              (subContainer && subContainer.style.display !== 'none') ? subContainer : null;
            if (container) {
                const wasTop = container === topContainer;
                container.style.display = 'flex';
                // Перерисовать содержимое
                if (wasTop) {
                    toggleFolderFiles(folderId, null);
                    setTimeout(() => toggleFolderFiles(folderId, null), 100);
                } else {
                    toggleSubfolderFiles(folderId);
                    setTimeout(() => toggleSubfolderFiles(folderId), 100);
                }
            }
        } catch (e) { alert('Ошибка: ' + e.message); }
    };
    inp.click();
}

async function deleteFile(id) {
    if (!confirm('Удалить файл?')) return;
    try {
        await api(`/api/folders/files/${id}`, {method: 'DELETE'});
        renderFolderTree();
    } catch (e) { alert('Ошибка: ' + e.message); }
}

async function deepScanFile(id) {
    // Без confirm — запускаем асинхронную задачу через API
    try {
        const taskData = await api('/api/scan/tasks', {
            method: 'POST',
            body: JSON.stringify({ file_id: id }),
        });
        // Переключиться на дашборд и показать прогресс
        switchTab('dashboard');
        await trackTaskProgress(taskData.id);
    } catch (e) { alert('Ошибка запуска проверки: ' + e.message); }
}

// ------------------ ADMIN ------------------

async function loadAdminSettings() {
    try {
        const s = await api('/api/admin/settings');
        document.getElementById('veto-threshold').value = s.veto_threshold;
        document.getElementById('judge-mode').value = s.judge_mode;
        document.getElementById('judge-retries').value = s.judge_max_retries;
        document.getElementById('new-admin-password').value = '';
    } catch (e) { /* silent */ }
}

async function saveAdminSettings() {
    const newPass = document.getElementById('new-admin-password').value;
    const body = {
        veto_threshold: parseInt(document.getElementById('veto-threshold').value),
        judge_mode: document.getElementById('judge-mode').value,
        judge_max_retries: parseInt(document.getElementById('judge-retries').value),
    };
    if (newPass && newPass.length >= 6) body.new_master_password = newPass;
    try {
        await api('/api/admin/settings', {method: 'PUT', body: JSON.stringify(body)});
        alert('Настройки сохранены.');
        document.getElementById('new-admin-password').value = '';
    } catch (e) { alert('Ошибка: ' + e.message); }
}

// ------------------ LLM PROVIDERS ------------------

async function renderLlmProviders() {
    try {
        const items = await api('/api/admin/llm-providers');
        const ensemble = items.filter(p => p.role === 'ensemble');
        const judges = items.filter(p => p.role === 'judge');
        renderProviderList('llm-providers-ensemble', ensemble, 'ensemble');
        renderProviderList('llm-providers-judge', judges, 'judge');
    } catch (e) {
        document.getElementById('llm-providers-ensemble').innerHTML =
            `<p style="font-size:12px; color: var(--accent-danger);">Ошибка: ${escapeHtml(e.message)}</p>`;
    }
}

function renderProviderList(containerId, items, role) {
    const container = document.getElementById(containerId);
    if (!items.length) {
        container.innerHTML = `<p style="font-size:12px; color: var(--text-muted);">${role === 'judge' ? 'Метасудья не назначен.' : 'Детекторы не добавлены.'}</p>`;
        return;
    }
    container.innerHTML = items.map(p => `
        <div class="provider-row ${p.enabled ? '' : 'disabled'}">
            <div>
                <div><strong>${escapeHtml(p.name)}</strong> — <code>${escapeHtml(p.model)}</code></div>
                <div class="provider-meta">${escapeHtml(p.base_url)} • key: <code>${escapeHtml(p.api_key_masked)}</code></div>
            </div>
            <div style="display:flex; gap: 6px; align-items: center;">
                <span style="font-size: 11px; color: ${p.enabled ? 'var(--accent-success)' : 'var(--text-muted)'}">
                    ${p.enabled ? 'ВКЛ' : 'ВЫКЛ'}
                </span>
                <button class="btn-secondary btn-sm" onclick="toggleLlmProvider(${p.id}, ${!p.enabled})">
                    ${p.enabled ? 'Выключить' : 'Включить'}
                </button>
                <button class="btn-secondary btn-sm" onclick="editLlmProvider(${p.id})">✎</button>
                <button class="btn-danger" onclick="deleteLlmProvider(${p.id})">✕</button>
            </div>
        </div>
    `).join('');
}

async function addLlmProvider(role) {
    const prefix = role === 'judge' ? 'judge' : 'llm';
    const name = document.getElementById(`${prefix}-name`).value.trim();
    const model = document.getElementById(`${prefix}-model`).value.trim();
    const base_url = document.getElementById(`${prefix}-baseurl`).value.trim();
    const api_key = document.getElementById(`${prefix}-apikey`).value.trim();
    if (!name || !model || !base_url || !api_key) {
        alert('Заполните все поля');
        return;
    }
    try {
        await api('/api/admin/llm-providers', {
            method: 'POST',
            body: JSON.stringify({name, model, base_url, api_key, role, enabled: true}),
        });
        [`${prefix}-name`, `${prefix}-model`, `${prefix}-baseurl`, `${prefix}-apikey`].forEach(id => document.getElementById(id).value = '');
        renderLlmProviders();
        refreshJudgeStatus();
    } catch (e) { alert('Ошибка: ' + e.message); }
}

async function toggleLlmProvider(id, enabled) {
    try {
        const url = `/api/admin/llm-providers/${id}/toggle?enabled=${enabled}`;
        await api(url, {method: 'POST'});
        renderLlmProviders();
        refreshJudgeStatus();
    } catch (e) { alert('Ошибка: ' + e.message); }
}

async function editLlmProvider(id) {
    const items = await api('/api/admin/llm-providers');
    const p = items.find(x => x.id === id);
    if (!p) return;
    const name = prompt('Имя:', p.name); if (name === null) return;
    const model = prompt('Модель:', p.model); if (model === null) return;
    const base_url = prompt('Base URL:', p.base_url); if (base_url === null) return;
    const api_key = prompt('API key (оставьте пустым — будет заглушка, изменить нельзя через UI):', ''); if (api_key === null) return;
    // Маскированный ключ нельзя восстановить — используем заглушку, сервер должен принять только при изменении.
    // Для простоты: если пользователь ввёл новый ключ, отправляем его; иначе — заглушку, которая приведёт к ошибке.
    // Лучше: не позволять менять ключ через prompt, а только через delete+add.
    try {
        await api(`/api/admin/llm-providers/${id}`, {
            method: 'PUT',
            body: JSON.stringify({
                name, model, base_url,
                api_key: api_key || '****-unchanged-****',
                role: p.role,
                enabled: p.enabled,
            }),
        });
        renderLlmProviders();
    } catch (e) { alert('Ошибка: ' + e.message + '\n\nСовет: для смены ключа удалите и добавьте провайдера заново.'); }
}

async function deleteLlmProvider(id) {
    if (!confirm('Удалить провайдера?')) return;
    try {
        await api(`/api/admin/llm-providers/${id}`, {method: 'DELETE'});
        renderLlmProviders();
        refreshJudgeStatus();
    } catch (e) { alert('Ошибка: ' + e.message); }
}

// ------------------ API KEYS ------------------

async function renderApiKeys() {
    const container = document.getElementById('api-keys-list');
    try {
        const items = await api('/api/admin/api-keys');
        if (!items.length) {
            container.innerHTML = '<p style="font-size:12px; color: var(--text-muted);">Ключей пока нет.</p>';
            return;
        }
        container.innerHTML = items.map(k => `
            <div class="apikey-row ${k.revoked ? 'revoked' : ''}" data-key-id="${k.id}">
                <div style="flex:1;">
                    <strong>${escapeHtml(k.name)}</strong>
                    <div class="provider-meta">
                        <code class="apikey-prefix">${escapeHtml(k.key_prefix)}</code> • создан ${k.created_at}
                        ${k.last_used_at ? '• посл. исп. ' + k.last_used_at : ''}
                        ${k.revoked ? '• <span style="color: var(--accent-danger)">REVOKED</span>' : ''}
                    </div>
                    <div class="apikey-full-value" id="apikey-full-${k.id}" style="display:none;"></div>
                </div>
                <div style="display:flex; gap: 6px;">
                    <button class="eye-toggle" onclick="toggleKeyVisibility(${k.id})" ${k.revoked ? 'disabled' : ''} title="Показать/скрыть ключ">
                        <i data-lucide="eye" style="width:14px;"></i>
                    </button>
                    <button class="btn-secondary btn-sm" onclick="rotateApiKey(${k.id})">Rotate</button>
                    <button class="btn-danger" onclick="revokeApiKey(${k.id})" ${k.revoked ? 'disabled' : ''}>Revoke</button>
                </div>
            </div>
        `).join('');
        if (window.lucide) lucide.createIcons();
    } catch (e) {
        container.innerHTML = `<p style="font-size:12px; color: var(--accent-danger);">Ошибка: ${escapeHtml(e.message)}</p>`;
    }
}

// Хранилище полных ключей в памяти (после создания/ротации)
const apiKeyStore = {};

function toggleKeyVisibility(keyId) {
    const el = document.getElementById(`apikey-full-${keyId}`);
    const btn = el.parentElement.parentElement.querySelector('.eye-toggle i');
    if (el.style.display === 'none') {
        if (apiKeyStore[keyId]) {
            el.textContent = apiKeyStore[keyId];
            el.style.display = 'block';
            btn.setAttribute('data-lucide', 'eye-off');
        } else {
            alert('Полный ключ недоступен — он показывается только один раз при создании. Создайте новый ключ через Rotate.');
        }
    } else {
        el.style.display = 'none';
        btn.setAttribute('data-lucide', 'eye');
    }
    if (window.lucide) lucide.createIcons();
}

async function createApiKey() {
    const name = document.getElementById('api-key-name').value.trim();
    if (!name) { alert('Введите имя'); return; }
    try {
        const data = await api('/api/admin/api-keys', {method: 'POST', body: JSON.stringify({name})});
        apiKeyStore[data.id] = data.key;
        document.getElementById('api-key-name').value = '';
        document.getElementById('apikey-once-value').value = data.key;
        document.getElementById('apikey-modal').style.display = 'flex';
        renderApiKeys();
    } catch (e) { alert('Ошибка: ' + e.message); }
}

function closeApiKeyModal() {
    document.getElementById('apikey-modal').style.display = 'none';
}

function copyApiKeyOnce() {
    const inp = document.getElementById('apikey-once-value');
    inp.select();
    document.execCommand('copy');
    alert('Ключ скопирован');
}

async function revokeApiKey(id) {
    if (!confirm('Отозвать ключ?')) return;
    try {
        await api(`/api/admin/api-keys/${id}`, {method: 'DELETE'});
        delete apiKeyStore[id];
        renderApiKeys();
    } catch (e) { alert('Ошибка: ' + e.message); }
}

async function rotateApiKey(id) {
    if (!confirm('Выпустить новый ключ (старый будет отозван)?')) return;
    try {
        const data = await api(`/api/admin/api-keys/${id}/rotate`, {method: 'POST'});
        apiKeyStore[data.id] = data.key;
        document.getElementById('apikey-once-value').value = data.key;
        document.getElementById('apikey-modal').style.display = 'flex';
        renderApiKeys();
    } catch (e) { alert('Ошибка: ' + e.message); }
}

// ------------------ UTIL ------------------

function escapeHtml(s) {
    if (s == null) return '';
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// ------------------ PASSWORD CHANGE ------------------

function openPasswordModal() {
    document.getElementById('new-password-input').value = '';
    document.getElementById('new-password-confirm').value = '';
    document.getElementById('password-modal').style.display = 'flex';
}

function closePasswordModal() {
    document.getElementById('password-modal').style.display = 'none';
}

async function submitPasswordChange() {
    const p1 = document.getElementById('new-password-input').value;
    const p2 = document.getElementById('new-password-confirm').value;
    if (!p1 || p1.length < 4) { alert('Пароль слишком короткий (минимум 4 символа)'); return; }
    if (p1 !== p2) { alert('Пароли не совпадают'); return; }
    try {
        await api('/api/admin/users/me/password', {
            method: 'PUT',
            body: JSON.stringify({ new_password: p1 }),
        });
        alert('Пароль успешно изменён.');
        closePasswordModal();
    } catch (e) { alert('Ошибка: ' + e.message); }
}

// ------------------ USERS MANAGEMENT ------------------

async function renderUsers() {
    const container = document.getElementById('users-list');
    if (!container) return;
    if (userRole !== 'admin') {
        const section = document.getElementById('users-section');
        if (section) section.style.display = 'none';
        return;
    }
    try {
        const items = await api('/api/admin/users');
        container.innerHTML = items.map(u => `
            <div class="apikey-row">
                <div>
                    <strong>${escapeHtml(u.login)}</strong>
                    ${u.display_name ? '— ' + escapeHtml(u.display_name) : ''}
                    <span style="font-size:11px; color: ${u.role==='admin'?'var(--accent-primary)':'var(--text-muted)'}; margin-left:6px;">
                        [${u.role === 'admin' ? 'АДМИН' : 'ПРЕПОД'}]
                    </span>
                    <div class="provider-meta">
                        создан ${u.created_at}
                        ${u.last_login_at ? ' • посл. вход ' + u.last_login_at : ''}
                    </div>
                </div>
                <div style="display:flex; gap: 6px;">
                    <button class="btn-secondary btn-sm" onclick="resetUserPassword(${u.id}, '${escapeHtml(u.login)}')">Сбросить пароль</button>
                    ${u.login !== 'admin' ? `<button class="btn-danger" onclick="deleteUser(${u.id}, '${escapeHtml(u.login)}')">✕</button>` : ''}
                </div>
            </div>
        `).join('');
    } catch (e) {
        container.innerHTML = `<p style="font-size:12px; color: var(--accent-danger);">Ошибка: ${escapeHtml(e.message)}</p>`;
    }
}

async function createUser() {
    const login = document.getElementById('user-login').value.trim();
    const display_name = document.getElementById('user-displayname').value.trim();
    const password = document.getElementById('user-password').value;
    if (!login || !password) { alert('Заполните логин и пароль'); return; }
    try {
        await api('/api/admin/users', {
            method: 'POST',
            body: JSON.stringify({ login, password, role: 'teacher', display_name: display_name || null }),
        });
        ['user-login', 'user-displayname', 'user-password'].forEach(id => document.getElementById(id).value = '');
        renderUsers();
    } catch (e) { alert('Ошибка: ' + e.message); }
}

async function resetUserPassword(uid, login) {
    const newPass = prompt(`Новый пароль для "${login}":`);
    if (!newPass) return;
    if (newPass.length < 4) { alert('Минимум 4 символа'); return; }
    try {
        await api(`/api/admin/users/${uid}/reset-password`, {
            method: 'POST',
            body: JSON.stringify({ new_password: newPass }),
        });
        alert(`Пароль для "${login}" сброшен.`);
    } catch (e) { alert('Ошибка: ' + e.message); }
}

async function deleteUser(uid, login) {
    if (!confirm(`Удалить пользователя "${login}"?`)) return;
    try {
        await api(`/api/admin/users/${uid}`, { method: 'DELETE' });
        renderUsers();
    } catch (e) { alert('Ошибка: ' + e.message); }
}
