/* Логика страницы детализированного отчёта /report/{id} */

const METRIC_NAMES = {
    burstiness: { name: "Burstiness (Всплесковость)", hint: "SD/Mean длин предложений. ИИ → 0.1–0.4, человек → 0.6–1.2.", direction: "low_bad" },
    lexical_diversity_ttr: { name: "Lexical Diversity (TTR)", hint: "Уникальные леммы / всего слов.", direction: "high_good" },
    sentence_length_variance: { name: "Sentence Length Variance", hint: "Дисперсия длин предложений.", direction: "neutral" },
    perplexity_proxy: { name: "Perplexity Proxy", hint: "Shannon-энтропия биграмм — прокси перплексии.", direction: "neutral" },
    synthetic_cliche_density: { name: "Cliche Density", hint: "Плотность синтетических клише на 1000 слов.", direction: "high_bad" },
    punctuation_entropy: { name: "Punctuation Entropy", hint: "Разнообразие пунктуации.", direction: "neutral" },
    sentence_starter_entropy: { name: "Starter Entropy", hint: "Энтропия первых слов предложений.", direction: "neutral" },
    yule_k: { name: "Yule's K", hint: "Текстовая характеристика богатства словаря.", direction: "neutral" },
    honore_r: { name: "Honore's R", hint: "Лексическое богатство (hapax-коррекция).", direction: "neutral" },
    sichel_s: { name: "Sichel's S", hint: "Доля hapax legomena среди словаря.", direction: "high_good" },
    hapax_ratio: { name: "Hapax Ratio", hint: "Слова, встретившиеся 1 раз / всего слов.", direction: "high_good" },
    connective_density: { name: "Connective Density", hint: "Логические маркеры / 1000 слов.", direction: "high_bad" },
    ngram_repetition_rate: { name: "N-gram Repetition", hint: "Доля повторяющихся 3-грамм.", direction: "high_bad" },
    sentence_rhythm_entropy: { name: "Rhythm Entropy", hint: "Энтропия распределения длин предложений.", direction: "neutral" },
    comma_per_sentence: { name: "Comma / Sentence", hint: "Среднее число запятых на предложение.", direction: "neutral" },
    avg_syllables_per_word: { name: "Avg Syllables / Word", hint: "Среднее число слогов в слове.", direction: "neutral" },
    function_word_ratio: { name: "Function Word Ratio", hint: "Доля служебных слов.", direction: "neutral" },
    lexical_density: { name: "Lexical Density", hint: "Content-words / всего слов.", direction: "neutral" },
    ttr_windowed: { name: "TTR Windowed", hint: "Скользящий TTR по окну 500 слов (MATTR-подобный).", direction: "high_good" },
    determiner_density: { name: "Determiner Density", hint: "Определители / 1000 слов.", direction: "neutral" },
    num_sentences: { name: "Предложений", hint: "Число предложений в тексте.", direction: "neutral" },
    num_words: { name: "Слов", hint: "Число слов в тексте.", direction: "neutral" },
    num_chars: { name: "Символов", hint: "Число символов в тексте.", direction: "neutral" },
};

async function fetchReport() {
    const r = await fetch(`/api/scan/reports/${REPORT_ID}`, {
        headers: API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {},
    });
    if (r.status === 401) {
        alert("Сессия истекла. Вернитесь на главную и войдите снова.");
        window.location.href = "/";
        return null;
    }
    if (!r.ok) {
        document.querySelector(".report-shell").innerHTML = `<p style="color:var(--accent-danger)">Ошибка загрузки отчёта: ${r.status}</p>`;
        return null;
    }
    return r.json();
}

function escapeHtml(s) {
    if (s == null) return "";
    return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

function renderGauge(score) {
    const circle = document.getElementById("r-gauge");
    circle.setAttribute("stroke-dasharray", `${score}, 100`);
    circle.classList.remove("danger","warning","success");
    if (score >= 75) circle.classList.add("danger");
    else if (score >= 50) circle.classList.add("warning");
    else circle.classList.add("success");
    document.getElementById("r-score").textContent = score + "%";
}

function metricColorClass(key, value) {
    const meta = METRIC_NAMES[key];
    if (!meta || typeof value !== "number") return "";
    const dir = meta.direction;
    if (dir === "high_good") {
        if (value > 0.65) return "m-good";
        if (value < 0.3) return "m-bad";
        return "m-warn";
    }
    if (dir === "high_bad") {
        if (value > 8) return "m-bad";
        if (value > 3) return "m-warn";
        return "m-good";
    }
    if (dir === "low_bad") {
        if (value < 0.3) return "m-bad";
        if (value < 0.5) return "m-warn";
        return "m-good";
    }
    return "";
}

function renderMetrics(m) {
    const cont = document.getElementById("r-metrics");
    if (!m) { cont.innerHTML = "<p>Метрики недоступны</p>"; return; }
    const html = Object.entries(m).map(([k, v]) => {
        const meta = METRIC_NAMES[k] || { name: k, hint: "" };
        const val = typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3)) : v;
        const cls = metricColorClass(k, v);
        return `<div class="metric-cell ${cls}" title="${escapeHtml(meta.hint)}">
            <span class="m-name">${escapeHtml(meta.name)}</span>
            <span class="m-value">${val}</span>
        </div>`;
    }).join("");
    cont.innerHTML = html;
}

function renderEnsemble(metrics) {
    const cont = document.getElementById("r-ensemble");
    const panel = metrics?.signal2_ensemble_panel || {};
    const ens = panel.ensemble || {};
    const det = ens.detectors || [];
    const judge = panel.judge || null;

    let html = "";

    // Три фактора
    const aiTextPct = metrics?.judge_status === "OFFLINE" ? metrics?.signal1?.burstiness !== undefined ? "н/д" : "н/д" : "—";
    // ИИ для текста = средний ai_score ансамбля (оценка синтаксиса)
    const ensembleScores = det.filter(d => d.error === undefined || d.error === null).map(d => d.ai_score || 0);
    const aiTextScore = ensembleScores.length ? Math.round(ensembleScores.reduce((a,b)=>a+b,0) / ensembleScores.length) : null;
    // ИИ для смыслов = вердикт метасудьи (оценка логики/человечности)
    const aiMeaningScore = judge && !judge.error ? judge.ai_score : null;
    // Итоговый
    const finalScore = window._reportData?.ai_score || 0;

    html += `<div class="three-factors-grid">`;
    html += `<div class="factor-card">
        <span class="meta-label">ИИ для написания текста</span>
        <div class="factor-value ${aiTextScore !== null && aiTextScore >= 50 ? 'danger' : aiTextScore !== null && aiTextScore >= 30 ? 'warn' : 'good'}">${aiTextScore !== null ? aiTextScore + '%' : 'н/д'}</div>
        <small>оценка ансамбля по синтаксису</small>
    </div>`;
    html += `<div class="factor-card">
        <span class="meta-label">ИИ для смыслов</span>
        <div class="factor-value ${aiMeaningScore !== null && aiMeaningScore >= 50 ? 'danger' : aiMeaningScore !== null && aiMeaningScore >= 30 ? 'warn' : 'good'}">${aiMeaningScore !== null ? aiMeaningScore + '%' : 'н/д'}</div>
        <small>вердикт метасудьи по логике</small>
    </div>`;
    html += `<div class="factor-card">
        <span class="meta-label">Итоговый вердикт</span>
        <div class="factor-value ${finalScore >= 50 ? 'danger' : finalScore >= 30 ? 'warn' : 'good'}">${finalScore}%</div>
        <small>финальная оценка</small>
    </div>`;
    html += `</div>`;

    // Ансамбль
    if (ens.detectors) {
        html += `<div class="judge-narrative passed"><strong>Ансамбль детекторов:</strong> median ai_score=${ens.ai_score_median ?? "—"},
            veto-голосов=${ens.veto_votes ?? 0}/${det.length},
            честность=${ens.academic_integrity || "—"}.</div>`;

        // Признаки человека/ИИ
        if (ens.human_signals_summary && ens.human_signals_summary.length) {
            html += `<div class="signals-block"><strong style="color:var(--accent-success);">Признаки человека:</strong><ul>`;
            ens.human_signals_summary.forEach(s => { html += `<li>${escapeHtml(s)}</li>`; });
            html += `</ul></div>`;
        }
        if (ens.llm_signals_summary && ens.llm_signals_summary.length) {
            html += `<div class="signals-block"><strong style="color:var(--accent-danger);">Признаки ИИ:</strong><ul>`;
            ens.llm_signals_summary.forEach(s => { html += `<li>${escapeHtml(s)}</li>`; });
            html += `</ul></div>`;
        }

        html += det.map(d => `
            <div class="ensemble-row">
                <div>
                    <span class="e-model">${escapeHtml(d.model)}</span>
                    ${d.error ? `<span style="color:var(--accent-danger); font-size:11px;"> ⚠ ${escapeHtml(d.error.substring(0, 100))}</span>` : ""}
                    ${d.human_signals && d.human_signals.length ? `<div style="font-size:11px; color:var(--accent-success); margin-top:2px;">✓ ${d.human_signals.map(s=>escapeHtml(s)).join('; ')}</div>` : ""}
                    ${d.llm_signals && d.llm_signals.length ? `<div style="font-size:11px; color:var(--accent-danger); margin-top:2px;">⚠ ${d.llm_signals.map(s=>escapeHtml(s)).join('; ')}</div>` : ""}
                </div>
                <div style="display:flex; gap:12px; align-items:center;">
                    <span class="e-score" style="color:${(d.ai_score||0)>50?'var(--accent-danger)':'var(--accent-success)'}">${d.ai_score ?? "—"}</span>
                    ${d.veto ? '<span class="veto-badge veto-triggered" style="font-size:10px;padding:3px 8px;">VETO</span>' : '<span class="veto-badge veto-passed" style="font-size:10px;padding:3px 8px;">PASS</span>'}
                </div>
            </div>
        `).join("");
    }

    // Метасудья — развёрнутый вердикт
    if (judge && !judge.error) {
        html += `<div class="judge-narrative ${judge.veto ? '' : 'passed'}">
            <strong>Вердикт метасудьи (${escapeHtml(judge.model || '—')}):</strong>
            <p style="margin-top:8px;">Итоговая оценка вероятности ИИ-генерации: <strong>${judge.ai_score}%</strong>.</p>
            <p>Академическая честность: <strong>${judge.integrity || "—"}</strong>.</p>
            ${judge.veto ? `<p style="color:var(--accent-danger); font-weight:600;">⚠ ПРИМЕНЕНО БЛОКИРУЮЩЕЕ ВЕТО</p>` : ''}
            ${judge.veto_reason ? `<p style="margin-top:8px;"><strong>Обоснование:</strong> ${escapeHtml(judge.veto_reason)}</p>` : ""}
            ${judge.fragments && judge.fragments.length ? `<p style="margin-top:8px;"><strong>Фрагменты, на которые опирается вердикт:</strong></p><ul>${judge.fragments.map(f=>`<li>${escapeHtml(f)}</li>`).join('')}</ul>` : ""}
        </div>`;
    } else if (judge && judge.error) {
        html += `<div class="judge-narrative"><strong>Метасудья:</strong> ошибка вызова — ${escapeHtml(judge.error)}.
            Использован вердикт ансамбля.</div>`;
    }

    cont.innerHTML = html || "<p>—</p>";
}

function renderFragments(fragments) {
    const cont = document.getElementById("r-fragments");
    if (!fragments || !fragments.length) {
        cont.innerHTML = "<p style='color:var(--text-muted);'>Подозрительные фрагменты не обнаружены.</p>";
        return;
    }
    cont.innerHTML = fragments.map(f => `<span class="fragment-pill">${escapeHtml(f)}</span>`).join("");
}

function renderText(text, fragments) {
    const cont = document.getElementById("r-text");
    if (!text) { cont.innerHTML = "<p style='color:var(--text-muted);'>Текст недоступен.</p>"; return; }
    let html = escapeHtml(text);
    const sorted = (fragments || []).filter(f => f && f.length >= 8).slice().sort((a, b) => b.length - a.length);
    for (const f of sorted) {
        const esc = escapeHtml(f);
        if (!esc || !html.includes(esc)) continue;
        if (html.includes(`<mark class="hl-veto">${esc}</mark>`) ||
            html.includes(`<mark class="hl-strong">${esc}</mark>`) ||
            html.includes(`<mark class="hl-ai">${esc}</mark>`)) continue;
        let cls = "hl-ai";
        if (f.length >= 120) cls = "hl-veto";
        else if (f.length >= 40) cls = "hl-strong";
        html = html.replace(esc, `<mark class="${cls}">${esc}</mark>`);
    }
    cont.innerHTML = html.replace(/\n/g, "<br>");
}

async function init() {
    const data = await fetchReport();
    if (!data) return;
    window._reportData = data;
    document.getElementById("r-title").textContent = data.title || "—";
    document.getElementById("r-date").textContent = data.created_at || "—";
    document.getElementById("r-source").textContent = data.source === "deep" ? "Глубокая проверка" : "Проверка текста";
    document.title = "Отчёт AI Radar — " + (data.title || "без названия");
    renderGauge(data.ai_score);

    const vetoEl = document.getElementById("r-veto");
    if (data.veto) {
        vetoEl.innerHTML = '<span class="veto-badge veto-triggered">ВЕТО АКТИВИРОВАНО</span>';
    } else {
        vetoEl.innerHTML = '<span class="veto-badge veto-passed">ПРОЙДЕНО</span>';
    }

    const integrityEl = document.getElementById("r-integrity");
    integrityEl.textContent = data.integrity || "—";
    integrityEl.style.color = data.integrity === "НАРУШЕНА" ? "var(--accent-danger)" : "var(--accent-success)";

    // Статус судьи в человекочитаемом виде
    const judgeStatusEl = document.getElementById("r-judge-status");
    const statusLabels = {
        "FULL": "Ансамбль + метасудья",
        "ENSEMBLE_ONLY": "Только ансамбль",
        "OFFLINE": "Только статистика",
    };
    judgeStatusEl.textContent = statusLabels[data.judge_status] || data.judge_status || "—";

    const m = data.metrics?.signal1 || data.metrics || {};
    renderMetrics(m);
    renderEnsemble(data.metrics);
    renderFragments(data.suspicious_fragments);
    renderText(data.text, data.suspicious_fragments);

    if (window.lucide) lucide.createIcons();
}

function downloadJSON() {
    if (!window._reportData) return;
    const blob = new Blob([JSON.stringify(window._reportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `AIRadar_Report_${window._reportData.id || 'scan'}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

document.addEventListener("DOMContentLoaded", init);
