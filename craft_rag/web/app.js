const $ = (selector) => document.querySelector(selector);
const form = $('#question-form');
const input = $('#question');
const askButton = $('#ask-button');
const statusPill = $('#status-pill');
const loading = $('#loading');
const welcome = $('#welcome');
const errorPanel = $('#error');
const answerPanel = $('#answer-panel');

function setStatus(status, message) {
  statusPill.className = `status-pill ${status}`;
  statusPill.querySelector('span').textContent = message;
}

async function checkStatus() {
  try {
    const response = await fetch('/api/status');
    const body = await response.json();
    if (!response.ok) throw new Error(body.message || 'Corpus could not be loaded');
    setStatus('ready', `${body.records} records ready`);
    return body;
  } catch (error) {
    setStatus('error', 'Corpus unavailable');
    throw error;
  }
}

function renderSources(sources) {
  const container = $('#sources');
  container.replaceChildren();
  if (!sources.length) {
    container.innerHTML = '<div class="welcome-card"><div><b>No matching evidence found</b><p>Try a broader question or clear the filters.</p></div></div>';
    return;
  }
  sources.forEach((source, index) => {
    const card = document.createElement('article');
    card.className = 'source-card';
    const title = document.createElement('b');
    title.textContent = source.title || 'Untitled source';
    const cluster = document.createElement('small');
    cluster.textContent = `${source.source_id || 'UNKNOWN'}${source.craft_cluster ? ` · ${source.craft_cluster}` : ''}`;
    const summary = document.createElement('button');
    summary.type = 'button';
    summary.className = 'source-summary';
    summary.setAttribute('aria-expanded', 'false');
    const number = document.createElement('span');
    number.className = 'source-number';
    number.textContent = String(index + 1).padStart(2, '0');
    const name = document.createElement('span');
    name.className = 'source-name';
    name.append(title, cluster);
    const badges = document.createElement('span');
    badges.className = 'badges';
    badges.innerHTML = `<span class="badge">${escapeHtml(source.evidence_level || 'E5')}</span><span class="badge grade">${escapeHtml(source.grade || 'C')}</span>`;
    const arrow = document.createElement('span');
    arrow.className = 'expand';
    arrow.textContent = '⌄';
    summary.append(number, name, badges, arrow);
    summary.addEventListener('click', () => {
      card.classList.toggle('open');
      summary.setAttribute('aria-expanded', String(card.classList.contains('open')));
    });

    const detail = document.createElement('div');
    detail.className = 'source-detail';
    const excerpt = document.createElement('p');
    excerpt.textContent = source.text || 'No text excerpt available.';
    detail.append(excerpt);
    const meta = document.createElement('div');
    meta.className = 'source-meta';
    [source.why_theme, source.process_stage].filter(Boolean).forEach((value) => {
      const tag = document.createElement('span');
      tag.textContent = value;
      meta.append(tag);
    });
    if (meta.childElementCount) detail.append(meta);
    if (source.caution) {
      const caution = document.createElement('p');
      caution.className = 'source-caution';
      caution.textContent = `Caution: ${source.caution}`;
      detail.append(caution);
    }
    if (source.link) {
      const link = document.createElement('a');
      link.className = 'source-link';
      link.href = source.link;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = 'Open source ↗';
      detail.append(link);
    }
    card.append(summary, detail);
    container.append(card);
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}

async function ask(question) {
  question = question.trim();
  if (!question) { input.focus(); return; }
  loading.classList.remove('hidden');
  welcome.classList.add('hidden');
  errorPanel.classList.add('hidden');
  answerPanel.classList.add('hidden');
  askButton.disabled = true;
  try {
    const response = await fetch('/api/query', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, craft_cluster: $('#cluster').value, why_theme: $('#theme').value})
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'Could not answer that question');
    $('#answer-text').textContent = body.answer;
    $('#result-count').textContent = `${body.sources.length} SOURCES`;
    $('#backend-note').textContent = `LOCAL RAG · ${Object.entries(body.backends).map(([key, value]) => `${key}: ${value}`).join(' · ')}`;
    renderSources(body.sources);
    answerPanel.classList.remove('hidden');
    setStatus('ready', 'Research corpus online');
  } catch (error) {
    errorPanel.textContent = error.message;
    errorPanel.classList.remove('hidden');
  } finally {
    loading.classList.add('hidden');
    askButton.disabled = false;
  }
}

form.addEventListener('submit', (event) => { event.preventDefault(); ask(input.value); });
document.querySelectorAll('[data-question]').forEach((button) => button.addEventListener('click', () => {
  input.value = button.dataset.question;
  ask(input.value);
}));

checkStatus().then(() => {
  // Populate exact filter values from the indexed corpus metadata.
  // The backend exposes values on demand through its metadata API.
  return fetch('/api/filters');
}).then((response) => response.json()).then((filters) => {
  [['cluster', filters.craft_clusters, 'All craft clusters'], ['theme', filters.why_themes, 'All themes']].forEach(([id, values]) => {
    const select = document.getElementById(id);
    values.forEach((value) => { const option = document.createElement('option'); option.value = value; option.textContent = value; select.append(option); });
  });
}).catch((error) => { console.warn('Prototype readiness check:', error); });
