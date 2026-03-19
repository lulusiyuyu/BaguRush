/* ============================================================
   BaguRush — 核心交互逻辑
   ============================================================ */

// === 全局状态 ===
const state = {
  sessionId: null,
  messages: [],
  plan: [],
  currentTopicIndex: 0,
  isInterviewing: false,
  isLoading: false,
  totalAsked: 0,
  maxQuestions: 8,
  resumeFile: null,
};

// === DOM 元素缓存 ===
const $ = (sel) => document.querySelector(sel);
const dom = {
  // Setup
  resumeInput: $('#resume-input'),
  uploadZone: $('#upload-zone'),
  uploadPlaceholder: $('#upload-placeholder'),
  uploadSuccess: $('#upload-success'),
  fileNameDisplay: $('#file-name-display'),
  nameInput: $('#name-input'),
  roleSelect: $('#role-select'),
  questionsSlider: $('#questions-slider'),
  qCountDisplay: $('#q-count-display'),
  followupSelect: $('#followup-select'),
  startBtn: $('#start-btn'),
  setupForm: $('#setup-form'),
  interviewInfo: $('#interview-info'),
  infoFile: $('#info-file'),
  infoRole: $('#info-role'),
  infoProgress: $('#info-progress'),

  // Header
  progressIndicator: $('#progress-indicator'),
  progressText: $('#progress-text'),
  endBtn: $('#end-btn'),
  endBtnSide: $('#end-btn-side'),

  // Chat
  welcomeView: $('#welcome-view'),
  messagesContainer: $('#messages-container'),
  typingIndicator: $('#typing-indicator'),
  inputBar: $('#input-bar'),
  answerTextarea: $('#answer-textarea'),
  submitBtn: $('#submit-btn'),

  // Right panel
  planSection: $('#plan-section'),
  planList: $('#plan-list'),
  evalCard: $('#eval-card'),
  evalScoreBadge: $('#eval-score-badge'),
  evalFeedback: $('#eval-feedback'),

  // Report
  reportView: $('#report-view'),
  reportStats: $('#report-stats'),
  reportContent: $('#report-content'),
  restartBtn: $('#restart-btn'),
  exportMdBtn: $('#export-md-btn'),

  // Pipeline
  agentPipeline: $('#agent-pipeline'),
  pipelineSteps: $('#pipeline-steps'),
  pipelineProgressFill: $('#pipeline-progress-fill'),
  pipelineLog: $('#pipeline-log'),
  llmStream: $('#llm-stream'),
  llmStreamContent: $('#llm-stream-content'),

  // Settings modal
  settingsBtn: $('#settings-btn'),
  settingsModal: $('#settings-modal'),
  settingsCloseBtn: $('#settings-close-btn'),
  settingsSaveBtn: $('#settings-save-btn'),
  settingsClearBtn: $('#settings-clear-btn'),
  settingsStatus: $('#settings-status'),
  apiProvider: $('#api-provider'),
  apiKeyInput: $('#api-key-input'),
  apiBaseUrl: $('#api-base-url'),
  apiModel: $('#api-model'),
  toggleKeyVis: $('#toggle-key-vis'),
};

// === Markdown renderer (markdown-it) ===
let md;
try {
  md = window.markdownit ? window.markdownit() : null;
} catch (e) {
  md = null;
}

// === API 配置管理（localStorage） ===

const API_CONFIG_KEY = 'bagurush_api_config';

const PROVIDER_DEFAULTS = {
  deepseek: { base_url: 'https://api.deepseek.com', model: 'deepseek-chat' },
  openai:   { base_url: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
  glm:      { base_url: 'https://open.bigmodel.cn/api/paas/v4', model: 'glm-4-flash' },
  custom:   { base_url: '', model: '' },
};

function loadApiConfig() {
  try {
    return JSON.parse(localStorage.getItem(API_CONFIG_KEY)) || null;
  } catch { return null; }
}

function saveApiConfig(cfg) {
  localStorage.setItem(API_CONFIG_KEY, JSON.stringify(cfg));
}

function clearApiConfig() {
  localStorage.removeItem(API_CONFIG_KEY);
}

/** 获取 LLM 配置的 HTTP 头，用于发送给后端 */
function getLlmHeaders() {
  const cfg = loadApiConfig();
  if (!cfg || !cfg.api_key) return {};
  return {
    'X-LLM-API-Key': cfg.api_key,
    'X-LLM-Base-URL': cfg.base_url || '',
    'X-LLM-Model': cfg.model || '',
  };
}

// === 工具函数 ===
function formatTime() {
  const d = new Date();
  return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function getScoreColor(score) {
  if (score >= 8) return 'var(--color-success)';
  if (score >= 5) return 'var(--color-warning)';
  return 'var(--color-danger)';
}

function scrollToBottom() {
  const c = dom.messagesContainer;
  if (c) {
    requestAnimationFrame(() => {
      c.scrollTop = c.scrollHeight;
    });
  }
}

// === DOM 渲染函数 ===

function renderInterviewerMessage(text) {
  const wrapper = document.createElement('div');
  wrapper.className = 'msg-row interviewer';
  wrapper.innerHTML = `
    <div class="avatar avatar-interviewer">🤖</div>
    <div class="msg-content">
      <div class="msg-interviewer">${escapeHtml(text)}</div>
      <span class="msg-time">${formatTime()}</span>
    </div>
  `;
  dom.typingIndicator.before(wrapper);
  state.messages.push({ role: 'interviewer', content: text });
  scrollToBottom();
}

function renderCandidateMessage(text) {
  const wrapper = document.createElement('div');
  wrapper.className = 'msg-row candidate';
  wrapper.innerHTML = `
    <div class="msg-content">
      <div class="msg-candidate">${escapeHtml(text)}</div>
      <span class="msg-time" style="text-align:right;">${formatTime()}</span>
    </div>
    <div class="avatar avatar-candidate">🧑‍💻</div>
  `;
  dom.typingIndicator.before(wrapper);
  state.messages.push({ role: 'candidate', content: text });
  scrollToBottom();
}

function renderSystemMessage(text) {
  const el = document.createElement('div');
  el.className = 'msg-system';
  el.textContent = text;
  dom.typingIndicator.before(el);
  scrollToBottom();
}

function renderPlan(planList) {
  state.plan = planList;
  dom.planSection.style.display = '';
  dom.planList.innerHTML = '';
  planList.forEach((item, i) => {
    const el = document.createElement('div');
    el.className = 'plan-item' + (i === state.currentTopicIndex ? ' active' : '');
    const icon = i < state.currentTopicIndex ? '✅' : (i === state.currentTopicIndex ? '●' : '○');
    el.innerHTML = `<span class="plan-icon">${icon}</span><span>${escapeHtml(item.topic || item.name || '话题 '+(i+1))}</span>`;
    dom.planList.appendChild(el);
  });
}

function updatePlanHighlight(idx) {
  state.currentTopicIndex = idx;
  const items = dom.planList.querySelectorAll('.plan-item');
  items.forEach((el, i) => {
    el.className = 'plan-item' + (i === idx ? ' active' : '') + (i < idx ? ' done' : '');
    const icon = i < idx ? '✅' : (i === idx ? '●' : '○');
    el.querySelector('.plan-icon').textContent = icon;
  });
}

function renderEvaluation(evalData) {
  if (!evalData) return;
  dom.evalCard.style.display = '';

  const overall = evalData.overall_score || 0;
  dom.evalScoreBadge.textContent = overall.toFixed(1);
  dom.evalScoreBadge.style.color = getScoreColor(overall);

  const dims = ['completeness', 'accuracy', 'depth', 'expression'];
  dims.forEach(d => {
    const val = evalData[d] || 0;
    const bar = $(`#bar-${d}`);
    const valEl = $(`#val-${d}`);
    if (bar) {
      bar.style.width = (val * 10) + '%';
      bar.style.background = getScoreColor(val);
    }
    if (valEl) valEl.textContent = val;
  });

  dom.evalFeedback.textContent = evalData.feedback || '';
}

function updateProgress(progressStr, difficulty) {
  let displayStr = progressStr;
  if (difficulty) {
    const diffLabel = {easy: '简单', medium: '中等', hard: '困难'}[difficulty] || difficulty;
    displayStr += ` · ${diffLabel}`;
  }
  dom.progressText.textContent = displayStr;
  dom.infoProgress.textContent = displayStr;
  const parts = progressStr.split('/');
  if (parts.length === 2) {
    state.totalAsked = parseInt(parts[0]) || 0;
    state.maxQuestions = parseInt(parts[1]) || 8;
  }
}

function renderReport(data) {
  // Stats cards
  const overall = data.overall_score || 0;
  const grade = data.grade || 'N/A';
  const numEvals = (data.evaluations || []).length;
  dom.reportStats.innerHTML = `
    <div class="stat-card">
      <div class="stat-label">综合得分</div>
      <div class="stat-value accent">${overall.toFixed(1)}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">评级</div>
      <div class="stat-value">${escapeHtml(grade)}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">题目数</div>
      <div class="stat-value">${numEvals}</div>
    </div>
  `;

  // Report markdown
  const reportMd = data.report || '暂无报告内容';
  if (md) {
    dom.reportContent.innerHTML = md.render(reportMd);
  } else {
    dom.reportContent.innerHTML = `<pre>${escapeHtml(reportMd)}</pre>`;
  }

  switchToReportView();
}

// === UI 状态切换 ===

function switchToInterviewMode() {
  dom.welcomeView.style.display = 'none';
  dom.messagesContainer.style.display = '';
  dom.inputBar.style.display = '';
  dom.setupForm.style.display = 'none';
  dom.interviewInfo.style.display = '';
  dom.progressIndicator.style.display = '';
  dom.endBtn.style.display = '';
  state.isInterviewing = true;

  dom.infoFile.textContent = state.resumeFile ? state.resumeFile.name : '-';
  dom.infoRole.textContent = dom.roleSelect.value;
}

function switchToReportView() {
  // hide main layout
  document.querySelector('.layout').style.display = 'none';
  document.querySelector('.header').style.display = 'none';
  dom.reportView.style.display = '';
}

function showTypingIndicator() {
  dom.typingIndicator.style.display = '';
  scrollToBottom();
}

function hideTypingIndicator() {
  dom.typingIndicator.style.display = 'none';
}

function setLoading(loading) {
  state.isLoading = loading;
  dom.submitBtn.disabled = loading;
  dom.answerTextarea.disabled = loading;
  if (loading) showTypingIndicator(); else hideTypingIndicator();
}

// === API 对接函数 ===

async function startInterview() {
  if (!state.resumeFile) {
    alert('请先上传简历文件');
    return;
  }

  dom.startBtn.disabled = true;
  dom.startBtn.textContent = '⏳ 正在规划面试...';

  // 隐藏设置表单，让流水线进度可见
  dom.setupForm.style.display = 'none';

  // 启动动态 Agent 流水线动画（每 2.5 秒推进一步）
  startPipelineAnimation(['解析简历', '检索岗位需求', '生成面试大纲', '生成第一题'], 2500);

  // 启动 LLM 数据流
  const streamId = generateStreamId();

  const formData = new FormData();
  formData.append('resume', state.resumeFile);
  formData.append('candidate_name', dom.nameInput.value || '候选人');
  formData.append('job_role', dom.roleSelect.value);
  formData.append('max_questions', dom.questionsSlider.value);
  formData.append('max_follow_ups', dom.followupSelect.value);

  // 先连接 SSE 再发请求
  connectLLMStream(streamId);

  try {
    const res = await fetch('/api/interview/start', {
      method: 'POST',
      body: formData,
      headers: { ...getLlmHeaders(), 'X-Stream-Id': streamId },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || '启动失败');
    }
    const data = await res.json();

    state.sessionId = data.session_id;
    state.maxQuestions = parseInt(dom.questionsSlider.value);

    // 停止动画，标记全部完成
    completePipeline(['解析简历', '检索岗位需求', '生成面试大纲', '生成第一题']);

    switchToInterviewMode();

    // 显示面试大纲
    if (data.interview_plan && data.interview_plan.length) {
      renderPlan(data.interview_plan);
      renderSystemMessage(`面试规划完成，共 ${data.interview_plan.length} 个话题`);
    }

    // 显示第一个问题
    if (data.first_question) {
      renderInterviewerMessage(data.first_question);
    }

    updateProgress(`0/${state.maxQuestions}`);
    dom.submitBtn.disabled = false;

  } catch (err) {
    alert('启动面试失败: ' + err.message);
    dom.startBtn.disabled = false;
    dom.startBtn.textContent = '🚀 开始面试';
    dom.setupForm.style.display = '';
    hidePipeline();
    hideLLMStream();
  }
}

async function submitAnswer() {
  const answer = dom.answerTextarea.value.trim();
  if (!answer || state.isLoading || !state.sessionId) return;

  renderCandidateMessage(answer);
  dom.answerTextarea.value = '';
  dom.answerTextarea.style.height = 'auto';
  setLoading(true);

  // 启动回答处理流水线动画
  startPipelineAnimation(['评估回答', '分析路由', '准备下一题'], 2000);

  // 启动 LLM 数据流
  const streamId = generateStreamId();
  connectLLMStream(streamId);

  try {
    const res = await fetch(`/api/interview/${state.sessionId}/answer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getLlmHeaders(), 'X-Stream-Id': streamId },
      body: JSON.stringify({ answer }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || '提交失败');
    }

    const data = await res.json();

    setLoading(false);

    // 停止动画，标记全部完成
    completePipeline(['评估回答', '分析路由', data.interview_ended ? '生成报告' : '准备下一题']);

    // 延迟后显示等待状态
    if (!data.interview_ended) {
      setTimeout(() => {
        showPipelineIdle();
      }, 1500);
    }

    // 更新评估
    if (data.evaluation) {
      renderEvaluation(data.evaluation);
    }

    // 更新进度
    if (data.progress) {
      updateProgress(data.progress, data.difficulty);
    }

    // 更新话题高亮（估算：通过 topic 名称匹配或递增）
    if (data.topic && state.plan.length) {
      const idx = state.plan.findIndex(p => p.topic === data.topic);
      if (idx >= 0) updatePlanHighlight(idx);
    }

    if (data.interview_ended) {
      renderSystemMessage('面试结束，正在生成报告...');
      setLoading(true);
      await showReport();
      setLoading(false);
    } else if (data.next_question) {
      const prefix = data.is_follow_up ? '（追问）' : '';
      renderInterviewerMessage(prefix + data.next_question);
    }

  } catch (err) {
    setLoading(false);
    hideLLMStream();
    renderSystemMessage('⚠️ 服务端错误: ' + err.message);
  }
}

async function showReport() {
  if (!state.sessionId) return;
  try {
    const res = await fetch(`/api/interview/${state.sessionId}/report`);
    if (!res.ok) {
      // 可能报告还没生成完，等一下重试
      await new Promise(r => setTimeout(r, 2000));
      const res2 = await fetch(`/api/interview/${state.sessionId}/report`);
      if (!res2.ok) throw new Error('获取报告失败');
      const data = await res2.json();
      renderReport(data);
      return;
    }
    const data = await res.json();
    renderReport(data);
  } catch (err) {
    renderSystemMessage('⚠️ 获取报告失败: ' + err.message);
  }
}

function resetAll() {
  state.sessionId = null;
  state.messages = [];
  state.plan = [];
  state.currentTopicIndex = 0;
  state.isInterviewing = false;
  state.isLoading = false;
  state.totalAsked = 0;
  state.resumeFile = null;

  // Reset UI
  dom.reportView.style.display = 'none';
  document.querySelector('.layout').style.display = '';
  document.querySelector('.header').style.display = '';
  dom.welcomeView.style.display = '';
  dom.messagesContainer.style.display = 'none';
  dom.inputBar.style.display = 'none';
  dom.setupForm.style.display = '';
  dom.interviewInfo.style.display = 'none';
  dom.progressIndicator.style.display = 'none';
  dom.endBtn.style.display = 'none';
  dom.planSection.style.display = 'none';
  dom.evalCard.style.display = 'none';
  hidePipeline();

  // Clear messages
  const msgs = dom.messagesContainer.querySelectorAll('.msg-row, .msg-system');
  msgs.forEach(m => m.remove());

  // Reset form
  dom.uploadPlaceholder.style.display = '';
  dom.uploadSuccess.style.display = 'none';
  dom.startBtn.disabled = true;
  dom.startBtn.textContent = '🚀 开始面试';
  dom.submitBtn.disabled = true;
  dom.answerTextarea.value = '';
}

// === 事件绑定 ===

// 文件上传
dom.uploadZone.addEventListener('click', () => dom.resumeInput.click());
dom.resumeInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFileSelect(file);
});

// 拖拽上传
dom.uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dom.uploadZone.classList.add('drag-over');
});
dom.uploadZone.addEventListener('dragleave', () => {
  dom.uploadZone.classList.remove('drag-over');
});
dom.uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dom.uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelect(file);
});

function handleFileSelect(file) {
  state.resumeFile = file;
  dom.uploadPlaceholder.style.display = 'none';
  dom.uploadSuccess.style.display = '';
  dom.fileNameDisplay.textContent = file.name;
  dom.startBtn.disabled = false;
}

// Slider
dom.questionsSlider.addEventListener('input', (e) => {
  dom.qCountDisplay.textContent = e.target.value;
});

// Start
dom.startBtn.addEventListener('click', startInterview);

// Submit
dom.submitBtn.addEventListener('click', submitAnswer);

// Ctrl+Enter
dom.answerTextarea.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') {
    e.preventDefault();
    submitAnswer();
  }
});

// Auto-resize textarea
dom.answerTextarea.addEventListener('input', () => {
  dom.answerTextarea.style.height = 'auto';
  dom.answerTextarea.style.height = Math.min(dom.answerTextarea.scrollHeight, 160) + 'px';
  dom.submitBtn.disabled = !dom.answerTextarea.value.trim() || state.isLoading;
});

// End interview buttons
async function endInterview() {
  if (!confirm('确定要结束当前面试吗？')) return;
  if (!state.sessionId) return;

  renderSystemMessage('面试手动结束，正在生成报告...');
  setLoading(true);
  startPipelineAnimation(['结束面试', '汇总评估', '生成报告'], 2000);

  // 启动 LLM 数据流
  const streamId = generateStreamId();
  connectLLMStream(streamId);

  try {
    const res = await fetch(`/api/interview/${state.sessionId}/end`, {
      method: 'POST',
      headers: { ...getLlmHeaders(), 'X-Stream-Id': streamId },
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || '结束面试失败');
    }

    const data = await res.json();
    completePipeline(['结束面试', '汇总评估', '生成报告']);
    setLoading(false);
    renderReport(data);

  } catch (err) {
    setLoading(false);
    hidePipeline();
    renderSystemMessage('⚠️ 生成报告失败: ' + err.message);
  }
}
dom.endBtn.addEventListener('click', endInterview);
dom.endBtnSide.addEventListener('click', endInterview);

// Restart
dom.restartBtn.addEventListener('click', resetAll);

// Export buttons
dom.exportMdBtn.addEventListener('click', exportMarkdown);

// === Agent 流水线进度函数 ===

let _pipelineTimer = null;

function showPipeline(steps, percent) {
  dom.agentPipeline.style.display = '';
  dom.pipelineSteps.innerHTML = '';
  steps.forEach(step => {
    const el = document.createElement('div');
    el.className = 'pipeline-step' + (step.status === 'done' ? ' done' : (step.status === 'active' ? ' active' : ''));
    const icon = step.status === 'done' ? '✅' : (step.status === 'active' ? '⏳' : '○');
    el.innerHTML = `<span class="pipeline-step-icon">${icon}</span><span>${step.label}${step.status === 'active' ? '<span class="pipeline-dots"></span>' : ''}</span>`;
    dom.pipelineSteps.appendChild(el);
  });
  dom.pipelineProgressFill.style.width = percent + '%';
}

/** 向流水线日志追加消息 */
function pipelineLogMsg(msg) {
  const el = document.createElement('div');
  el.className = 'pipeline-log-entry';
  const t = new Date().toLocaleTimeString('zh-CN', { hour:'2-digit', minute:'2-digit', second:'2-digit' });
  el.innerHTML = `<span class="log-time">[${t}]</span> <span class="log-msg">${escapeHtml(msg)}</span>`;
  dom.pipelineLog.appendChild(el);
  dom.pipelineLog.scrollTop = dom.pipelineLog.scrollHeight;
}

function clearPipelineLog() {
  dom.pipelineLog.innerHTML = '';
}

/** 启动动态流水线动画，逐步推进每个步骤 */
function startPipelineAnimation(stepLabels, intervalMs) {
  stopPipelineAnimation();
  const steps = stepLabels.map(label => ({ label, status: 'pending' }));
  let idx = 0;
  steps[0].status = 'active';
  showPipeline(steps, 5);
  pipelineLogMsg(`→ ${stepLabels[0]}...`);

  _pipelineTimer = setInterval(() => {
    if (idx < steps.length - 1) {
      steps[idx].status = 'done';
      pipelineLogMsg(`✓ ${steps[idx].label} 完成`);
      idx++;
      steps[idx].status = 'active';
      pipelineLogMsg(`→ ${steps[idx].label}...`);
      const pct = Math.min(85, 5 + Math.round((idx / steps.length) * 80));
      showPipeline(steps, pct);
    } else {
      clearInterval(_pipelineTimer);
      _pipelineTimer = null;
    }
  }, intervalMs);
}

function stopPipelineAnimation() {
  if (_pipelineTimer) {
    clearInterval(_pipelineTimer);
    _pipelineTimer = null;
  }
}

function completePipeline(stepLabels) {
  stopPipelineAnimation();
  const steps = stepLabels.map(label => ({ label, status: 'done' }));
  showPipeline(steps, 100);
  pipelineLogMsg(`✓ 全部完成`);
}

function showPipelineIdle() {
  stopPipelineAnimation();
  dom.agentPipeline.style.display = '';
  dom.pipelineSteps.innerHTML = '<div class="pipeline-step done"><span class="pipeline-step-icon">✅</span><span>等待你的回答</span></div>';
  dom.pipelineProgressFill.style.width = '100%';
}

function hidePipeline() {
  stopPipelineAnimation();
  dom.agentPipeline.style.display = 'none';
  clearPipelineLog();
  hideLLMStream();
}

// === 报告导出函数 ===

// 保存最后一份报告的原始 Markdown
let _lastReportMd = '';

const _origRenderReport = renderReport;
renderReport = function(data) {
  _lastReportMd = data.report || '';
  _origRenderReport(data);
};

function exportMarkdown() {
  const reportMd = _lastReportMd || dom.reportContent.textContent || '';
  if (!reportMd) { alert('暂无报告内容'); return; }
  const blob = new Blob([reportMd], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `BaguRush_Report_${(state.sessionId || 'unknown').slice(0, 8)}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

// === LLM 数据流 SSE ===

let _eventSource = null;
let _currentTokenSpan = null;

function generateStreamId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function connectLLMStream(streamId) {
  disconnectLLMStream();
  // 渐入显示
  dom.llmStream.style.display = '';
  dom.llmStream.classList.remove('fade-out');
  requestAnimationFrame(() => {
    dom.llmStream.classList.add('visible');
  });
  dom.llmStreamContent.textContent = '';
  _currentTokenSpan = null;

  _eventSource = new EventSource(`/api/interview/stream/${streamId}`);
  _eventSource.onmessage = (e) => {
    try {
      handleStreamEvent(JSON.parse(e.data));
    } catch {}
  };
  _eventSource.onerror = () => {
    disconnectLLMStream();
  };
}

function disconnectLLMStream() {
  if (_eventSource) {
    _eventSource.close();
    _eventSource = null;
  }
}

function hideLLMStream() {
  disconnectLLMStream();
  // 渐出隐藏
  dom.llmStream.classList.add('fade-out');
  dom.llmStream.classList.remove('visible');
  setTimeout(() => {
    if (!dom.llmStream.classList.contains('visible')) {
      dom.llmStream.style.display = 'none';
      dom.llmStream.classList.remove('fade-out');
    }
  }, 450);
}

function handleStreamEvent(evt) {
  const c = dom.llmStreamContent;
  switch (evt.type) {
    case 'llm_start': {
      const el = document.createElement('span');
      el.className = 'stream-llm-start';
      el.textContent = `🧠 ${evt.data || 'LLM'} →`;
      c.appendChild(el);
      _currentTokenSpan = document.createElement('span');
      _currentTokenSpan.className = 'stream-tokens';
      c.appendChild(_currentTokenSpan);
      break;
    }
    case 'token': {
      if (_currentTokenSpan) {
        _currentTokenSpan.textContent += evt.data;
        // 防止过长，只保留最近 600 字符
        if (_currentTokenSpan.textContent.length > 600) {
          _currentTokenSpan.textContent = '…' + _currentTokenSpan.textContent.slice(-500);
        }
      }
      break;
    }
    case 'llm_end': {
      const el = document.createElement('span');
      el.className = 'stream-done';
      el.textContent = ' ✓\n';
      c.appendChild(el);
      _currentTokenSpan = null;
      break;
    }
    case 'llm_error': {
      const el = document.createElement('span');
      el.className = 'stream-error';
      el.textContent = `✗ ${evt.data}`;
      c.appendChild(el);
      break;
    }
    case 'tool_start': {
      const el = document.createElement('span');
      el.className = 'stream-tool';
      el.textContent = `🔧 调用工具: ${evt.data}`;
      c.appendChild(el);
      break;
    }
    case 'tool_end': {
      const el = document.createElement('span');
      el.className = 'stream-done';
      el.textContent = ' ✓\n';
      c.appendChild(el);
      break;
    }
    case 'done': {
      disconnectLLMStream();
      // 2 秒后渐出隐藏
      setTimeout(() => {
        if (!_eventSource) hideLLMStream();
      }, 2000);
      break;
    }
  }
  c.scrollTop = c.scrollHeight;
}

// === 设置面板（API 密钥配置）===

function openSettings() {
  const cfg = loadApiConfig();
  if (cfg) {
    dom.apiProvider.value = cfg.provider || 'deepseek';
    dom.apiKeyInput.value = cfg.api_key || '';
    dom.apiBaseUrl.value = cfg.base_url || '';
    dom.apiModel.value = cfg.model || '';
  } else {
    dom.apiProvider.value = 'deepseek';
    dom.apiKeyInput.value = '';
    dom.apiBaseUrl.value = PROVIDER_DEFAULTS.deepseek.base_url;
    dom.apiModel.value = PROVIDER_DEFAULTS.deepseek.model;
  }
  dom.settingsStatus.textContent = '';
  dom.settingsStatus.className = 'settings-status';
  dom.settingsModal.style.display = '';
}

function closeSettings() {
  dom.settingsModal.style.display = 'none';
}

function showSettingsStatus(msg, isError) {
  dom.settingsStatus.textContent = msg;
  dom.settingsStatus.className = 'settings-status ' + (isError ? 'error' : 'success');
  setTimeout(() => { dom.settingsStatus.textContent = ''; }, 3000);
}

dom.settingsBtn.addEventListener('click', openSettings);
dom.settingsCloseBtn.addEventListener('click', closeSettings);
dom.settingsModal.addEventListener('click', (e) => {
  if (e.target === dom.settingsModal) closeSettings();
});

dom.apiProvider.addEventListener('change', () => {
  const p = dom.apiProvider.value;
  const defaults = PROVIDER_DEFAULTS[p] || PROVIDER_DEFAULTS.custom;
  dom.apiBaseUrl.value = defaults.base_url;
  dom.apiModel.value = defaults.model;
});

dom.toggleKeyVis.addEventListener('click', () => {
  const isPassword = dom.apiKeyInput.type === 'password';
  dom.apiKeyInput.type = isPassword ? 'text' : 'password';
  dom.toggleKeyVis.textContent = isPassword ? '🙈' : '👁️';
});

dom.settingsSaveBtn.addEventListener('click', () => {
  const key = dom.apiKeyInput.value.trim();
  if (!key) {
    showSettingsStatus('请输入 API Key', true);
    return;
  }
  saveApiConfig({
    provider: dom.apiProvider.value,
    api_key: key,
    base_url: dom.apiBaseUrl.value.trim(),
    model: dom.apiModel.value.trim(),
  });
  showSettingsStatus('✅ 已保存到浏览器');
});

dom.settingsClearBtn.addEventListener('click', () => {
  clearApiConfig();
  dom.apiKeyInput.value = '';
  dom.apiBaseUrl.value = '';
  dom.apiModel.value = '';
  showSettingsStatus('已清除配置');
});

// === 初始化 ===
{
  const cfg = loadApiConfig();
  if (!cfg || !cfg.api_key) {
    console.log('[BaguRush] ⚠️ 未配置 API Key，点击 ⚙️ 进行设置');
  }
}
console.log('[BaguRush] 🚀 前端已加载');
