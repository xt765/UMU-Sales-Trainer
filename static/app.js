/**
 * AI Sales Trainer Chatbot - Frontend Application
 * 
 * This module handles the frontend logic for the AI sales training chatbot,
 * including session management, message handling, and semantic point coverage tracking.
 */

const API_BASE = '/api/v1';

const STAGE_LABELS = {
  'opening': '开场破冰',
  'needs_discovery': '需求探查',
  'presentation': '产品呈现',
  'objection_handling': '异议处理',
  'closing': '缔结成交'
};

const DIMENSION_LABELS = {
  'clarity': '清晰度',
  'professionalism': '专业性',
  'persuasiveness': '说服力'
};

const appState = {
  currentSessionId: null,
  sessions: [],
  currentSessionNumber: 0,
  messageCount: 0,
  isLoading: false,
  coverageData: []
};

const elements = {
  topbarSessionInfo: document.getElementById('topbarSessionInfo'),
  sessionDrawer: document.getElementById('sessionDrawer'),
  drawerTrigger: document.getElementById('drawerTrigger'),
  drawerBadge: document.getElementById('drawerBadge'),
  sessionListContainer: document.getElementById('sessionListContainer'),
  btnNewChat: document.getElementById('btnNewChat'),
  btnDeleteSession: document.getElementById('btnDeleteSession'),
  btnSend: document.getElementById('btnSend'),
  chatInput: document.getElementById('chatInput'),
  inputForm: document.getElementById('inputForm'),
  chatMessages: document.getElementById('chatMessages'),
  coverageItems: document.getElementById('coverageItems'),
  progressValue: document.getElementById('progressValue'),
  progressFill: document.getElementById('progressFill'),
  toast: document.getElementById('toast'),
  loadingOverlay: document.getElementById('loadingOverlay')
};

function showToast(message, type = 'success') {
  const toast = elements.toast;
  toast.textContent = message;
  toast.className = `toast ${type} show`;
  
  setTimeout(() => {
    toast.classList.remove('show');
  }, 3000);
}

function setLoading(loading) {
  appState.isLoading = loading;
  
  if (loading) {
    elements.loadingOverlay.classList.add('show');
  } else {
    elements.loadingOverlay.classList.remove('show');
  }
  
  updateButtonStates();
}

function updateButtonStates() {
  const hasSession = !!appState.currentSessionId;
  
  elements.btnDeleteSession.disabled = !hasSession || appState.isLoading;
  elements.btnSend.disabled = !hasSession || appState.isLoading || !elements.chatInput.value.trim();
}

function formatTime(date) {
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function addMessage(content, isUser) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `msg ${isUser ? 'msg--user' : 'msg--ai'}`;

  const avatarDiv = document.createElement('div');
  avatarDiv.className = 'msg-avatar';
  avatarDiv.textContent = isUser ? '我' : 'AI';

  const contentDiv = document.createElement('div');
  contentDiv.className = 'msg-bubble';
  contentDiv.innerHTML = content.replace(/\n/g, '<br>');

  const timeSpan = document.createElement('div');
  timeSpan.className = 'message-time';
  timeSpan.textContent = formatTime(new Date());

  contentDiv.appendChild(timeSpan);
  messageDiv.appendChild(avatarDiv);
  messageDiv.appendChild(contentDiv);

  elements.chatMessages.appendChild(messageDiv);
  scrollToBottom();
}

function addTypingIndicator() {
  const typingDiv = document.createElement('div');
  typingDiv.className = 'typing-indicator';
  typingDiv.id = 'typingIndicator';
  
  const bubbleDiv = document.createElement('div');
  bubbleDiv.className = 'typing-bubble';
  
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement('div');
    dot.className = 'typing-dot';
    bubbleDiv.appendChild(dot);
  }
  
  typingDiv.appendChild(bubbleDiv);
  elements.chatMessages.appendChild(typingDiv);
  scrollToBottom();
}

function removeTypingIndicator() {
  const indicator = document.getElementById('typingIndicator');
  if (indicator) {
    indicator.remove();
  }
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
  });
}

function updateCoverageDisplay(coverageData) {
  if (!coverageData || coverageData.length === 0) return;

  appState.coverageData = coverageData;

  const itemsContainer = elements.coverageItems;
  itemsContainer.innerHTML = '';

  let coveredCount = 0;

  coverageData.forEach((item, index) => {
    let statusClass = 'cov-item--off';
    if (item.status === 'covered') {
      statusClass = 'cov-item--on';
      coveredCount++;
    } else if (item.status === 'pending') {
      statusClass = 'cov-item--partial';
      coveredCount += 0.5;
    }

    const itemDiv = document.createElement('div');
    itemDiv.className = `cov-item ${statusClass}`;
    itemDiv.style.animationDelay = `${index * 100}ms`;

    itemDiv.innerHTML = `
      <span class="cov-dot"></span>
      <div class="cov-info">
        <strong>${escapeHtml(item.description)}</strong>
      </div>
    `;

    itemsContainer.appendChild(itemDiv);
  });

  const percentage = Math.round((coveredCount / coverageData.length) * 100);
  elements.progressValue.textContent = `${percentage}%`;
  elements.progressFill.style.width = `${percentage}%`;

  lucide.createIcons();
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function updateExpressionDisplay(expression, suggestions) {
  if (!expression) return;
  
  let exprContainer = document.getElementById('expressionAnalysis');
  if (!exprContainer) {
    const coverageCard = document.querySelector('.coverage-card');
    if (coverageCard) {
      exprContainer = document.createElement('div');
      exprContainer.id = 'expressionAnalysis';
      exprContainer.className = 'expression-analysis';
      exprContainer.innerHTML = `
        <h3>表达能力分析</h3>
        <div class="expression-metrics">
          <div class="metric" data-metric="clarity">
            <span class="metric-label">清晰度</span>
            <span class="metric-value">-</span>
          </div>
          <div class="metric" data-metric="professionalism">
            <span class="metric-label">专业性</span>
            <span class="metric-value">-</span>
          </div>
          <div class="metric" data-metric="persuasiveness">
            <span class="metric-label">说服力</span>
            <span class="metric-value">-</span>
          </div>
        </div>
      `;
      coverageCard.parentNode.insertBefore(exprContainer, coverageCard.nextSibling);
    }
  }
  
  if (exprContainer) {
    const clarityEl = exprContainer.querySelector('[data-metric="clarity"] .metric-value');
    const proEl = exprContainer.querySelector('[data-metric="professionalism"] .metric-value');
    const persEl = exprContainer.querySelector('[data-metric="persuasiveness"] .metric-value');
    
    if (clarityEl) clarityEl.textContent = expression.clarity || 0;
    if (proEl) proEl.textContent = expression.professionalism || 0;
    if (persEl) persEl.textContent = expression.persuasiveness || 0;
    
    [clarityEl, proEl, persEl].forEach(el => {
      if (el) {
        const val = parseInt(el.textContent) || 0;
        el.className = 'metric-value ' + (val >= 7 ? 'high' : val >= 4 ? 'medium' : 'low');
      }
    });

    const existingSuggest = exprContainer.querySelector('.suggestion-list');
    if (existingSuggest) existingSuggest.remove();

    if (suggestions && suggestions.length > 0) {
      const suggestDiv = document.createElement('div');
      suggestDiv.className = 'suggestion-list';
      suggestDiv.innerHTML = '<h4>改进建议</h4>' +
        suggestions.map(s => {
          const dimLabel = DIMENSION_LABELS[s.dimension] || s.dimension;
          return `
          <div class="suggestion-item">
            <span class="suggestion-dim">${escapeHtml(dimLabel)}(${s.current_score}分)</span>
            <span class="suggestion-advice">${escapeHtml(s.advice)}</span>
            ${s.example ? `<div class="suggestion-example">${escapeHtml(s.example)}</div>` : ''}
          </div>`;
        }).join('');
      exprContainer.querySelector('.expression-metrics').after(suggestDiv);
    }
  }
}

function updateOverallScore(score) {
  let scoreEl = document.getElementById('overallScoreValue');
  if (!scoreEl) {
    const coverageCard = document.querySelector('.coverage-card');
    if (coverageCard) {
      const scoreDiv = document.createElement('div');
      scoreDiv.className = 'overall-score-display';
      scoreDiv.innerHTML = `<span>综合评分: </span><strong id="overallScoreValue">-</strong>`;
      coverageCard.parentNode.insertBefore(scoreDiv, coverageCard.nextSibling);
      scoreEl = document.getElementById('overallScoreValue');
    }
  }
  if (scoreEl) {
    scoreEl.textContent = Math.round(score);
    scoreEl.className = score >= 80 ? 'score-excellent' : score >= 60 ? 'score-good' : 'score-needs-work';
  }
}

function updateSidebarProfile(customer) {
  if (!customer) return;

  const profileSection = document.querySelector('.customer-card');
  if (!profileSection) return;

  const name = customer.name || '客户';
  const firstChar = name.charAt(0);
  const position = customer.position || '';
  const hospital = customer.hospital || '';
  const concerns = customer.concerns || [];

  const avatarEl = profileSection.querySelector('.avatar-text');
  if (avatarEl) avatarEl.textContent = firstChar;

  const nameEl = profileSection.querySelector('.customer-name');
  if (nameEl) {
    const title = position ? `${name} · ${position}` : name;
    nameEl.textContent = title;
  }

  const roleEl = profileSection.querySelector('.customer-role');
  if (roleEl) roleEl.textContent = hospital || '未知机构';

  const detailsEl = profileSection.querySelector('.concern-list');
  if (detailsEl && concerns.length > 0) {
    const iconMap = ['target', 'shield-check', 'file-text', 'heart-pulse', 'trending-up'];
    detailsEl.innerHTML = concerns.slice(0, 4).map((concern, i) => `
      <li class="concern-item">
        <i data-lucide="${iconMap[i % iconMap.length] || 'check'}" style="width:14px;height:14px"></i>
        <span>${escapeHtml(concern)}</span>
      </li>
    `).join('');
    lucide.createIcons();
  }
}

function updateConversationInsight(analysis) {
  if (!analysis || !analysis.stage) return;

  let insightEl = document.getElementById('conversationInsight');
  if (!insightEl) {
    const customerCard = document.querySelector('.customer-card');
    const coverageCard = document.querySelector('.coverage-card');

    if (customerCard && coverageCard) {
      insightEl = document.createElement('div');
      insightEl.id = 'conversationInsight';
      insightEl.className = 'conversation-insight';
      customerCard.parentNode.insertBefore(insightEl, coverageCard);
    }
  }

  if (!insightEl) return;

  const stageLabel = STAGE_LABELS[analysis.stage] || analysis.stage;
  const stageClass = `stage-${analysis.stage}`;
  const sentimentIcon = analysis.sentiment === 'positive' ? 'thumbs-up' : analysis.sentiment === 'cautious' ? 'alert-circle' : 'minus-circle';

  let objectionsHtml = '';
  if (analysis.objections && analysis.objections.length > 0) {
    objectionsHtml = '<div class="objection-tags">' +
      analysis.objections.map(obj => `<span class="objection-tag">${escapeHtml(obj)}</span>`).join('') +
      '</div>';
  }

  insightEl.innerHTML = `
    <div class="insight-header">
      <span class="stage-badge ${stageClass}">${stageLabel}</span>
      <i data-lucide="${sentimentIcon}" class="sentiment-icon"></i>
    </div>
    ${analysis.intent ? `<p class="intent-text">${escapeHtml(analysis.intent)}</p>` : ''}
    ${objectionsHtml}
  `;
  insightEl.style.display = 'block';
  lucide.createIcons();
}

async function createSession() {
  try {
    setLoading(true);
    
    const response = await fetch(`${API_BASE}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        customer_profile: {
          name: "张医生",
          position: "内分泌科主任",
          hospital: "市第一人民医院",
          personality_type: "ANALYTICAL"
        },
        product_info: {
          product_name: "糖宁胶囊",
          category: "降血糖药物",
          core_benefits: [
            "HbA1c改善: 平均降低0.8%",
            "安全性高: 低血糖风险<1%",
            "服用方便: 每日一次"
          ],
          key_selling_points: {
            SP_EFFICACY: {
              description: "显著降低HbA1c水平",
              keywords: ["HbA1c", "降糖", "疗效"],
              weight: 1.0
            },
            SP_SAFETY: {
              description: "低血糖风险极低",
              keywords: ["安全", "副作用", "低血糖"],
              weight: 0.9
            },
            SP_CONVENIENCE: {
              description: "每日一次服药便利",
              keywords: ["方便", "依从性", "每日"],
              weight: 0.8
            }
          }
        }
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    appState.currentSessionId = data.session_id;
    appState.currentSessionNumber += 1;
    appState.messageCount = 0;
    
    appState.sessions.unshift({
      id: data.session_id,
      number: appState.currentSessionNumber,
      startTime: new Date(),
      turns: 0,
      status: 'active',
      coverage: 0
    });
    
    updateTopBarSession();
    updateDrawerBadge();
    clearChatMessages();
    resetChatToInitialState();
    resetCoverageDisplay();
    
    updateSidebarProfile({
      name: "张医生",
      position: "内分泌科主任",
      hospital: "市第一人民医院",
      concerns: ["HbA1c控制效果", "低血糖风险", "患者依从性"]
    });
    
    const exprEl = document.getElementById('expressionAnalysis');
    if (exprEl) exprEl.remove();
    const scoreEl = document.querySelector('.overall-score-display');
    if (scoreEl) scoreEl.remove();
    
    showToast('会话创建成功！', 'success');

    setLoading(false);
    updateButtonStates();
    
  } catch (error) {
    console.error('Error creating session:', error);
    showToast('创建会话失败，请重试', 'error');
  } finally {
    setLoading(false);
  }
}

async function deleteSession() {
  if (!appState.currentSessionId) return;

  try {
    setLoading(true);

    const response = await fetch(`${API_BASE}/sessions/${appState.currentSessionId}?hard=true`, {
      method: 'DELETE'
    });

    if (!response.ok && response.status !== 404) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    appState.sessions = appState.sessions.filter(s => s.id !== appState.currentSessionId);
    appState.currentSessionId = null;
    appState.messageCount = 0;

    updateTopBarSession();
    updateDrawerBadge();
    resetChatToInitialState();
    resetCoverageDisplay();

    const exprEl = document.getElementById('expressionAnalysis');
    if (exprEl) exprEl.remove();
    const scoreEl = document.querySelector('.overall-score-display');
    if (scoreEl) scoreEl.remove();

    showToast('会话已永久删除', 'success');
    updateButtonStates();
    loadSessionList();

  } catch (error) {
    console.error('Error deleting session:', error);
    showToast('删除会话失败，请重试', 'error');
  } finally {
    setLoading(false);
  }
}

function updateGuidancePanel(guidance) {
  let panel = document.getElementById('guidancePanel');
  if (!panel) {
    panel = document.createElement('div');
    panel.id = 'guidancePanel';
    const anchor = document.getElementById('guidancePanelAnchor');
    if (anchor) {
      anchor.appendChild(panel);
    } else {
      elements.chatMessages.parentNode.insertBefore(panel, elements.chatMessages);
    }
  }

  if (!guidance || !guidance.is_actionable) {
    panel.className = 'guidance-panel guidance-panel--excellent';
    const summary = guidance?.summary || '表现优秀';
    const score = guidance?.overall_score ?? 0;
    panel.innerHTML = `
      <div class="guidance-header" onclick="toggleGuidance()">
        <span class="guidance-summary">
          <i data-lucide="award"></i>
          <span class="excellent-badge">${escapeHtml(summary)}${score > 0 ? `（${score}分）` : ''}</span>
        </span>
        <i data-lucide="chevron-down" class="guidance-toggle-icon"></i>
      </div>
      <div class="guidance-body excellent-detail" style="display:none">
        ${guidance?.priority_list ? guidance.priority_list.map(item => `
          <div class="guidance-item urgency-${item.urgency || 'low'}">
            <div class="gap-text">${escapeHtml(item.gap)}</div>
            <div class="suggestion-text">${escapeHtml(item.suggestion)}</div>
            ${item.talking_point ? `<div class="talking-point">${escapeHtml(item.talking_point)}</div>` : ''}
          </div>
        `).join('') : '<p>语义覆盖全面，表达能力均衡，暂无紧急改进项。</p>'}
      </div>
    `;
  } else {
    panel.className = 'guidance-panel';
    const urgencyIcons = { high: 'alert-triangle', medium: 'alert-circle', low: 'info' };

    panel.innerHTML = `
      <div class="guidance-header" onclick="toggleGuidance()">
        <span class="guidance-summary">
          <i data-lucide="lightbulb"></i>
          ${escapeHtml(guidance.summary)}
        </span>
        <i data-lucide="chevron-up" class="guidance-toggle-icon"></i>
      </div>
      <div class="guidance-body">
        ${guidance.priority_list.map(item => `
          <div class="guidance-item urgency-${item.urgency}">
            <div class="gap-text">${escapeHtml(item.gap)}</div>
            <div class="suggestion-text">${escapeHtml(item.suggestion)}</div>
            ${item.talking_point ? `<div class="talking-point">${escapeHtml(item.talking_point)}</div>` : ''}
          </div>
        `).join('')}
      </div>
    `;
  }

  panel.style.display = 'block';
  lucide.createIcons();
}

function toggleGuidance() {
  const body = document.querySelector('.guidance-body');
  const icon = document.querySelector('.guidance-toggle-icon');
  if (body) {
    body.style.display = body.style.display === 'none' ? 'block' : 'none';
    if (icon) icon.setAttribute('data-lucide', body.style.display === 'none' ? 'chevron-down' : 'chevron-up');
    lucide.createIcons();
  }
}

async function sendMessageToAI(message, isInitial = false) {
  if (!appState.currentSessionId) return;
  
  try {
    if (!isInitial) addTypingIndicator();
    
    const payload = isInitial ? { content: "" } : { content: message };
    
    const response = await fetch(`${API_BASE}/sessions/${appState.currentSessionId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!isInitial) removeTypingIndicator();
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.ai_response && data.ai_response.trim()) {
      addMessage(data.ai_response, false);
    }
    
    if (data.evaluation) {
      appState.lastEvaluation = data.evaluation;

      if (data.evaluation.coverage_status) {
        const labels = data.evaluation.coverage_labels || {};

        const coverageData = Object.entries(data.evaluation.coverage_status).map(([id, status]) => ({
          name: id,
          description: labels[id] || id,
          status: status
        }));

        if (coverageData.length > 0) {
          updateCoverageDisplay(coverageData);
        }
      }

      if (data.evaluation.expression_analysis) {
        updateExpressionDisplay(data.evaluation.expression_analysis, data.evaluation.suggestions);
      }

      if (data.evaluation.overall_score !== undefined) {
        updateOverallScore(data.evaluation.overall_score);
      }

      if (data.evaluation.conversation_analysis) {
        updateConversationInsight(data.evaluation.conversation_analysis);
      }
    }

    if (data.guidance) {
      updateGuidancePanel(data.guidance);
    } else {
      updateGuidancePanel(null);
    }
    
    appState.messageCount++;
    updateTopBarSession();
    
    if (data.is_complete) {
      showToast('训练完成！', 'success');
    }
    
  } catch (error) {
    console.error('Error sending message:', error);
    if (!isInitial) removeTypingIndicator();
    showToast('发送消息失败，请重试', 'error');
  }
}

async function handleSendMessage(e) {
  e.preventDefault();
  
  const message = elements.chatInput.value.trim();
  if (!message || !appState.currentSessionId || appState.isLoading) return;
  
  addMessage(message, true);
  elements.chatInput.value = '';
  elements.chatInput.style.height = 'auto';
  
  updateButtonStates();
  
  updateMessageCount();
  
  await sendMessageToAI(message);
}

function autoResizeTextarea() {
  const textarea = elements.chatInput;
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
}

elements.btnNewChat.addEventListener('click', createSession);
elements.btnDeleteSession.addEventListener('click', deleteSession);
elements.inputForm.addEventListener('submit', handleSendMessage);

elements.chatInput.addEventListener('input', () => {
  updateButtonStates();
  autoResizeTextarea();
});

elements.chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!elements.btnSend.disabled) {
      handleSendMessage(e);
    }
  }
});

document.addEventListener('DOMContentLoaded', () => {
  updateButtonStates();
  initDrawerHover();
  restoreSessionsFromBackend();
});

async function restoreSessionsFromBackend() {
  try {
    const resp = await fetch(`${API_BASE}/sessions`);
    if (!resp.ok) return;
    
    const data = await resp.json();
    
    if (data.sessions && data.sessions.length > 0) {
      appState.sessions = data.sessions.map((s, idx) => ({
        id: s.session_id,
        number: idx + 1,
        startTime: new Date(s.created_at),
        turns: s.message_count || 0,
        status: s.status || 'active',
        coverage: 0
      }));
      
      const activeSession = appState.sessions.find(s => s.status === 'active');
      
      if (activeSession && !appState.currentSessionId) {
        appState.currentSessionId = activeSession.id;
        appState.currentSessionNumber = activeSession.number;
        appState.messageCount = Math.floor(activeSession.turns / 2);
        
        try {
          const msgsResp = await fetch(`${API_BASE}/sessions/${activeSession.id}/messages`);
          if (msgsResp.ok) {
            const msgsData = await msgsResp.json();
            clearChatMessages();
            if (msgsData.messages && msgsData.messages.length > 0) {
              for (const msg of msgsData.messages) {
                addMessage(msg.content, msg.role === 'user', false);
              }
            } else {
              resetChatToInitialState();
            }
          }
          
          const evalResp = await fetch(`${API_BASE}/sessions/${activeSession.id}/evaluation`);
          if (evalResp.ok) {
            const evalData = await evalResp.json();
            if (evalData.coverage_status) {
        const coverageData = Object.entries(evalData.coverage_status).map(([id, status]) => ({
          name: id,
          description: evalData.coverage_labels?.[id] || id,
          status: status
        }));
              if (coverageData.length > 0) updateCoverageDisplay(coverageData);
            }
            if (evalData.expression_analysis) updateExpressionDisplay(evalData.expression_analysis);
            if (evalData.overall_score !== undefined) updateOverallScore(evalData.overall_score);
          }
        } catch (e) {
          console.warn('Failed to restore session details:', e);
          resetChatToInitialState();
          resetCoverageDisplay();
        }
        
        updateTopBarSession();
        updateSidebarProfile({
          name: "张医生",
          position: "内分泌科主任",
          hospital: "市第一人民医院",
          concerns: ["HbA1c控制效果", "低血糖风险", "患者依从性"]
        });
      }
      
      updateDrawerBadge();
      loadSessionList();
    }
  } catch (e) {
    console.warn('Failed to restore sessions from backend:', e);
  }
}

function updateTopBarSession() {
  if (!elements.topbarSessionInfo) return;
  
  if (!appState.currentSessionId) {
    elements.topbarSessionInfo.innerHTML = `
      <span class="no-session">暂无活动会话</span>
    `;
    return;
  }

  const session = appState.sessions.find(s => s.id === appState.currentSessionId);
  
  elements.topbarSessionInfo.innerHTML = `
    <div class="session-status-card">
      <span class="status-dot active"></span>
      <strong>训练 #${appState.currentSessionNumber}</strong>
      <span class="meta-separator">·</span>
      <span class="status-label">进行中</span>
      <span class="meta-separator">·</span>
      <span class="turn-count">第 ${appState.messageCount} 轮</span>
    </div>
  `;
}

function clearChatMessages() {
  if (elements.chatMessages) {
    elements.chatMessages.innerHTML = '';
  }
}

function resetChatToInitialState() {
  if (elements.chatMessages) {
    elements.chatMessages.innerHTML = `
      <div class="msg msg--ai">
        <div class="msg-avatar">AI</div>
        <div class="msg-bubble">
          您好！我是您的 AI 销售训练助手。今天我们将模拟一场与张主任（内分泌科主任）关于<strong>糖宁胶囊</strong>的销售对话。<br><br>
          张主任关注：HbA1c改善效果、药物安全性、患者依从性，注重循证医学证据。<br><br>
          请开始您的销售开场…
        </div>
      </div>
    `;
  }
}

function resetCoverageDisplay() {
  updateCoverageDisplay([
    { name: 'SP-001', description: 'HbA1c改善', status: 'not-covered' },
    { name: 'SP-002', description: '安全性', status: 'not-covered' },
    { name: 'SP-003', description: '服用方便', status: 'not-covered' }
  ]);
}

function updateMessageCount() {
  appState.messageCount += 1;
  
  const session = appState.sessions.find(s => s.id === appState.currentSessionId);
  if (session) {
    session.turns = appState.messageCount;
  }
  
  updateTopBarSession();
  loadSessionList();
}

function updateDrawerBadge() {
  if (elements.drawerBadge) {
    elements.drawerBadge.textContent = appState.sessions.length.toString();
  }
}

function initDrawerHover() {
  const drawer = elements.sessionDrawer;
  const trigger = elements.drawerTrigger;
  
  if (!drawer || !trigger) return;
  
  let hideTimer = null;

  trigger.addEventListener('mouseenter', () => {
    clearTimeout(hideTimer);
    drawer.classList.add('visible');
    loadSessionList();
  });

  drawer.addEventListener('mouseleave', () => {
    hideTimer = setTimeout(() => {
      if (!drawer.matches(':hover')) {
        drawer.classList.remove('visible');
      }
    }, 300);
  });

  drawer.addEventListener('mouseenter', () => {
    clearTimeout(hideTimer);
  });
}

function closeDrawer() {
  if (elements.sessionDrawer) {
    elements.sessionDrawer.classList.remove('visible');
  }
}

function loadSessionList() {
  if (!elements.sessionListContainer) return;
  
  if (appState.sessions.length === 0) {
    elements.sessionListContainer.innerHTML = '<p class="drawer-empty">暂无历史会话</p>';
    return;
  }
  
  const html = appState.sessions.map(session => {
    const timeStr = formatTime(new Date(session.startTime));
    const isActive = session.id === appState.currentSessionId;
    
    return `
      <div class="session-list-item ${isActive ? 'active' : ''}" 
           data-session-id="${session.id}"
           onclick="switchSession('${session.id}')">
        <div class="item-main">
          <span class="item-number">训练 #${session.number}</span>
          <span class="item-time">${timeStr}</span>
        </div>
        <div class="item-meta">
          <span class="item-turns">${session.turns}轮对话</span>
          <span class="item-coverage">覆盖率 ${session.coverage}%</span>
        </div>
        <div class="item-status-badge ${session.status}">${session.status === 'active' ? '进行中' : '已完成'}</div>
      </div>
    `;
  }).join('');
  
  elements.sessionListContainer.innerHTML = html;
  
  lucide.createIcons();
}

async function switchSession(sessionId) {
  if (sessionId === appState.currentSessionId) {
    closeDrawer();
    return;
  }
  
  try {
    setLoading(true);
    closeDrawer();
    
    const [msgsResp, evalResp] = await Promise.all([
      fetch(`${API_BASE}/sessions/${sessionId}/messages`),
      fetch(`${API_BASE}/sessions/${sessionId}/evaluation`),
    ]);
    
    if (!msgsResp.ok) throw new Error(`Messages HTTP ${msgsResp.status}`);
    
    const msgsData = await msgsResp.json();
    const evalData = evalResp.ok ? await evalResp.json() : null;
    
    appState.currentSessionId = sessionId;
    
    const session = appState.sessions.find(s => s.id === sessionId);
    if (session) {
      appState.currentSessionNumber = session.number;
      appState.messageCount = session.turns || 0;
    } else {
      const idx = appState.sessions.findIndex(s => s.id === sessionId);
      appState.currentSessionNumber = idx >= 0 ? idx + 1 : appState.sessions.length + 1;
      appState.messageCount = msgsData.total ? Math.floor(msgsData.total / 2) : 0;
    }
    
    clearChatMessages();
    
    if (msgsData.messages && msgsData.messages.length > 0) {
      for (const msg of msgsData.messages) {
        const isUser = msg.role === 'user';
        addMessage(msg.content, isUser, false);
      }
    } else {
      resetChatToInitialState();
    }
    
    resetCoverageDisplay();
    
    const exprEl = document.getElementById('expressionAnalysis');
    if (exprEl) exprEl.remove();
    const scoreEl = document.querySelector('.overall-score-display');
    if (scoreEl) scoreEl.remove();
    
    if (evalData && evalData.coverage_status) {
      const coverageData = Object.entries(evalData.coverage_status).map(([id, status]) => ({
        name: id,
        description: evalData.coverage_labels?.[id] || id,
        status: status
      }));
      
      if (coverageData.length > 0) updateCoverageDisplay(coverageData);
    }
    
    if (evalData && evalData.expression_analysis) {
      updateExpressionDisplay(evalData.expression_analysis);
    }
    
    if (evalData && evalData.overall_score !== undefined) {
      updateOverallScore(evalData.overall_score);
    }
    
    updateTopBarSession();
    updateSidebarProfile({
      name: "张医生",
      position: "内分泌科主任",
      hospital: "市第一人民医院",
      concerns: ["HbA1c控制效果", "低血糖风险", "患者依从性"]
    });
    updateButtonStates();
    loadSessionList();
    
    showToast(`已切换到训练 #${appState.currentSessionNumber}`, 'success');
    
  } catch (error) {
    console.error('Error switching session:', error);
    showToast('切换会话失败，请重试', 'error');
  } finally {
    setLoading(false);
  }
}

async function clearHistory() {
  if (!confirm('确定要永久删除所有历史会话吗？此操作不可恢复！')) return;

  try {
    setLoading(true);

    const response = await fetch(`${API_BASE}/sessions?hard=true`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    appState.sessions = [];
    appState.currentSessionId = null;
    appState.currentSessionNumber = 0;
    appState.messageCount = 0;

    updateTopBarSession();
    updateDrawerBadge();
    resetChatToInitialState();
    resetCoverageDisplay();

    const exprEl = document.getElementById('expressionAnalysis');
    if (exprEl) exprEl.remove();
    const scoreEl = document.querySelector('.overall-score-display');
    if (scoreEl) scoreEl.remove();

    updateButtonStates();
    loadSessionList();

    showToast('全部历史已永久删除', 'success');
    closeDrawer();

  } catch (error) {
    console.error('Error clearing history:', error);
    showToast('清空历史失败，请重试', 'error');
  } finally {
    setLoading(false);
  }
}
