/**
 * AI Sales Trainer Chatbot - Frontend Application
 * 
 * This module handles the frontend logic for the AI sales training chatbot,
 * including session management, message handling, and semantic point coverage tracking.
 */

const API_BASE = '/api/v1';

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
  messageDiv.className = `message ${isUser ? 'user' : 'ai'}`;
  
  const avatarDiv = document.createElement('div');
  avatarDiv.className = 'message-avatar';
  avatarDiv.textContent = isUser ? '我' : 'AI';
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
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
    const itemDiv = document.createElement('div');
    
    let statusClass = 'not-covered';
    if (item.status === 'covered') {
      statusClass = 'covered';
      coveredCount++;
    } else if (item.status === 'pending') {
      statusClass = 'pending';
      coveredCount += 0.5;
    }
    
    itemDiv.className = `coverage-item ${statusClass}`;
    itemDiv.style.animationDelay = `${index * 100}ms`;
    
    itemDiv.innerHTML = `
      <div class="item-info">
        <h4>${escapeHtml(item.description)}</h4>
      </div>
      <div class="item-status ${statusClass}"></div>
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

function updateExpressionDisplay(expression) {
  if (!expression) return;
  
  let exprContainer = document.getElementById('expressionAnalysis');
  if (!exprContainer) {
    const coverageSection = document.querySelector('.coverage-section');
    if (coverageSection) {
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
      coverageSection.parentNode.insertBefore(exprContainer, coverageSection.nextSibling);
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
  }
}

function updateOverallScore(score) {
  let scoreEl = document.getElementById('overallScoreValue');
  if (!scoreEl) {
    const coverageSection = document.querySelector('.coverage-section');
    if (coverageSection) {
      const scoreDiv = document.createElement('div');
      scoreDiv.className = 'overall-score-display';
      scoreDiv.innerHTML = `<span>综合评分: </span><strong id="overallScoreValue">-</strong>`;
      coverageSection.parentNode.insertBefore(scoreDiv, coverageSection.nextSibling);
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
  
  const profileSection = document.querySelector('.customer-profile');
  if (!profileSection) return;
  
  const name = customer.name || '客户';
  const firstChar = name.charAt(0);
  const position = customer.position || '';
  const hospital = customer.hospital || '';
  const concerns = customer.concerns || [];
  
  const avatarEl = profileSection.querySelector('.profile-avatar');
  if (avatarEl) avatarEl.textContent = firstChar;
  
  const nameEl = profileSection.querySelector('.profile-info h3');
  if (nameEl) {
    const title = position ? `${name} · ${position}` : name;
    nameEl.textContent = title;
  }
  
  const posEl = profileSection.querySelector('.profile-info p');
  if (posEl) posEl.textContent = hospital || '未知机构';
  
  const detailsEl = profileSection.querySelector('.profile-details');
  if (detailsEl && concerns.length > 0) {
    const iconMap = ['target', 'shield-check', 'file-text', 'heart-pulse', 'trending-up'];
    detailsEl.innerHTML = concerns.slice(0, 4).map((concern, i) => `
      <div class="detail-item">
        <i data-lucide="${iconMap[i % iconMap.length] || 'check'}" class="detail-icon"></i>
        <span>${escapeHtml(concern)}</span>
      </div>
    `).join('');
    lucide.createIcons();
  }
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
    
    const response = await fetch(`${API_BASE}/sessions/${appState.currentSessionId}`, {
      method: 'DELETE'
    });
    
    if (!response.ok && response.status !== 404) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const session = appState.sessions.find(s => s.id === appState.currentSessionId);
    if (session) {
      session.status = 'completed';
      session.endTime = new Date();
    }
    
    appState.currentSessionId = null;
    appState.messageCount = 0;
    
    updateTopBarSession();
    resetChatToInitialState();
    resetCoverageDisplay();
    
    showToast('会话已结束', 'success');
    
    updateButtonStates();
    
  } catch (error) {
    console.error('Error deleting session:', error);
    showToast('结束会话失败', 'error');
  } finally {
    setLoading(false);
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
        const SEMANTIC_LABELS = {
          'SP-001': 'HbA1c改善',
          'SP-002': '安全性',
          'SP-003': '服用方便',
          'SP_EFFICACY': '疗效数据',
          'SP_SAFETY': '安全性论证',
          'SP_CONVENIENCE': '用药便利'
        };
        
        const coverageData = Object.entries(data.evaluation.coverage_status).map(([id, status]) => ({
          name: id,
          description: SEMANTIC_LABELS[id] || id,
          status: status
        }));
        
        if (coverageData.length > 0) {
          updateCoverageDisplay(coverageData);
        }
      }
      
      if (data.evaluation.expression_analysis) {
        updateExpressionDisplay(data.evaluation.expression_analysis);
      }
      
      if (data.evaluation.overall_score !== undefined) {
        updateOverallScore(data.evaluation.overall_score);
      }
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
      <div class="message ai">
        <div class="message-avatar">AI</div>
        <div class="message-content">
          您好！我是您的AI销售训练助手。今天我们将模拟一场与张主任（内分泌科主任）关于<strong>糖宁胶囊</strong>的销售对话。<br><br>
          张主任关注：HbA1c改善效果、药物安全性、患者依从性，注重循证医学证据。<br><br>
          请开始您的销售开场...
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
    elements.sessionListContainer.innerHTML = '<p class="empty-list">暂无历史会话</p>';
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

function clearHistory() {
  if (!confirm('确定要清空所有历史会话吗？')) return;
  
  appState.sessions = [];
  appState.currentSessionId = null;
  appState.currentSessionNumber = 0;
  appState.messageCount = 0;
  
  updateTopBarSession();
  updateDrawerBadge();
  resetChatToInitialState();
  resetCoverageDisplay();
  updateButtonStates();
  loadSessionList();
  
  showToast('历史已清空', 'success');
  closeDrawer();
}
