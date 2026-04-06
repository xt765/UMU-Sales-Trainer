/**
 * AI Sales Trainer Chatbot - Frontend Application
 * 
 * This module handles the frontend logic for the AI sales training chatbot,
 * including session management, message handling, and semantic point coverage tracking.
 */

const API_BASE = '/api/v1';

const appState = {
  currentSessionId: null,
  isLoading: false,
  coverageData: []
};

const elements = {
  sessionIdDisplay: document.getElementById('sessionIdDisplay'),
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
        <h4>${escapeHtml(item.name)}</h4>
        <p>${escapeHtml(item.description)}</p>
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

async function createSession() {
  try {
    setLoading(true);
    
    const response = await fetch(`${API_BASE}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    appState.currentSessionId = data.session_id;
    
    elements.sessionIdDisplay.innerHTML = `<span class="session-id">${data.session_id}</span>`;
    
    elements.chatMessages.innerHTML = '';
    
    showToast('会话创建成功！', 'success');
    
    updateButtonStates();
    
    await sendMessageToAI('', true);
    
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
    
    appState.currentSessionId = null;
    elements.sessionIdDisplay.innerHTML = '<span class="session-empty">暂无活动会话</span>';
    elements.chatMessages.innerHTML = `
      <div class="message ai">
        <div class="message-avatar">AI</div>
        <div class="message-content">
          您好！我是您的AI销售训练助手。今天我们将模拟一场与张主任的销售对话。<br><br>
          张主任是内分泌科主任，关注糖尿病控制率和药物安全性，注重循证医学证据。<br><br>
          请开始您的销售开场...
        </div>
      </div>
    `;
    
    updateCoverageDisplay([
      { name: 'HbA1c 改善', description: '提及降低糖化血红蛋白效果', status: 'not-covered' },
      { name: '低血糖风险', description: '提及低血糖风险低', status: 'not-covered' },
      { name: '用药便利性', description: '提及一周一次给药', status: 'not-covered' }
    ]);
    
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
    addTypingIndicator();
    
    const payload = isInitial ? {} : { message };
    
    const response = await fetch(`${API_BASE}/sessions/${appState.currentSessionId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    removeTypingIndicator();
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.response) {
      addMessage(data.response, false);
    }
    
    if (data.semantic_coverage) {
      updateCoverageDisplay(data.semantic_coverage);
    }
    
    if (data.is_complete) {
      showToast('训练完成！', 'success');
    }
    
  } catch (error) {
    console.error('Error sending message:', error);
    removeTypingIndicator();
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
});
