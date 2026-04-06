const API_BASE_URL = 'http://localhost:8000/api/v1';

const state = {
    currentSessionId: null,
    messages: [],
    isLoading: false,
    coverageStatus: {
        'hba1c': 'not-covered',
        'hypoglycemia': 'not-covered',
        'convenience': 'not-covered'
    },
    progress: 0
};

let elements = {};

function initApp() {
    cacheElements();
    bindEventListeners();
    updateUI();
}

function cacheElements() {
    elements = {
        chatMessages: document.getElementById('chatMessages'),
        chatInput: document.getElementById('chatInput'),
        btnSend: document.getElementById('btnSend'),
        btnNewChat: document.getElementById('btnNewChat'),
        btnDeleteSession: document.getElementById('btnDeleteSession'),
        progressFill: document.getElementById('progressFill'),
        progressText: document.getElementById('progressText'),
        sessionIdDisplay: document.getElementById('sessionIdDisplay'),
        coverageItems: document.getElementById('coverageItems'),
        toast: document.getElementById('toast'),
        loadingOverlay: document.getElementById('loadingOverlay'),
        inputForm: document.getElementById('inputForm')
    };
}

function bindEventListeners() {
    if (elements.btnSend) {
        elements.btnSend.addEventListener('click', handleSendMessage);
    }
    if (elements.inputForm) {
        elements.inputForm.addEventListener('submit', (e) => {
            e.preventDefault();
            handleSendMessage();
        });
    }
    if (elements.btnNewChat) {
        elements.btnNewChat.addEventListener('click', handleCreateSession);
    }
    if (elements.btnDeleteSession) {
        elements.btnDeleteSession.addEventListener('click', handleDeleteSession);
    }
    if (elements.chatInput) {
        elements.chatInput.addEventListener('input', handleInputChange);
        elements.chatInput.addEventListener('keydown', handleKeyDown);
    }
}

function handleInputChange() {
    autoResize();
    elements.btnSend.disabled = !elements.chatInput.value.trim() || !state.currentSessionId;
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
    }
}

function autoResize() {
    if (elements.chatInput) {
        elements.chatInput.style.height = 'auto';
        elements.chatInput.style.height = Math.min(elements.chatInput.scrollHeight, 120) + 'px';
    }
}

async function handleCreateSession() {
    try {
        setLoading(true);
        const session = await createSession();
        state.currentSessionId = session.id;
        state.messages = [];
        state.coverageStatus = {
            'hba1c': 'not-covered',
            'hypoglycemia': 'not-covered',
            'convenience': 'not-covered'
        };
        state.progress = 0;
        clearMessages();
        addMessage({
            role: 'ai',
            content: '您好！我是您的AI销售训练助手。今天我们将模拟一场与张主任的销售对话。\n\n张主任是内分泌科主任，关注糖尿病控制率和药物安全性，注重循证医学证据。\n\n请开始您的销售开场...'
        });
        updateUI();
        showToast('会话创建成功', 'success');
    } catch (error) {
        showToast(`创建会话失败: ${error.message}`, 'error');
    } finally {
        setLoading(false);
    }
}

async function handleSendMessage() {
    const message = elements.chatInput?.value.trim();
    if (!message) return;

    if (!state.currentSessionId) {
        showToast('请先创建会话', 'warning');
        return;
    }

    try {
        setLoading(true);
        addMessage({ role: 'user', content: message });
        elements.chatInput.value = '';
        elements.chatInput.style.height = 'auto';
        elements.btnSend.disabled = true;

        const response = await sendMessage(state.currentSessionId, message);
        addMessage({ role: 'ai', content: response.ai_response });
        
        // Update coverage status
        if (response.pending_points) {
            updateCoverageStatus(response.pending_points);
        }
        
        await fetchSessionStatus(state.currentSessionId);
        updateUI();
    } catch (error) {
        showToast(`发送消息失败: ${error.message}`, 'error');
    } finally {
        setLoading(false);
    }
}

async function handleDeleteSession() {
    if (!state.currentSessionId) return;
    if (!confirm('确定要结束当前会话吗？')) return;

    try {
        setLoading(true);
        await deleteSession(state.currentSessionId);
        state.currentSessionId = null;
        state.messages = [];
        state.coverageStatus = {
            'hba1c': 'not-covered',
            'hypoglycemia': 'not-covered',
            'convenience': 'not-covered'
        };
        state.progress = 0;
        clearMessages();
        addMessage({
            role: 'ai',
            content: '您好！我是您的AI销售训练助手。今天我们将模拟一场与张主任的销售对话。\n\n张主任是内分泌科主任，关注糖尿病控制率和药物安全性，注重循证医学证据。\n\n请开始您的销售开场...'
        });
        updateUI();
        showToast('会话已结束', 'success');
    } catch (error) {
        showToast(`删除会话失败: ${error.message}`, 'error');
    } finally {
        setLoading(false);
    }
}

function addMessage(message) {
    state.messages.push(message);
    renderMessages();
}

function clearMessages() {
    state.messages = [];
    renderMessages();
}

function renderMessages() {
    if (!elements.chatMessages) return;

    const html = state.messages.map(msg => `
        <div class="message ${msg.role}">
            <div class="message-avatar">${msg.role === 'ai' ? 'AI' : '我'}</div>
            <div class="message-content">${escapeHtml(msg.content)}</div>
        </div>
    `).join('');

    elements.chatMessages.innerHTML = html;
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
}

function updateCoverageStatus(pendingPoints) {
    // Reset all to not-covered first
    Object.keys(state.coverageStatus).forEach(key => {
        state.coverageStatus[key] = 'not-covered';
    });

    // Mark pending points
    pendingPoints.forEach(point => {
        if (state.coverageStatus[point]) {
            state.coverageStatus[point] = 'pending';
        }
    });

    // Calculate progress
    const coveredCount = Object.values(state.coverageStatus).filter(status => 
        status === 'covered'
    ).length;
    const totalCount = Object.keys(state.coverageStatus).length;
    state.progress = Math.round((coveredCount / totalCount) * 100);
}

function updateUI() {
    updateSessionDisplay();
    updateCoverageDisplay();
    updateProgressDisplay();
    updateButtonStates();
}

function updateSessionDisplay() {
    if (!elements.sessionIdDisplay) return;
    if (state.currentSessionId) {
        elements.sessionIdDisplay.innerHTML = state.currentSessionId;
    } else {
        elements.sessionIdDisplay.innerHTML = '<span class="session-empty">暂无活动会话</span>';
    }
}

function updateCoverageDisplay() {
    if (!elements.coverageItems) return;

    const coverageElements = elements.coverageItems.querySelectorAll('.coverage-item');
    coverageElements.forEach((element, index) => {
        const keys = ['hba1c', 'hypoglycemia', 'convenience'];
        const key = keys[index];
        if (key) {
            const status = state.coverageStatus[key];
            element.className = `coverage-item ${status}`;
            const statusElement = element.querySelector('.item-status');
            if (statusElement) {
                statusElement.className = `item-status ${status}`;
            }
        }
    });
}

function updateProgressDisplay() {
    if (elements.progressFill) {
        elements.progressFill.style.width = `${state.progress}%`;
    }
    if (elements.progressText) {
        elements.progressText.textContent = `${state.progress}% 覆盖`;
    }
}

function updateButtonStates() {
    if (elements.btnSend) {
        elements.btnSend.disabled = !state.currentSessionId;
    }
    if (elements.btnDeleteSession) {
        elements.btnDeleteSession.disabled = !state.currentSessionId;
    }
}

function setLoading(isLoading) {
    state.isLoading = isLoading;
    if (elements.loadingOverlay) {
        elements.loadingOverlay.classList.toggle('show', isLoading);
    }
}

function showToast(message, type = 'info') {
    if (!elements.toast) return;
    
    elements.toast.textContent = message;
    elements.toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 3000);
}

async function createSession() {
    const response = await fetch(`${API_BASE_URL}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            customer_profile_id: 'default',
            product_id: 'default'
        })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
}

async function sendMessage(sessionId, message) {
    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
}

async function deleteSession(sessionId) {
    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
        method: 'DELETE'
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
}

async function fetchSessionStatus(sessionId) {
    try {
        const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/status`);
        if (!response.ok) return;
        const status = await response.json();
        if (status.progress !== undefined) {
            state.progress = status.progress;
        }
    } catch (error) {
        console.warn('获取会话状态失败:', error);
    }
}

// Initialize app
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}

// Add typing indicator function
function showTypingIndicator() {
    if (!elements.chatMessages) return;
    
    const typingHtml = `
        <div class="typing-indicator">
            <div class="message-avatar">AI</div>
            <div class="typing-bubble">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    elements.chatMessages.insertAdjacentHTML('beforeend', typingHtml);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const typingElement = elements.chatMessages.querySelector('.typing-indicator');
    if (typingElement) {
        typingElement.remove();
    }
}
