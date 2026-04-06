/**
 * AI销售训练Chatbot系统 - 前端应用逻辑
 * @description 处理与后端API的交互，包括会话管理、消息发送、评估获取等功能
 * @version 1.0.0
 */

const API_BASE_URL = 'http://localhost:8000/api/v1';

/**
 * 状态管理
 * @typedef {Object} AppState
 * @property {string|null} currentSessionId - 当前会话ID
 * @property {Array<Object>} messages - 消息列表
 * @property {boolean} isLoading - 加载状态
 * @property {number} progress - 进度百分比
 * @property {string|null} error - 错误信息
 */

/** @type {AppState} */
const state = {
    currentSessionId: null,
    messages: [],
    isLoading: false,
    progress: 0,
    error: null,
};

/**
 * DOM元素缓存
 * @typedef {Object} DOMElements
 */

/** @type {DOMElements} */
let elements = {};

/**
 * 初始化应用
 * @description 缓存DOM元素，绑定事件监听器
 */
function initApp() {
    cacheElements();
    bindEventListeners();
    updateUI();
}

/**
 * 缓存常用DOM元素
 */
function cacheElements() {
    elements = {
        messageList: document.getElementById('messageList'),
        messageInput: document.getElementById('messageInput'),
        sendButton: document.getElementById('sendButton'),
        newSessionButton: document.getElementById('newSessionButton'),
        deleteSessionButton: document.getElementById('deleteSessionButton'),
        progressBar: document.getElementById('progressBar'),
        progressText: document.getElementById('progressText'),
        loadingIndicator: document.getElementById('loadingIndicator'),
        errorDisplay: document.getElementById('errorDisplay'),
        sessionIdDisplay: document.getElementById('sessionIdDisplay'),
        healthStatus: document.getElementById('healthStatus'),
    };
}

/**
 * 绑定事件监听器
 */
function bindEventListeners() {
    if (elements.sendButton) {
        elements.sendButton.addEventListener('click', handleSendMessage);
    }
    if (elements.messageInput) {
        elements.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });
    }
    if (elements.newSessionButton) {
        elements.newSessionButton.addEventListener('click', handleCreateSession);
    }
    if (elements.deleteSessionButton) {
        elements.deleteSessionButton.addEventListener('click', handleDeleteSession);
    }
}

/**
 * 处理创建新会话
 * @returns {Promise<void>}
 */
async function handleCreateSession() {
    try {
        setLoading(true);
        clearError();
        const session = await createSession();
        state.currentSessionId = session.id;
        state.messages = [];
        await fetchSessionStatus(session.id);
        updateUI();
    } catch (error) {
        setError(`创建会话失败: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

/**
 * 处理发送消息
 * @returns {Promise<void>}
 */
async function handleSendMessage() {
    const message = elements.messageInput?.value.trim();
    if (!message) return;

    if (!state.currentSessionId) {
        setError('请先创建会话');
        return;
    }

    try {
        setLoading(true);
        clearError();

        addMessageToList({ role: 'user', content: message });
        elements.messageInput.value = '';

        const response = await sendMessage(state.currentSessionId, message);
        addMessageToList({ role: 'assistant', content: response.response });
        await fetchSessionStatus(state.currentSessionId);
        updateUI();
    } catch (error) {
        setError(`发送消息失败: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

/**
 * 处理删除会话
 * @returns {Promise<void>}
 */
async function handleDeleteSession() {
    if (!state.currentSessionId) {
        setError('没有活动的会话');
        return;
    }

    if (!confirm('确定要删除当前会话吗？')) {
        return;
    }

    try {
        setLoading(true);
        clearError();
        await deleteSession(state.currentSessionId);
        state.currentSessionId = null;
        state.messages = [];
        state.progress = 0;
        updateUI();
    } catch (error) {
        setError(`删除会话失败: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

/**
 * 添加消息到列表
 * @param {Object} message - 消息对象
 * @param {string} message.role - 角色 ('user' | 'assistant')
 * @param {string} message.content - 消息内容
 */
function addMessageToList(message) {
    state.messages.push(message);
    renderMessages();
}

/**
 * 渲染消息列表
 */
function renderMessages() {
    if (!elements.messageList) return;

    elements.messageList.innerHTML = state.messages
        .map((msg) => {
            const roleClass = msg.role === 'user' ? 'user-message' : 'assistant-message';
            const roleLabel = msg.role === 'user' ? '用户' : 'AI教练';
            return `
                <div class="message ${roleClass}">
                    <div class="message-role">${roleLabel}</div>
                    <div class="message-content">${escapeHtml(msg.content)}</div>
                </div>
            `;
        })
        .join('');

    elements.messageList.scrollTop = elements.messageList.scrollHeight;
}

/**
 * HTML转义防止XSS
 * @param {string} text - 原始文本
 * @returns {string} 转义后的文本
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * 更新UI状态
 */
function updateUI() {
    updateProgressBar();
    updateSessionDisplay();
    updateButtonStates();
    renderMessages();
}

/**
 * 更新进度条
 */
function updateProgressBar() {
    if (elements.progressBar) {
        elements.progressBar.style.width = `${state.progress}%`;
    }
    if (elements.progressText) {
        elements.progressText.textContent = `${state.progress}%`;
    }
}

/**
 * 更新会话显示
 */
function updateSessionDisplay() {
    if (elements.sessionIdDisplay) {
        elements.sessionIdDisplay.textContent = state.currentSessionId
            ? `会话ID: ${state.currentSessionId}`
            : '无活动会话';
    }
}

/**
 * 更新按钮状态
 */
function updateButtonStates() {
    if (elements.sendButton) {
        elements.sendButton.disabled = state.isLoading || !state.currentSessionId;
    }
    if (elements.deleteSessionButton) {
        elements.deleteSessionButton.disabled = !state.currentSessionId;
    }
}

/**
 * 设置加载状态
 * @param {boolean} isLoading - 是否加载中
 */
function setLoading(isLoading) {
    state.isLoading = isLoading;
    if (elements.loadingIndicator) {
        elements.loadingIndicator.style.display = isLoading ? 'block' : 'none';
    }
    updateButtonStates();
}

/**
 * 设置错误信息
 * @param {string} message - 错误信息
 */
function setError(message) {
    state.error = message;
    if (elements.errorDisplay) {
        elements.errorDisplay.textContent = message;
        elements.errorDisplay.style.display = 'block';
    }
}

/**
 * 清除错误信息
 */
function clearError() {
    state.error = null;
    if (elements.errorDisplay) {
        elements.errorDisplay.style.display = 'none';
    }
}

/**
 * 创建新会话
 * @returns {Promise<Object>} 会话对象
 * @throws {Error} API错误
 */
async function createSession() {
    const response = await fetch(`${API_BASE_URL}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
}

/**
 * 发送消息
 * @param {string} sessionId - 会话ID
 * @param {string} message - 消息内容
 * @returns {Promise<Object>} 响应对象
 * @throws {Error} API错误
 */
async function sendMessage(sessionId, message) {
    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
}

/**
 * 获取会话评估
 * @param {string} sessionId - 会话ID
 * @returns {Promise<Object>} 评估结果
 * @throws {Error} API错误
 */
async function getEvaluation(sessionId) {
    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/evaluation`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
}

/**
 * 删除会话
 * @param {string} sessionId - 会话ID
 * @returns {Promise<void>}
 * @throws {Error} API错误
 */
async function deleteSession(sessionId) {
    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
}

/**
 * 获取会话状态
 * @param {string} sessionId - 会话ID
 * @returns {Promise<Object>} 状态对象
 * @throws {Error} API错误
 */
async function fetchSessionStatus(sessionId) {
    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/status`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const status = await response.json();
    state.progress = status.progress || 0;
    return status;
}

/**
 * 健康检查
 * @returns {Promise<Object>} 健康状态
 * @throws {Error} API错误
 */
async function checkHealth() {
    const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
}

/**
 * 初始化健康检查状态显示
 * @returns {Promise<void>}
 */
async function initHealthCheck() {
    try {
        const health = await checkHealth();
        if (elements.healthStatus) {
            elements.healthStatus.textContent = '后端服务: 在线';
            elements.healthStatus.className = 'health-status online';
        }
    } catch (error) {
        if (elements.healthStatus) {
            elements.healthStatus.textContent = '后端服务: 离线';
            elements.healthStatus.className = 'health-status offline';
        }
    }
}

/**
 * 导出评估报告
 * @param {string} sessionId - 会话ID
 * @returns {Promise<void>}
 */
async function exportEvaluation(sessionId) {
    try {
        const evaluation = await getEvaluation(sessionId);
        const blob = new Blob([JSON.stringify(evaluation, null, 2)], {
            type: 'application/json',
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `evaluation_${sessionId}.json`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        setError(`导出评估失败: ${error.message}`);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}

export {
    createSession,
    sendMessage,
    getEvaluation,
    deleteSession,
    fetchSessionStatus,
    checkHealth,
    exportEvaluation,
};
