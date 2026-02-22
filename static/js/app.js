// LLM Unified MK12 - Frontend Application (WebSocket streaming + memory management)

class App {
    constructor() {
        this.user = null;
        this.currentConversationId = null;
        this.isStreaming = false;

        // WebSocket state
        this.ws = null;
        this._wsReconnectDelay = 1000;
        this._wsReconnecting = false;

        // State for the currently active streaming response
        this._streamingState = null;

        // Settings listeners flag
        this._settingsListenersAdded = false;

        this.init();
    }

    async init() {
        this.bindEvents();
        await this.checkAuth();
    }

    bindEvents() {
        // Auth tabs
        document.querySelectorAll('.auth-tabs .tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchAuthTab(tab.dataset.tab));
        });

        // Auth forms
        document.getElementById('login-form').addEventListener('submit', (e) => this.handleLogin(e));
        document.getElementById('register-form').addEventListener('submit', (e) => this.handleRegister(e));

        // Chat
        document.getElementById('chat-form').addEventListener('submit', (e) => this.handleSendMessage(e));
        document.getElementById('message-input').addEventListener('keydown', (e) => this.handleInputKeydown(e));
        document.getElementById('message-input').addEventListener('input', (e) => this.autoResizeInput(e.target));

        // Stop button
        document.getElementById('stop-btn').addEventListener('click', () => this.handleStopStreaming());

        // Scroll-to-bottom button
        document.getElementById('scroll-bottom-btn').addEventListener('click', () => {
            const msgs = document.getElementById('messages');
            msgs.scrollTop = msgs.scrollHeight;
        });
        document.getElementById('messages').addEventListener('scroll', () => {
            this._updateScrollBtn();
        });

        // MK13: tag button toggle (event delegation on messages container)
        document.getElementById('messages').addEventListener('click', (e) => {
            if (e.target.matches('.tag-btn:not(.submit-tags)')) {
                e.target.classList.toggle('selected');
            }
        });

        // Memory Map panel
        document.getElementById('memory-map-btn').addEventListener('click', () => {
            if (typeof window._openMemoryMap === 'function') window._openMemoryMap();
        });
        document.getElementById('close-memory-map-btn').addEventListener('click', () => {
            if (typeof window._closeMemoryMap === 'function') window._closeMemoryMap();
        });
        document.getElementById('memory-map-refresh-btn').addEventListener('click', () => {
            if (typeof window._refreshMemoryMap === 'function') window._refreshMemoryMap();
        });

        // Conversation search
        document.getElementById('conversation-search').addEventListener('input', (e) => {
            this.filterConversations(e.target.value);
        });

        // Sidebar
        document.getElementById('new-chat-btn').addEventListener('click', () => this.newChat());
        document.getElementById('logout-btn').addEventListener('click', () => this.logout());
        document.getElementById('toggle-sidebar').addEventListener('click', () => this.toggleSidebar());
        document.getElementById('sidebar-close-btn').addEventListener('click', () => this.closeSidebar());

        // Settings
        document.getElementById('settings-btn').addEventListener('click', () => this.openSettings());
        document.querySelectorAll('.close-modal').forEach(btn => {
            btn.addEventListener('click', () => this.closeModals());
        });

        // Close modal on backdrop click
        document.getElementById('settings-modal').addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) this.closeModals();
        });

        // Custom Tools Panel
        this._bindToolsPanel();

        // Research Panel
        this._bindResearchPanel();
    }

    // --- WebSocket ---

    connectWebSocket() {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${location.host}/ws/chat`);

        this.ws.addEventListener('open', () => {
            this._wsReconnectDelay = 1000;
            this._wsReconnecting = false;
        });

        this.ws.addEventListener('message', (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWsMessage(data);
            } catch (e) {
                // Ignore parse errors
            }
        });

        this.ws.addEventListener('close', (event) => {
            if (event.code === 4001) return; // Unauthorized — don't reconnect

            // If we were streaming, mark it as failed
            if (this.isStreaming) {
                const state = this._streamingState;
                if (state) {
                    state.contentDiv.innerHTML = state.toolOutputsHtml +
                        this.formatMessage(state.fullResponse) +
                        '<p class="cancelled-msg">— connection lost —</p>';
                }
                this.finishStreaming();
            }

            // Reconnect with exponential backoff
            this._wsReconnecting = true;
            setTimeout(() => {
                if (this.user) this.connectWebSocket();
            }, this._wsReconnectDelay);
            this._wsReconnectDelay = Math.min(this._wsReconnectDelay * 2, 30000);
        });

        this.ws.addEventListener('error', () => {
            // onclose fires after onerror — reconnect handled there
        });
    }

    handleWsMessage(data) {
        const state = this._streamingState;

        switch (data.type) {
            case 'conversation_id':
                this.currentConversationId = data.id;
                break;

            case 'status':
                if (window.neuralBg && data.status) window.neuralBg.setStatus(data.status);
                break;

            case 'token':
                if (!state) break;
                if (!state.firstToken) state.firstToken = true;
                if (window.neuralBg && window.neuralBg.status === 'memory') {
                    window.neuralBg.setStatus('thinking');
                }
                state.fullResponse += data.content;
                state.contentDiv.innerHTML = state.toolOutputsHtml +
                    this.formatMessage(state.fullResponse) +
                    '<span class="streaming-cursor"></span>';
                this._scrollIfAtBottom(state.messagesContainer);
                break;

            case 'command_response':
                if (!state) break;
                if (data.content === '__DOWNLOAD_FINDINGS__') {
                    this.downloadFindings();
                    state.fullResponse = 'Downloading research findings...';
                } else {
                    state.fullResponse = data.content;
                }
                state.contentDiv.innerHTML = state.toolOutputsHtml + this.formatMessage(state.fullResponse);
                // Commands (?help, ?suggestions, etc.) don't get feedback buttons
                if (state.messageEl) {
                    const fb = state.messageEl.querySelector('.message-feedback');
                    if (fb) fb.remove();
                }
                this.loadConversations();
                this.finishStreaming();
                break;

            case 'tool_call':
                if (!state) break;
                if (window.neuralBg) window.neuralBg.setStatus('tool');
                state.contentDiv.innerHTML = state.toolOutputsHtml +
                    `<p class="tool-call">🔧 Calling: ${data.name}</p><span class="streaming-cursor"></span>`;
                state.messagesContainer.scrollTop = state.messagesContainer.scrollHeight;
                break;

            case 'tool_output': {
                if (!state) break;
                const outputPreview = data.output.length > 100
                    ? data.output.substring(0, 100) + '...'
                    : data.output;
                const toolId = `tool-${Date.now()}`;
                state.toolOutputsHtml += `
                    <div class="tool-section" id="${toolId}">
                        <div class="tool-header" onclick="app.toggleToolOutput('${toolId}')">
                            <span class="tool-toggle">▶</span>
                            <span class="tool-name">🔧 ${data.name}</span>
                            <span class="tool-preview">${this.escapeHtml(outputPreview)}</span>
                        </div>
                        <pre class="tool-output-full hidden">${this.escapeHtml(data.output)}</pre>
                    </div>`;
                state.contentDiv.innerHTML = state.toolOutputsHtml + '<span class="streaming-cursor"></span>';
                this._scrollIfAtBottom(state.messagesContainer);
                break;
            }

            case 'approval_required':
                if (!state) break;
                state.contentDiv.innerHTML = state.toolOutputsHtml + this.renderApprovalDialog(data.approval);
                state.messagesContainer.scrollTop = state.messagesContainer.scrollHeight;
                this.finishStreaming();
                if (window.neuralBg) window.neuralBg.setStatus('approval');
                break;

            case 'done': {
                if (!state) break;
                let statsHtml = '';
                if (data.stats && data.stats.tokens_per_sec) {
                    statsHtml = `<div class="generation-stats">${data.stats.tokens} tokens in ${data.stats.elapsed}s (${data.stats.tokens_per_sec} tok/s)</div>`;
                }
                state.contentDiv.innerHTML = state.toolOutputsHtml +
                    this.formatMessage(state.fullResponse) + statsHtml;
                // MK13: attach message_id to the message div for feedback routing
                if (data.message_id && state.messageEl) {
                    state.messageEl.dataset.messageId = data.message_id;
                    const fb = state.messageEl.querySelector('.message-feedback');
                    if (fb) fb.dataset.messageId = data.message_id;
                }
                this.loadConversations();
                this.finishStreaming();
                break;
            }

            // MK13: confidence badge
            case 'confidence': {
                if (!this._settings || !this._settings.show_confidence) break;
                // Find message div by message_id (set on 'done') — may arrive before or after done
                this._pendingConfidence = data;
                this._applyPendingConfidence();
                break;
            }

            // MK13: gate decision (optional transparency display — no-op for now)
            case 'gate_decision':
                break;

            case 'error':
                if (state) {
                    state.contentDiv.innerHTML = state.toolOutputsHtml +
                        this.formatMessage(state.fullResponse) +
                        `<p style="color: var(--error);">Error: ${this.escapeHtml(data.message)}</p>`;
                }
                this.finishStreaming();
                break;

            case 'cancelled':
                if (state) {
                    state.contentDiv.innerHTML = state.toolOutputsHtml +
                        this.formatMessage(state.fullResponse) +
                        '<p class="cancelled-msg">— cancelled —</p>';
                }
                this.finishStreaming();
                break;

            case 'server_push':
                this.showServerPush(data.content);
                break;
        }
    }

    _scrollIfAtBottom(container) {
        // Only auto-scroll if the user is within 120px of the bottom
        const distFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
        if (distFromBottom < 120) {
            container.scrollTop = container.scrollHeight;
        }
        this._updateScrollBtn();
    }

    _updateScrollBtn() {
        const msgs = document.getElementById('messages');
        const btn = document.getElementById('scroll-bottom-btn');
        if (!msgs || !btn) return;
        const distFromBottom = msgs.scrollHeight - msgs.scrollTop - msgs.clientHeight;
        btn.classList.toggle('hidden', distFromBottom < 120);
    }

    finishStreaming() {
        this.isStreaming = false;
        this._streamingState = null;
        this.updateInputButtons();
        if (window.neuralBg) window.neuralBg.deactivate();
    }

    updateInputButtons() {
        const sendBtn = document.getElementById('send-btn');
        const stopBtn = document.getElementById('stop-btn');
        sendBtn.classList.toggle('hidden', this.isStreaming);
        stopBtn.classList.toggle('hidden', !this.isStreaming);
    }

    handleStopStreaming() {
        if (!this.isStreaming || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        this.ws.send(JSON.stringify({ type: 'cancel' }));
    }

    showServerPush(content) {
        const el = document.createElement('div');
        el.className = 'server-push-notification';
        el.textContent = content;
        document.body.appendChild(el);
        setTimeout(() => el.remove(), 5000);
    }

    // --- Auth ---

    async checkAuth() {
        try {
            const response = await fetch('/api/auth/me');
            if (response.ok) {
                const data = await response.json();
                this.user = data.user;
                this.showChatScreen();
                this.connectWebSocket();
                await this.loadConversations();
            } else {
                this.showAuthScreen();
            }
        } catch (error) {
            this.showAuthScreen();
        }
    }

    switchAuthTab(tab) {
        document.querySelectorAll('.auth-tabs .tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tab);
        });
        document.getElementById('login-form').classList.toggle('hidden', tab !== 'login');
        document.getElementById('register-form').classList.toggle('hidden', tab !== 'register');
    }

    async handleLogin(e) {
        e.preventDefault();
        const form = e.target;
        const username = form.username.value;
        const password = form.password.value;

        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            const data = await response.json();

            if (response.ok) {
                this.user = data.user;
                this.showChatScreen();
                this.connectWebSocket();
                await this.loadConversations();
            } else {
                document.getElementById('login-error').textContent = data.detail || 'Login failed';
            }
        } catch (error) {
            document.getElementById('login-error').textContent = 'Connection error';
        }
    }

    async handleRegister(e) {
        e.preventDefault();
        const form = e.target;
        const username = form.username.value;
        const password = form.password.value;
        const confirm = form.confirm.value;

        if (password !== confirm) {
            document.getElementById('register-error').textContent = 'Passwords do not match';
            return;
        }

        try {
            const response = await fetch('/api/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            const data = await response.json();

            if (response.ok) {
                this.user = data.user;
                this.showChatScreen();
                this.connectWebSocket();
                await this.loadConversations();
            } else {
                document.getElementById('register-error').textContent = data.detail || 'Registration failed';
            }
        } catch (error) {
            document.getElementById('register-error').textContent = 'Connection error';
        }
    }

    async logout() {
        if (this.ws) {
            this.ws.close(1000, 'Logout');
            this.ws = null;
        }
        await fetch('/api/auth/logout', { method: 'POST' });
        this.user = null;
        this.currentConversationId = null;
        this.showAuthScreen();
    }

    showAuthScreen() {
        document.getElementById('auth-screen').classList.remove('hidden');
        document.getElementById('chat-screen').classList.add('hidden');
    }

    showChatScreen() {
        document.getElementById('auth-screen').classList.add('hidden');
        document.getElementById('chat-screen').classList.remove('hidden');
        document.getElementById('username-display').textContent = this.user.username;
    }

    // --- Conversations ---

    async loadConversations() {
        try {
            const response = await fetch('/api/conversations');
            const data = await response.json();
            this.renderConversationList(data.conversations);
        } catch (error) {
            console.error('Failed to load conversations:', error);
        }
    }

    renderConversationList(conversations) {
        const container = document.getElementById('conversation-list');
        container.innerHTML = '';

        conversations.forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            item.dataset.id = conv.id;
            if (conv.id === this.currentConversationId) {
                item.classList.add('active');
            }

            item.innerHTML = `
                <span class="title">${this.escapeHtml(conv.title)}</span>
                <div class="actions">
                    <button class="rename-btn" title="Rename">✏</button>
                    <button class="delete-btn" title="Delete">🗑</button>
                </div>
            `;

            item.querySelector('.title').addEventListener('click', () => this.loadConversation(conv.id));
            item.querySelector('.rename-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                this.renameConversation(conv.id, conv.title);
            });
            item.querySelector('.delete-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteConversation(conv.id);
            });

            container.appendChild(item);
        });
    }

    async loadConversation(id) {
        try {
            const response = await fetch(`/api/conversations/${id}`);
            if (!response.ok) return;

            const data = await response.json();
            this.currentConversationId = id;

            document.getElementById('chat-title').textContent = data.title || 'Chat';

            document.querySelectorAll('.conversation-item').forEach(item => {
                item.classList.toggle('active', item.dataset.id === id);
            });

            const messagesContainer = document.getElementById('messages');
            messagesContainer.innerHTML = '';

            // Use the conversation's updated_at as the timestamp for all loaded messages
            const convTs = (data.updated_at || Date.now() / 1000) * 1000;
            data.messages.forEach(msg => {
                this.appendMessage(msg.role, msg.content, convTs);
            });

            messagesContainer.scrollTop = messagesContainer.scrollHeight;

            document.querySelector('.sidebar').classList.remove('open');
        } catch (error) {
            console.error('Failed to load conversation:', error);
        }
    }

    async deleteConversation(id) {
        if (!confirm('Delete this conversation?')) return;

        try {
            await fetch(`/api/conversations/${id}`, { method: 'DELETE' });

            if (this.currentConversationId === id) {
                this.newChat();
            }

            await this.loadConversations();
        } catch (error) {
            console.error('Failed to delete conversation:', error);
        }
    }

    async renameConversation(id, currentTitle) {
        const newTitle = prompt('Rename conversation:', currentTitle);
        if (!newTitle || newTitle === currentTitle) return;

        try {
            await fetch(`/api/conversations/${id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: newTitle })
            });

            await this.loadConversations();

            if (this.currentConversationId === id) {
                document.getElementById('chat-title').textContent = newTitle;
            }
        } catch (error) {
            console.error('Failed to rename conversation:', error);
        }
    }

    filterConversations(query) {
        const q = query.toLowerCase().trim();
        document.querySelectorAll('#conversation-list .conversation-item').forEach(item => {
            const title = item.querySelector('.title').textContent.toLowerCase();
            item.style.display = (!q || title.includes(q)) ? '' : 'none';
        });
    }

    newChat() {
        this.currentConversationId = null;
        document.getElementById('chat-title').textContent = 'New Chat';
        document.getElementById('messages').innerHTML = `
            <div class="welcome-message">
                <h2>How can I help you today?</h2>
                <p>Ask me anything. I can search the web, run commands, and remember our conversations.</p>
            </div>
        `;

        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.remove('active');
        });

        document.querySelector('.sidebar').classList.remove('open');
    }

    // --- Chat ---

    handleInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.getElementById('chat-form').requestSubmit();
        }
    }

    autoResizeInput(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }

    handleSendMessage(e) {
        e.preventDefault();

        if (this.isStreaming) return;

        const input = document.getElementById('message-input');
        const message = input.value.trim();
        if (!message) return;

        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.showServerPush('Not connected — reconnecting...');
            this.connectWebSocket();
            return;
        }

        input.value = '';
        input.style.height = 'auto';

        const welcome = document.querySelector('.welcome-message');
        if (welcome) welcome.remove();

        this.appendMessage('user', message);

        const assistantMessage = this.appendMessage('assistant', '');
        const contentDiv = assistantMessage.querySelector('.content');
        contentDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

        const messagesContainer = document.getElementById('messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        this.isStreaming = true;
        this.updateInputButtons();

        if (window.neuralBg) window.neuralBg.setStatus('memory');

        this._streamingState = {
            contentDiv,
            messagesContainer,
            messageEl: assistantMessage,  // MK13: track message div for message_id
            fullResponse: '',
            toolOutputsHtml: '',
            firstToken: false,
        };
        this._pendingConfidence = null;  // MK13: hold confidence until message_id arrives

        this.ws.send(JSON.stringify({
            type: 'message',
            content: message,
            conversation_id: this.currentConversationId,
        }));
    }

    appendMessage(role, content, timestamp = Date.now()) {
        const messagesContainer = document.getElementById('messages');

        const message = document.createElement('div');
        message.className = `message ${role}`;
        message.dataset.timestamp = timestamp;

        const avatar = role === 'user' ? '👤' : '🤖';
        const timeStr = this.formatTimestamp(timestamp);
        // Only show copy button and feedback on assistant messages
        const copyBtn = role === 'assistant'
            ? `<button class="copy-msg-btn" onclick="app.copyMessage(this)" title="Copy message">Copy</button>`
            : '';
        const feedbackBtns = role === 'assistant'
            ? `<div class="message-feedback">
                <button class="feedback-btn upvote" onclick="app.sendFeedback(this, 'positive')" title="Good response">👍</button>
                <button class="feedback-btn downvote" onclick="app.showFeedbackTags(this)" title="Bad response">👎</button>
                <div class="feedback-tags hidden">
                    <button class="tag-btn" data-tag="wrong">Wrong</button>
                    <button class="tag-btn" data-tag="irrelevant">Irrelevant</button>
                    <button class="tag-btn" data-tag="hallucinated">Hallucinated</button>
                    <button class="tag-btn" data-tag="too_verbose">Too verbose</button>
                    <button class="tag-btn submit-tags" onclick="app.submitFeedbackTags(this)">Submit</button>
                </div>
               </div>`
            : '';

        message.innerHTML = `
            ${copyBtn}
            <div class="avatar">${avatar}</div>
            <div class="content">${this.formatMessage(content)}</div>
            <div class="message-meta">
                <span class="message-time">${timeStr}</span>
                ${feedbackBtns}
            </div>
        `;

        messagesContainer.appendChild(message);
        return message;
    }

    formatMessage(content) {
        if (!content) return '';

        // Protect code blocks before any other processing, wrapping each in a
        // copy-button container so it doesn't interfere with line parsing below.
        const codeBlocks = [];
        let processed = content.replace(/```(\w*)\n?([\s\S]*?)```/g, (_match, lang, code) => {
            const idx = codeBlocks.length;
            const escaped = this.escapeHtml(code.trim());
            const langLabel = lang ? lang : 'code';
            codeBlocks.push(
                `<div class="code-block-wrapper">` +
                `<div class="code-block-header">` +
                `<span class="code-lang">${this.escapeHtml(langLabel)}</span>` +
                `<button class="copy-code-btn" onclick="app.copyCode(this)" title="Copy code">Copy</button>` +
                `</div>` +
                `<pre><code class="language-${lang}">${escaped}</code></pre>` +
                `</div>`
            );
            return `\x00CB${idx}\x00`;
        });

        const lines = processed.split('\n');
        const result = [];
        let inList = null;
        let inTable = false;
        let tableRows = [];       // array of string[]
        let tableHasHeader = false;

        const closeList = () => {
            if (inList) { result.push(`</${inList}>`); inList = null; }
        };

        const closeTable = () => {
            if (!inTable) return;
            inTable = false;
            if (tableRows.length === 0) return;
            let html = '<div class="table-wrapper"><table>';
            const bodyStart = tableHasHeader ? 1 : 0;
            if (tableHasHeader) {
                html += '<thead><tr>';
                tableRows[0].forEach(cell => html += `<th>${this.inlineFormat(cell)}</th>`);
                html += '</tr></thead>';
            }
            html += '<tbody>';
            tableRows.slice(bodyStart).forEach(row => {
                html += '<tr>';
                row.forEach(cell => html += `<td>${this.inlineFormat(cell)}</td>`);
                html += '</tr>';
            });
            html += '</tbody></table></div>';
            result.push(html);
            tableRows = [];
            tableHasHeader = false;
        };

        const closeAll = () => { closeList(); closeTable(); };

        for (const line of lines) {
            // Restore code blocks
            if (/\x00CB\d+\x00/.test(line)) {
                closeAll();
                result.push(line.replace(/\x00CB(\d+)\x00/g, (_, i) => codeBlocks[+i]));
                continue;
            }

            const trimmed = line.trim();

            // Table separator row  (| --- | --- |)
            if (/^\|[\s\-:|]+\|$/.test(trimmed)) {
                if (inTable && tableRows.length > 0) tableHasHeader = true;
                continue;
            }

            // Table data row  (| cell | cell |)
            if (/^\|.+\|$/.test(trimmed)) {
                closeList();
                if (!inTable) { inTable = true; tableRows = []; tableHasHeader = false; }
                const cells = trimmed.slice(1, -1).split('|').map(c => c.trim());
                tableRows.push(cells);
                continue;
            }

            // Any non-table line ends the table
            closeTable();

            // Headings
            const hMatch = line.match(/^(#{1,6})\s+(.+)/);
            if (hMatch) {
                closeList();
                const level = hMatch[1].length;
                result.push(`<h${level}>${this.inlineFormat(hMatch[2])}</h${level}>`);
                continue;
            }

            // Horizontal rule
            if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
                closeList();
                result.push('<hr>');
                continue;
            }

            // Blockquote
            const bqMatch = line.match(/^>\s?(.*)/);
            if (bqMatch) {
                closeList();
                result.push(`<blockquote>${this.inlineFormat(bqMatch[1])}</blockquote>`);
                continue;
            }

            // Unordered list
            const ulMatch = line.match(/^[-*+]\s+(.*)/);
            if (ulMatch) {
                if (inList !== 'ul') { closeList(); result.push('<ul>'); inList = 'ul'; }
                result.push(`<li>${this.inlineFormat(ulMatch[1])}</li>`);
                continue;
            }

            // Ordered list
            const olMatch = line.match(/^\d+[.)]\s+(.*)/);
            if (olMatch) {
                if (inList !== 'ol') { closeList(); result.push('<ol>'); inList = 'ol'; }
                result.push(`<li>${this.inlineFormat(olMatch[1])}</li>`);
                continue;
            }

            // Regular text
            closeList();
            if (trimmed === '') {
                result.push('<br>');
            } else {
                result.push(`<p>${this.inlineFormat(line)}</p>`);
            }
        }

        closeAll();
        return result.join('');
    }

    inlineFormat(text) {
        let s = this.escapeHtml(text);
        // Inline code (protect first so other patterns don't touch it)
        const codes = [];
        s = s.replace(/`([^`]+)`/g, (_, c) => {
            codes.push(`<code>${c}</code>`);
            return `\x00IC${codes.length - 1}\x00`;
        });
        // Links: [text](url) — only http/https to prevent javascript: injection
        s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
        // Bold
        s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        // Italic
        s = s.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        // Strikethrough
        s = s.replace(/~~([^~]+)~~/g, '<del>$1</del>');
        // Restore inline code
        s = s.replace(/\x00IC(\d+)\x00/g, (_, i) => codes[+i]);
        return s;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    copyCode(btn) {
        const code = btn.closest('.code-block-wrapper').querySelector('pre code');
        navigator.clipboard.writeText(code.textContent).then(() => {
            btn.textContent = 'Copied!';
            btn.classList.add('copied');
            setTimeout(() => {
                btn.textContent = 'Copy';
                btn.classList.remove('copied');
            }, 2000);
        }).catch(() => {});
    }

    copyMessage(btn) {
        // Walk up to the message div and grab the raw text of .content
        const msgDiv = btn.closest('.message');
        const content = msgDiv.querySelector('.content').innerText;
        navigator.clipboard.writeText(content).then(() => {
            btn.textContent = 'Copied!';
            btn.classList.add('copied');
            setTimeout(() => {
                btn.textContent = 'Copy';
                btn.classList.remove('copied');
            }, 2000);
        }).catch(() => {});
    }

    // --- MK13: Feedback ---

    sendFeedback(btn, value) {
        const msgDiv = btn.closest('.message');
        const messageId = msgDiv && msgDiv.dataset.messageId;
        if (!messageId) return;
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'feedback', message_id: messageId, value, tags: [] }));
        }
        // Mark button active, dim the other
        const feedback = btn.closest('.message-feedback');
        feedback.querySelectorAll('.feedback-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }

    showFeedbackTags(btn) {
        const msgDiv = btn.closest('.message');
        const messageId = msgDiv && msgDiv.dataset.messageId;
        if (!messageId) return;
        const feedback = btn.closest('.message-feedback');
        const tags = feedback.querySelector('.feedback-tags');
        if (tags) tags.classList.toggle('hidden');
        btn.classList.add('active');
    }

    submitFeedbackTags(btn) {
        const feedback = btn.closest('.message-feedback');
        const msgDiv = btn.closest('.message');
        const messageId = msgDiv && msgDiv.dataset.messageId;
        if (!messageId) return;
        const selected = [];
        feedback.querySelectorAll('.tag-btn:not(.submit-tags)').forEach(tb => {
            if (tb.classList.contains('selected')) selected.push(tb.dataset.tag);
        });
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'feedback', message_id: messageId, value: 'negative', tags: selected }));
        }
        feedback.querySelector('.feedback-tags').classList.add('hidden');
    }

    // --- MK13: Confidence Badge ---

    _applyPendingConfidence() {
        const pending = this._pendingConfidence;
        if (!pending || !pending.message_id) return;
        const msgDiv = document.querySelector(`.message[data-message-id="${pending.message_id}"]`);
        if (!msgDiv) return;  // message_id not set yet — 'done' will trigger another call
        this._pendingConfidence = null;
        this.showConfidenceBadge(msgDiv, pending.value);
    }

    showConfidenceBadge(msgDiv, value) {
        const meta = msgDiv.querySelector('.message-meta');
        if (!meta || meta.querySelector('.confidence-badge')) return;
        const cls = value >= 0.65 ? 'confidence-high' : value >= 0.4 ? 'confidence-medium' : 'confidence-low';
        const pct = Math.round(value * 100);
        const badge = document.createElement('span');
        badge.className = `confidence-badge ${cls}`;
        badge.title = `Confidence: ${pct}%`;
        badge.textContent = '●';
        meta.insertBefore(badge, meta.firstChild);
    }

    async loadFeedbackStats() {
        try {
            const r = await fetch('/api/feedback/stats');
            if (!r.ok) return;
            const s = await r.json();
            const el = document.getElementById('feedback-stats');
            if (!el) return;
            let text = `${s.total} interactions`;
            if (s.rated > 0) {
                text += ` · ${s.positive} 👍 ${s.negative} 👎`;
            }
            const topTags = Object.entries(s.by_tag || {}).sort((a, b) => b[1] - a[1]).slice(0, 3);
            if (topTags.length) {
                text += ' · Flagged: ' + topTags.map(([tag, n]) => `${tag} (${n})`).join(', ');
            }
            el.textContent = text;
        } catch (_) {}
    }

    formatTimestamp(ts) {
        const d   = new Date(ts);
        const now = new Date();
        // Explicitly resolve the browser's local timezone so Linux systems that
        // launch the browser without TZ set don't fall back to UTC.
        const tz  = Intl.DateTimeFormat().resolvedOptions().timeZone;
        const fmt  = (date, opts) => date.toLocaleDateString(undefined, { timeZone: tz, ...opts });
        const isToday     = fmt(d, {}) === fmt(now, {});
        const isYesterday = fmt(d, {}) === fmt(new Date(now - 86400000), {});
        const timeStr = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', timeZone: tz });
        if (isToday)     return `Today at ${timeStr}`;
        if (isYesterday) return `Yesterday at ${timeStr}`;
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: tz }) + ` at ${timeStr}`;
    }

    // --- UI ---

    toggleSidebar() {
        document.querySelector('.sidebar').classList.toggle('open');
    }

    closeSidebar() {
        document.querySelector('.sidebar').classList.remove('open');
    }

    _closeAllPanels() {
        document.getElementById('tools-panel').classList.remove('open');
        document.getElementById('research-panel').classList.remove('open');
        document.getElementById('memory-map-panel')?.classList.remove('open');
    }

    async openSettings() {
        document.getElementById('settings-modal').classList.remove('hidden');
        await this.loadSettings();
    }

    async loadSettings() {
        try {
            const response = await fetch('/api/settings');
            if (!response.ok) return;

            const data = await response.json();
            this._settings = data;  // MK13: cache for show_confidence check

            document.getElementById('research-toggle').checked = data.research_enabled;
            document.getElementById('self-improve-toggle').checked = data.self_improve_enabled;
            document.getElementById('search-provider-select').value = data.search_provider || 'google';

            // MK13: new settings
            const gateToggle = document.getElementById('decision-gate-toggle');
            const gateSens = document.getElementById('gate-sensitivity');
            const gateSensVal = document.getElementById('gate-sensitivity-value');
            const feedbackToggle = document.getElementById('feedback-toggle');
            const confidenceToggle = document.getElementById('confidence-toggle');

            if (gateToggle) gateToggle.checked = data.decision_gate_enabled !== false;
            if (gateSens) {
                gateSens.value = data.decision_gate_sensitivity ?? 0.5;
                if (gateSensVal) gateSensVal.textContent = parseFloat(gateSens.value).toFixed(2);
            }
            if (feedbackToggle) feedbackToggle.checked = data.show_feedback_buttons !== false;
            if (confidenceToggle) confidenceToggle.checked = data.show_confidence !== false;

            if (!this._settingsListenersAdded) {
                document.getElementById('research-toggle').addEventListener('change', (e) => {
                    this.updateSetting('research_enabled', e.target.checked);
                });
                document.getElementById('self-improve-toggle').addEventListener('change', (e) => {
                    this.updateSetting('self_improve_enabled', e.target.checked);
                });
                document.getElementById('search-provider-select').addEventListener('change', (e) => {
                    this.updateSetting('search_provider', e.target.value);
                });
                document.getElementById('refresh-analytics-btn').addEventListener('click', () => {
                    this.loadAnalytics();
                });

                // Memory management buttons
                document.getElementById('export-memory-btn').addEventListener('click', () => {
                    this.exportMemory();
                });
                document.getElementById('import-memory-btn').addEventListener('click', () => {
                    document.getElementById('import-memory-file').click();
                });
                document.getElementById('import-memory-file').addEventListener('change', (e) => {
                    this.handleMemoryImportFile(e);
                });
                document.getElementById('delete-memories-btn').addEventListener('click', () => {
                    this.deleteAllMemories();
                });

                // MK13: new setting listeners
                if (gateToggle) gateToggle.addEventListener('change', (e) => {
                    this.updateSetting('decision_gate_enabled', e.target.checked);
                    if (this._settings) this._settings.decision_gate_enabled = e.target.checked;
                });
                if (gateSens) gateSens.addEventListener('input', (e) => {
                    const v = parseFloat(e.target.value);
                    if (gateSensVal) gateSensVal.textContent = v.toFixed(2);
                    this.updateSetting('decision_gate_sensitivity', v);
                });
                if (feedbackToggle) feedbackToggle.addEventListener('change', (e) => {
                    this.updateSetting('show_feedback_buttons', e.target.checked);
                    if (this._settings) this._settings.show_feedback_buttons = e.target.checked;
                    // Toggle visibility of existing feedback buttons
                    document.querySelectorAll('.message-feedback').forEach(el => {
                        el.style.display = e.target.checked ? '' : 'none';
                    });
                });
                if (confidenceToggle) confidenceToggle.addEventListener('change', (e) => {
                    this.updateSetting('show_confidence', e.target.checked);
                    if (this._settings) this._settings.show_confidence = e.target.checked;
                });

                this._settingsListenersAdded = true;
            }

            // MK13: load feedback stats + analytics into settings panel
            this.loadFeedbackStats();
            this.loadAnalytics();
        } catch (error) {
            console.error('Failed to load settings:', error);
        }
    }

    async loadAnalytics() {
        const content = document.getElementById('analytics-content');
        if (!content) return;
        try {
            const ar = await fetch('/api/analytics').then(r => r.json());
            const { interactions: int, confidence: conf,
                    gate_decisions: gate, feedback_patterns: fp,
                    behavior_profile: bp } = ar;

            const total = int.total || 0;
            const rated = int.rated || 0;
            const rateStr = total > 0 ? `${Math.round(rated / total * 100)}%` : '—';

            // Gate distribution bar
            const gTotal = (gate.answer || 0) + (gate.search || 0) + (gate.ask || 0) || 1;
            const aPct = Math.round((gate.answer || 0) / gTotal * 100);
            const sPct = Math.round((gate.search || 0) / gTotal * 100);
            const qPct = 100 - aPct - sPct;

            const confDot = v => v == null ? '—'
                : `<span style="color:${v >= 0.65 ? '#3fb950' : v >= 0.4 ? '#e6b428' : '#f05046'}">${Math.round(v * 100)}%</span>`;

            const row = (label, val) =>
                `<div class="analytics-row"><span class="analytics-label">${label}</span><span class="analytics-value">${val}</span></div>`;

            const tagStr = (fp.top_tags || []).map(([t, n]) => `${t} (${n})`).join(', ') || '—';

            content.innerHTML = `
                ${row('Total interactions', total)}
                ${row('Rated', `${rated} &middot; ${rateStr} rated`)}
                ${row('Feedback', `${int.positive} 👍&nbsp; ${int.negative} 👎`)}
                <hr class="analytics-section-sep">
                <div class="analytics-row">
                    <span class="analytics-label">Gate decisions</span>
                    <span class="analytics-value">${gate.answer || 0} answer &middot; ${gate.search || 0} search &middot; ${gate.ask || 0} ask</span>
                </div>
                <div class="analytics-bar-row">
                    <div class="analytics-bar-answer" style="flex:${aPct}"></div>
                    <div class="analytics-bar-search"  style="flex:${sPct}"></div>
                    <div class="analytics-bar-ask"     style="flex:${qPct}"></div>
                </div>
                ${row('Avg response confidence', confDot(conf.avg_response))}
                ${row('Confidence (hi / mid / lo)', `${conf.high} / ${conf.medium} / ${conf.low}`)}
                <hr class="analytics-section-sep">
                ${row('Low-conf negative rate', `${Math.round((fp.low_conf_negative_rate || 0) * 100)}%`)}
                ${row('Search helped rate', `${Math.round((fp.search_helped_rate || 0) * 100)}%`)}
                ${row('Hallucination rate', `${Math.round((fp.hallucination_rate || 0) * 100)}%`)}
                ${row('Top flags', tagStr)}
                <hr class="analytics-section-sep">
                ${row('Search threshold', bp.search_threshold)}
                ${row('Uncertainty behavior', bp.uncertainty_behavior)}
            `;
        } catch (e) {
            if (content) content.innerHTML =
                `<span style="color:var(--text-secondary);font-size:0.82rem">No data yet — start chatting and rating responses.</span>`;
        }
    }

    formatTimeAgo(date) {
        const seconds = Math.floor((new Date() - date) / 1000);
        if (seconds < 60) return 'just now';
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes}m ago`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ago`;
        const days = Math.floor(hours / 24);
        return `${days}d ago`;
    }

    async updateSetting(key, value) {
        try {
            await fetch('/api/settings', {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ [key]: value })
            });
        } catch (error) {
            console.error('Failed to update setting:', error);
        }
    }

    closeModals() {
        document.querySelectorAll('.modal').forEach(modal => {
            modal.classList.add('hidden');
        });
    }

    toggleToolOutput(toolId) {
        const section = document.getElementById(toolId);
        if (!section) return;

        const toggle = section.querySelector('.tool-toggle');
        const preview = section.querySelector('.tool-preview');
        const fullOutput = section.querySelector('.tool-output-full');

        if (fullOutput.classList.contains('hidden')) {
            fullOutput.classList.remove('hidden');
            preview.classList.add('hidden');
            toggle.textContent = '▼';
        } else {
            fullOutput.classList.add('hidden');
            preview.classList.remove('hidden');
            toggle.textContent = '▶';
        }
    }

    async downloadFindings() {
        try {
            const response = await fetch('/api/research/findings/download', {
                credentials: 'include'
            });

            if (!response.ok) throw new Error('Failed to download findings');

            const blob = await response.blob();
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = 'research_findings.txt';

            if (contentDisposition) {
                const match = contentDisposition.match(/filename="(.+)"/);
                if (match) filename = match[1];
            }

            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Download failed:', error);
        }
    }

    // --- Memory Management ---

    async exportMemory() {
        const btn = document.getElementById('export-memory-btn');
        btn.textContent = 'Exporting...';
        btn.disabled = true;

        try {
            const response = await fetch('/api/memory/export');
            if (!response.ok) throw new Error('Export failed');

            const data = await response.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
            const a = document.createElement('a');
            a.href = url;
            a.download = `memory_export_${timestamp}.json`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            alert(`Export error: ${err.message}`);
        } finally {
            btn.textContent = 'Export Memory';
            btn.disabled = false;
        }
    }

    async handleMemoryImportFile(e) {
        const file = e.target.files[0];
        if (!file) return;

        // Reset the input so the same file can be picked again later
        e.target.value = '';

        let data;
        try {
            data = JSON.parse(await file.text());
        } catch {
            alert('Invalid JSON file');
            return;
        }

        if (!data.memories) {
            alert('Invalid export format — missing "memories" field');
            return;
        }

        if (!confirm(`Import ${data.memories.length} memories? This will replace all current memories.`)) return;

        const btn = document.getElementById('import-memory-btn');
        btn.textContent = 'Importing...';
        btn.disabled = true;

        try {
            const response = await fetch('/api/memory/import', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            const result = await response.json();

            if (result.error) {
                alert(`Import failed: ${result.error}`);
            } else {
                alert(`Imported ${result.imported} memories${result.failed ? ` (${result.failed} failed)` : ''}`);
                await this.loadAnalytics();
            }
        } catch (err) {
            alert(`Import error: ${err.message}`);
        } finally {
            btn.textContent = 'Import Memory';
            btn.disabled = false;
        }
    }

    async deleteAllMemories() {
        if (!confirm('Delete ALL memories? This cannot be undone.')) return;
        if (!confirm('Second confirmation: permanently wipe all learned facts and preferences?')) return;

        const btn = document.getElementById('delete-memories-btn');
        btn.textContent = 'Deleting...';
        btn.disabled = true;

        try {
            await fetch('/api/memory', { method: 'DELETE' });
            await this.loadAnalytics();
            this.showServerPush('All memories deleted');
        } catch (err) {
            alert(`Delete error: ${err.message}`);
        } finally {
            btn.textContent = 'Delete All Memories';
            btn.disabled = false;
        }
    }

    // --- Approval ---

    renderApprovalDialog(approval) {
        let actionLabel = approval.action;
        if (approval.action === 'create_file') actionLabel = 'Create File';
        else if (approval.action === 'execute_command') actionLabel = 'Run Command';

        const executableBadge = approval.executable ? '<span class="badge executable">executable</span>' : '';

        const contentPreview = approval.content.length > 2000
            ? approval.content.substring(0, 2000) + '\n\n[... truncated ...]'
            : approval.content;

        return `
            <div class="approval-dialog" data-approval-id="${approval.id}">
                <div class="approval-header">
                    <span class="approval-icon">⚠️</span>
                    <span class="approval-title">${actionLabel} Requested</span>
                    ${executableBadge}
                </div>
                <div class="approval-path">
                    <strong>Path:</strong> <code>${this.escapeHtml(approval.path)}</code>
                </div>
                <div class="approval-content">
                    <strong>Content:</strong>
                    <pre class="file-preview">${this.escapeHtml(contentPreview)}</pre>
                </div>
                <div class="approval-buttons">
                    <button class="btn approve-btn" onclick="app.handleApproval(true)">
                        ✓ Approve
                    </button>
                    <button class="btn deny-btn" onclick="app.handleApproval(false)">
                        ✗ Deny
                    </button>
                </div>
            </div>
        `;
    }

    async handleApproval(approved) {
        const dialog = document.querySelector('.approval-dialog');
        if (!dialog) return;

        const buttons = dialog.querySelectorAll('button');
        buttons.forEach(btn => btn.disabled = true);

        if (window.neuralBg) window.neuralBg.deactivate();

        try {
            const response = await fetch('/api/approve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ approved })
            });

            const data = await response.json();

            const resultClass = approved ? 'approval-success' : 'approval-denied';
            const resultIcon = approved ? '✓' : '✗';
            dialog.innerHTML = `
                <div class="${resultClass}">
                    <span>${resultIcon}</span> ${this.escapeHtml(data.message)}
                </div>
            `;
        } catch (error) {
            dialog.innerHTML = `
                <div class="approval-error">
                    Error: ${this.escapeHtml(error.message)}
                </div>
            `;
        }
    }
    // --- Custom Tools Panel ---

    _bindToolsPanel() {
        document.getElementById('tools-panel-btn').addEventListener('click', () => this.openToolsPanel());
        document.getElementById('close-tools-panel-btn').addEventListener('click', () => this.closeToolsPanel());
        document.getElementById('tools-propose-btn').addEventListener('click', () => this.proposeTool());
        document.getElementById('tools-propose-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.proposeTool(); }
        });
    }

    openToolsPanel() {
        this._closeAllPanels();
        document.getElementById('tools-panel').classList.add('open');
        this.loadTools();
    }

    closeToolsPanel() {
        document.getElementById('tools-panel').classList.remove('open');
    }

    async loadTools() {
        const list = document.getElementById('tools-list');
        list.innerHTML = '<div class="tools-loading">Loading...</div>';
        try {
            const r = await fetch('/api/tools');
            if (!r.ok) throw new Error('Failed to load');
            const data = await r.json();
            this._renderTools(data);
        } catch (e) {
            list.innerHTML = `<div class="tools-loading">Error loading tools</div>`;
        }
    }

    _renderTools({ installed, code_ready, proposals }) {
        const list = document.getElementById('tools-list');
        let html = '';

        // Installed
        html += `<div class="tools-section-heading">Installed (${installed.length})</div>`;
        if (installed.length === 0) {
            html += `<div class="tools-empty">No custom tools installed yet.</div>`;
        } else {
            installed.forEach(t => {
                html += this._toolCard(t, [
                    { label: 'Remove', cls: 'danger', action: `app.removeTool('${t.id}')` },
                ]);
            });
        }

        // Code ready — awaiting install
        if (code_ready.length > 0) {
            html += `<div class="tools-section-heading">Ready to Install (${code_ready.length})</div>`;
            code_ready.forEach(t => {
                html += this._toolCard(t, [
                    { label: 'View Code', cls: '', action: `app.toggleToolCode('${t.id}')` },
                    { label: 'Install', cls: 'primary', action: `app.installTool('${t.id}')` },
                ], t.code);
            });
        }

        // Proposals — awaiting code generation
        if (proposals.length > 0) {
            html += `<div class="tools-section-heading">Proposals (${proposals.length})</div>`;
            proposals.forEach(t => {
                html += this._toolCard(t, [
                    { label: 'Generate Code', cls: 'primary', action: `app.generateToolCode('${t.id}')` },
                ]);
            });
        }

        list.innerHTML = html || '<div class="tools-empty">No tools or proposals yet.</div>';
    }

    _toolCard(tool, actions, code = '') {
        const statusLabels = { installed: 'installed', code_ready: 'code ready', proposal: 'proposal' };
        const label = statusLabels[tool.status] || tool.status;
        const actionsHtml = actions.map(a =>
            `<button class="tool-action-btn ${a.cls}" onclick="${a.action}">${a.label}</button>`
        ).join('');
        const codeHtml = code
            ? `<div class="tool-code-viewer" id="code-${tool.id}"><pre>${this.escapeHtml(code)}</pre></div>`
            : '';
        return `
            <div class="tool-card" id="card-${tool.id}">
                <div class="tool-card-top">
                    <span class="tool-card-name">${this.escapeHtml(tool.name || '(unnamed)')}</span>
                    <span class="tool-card-status tool-status-${tool.status}">${label}</span>
                </div>
                <div class="tool-card-desc">${this.escapeHtml(tool.description)}</div>
                ${codeHtml}
                <div class="tool-card-actions">${actionsHtml}</div>
            </div>`;
    }

    toggleToolCode(toolId) {
        const viewer = document.getElementById(`code-${toolId}`);
        if (viewer) viewer.classList.toggle('open');
    }

    async generateToolCode(toolId) {
        const btn = document.querySelector(`#card-${toolId} .tool-action-btn.primary`);
        if (btn) { btn.disabled = true; btn.textContent = 'Generating...'; }
        try {
            const r = await fetch(`/api/tools/${toolId}/generate`, { method: 'POST' });
            const data = await r.json();
            if (!r.ok) throw new Error(data.detail || 'Failed');
            this.loadTools();
        } catch (e) {
            if (btn) { btn.disabled = false; btn.textContent = 'Generate Code'; }
            this.showServerPush(`Code generation failed: ${e.message}`);
        }
    }

    async installTool(toolId) {
        const btn = document.querySelector(`#card-${toolId} .tool-action-btn.primary`);
        if (btn) { btn.disabled = true; btn.textContent = 'Installing...'; }
        try {
            const r = await fetch(`/api/tools/${toolId}/install`, { method: 'POST' });
            const data = await r.json();
            if (!r.ok) throw new Error(data.detail || 'Install failed');
            this.showServerPush(`✓ ${data.message}`);
            this.loadTools();
        } catch (e) {
            if (btn) { btn.disabled = false; btn.textContent = 'Install'; }
            this.showServerPush(`Install failed: ${e.message}`);
        }
    }

    async removeTool(toolId) {
        if (!confirm('Remove this tool? It will be uninstalled from this session.')) return;
        try {
            const r = await fetch(`/api/tools/${toolId}`, { method: 'DELETE' });
            const data = await r.json();
            if (!r.ok) throw new Error(data.detail || 'Remove failed');
            this.showServerPush(`✓ ${data.message}`);
            this.loadTools();
        } catch (e) {
            this.showServerPush(`Remove failed: ${e.message}`);
        }
    }

    async proposeTool() {
        const input = document.getElementById('tools-propose-input');
        const btn = document.getElementById('tools-propose-btn');
        const status = document.getElementById('tools-propose-status');
        const desc = input.value.trim();
        if (!desc) return;

        btn.disabled = true;
        status.className = 'tools-propose-status';
        status.textContent = 'Proposing tool...';
        status.classList.remove('hidden');

        try {
            const r = await fetch('/api/tools/propose', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description: desc }),
            });
            const data = await r.json();
            if (!r.ok) throw new Error(data.detail || 'Failed');
            input.value = '';
            status.className = 'tools-propose-status success';
            status.textContent = `✓ Proposed: ${data.name} — click Generate Code to continue.`;
            this.loadTools();
        } catch (e) {
            status.className = 'tools-propose-status error';
            status.textContent = e.message;
        } finally {
            btn.disabled = false;
        }
    }
    // ── Research Panel ──────────────────────────────────────────────────

    _bindResearchPanel() {
        document.getElementById('research-panel-btn').addEventListener('click', () => this.openResearchPanel());
        document.getElementById('close-research-panel-btn').addEventListener('click', () => this.closeResearchPanel());
    }

    openResearchPanel() {
        this._closeAllPanels();
        document.getElementById('research-panel').classList.add('open');
        this.loadResearch();
    }

    closeResearchPanel() {
        document.getElementById('research-panel').classList.remove('open');
    }

    async loadResearch() {
        const list = document.getElementById('research-list');
        list.innerHTML = '<div class="research-loading">Loading...</div>';
        try {
            const r = await fetch('/api/research/data');
            if (!r.ok) throw new Error('Failed to load research data');
            const data = await r.json();
            this._renderResearch(data);
        } catch (e) {
            list.innerHTML = `<div class="research-loading">Error loading research: ${e.message}</div>`;
        }
    }

    _renderResearch({ topics, insights, stats }) {
        const list = document.getElementById('research-list');
        let html = '';

        // ── Insights section ──
        const quotaStr = stats.enabled
            ? `${stats.quota_remaining} quota remaining`
            : 'research disabled';
        html += `<div class="research-section-heading">Insights (${insights.length})</div>`;
        if (insights.length === 0) {
            html += `<div class="research-empty">No insights yet — enable research to start collecting findings.</div>`;
        } else {
            insights.forEach((ins, idx) => {
                const confLevel = ins.confidence >= 0.8 ? 'high' : ins.confidence >= 0.6 ? 'mid' : 'low';
                const confPct = Math.round(ins.confidence * 100);
                html += `
                <div class="research-card">
                    <div class="research-card-top">
                        <div class="research-card-fact">${this._escHtml(ins.fact)}</div>
                    </div>
                    <div class="research-card-meta">
                        <span>${this._escHtml(ins.topic)}</span>
                        <span class="research-confidence ${confLevel}">${confPct}%</span>
                        ${ins.shared ? '<span style="color:#3fb950;font-size:0.68rem">shared</span>' : ''}
                    </div>
                    ${ins.why_interesting ? `<div class="research-card-reason">${this._escHtml(ins.why_interesting)}</div>` : ''}
                    <div class="research-card-actions">
                        <button class="research-delete-btn" onclick="app.deleteInsight(${idx})" title="Delete insight">Delete</button>
                    </div>
                </div>`;
            });
        }

        // ── Topics section ──
        html += `<div class="research-section-heading">Topics (${topics.length}) &mdash; <span style="font-weight:400">${quotaStr}</span></div>`;
        if (topics.length === 0) {
            html += `<div class="research-empty">No topics queued. The system will add topics automatically from conversation patterns.</div>`;
        } else {
            topics.forEach((t, idx) => {
                const stars = '★'.repeat(t.priority) + '☆'.repeat(5 - t.priority);
                html += `
                <div class="research-card">
                    <div class="research-card-top">
                        <div class="research-card-name">${this._escHtml(t.name)}</div>
                        ${t.researched ? '<span class="research-researched" title="Already researched">✓</span>' : ''}
                    </div>
                    <div class="research-card-meta">
                        <span class="research-priority" title="Priority">${stars}</span>
                        ${t.search_query !== t.name ? `<span title="Search query">${this._escHtml(t.search_query)}</span>` : ''}
                    </div>
                    ${t.reason ? `<div class="research-card-reason">${this._escHtml(t.reason)}</div>` : ''}
                    <div class="research-card-actions">
                        <button class="research-delete-btn" onclick="app.deleteTopic(${idx})" title="Delete topic">Delete</button>
                    </div>
                </div>`;
            });
        }

        list.innerHTML = html;
    }

    _escHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    async deleteInsight(idx) {
        try {
            const r = await fetch(`/api/research/insight/${idx}`, { method: 'DELETE' });
            if (!r.ok) throw new Error('Delete failed');
            this.loadResearch();
        } catch (e) {
            this.showServerPush(`Delete failed: ${e.message}`);
        }
    }

    async deleteTopic(idx) {
        try {
            const r = await fetch(`/api/research/topic/${idx}`, { method: 'DELETE' });
            if (!r.ok) throw new Error('Delete failed');
            this.loadResearch();
        } catch (e) {
            this.showServerPush(`Delete failed: ${e.message}`);
        }
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
    window._closeAllPanels = () => window.app._closeAllPanels();
});
