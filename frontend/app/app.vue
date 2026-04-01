<template>
  <v-app>
    <v-main>
      <div class="page-bg">
        <div class="chat-shell">
          <header class="hero">
            <h1>MCP Agent Chat</h1>
            <p>Ask anything. Backend: FastAPI + Groq + MCP (Playwright, Airbnb, DuckDuckGo).</p>
          </header>

          <section class="chat-card">
            <div ref="messagePanel" class="messages-panel">
              <div v-if="messages.length === 0" class="empty-state">
                Ask a question to start the conversation.
              </div>

              <article
                v-for="(m, i) in messages"
                :key="i"
                :class="['message-row', m.role === 'user' ? 'from-user' : 'from-assistant']"
              >
                <div class="message-label">
                  {{ m.role === 'user' ? 'You' : 'Assistant' }}
                </div>
                <div class="message-bubble">
                  {{ m.content }}
                </div>
                <details
                  v-if="m.role === 'assistant' && m.toolResults && m.toolResults.length > 0"
                  class="tool-status"
                >
                  <summary>
                    Tool status ({{ m.toolResults.length }} result{{ m.toolResults.length > 1 ? 's' : '' }})
                  </summary>
                  <div
                    v-for="(tool, idx) in m.toolResults"
                    :key="`${i}-${idx}-${tool.tool_name}`"
                    class="tool-item"
                  >
                    <div class="tool-head">
                      <span class="tool-name">{{ tool.tool_name }}</span>
                      <span class="tool-badge" :class="tool.isError ? 'bad' : 'good'">
                        {{ tool.isError ? 'Failed' : 'OK' }}
                      </span>
                    </div>
                    <div class="tool-meta">Server: {{ tool.server_name || 'unknown' }}</div>
                    <div v-if="tool.error" class="tool-error">
                      Error: {{ tool.error }}
                    </div>
                    <div v-else-if="tool.content" class="tool-content">
                      {{ tool.content }}
                    </div>
                    <div v-else class="tool-empty">
                      No output returned.
                    </div>
                  </div>
                </details>
              </article>

              <div v-if="loading" class="typing-indicator">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
                <span>Thinking with MCP tools...</span>
              </div>
            </div>

            <form class="input-row" @submit.prevent="onSend">
              <input
                v-model="input"
                :disabled="loading"
                placeholder="Ask something..."
                type="text"
              />
              <button type="submit" :disabled="loading || !input.trim()">Send</button>
            </form>
          </section>

          <footer class="footer-note">
            Backend endpoint: <code>{{ activeBackendUrl }}/api/agent</code>
          </footer>
        </div>
      </div>
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'

const input = ref('')
const loading = ref(false)
const messagePanel = ref<HTMLElement | null>(null)
const config = useRuntimeConfig()
const backendBaseUrl = String(config.public.backendBaseUrl || 'http://127.0.0.1:8000').replace(/\/$/, '')
const backendCandidates = Array.from(new Set([backendBaseUrl, 'http://127.0.0.1:8000', 'http://localhost:8000']))
const activeBackendUrl = ref(backendBaseUrl)

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  toolResults?: ToolResult[]
}

const messages = ref<ChatMessage[]>([])

interface ToolResult {
  tool_name: string
  server_name?: string
  isError?: boolean
  content?: string
  error?: string
}

interface AgentResponse {
  final: string
  tool_calls?: Array<{ name?: string }>
  tool_results?: ToolResult[]
}

watch(
  messages,
  async () => {
    await nextTick()
    if (!messagePanel.value) return
    messagePanel.value.scrollTop = messagePanel.value.scrollHeight
  },
  { deep: true }
)

const onSend = async () => {
  const text = input.value.trim()
  if (!text || loading.value) return

  messages.value.push({ role: 'user', content: text })
  input.value = ''
  loading.value = true

  try {
    let lastError: unknown = null
    let res: AgentResponse | null = null

    for (const baseUrl of backendCandidates) {
      try {
        res = await $fetch<AgentResponse>(`${baseUrl}/api/agent`, {
          method: 'POST',
          body: { prompt: text },
        })
        activeBackendUrl.value = baseUrl
        break
      } catch (innerError) {
        lastError = innerError
      }
    }

    if (!res) {
      throw lastError || new Error('Could not reach backend server')
    }

    messages.value.push({
      role: 'assistant',
      content: res.final || '(no response)',
      toolResults: res.tool_results || [],
    })
  } catch (e: any) {
    messages.value.push({
      role: 'assistant',
      content: `Error calling backend: ${e?.data?.detail || e?.message || e}`,
    })
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
:global(body) {
  margin: 0;
  font-family: "Segoe UI", "Inter", sans-serif;
}

.page-bg {
  min-height: 100vh;
  background:
    radial-gradient(circle at 15% 15%, rgba(255, 189, 89, 0.18), transparent 32%),
    radial-gradient(circle at 85% 10%, rgba(74, 144, 226, 0.16), transparent 28%),
    linear-gradient(140deg, #111318 0%, #191c23 55%, #1d1a24 100%);
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 24px 12px;
  color: #eef2ff;
}

.chat-shell {
  width: min(980px, 100%);
}

.hero {
  text-align: center;
  margin-bottom: 18px;
}

.hero h1 {
  margin: 0;
  font-size: clamp(1.8rem, 3.2vw, 2.6rem);
  letter-spacing: 0.03em;
}

.hero p {
  margin: 10px 0 0;
  color: #c8d1f0;
}

.chat-card {
  background: rgba(13, 16, 25, 0.78);
  border: 1px solid rgba(147, 170, 228, 0.25);
  border-radius: 18px;
  box-shadow: 0 24px 50px rgba(0, 0, 0, 0.38);
  overflow: hidden;
  backdrop-filter: blur(8px);
}

.messages-panel {
  min-height: 48vh;
  max-height: 64vh;
  overflow-y: auto;
  padding: 18px;
}

.empty-state {
  text-align: center;
  color: #9ea8c7;
  margin-top: 32px;
}

.message-row {
  display: flex;
  flex-direction: column;
  margin-bottom: 14px;
}

.from-user {
  align-items: flex-end;
}

.from-assistant {
  align-items: flex-start;
}

.message-label {
  font-size: 0.74rem;
  margin-bottom: 5px;
  color: #9da8ca;
  text-transform: uppercase;
  letter-spacing: 0.07em;
}

.message-bubble {
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.5;
  border-radius: 14px;
  padding: 12px 14px;
  width: fit-content;
  max-width: min(720px, 92%);
}

.from-user .message-bubble {
  background: linear-gradient(145deg, #2556cf, #3f70e0);
  color: #eaf1ff;
}

.from-assistant .message-bubble {
  background: rgba(225, 234, 255, 0.12);
  border: 1px solid rgba(170, 191, 240, 0.2);
  color: #edf2ff;
}

.tool-status {
  margin-top: 8px;
  width: min(720px, 92%);
  background: rgba(14, 19, 30, 0.62);
  border: 1px solid rgba(156, 176, 226, 0.25);
  border-radius: 10px;
  padding: 8px 10px;
}

.tool-status summary {
  cursor: pointer;
  color: #c8d7ff;
  font-size: 0.86rem;
}

.tool-item {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid rgba(156, 176, 226, 0.2);
}

.tool-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}

.tool-name {
  color: #f1f5ff;
  font-size: 0.86rem;
  font-weight: 600;
}

.tool-badge {
  font-size: 0.73rem;
  padding: 2px 7px;
  border-radius: 999px;
}

.tool-badge.good {
  background: rgba(72, 190, 129, 0.2);
  color: #9ce9ba;
}

.tool-badge.bad {
  background: rgba(241, 97, 97, 0.2);
  color: #ffb0b0;
}

.tool-meta {
  color: #9fb1df;
  font-size: 0.78rem;
  margin-top: 4px;
}

.tool-content,
.tool-error,
.tool-empty {
  white-space: pre-wrap;
  word-break: break-word;
  margin-top: 6px;
  font-size: 0.82rem;
  line-height: 1.4;
}

.tool-content {
  color: #dde6ff;
}

.tool-error {
  color: #ffc2c2;
}

.tool-empty {
  color: #b7c3e6;
}

.typing-indicator {
  margin-top: 8px;
  color: #b3bfdf;
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 0.92rem;
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #8ea8f8;
  animation: pulse 1.2s infinite ease-in-out;
}

.dot:nth-child(2) {
  animation-delay: 0.15s;
}

.dot:nth-child(3) {
  animation-delay: 0.3s;
}

@keyframes pulse {
  0%, 100% {
    opacity: 0.35;
    transform: translateY(0);
  }
  50% {
    opacity: 1;
    transform: translateY(-2px);
  }
}

.input-row {
  padding: 14px;
  border-top: 1px solid rgba(147, 170, 228, 0.25);
  display: flex;
  gap: 10px;
  background: rgba(8, 10, 16, 0.5);
}

.input-row input {
  flex: 1;
  border: 1px solid rgba(166, 184, 231, 0.35);
  background: rgba(11, 15, 24, 0.74);
  color: #eff3ff;
  padding: 12px 13px;
  border-radius: 10px;
  font-size: 1rem;
  outline: none;
}

.input-row input:focus {
  border-color: #8cb6ff;
  box-shadow: 0 0 0 3px rgba(140, 182, 255, 0.2);
}

.input-row button {
  border: none;
  border-radius: 10px;
  padding: 0 18px;
  font-weight: 600;
  cursor: pointer;
  background: linear-gradient(145deg, #ff9c47, #ff7f49);
  color: #111520;
}

.input-row button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.footer-note {
  text-align: center;
  margin-top: 12px;
  color: #afbbde;
}

.footer-note code {
  color: #ffd7af;
}

@media (max-width: 700px) {
  .page-bg {
    padding: 12px;
  }

  .messages-panel {
    min-height: 52vh;
    max-height: 68vh;
    padding: 12px;
  }

  .input-row {
    padding: 10px;
    gap: 8px;
  }

  .input-row button {
    padding: 0 14px;
  }
}
</style>
