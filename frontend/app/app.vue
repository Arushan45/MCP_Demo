<template>
  <v-app>
    <v-main class="bg-grey-darken-4">
      <v-container class="fill-height d-flex flex-column" max-width="800">
        <header class="py-4 text-center">
          <h1 class="text-h4 font-weight-bold">MCP Agent Chat</h1>
          <p class="text-body-2 text-grey-lighten-1">
            Ask anything. Backend: FastAPI + Groq + MCP (Playwright, Airbnb, DuckDuckGo).
          </p>
        </header>

        <v-card class="flex-grow-1 d-flex flex-column" elevation="4">
          <v-card-text class="flex-grow-1 overflow-y-auto chat-scroll">
            <div v-if="messages.length === 0" class="text-grey text-center mt-8">
              Start by typing a question below.
            </div>
            <div v-for="(m, i) in messages" :key="i" class="mb-4">
              <div class="d-flex" :class="m.role === 'user' ? 'justify-end' : 'justify-start'">
                <v-chip
                  :color="m.role === 'user' ? 'indigo' : 'grey-darken-3'"
                  text-color="white"
                  class="py-3 px-4"
                >
                  <span v-if="m.role === 'user'">{{ m.content }}</span>
                  <span v-else style="white-space: pre-wrap">{{ m.content }}</span>
                </v-chip>
              </div>
            </div>
            <div v-if="loading" class="d-flex align-center mt-2">
              <v-progress-circular indeterminate size="20" width="3" color="indigo" class="mr-2" />
              <span class="text-caption text-grey-lighten-1">Thinking with MCP tools…</span>
            </div>
          </v-card-text>

          <v-divider />

          <v-card-actions class="pa-3">
            <v-text-field
              v-model="input"
              :disabled="loading"
              placeholder="Ask something…"
              variant="outlined"
              hide-details
              density="comfortable"
              class="flex-grow-1 mr-2"
              @keyup.enter="onSend"
            />
            <v-btn
              color="indigo"
              :disabled="loading || !input.trim()"
              @click="onSend"
            >
              Send
            </v-btn>
          </v-card-actions>
        </v-card>

        <footer class="mt-3 text-center text-caption text-grey-lighten-1">
          Backend endpoint: <code>http://localhost:8000/api/agent</code>
        </footer>
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup lang="ts">
const input = ref('')
const loading = ref(false)

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

const messages = ref<ChatMessage[]>([])

const onSend = async () => {
  const text = input.value.trim()
  if (!text || loading.value) return

  messages.value.push({ role: 'user', content: text })
  input.value = ''
  loading.value = true

  try {
    const res = await $fetch<{
      final: string
    }>('http://localhost:8000/api/agent', {
      method: 'POST',
      body: { prompt: text },
    })

    messages.value.push({
      role: 'assistant',
      content: res.final || '(no response)',
    })
  } catch (e: any) {
    messages.value.push({
      role: 'assistant',
      content: `Error calling backend: ${e?.message || e}`,
    })
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.chat-scroll {
  max-height: 60vh;
}
</style>
