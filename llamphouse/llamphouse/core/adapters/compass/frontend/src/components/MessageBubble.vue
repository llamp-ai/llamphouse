<script setup lang="ts">
import type { Message, MessageContent } from '../api/client'
import { formatTs } from '../api/client'

defineProps<{
  message: Message
}>()

function extractText(content: MessageContent[] | undefined): string {
  if (!content || !Array.isArray(content)) return ''
  return content
    .filter((c) => c.type === 'text' && c.text != null)
    .map((c) => {
      const t = c.text!
      return typeof t === 'string' ? t : t.value
    })
    .join('\n')
}
</script>

<template>
  <div class="msg" :class="`msg--${message.role}`">
    <div class="msg__header">
      <div class="msg__role-group">
        <span class="msg__role">{{ message.role }}</span>
        <span v-if="message.agent_name" class="msg__agent">{{ message.agent_name }}</span>
      </div>
      <span class="msg__time">{{ formatTs(message.created_at) }}</span>
    </div>
    <div class="msg__body">{{ extractText(message.content) }}</div>
    <div v-if="message.run_id" class="msg__meta">
      <span class="mono">run {{ message.run_id.slice(0, 8) }}</span>
    </div>
  </div>
</template>

<style scoped>
.msg {
  padding: 14px 16px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  background: var(--bg-surface);
}

.msg--user {
  background: var(--accent-dim);
  border-color: rgba(79, 70, 229, 0.15);
}

.msg--assistant {
  background: var(--bg-surface);
}

.msg--system {
  background: var(--bg-hover);
  border-color: var(--border-light);
}

.msg__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.msg__role-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.msg__role {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-secondary);
}

.msg__agent {
  font-size: 0.7rem;
  font-weight: 500;
  color: var(--text-muted);
  padding: 1px 7px;
  border-radius: 4px;
  background: var(--bg-hover);
  border: 1px solid var(--border-light);
}

.msg--user .msg__role  { color: var(--accent); }
.msg--assistant .msg__role { color: var(--success); }

.msg__time {
  font-size: 0.7rem;
  color: var(--text-muted);
}

.msg__body {
  font-size: 0.875rem;
  line-height: 1.65;
  white-space: pre-wrap;
  word-break: break-word;
}

.msg__meta {
  margin-top: 8px;
  font-size: 0.7rem;
  color: var(--text-muted);
}
</style>
