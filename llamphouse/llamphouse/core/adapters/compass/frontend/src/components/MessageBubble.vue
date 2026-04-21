<script setup lang="ts">
import { ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import type { Message, MessageContent } from '../api/client'
import { formatTs } from '../api/client'

type RenderMode = 'plain' | 'markdown' | 'json'

const props = defineProps<{
  message: Message
  renderMode?: RenderMode
  threadId?: string
}>()

const router = useRouter()

// Per-message mode — starts from the global prop, can be overridden locally.
// When the global prop changes, reset to follow it.
const localMode = ref<RenderMode>(props.renderMode ?? 'plain')
watch(
  () => props.renderMode,
  (v) => { localMode.value = v ?? 'plain' },
)

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

// ── Lightweight markdown → HTML ──────────────────────────────────────────────
function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function renderMarkdown(src: string): string {
  const lines = src.split('\n')
  const out: string[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]

    // Fenced code block
    if (/^```/.test(line)) {
      const lang = line.slice(3).trim()
      const codeLines: string[] = []
      i++
      while (i < lines.length && !/^```/.test(lines[i])) {
        codeLines.push(lines[i])
        i++
      }
      i++ // skip closing ```
      out.push(`<pre class="md-pre"><code${lang ? ` class="language-${escHtml(lang)}"` : ''}>${escHtml(codeLines.join('\n'))}</code></pre>`)
      continue
    }

    // Headings
    const heading = line.match(/^(#{1,6})\s+(.+)$/)
    if (heading) {
      const level = heading[1].length
      out.push(`<h${level} class="md-h${level}">${inlineMarkdown(heading[2])}</h${level}>`)
      i++
      continue
    }

    // Blockquote
    if (/^>\s?/.test(line)) {
      const bqLines: string[] = []
      while (i < lines.length && /^>\s?/.test(lines[i])) {
        bqLines.push(lines[i].replace(/^>\s?/, ''))
        i++
      }
      out.push(`<blockquote class="md-blockquote">${bqLines.map(inlineMarkdown).join('<br>')}</blockquote>`)
      continue
    }

    // Unordered list
    if (/^[-*+]\s/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^[-*+]\s/.test(lines[i])) {
        items.push(`<li>${inlineMarkdown(lines[i].replace(/^[-*+]\s/, ''))}</li>`)
        i++
      }
      out.push(`<ul class="md-ul">${items.join('')}</ul>`)
      continue
    }

    // Ordered list
    if (/^\d+\.\s/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
        items.push(`<li>${inlineMarkdown(lines[i].replace(/^\d+\.\s/, ''))}</li>`)
        i++
      }
      out.push(`<ol class="md-ol">${items.join('')}</ol>`)
      continue
    }

    // Horizontal rule
    if (/^(---+|\*\*\*+|___+)\s*$/.test(line)) {
      out.push('<hr class="md-hr">')
      i++
      continue
    }

    // Blank line → paragraph break
    if (line.trim() === '') {
      out.push('<div class="md-spacer"></div>')
      i++
      continue
    }

    // Paragraph line
    out.push(`<p class="md-p">${inlineMarkdown(line)}</p>`)
    i++
  }

  return out.join('')
}

function inlineMarkdown(s: string): string {
  return escHtml(s)
    // Code span (do first so backticks aren't processed further)
    .replace(/`([^`]+)`/g, '<code class="md-code">$1</code>')
    // Bold+italic
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/__(.+?)__/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/_(.+?)_/g, '<em>$1</em>')
    // Strikethrough
    .replace(/~~(.+?)~~/g, '<del>$1</del>')
    // Links
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a class="md-link" href="$2" target="_blank" rel="noopener">$1</a>')
}

// ── JSON mode ────────────────────────────────────────────────────────────────
function renderJson(content: MessageContent[]): string {
  const text = extractText(content)
  try {
    return JSON.stringify(JSON.parse(text), null, 2)
  } catch {
    return text
  }
}

const MODES: RenderMode[] = ['plain', 'markdown', 'json']
const MODE_LABELS: Record<RenderMode, string> = { plain: 'Plain', markdown: 'MD', json: 'JSON' }
</script>

<template>
  <div class="msg" :class="`msg--${message.role}`">
    <div class="msg__header">
      <div class="msg__role-group">
        <span class="msg__role">{{ message.role }}</span>
        <span v-if="message.agent_name" class="msg__agent">{{ message.agent_name }}</span>
      </div>
      <div class="msg__header-right">
        <div class="seg-control">
          <button
            v-for="mode in MODES"
            :key="mode"
            class="seg-control__btn"
            :class="{ 'seg-control__btn--active': localMode === mode }"
            @click="localMode = mode"
          >
            {{ MODE_LABELS[mode] }}
          </button>
        </div>
        <span class="msg__time">{{ formatTs(message.created_at) }}</span>
      </div>
    </div>

    <!-- plain (default) -->
    <div v-if="localMode === 'plain'" class="msg__body">
      {{ extractText(message.content) }}
    </div>

    <!-- markdown -->
    <div
      v-else-if="localMode === 'markdown'"
      class="msg__body msg__body--markdown"
      v-html="renderMarkdown(extractText(message.content))"
    />

    <!-- json -->
    <pre v-else-if="localMode === 'json'" class="msg__body msg__body--json">{{ renderJson(message.content) }}</pre>

    <div v-if="message.run_id" class="msg__meta">
      <button
        v-if="props.threadId"
        class="msg__run-btn"
        @click="router.push(`/threads/${props.threadId}/runs/${message.run_id}`)"
      >
        <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        run {{ message.run_id.slice(0, 8) }}
      </button>
      <span v-else class="mono">run {{ message.run_id.slice(0, 8) }}</span>
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
  gap: 8px;
}

.msg__header-right {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}

/* Segmented control (per-message) */
.seg-control {
  display: inline-flex;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
  background: var(--bg-hover);
}

.seg-control__btn {
  padding: 2px 8px;
  font-size: 0.68rem;
  font-weight: 500;
  background: transparent;
  border: none;
  border-right: 1px solid var(--border);
  color: var(--text-muted);
  cursor: pointer;
  transition: background 0.12s, color 0.12s;
  line-height: 1.6;
}

.seg-control__btn:last-child {
  border-right: none;
}

.seg-control__btn:hover {
  background: var(--bg-surface);
  color: var(--text-primary);
}

.seg-control__btn--active {
  background: var(--accent);
  color: #fff;
}

.seg-control__btn--active:hover {
  background: var(--accent);
  color: #fff;
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

/* ── Markdown mode ────────────────────────────────────────── */
.msg__body--markdown {
  white-space: normal;
}

.msg__body--markdown :deep(.md-p) {
  margin: 0 0 0.5em;
}
.msg__body--markdown :deep(.md-spacer) {
  height: 0.4em;
}
.msg__body--markdown :deep(.md-h1),
.msg__body--markdown :deep(.md-h2),
.msg__body--markdown :deep(.md-h3),
.msg__body--markdown :deep(.md-h4),
.msg__body--markdown :deep(.md-h5),
.msg__body--markdown :deep(.md-h6) {
  font-weight: 700;
  margin: 0.75em 0 0.25em;
  line-height: 1.3;
}
.msg__body--markdown :deep(.md-h1) { font-size: 1.3em; }
.msg__body--markdown :deep(.md-h2) { font-size: 1.15em; }
.msg__body--markdown :deep(.md-h3) { font-size: 1.05em; }
.msg__body--markdown :deep(.md-h4),
.msg__body--markdown :deep(.md-h5),
.msg__body--markdown :deep(.md-h6) { font-size: 0.95em; }

.msg__body--markdown :deep(.md-ul),
.msg__body--markdown :deep(.md-ol) {
  margin: 0.3em 0 0.5em 1.4em;
  padding: 0;
}
.msg__body--markdown :deep(li) {
  margin: 0.1em 0;
}
.msg__body--markdown :deep(.md-blockquote) {
  border-left: 3px solid var(--border);
  margin: 0.4em 0;
  padding: 0.2em 0.8em;
  color: var(--text-muted);
  font-style: italic;
}
.msg__body--markdown :deep(.md-pre) {
  background: var(--bg-hover);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  margin: 0.5em 0;
  overflow-x: auto;
  font-family: var(--font-mono, monospace);
  font-size: 0.8rem;
  line-height: 1.5;
}
.msg__body--markdown :deep(.md-code) {
  font-family: var(--font-mono, monospace);
  font-size: 0.82em;
  background: var(--bg-hover);
  border: 1px solid var(--border-light);
  border-radius: 3px;
  padding: 1px 5px;
}
.msg__body--markdown :deep(.md-link) {
  color: var(--accent);
  text-decoration: underline;
  text-underline-offset: 2px;
}
.msg__body--markdown :deep(.md-hr) {
  border: none;
  border-top: 1px solid var(--border);
  margin: 0.75em 0;
}

/* ── JSON mode ────────────────────────────────────────────── */
.msg__body--json {
  white-space: pre;
  font-family: var(--font-mono, monospace);
  font-size: 0.78rem;
  line-height: 1.55;
  background: var(--bg-hover);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
}

.msg__meta {
  margin-top: 8px;
  font-size: 0.7rem;
  color: var(--text-muted);
}

.msg__run-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 0.7rem;
  font-family: var(--font-mono, monospace);
  color: var(--text-muted);
  background: transparent;
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  padding: 2px 8px;
  cursor: pointer;
  transition: color 0.12s, border-color 0.12s, background 0.12s;
}

.msg__run-btn:hover {
  color: var(--accent);
  border-color: var(--accent);
  background: var(--accent-dim);
}
</style>
