<script setup lang="ts">
import { ref, watch, computed } from 'vue'

type RenderMode = 'plain' | 'markdown' | 'json'

const props = withDefaults(defineProps<{
  content: string
  defaultMode?: RenderMode
  compact?: boolean
  maxHeight?: number
}>(), {
  defaultMode: 'plain',
  compact: false,
  maxHeight: 320,
})

const localMode = ref<RenderMode>(props.defaultMode)
watch(() => props.defaultMode, (v) => { localMode.value = v })

// Auto-pick: if content is valid JSON, default to JSON view
const looksLikeJson = computed(() => {
  const t = (props.content || '').trim()
  if (!(t.startsWith('{') || t.startsWith('['))) return false
  try { JSON.parse(t); return true } catch { return false }
})

// Prefer JSON when content is JSON and user hasn't overridden
watch(looksLikeJson, (v) => {
  if (v && localMode.value === 'plain' && props.defaultMode === 'plain') {
    localMode.value = 'json'
  }
}, { immediate: true })

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function inlineMarkdown(s: string): string {
  return escHtml(s)
    .replace(/`([^`]+)`/g, '<code class="md-code">$1</code>')
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/__(.+?)__/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/_(.+?)_/g, '<em>$1</em>')
    .replace(/~~(.+?)~~/g, '<del>$1</del>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a class="md-link" href="$2" target="_blank" rel="noopener">$1</a>')
}

function renderMarkdown(src: string): string {
  const lines = src.split('\n')
  const out: string[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (/^```/.test(line)) {
      const lang = line.slice(3).trim()
      const codeLines: string[] = []
      i++
      while (i < lines.length && !/^```/.test(lines[i])) { codeLines.push(lines[i]); i++ }
      i++
      out.push(`<pre class="md-pre"><code${lang ? ` class="language-${escHtml(lang)}"` : ''}>${escHtml(codeLines.join('\n'))}</code></pre>`)
      continue
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/)
    if (heading) { out.push(`<h${heading[1].length} class="md-h${heading[1].length}">${inlineMarkdown(heading[2])}</h${heading[1].length}>`); i++; continue }
    if (/^>\s?/.test(line)) {
      const bq: string[] = []
      while (i < lines.length && /^>\s?/.test(lines[i])) { bq.push(lines[i].replace(/^>\s?/, '')); i++ }
      out.push(`<blockquote class="md-blockquote">${bq.map(inlineMarkdown).join('<br>')}</blockquote>`)
      continue
    }
    if (/^[-*+]\s/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^[-*+]\s/.test(lines[i])) { items.push(`<li>${inlineMarkdown(lines[i].replace(/^[-*+]\s/, ''))}</li>`); i++ }
      out.push(`<ul class="md-ul">${items.join('')}</ul>`); continue
    }
    if (/^\d+\.\s/.test(line)) {
      const items: string[] = []
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) { items.push(`<li>${inlineMarkdown(lines[i].replace(/^\d+\.\s/, ''))}</li>`); i++ }
      out.push(`<ol class="md-ol">${items.join('')}</ol>`); continue
    }
    if (/^(---+|\*\*\*+|___+)\s*$/.test(line)) { out.push('<hr class="md-hr">'); i++; continue }
    if (line.trim() === '') { out.push('<div class="md-spacer"></div>'); i++; continue }
    out.push(`<p class="md-p">${inlineMarkdown(line)}</p>`)
    i++
  }
  return out.join('')
}

const jsonPretty = computed(() => {
  try { return JSON.stringify(JSON.parse(props.content), null, 2) }
  catch { return props.content }
})

const MODES: RenderMode[] = ['plain', 'markdown', 'json']
const LABELS: Record<RenderMode, string> = { plain: 'Plain', markdown: 'MD', json: 'JSON' }
</script>

<template>
  <div class="content-block" :class="{ 'content-block--compact': compact }">
    <div class="content-block__toolbar">
      <div class="seg-control">
        <button
          v-for="m in MODES"
          :key="m"
          class="seg-control__btn"
          :class="{ 'seg-control__btn--active': localMode === m }"
          @click.stop="localMode = m"
          type="button"
        >{{ LABELS[m] }}</button>
      </div>
    </div>
    <div
      class="content-block__body"
      :style="{ maxHeight: maxHeight + 'px' }"
    >
      <div v-if="localMode === 'plain'" class="content-block__plain">{{ content }}</div>
      <div v-else-if="localMode === 'markdown'" class="content-block__md" v-html="renderMarkdown(content)" />
      <pre v-else class="content-block__json">{{ jsonPretty }}</pre>
    </div>
  </div>
</template>

<style scoped>
.content-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}

.content-block__toolbar {
  display: flex;
  justify-content: flex-end;
}

.seg-control {
  display: inline-flex;
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
  background: var(--bg-secondary);
}

.seg-control__btn {
  background: transparent;
  border: none;
  padding: 2px 8px;
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--text-muted);
  cursor: pointer;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  border-right: 1px solid var(--border);
}

.seg-control__btn:last-child { border-right: none; }

.seg-control__btn:hover {
  background: var(--bg-hover);
  color: var(--text-primary);
}

.seg-control__btn--active {
  background: var(--accent);
  color: white;
}

.content-block__body {
  overflow: auto;
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--bg-secondary);
  padding: 8px 10px;
  font-size: 0.82rem;
  line-height: 1.5;
}

.content-block--compact .content-block__body {
  padding: 6px 8px;
  font-size: 0.78rem;
}

.content-block__plain {
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text-primary);
}

.content-block__json {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 0.75rem;
  color: var(--text-primary);
}

.content-block__md {
  color: var(--text-primary);
}

.content-block__md :deep(.md-p) { margin: 0 0 6px 0; }
.content-block__md :deep(.md-p:last-child) { margin-bottom: 0; }
.content-block__md :deep(.md-h1),
.content-block__md :deep(.md-h2),
.content-block__md :deep(.md-h3),
.content-block__md :deep(.md-h4),
.content-block__md :deep(.md-h5),
.content-block__md :deep(.md-h6) {
  margin: 8px 0 4px 0;
  font-weight: 700;
  line-height: 1.25;
}
.content-block__md :deep(.md-h1) { font-size: 1.05rem; }
.content-block__md :deep(.md-h2) { font-size: 1rem; }
.content-block__md :deep(.md-h3) { font-size: 0.9rem; }
.content-block__md :deep(.md-ul),
.content-block__md :deep(.md-ol) { margin: 4px 0 6px 20px; padding: 0; }
.content-block__md :deep(.md-ul li),
.content-block__md :deep(.md-ol li) { margin: 2px 0; }
.content-block__md :deep(.md-blockquote) {
  border-left: 3px solid var(--border);
  padding: 4px 8px;
  color: var(--text-secondary);
  margin: 4px 0;
  background: var(--bg-hover);
  border-radius: 0 6px 6px 0;
}
.content-block__md :deep(.md-code) {
  background: var(--bg-hover);
  padding: 1px 4px;
  border-radius: 3px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 0.85em;
}
.content-block__md :deep(.md-pre) {
  background: var(--bg-hover);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
  overflow-x: auto;
  margin: 6px 0;
  font-size: 0.75rem;
}
.content-block__md :deep(.md-hr) {
  border: none;
  border-top: 1px solid var(--border);
  margin: 8px 0;
}
.content-block__md :deep(.md-spacer) { height: 4px; }
.content-block__md :deep(.md-link) {
  color: var(--accent);
  text-decoration: underline;
}
</style>
