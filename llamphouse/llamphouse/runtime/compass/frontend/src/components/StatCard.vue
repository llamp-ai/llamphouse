<script setup lang="ts">
defineProps<{
  icon: string
  label: string
  value: string | number
  sub?: string
  variant?: 'default' | 'success' | 'warning' | 'error'
}>()
</script>

<template>
  <div class="stat-card" :class="variant ? `stat-card--${variant}` : ''">
    <div class="stat-card__icon">{{ icon }}</div>
    <div class="stat-card__body">
      <div class="stat-card__value">{{ value }}</div>
      <div class="stat-card__label">{{ label }}</div>
      <div v-if="sub" class="stat-card__sub">{{ sub }}</div>
    </div>
  </div>
</template>

<style scoped>
.stat-card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px;
  display: flex;
  align-items: flex-start;
  gap: 14px;
  transition: border-color var(--transition);
  box-shadow: var(--shadow-sm);
}

.stat-card:hover {
  border-color: var(--border-light);
}

.stat-card__icon {
  font-size: 0.85rem;
  font-weight: 700;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--accent-dim);
  color: var(--accent);
  border-radius: var(--radius-md);
  flex-shrink: 0;
  font-family: var(--font-mono);
}

.stat-card--success .stat-card__icon { background: var(--success-dim); color: var(--success); }
.stat-card--warning .stat-card__icon { background: var(--warning-dim); color: var(--warning); }
.stat-card--error .stat-card__icon   { background: var(--error-dim); color: var(--error); }

.stat-card__value {
  font-size: 1.5rem;
  font-weight: 700;
  line-height: 1.2;
}

.stat-card__label {
  font-size: 0.8rem;
  color: var(--text-secondary);
  margin-top: 2px;
}

.stat-card__sub {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-top: 4px;
}
</style>
