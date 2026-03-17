<script setup lang="ts">
withDefaults(defineProps<{
  columns: { key: string; label: string; mono?: boolean; width?: string }[]
  rows?: Record<string, any>[]
  clickable?: boolean
}>(), {
  rows: () => [],
})

const emit = defineEmits<{
  rowClick: [row: Record<string, any>]
}>()
</script>

<template>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th
            v-for="col in columns"
            :key="col.key"
            :style="col.width ? { width: col.width } : {}"
          >
            {{ col.label }}
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="(row, i) in rows"
          :key="i"
          :class="{ clickable }"
          @click="clickable && emit('rowClick', row)"
        >
          <td
            v-for="col in columns"
            :key="col.key"
            :class="{ mono: col.mono }"
          >
            <slot :name="col.key" :row="row" :value="row[col.key]">
              {{ row[col.key] ?? '—' }}
            </slot>
          </td>
        </tr>
        <tr v-if="rows.length === 0">
          <td :colspan="columns.length" style="text-align: center; color: var(--text-muted); padding: 32px;">
            No data
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
