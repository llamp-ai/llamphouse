/**
 * Persistent table preferences stored in localStorage.
 *
 * Each named table (identified by `tableId`) keeps:
 *   - hiddenCols   — keys of hidden columns
 *   - widths       — manually resized column widths in px
 *   - sort         — active sort key + direction
 *
 * Filters are intentionally NOT persisted — they are ephemeral and
 * restoring old filter values on page load is more annoying than helpful.
 */

import { ref, watch } from 'vue'

export interface SortState {
  key: string
  dir: 'asc' | 'desc'
}

interface StoredPrefs {
  hiddenCols: string[]
  widths: Record<string, number>
  sort: SortState | null
}

const STORAGE_PREFIX = 'compass.table.'

function load(tableId: string): StoredPrefs {
  try {
    const raw = localStorage.getItem(STORAGE_PREFIX + tableId)
    if (raw) return JSON.parse(raw) as StoredPrefs
  } catch {
    // corrupted storage — ignore
  }
  return { hiddenCols: [], widths: {}, sort: null }
}

function save(tableId: string, prefs: StoredPrefs) {
  try {
    localStorage.setItem(STORAGE_PREFIX + tableId, JSON.stringify(prefs))
  } catch {
    // storage full or private mode — ignore
  }
}

export function useTablePrefs(tableId: string | undefined) {
  const stored = tableId ? load(tableId) : { hiddenCols: [], widths: {}, sort: null }

  const hiddenCols = ref<string[]>(stored.hiddenCols)
  const colWidths  = ref<Record<string, number>>(stored.widths)
  const sort       = ref<SortState | null>(stored.sort)

  function persist() {
    if (!tableId) return
    save(tableId, {
      hiddenCols: hiddenCols.value,
      widths: colWidths.value,
      sort: sort.value,
    })
  }

  function setWidth(key: string, px: number) {
    colWidths.value = { ...colWidths.value, [key]: px }
    persist()
  }

  function toggleHidden(key: string) {
    const idx = hiddenCols.value.indexOf(key)
    hiddenCols.value = idx === -1
      ? [...hiddenCols.value, key]
      : hiddenCols.value.filter((k) => k !== key)
    persist()
  }

  function setSort(key: string) {
    if (sort.value?.key === key) {
      sort.value = sort.value.dir === 'asc'
        ? { key, dir: 'desc' }
        : null
    } else {
      sort.value = { key, dir: 'asc' }
    }
    persist()
  }

  function resetPrefs() {
    hiddenCols.value = []
    colWidths.value = {}
    sort.value = null
    persist()
  }

  return { hiddenCols, colWidths, sort, setWidth, toggleHidden, setSort, resetPrefs }
}
