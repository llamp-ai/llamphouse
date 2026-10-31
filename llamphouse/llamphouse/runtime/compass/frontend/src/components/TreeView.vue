<script setup lang="ts">
/* A tree of runs or threads: forks, calls, the current one marked. */
import type { TreeNode } from '../api/client'
import Badge from './Badge.vue'
defineProps<{ node: TreeNode; kind: 'run' | 'thread'; root?: boolean }>()
</script>
<template>
  <ul :class="root ? 'tree-root' : 'tree'">
    <li>
      <span class="pill">{{ node.kind }}</span>
      <router-link :to="kind === 'run' ? `/runs/${node.id}` : `/threads/${node.id}`" class="mono"
                   :style="node.current ? 'font-weight:700' : ''">{{ node.id }}</router-link>
      <span class="muted small"> {{ node.agent }}</span>
      <Badge :status="node.status" style="margin-left:6px" />
      <span v-if="node.current" class="pill" style="margin-left:6px">this one</span>
      <TreeView v-for="c in node.children" :key="c.id" :node="c" :kind="kind" />
    </li>
  </ul>
</template>
