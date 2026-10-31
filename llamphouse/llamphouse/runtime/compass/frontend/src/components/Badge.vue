<script setup lang="ts">
import { statusBadge } from '../api/client'
defineProps<{ status: string }>()
</script>
<template><span class="badge" :class="statusBadge(status)">{{ status.replace('_', ' ') }}</span></template>
