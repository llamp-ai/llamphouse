import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory('/compass/'),
  routes: [
    {
      path: '/',
      name: 'overview',
      component: () => import('../views/OverviewView.vue'),
    },
    {
      path: '/threads',
      name: 'threads',
      component: () => import('../views/ThreadsView.vue'),
    },
    {
      path: '/threads/:threadId',
      name: 'thread-detail',
      component: () => import('../views/ThreadDetailView.vue'),
    },
    {
      path: '/threads/:threadId/runs/:runId',
      name: 'run-detail',
      component: () => import('../views/RunDetailView.vue'),
    },
    {
      path: '/assistants',
      name: 'assistants',
      component: () => import('../views/AssistantsView.vue'),
    },
    {
      path: '/traces',
      name: 'traces',
      component: () => import('../views/TracesView.vue'),
    },
    {
      path: '/traces/:runId',
      name: 'trace-detail',
      component: () => import('../views/TraceDetailView.vue'),
    },
    {
      path: '/compare',
      name: 'compare',
      component: () => import('../views/CompareView.vue'),
    },
  ],
})

export default router
