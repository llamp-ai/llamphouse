import { createRouter, createWebHistory } from 'vue-router'

const view = (name: string) => () => import(`../views/${name}.vue`)

const router = createRouter({
  history: createWebHistory(document.querySelector('base')?.getAttribute('href') ?? '/compass/'),
  routes: [
    { path: '/', name: 'overview', component: view('OverviewView') },
    { path: '/agents', name: 'agents', component: view('AgentsView') },
    { path: '/agents/:name', name: 'agent', component: view('AgentDetailView') },
    { path: '/threads', name: 'threads', component: view('ThreadsView') },
    { path: '/threads/:threadId', name: 'thread', component: view('ThreadDetailView') },
    { path: '/threads/:threadId/runs/:runId', redirect: to => `/runs/${to.params.runId}` },
    { path: '/runs', name: 'runs', component: view('RunsView') },
    { path: '/runs/:runId', name: 'run', component: view('RunDetailView') },
    { path: '/compare', name: 'compare', component: view('CompareView') },
    { path: '/traces', name: 'traces', component: view('TracesView') },
    { path: '/traces/:runId', name: 'trace-detail', component: view('TraceDetailView') },
    { path: '/triggers', name: 'triggers', component: view('TriggersView') },
    { path: '/subscriptions', name: 'subscriptions', component: view('SubscriptionsView') },
    { path: '/dashboards', name: 'dashboards', component: view('DashboardsView') },
    { path: '/dashboards/:dashboardId', name: 'dashboard', component: view('DashboardView') },
    { path: '/dashboards/:dashboardId/present', name: 'dashboard-present', component: view('DashboardPresentView'), meta: { fullscreen: true } },
    { path: '/learn', name: 'learn', component: view('LearnView') },
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
})

export default router
