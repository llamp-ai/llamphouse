import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  base: '/compass/',
  build: {
    outDir: '../static',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // Stable filenames — no content hashes.
        // Compass assets are served by the same runtime at the same version,
        // so cache-busting hashes are unnecessary and create noisy git diffs.
        entryFileNames: 'assets/[name].js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name][extname]',
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/compass/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
})
