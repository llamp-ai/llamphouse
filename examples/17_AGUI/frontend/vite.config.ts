import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// The page talks to the LLAMPHouse server through this dev server, so the
// browser sees one origin: /ag-ui goes to the server (`llamphouse up`).
const server = process.env.LLAMPHOUSE_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: { "/ag-ui": { target: server, changeOrigin: true } },
  },
  preview: {
    proxy: { "/ag-ui": { target: server, changeOrigin: true } },
  },
});
