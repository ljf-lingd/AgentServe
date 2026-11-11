import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/app/',
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(
      ['/auth', '/sessions', '/tasks', '/admin', '/catalog', '/healthz'].map((path) => [
        path, { target: process.env.AGENTSERVE_WEB_API ?? 'http://127.0.0.1:8766', changeOrigin: false },
      ]),
    ),
  },
})
