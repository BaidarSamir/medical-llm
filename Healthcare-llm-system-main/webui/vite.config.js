import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/analyze': 'http://127.0.0.1:9001',
      '/health': 'http://127.0.0.1:9001',
      '/departments': 'http://127.0.0.1:9001'
    }
  }
})
