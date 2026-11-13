import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// `npm run dev` proxies /api to the FastAPI dev server, mirroring Nginx in production.
const proxy = { '/api': process.env.API_URL ?? 'http://localhost:8000' }

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy },
  preview: { proxy },
})
