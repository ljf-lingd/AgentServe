import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  timeout: 120_000,
  expect: { timeout: 65_000 },
  workers: 1,
  use: {
    baseURL: process.env.AGENTSERVE_TEST_URL ?? 'http://127.0.0.1:8766',
    channel: process.env.AGENTSERVE_TEST_CHANNEL ?? 'chromium',
    acceptDownloads: true,
  },
})
