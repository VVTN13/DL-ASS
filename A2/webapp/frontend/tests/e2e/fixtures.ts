import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { test as base, expect } from '@playwright/test'
import type { APIRequestContext } from '@playwright/test'

import { HomePage } from '../pages/HomePage'
import { RecordingPage } from '../pages/RecordingPage'

const SAMPLES = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../backend/samples')
const PSG = readFileSync(path.join(SAMPLES, 'SC4031-excerpt-PSG.edf'))
const HYPNOGRAM = readFileSync(path.join(SAMPLES, 'SC4031-excerpt-Hypnogram.edf'))

export const TRANSFORMER = 'transformer-a2-v1'
export const ATTNSLEEP = 'attnsleep-a2-v1'
export const N_EPOCHS = 120

export interface UploadFile {
  name: string
  mimeType: string
  buffer: Buffer
}

interface RecordingJson {
  id: string
  filename: string
  jobs: { id: string; model_id: string; status: string }[]
}

/**
 * The one-hour sample with a unique "recording identification" header field.
 * The signal is untouched but the SHA-256 differs, so every test owns its recording
 * even though the server de-duplicates identical uploads.
 */
export function uniquePsg(label: string): UploadFile {
  const buffer = Buffer.from(PSG)
  const tag = `Startdate X X X ${label}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  buffer.write(tag.slice(0, 80).padEnd(80), 88, 'ascii')
  return { name: `${label}-PSG.edf`, mimeType: 'application/octet-stream', buffer }
}

export function hypnogramFile(): UploadFile {
  return { name: 'night-Hypnogram.edf', mimeType: 'application/octet-stream', buffer: HYPNOGRAM }
}

/** Same signal with the second channel renamed: a valid EDF the models cannot use. */
export function psgWithWrongChannel(): UploadFile {
  const file = uniquePsg('wrong-channel')
  file.buffer.write('EEG C3-A2'.padEnd(16), 256 + 16, 'ascii')
  return file
}

class Api {
  private readonly created: string[] = []
  private readonly request: APIRequestContext

  constructor(request: APIRequestContext) {
    this.request = request
  }

  /** Upload through the API (fast setup for tests that are not about the upload form). */
  async createRecording(options: { label: string; hypnogram?: boolean; models?: string[] }) {
    const file = uniquePsg(options.label)
    const form = new FormData()
    form.append('edf', new File([new Uint8Array(file.buffer)], file.name))
    if (options.hypnogram) {
      form.append('hypnogram', new File([new Uint8Array(HYPNOGRAM)], 'night-Hypnogram.edf'))
    }
    for (const model of options.models ?? []) form.append('model_ids', model)
    const response = await this.request.post('/api/v1/recordings', { multipart: form })
    expect(response.status()).toBe(201)
    const recording = (await response.json()) as RecordingJson
    this.created.push(recording.id)
    return recording
  }

  async waitForJobs(recordingId: string): Promise<RecordingJson> {
    let recording!: RecordingJson
    await expect
      .poll(
        async () => {
          recording = await (await this.request.get(`/api/v1/recordings/${recordingId}`)).json()
          return recording.jobs.every((job) => job.status === 'succeeded')
        },
        { timeout: 150_000, intervals: [500] },
      )
      .toBe(true)
    return recording
  }

  track(recordingId: string) {
    this.created.push(recordingId)
  }

  async cleanup() {
    for (const id of this.created) await this.request.delete(`/api/v1/recordings/${id}`)
  }
}

interface Fixtures {
  api: Api
  homePage: HomePage
  recordingPage: RecordingPage
}

export const test = base.extend<Fixtures>({
  api: async ({ request }, provide) => {
    const api = new Api(request)
    await provide(api)
    await api.cleanup()
  },
  homePage: async ({ page }, provide) => provide(new HomePage(page)),
  recordingPage: async ({ page }, provide) => provide(new RecordingPage(page)),
})

export { expect }
