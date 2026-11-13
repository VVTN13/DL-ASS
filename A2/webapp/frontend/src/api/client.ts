import createClient from 'openapi-fetch'

import type { components, paths } from './schema'

export type Model = components['schemas']['ModelPublic']
export type Recording = components['schemas']['RecordingPublic']
export type Job = components['schemas']['JobPublic']
export type Predictions = components['schemas']['PredictionsPublic']
export type Metrics = components['schemas']['MetricsPublic']
export type Epoch = components['schemas']['EpochPublic']
export type MetricsScope = Metrics['scope']

export const api = createClient<paths>({ baseUrl: '/' })

/** A failed request, carrying the backend's stable `code` (for example `EDF_MISSING_CHANNEL`). */
export class ApiError extends Error {
  readonly status: number
  readonly code: string

  constructor(status: number, code: string, message: string) {
    super(message)
    this.status = status
    this.code = code
  }
}

function toApiError(status: number, body: unknown): ApiError {
  const detail = (body as { detail?: unknown } | undefined)?.detail
  if (detail && typeof detail === 'object' && 'code' in detail && 'message' in detail) {
    return new ApiError(status, String(detail.code), String(detail.message))
  }
  if (status === 413) return new ApiError(status, 'UPLOAD_TOO_LARGE', 'The file is too large.')
  return new ApiError(status, 'REQUEST_FAILED', `The request failed (HTTP ${status}).`)
}

/** Turn an openapi-fetch result into data, or throw an `ApiError`. */
export async function unwrap<T>(
  request: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  const { data, error, response } = await request
  if (!response.ok || error !== undefined) throw toApiError(response.status, error)
  return data as T
}

export interface UploadInput {
  edf: File
  hypnogram: File | null
  modelIds: string[]
}

/** Multipart upload; `fetch` builds the boundary, so this bypasses the JSON client. */
export async function uploadRecording(input: UploadInput): Promise<Recording> {
  const form = new FormData()
  form.append('edf', input.edf)
  if (input.hypnogram) form.append('hypnogram', input.hypnogram)
  for (const modelId of input.modelIds) form.append('model_ids', modelId)
  const response = await fetch('/api/v1/recordings', { method: 'POST', body: form })
  const body: unknown = await response.json().catch(() => undefined)
  if (!response.ok) throw toApiError(response.status, body)
  return body as Recording
}

export function exportUrl(jobId: string, format: 'csv' | 'json'): string {
  return `/api/v1/jobs/${jobId}/export?format=${format}`
}
