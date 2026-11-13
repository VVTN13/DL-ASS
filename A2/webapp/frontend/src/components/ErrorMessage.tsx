import { ApiError } from '../api/client'

/** Shows a failed request. `data-code` exposes the backend's stable error code. */
export function ErrorMessage({ error }: { error: unknown }) {
  if (!error) return null
  const code = error instanceof ApiError ? error.code : 'NETWORK_ERROR'
  const message =
    error instanceof ApiError ? error.message : 'Could not reach the server. Check your connection.'
  return (
    <p
      role="alert"
      data-code={code}
      className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800"
    >
      {message}
    </p>
  )
}
