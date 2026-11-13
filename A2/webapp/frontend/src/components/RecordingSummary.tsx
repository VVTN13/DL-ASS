import type { ReactNode } from 'react'

import type { Recording } from '../api/client'
import { formatDuration } from '../lib/stages'

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className="mt-0.5 text-sm">{children}</dd>
    </div>
  )
}

/** What the server read from the file, so the user can check the input before trusting output. */
export function RecordingSummary({ recording }: { recording: Recording }) {
  const used = new Set(['EEG Fpz-Cz', 'EEG Pz-Oz'])
  const ignored = recording.channels.filter((channel) => !used.has(channel.label))
  return (
    <section className="card" aria-labelledby="input-title">
      <h2 id="input-title" className="card-title">
        Input
      </h2>
      <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-3 md:grid-cols-4" data-testid="recording-summary">
        <Field label="Channels used">
          {recording.channels
            .filter((channel) => used.has(channel.label))
            .map((channel) => `${channel.label} (${channel.sampling_rate} Hz, ${channel.unit})`)
            .join(' · ')}
        </Field>
        <Field label="Duration">{formatDuration(recording.duration_seconds)}</Field>
        <Field label="Epochs (30 s)">
          {recording.n_epochs}
          {recording.dropped_seconds > 0 && (
            <span className="text-slate-500"> · last {recording.dropped_seconds} s not scored</span>
          )}
        </Field>
        <Field label="Hypnogram">{recording.has_hypnogram ? 'uploaded' : 'none'}</Field>
        <Field label="SHA-256">
          <code className="text-xs" title={recording.sha256}>
            {recording.sha256.slice(0, 16)}…
          </code>
        </Field>
        <Field label="Size">{(recording.size_bytes / 1024 / 1024).toFixed(1)} MiB</Field>
        <Field label="Kept until">
          {recording.expires_at ? new Date(recording.expires_at).toLocaleDateString() : 'no expiry'}
        </Field>
        {ignored.length > 0 && (
          <Field label="Channels ignored">{ignored.map((channel) => channel.label).join(', ')}</Field>
        )}
      </dl>
    </section>
  )
}
