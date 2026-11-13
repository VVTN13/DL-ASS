import { useState } from 'react'
import type { FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'

import type { Model } from '../api/client'
import { useUploadRecording } from '../api/queries'
import { ErrorMessage } from './ErrorMessage'

const FILE_INPUT =
  'block w-full max-w-md text-sm text-slate-700 file:mr-3 file:rounded-lg file:border file:border-slate-300 file:bg-white file:px-3 file:py-1.5 file:text-sm file:font-medium hover:file:bg-slate-100'

export function UploadForm({ models }: { models: Model[] }) {
  const navigate = useNavigate()
  const upload = useUploadRecording()
  const runnable = models.filter((model) => model.available)
  const [edf, setEdf] = useState<File | null>(null)
  const [hypnogram, setHypnogram] = useState<File | null>(null)
  const [unchecked, setUnchecked] = useState<Set<string>>(new Set())
  const modelIds = runnable.filter((model) => !unchecked.has(model.id)).map((model) => model.id)

  function toggle(modelId: string) {
    setUnchecked((current) => {
      const next = new Set(current)
      if (!next.delete(modelId)) next.add(modelId)
      return next
    })
  }

  function submit(event: FormEvent) {
    event.preventDefault()
    if (!edf) return
    upload.mutate(
      { edf, hypnogram, modelIds },
      { onSuccess: (recording) => navigate(`/recordings/${recording.id}`) },
    )
  }

  return (
    <section className="card" aria-labelledby="upload-title">
      <h2 id="upload-title" className="card-title">
        Upload recording
      </h2>
      <form onSubmit={submit} className="mt-4 space-y-4">
        <div>
          <label htmlFor="edf" className="block text-sm font-medium">
            PSG EDF
          </label>
          <p className="mb-1 text-xs text-slate-500">
            Sleep-EDF compatible: EEG Fpz-Cz and Pz-Oz at 100 Hz, up to 64 MiB
          </p>
          <input
            id="edf"
            type="file"
            accept=".edf"
            required
            className={FILE_INPUT}
            onChange={(event) => setEdf(event.target.files?.[0] ?? null)}
          />
        </div>
        <div>
          <label htmlFor="hypnogram" className="block text-sm font-medium">
            Hypnogram EDF
          </label>
          <p className="mb-1 text-xs text-slate-500">
            Optional; enables agreement metrics. It never changes the predictions.
          </p>
          <input
            id="hypnogram"
            type="file"
            accept=".edf"
            className={FILE_INPUT}
            onChange={(event) => setHypnogram(event.target.files?.[0] ?? null)}
          />
        </div>
        <fieldset className="max-w-md rounded-lg border border-slate-200 px-3 pb-2">
          <legend className="px-1 text-sm font-medium">Models</legend>
          <div className="flex flex-wrap gap-x-6 gap-y-1">
            {runnable.map((model) => (
              <label key={model.id} className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="size-4 accent-slate-900"
                  checked={!unchecked.has(model.id)}
                  onChange={() => toggle(model.id)}
                />
                {model.name}
              </label>
            ))}
            {!runnable.length && (
              <p className="text-sm text-slate-500">No model is installed on this server.</p>
            )}
          </div>
        </fieldset>
        <ErrorMessage error={upload.error} />
        <button type="submit" className="button button-primary" disabled={!edf || upload.isPending}>
          {upload.isPending ? 'Uploading…' : modelIds.length ? 'Upload and run' : 'Upload'}
        </button>
      </form>
    </section>
  )
}
