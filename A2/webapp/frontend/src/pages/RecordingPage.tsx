import { useMemo } from 'react'
import { Link, useParams } from 'react-router-dom'

import { ApiError } from '../api/client'
import { useModels, useRecording } from '../api/queries'
import { ErrorMessage } from '../components/ErrorMessage'
import { JobsPanel } from '../components/JobsPanel'
import { RecordingSummary } from '../components/RecordingSummary'
import { Results } from '../components/Results'
import { MODEL_COLORS } from '../lib/stages'
import { NotFoundPage } from './NotFoundPage'

export function RecordingPage() {
  const { recordingId = '' } = useParams()
  const recording = useRecording(recordingId)
  const models = useModels()

  const jobs = recording.data?.jobs
  const finished = useMemo(
    () =>
      (models.data ?? []).flatMap((model, index) => {
        const job = jobs?.find(
          (candidate) => candidate.model_id === model.id && candidate.status === 'succeeded',
        )
        return job ? [{ job, model, color: MODEL_COLORS[index % MODEL_COLORS.length] }] : []
      }),
    [models.data, jobs],
  )

  const error = recording.error ?? models.error
  if (error instanceof ApiError && (error.status === 404 || error.status === 422)) {
    return <NotFoundPage message="This recording does not exist or has been deleted." />
  }
  if (error) return <ErrorMessage error={error} />
  if (!recording.data || !models.data) return <p className="text-sm text-slate-500">Loading…</p>

  return (
    <>
      <div>
        <Link to="/" className="text-sm text-sky-700 hover:underline">
          ← All recordings
        </Link>
        <h1 className="mt-1 flex flex-wrap items-center gap-2 text-xl font-semibold">
          {recording.data.filename}
          {recording.data.is_sample && (
            <span className="rounded-full bg-violet-100 px-2 py-0.5 text-xs font-medium text-violet-800">
              sample
            </span>
          )}
        </h1>
      </div>
      <RecordingSummary recording={recording.data} />
      <JobsPanel recording={recording.data} models={models.data} />
      {finished.length > 0 ? (
        <Results recording={recording.data} finished={finished} />
      ) : (
        <p className="card text-sm text-slate-600" data-testid="no-results">
          Results appear here as soon as a model has finished.
        </p>
      )}
    </>
  )
}
