import { exportUrl } from '../api/client'
import type { Job, Model, Recording } from '../api/client'
import { useCreateJob } from '../api/queries'
import { MODEL_COLORS } from '../lib/stages'
import { ErrorMessage } from './ErrorMessage'

function runtimeSeconds(job: Job): string {
  if (!job.started_at || !job.finished_at) return ''
  const seconds = (Date.parse(job.finished_at) - Date.parse(job.started_at)) / 1000
  return `${seconds.toFixed(1)} s`
}

interface RowProps {
  model: Model
  job: Job | undefined
  color: string
  pending: boolean
  onRun: () => void
}

function ModelRow({ model, job, color, pending, onRun }: RowProps) {
  const runButton = (label: string) => (
    <button type="button" className="button" disabled={pending || !model.available} onClick={onRun}>
      {label}
    </button>
  )
  return (
    <li
      className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-slate-100 py-3 last:border-0"
      data-testid="model-job"
      data-model={model.id}
      data-status={job?.status ?? 'none'}
    >
      <div className="flex w-36 items-center gap-2 font-medium">
        <span className="h-0.5 w-4 rounded" style={{ backgroundColor: color }} aria-hidden />
        {model.name}
      </div>
      {!job && (
        <>
          <span className="text-sm text-slate-500">
            {model.available ? 'Not run on this recording yet.' : 'Checkpoints not installed.'}
          </span>
          {runButton(`Run ${model.name}`)}
        </>
      )}
      {job && (job.status === 'queued' || job.status === 'running') && (
        <div className="flex flex-1 items-center gap-3 text-sm text-slate-600">
          <progress
            className="h-2 w-48 accent-sky-600"
            value={job.progress}
            max={1}
            aria-label={`${model.name} progress`}
          />
          {job.status === 'queued' ? 'Queued' : `${job.stage ?? 'running'}…`}
        </div>
      )}
      {job?.status === 'succeeded' && (
        <>
          <span className="text-sm text-emerald-700">Completed in {runtimeSeconds(job)}</span>
          <a className="button" href={exportUrl(job.id, 'csv')} download>
            Download CSV
          </a>
          <a className="button" href={exportUrl(job.id, 'json')} download>
            Download JSON
          </a>
        </>
      )}
      {job?.status === 'failed' && (
        <>
          <span className="text-sm text-red-700" role="alert" data-code={job.error_code}>
            Failed: {job.error_message}
          </span>
          {runButton('Run again')}
        </>
      )}
    </li>
  )
}

interface Props {
  recording: Recording
  models: Model[]
}

export function JobsPanel({ recording, models }: Props) {
  const createJob = useCreateJob(recording.id)
  // Jobs arrive oldest first; the newest one per model is the one that matters.
  const latest = new Map(recording.jobs.map((job) => [job.model_id, job]))
  return (
    <section className="card" aria-labelledby="models-title">
      <h2 id="models-title" className="card-title">
        Models
      </h2>
      <ErrorMessage error={createJob.error} />
      <ul className="mt-1">
        {models.map((model, index) => (
          <ModelRow
            key={model.id}
            model={model}
            job={latest.get(model.id)}
            color={MODEL_COLORS[index % MODEL_COLORS.length]}
            pending={createJob.isPending}
            onRun={() => createJob.mutate(model.id)}
          />
        ))}
      </ul>
    </section>
  )
}
