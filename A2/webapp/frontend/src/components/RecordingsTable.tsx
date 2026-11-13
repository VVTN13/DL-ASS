import { Link } from 'react-router-dom'

import type { Model, Recording } from '../api/client'
import { useDeleteRecording } from '../api/queries'
import { formatDuration } from '../lib/stages'
import { ErrorMessage } from './ErrorMessage'
import { JobBadge } from './JobBadge'

interface Props {
  recordings: Recording[]
  models: Model[]
}

export function RecordingsTable({ recordings, models }: Props) {
  const remove = useDeleteRecording()
  const names = new Map(models.map((model) => [model.id, model.name]))

  function confirmDelete(recording: Recording) {
    if (window.confirm(`Delete ${recording.filename} and its results?`)) remove.mutate(recording.id)
  }

  return (
    <section className="card" aria-labelledby="recordings-title">
      <h2 id="recordings-title" className="card-title">
        Recordings
      </h2>
      <ErrorMessage error={remove.error} />
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200">
              {['File', 'Duration', 'Epochs', 'Hypnogram', 'Jobs', ''].map((heading) => (
                <th key={heading} scope="col" className="label px-2 py-2">
                  {heading}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {recordings.map((recording) => (
              <tr key={recording.id} className="border-b border-slate-100" data-testid="recording-row">
                <td className="px-2 py-2">
                  <Link
                    to={`/recordings/${recording.id}`}
                    className="font-medium text-sky-700 hover:underline"
                  >
                    {recording.filename}
                  </Link>
                  {recording.is_sample && (
                    <span className="ml-2 rounded-full bg-violet-100 px-2 py-0.5 text-xs font-medium text-violet-800">
                      sample
                    </span>
                  )}
                </td>
                <td className="px-2 py-2 tabular-nums">{formatDuration(recording.duration_seconds)}</td>
                <td className="px-2 py-2 tabular-nums">{recording.n_epochs}</td>
                <td className="px-2 py-2">{recording.has_hypnogram ? 'yes' : '–'}</td>
                <td className="px-2 py-2">
                  <div className="flex flex-wrap gap-1">
                    {recording.jobs.map((job) => (
                      <JobBadge key={job.id} job={job} name={names.get(job.model_id) ?? job.model_id} />
                    ))}
                    {!recording.jobs.length && <span className="text-slate-400">not run yet</span>}
                  </div>
                </td>
                <td className="px-2 py-2 text-right">
                  {!recording.is_sample && (
                    <button
                      type="button"
                      className="text-xs font-medium text-slate-500 hover:text-red-700"
                      aria-label={`Delete ${recording.filename}`}
                      disabled={remove.isPending}
                      onClick={() => confirmDelete(recording)}
                    >
                      Delete
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!recordings.length && <p className="px-2 py-3 text-sm text-slate-500">No recordings yet.</p>}
      </div>
    </section>
  )
}
