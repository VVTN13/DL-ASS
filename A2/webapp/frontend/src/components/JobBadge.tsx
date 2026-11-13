import type { Job } from '../api/client'

const STYLES: Record<Job['status'], string> = {
  queued: 'bg-slate-100 text-slate-700',
  running: 'bg-sky-100 text-sky-800',
  succeeded: 'bg-emerald-100 text-emerald-800',
  failed: 'bg-red-100 text-red-800',
}

export function JobBadge({ job, name }: { job: Job; name: string }) {
  const detail =
    job.status === 'running'
      ? `${job.stage ?? 'running'} ${Math.round(job.progress * 100)}%`
      : job.status === 'failed'
        ? `failed · ${job.error_code}`
        : job.status
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${STYLES[job.status]}`}
      data-testid="job-badge"
      data-status={job.status}
      data-model={job.model_id}
    >
      {name}: {detail}
    </span>
  )
}
