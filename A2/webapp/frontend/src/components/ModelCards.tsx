import type { Model } from '../api/client'
import { MODEL_COLORS, formatScore } from '../lib/stages'

function Metric({ label, value }: { label: string; value: number | undefined }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className="text-xl font-semibold tabular-nums">{formatScore(value)}</dd>
    </div>
  )
}

function ModelCard({ model, color }: { model: Model; color: string }) {
  const metrics = model.published_metrics as Record<string, number> | null
  const channels = model.input_channels.join(', ')
  return (
    <article className="card" data-testid="model-card" data-model={model.id}>
      <h2 className="card-title flex items-center gap-2">
        <span className="h-0.5 w-4 rounded" style={{ backgroundColor: color }} aria-hidden />
        {model.name}
      </h2>
      <p className="mt-1 text-sm text-slate-600">5-fold soft voting · input {channels}</p>
      {model.synthetic ? (
        <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-900">
          Synthetic weights (test build): predictions are not meaningful.
        </p>
      ) : metrics ? (
        <>
          <dl className="mt-3 flex gap-8">
            <Metric label="Test accuracy" value={metrics.accuracy} />
            <Metric label="Test macro-F1" value={metrics.macro_f1} />
            <Metric label="Test kappa" value={metrics.kappa} />
          </dl>
          <p className="mt-2 text-xs text-slate-500">
            Reported on the A2 held-out test participants, not on your recording.
          </p>
        </>
      ) : null}
      {!model.available && (
        <p className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-800">
          Checkpoints are not installed on this server, so this model cannot run.
        </p>
      )}
    </article>
  )
}

export function ModelCards({ models }: { models: Model[] }) {
  return (
    <section className="grid gap-5 md:grid-cols-2" aria-label="Models">
      {models.map((model, index) => (
        <ModelCard key={model.id} model={model} color={MODEL_COLORS[index % MODEL_COLORS.length]} />
      ))}
    </section>
  )
}
