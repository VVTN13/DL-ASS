import { useState } from 'react'

import type { Metrics, MetricsScope } from '../api/client'
import { useMetrics } from '../api/queries'
import { formatPercent, formatScore } from '../lib/stages'
import { ErrorMessage } from './ErrorMessage'
import type { ModelResult } from './Results'

const SCOPES: { value: MetricsScope; label: string; help: string }[] = [
  { value: 'all', label: 'All scored epochs', help: 'Every epoch the hypnogram labels.' },
  {
    value: 'benchmark',
    label: 'A2 benchmark window',
    help: 'Only epochs within 30 minutes of scored sleep, as in the A2 report.',
  },
]

function ConfusionMatrix({ metrics, name }: { metrics: Metrics; name: string }) {
  return (
    <table className="text-center text-xs tabular-nums" data-testid="confusion-matrix">
      <caption className="mb-1 text-left text-sm font-medium">
        {name}: reference (rows) vs predicted (columns)
      </caption>
      <thead>
        <tr>
          <td />
          {metrics.class_names.map((label) => (
            <th key={label} scope="col" className="px-2 py-1 font-medium text-slate-500">
              {label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {metrics.confusion_matrix.map((row, reference) => {
          const support = row.reduce((sum, count) => sum + count, 0)
          return (
            <tr key={metrics.class_names[reference]}>
              <th scope="row" className="px-2 py-1 text-right font-medium text-slate-500">
                {metrics.class_names[reference]}
              </th>
              {row.map((count, predicted) => {
                const share = support ? count / support : 0
                return (
                  <td
                    key={predicted}
                    className="size-11 border border-white"
                    style={{
                      backgroundColor: `rgba(0, 114, 178, ${share})`,
                      color: share > 0.55 ? 'white' : undefined,
                    }}
                  >
                    {count}
                  </td>
                )
              })}
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

interface Props {
  results: ModelResult[]
  active: ModelResult
}

/** Agreement with the uploaded hypnogram. Never shown for recordings without labels. */
export function MetricsPanel({ results, active }: Props) {
  const [scope, setScope] = useState<MetricsScope>('all')
  const queries = useMetrics(results.map((result) => result.job), scope, true)
  const rows = results.map((result, index) => ({ result, metrics: queries[index].data }))
  const activeMetrics = rows.find((row) => row.result === active)?.metrics
  const failure = queries.find((query) => query.error)?.error

  return (
    <section className="card" aria-labelledby="metrics-title">
      <div className="flex flex-wrap items-center gap-3">
        <h2 id="metrics-title" className="card-title mr-auto">
          Agreement with the uploaded hypnogram
        </h2>
        <div role="radiogroup" aria-label="Epochs to score" className="flex gap-1">
          {SCOPES.map((option) => (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={scope === option.value}
              title={option.help}
              className={`button ${scope === option.value ? 'button-primary' : ''}`}
              onClick={() => setScope(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
      <p className="mt-1 text-xs text-slate-500">
        Measured on this recording only. {SCOPES.find((option) => option.value === scope)?.help}
      </p>
      <ErrorMessage error={failure} />
      <table className="mt-3 w-full text-left text-sm" data-testid="metrics-table">
        <thead>
          <tr className="border-b border-slate-200">
            {['Model', 'Accuracy', 'Balanced acc.', 'Macro-F1', 'Kappa', 'Scored epochs'].map(
              (heading, index) => (
                <th key={heading} scope="col" className={`label py-1.5 ${index ? 'text-right' : ''}`}>
                  {heading}
                </th>
              ),
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ result, metrics }) => (
            <tr key={result.model.id} className="border-b border-slate-100 tabular-nums" data-model={result.model.id}>
              <th scope="row" className="py-1.5 font-medium">
                {result.model.name}
              </th>
              <td className="py-1.5 text-right">{formatPercent(metrics?.accuracy)}</td>
              <td className="py-1.5 text-right">{formatPercent(metrics?.balanced_accuracy)}</td>
              <td className="py-1.5 text-right">{formatScore(metrics?.macro_f1)}</td>
              <td className="py-1.5 text-right">{formatScore(metrics?.kappa)}</td>
              <td className="py-1.5 text-right">
                {metrics ? `${metrics.n_scored_epochs} of ${metrics.n_epochs}` : '–'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {activeMetrics && (
        <div className="mt-5 flex flex-wrap gap-10">
          <ConfusionMatrix metrics={activeMetrics} name={active.model.name} />
          <table className="self-start text-sm tabular-nums" data-testid="per-class-metrics">
            <caption className="mb-1 text-left text-sm font-medium">
              {active.model.name}: per stage
            </caption>
            <thead>
              <tr className="border-b border-slate-200">
                {['Stage', 'Precision', 'Recall', 'F1', 'Epochs'].map((heading, index) => (
                  <th key={heading} scope="col" className={`label px-2 py-1 ${index ? 'text-right' : ''}`}>
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {activeMetrics.per_class.map((row) => (
                <tr key={row.name} className="border-b border-slate-100">
                  <th scope="row" className="px-2 py-1 text-left font-normal">
                    {row.name}
                  </th>
                  <td className="px-2 py-1 text-right">{formatScore(row.precision)}</td>
                  <td className="px-2 py-1 text-right">{formatScore(row.recall)}</td>
                  <td className="px-2 py-1 text-right">{formatScore(row.f1)}</td>
                  <td className="px-2 py-1 text-right">{row.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
