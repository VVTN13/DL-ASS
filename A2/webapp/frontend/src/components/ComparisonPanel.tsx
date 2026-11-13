import { stageCounts } from '../lib/compare'
import { EPOCH_SECONDS, STAGE_LABELS, formatPercent } from '../lib/stages'
import type { ModelResult } from './Results'

interface Props {
  results: ModelResult[]
  reference: (number | null)[] | null
  /** Epochs where the first two models disagree. */
  disagreeing: number[]
}

/** Stage distribution per source, plus how often the two models agree epoch by epoch. */
export function ComparisonPanel({ results, reference, disagreeing }: Props) {
  const nEpochs = results[0].predictions.stages.length
  const columns = results.map((result) => ({
    name: result.model.name,
    counts: stageCounts(result.predictions.stages),
  }))
  if (reference) columns.push({ name: 'Reference', counts: stageCounts(reference) })

  return (
    <section className="card" aria-labelledby="comparison-title">
      <h2 id="comparison-title" className="card-title">
        {results.length > 1 ? 'Model comparison' : 'Stage distribution'}
      </h2>
      {results.length > 1 && (
        <p className="mt-2 text-sm" data-testid="agreement">
          {results[0].model.name} and {results[1].model.name} agree on{' '}
          <strong>{formatPercent(1 - disagreeing.length / nEpochs)}</strong> of epochs (
          {disagreeing.length} of {nEpochs} differ). Disagreements are shaded red in the hypnogram.
        </p>
      )}
      <table className="mt-3 w-full max-w-xl text-left text-sm" data-testid="stage-distribution">
        <thead>
          <tr className="border-b border-slate-200">
            <th scope="col" className="label py-1.5">
              Stage
            </th>
            {columns.map((column) => (
              <th key={column.name} scope="col" className="label py-1.5 text-right">
                {column.name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {STAGE_LABELS.map((label, stage) => (
            <tr key={label} className="border-b border-slate-100">
              <th scope="row" className="py-1.5 font-normal">
                {label}
              </th>
              {columns.map((column) => (
                <td key={column.name} className="py-1.5 text-right tabular-nums">
                  {((column.counts[stage] * EPOCH_SECONDS) / 60).toFixed(1)} min
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
