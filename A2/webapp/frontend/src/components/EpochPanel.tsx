import { useEffect } from 'react'

import type { Epoch } from '../api/client'
import { WaveformChart } from '../charts/WaveformChart'
import { EPOCH_SECONDS, STAGE_COLORS, STAGE_NAMES, formatClock } from '../lib/stages'
import type { ModelResult } from './Results'

interface Props {
  results: ModelResult[]
  reference: (number | null)[] | null
  nEpochs: number
  selected: number
  epoch: Epoch | undefined
  onSelect: (epochIndex: number) => void
  /** Epoch to jump to, or undefined when the two models never disagree. */
  nextDisagreement: number | undefined
}

function ProbabilityBars({ result, selected }: { result: ModelResult; selected: number }) {
  const row = result.predictions.probabilities[selected]
  return (
    <div data-testid="epoch-probabilities" data-model={result.model.id}>
      <h4 className="mb-1 flex items-center gap-2 text-sm font-medium">
        <span className="h-0.5 w-4 rounded" style={{ backgroundColor: result.color }} aria-hidden />
        {result.model.name}
      </h4>
      <ul className="space-y-1">
        {STAGE_NAMES.map((name, stage) => (
          <li key={name} className="flex items-center gap-2 text-xs tabular-nums">
            <span className="w-8 text-slate-600">{name}</span>
            <span className="h-2 flex-1 overflow-hidden rounded bg-slate-100">
              <span
                className="block h-full rounded"
                style={{ width: `${row[stage] * 100}%`, backgroundColor: STAGE_COLORS[stage] }}
              />
            </span>
            <span className="w-12 text-right">{row[stage].toFixed(3)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function EpochPanel(props: Props) {
  const { results, reference, nEpochs, selected, epoch, onSelect, nextDisagreement } = props

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null
      if (target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)) return
      if (event.key === 'ArrowLeft' && selected > 0) onSelect(selected - 1)
      if (event.key === 'ArrowRight' && selected < nEpochs - 1) onSelect(selected + 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selected, nEpochs, onSelect])

  const parts = [
    `Epoch ${selected}`,
    formatClock(selected * EPOCH_SECONDS),
    ...results.map(
      (result) => `${result.model.name} ${STAGE_NAMES[result.predictions.stages[selected]]}`,
    ),
  ]
  const referenceStage = reference?.[selected]
  if (reference) parts.push(`Reference ${referenceStage == null ? 'unscored' : STAGE_NAMES[referenceStage]}`)

  return (
    <section className="card" aria-labelledby="epoch-title">
      <div className="flex flex-wrap items-center gap-2">
        <h2 id="epoch-title" className="card-title mr-auto">
          Selected epoch
        </h2>
        <button
          type="button"
          className="button"
          disabled={selected === 0}
          onClick={() => onSelect(selected - 1)}
        >
          ← Previous
        </button>
        <button
          type="button"
          className="button"
          disabled={selected >= nEpochs - 1}
          onClick={() => onSelect(selected + 1)}
        >
          Next →
        </button>
        {results.length > 1 && (
          <button
            type="button"
            className="button"
            disabled={nextDisagreement === undefined}
            onClick={() => nextDisagreement !== undefined && onSelect(nextDisagreement)}
          >
            Next disagreement
          </button>
        )}
      </div>
      <p className="mt-2 text-sm font-medium" data-testid="epoch-summary" aria-live="polite">
        {parts.join(' · ')}
      </p>
      <p className="text-xs text-slate-500">
        Click a chart or use the ← → keys. Epoch {selected} covers{' '}
        {formatClock(selected * EPOCH_SECONDS)}–{formatClock((selected + 1) * EPOCH_SECONDS)} of{' '}
        {nEpochs} epochs.
      </p>
      <div className="mt-3 grid gap-6 lg:grid-cols-[2fr_1fr]">
        {epoch ? (
          <WaveformChart epoch={epoch} />
        ) : (
          <p className="h-72 text-sm text-slate-500">Loading waveform…</p>
        )}
        <div className="space-y-4">
          {results.map((result) => (
            <ProbabilityBars key={result.model.id} result={result} selected={selected} />
          ))}
        </div>
      </div>
    </section>
  )
}
