import { useCallback, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'

import type { Job, Model, Predictions, Recording } from '../api/client'
import { useEpoch, usePredictions } from '../api/queries'
import { HypnogramChart } from '../charts/HypnogramChart'
import type { HypnogramTrack } from '../charts/HypnogramChart'
import { ProbabilityChart } from '../charts/ProbabilityChart'
import { disagreements, nextAfter, toRuns } from '../lib/compare'
import { REFERENCE_COLOR } from '../lib/stages'
import { ComparisonPanel } from './ComparisonPanel'
import { EpochPanel } from './EpochPanel'
import { ErrorMessage } from './ErrorMessage'
import { MetricsPanel } from './MetricsPanel'

/** One finished job with everything needed to draw it. */
export interface ModelResult {
  job: Job
  model: Model
  color: string
  predictions: Predictions
}

interface Props {
  recording: Recording
  /** Finished jobs with their model and colour, in display order. */
  finished: Omit<ModelResult, 'predictions'>[]
}

/**
 * Everything shown once at least one model has finished. The selected epoch and model live
 * in the URL (`?epoch=120&model=transformer-a2-v1`), so a view can be shared and reloaded.
 */
export function Results({ recording, finished }: Props) {
  const [params, setParams] = useSearchParams()
  const jobs = useMemo(() => finished.map((entry) => entry.job), [finished])
  const { data: loaded, error: failure } = usePredictions(jobs)
  const results = useMemo(
    () =>
      finished.flatMap((entry, index) => {
        const predictions = loaded[index]
        return predictions ? [{ ...entry, predictions }] : []
      }),
    [finished, loaded],
  )

  const nEpochs = recording.n_epochs
  const requested = Number.parseInt(params.get('epoch') ?? '0', 10)
  const selected = Number.isFinite(requested) ? Math.max(0, Math.min(nEpochs - 1, requested)) : 0
  const active = results.find((result) => result.model.id === params.get('model')) ?? results[0]

  const setParam = useCallback(
    (name: string, value: string) =>
      setParams(
        (current) => {
          const next = new URLSearchParams(current)
          next.set(name, value)
          return next
        },
        { replace: true },
      ),
    [setParams],
  )
  const selectEpoch = useCallback((epoch: number) => setParam('epoch', String(epoch)), [setParam])

  const reference = results[0]?.predictions.reference_stages ?? null
  const disagreeing = useMemo(
    () =>
      results.length > 1
        ? disagreements(results[0].predictions.stages, results[1].predictions.stages)
        : [],
    [results],
  )
  const highlighted = useMemo(() => toRuns(disagreeing), [disagreeing])
  const tracks = useMemo(() => {
    const lines: HypnogramTrack[] = results.map((result) => ({
      name: result.model.name,
      color: result.color,
      stages: result.predictions.stages,
    }))
    if (reference) {
      lines.push({ name: 'Reference', color: REFERENCE_COLOR, stages: reference, dashed: true })
    }
    return lines
  }, [results, reference])

  const epoch = useEpoch(recording.id, selected, recording.waveform_ready)

  if (failure) return <ErrorMessage error={failure} />
  if (!active) return <p className="text-sm text-slate-500">Loading predictions…</p>

  return (
    <>
      <section className="card" aria-labelledby="hypnogram-title">
        <h2 id="hypnogram-title" className="card-title">
          Hypnogram
        </h2>
        <p className="text-xs text-slate-500">
          Predicted stage for each 30 s epoch from the start of the file. Drag the slider to zoom.
        </p>
        <HypnogramChart
          tracks={tracks}
          nEpochs={nEpochs}
          selected={selected}
          highlighted={highlighted}
          onSelect={selectEpoch}
        />
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <h3 className="mr-auto text-sm font-semibold">Stage probabilities</h3>
          {results.length > 1 && (
            <div role="tablist" aria-label="Model shown in detail" className="flex gap-1">
              {results.map((result) => (
                <button
                  key={result.model.id}
                  type="button"
                  role="tab"
                  aria-selected={result === active}
                  className={`button ${result === active ? 'button-primary' : ''}`}
                  onClick={() => setParam('model', result.model.id)}
                >
                  {result.model.name}
                </button>
              ))}
            </div>
          )}
        </div>
        <ProbabilityChart probabilities={active.predictions.probabilities} onSelect={selectEpoch} />
      </section>

      <EpochPanel
        results={results}
        reference={reference}
        nEpochs={nEpochs}
        selected={selected}
        epoch={epoch.data}
        onSelect={selectEpoch}
        nextDisagreement={nextAfter(disagreeing, selected)}
      />
      <ComparisonPanel results={results} reference={reference} disagreeing={disagreeing} />
      {recording.has_hypnogram && <MetricsPanel results={results} active={active} />}
    </>
  )
}
