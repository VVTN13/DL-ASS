import { useMemo } from 'react'

import { EPOCH_SECONDS, STAGE_COLORS, STAGE_NAMES, formatClock } from '../lib/stages'
import { NIGHT_GROUP } from './HypnogramChart'
import { useEChart } from './useEChart'

interface Props {
  probabilities: number[][]
  onSelect: (epochIndex: number) => void
}

/** Stacked class probabilities across the night, zoomed together with the hypnogram. */
export function ProbabilityChart({ probabilities, onSelect }: Props) {
  const nEpochs = probabilities.length
  const option = useMemo(
    () => ({
      animation: false,
      grid: { left: 48, right: 16, top: 36, bottom: 28 },
      legend: { top: 0, left: 40, icon: 'roundRect', itemWidth: 14, itemHeight: 8 },
      tooltip: {
        trigger: 'axis',
        formatter: (rows: { axisValue: number; marker: string; seriesName: string; data: number[] }[]) => {
          const epoch = Math.min(nEpochs - 1, Math.floor(rows[0].axisValue))
          const lines = rows.map(
            ({ marker, seriesName, data }) => `${marker}${seriesName}: <b>${data[1].toFixed(3)}</b>`,
          )
          return [`Epoch ${epoch} · ${formatClock(epoch * EPOCH_SECONDS)}`, ...lines].join('<br/>')
        },
      },
      xAxis: {
        type: 'value',
        min: 0,
        max: nEpochs,
        minInterval: 1,
        axisLabel: {
          formatter: (epoch: number) => formatClock(epoch * EPOCH_SECONDS).slice(0, 5),
        },
        splitLine: { show: false },
      },
      yAxis: { type: 'value', min: 0, max: 1, interval: 0.5 },
      // No controls of its own: this zoom only follows the hypnogram through the chart group.
      dataZoom: [
        { type: 'inside', zoomOnMouseWheel: false, moveOnMouseMove: false, moveOnMouseWheel: false },
      ],
      series: STAGE_NAMES.map((name, stage) => ({
        name,
        type: 'line',
        stack: 'probability',
        step: 'end',
        showSymbol: false,
        lineStyle: { width: 0 },
        areaStyle: { opacity: 0.9 },
        itemStyle: { color: STAGE_COLORS[stage] },
        emphasis: { disabled: true },
        data: probabilities.map((row, epoch) => [epoch, row[stage]]),
      })),
    }),
    [probabilities, nEpochs],
  )
  const container = useEChart(option, {
    group: NIGHT_GROUP,
    onSelect: (x) => onSelect(Math.max(0, Math.min(nEpochs - 1, Math.floor(x)))),
  })
  return (
    <div
      ref={container}
      className="h-44 w-full"
      role="img"
      aria-label="Predicted probability of each sleep stage across the recording"
      data-testid="probability-chart"
    />
  )
}
