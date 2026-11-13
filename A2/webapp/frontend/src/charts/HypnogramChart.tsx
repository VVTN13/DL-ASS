import { useMemo } from 'react'

import { EPOCH_SECONDS, HYPNOGRAM_ROWS, formatClock, hypnogramRow } from '../lib/stages'
import { useEChart } from './useEChart'

export interface HypnogramTrack {
  name: string
  color: string
  stages: (number | null)[]
  dashed?: boolean
}

interface Props {
  tracks: HypnogramTrack[]
  nEpochs: number
  selected: number
  /** Half-open epoch ranges to shade, for example where two models disagree. */
  highlighted?: [number, number][]
  onSelect: (epochIndex: number) => void
}

const SELECTED_STYLE = { color: 'rgba(15, 23, 42, 0.18)' }
const HIGHLIGHT_STYLE = { color: 'rgba(220, 38, 38, 0.10)' }

export const NIGHT_GROUP = 'night'

interface TooltipRow {
  axisValue: number
  marker: string
  seriesName: string
  data: [number, number | null]
}

export function HypnogramChart({ tracks, nEpochs, selected, highlighted = [], onSelect }: Props) {
  const option = useMemo(() => {
    const areas = [
      ...highlighted.map(([start, end]) => [
        { xAxis: start, itemStyle: HIGHLIGHT_STYLE },
        { xAxis: end },
      ]),
      [{ xAxis: selected, itemStyle: SELECTED_STYLE }, { xAxis: selected + 1 }],
    ]
    return {
      animation: false,
      grid: { left: 48, right: 16, top: 36, bottom: 64 },
      legend: { top: 0, left: 40, icon: 'roundRect', itemWidth: 14, itemHeight: 3 },
      tooltip: {
        trigger: 'axis',
        formatter: (rows: TooltipRow[]) => {
          const epoch = Math.min(nEpochs - 1, Math.floor(rows[0].axisValue))
          const lines = rows.map(({ marker, seriesName, data }) => {
            const stage = data[1] == null ? '–' : HYPNOGRAM_ROWS[data[1]]
            return `${marker}${seriesName}: <b>${stage}</b>`
          })
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
      yAxis: {
        type: 'value',
        min: 0,
        max: HYPNOGRAM_ROWS.length - 1,
        interval: 1,
        axisLabel: { formatter: (row: number) => HYPNOGRAM_ROWS[row] },
      },
      dataZoom: [
        { type: 'inside', zoomOnMouseWheel: 'ctrl', moveOnMouseMove: false, moveOnMouseWheel: false },
        { type: 'slider', height: 18, bottom: 12, labelFormatter: '' },
      ],
      series: tracks.map((track, index) => {
        const points: [number, number | null][] = track.stages.map((stage, epoch) => [
          epoch,
          stage == null ? null : hypnogramRow(stage),
        ])
        // A step line holds each value until the next point: close the final epoch.
        if (points.length) points.push([points.length, points[points.length - 1][1]])
        return {
          name: track.name,
          type: 'line',
          step: 'end',
          showSymbol: false,
          data: points,
          lineStyle: { width: 1.6, color: track.color, type: track.dashed ? 'dashed' : 'solid' },
          itemStyle: { color: track.color },
          emphasis: { disabled: true },
          markArea: index === 0 ? { silent: true, data: areas } : undefined,
        }
      }),
    }
  }, [tracks, nEpochs, selected, highlighted])

  const container = useEChart(option, {
    group: NIGHT_GROUP,
    onSelect: (x) => onSelect(Math.max(0, Math.min(nEpochs - 1, Math.floor(x)))),
  })
  return (
    <div
      ref={container}
      className="h-72 w-full"
      role="img"
      aria-label="Hypnogram: predicted sleep stage for every 30-second epoch"
      data-testid="hypnogram-chart"
    />
  )
}
