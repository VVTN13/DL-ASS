import { useMemo } from 'react'

import type { Epoch } from '../api/client'
import { useEChart } from './useEChart'

const COLORS = ['#0f766e', '#7c3aed']

/** Symmetric y-range in steps of 50 µV, so neighbouring epochs are easy to compare. */
function amplitudeRange(samples: number[]): number {
  let peak = 0
  for (const value of samples) peak = Math.max(peak, Math.abs(value))
  return Math.max(50, Math.ceil(peak / 50) * 50)
}

export function WaveformChart({ epoch }: { epoch: Epoch }) {
  const option = useMemo(() => {
    const tops = ['8%', '54%']
    return {
      animation: false,
      tooltip: { trigger: 'axis', valueFormatter: (value: number) => `${value.toFixed(1)} µV` },
      grid: tops.map((top) => ({ left: 56, right: 16, top, height: '36%' })),
      xAxis: epoch.channels.map((_, index) => ({
        gridIndex: index,
        type: 'value',
        min: 0,
        max: epoch.channels[index].samples.length / epoch.sampling_rate,
        interval: 5,
        axisLabel: { show: index === epoch.channels.length - 1, formatter: '{value} s' },
      })),
      yAxis: epoch.channels.map((channel, index) => {
        const range = amplitudeRange(channel.samples)
        return {
          gridIndex: index,
          type: 'value',
          min: -range,
          max: range,
          interval: range,
          name: `${channel.label.replace('EEG ', '')} (µV)`,
          nameLocation: 'middle',
          nameGap: 40,
        }
      }),
      series: epoch.channels.map((channel, index) => ({
        name: channel.label,
        type: 'line',
        xAxisIndex: index,
        yAxisIndex: index,
        showSymbol: false,
        sampling: 'lttb',
        lineStyle: { width: 1, color: COLORS[index % COLORS.length] },
        itemStyle: { color: COLORS[index % COLORS.length] },
        data: channel.samples.map((value, sample) => [sample / epoch.sampling_rate, value]),
      })),
    }
  }, [epoch])
  const container = useEChart(option)
  return (
    <div
      ref={container}
      className="h-72 w-full"
      role="img"
      aria-label={`Filtered EEG waveform of epoch ${epoch.epoch_index}`}
      data-testid="waveform-chart"
    />
  )
}
