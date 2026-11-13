import { BarChart, LineChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  TooltipComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import type { EChartsCoreOption, EChartsType } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { useEffect, useRef } from 'react'

echarts.use([
  BarChart,
  LineChart,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkAreaComponent,
  TooltipComponent,
  CanvasRenderer,
])

interface Options {
  /** Charts sharing a group keep their zoom and hover position in sync. */
  group?: string
  /** Called with the x-axis value under a click anywhere inside the plot area. */
  onSelect?: (x: number) => void
}

/** Mount an ECharts instance in a div, keep it sized, and apply `option` whenever it changes. */
export function useEChart(option: EChartsCoreOption, { group, onSelect }: Options = {}) {
  const container = useRef<HTMLDivElement>(null)
  const chart = useRef<EChartsType | null>(null)
  const select = useRef(onSelect)

  useEffect(() => {
    select.current = onSelect
  }, [onSelect])

  useEffect(() => {
    const element = container.current
    if (!element) return
    const instance = echarts.init(element)
    chart.current = instance
    if (group) {
      instance.group = group
      echarts.connect(group)
    }
    instance.getZr().on('click', (event) => {
      const point = [event.offsetX, event.offsetY]
      if (!instance.containPixel({ gridIndex: 0 }, point)) return
      select.current?.(instance.convertFromPixel({ gridIndex: 0 }, point)[0])
    })
    const observer = new ResizeObserver(() => instance.resize())
    observer.observe(element)
    return () => {
      observer.disconnect()
      instance.dispose()
      chart.current = null
    }
  }, [group])

  useEffect(() => {
    // Replace the series but merge everything else, so the user's zoom survives updates.
    chart.current?.setOption(option, { replaceMerge: ['series'] })
  }, [option])

  return container
}
