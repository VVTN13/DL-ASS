/** Class order used by the models and the API: index 0..4. */
export const STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'REM'] as const
export const STAGE_LABELS = ['Wake', 'N1', 'N2', 'N3', 'REM'] as const
export const STAGE_COLORS = ['#e69f00', '#56b4e9', '#0072b2', '#0b2545', '#cc79a7'] as const
export const EPOCH_SECONDS = 30

/** Hypnogram rows from bottom to top: deep sleep at the bottom, wake at the top. */
export const HYPNOGRAM_ROWS = ['N3', 'N2', 'N1', 'REM', 'W'] as const
const ROW_OF_STAGE = [4, 2, 1, 0, 3]

export function hypnogramRow(stage: number): number {
  return ROW_OF_STAGE[stage]
}

export const MODEL_COLORS = ['#2563eb', '#dc2626', '#059669', '#7c3aed']
export const REFERENCE_COLOR = '#0f172a'

/** Seconds since the start of the recording as `HH:MM:SS`. */
export function formatClock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds))
  const parts = [Math.floor(total / 3600), Math.floor((total % 3600) / 60), total % 60]
  return parts.map((part) => String(part).padStart(2, '0')).join(':')
}

export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  return hours ? `${hours} h ${String(minutes).padStart(2, '0')} min` : `${minutes} min`
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  return value == null ? '–' : `${(value * 100).toFixed(digits)}%`
}

export function formatScore(value: number | null | undefined): string {
  return value == null ? '–' : value.toFixed(3)
}
