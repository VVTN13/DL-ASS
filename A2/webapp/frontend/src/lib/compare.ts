/** Epoch-level comparisons between two stage sequences. Pure functions, no React. */

export function disagreements(a: number[], b: number[]): number[] {
  const indices: number[] = []
  for (let index = 0; index < Math.min(a.length, b.length); index++) {
    if (a[index] !== b[index]) indices.push(index)
  }
  return indices
}

/** Merge sorted epoch indices into half-open `[start, end)` runs. */
export function toRuns(indices: number[]): [number, number][] {
  const runs: [number, number][] = []
  for (const index of indices) {
    const last = runs[runs.length - 1]
    if (last && last[1] === index) last[1] = index + 1
    else runs.push([index, index + 1])
  }
  return runs
}

export function stageCounts(stages: (number | null)[]): number[] {
  const counts = [0, 0, 0, 0, 0]
  for (const stage of stages) if (stage != null) counts[stage] += 1
  return counts
}

/** The next index after `current` in a sorted list, wrapping to the first. */
export function nextAfter(sorted: number[], current: number): number | undefined {
  return sorted.find((index) => index > current) ?? sorted[0]
}
