import { expect } from '@playwright/test'
import type { Locator, Page } from '@playwright/test'

export class RecordingPage {
  readonly page: Page
  readonly summary: Locator
  readonly epochSummary: Locator
  readonly hypnogram: Locator
  readonly waveform: Locator
  readonly agreement: Locator
  readonly stageDistribution: Locator
  readonly metricsTable: Locator
  readonly confusionMatrix: Locator
  readonly previous: Locator
  readonly next: Locator
  readonly nextDisagreement: Locator

  constructor(page: Page) {
    this.page = page
    this.summary = page.getByTestId('recording-summary')
    this.epochSummary = page.getByTestId('epoch-summary')
    this.hypnogram = page.getByTestId('hypnogram-chart')
    this.waveform = page.getByTestId('waveform-chart')
    this.agreement = page.getByTestId('agreement')
    this.stageDistribution = page.getByTestId('stage-distribution')
    this.metricsTable = page.getByTestId('metrics-table')
    this.confusionMatrix = page.getByTestId('confusion-matrix')
    this.previous = page.getByRole('button', { name: '← Previous' })
    this.next = page.getByRole('button', { name: 'Next →' })
    this.nextDisagreement = page.getByRole('button', { name: 'Next disagreement' })
  }

  async goto(recordingId: string, query = '') {
    await this.page.goto(`/recordings/${recordingId}${query}`)
    await this.page.getByRole('heading', { name: 'Models', exact: true }).waitFor()
  }

  recordingId(): string {
    return new URL(this.page.url()).pathname.split('/').pop() ?? ''
  }

  modelJob(name: string): Locator {
    return this.page.getByTestId('model-job').filter({ hasText: name })
  }

  async run(name: string) {
    await this.page.getByRole('button', { name: `Run ${name}` }).click()
  }

  async expectFinished(name: string) {
    await expect(this.modelJob(name)).toHaveAttribute('data-status', 'succeeded', {
      timeout: 150_000,
    })
  }

  modelTab(name: string): Locator {
    return this.page.getByRole('tab', { name })
  }

  scope(name: string): Locator {
    return this.page.getByRole('radio', { name })
  }

  /** Click the hypnogram at a horizontal fraction of its plot area (0 = start, 1 = end). */
  async clickHypnogramAt(fraction: number) {
    const box = await this.hypnogram.boundingBox()
    if (!box) throw new Error('The hypnogram is not visible.')
    // Plot area insets match the chart's grid: 48 px left, 16 px right.
    // A positioned locator click scrolls the chart into view first; a raw mouse click would not.
    await this.hypnogram.click({
      position: { x: 48 + (box.width - 64) * fraction, y: box.height / 2 },
    })
  }

  async epochIndex(): Promise<number> {
    const match = /^Epoch (\d+)/.exec(await this.epochSummary.innerText())
    if (!match) throw new Error('No epoch is selected.')
    return Number(match[1])
  }
}
