import { ATTNSLEEP, N_EPOCHS, TRANSFORMER, expect, test } from './fixtures'

const STAGE = '(W|N1|N2|N3|REM)'

test.describe('results viewer', () => {
  let recordingId: string

  // One recording with a hypnogram and both models, shared by the read-only tests below.
  test.beforeEach(async ({ api, recordingPage }) => {
    const recording = await api.createRecording({
      label: 'results',
      hypnogram: true,
      models: [TRANSFORMER, ATTNSLEEP],
    })
    recordingId = recording.id
    await api.waitForJobs(recordingId)
    await recordingPage.goto(recordingId)
    await expect(recordingPage.epochSummary).toBeVisible()
  })

  test('epoch navigation keeps the summary and the URL in sync', async ({ recordingPage, page }) => {
    const summary = recordingPage.epochSummary
    await expect(summary).toHaveText(
      new RegExp(`^Epoch 0 · 00:00:00 · Transformer ${STAGE} · AttnSleep ${STAGE} · Reference W$`),
    )
    await expect(recordingPage.previous).toBeDisabled()

    await recordingPage.next.click()
    await recordingPage.next.click()
    await expect(summary).toContainText('Epoch 2 · 00:01:00')
    await expect(page).toHaveURL(/[?&]epoch=2/)

    await page.keyboard.press('ArrowRight')
    await expect(summary).toContainText('Epoch 3 · 00:01:30')
    await expect(summary).toContainText('Reference N1') // from the uploaded hypnogram
    await page.keyboard.press('ArrowLeft')
    await recordingPage.previous.click()
    await expect(summary).toContainText('Epoch 1 · 00:00:30')

    await page.reload()
    await expect(summary).toContainText('Epoch 1 · 00:00:30')
  })

  test('a deep link opens the same epoch and model', async ({ recordingPage, page }) => {
    await recordingPage.goto(recordingId, `?epoch=${N_EPOCHS - 1}&model=${ATTNSLEEP}`)

    await expect(recordingPage.epochSummary).toContainText('Epoch 119 · 00:59:30')
    await expect(recordingPage.next).toBeDisabled()
    await expect(recordingPage.modelTab('AttnSleep')).toHaveAttribute('aria-selected', 'true')
    await expect(page.getByText('AttnSleep: reference (rows) vs predicted (columns)')).toBeVisible()

    await recordingPage.modelTab('Transformer').click()
    await expect(page).toHaveURL(new RegExp(`model=${TRANSFORMER}`))
    await expect(page.getByText('Transformer: per stage')).toBeVisible()

    // Out-of-range values fall back to a valid epoch instead of breaking the page.
    await recordingPage.goto(recordingId, '?epoch=9999')
    await expect(recordingPage.epochSummary).toContainText('Epoch 119')
  })

  test('clicking the hypnogram selects the epoch under the pointer', async ({ recordingPage }) => {
    await recordingPage.clickHypnogramAt(0.5)
    await expect.poll(() => recordingPage.epochIndex()).toBeGreaterThanOrEqual(58)
    expect(await recordingPage.epochIndex()).toBeLessThanOrEqual(61)

    await recordingPage.clickHypnogramAt(0.9)
    await expect.poll(() => recordingPage.epochIndex()).toBeGreaterThanOrEqual(106)
    await expect(recordingPage.waveform).toHaveAttribute(
      'aria-label',
      `Filtered EEG waveform of epoch ${await recordingPage.epochIndex()}`,
    )
  })

  test('the two models are compared epoch by epoch', async ({ recordingPage }) => {
    await expect(recordingPage.agreement).toContainText(
      /Transformer and AttnSleep agree on [\d.]+% of epochs \(\d+ of 120 differ\)/,
    )
    const differing = Number(/\((\d+) of/.exec(await recordingPage.agreement.innerText())?.[1])

    // Each source's stage durations add up to the one-hour recording.
    const rows = await recordingPage.stageDistribution.locator('tbody tr').allInnerTexts()
    expect(rows).toHaveLength(5)
    for (const column of [1, 2, 3]) {
      const minutes = rows.map((row) => Number.parseFloat(row.split('\t')[column]))
      expect(minutes.reduce((sum, value) => sum + value, 0)).toBeCloseTo(60, 5)
    }

    if (differing === 0) {
      await expect(recordingPage.nextDisagreement).toBeDisabled()
      return
    }
    await recordingPage.nextDisagreement.click()
    const summary = await recordingPage.epochSummary.innerText()
    const [, transformer, attnsleep] =
      new RegExp(`Transformer ${STAGE} · AttnSleep ${STAGE}`).exec(summary) ?? []
    expect(transformer).toBeTruthy()
    expect(transformer).not.toBe(attnsleep)
  })

  test('metrics against the hypnogram follow the selected scope', async ({ recordingPage }) => {
    const table = recordingPage.metricsTable
    for (const model of [TRANSFORMER, ATTNSLEEP]) {
      const row = table.locator(`tr[data-model="${model}"]`)
      await expect(row).toContainText('120 of 120')
      await expect(row).toContainText(/\d+\.\d%/)
    }
    await expect(recordingPage.scope('All scored epochs')).toHaveAttribute('aria-checked', 'true')

    const cells = await recordingPage.confusionMatrix.locator('tbody td').allInnerTexts()
    expect(cells).toHaveLength(25)
    expect(cells.reduce((sum, cell) => sum + Number(cell), 0)).toBe(N_EPOCHS)

    await recordingPage.scope('A2 benchmark window').click()
    await expect(recordingPage.scope('A2 benchmark window')).toHaveAttribute('aria-checked', 'true')
    await expect(recordingPage.page.getByText('within 30 minutes of scored sleep')).toBeVisible()
    await expect(table.locator(`tr[data-model="${TRANSFORMER}"]`)).toContainText('120 of 120')
  })
})
