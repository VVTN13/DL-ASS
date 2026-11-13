import { readFileSync } from 'node:fs'

import { N_EPOCHS, expect, hypnogramFile, psgWithWrongChannel, test, uniquePsg } from './fixtures'

test.describe('upload', () => {
  test('upload, run one model, inspect and download the result', async ({
    homePage,
    recordingPage,
    api,
    page,
  }) => {
    const file = uniquePsg('upload-run')
    await homePage.goto()
    await homePage.upload(file, { models: ['Transformer'] })

    await expect(page).toHaveURL(/\/recordings\/[0-9a-f-]{36}$/)
    api.track(recordingPage.recordingId())
    await expect(page.getByRole('heading', { name: file.name })).toBeVisible()
    await expect(recordingPage.summary).toContainText('EEG Fpz-Cz (100 Hz, uV)')
    await expect(recordingPage.summary).toContainText('1 h 00 min')
    await expect(recordingPage.summary).toContainText(String(N_EPOCHS))

    await recordingPage.expectFinished('Transformer')
    await expect(recordingPage.modelJob('AttnSleep')).toHaveAttribute('data-status', 'none')
    await expect(recordingPage.hypnogram).toBeVisible()
    await expect(recordingPage.waveform).toBeVisible()
    await expect(recordingPage.epochSummary).toHaveText(
      /^Epoch 0 · 00:00:00 · Transformer (W|N1|N2|N3|REM)$/,
    )
    // No hypnogram was uploaded, so nothing may claim an accuracy.
    await expect(page.getByText('Agreement with the uploaded hypnogram')).toHaveCount(0)
    await expect(page.getByText(/accuracy/i)).toHaveCount(0)

    const download = page.waitForEvent('download')
    await recordingPage.modelJob('Transformer').getByRole('link', { name: 'Download CSV' }).click()
    const csv = await download
    expect(csv.suggestedFilename()).toBe('upload-run-PSG_transformer-a2-v1.csv')
    const lines = readFileSync(await csv.path(), 'utf8').trim().split('\n')
    expect(lines[0]).toBe(
      'epoch_index,onset_seconds,stage,prob_W,prob_N1,prob_N2,prob_N3,prob_REM,reference_stage',
    )
    expect(lines).toHaveLength(N_EPOCHS + 1)
    expect(lines[N_EPOCHS]).toMatch(/^119,3570,(W|N1|N2|N3|REM),/)
  })

  test('a second model can be started from the recording page', async ({ api, recordingPage }) => {
    const recording = await api.createRecording({ label: 'second-model' })
    await recordingPage.goto(recording.id)
    await expect(recordingPage.page.getByTestId('no-results')).toBeVisible()

    await recordingPage.run('AttnSleep')
    await recordingPage.expectFinished('AttnSleep')
    await expect(recordingPage.epochSummary).toContainText('AttnSleep')
    await expect(recordingPage.modelJob('Transformer')).toHaveAttribute('data-status', 'none')
  })

  test('a file that is not EDF is rejected with a reason', async ({ homePage, page }) => {
    await homePage.goto()
    await homePage.upload({
      name: 'notes.edf',
      mimeType: 'application/octet-stream',
      buffer: Buffer.from('these are not the samples you are looking for'.repeat(20)),
    })

    await expect(homePage.error).toHaveAttribute('data-code', 'EDF_INVALID')
    await expect(homePage.error).toContainText('Not a valid EDF file')
    await expect(page).toHaveURL(/\/$/)
    await expect(homePage.row('notes.edf')).toHaveCount(0)
  })

  test('an EDF without the required channels names what is missing', async ({ homePage }) => {
    await homePage.goto()
    await homePage.upload(psgWithWrongChannel())

    await expect(homePage.error).toHaveAttribute('data-code', 'EDF_MISSING_CHANNEL')
    await expect(homePage.error).toContainText('Missing channel(s) EEG Pz-Oz')
    await expect(homePage.error).toContainText('EEG C3-A2')
  })

  test('a broken hypnogram is rejected and no recording is kept', async ({ homePage }) => {
    const file = uniquePsg('bad-hypnogram')
    await homePage.goto()
    await homePage.upload(file, {
      hypnogram: { ...hypnogramFile(), buffer: Buffer.from('not annotations') },
    })

    await expect(homePage.error).toHaveAttribute('data-code', 'HYPNOGRAM_INVALID')
    await homePage.page.reload()
    await expect(homePage.row(file.name)).toHaveCount(0)
  })

  test('uploading the same file twice opens the existing recording', async ({
    homePage,
    recordingPage,
    api,
    page,
  }) => {
    const file = uniquePsg('duplicate')
    await homePage.goto()
    await homePage.upload(file, { models: [] })
    await expect(page).toHaveURL(/\/recordings\//)
    const first = recordingPage.recordingId()
    api.track(first)

    await homePage.goto()
    await homePage.upload({ ...file, name: 'renamed-copy.edf' }, { models: [] })
    await expect(page).toHaveURL(new RegExp(`/recordings/${first}$`))
    await expect(page.getByRole('heading', { name: file.name })).toBeVisible()

    await homePage.goto()
    await expect(homePage.row(file.name)).toHaveCount(1)
    await expect(homePage.row('renamed-copy.edf')).toHaveCount(0)
  })
})
