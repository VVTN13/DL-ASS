import { TRANSFORMER, expect, test } from './fixtures'

test.describe('recordings list', () => {
  test('shows job status and removes a recording on delete', async ({ api, homePage, page }) => {
    const recording = await api.createRecording({ label: 'to-delete', models: [TRANSFORMER] })
    await api.waitForJobs(recording.id)
    await homePage.goto()

    const row = homePage.row(recording.filename)
    await expect(row.getByTestId('job-badge')).toHaveAttribute('data-status', 'succeeded')
    await expect(row.getByTestId('job-badge')).toHaveText('Transformer: succeeded')

    page.once('dialog', (dialog) => dialog.accept())
    await row.getByRole('button', { name: `Delete ${recording.filename}` }).click()
    await expect(row).toHaveCount(0)

    await page.goto(`/recordings/${recording.id}`)
    await expect(page.getByText('This recording does not exist or has been deleted.')).toBeVisible()
  })

  test('progress is visible without reloading while a job runs', async ({ api, homePage }) => {
    const recording = await api.createRecording({ label: 'live-status', models: [TRANSFORMER] })
    await homePage.goto()

    const badge = homePage.row(recording.filename).getByTestId('job-badge')
    await expect(badge).toBeVisible()
    await expect(badge).toHaveAttribute('data-status', 'succeeded', { timeout: 150_000 })
  })

  test('cancelling the confirmation keeps the recording', async ({ api, homePage, page }) => {
    const recording = await api.createRecording({ label: 'keep-me' })
    await homePage.goto()

    page.once('dialog', (dialog) => dialog.dismiss())
    await homePage.row(recording.filename).getByRole('button', { name: /Delete/ }).click()
    await expect(homePage.row(recording.filename)).toHaveCount(1)
  })
})
