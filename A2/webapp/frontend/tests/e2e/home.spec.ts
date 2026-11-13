import { expect, test } from './fixtures'

test.describe('home page', () => {
  test('lists both models and the bundled sample recording', async ({ homePage, page }) => {
    await homePage.goto()

    await expect(page).toHaveTitle('A2 Sleep Staging')
    await expect(page.getByText('not for clinical use')).toBeVisible()
    for (const name of ['Transformer', 'AttnSleep']) {
      await expect(homePage.modelCard(name)).toContainText('5-fold soft voting')
      await expect(homePage.modelCheckbox(name)).toBeChecked()
    }
    await expect(homePage.modelCard('Transformer')).toContainText('EEG Fpz-Cz, EEG Pz-Oz')

    const sample = homePage.row('SC4031-excerpt-PSG.edf')
    await expect(sample).toContainText('sample')
    await expect(sample).toContainText('1 h 00 min')
    await expect(sample.getByRole('button', { name: /Delete/ })).toHaveCount(0)
  })

  test('the upload button needs a file', async ({ homePage }) => {
    await homePage.goto()
    await expect(homePage.submit).toBeDisabled()
  })

  test('an unknown address shows a way back', async ({ page }) => {
    await page.goto('/nothing-here')
    await expect(page.getByRole('heading', { name: 'Not found' })).toBeVisible()
    await page.getByRole('link', { name: 'Back to recordings' }).click()
    await expect(page.getByRole('heading', { name: 'Recordings' })).toBeVisible()
  })
})
