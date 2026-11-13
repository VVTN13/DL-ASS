import type { Locator, Page } from '@playwright/test'

import type { UploadFile } from '../e2e/fixtures'

export class HomePage {
  readonly page: Page
  readonly psgInput: Locator
  readonly hypnogramInput: Locator
  readonly submit: Locator
  readonly error: Locator

  constructor(page: Page) {
    this.page = page
    this.psgInput = page.getByLabel('PSG EDF')
    this.hypnogramInput = page.getByLabel('Hypnogram EDF')
    this.submit = page.getByRole('button', { name: /^Upload/ })
    this.error = page.getByRole('alert')
  }

  async goto() {
    await this.page.goto('/')
    await this.page.getByRole('heading', { name: 'Recordings' }).waitFor()
  }

  modelCard(name: string): Locator {
    return this.page.getByTestId('model-card').filter({ hasText: name })
  }

  modelCheckbox(name: string): Locator {
    return this.page.getByRole('group', { name: 'Models' }).getByLabel(name)
  }

  row(filename: string): Locator {
    return this.page.getByTestId('recording-row').filter({ hasText: filename })
  }

  /** Fill the form and submit. `models` lists the checkboxes that should stay ticked. */
  async upload(psg: UploadFile, options: { hypnogram?: UploadFile; models?: string[] } = {}) {
    await this.psgInput.setInputFiles(psg)
    if (options.hypnogram) await this.hypnogramInput.setInputFiles(options.hypnogram)
    if (options.models) {
      for (const checkbox of await this.page.getByRole('group', { name: 'Models' }).getByRole('checkbox').all()) {
        const label = (await checkbox.locator('xpath=..').innerText()).trim()
        await checkbox.setChecked(options.models.includes(label))
      }
    }
    await this.submit.click()
  }
}
