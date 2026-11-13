import { useModels, useRecordings } from '../api/queries'
import { ErrorMessage } from '../components/ErrorMessage'
import { ModelCards } from '../components/ModelCards'
import { RecordingsTable } from '../components/RecordingsTable'
import { UploadForm } from '../components/UploadForm'

export function HomePage() {
  const models = useModels()
  const recordings = useRecordings()
  if (models.error || recordings.error) {
    return <ErrorMessage error={models.error ?? recordings.error} />
  }
  if (!models.data || !recordings.data) {
    return <p className="text-sm text-slate-500">Loading…</p>
  }
  return (
    <>
      <ModelCards models={models.data} />
      <UploadForm models={models.data} />
      <RecordingsTable recordings={recordings.data.data} models={models.data} />
    </>
  )
}
