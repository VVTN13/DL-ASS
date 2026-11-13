import { Link } from 'react-router-dom'

export function NotFoundPage({ message = 'This page does not exist.' }: { message?: string }) {
  return (
    <section className="card">
      <h1 className="card-title">Not found</h1>
      <p className="mt-1 text-sm text-slate-600">{message}</p>
      <Link to="/" className="button mt-4">
        Back to recordings
      </Link>
    </section>
  )
}
