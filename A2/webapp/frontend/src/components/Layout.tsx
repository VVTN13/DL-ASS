import { Link, Outlet } from 'react-router-dom'

export function Layout() {
  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 py-6 sm:px-6">
      <header className="mb-6">
        <Link to="/" className="text-2xl font-semibold tracking-tight text-slate-900">
          A2 Sleep Staging
        </Link>
        <p className="mt-1 text-sm text-slate-600">
          Transformer and AttnSleep, five-fold soft voting · research prototype, not for clinical
          use
        </p>
      </header>
      <main className="flex-1 space-y-5">
        <Outlet />
      </main>
      <footer className="mt-10 border-t border-slate-200 pt-4 text-xs text-slate-500">
        Models trained on Sleep-EDF Expanded (sleep-cassette) for CO3133 Assignment 2. Uploads are
        visible to everyone who can open this page — do not upload data that identifies a person.
      </footer>
    </div>
  )
}
