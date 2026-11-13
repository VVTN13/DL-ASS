import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import type { UseQueryResult } from '@tanstack/react-query'

import { api, unwrap, uploadRecording } from './client'
import type { Job, MetricsScope, Predictions, Recording } from './client'

const POLL_MS = 1500

export function isActive(job: Job): boolean {
  return job.status === 'queued' || job.status === 'running'
}

function pollWhileActive(recordings: Recording[] | undefined): number | false {
  return recordings?.some((recording) => recording.jobs.some(isActive)) ? POLL_MS : false
}

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: () => unwrap(api.GET('/api/v1/models')),
    staleTime: Infinity,
  })
}

export function useRecordings() {
  return useQuery({
    queryKey: ['recordings'],
    queryFn: () => unwrap(api.GET('/api/v1/recordings')),
    refetchInterval: (query) => pollWhileActive(query.state.data?.data),
  })
}

export function useRecording(recordingId: string) {
  return useQuery({
    queryKey: ['recordings', recordingId],
    queryFn: () =>
      unwrap(
        api.GET('/api/v1/recordings/{recording_id}', {
          params: { path: { recording_id: recordingId } },
        }),
      ),
    refetchInterval: (query) => pollWhileActive(query.state.data && [query.state.data]),
    retry: false,
  })
}

export function useUploadRecording() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: uploadRecording,
    onSuccess: (recording) => {
      queryClient.setQueryData(['recordings', recording.id], recording)
      return queryClient.invalidateQueries({ queryKey: ['recordings'], exact: true })
    },
  })
}

export function useDeleteRecording() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (recordingId: string) =>
      unwrap(
        api.DELETE('/api/v1/recordings/{recording_id}', {
          params: { path: { recording_id: recordingId } },
        }),
      ),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['recordings'] }),
  })
}

export function useCreateJob(recordingId: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (modelId: string) =>
      unwrap(
        api.POST('/api/v1/jobs', { body: { recording_id: recordingId, model_id: modelId } }),
      ),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['recordings'] }),
  })
}

function combinePredictions(results: UseQueryResult<Predictions>[]) {
  return {
    data: results.map((result) => result.data),
    error: results.find((result) => result.error)?.error ?? null,
  }
}

/**
 * Predictions never change once a job has succeeded, so they are cached for the session.
 * `combine` keeps the returned object referentially stable between polls.
 */
export function usePredictions(jobs: Job[]) {
  return useQueries({
    queries: jobs.map((job) => ({
      queryKey: ['predictions', job.id],
      queryFn: () =>
        unwrap(
          api.GET('/api/v1/jobs/{job_id}/predictions', { params: { path: { job_id: job.id } } }),
        ),
      staleTime: Infinity,
    })),
    combine: combinePredictions,
  })
}

export function useMetrics(jobs: Job[], scope: MetricsScope, enabled: boolean) {
  return useQueries({
    queries: jobs.map((job) => ({
      queryKey: ['metrics', job.id, scope],
      queryFn: () =>
        unwrap(
          api.GET('/api/v1/jobs/{job_id}/metrics', {
            params: { path: { job_id: job.id }, query: { scope } },
          }),
        ),
      staleTime: Infinity,
      enabled,
      retry: false,
    })),
  })
}

export function useEpoch(recordingId: string, epochIndex: number, enabled: boolean) {
  return useQuery({
    queryKey: ['epochs', recordingId, epochIndex],
    queryFn: () =>
      unwrap(
        api.GET('/api/v1/recordings/{recording_id}/epochs/{epoch_index}', {
          params: { path: { recording_id: recordingId, epoch_index: epochIndex } },
        }),
      ),
    staleTime: Infinity,
    enabled,
    placeholderData: (previous) => previous,
  })
}
