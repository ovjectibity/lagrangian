<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import DropdownList from './dropdownlist.svelte';
  import type { TrainingExperiment } from "../commondef/training";

  const DUMMY_EXPERIMENTS: TrainingExperiment[] = [
    {
      experiment_id: 'exp-001',
      experiment_name: 'amber-falcon-run',
      created_at: '2026-03-01T00:00:00Z',
      experiment_runs: [
        { experiment_id: 'exp-001', run_name: 'run-alpha', run_id: 'run-1', created_at: '', run_start_at: '', run_end_at: '', training_graph: {} as any, training_update: [] },
        { experiment_id: 'exp-001', run_name: 'run-beta', run_id: 'run-2', created_at: '', run_start_at: '', run_end_at: '', training_graph: {} as any, training_update: [] },
      ],
    },
    {
      experiment_id: 'exp-002',
      experiment_name: 'cobalt-river-dawn',
      created_at: '2026-03-10T00:00:00Z',
      experiment_runs: [
        { experiment_id: 'exp-002', run_name: 'run-alpha', run_id: 'run-1', created_at: '', run_start_at: '', run_end_at: '', training_graph: {} as any, training_update: [] },
      ],
    },
    {
      experiment_id: 'exp-003',
      experiment_name: 'silver-marsh-echo',
      created_at: '2026-03-20T00:00:00Z',
      experiment_runs: [],
    },
  ];

  const experiments: TrainingExperiment[] = DUMMY_EXPERIMENTS;

  let openMenuId = $state<string | null>(null);

  function onOpenExperiment(_experiment: TrainingExperiment): void {
    // no-op for now
  }

  function toggleMenu(event: MouseEvent, experimentId: string): void {
    event.stopPropagation();
    openMenuId = openMenuId === experimentId ? null : experimentId;
  }

  function onMenuSelect(_item: string, _experiment: TrainingExperiment): void {
    openMenuId = null;
    // no-op for now
  }

  function handleGlobalClick(): void {
    openMenuId = null;
  }

  onMount(() => {
    document.addEventListener('click', handleGlobalClick);
  });

  onDestroy(() => {
    document.removeEventListener('click', handleGlobalClick);
  });

  function formatDate(iso: string): string {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
  }
</script>

<div class="experiments-table-container">
  <table class="experiments-table">
    <thead>
      <tr>
        <th>Experiment ID</th>
        <th>Name</th>
        <th>Created at</th>
        <th>Experiment runs</th>
        <th>Actions</th>
      </tr>
    </thead>
    <tbody>
      {#if experiments.length === 0}
        <tr>
          <td colspan="5" class="experiments-table-empty">No experiments yet.</td>
        </tr>
      {/if}
      {#each experiments as experiment}
        <tr>
          <td>{experiment.experiment_id}</td>
          <td>{experiment.experiment_name}</td>
          <td>{formatDate(experiment.created_at)}</td>
          <td>
            <div class="experiments-table-runs-cell">
              <span>{experiment.experiment_runs.length}</span>
              <button class="experiments-table-action" onclick={() => onOpenExperiment(experiment)} title="View runs">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
                  <path fill-rule="evenodd" d="M4.22 11.78a.75.75 0 0 1 0-1.06l5.44-5.44H6.75a.75.75 0 0 1 0-1.5h4.5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-1.5 0V6.56l-5.44 5.44a.75.75 0 0 1-1.06 0Z" clip-rule="evenodd" />
                </svg>
              </button>
            </div>
          </td>
          <td>
            <div class="experiments-table-actions-cell">
              <button class="experiments-table-action" onclick={(e) => toggleMenu(e, experiment.experiment_id)} title="Actions">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
                  <path d="M8 2a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3ZM8 6.5a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3ZM9.5 12.5a1.5 1.5 0 1 0-3 0 1.5 1.5 0 0 0 3 0Z" />
                </svg>
              </button>
              <div class="experiments-table-menu-anchor">
                <DropdownList
                  items={['Set as current']}
                  show={openMenuId === experiment.experiment_id}
                  onSelect={(item) => onMenuSelect(item, experiment)}
                  position="bottom-right"
                />
              </div>
            </div>
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>
