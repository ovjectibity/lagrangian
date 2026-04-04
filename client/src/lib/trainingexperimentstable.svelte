<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import DropdownList from './dropdownlist.svelte';
  import ExperimentRunCardLayout
    from './experimentruncardlayout.svelte';
  import type {
    TrainingExperiment,
    TrainingExperimentRun,
  } from "../commondef/messages";

  const {
    experiments = [],
    experimentRunsMap = {},
    activeRunId,
    onLoadRun,
  } = $props<{
    experiments?: TrainingExperiment[];
    experimentRunsMap?: Record<string, TrainingExperimentRun[]>;
    activeRunId?: string;
    onLoadRun?: (
      expId: string,
      runId: string,
    ) => void;
  }>();

  let openMenuId = $state<string | null>(null);
  let selectedExperiment = $state<
    TrainingExperiment | null
  >(null);

  function onOpenExperiment(
    experiment: TrainingExperiment
  ): void {
    selectedExperiment = experiment;
  }

  function onBackToTable(): void {
    selectedExperiment = null;
  }

  function toggleMenu(
    event: MouseEvent,
    experimentId: string,
  ): void {
    event.stopPropagation();
    openMenuId =
      openMenuId === experimentId
        ? null
        : experimentId;
  }

  function onMenuSelect(
    _item: string,
    _experiment: TrainingExperiment,
  ): void {
    openMenuId = null;
  }

  function handleGlobalClick(): void {
    openMenuId = null;
  }

  onMount(() => {
    document.addEventListener(
      'click',
      handleGlobalClick,
    );
  });

  onDestroy(() => {
    document.removeEventListener(
      'click',
      handleGlobalClick,
    );
  });

  function formatDate(iso: string): string {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString(
      'en-GB',
      {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
      },
    );
  }
</script>

{#if selectedExperiment}
  <ExperimentRunCardLayout
    experimentId={selectedExperiment.training_experiment_id}
    runs={experimentRunsMap[selectedExperiment.training_experiment_id] ?? []}
    currentRunId={activeRunId ?? ""}
    onLoadRun={onLoadRun}
    onBack={onBackToTable}
  />
{:else}
  <div class="experiments-table-container">
    <table class="experiments-table">
      <thead>
        <tr>
          <th>Experiment ID</th>
          <th>Name</th>
          <th>Created at</th>
          <th>Runs</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {#if experiments.length === 0}
          <tr>
            <td
              colspan="5"
              class="experiments-table-empty"
            >
              No experiments yet.
            </td>
          </tr>
        {/if}
        {#each experiments as experiment}
          <tr>
            <td>
              {experiment.training_experiment_id}
            </td>
            <td>
              {experiment.experiment_name}
            </td>
            <td>
              {formatDate(experiment.created_at)}
            </td>
            <td>
              <div
                class="experiments-table-runs-cell"
              >
                <span>
                  {experiment.training_run_ids.length}
                </span>
                <button
                  class="experiments-table-action"
                  onclick={() =>
                    onOpenExperiment(experiment)}
                  title="View runs"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 16 16"
                    fill="currentColor"
                  >
                    <path
                      fill-rule="evenodd"
                      d="M4.22 11.78a.75.75 0 0 1 0-1.06l5.44-5.44H6.75a.75.75 0 0 1 0-1.5h4.5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-1.5 0V6.56l-5.44 5.44a.75.75 0 0 1-1.06 0Z"
                      clip-rule="evenodd"
                    />
                  </svg>
                </button>
              </div>
            </td>
            <td>
              <div
                class="experiments-table-actions-cell"
              >
                <button
                  class="experiments-table-action"
                  onclick={(e) =>
                    toggleMenu(
                      e,
                      experiment.training_experiment_id,
                    )}
                  title="Actions"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 16 16"
                    fill="currentColor"
                  >
                    <path
                      d="M8 2a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3ZM8 6.5a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3ZM9.5 12.5a1.5 1.5 0 1 0-3 0 1.5 1.5 0 0 0 3 0Z"
                    />
                  </svg>
                </button>
                <div
                  class="experiments-table-menu-anchor"
                >
                  <DropdownList
                    items={['Set as current']}
                    show={openMenuId ===
                      experiment.training_experiment_id}
                    onSelect={(item) =>
                      onMenuSelect(
                        item,
                        experiment,
                      )}
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
{/if}
