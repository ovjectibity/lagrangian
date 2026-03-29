<script lang="ts">
  import type {
    TrainingExperimentRun,
  } from "../commondef/training";

  import ExperimentRunCard
    from "./experimentruncard.svelte";

  const {
    runs,
    currentRunId = "",
    onLoadRun,
    onBack,
  } = $props<{
    runs: TrainingExperimentRun[];
    currentRunId?: string;
    onLoadRun?: (
      run: TrainingExperimentRun
    ) => void;
    onBack?: () => void;
  }>();
</script>

<div class="run-card-layout-container">
  <div class="run-card-layout-header">
    <button
      class="run-card-layout-back"
      onclick={() => onBack?.()}
      title="Back to experiments"
    >
      ← Back
    </button>
  </div>
  <div class="run-card-layout-grid">
    {#each runs as run (run.run_id)}
      <ExperimentRunCard
        {run}
        isCurrent={run.run_id === currentRunId}
        {onLoadRun}
      />
    {/each}
    {#if runs.length === 0}
      <div class="run-card-layout-empty">
        No runs for this experiment.
      </div>
    {/if}
  </div>
</div>
