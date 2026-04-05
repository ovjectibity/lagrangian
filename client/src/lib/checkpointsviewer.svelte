<script lang="ts">
  import type { CheckpointReference } from '../commondef/training';

  const { checkpoints = [] } = $props<{ checkpoints?: CheckpointReference[] }>();

  const formatTimestamp = (iso: string): string =>
    new Date(iso).toLocaleString('en-GB', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
</script>

<div class="checkpoints-viewer">
  {#if checkpoints.length === 0}
    <div class="checkpoints-viewer-empty">No checkpoints saved yet.</div>
  {:else}
    {#each checkpoints as ckpt}
      <div class="experiment-run-card">
        <div class="experiment-run-card-body">
          <div class="experiment-run-card-attrs">
            <div class="experiment-run-card-row">
              <span class="experiment-run-card-label">Checkpoint ID</span>
              <span class="experiment-run-card-value">{ckpt.id}</span>
            </div>
            <div class="experiment-run-card-row">
              <span class="experiment-run-card-label">Experiment ID</span>
              <span class="experiment-run-card-value">{ckpt.training_experiment_id}</span>
            </div>
            <div class="experiment-run-card-row">
              <span class="experiment-run-card-label">Run ID</span>
              <span class="experiment-run-card-value">{ckpt.training_run_id}</span>
            </div>
            <div class="experiment-run-card-row">
              <span class="experiment-run-card-label">Saved at</span>
              <span class="experiment-run-card-value">{formatTimestamp(ckpt.created_at)}</span>
            </div>
          </div>
        </div>
      </div>
    {/each}
  {/if}
</div>
