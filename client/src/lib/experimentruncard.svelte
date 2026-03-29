<script lang="ts">
  import type {
    TrainingExperimentRun,
  } from "../commondef/training";

  import {
    NeoCard,
    NeoButton,
  } from "@dvcol/neo-svelte";

  const {
    run,
    isCurrent = false,
    onLoadRun,
  } = $props<{
    run: TrainingExperimentRun;
    isCurrent?: boolean;
    onLoadRun?: (
      run: TrainingExperimentRun
    ) => void;
  }>();

  const formatTimestamp = (
    iso: string
  ): string => {
    if (!iso) return "—";
    return new Date(iso).toLocaleString(
      "en-GB",
      {
        day: "2-digit",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }
    );
  };

  const computeRuntime = (
    start: string,
    end: string
  ): string => {
    if (!start) return "—";
    const s = new Date(start).getTime();
    const e = end
      ? new Date(end).getTime()
      : Date.now();
    const diffMs = e - s;
    const mins = Math.floor(diffMs / 60000);
    const secs = Math.floor(
      (diffMs % 60000) / 1000
    );
    return `${mins}m ${secs}s`;
  };

  const deriveStatus = (
    run: TrainingExperimentRun
  ): string => {
    if (run.run_end_at) return "Completed";
    if (run.run_start_at) return "Running";
    return "Failed";
  };

  const status = $derived(deriveStatus(run));
</script>

<NeoCard
  class={[
    "experiment-run-card",
    isCurrent
      ? "experiment-run-card--current"
      : "",
  ].join(" ")}
  elevation={-2}
  pressed
  hover={0}
  spacing="0"
>
  {#snippet children()}
    <div class="experiment-run-card-body">
      <div class="experiment-run-card-attrs">
        <div class="experiment-run-card-row">
          <span
            class="experiment-run-card-label"
          >
            Run ID
          </span>
          <span
            class="experiment-run-card-value"
          >
            {run.run_id}
          </span>
        </div>
        <div class="experiment-run-card-row">
          <span
            class="experiment-run-card-label"
          >
            Start
          </span>
          <span
            class="experiment-run-card-value"
          >
            {formatTimestamp(run.run_start_at)}
          </span>
        </div>
        <div class="experiment-run-card-row">
          <span
            class="experiment-run-card-label"
          >
            Runtime
          </span>
          <span
            class="experiment-run-card-value"
          >
            {computeRuntime(
              run.run_start_at,
              run.run_end_at
            )}
          </span>
        </div>
        <div class="experiment-run-card-row">
          <span
            class="experiment-run-card-label"
          >
            Status
          </span>
          <span
            class={[
              "experiment-run-card-value",
              "experiment-run-card-status",
              `experiment-run-card-status--${status.toLowerCase()}`,
            ].join(" ")}
          >
            {status}
          </span>
        </div>
      </div>
      <div class="experiment-run-card-cta">
        <NeoButton
          onclick={() => onLoadRun?.(run)}
        >
          Load run
        </NeoButton>
      </div>
    </div>
  {/snippet}
</NeoCard>
