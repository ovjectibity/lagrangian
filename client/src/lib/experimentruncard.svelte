<script lang="ts">
  import type {
    TrainingExperimentRun,
  } from "../commondef/messages";

  const {
    run,
    experimentId,
    isCurrent = false,
    onLoadRun,
  } = $props<{
    run: TrainingExperimentRun;
    experimentId: string;
    isCurrent?: boolean;
    onLoadRun?: (
      expId: string,
      runId: string,
    ) => void;
  }>();

  const formatTimestamp = (
    iso: string | null
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
    start: string | null,
    end: string | null
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

  const status = $derived(run.run_status);
</script>

<div
  class="experiment-run-card"
  class:experiment-run-card--current={isCurrent}
>
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
          {run.id}
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
      <button
        class="experiment-run-card-button"
        onclick={() =>
          onLoadRun?.(experimentId, run.id)}
      >
        Load run
      </button>
    </div>
  </div>
</div>
