<script lang="ts">
  const {
    isRunning = false,
    isDisabled = false,
    onToggle,
    startLabel = "Start training",
    stopLabel = "Stop training",
  } = $props<{
    isRunning?: boolean;
    isDisabled?: boolean;
    onToggle?: () => void;
    startLabel?: string;
    stopLabel?: string;
  }>();
</script>

<div class="play-pause-cta">
  <button
    class="play-pause-cta-button"
    disabled={isDisabled}
    onclick={() => onToggle?.()}
    title={isRunning ? stopLabel : startLabel}
    aria-label={isRunning ? stopLabel : startLabel}
  >
    {#if isRunning}
      <svg
        class="play-pause-cta-icon"
        viewBox="0 0 24 24"
        fill="currentColor"
        aria-hidden="true"
      >
        <rect x="6" y="5" width="4" height="14" rx="1" />
        <rect x="14" y="5" width="4" height="14" rx="1" />
      </svg>
    {:else}
      <svg
        class="play-pause-cta-icon"
        viewBox="0 0 24 24"
        fill="currentColor"
        aria-hidden="true"
      >
        <path d="M8 5.5v13a1 1 0 0 0 1.53.85l9.5-6.5a1 1 0 0 0 0-1.7l-9.5-6.5A1 1 0 0 0 8 5.5z" />
      </svg>
    {/if}
    <span class="play-pause-cta-label">
      {isRunning ? stopLabel : startLabel}
    </span>
  </button>
</div>
