<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  const {
    isRunning = false,
    isDisabled = false,
    onToggle,
    startLabel = "Start training",
    stopLabel = "Stop training",
    showNewRunOption = false,
    onStartNewRun,
  } = $props<{
    isRunning?: boolean;
    isDisabled?: boolean;
    onToggle?: () => void;
    startLabel?: string;
    stopLabel?: string;
    showNewRunOption?: boolean;
    onStartNewRun?: () => void;
  }>();

  let dropdownOpen = $state(false);
  let dropdownAnchor = $state<HTMLDivElement | undefined>(undefined);

  function handleGlobalClick(e: MouseEvent) {
    if (
      dropdownOpen &&
      dropdownAnchor &&
      !dropdownAnchor.contains(e.target as Node)
    ) {
      dropdownOpen = false;
    }
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
</script>

<div class="topbar-item">
  {#if showNewRunOption && !isRunning}
    <div
      class="playpause-split"
      bind:this={dropdownAnchor}
    >
      <button
        class="topbar-button topbar-cta-button playpause-split-main"
        disabled={isDisabled}
        onclick={() => onToggle?.()}
      >
        <svg
          class="play-pause-cta-icon"
          viewBox="0 0 24 24"
          fill="currentColor"
          aria-hidden="true"
        >
          <path d="M8 5.5v13a1 1 0 0 0 1.53.85l9.5-6.5a1 1 0 0 0 0-1.7l-9.5-6.5A1 1 0 0 0 8 5.5z" />
        </svg>
        <span>{startLabel}</span>
      </button>
      <button
        class="topbar-button topbar-cta-button playpause-split-chevron"
        disabled={isDisabled}
        onclick={(e) => {
          e.stopPropagation();
          dropdownOpen = !dropdownOpen;
        }}
        title="More options"
      >
        <svg
          viewBox="0 0 16 16"
          fill="currentColor"
          class="playpause-chevron-icon"
          aria-hidden="true"
        >
          <path
            fill-rule="evenodd"
            d="M4.22 6.22a.75.75 0 0 1 1.06 0L8 8.94l2.72-2.72a.75.75 0 1 1 1.06 1.06l-3.25 3.25a.75.75 0 0 1-1.06 0L4.22 7.28a.75.75 0 0 1 0-1.06Z"
            clip-rule="evenodd"
          />
        </svg>
      </button>
      {#if dropdownOpen}
        <div class="playpause-dropdown">
          <button
            class="playpause-dropdown-item"
            onclick={() => {
              dropdownOpen = false;
              onStartNewRun?.();
            }}
          >
            Start new run
          </button>
        </div>
      {/if}
    </div>
  {:else}
    <button
      class="topbar-button topbar-cta-button"
      disabled={isDisabled}
      onclick={() => onToggle?.()}
    >
      {#if isRunning}
        <svg
          class="play-pause-cta-icon"
          viewBox="0 0 24 24"
          fill="currentColor"
          aria-hidden="true"
        >
          <rect
            x="6" y="5"
            width="4" height="14" rx="1"
          />
          <rect
            x="14" y="5"
            width="4" height="14" rx="1"
          />
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
      <span>
        {isRunning ? stopLabel : startLabel}
      </span>
    </button>
  {/if}
</div>
