<script lang="ts">
  import {
    onMount, onDestroy
  } from 'svelte';
  import DropdownList
    from './dropdownlist.svelte';

  const {
    onImport,
    showExperimentsMenu = false,
    onCreateExperiment,
    onCreateRun,
  } = $props<{
    onImport?: () => void;
    showExperimentsMenu?: boolean;
    onCreateExperiment?: () => void;
    onCreateRun?: () => void;
  }>();

  let fileMenuOpen = $state(false);
  let experimentsMenuOpen = $state(false);
  let containerEl: HTMLDivElement;

  const fileMenuItems = ['Import'];
  const experimentsMenuItems = [
    'Create & load new experiment',
    'Create & load new run',
  ];

  const toggleFile = (e: MouseEvent) => {
    e.stopPropagation();
    experimentsMenuOpen = false;
    fileMenuOpen = !fileMenuOpen;
  };

  const toggleExperiments = (
    e: MouseEvent
  ) => {
    e.stopPropagation();
    fileMenuOpen = false;
    experimentsMenuOpen = !experimentsMenuOpen;
  };

  const handleFileSelect = (
    item: string
  ): void => {
    fileMenuOpen = false;
    if (item === 'Import') onImport?.();
  };

  const handleExperimentsSelect = (
    item: string
  ): void => {
    experimentsMenuOpen = false;
    if (
      item === 'Create & load new experiment'
    ) {
      onCreateExperiment?.();
    } else if (
      item === 'Create & load new run'
    ) {
      onCreateRun?.();
    }
  };

  const handleGlobalClick = (
    e: MouseEvent
  ) => {
    if (
      containerEl &&
      !containerEl.contains(e.target as Node)
    ) {
      fileMenuOpen = false;
      experimentsMenuOpen = false;
    }
  };

  onMount(() => {
    document.addEventListener(
      'click', handleGlobalClick
    );
  });

  onDestroy(() => {
    document.removeEventListener(
      'click', handleGlobalClick
    );
  });
</script>

<div
  class="topbar-menubar"
  bind:this={containerEl}
>
  <button
    class="topbar-button
      topbar-menubar-button"
    onclick={toggleFile}
  >
    File
  </button>
  <div class="topbar-menu-anchor">
    <DropdownList
      items={fileMenuItems}
      show={fileMenuOpen}
      onSelect={handleFileSelect}
      position="bottom-left"
    />
  </div>

  {#if showExperimentsMenu}
    <button
      class="topbar-button
        topbar-menubar-button"
      onclick={toggleExperiments}
    >
      Experiments
    </button>
    <div class="topbar-menu-anchor">
      <DropdownList
        items={experimentsMenuItems}
        show={experimentsMenuOpen}
        onSelect={handleExperimentsSelect}
        position="bottom-left"
      />
    </div>
  {/if}
</div>
