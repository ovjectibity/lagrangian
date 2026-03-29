<script lang="ts">
  import {
    onMount, onDestroy
  } from 'svelte';
  import DropdownList
    from './dropdownlist.svelte';

  const { onImport } = $props<{
    onImport?: () => void;
  }>();

  let fileMenuOpen = $state(false);
  let containerEl: HTMLDivElement;

  const fileMenuItems = ['Import'];

  const toggle = (e: MouseEvent) => {
    e.stopPropagation();
    fileMenuOpen = !fileMenuOpen;
  };

  const handleSelect = (
    item: string
  ): void => {
    fileMenuOpen = false;
    if (item === 'Import') onImport?.();
  };

  const handleGlobalClick = (
    e: MouseEvent
  ) => {
    if (
      fileMenuOpen &&
      containerEl &&
      !containerEl.contains(
        e.target as Node
      )
    ) {
      fileMenuOpen = false;
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
    onclick={toggle}
  >
    File
  </button>
  <div class="topbar-menu-anchor">
    <DropdownList
      items={fileMenuItems}
      show={fileMenuOpen}
      onSelect={handleSelect}
      position="bottom-left"
    />
  </div>
</div>
