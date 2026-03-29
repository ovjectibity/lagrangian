<script lang="ts">
  import {
    onMount, onDestroy
  } from 'svelte';
  import DropdownList
    from './dropdownlist.svelte';

  const {
    options,
    selected,
    onSelect,
  } = $props<{
    options: string[];
    selected: string;
    onSelect: (mode: string) => void;
  }>();

  let menuOpen = $state(false);
  let containerEl: HTMLDivElement;

  const toggle = (e: MouseEvent) => {
    e.stopPropagation();
    menuOpen = !menuOpen;
  };

  const handleSelect = (
    item: string
  ): void => {
    menuOpen = false;
    onSelect(item);
  };

  const handleGlobalClick = (
    e: MouseEvent
  ) => {
    if (
      menuOpen &&
      containerEl &&
      !containerEl.contains(
        e.target as Node
      )
    ) {
      menuOpen = false;
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
  class="topbar-item"
  bind:this={containerEl}
>
  <button
    class="topbar-button"
    onclick={toggle}
  >
    {selected}
  </button>
  <div class="topbar-menu-anchor">
    <DropdownList
      items={options}
      show={menuOpen}
      onSelect={handleSelect}
      position="bottom-left"
    />
  </div>
</div>
