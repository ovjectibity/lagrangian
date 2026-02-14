<script lang="ts">
  import DropdownList from './dropdownlist.svelte';
  import { onMount, onDestroy } from 'svelte';

  const { onImport } = $props<{
    onImport?: () => void;
  }>();

  let showFileMenu = $state(false);
  let menuContainer: HTMLDivElement;

  function toggleFileMenu() {
    showFileMenu = !showFileMenu;
  }

  const handleFileMenuSelect = (item: string): void => {
    showFileMenu = false;
    if (item === 'Import') {
      onImport?.();
    }
  };

  function handleClickOutside(event: MouseEvent) {
    if (menuContainer && !menuContainer.contains(event.target as Node)) {
      showFileMenu = false;
    }
  }

  onMount(() => {
    document.addEventListener('click', handleClickOutside);
  });

  onDestroy(() => {
    document.removeEventListener('click', handleClickOutside);
  });
</script>

<div class="menu-bar">
  <div class="menu-bar-item" style="position: relative;" bind:this={menuContainer}>
    <button class="menu-button" onclick={toggleFileMenu}>File</button>
    <DropdownList
      items={['Import']}
      show={showFileMenu}
      onSelect={handleFileMenuSelect}
    />
  </div>
</div>
