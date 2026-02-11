<script lang="ts">
  const { items, show, onSelect, selectedItem, position = 'bottom-left' } = $props<{
    items: string[];
    show: boolean;
    onSelect?: (item: string) => void;
    selectedItem?: string;
    position?: 'bottom-left' | 'bottom-right';
  }>();
</script>

{#if show}
  <div class={`dropdown-list ${position}`}>
    {#each items as item}
      <button
        class={`dropdown-item ${selectedItem === item ? 'selected' : ''}`}
        onclick={() => onSelect?.(item)}
      >
        {item}
      </button>
    {/each}
  </div>
{/if}
