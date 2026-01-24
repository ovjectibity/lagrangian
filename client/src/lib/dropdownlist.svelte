<script lang="ts">
  const { items, show } = $props<{ items: string[]; show: boolean }>();
</script>

{#if show}
  <div class="dropdown-list">
    {#each items as item}
      <button class="dropdown-item">{item}</button>
    {/each}
  </div>
{/if}
