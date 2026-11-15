<script lang="ts">
  import type { ComponentProps } from "svelte";
  import Card from "./card.svelte";

  let {
    ariaLabel = "Cards",
    items,
    maxColumns = 3,
  }: {
    ariaLabel?: string;
    items: ReadonlyArray<ComponentProps<typeof Card>>;
    maxColumns?: 2 | 3;
  } = $props();
</script>

<section aria-label={ariaLabel} class="lg-cards-view">
  {#if items.length}
    <ul
      class="lg-cards-view__grid"
      class:lg-cards-view__grid--three={maxColumns === 3}
    >
      {#each items as item}
        <li><Card {...item} /></li>
      {/each}
    </ul>
  {:else}
    <p class="lg-cards-view__empty">No cards to display.</p>
  {/if}
</section>
