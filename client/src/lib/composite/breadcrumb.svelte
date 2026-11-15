<script lang="ts">
  import Button from "../atomic/button.svelte";

  let {
    ariaLabel = "Breadcrumb",
    items,
  }: {
    ariaLabel?: string;
    items: ReadonlyArray<{
      label: string;
      onclick?: (event: MouseEvent) => void;
    }>;
  } = $props();
</script>

<nav aria-label={ariaLabel} class="lg-breadcrumb">
  <ol>
    {#each items as item, index}
      <li>
        {#if index > 0}
          <svg aria-hidden="true" viewBox="0 0 24 24">
            <path d="m9 6 6 6-6 6" />
          </svg>
        {/if}
        <Button
          active={index === items.length - 1}
          aria-current={index === items.length - 1
            ? "page" : undefined}
          onclick={item.onclick}
          variant="type-1"
        >
          {item.label}
        </Button>
      </li>
    {/each}
  </ol>
</nav>
