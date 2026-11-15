<script lang="ts">
  import Button from "../atomic/button.svelte";

  let {
    ariaLabel,
    options,
    value = $bindable(),
  }: {
    ariaLabel: string;
    options: readonly string[];
    value: string;
  } = $props();
</script>

<div aria-label={ariaLabel} class="lg-tab-switcher" role="group">
  {#each options as option}
    <Button
      active={option === value}
      class="lg-tab-switcher__button"
      onclick={() => (value = option)}
      variant="type-2"
    >
      {option}
    </Button>
  {/each}
</div>
