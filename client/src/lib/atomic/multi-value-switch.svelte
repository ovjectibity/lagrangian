<script lang="ts">
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

<div
  aria-label={ariaLabel}
  class="lg-multi-value-switch"
  role="radiogroup"
>
  {#each options as option}
    <button
      aria-checked={option === value}
      class:lg-multi-value-switch__option--selected={option === value}
      class="lg-multi-value-switch__option"
      onclick={() => (value = option)}
      role="radio"
      type="button"
    >
      {option}
    </button>
  {/each}
</div>
