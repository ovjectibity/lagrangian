<script lang="ts">
  import type { Snippet } from "svelte";
  import type { HTMLButtonAttributes } from "svelte/elements";

  let {
    active = false,
    children,
    class: className = "",
    hasActiveState = true,
    variant = "type-1",
    ...attributes
  }: HTMLButtonAttributes & {
    active?: boolean;
    children?: Snippet;
    hasActiveState?: boolean;
    variant?:
      | "type-1"
      | "type-2"
      | "type-3"
      | "type-4";
  } = $props();

  const isActive = $derived(hasActiveState && active);
</script>

<button
  aria-pressed={hasActiveState ? active : undefined}
  class={
    `lg-button lg-button--${variant} ${
      isActive ? "lg-button--active" : ""
    } ${className}`
  }
  {...attributes}
>
  {@render children?.()}
</button>
