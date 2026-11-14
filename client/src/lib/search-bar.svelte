<script lang="ts">
  let {
    ariaLabel,
    placeholder = "Search",
    value = $bindable(),
  }: {
    ariaLabel: string;
    placeholder?: string;
    value: string;
  } = $props();
</script>

<label class="lg-search-bar">
  <svg aria-hidden="true" viewBox="0 0 24 24">
    <circle cx="11" cy="11" r="6" />
    <path d="m16 16 4 4" />
  </svg>
  <input
    aria-label={ariaLabel}
    bind:value
    placeholder={placeholder}
    type="search"
  />
</label>
