<script lang="ts">
  import type { NeoListSelectEvent } from '@dvcol/neo-svelte';

  import { NeoThemeProvider, NeoButtonGroup, NeoButton, NeoPopSelect } from '@dvcol/neo-svelte';

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

  const handleSelect = (event: NeoListSelectEvent): void => {
    menuOpen = false;
    const current = Array.isArray(event.current) ? event.current[0] : event.current;
    if (current?.item?.value) onSelect(current.item.value as string);
  };
</script>

<NeoThemeProvider target="self">
  <NeoPopSelect
    items={options}
    bind:open={menuOpen}
    openOnHover={false}
    openOnClick={false}
    onSelect={handleSelect}
    width={{ min: '120px' }}
    tooltipProps={{ placement: 'bottom-start', portal: false, borderless: true, class: 'topbar-popselect-tooltip' }}
  >
    {#snippet children(_floating, toggle)}
      <NeoButtonGroup class="topbar-button-group topbar-mode-group" pressed>
        <NeoButton onclick={() => toggle()}>{selected}</NeoButton>
      </NeoButtonGroup>
    {/snippet}
  </NeoPopSelect>
</NeoThemeProvider>
