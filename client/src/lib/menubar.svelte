<script lang="ts">
  import type { NeoListSelectEvent } from '@dvcol/neo-svelte';

  import { NeoThemeProvider, NeoButtonGroup, NeoButton, NeoPopSelect } from '@dvcol/neo-svelte';

  const { onImport } = $props<{
    onImport?: () => void;
  }>();

  let fileMenuOpen = $state(false);

  const fileMenuItems = ['Import'];

  const handleFileMenuSelect = (event: NeoListSelectEvent): void => {
    fileMenuOpen = false;
    const selected = Array.isArray(event.current) ? event.current[0] : event.current;
    if (selected?.item?.value === 'Import') onImport?.();
  };
</script>

<NeoThemeProvider target="self">
  <NeoPopSelect
    items={fileMenuItems}
    bind:open={fileMenuOpen}
    openOnHover={false}
    openOnClick={false}
    onSelect={handleFileMenuSelect}
    width={{ max: '120px' }}
    tooltipProps={{ placement: 'bottom-start', portal: false, borderless: true, class: 'topbar-popselect-tooltip' }}
  >
    {#snippet children(_floating, toggle)}
      <NeoButtonGroup class="topbar-button-group menubar-button-group" pressed>
        <NeoButton onclick={() => toggle()}>File</NeoButton>
      </NeoButtonGroup>
    {/snippet}
  </NeoPopSelect>
</NeoThemeProvider>
