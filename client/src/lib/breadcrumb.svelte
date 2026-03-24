<script lang="ts">
  import { NeoThemeProvider, NeoButtonGroup, NeoButton } from '@dvcol/neo-svelte';

  const {
    experimentName,
    runName,
  } = $props<{
    experimentName?: string;
    runName?: string;
  }>();
</script>

<NeoThemeProvider target="self">
  <NeoButtonGroup class="topbar-button-group topbar-breadcrumb-group" pressed>
    <NeoButton disabled>
      <span class="breadcrumb-segment" class:breadcrumb-segment-placeholder={!experimentName}>
        {experimentName ?? 'Not set'}
      </span>
      <span class="breadcrumb-separator">›</span>
      <span class="breadcrumb-segment" class:breadcrumb-segment-placeholder={!runName}>
        {runName ?? 'Not set'}
      </span>
    </NeoButton>
  </NeoButtonGroup>
</NeoThemeProvider>
