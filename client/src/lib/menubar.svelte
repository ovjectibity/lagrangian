<script lang="ts">
  import type { NeoMenuItem } from '@dvcol/neo-svelte';

  import { NeoThemeProvider, NeoButtonGroup, NeoButton, NeoMenu } from '@dvcol/neo-svelte';

  const { onImport } = $props<{
    onImport?: () => void;
  }>();

  let fileMenuOpen = $state(false);

  const fileMenuItems: NeoMenuItem[] = [
    { value: 'Import', label: 'Import' },
  ];

  const handleFileMenuSelect = (item: NeoMenuItem): void => {
    fileMenuOpen = false;
    if (item.value === 'Import') {
      onImport?.();
    }
  };
</script>

<NeoThemeProvider target="self">
  <NeoButtonGroup class="menubar-button-group" rounded={false}>
    <NeoMenu
      items={fileMenuItems}
      bind:open={fileMenuOpen}
      openOnHover={false}
      portal={false}
      placement="bottom-start"
      rounded="2px"
      onSelect={handleFileMenuSelect}
    >
      <NeoButton toggle bind:checked={fileMenuOpen}>File</NeoButton>
    </NeoMenu>
  </NeoButtonGroup>
</NeoThemeProvider>
