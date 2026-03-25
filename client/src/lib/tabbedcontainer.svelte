<script lang="ts">
  import type { NeoListSelectEvent } from '@dvcol/neo-svelte';
  import type { Component } from 'svelte';

  import { NeoThemeProvider, NeoTabs, NeoTab, NeoTabsCard, NeoTabPanel, NeoPopSelect, NeoButton } from '@dvcol/neo-svelte';

  export type TabbedTab = {
    id: string;
    name: string;
    component?: Component<any>;
    props?: Record<string, unknown>;
  };

  const {
    tabs,
    showTabChannel = false,
    tabChannelOptions = [],
    selectedTabChannel = '',
    onTabChannelChange,
    activeTabId = '',
    onTabChange,
  } = $props<{
    tabs: TabbedTab[];
    showTabChannel?: boolean;
    tabChannelOptions?: string[];
    selectedTabChannel?: string;
    onTabChannelChange?: (channel: string) => void;
    activeTabId?: string;
    onTabChange?: (tabId: string) => void;
  }>();

  let internalActiveTabId = $state<string | undefined>(undefined);

  $effect(() => {
    if (activeTabId) {
      const exists = tabs.some((tab) => tab.id === activeTabId);
      if (exists && activeTabId !== internalActiveTabId) {
        internalActiveTabId = activeTabId;
      }
    }
  });

  $effect(() => {
    if (!internalActiveTabId || !tabs.some((tab) => tab.id === internalActiveTabId)) {
      internalActiveTabId = tabs[0]?.id;
    }
  });

  const handleTabChange = (tabId: string | number | undefined) => {
    if (tabId && typeof tabId === 'string') {
      internalActiveTabId = tabId;
      onTabChange?.(tabId);
    }
  };

  let tabChannelOpen = $state(false);

  const handleTabChannelSelect = (event: NeoListSelectEvent): void => {
    tabChannelOpen = false;
    const selected = Array.isArray(event.current) ? event.current[0] : event.current;
    if (selected?.item?.value) {
      onTabChannelChange?.(selected.item.value as string);
    }
  };

  let activeTab = $derived(
    tabs.find((tab) => tab.id === internalActiveTabId) ?? tabs[0]
  );
</script>

{#snippet panes()}
  <NeoTabsCard animate={false} class="tabbed-neo-card">
    {#each tabs as tab (tab.id)}
      <NeoTabPanel tabId={tab.id}>
        {#if tab.component}
          {@const ActiveComponent = tab.component}
          <ActiveComponent {...(tab.props ?? {})} />
        {/if}
      </NeoTabPanel>
    {/each}
  </NeoTabsCard>
{/snippet}

<NeoThemeProvider target="self">
  <NeoTabs
    class="tabbed-neo-tabs"
    {panes}
    active={internalActiveTabId}
    onchange={handleTabChange}
    pressed
    slide={false}
    toggle={false}
  >
    {#each tabs as tab (tab.id)}
      <NeoTab tabId={tab.id}>{tab.name}</NeoTab>
    {/each}

    {#if showTabChannel}
      <div class="tabbed-channel-spacer"></div>
      <NeoPopSelect
        items={tabChannelOptions}
        bind:open={tabChannelOpen}
        openOnHover={false}
        openOnClick={false}
        onSelect={handleTabChannelSelect}
        width={{ min: '120px' }}
        tooltipProps={{ placement: 'bottom-end', portal: false, borderless: true, class: 'topbar-popselect-tooltip' }}
        buttonProps={{ style: 'font-size: 13px' }}
      >
        {#snippet children(_floating, toggle)}
          <NeoButton onclick={() => toggle()}>{selectedTabChannel}</NeoButton>
        {/snippet}
      </NeoPopSelect>
    {/if}
  </NeoTabs>
</NeoThemeProvider>
