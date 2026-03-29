<script lang="ts">
  import {
    onMount, onDestroy
  } from 'svelte';
  import type { Component } from 'svelte';
  import DropdownList
    from './dropdownlist.svelte';

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
    onTabChannelChange?: (
      channel: string
    ) => void;
    activeTabId?: string;
    onTabChange?: (tabId: string) => void;
  }>();

  let internalActiveTabId = $state<
    string | undefined
  >(undefined);

  $effect(() => {
    if (activeTabId) {
      const exists = tabs.some(
        (tab: any) => tab.id === activeTabId
      );
      if (
        exists &&
        activeTabId !== internalActiveTabId
      ) {
        internalActiveTabId = activeTabId;
      }
    }
  });

  $effect(() => {
    if (
      !internalActiveTabId ||
      !tabs.some(
        (tab: any) =>
          tab.id === internalActiveTabId
      )
    ) {
      internalActiveTabId = tabs[0]?.id;
    }
  });

  const handleTabClick = (
    tabId: string
  ) => {
    internalActiveTabId = tabId;
    onTabChange?.(tabId);
  };

  let tabChannelOpen = $state(false);
  let channelMenuEl: HTMLDivElement;

  const toggleTabChannel = (
    e: MouseEvent
  ) => {
    e.stopPropagation();
    tabChannelOpen = !tabChannelOpen;
  };

  const handleTabChannelSelect = (
    item: string
  ): void => {
    tabChannelOpen = false;
    onTabChannelChange?.(item);
  };

  const handleGlobalClick = (
    e: MouseEvent
  ) => {
    if (
      tabChannelOpen &&
      channelMenuEl &&
      !channelMenuEl.contains(
        e.target as Node
      )
    ) {
      tabChannelOpen = false;
    }
  };

  onMount(() => {
    document.addEventListener(
      'click', handleGlobalClick
    );
  });

  onDestroy(() => {
    document.removeEventListener(
      'click', handleGlobalClick
    );
  });

  let activeTab = $derived(
    tabs.find(
      (tab: any) =>
        tab.id === internalActiveTabId
    ) ?? tabs[0]
  );
</script>

<div class="tabbed-container">
  <div class="tabbed-tabs">
    <div class="tabbed-tabs-left">
      {#each tabs as tab (tab.id)}
        <button
          class="tabbed-tab"
          class:active={
            tab.id === internalActiveTabId
          }
          onclick={() =>
            handleTabClick(tab.id)}
        >
          {tab.name}
        </button>
      {/each}
    </div>
    {#if showTabChannel}
      <div class="tabbed-tabs-right">
        <div
          class="tab-channel-menu"
          bind:this={channelMenuEl}
        >
          <button
            class="topbar-button"
            onclick={toggleTabChannel}
          >
            {selectedTabChannel}
          </button>
          <div
            class="topbar-menu-anchor"
          >
            <DropdownList
              items={tabChannelOptions}
              show={tabChannelOpen}
              onSelect={
                handleTabChannelSelect
              }
              position="bottom-right"
            />
          </div>
        </div>
      </div>
    {/if}
  </div>
  <div class="tabbed-content">
    {#if activeTab?.component}
      {@const ActiveComponent =
        activeTab.component}
      <ActiveComponent
        {...(activeTab.props ?? {})}
      />
    {/if}
  </div>
</div>
