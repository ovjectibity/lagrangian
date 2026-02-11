<script lang="ts">
  import { onDestroy, onMount } from "svelte";

  import DropdownList from "./dropdownlist.svelte";
  import type { Component } from "svelte";

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
    selectedTabChannel = "",
    onTabChannelChange,
    activeTabId = "",
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

  let internalActiveTabId = $state('');
  let activeTab = $derived(
    tabs.find((tab: TabbedTab) => tab.id === internalActiveTabId) ?? tabs[0]
  );
  let showTabChannelMenu = $state(false);
  let tabChannelMenuContainer = $state<HTMLDivElement | undefined>(undefined);

  $effect(() => {
    if (activeTab && activeTab.id !== internalActiveTabId) {
      internalActiveTabId = activeTab.id;
    }
    const activeExists = tabs.some((tab: TabbedTab) => tab.id === internalActiveTabId);
    if (!activeExists) {
      internalActiveTabId = tabs[0]?.id ?? '';
      if (internalActiveTabId) {
        onTabChange?.(internalActiveTabId);
      }
    }
  });

  $effect(() => {
    if (!activeTabId) {
      return;
    }
    const controlledExists = tabs.some((tab: TabbedTab) => tab.id === activeTabId);
    if (!controlledExists) {
      return;
    }
    if (activeTabId !== internalActiveTabId) {
      internalActiveTabId = activeTabId;
    }
  });

  const selectTab = (tabId: string) => {
    internalActiveTabId = tabId;
    onTabChange?.(tabId);
  };

  const toggleTabChannelMenu = () => {
    showTabChannelMenu = !showTabChannelMenu;
  };

  const selectTabChannel = (channel: string) => {
    onTabChannelChange?.(channel);
    showTabChannelMenu = false;
  };

  const handleClickOutside = (event: MouseEvent) => {
    if (!showTabChannelMenu || !showTabChannel) {
      return;
    }
    if (
      tabChannelMenuContainer &&
      !tabChannelMenuContainer.contains(event.target as Node)
    ) {
      showTabChannelMenu = false;
    }
  };

  const handleEscape = (event: KeyboardEvent) => {
    if (event.key === "Escape") {
      showTabChannelMenu = false;
    }
  };

  onMount(() => {
    document.addEventListener("click", handleClickOutside);
    window.addEventListener("keydown", handleEscape);
  });

  onDestroy(() => {
    document.removeEventListener("click", handleClickOutside);
    window.removeEventListener("keydown", handleEscape);
  });
</script>

<div class="tabbed-container">
  <div class="tabbed-tabs">
    <div class="tabbed-tabs-left">
      {#each tabs as tab}
        <button
          class={`tabbed-tab ${internalActiveTabId === tab.id ? 'active' : ''}`}
          onclick={() => selectTab(tab.id)}
        >
          {tab.name}
        </button>
      {/each}
    </div>
    {#if showTabChannel}
      <div class="tabbed-tabs-right">
        <div
          class="mode-menu tab-channel-menu"
          bind:this={tabChannelMenuContainer}
        >
          <button class="menu-button" onclick={toggleTabChannelMenu}>
            {selectedTabChannel}
          </button>
          <DropdownList
            items={tabChannelOptions}
            show={showTabChannelMenu}
            onSelect={selectTabChannel}
            selectedItem={selectedTabChannel}
            position="bottom-right"
          />
        </div>
      </div>
    {/if}
  </div>
  <div class="tabbed-content">
    {#if activeTab}
      {#if activeTab.component}
        {@const ActiveComponent = activeTab.component}
        <ActiveComponent {...(activeTab.props ?? {})} />
      {/if}
    {/if}
  </div>
</div>
