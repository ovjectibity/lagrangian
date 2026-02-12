<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Viewport from './lib/viewport.svelte';
  import MenuBar from './lib/menubar.svelte';
  import DropdownList from './lib/dropdownlist.svelte';
  import TabbedContainer from './lib/tabbedcontainer.svelte';
  import Logger from './lib/logger.svelte';
  import TrainingGraphEditor from './lib/traininggrapheditor.svelte';
  import PlayPauseCta from './lib/playpausecta.svelte';
  import RewardEditor from './lib/rewardeditor.svelte';
  import TrainingRuntimeDashboard from './lib/trainingruntimedashboard.svelte';
  import type { TabbedTab } from './lib/tabbedcontainer.svelte';
  import type { WorldBody } from './common/types';
  import type { ClientMessage, ModelForward } from "./common/messages";

  let websocket: WebSocket | null = null;
  let worldBody: WorldBody | undefined = $state(undefined);
  let simulUpdateData: ModelForward | undefined = $state(undefined);
  let showModeMenu = $state(false);
  let selectedMode = $state('Modeling');
  const modeOptions = ['Modeling', 'Training'];
  const logTabChannels = ['Training logs'];
  let selectedLogTabChannel = $state('Training logs');
  let isTrainingRunning = $state(false);
  let selectedEditorTabId = $state('model-editor');

  function openRewardEditor() {
    selectedEditorTabId = 'reward-editor';
  }

  let editorTabs = $derived<TabbedTab[]>([
    {
      id: 'model-editor',
      name: 'Model Editor',
      component: Viewport,
      props: {
        width: 800,
        height: 500,
        worldBody,
        simulUpdateData
      }
    },
    {
      id: 'training-graph',
      name: 'Training setup',
      component: TrainingGraphEditor,
      props: {
        onOpenRewardEditor: openRewardEditor
      }
    },
    {
      id: 'reward-editor',
      name: 'Reward editor',
      component: RewardEditor
    },
    {
      id: 'training-runtime-dashboard',
      name: 'Training runtime dashboard',
      component: TrainingRuntimeDashboard
    }
  ]);
  let logTabs = $derived<TabbedTab[]>([
    {
      id: 'logs',
      name: 'Logs',
      component: Logger
    }
  ]);
  let modeMenuContainer: HTMLDivElement;

  const toggleModeMenu = () => {
    showModeMenu = !showModeMenu;
  };

  const selectMode = (mode: string) => {
    selectedMode = mode;
    showModeMenu = false;
    if (mode !== 'Training') {
      isTrainingRunning = false;
    }
  };

  const toggleTrainingState = () => {
    isTrainingRunning = !isTrainingRunning;
  };

  const handleGlobalClick = (event: MouseEvent) => {
    if (!showModeMenu) {
      return;
    }
    if (modeMenuContainer && !modeMenuContainer.contains(event.target as Node)) {
      showModeMenu = false;
    }
  };

  const handleGlobalKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      showModeMenu = false;
    }
  };

  const handleKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'l' && websocket && websocket.readyState === WebSocket.OPEN) {
      console.log("Sending 'get_test_model' message...");
      websocket.send("get_test_model");
    } else if (event.key === 's' && websocket && websocket.readyState === WebSocket.OPEN) {
      console.log("Initiating simualtion loop...");
      websocket.send("run_simul");
    } else if (event.key === 'g' && websocket && websocket.readyState === WebSocket.OPEN) {
      console.log("Getting test model 2...");
      websocket.send("get_test_model_2");
    } else if (event.key === 'r' && websocket && websocket.readyState === WebSocket.OPEN) {
      console.log("Getting test model 3...");
      websocket.send("get_test_model_3");
    }
  };

  onMount(() => {
    websocket = new WebSocket("ws://localhost:8000/api/viewport");
    websocket.binaryType = 'arraybuffer'; // Important for receiving binary data

    websocket.onopen = () => {
      console.log("WebSocket connection established.");
    };

    websocket.onmessage = (event) => {
      if (typeof event.data === 'string') {
        // Handle JSON messages
        try {
          const clientMessage: ClientMessage = JSON.parse(event.data);
          if (clientMessage.type === "SetModelMessage") {
            worldBody = clientMessage.content.set_model;
            simulUpdateData = undefined; // Clear simulation data on new model load
          } else if (clientMessage.type === "LoadModelMessage") {
            worldBody = clientMessage.content.model;
            simulUpdateData = undefined; // Clear simulation data on new model load
          }
        } catch (e) {
          console.error("Failed to parse WebSocket JSON message:", e);
        }
      } else if (event.data instanceof ArrayBuffer) {
        // Handle binary messages
        const buffer = new DataView(event.data);
        let offset = 0;

        const messageType = buffer.getUint8(offset);
        offset += 1;

        if (messageType === 7) { // Hardcoded type for SimulUpdateMessage
          const idLen = buffer.getUint16(offset, true); // true for little-endian
          offset += 2;
          const idBytes = new Uint8Array(event.data, offset, idLen);
          const id = new TextDecoder().decode(idBytes);
          offset += idLen;

          const nbody = buffer.getUint16(offset, true);
          offset += 2;

          const xpos: number[] = [];
          for (let i = 0; i < nbody * 3; i++) {
            xpos.push(buffer.getFloat64(offset, true));
            offset += 8;
          }

          const xquat: number[] = [];
          for (let i = 0; i < nbody * 4; i++) {
            xquat.push(buffer.getFloat64(offset, true));
            offset += 8;
          }

          simulUpdateData = { xpos, xquat };
        } else {
          console.warn("Received unknown binary message type:", messageType);
        }
      }
    };

    websocket.onclose = (event) => {
      console.log("WebSocket connection closed:", event);
      websocket = null;
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    window.addEventListener('keydown', handleKeyDown);
    document.addEventListener('click', handleGlobalClick);
    window.addEventListener('keydown', handleGlobalKeyDown);

    return () => {
      if (websocket) {
        websocket.close();
      }
      window.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('click', handleGlobalClick);
      window.removeEventListener('keydown', handleGlobalKeyDown);
    };
  });

  onDestroy(() => {
    if (websocket) {
        websocket.close();
    }
    window.removeEventListener('keydown', handleKeyDown);
    document.removeEventListener('click', handleGlobalClick);
    window.removeEventListener('keydown', handleGlobalKeyDown);
  });
</script>

<div class="main-container">
  <div class="topbar-container">
    <div class="mode-menu" bind:this={modeMenuContainer}>
      <button class="menu-button" onclick={toggleModeMenu}>
        {selectedMode}
      </button>
      <DropdownList
        items={modeOptions}
        show={showModeMenu}
        onSelect={selectMode}
        selectedItem={selectedMode}
        position="bottom-left"
      />
    </div>
    {#if selectedMode === 'Training'}
      <PlayPauseCta isRunning={isTrainingRunning} onToggle={toggleTrainingState} />
    {/if}
    <MenuBar/>
  </div>
  <!-- <div class="views-container"> -->
  <div class="editors-tabs-pane">
    <TabbedContainer
      tabs={editorTabs}
      activeTabId={selectedEditorTabId}
      onTabChange={(tabId) => {
        selectedEditorTabId = tabId;
      }}
    />
  </div>
  <div class="logs-tabs-pane">
    <TabbedContainer
      tabs={logTabs}
      showTabChannel={true}
      tabChannelOptions={logTabChannels}
      selectedTabChannel={selectedLogTabChannel}
      onTabChannelChange={(channel) => {
        selectedLogTabChannel = channel;
      }}
    />
  </div>
  <!-- </div> -->
</div>
