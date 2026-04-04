<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Viewport from './lib/viewport.svelte';
  import MenuBar from './lib/menubar.svelte';
  import DropdownList from './lib/dropdownlist.svelte';
  import ModeSelector from './lib/modeselector.svelte';
  import TabbedContainer from './lib/tabbedcontainer.svelte';
  import Logger from './lib/logger.svelte';
  import TrainingGraphEditor from './lib/traininggrapheditor.svelte';
  import TestingGraphEditor from './lib/testinggrapheditor.svelte';
  import PlayPauseCta from './lib/playpausecta.svelte';
  import Breadcrumb from './lib/breadcrumb.svelte';
  import RewardEditor from './lib/rewardeditor.svelte';
  import TrainingRuntimeDashboard from './lib/trainingruntimedashboard.svelte';
  import TrainingExperimentsTable from './lib/trainingexperimentstable.svelte';
  import type { TabbedTab } from './lib/tabbedcontainer.svelte';
  import type { WorldBody } from './commondef/model';
  import type {
    MetricsData,
    TrainingMetrics,
  } from './commondef/training';
  import type {
    ClientMessage,
    ModelForward,
    TrainingExperiment,
    TrainingExperimentRun,
    ServerMessage,
  } from './commondef/messages';
  import { buildTrainingController,
    buildSimulationParams
   } from './workflows/workflowgraph';

  let websocket: WebSocket | null = null;
  const trainingTaskId = 'training-task-1';
  const testingTaskId = 'testing-task-1';
  let worldBody: WorldBody | undefined = $state(undefined);
  let simulUpdateData: ModelForward | undefined = $state(undefined);
  let showModeMenu = $state(false);
  let selectedMode = $state('Training');
  const modeOptions = ['Modeling', 'Training', 'Testing'];
  const logTabChannels = ['Training logs'];
  let selectedLogTabChannel = $state('Training logs');
  let activeExperiment = $state<TrainingExperiment | null>(null);
  let activeExperimentRuns = $state<TrainingExperimentRun[]>([]);
  let activeRun = $state<TrainingExperimentRun | null>(null);
  let trainingGraphEdited = $state(false);
  let isTrainingRunning = $state(false);
  let isTestingRunning = $state(false);
  let isTrainingActionDisabled = $derived(
    !isTrainingRunning && isTestingRunning
  );
  let isTestingActionDisabled = $derived(
    !isTestingRunning && isTrainingRunning
  );
  let selectedEditorTabId = $state('model-editor');
  let trainingLogs = $state<string[]>([]);
  type TrainingMetricType = TrainingMetrics['type'];
  type TrainingMetricsByType = Record<TrainingMetricType, MetricsData>;

  const createEmptyMetricsData = (): MetricsData => ({ x_axis: [], y_axis: [] });

  const createEmptyTrainingMetricsByType = (): TrainingMetricsByType => ({
    MeanReward: createEmptyMetricsData(),
    MeanPolicyLoss: createEmptyMetricsData(),
    ValueLoss: createEmptyMetricsData(),
    MeanEntropy: createEmptyMetricsData(),
    PolicyGradientNorm: createEmptyMetricsData(),
    ValueGradientNorm: createEmptyMetricsData(),
  });

  let trainingMetricsByType = $state<TrainingMetricsByType>(
    createEmptyTrainingMetricsByType(),
  );

  const appendTrainingMetrics = (metrics: TrainingMetrics[]): void => {
    const nextMetricsByType: TrainingMetricsByType = {
      MeanReward: {
        x_axis: [...trainingMetricsByType.MeanReward.x_axis],
        y_axis: [...trainingMetricsByType.MeanReward.y_axis],
      },
      MeanPolicyLoss: {
        x_axis: [...trainingMetricsByType.MeanPolicyLoss.x_axis],
        y_axis: [...trainingMetricsByType.MeanPolicyLoss.y_axis],
      },
      ValueLoss: {
        x_axis: [...trainingMetricsByType.ValueLoss.x_axis],
        y_axis: [...trainingMetricsByType.ValueLoss.y_axis],
      },
      MeanEntropy: {
        x_axis: [...trainingMetricsByType.MeanEntropy.x_axis],
        y_axis: [...trainingMetricsByType.MeanEntropy.y_axis],
      },
      PolicyGradientNorm: {
        x_axis: [
          ...trainingMetricsByType
            .PolicyGradientNorm
            .x_axis
        ],
        y_axis: [
          ...trainingMetricsByType
            .PolicyGradientNorm
            .y_axis
        ],
      },
      ValueGradientNorm: {
        x_axis: [
          ...trainingMetricsByType
            .ValueGradientNorm
            .x_axis
        ],
        y_axis: [
          ...trainingMetricsByType
            .ValueGradientNorm
            .y_axis
        ],
      },
    };
    for (const metric of metrics) {
      const existing = nextMetricsByType[metric.type];
      nextMetricsByType[metric.type] = {
        x_axis: [...existing.x_axis, ...metric.content.x_axis],
        y_axis: [...existing.y_axis, ...metric.content.y_axis],
      };
    }
    trainingMetricsByType = nextMetricsByType;
  };

  const appendTrainingLogs = (logs: string[]): void => {
    if (logs.length === 0) {
      return;
    }
    trainingLogs = [...trainingLogs, ...logs];
  };

  function openRewardEditor() {
    selectedEditorTabId = 'reward-editor';
  }

  function createExperiment() {
    sendServerMessage({ type: 'CreateExperimentMessage' });
  }

  function createRun() {
    if (!activeExperiment) return;
    sendServerMessage({
      type: 'CreateRunMessage',
      content: {
        training_experiment_id:
          activeExperiment.training_experiment_id,
      },
    });
  }

  function loadRun(expId: string, runId: string) {
    sendServerMessage({
      type: 'LoadRunMessage',
      content: {
        training_experiment_id: expId,
        run_id: runId,
      },
    });
  }

  function startNewRun() {
    createRun();
  }

  let editorTabs = $derived<TabbedTab[]>([
    ...(selectedMode === 'Testing'
      ? [
          {
            id: 'testing-graph',
            name: 'Testing setup',
            component: TestingGraphEditor
          } satisfies TabbedTab
        ]
      : [
          {
            id: 'training-experiments',
            name: 'Experiments',
            component: TrainingExperimentsTable,
            props: {
              experiments: activeExperiment
                ? [activeExperiment]
                : [],
              runs: activeExperimentRuns,
              activeRunId: activeRun?.id,
              onLoadRun: loadRun,
            },
          } satisfies TabbedTab,
        ]),
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
    ...(selectedMode === 'Testing'
      ? []
      : [
          {
            id: 'training-graph',
            name: 'Training setup',
            component: TrainingGraphEditor,
            props: {
              onOpenRewardEditor: openRewardEditor
            }
          } satisfies TabbedTab,
          {
            id: 'reward-editor',
            name: 'Reward editor',
            component: RewardEditor
          } satisfies TabbedTab,
          {
            id: 'training-runtime-dashboard',
            name: 'Training dashboard',
            component: TrainingRuntimeDashboard,
            props: {
              metricsByType: trainingMetricsByType
            }
          } satisfies TabbedTab
        ])
  ]);
  let logTabs = $derived<TabbedTab[]>([
    {
      id: 'logs',
      name: 'Logs',
      component: Logger,
      props: {
        logs: trainingLogs
      }
    }
  ]);
  let modeMenuContainer = $state<HTMLDivElement | undefined>(undefined);

  const toggleModeMenu = () => {
    showModeMenu = !showModeMenu;
  };

  const selectMode = (mode: string) => {
    selectedMode = mode;
    showModeMenu = false;
    selectedEditorTabId = 'model-editor';
  };

  const sendServerMessage = (message: ServerMessage): boolean => {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket is not connected. Could not send message:', message.type);
      return false;
    }
    websocket.send(JSON.stringify(message));
    return true;
  };

  const loadModelRuntime = () => {
    sendServerMessage({
      type: 'LoadModelMessage'
    });
  };

  const startTraining = () => {
    if (isTestingRunning) {
      return;
    }
    const wasSent = sendServerMessage({
      type: 'StartTrainingMessage',
      content: {
        id: trainingTaskId,
        training_graph: {
          training_job_controller: buildTrainingController(),
        },
      },
    });
    if (wasSent) {
      trainingMetricsByType = createEmptyTrainingMetricsByType();
      trainingLogs = [];
      isTrainingRunning = true;
    }
  };

  const stopTraining = () => {
    const wasSent = sendServerMessage({
      type: 'StopTrainingMessage',
      content: { id: trainingTaskId },
    });
    if (wasSent) {
      isTrainingRunning = false;
    }
  };

  const toggleTrainingState = () => {
    if (isTrainingRunning) {
      stopTraining();
      return;
    }
    startTraining();
  };


  const startTesting = () => {
    if (isTrainingRunning) {
      return;
    }
    const wasSent = sendServerMessage({
      type: 'StartSimulationMessage',
      content: {
        id: testingTaskId,
        simulation_params: buildSimulationParams(),
      },
    });
    if (wasSent) {
      isTestingRunning = true;
    }
  };

  const stopTesting = () => {
    const wasSent = sendServerMessage({
      type: 'StopSimulationMessage',
      content: { id: testingTaskId },
    });
    if (wasSent) {
      isTestingRunning = false;
    }
  };

  const toggleTestingState = () => {
    if (isTestingRunning) {
      stopTesting();
      return;
    }
    startTesting();
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
    websocket = new WebSocket("ws://localhost:8000/api/workloads");
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
          } else if (clientMessage.type === "SimulUpdateMessage") {
            const latestUpdate =
              clientMessage.content.updates[clientMessage.content.updates.length - 1];
            if (latestUpdate) {
              simulUpdateData = latestUpdate;
            }
          } else if (clientMessage.type === 'TrainingUpdatesMessage') {
            if (clientMessage.content.id === trainingTaskId) {
              for (const update of clientMessage.content.updates) {
                appendTrainingMetrics(update.metrics);
                appendTrainingLogs(update.logs);
              }
            }
          } else if (clientMessage.type === 'StopTrainingMessage') {
            isTrainingRunning = false;
          } else if (clientMessage.type === 'ActiveContextMessage') {
            activeExperiment = clientMessage.content.experiment;
            activeRun = clientMessage.content.run;
            if (activeRun) {
              activeExperimentRuns = [activeRun];
            }
          } else if (clientMessage.type === 'ExperimentCreatedMessage') {
            activeExperiment = clientMessage.content.experiment;
            activeRun = null;
            activeExperimentRuns = [];
          } else if (clientMessage.type === 'RunCreatedMessage') {
            activeRun = clientMessage.content.run;
            activeExperimentRuns = [...activeExperimentRuns, activeRun];
          } else if (clientMessage.type === 'RunLoadedMessage') {
            activeRun = clientMessage.content.run;
            trainingMetricsByType = createEmptyTrainingMetricsByType();
            trainingLogs = [];
          } else if (clientMessage.type === 'StopSimulationMessage') {
            if (clientMessage.content.id === testingTaskId) {
              isTestingRunning = false;
            }
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
      isTrainingRunning = false;
      isTestingRunning = false;
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
    <ModeSelector
      options={modeOptions}
      selected={selectedMode}
      onSelect={selectMode}
    />
    {#if selectedMode === 'Training'}
      <Breadcrumb
        experimentName={activeExperiment?.experiment_name}
        runName={activeRun?.id}
      />
      <PlayPauseCta
        isRunning={isTrainingRunning}
        isDisabled={isTrainingActionDisabled}
        onToggle={toggleTrainingState}
        startLabel="Start training"
        stopLabel="Stop training"
        showNewRunOption={trainingGraphEdited && !isTrainingRunning}
        onStartNewRun={startNewRun}
      />
    {:else if selectedMode === 'Testing'}
      <PlayPauseCta
        isRunning={isTestingRunning}
        isDisabled={isTestingActionDisabled}
        onToggle={toggleTestingState}
        startLabel="Run simulation"
        stopLabel="Stop simulation"
      />
    {/if}
    <MenuBar
      onImport={loadModelRuntime}
      showExperimentsMenu={selectedMode === 'Training'}
      onCreateExperiment={createExperiment}
      onCreateRun={createRun}
    />
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
