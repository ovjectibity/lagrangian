<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Viewport from './lib/viewport.svelte';
  import MenuBar from './lib/menubar.svelte';
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
  import CheckpointsViewer from './lib/checkpointsviewer.svelte';
  import type { TabbedTab } from './lib/tabbedcontainer.svelte';
  import type { WorldBody } from './commondef/model';
  import type {
    MetricsData,
    TrainingMetrics,
  } from './commondef/training';
  import type {
    ClientMessage,
    ModelForward,
  } from './commondef/messages';
  import type {
    CheckpointReference,
    TrainingExperiment,
    TrainingExperimentRun,
  } from './commondef/experiments';
  import {
    buildTrainingController,
    buildSimulationParams,
  } from './workflows/workflowgraph';

  // ── SSE handles + simulation ID ──────────────
  let trainingEventSource: EventSource | null = null;
  let simulEventSource: EventSource | null = null;
  let currentSimulId: string | null = null;

  // ── UI state ─────────────────────────────────
  let worldBody: WorldBody | undefined =
    $state(undefined);
  let simulUpdateData: ModelForward | undefined =
    $state(undefined);
  let showModeMenu = $state(false);
  let selectedMode = $state('Training');
  const modeOptions = ['Modeling', 'Training', 'Testing'];
  const logTabChannels = ['Training logs'];
  let selectedLogTabChannel = $state('Training logs');
  let experiments = $state<TrainingExperiment[]>([]);
  let experimentRunsMap =
    $state<Record<string, TrainingExperimentRun[]>>({});
  let activeExperiment =
    $state<TrainingExperiment | null>(null);
  let activeRun =
    $state<TrainingExperimentRun | null>(null);
  let isTrainingRunning = $state(false);
  let isTestingRunning = $state(false);
  let runCheckpoints = $state<CheckpointReference[]>([]);
  let allCheckpoints = $state<CheckpointReference[]>([]);
  let pendingCheckpointNodeId =
    $state<string | null>(null);

  const buildCheckpointOptions = (
    ckpts: CheckpointReference[],
  ): Array<{ label: string; value: string }> => {
    const opts: Array<{ label: string; value: string }> =
      [];
    let lastGroup = '';
    for (const ckpt of ckpts) {
      const group =
        `${ckpt.training_experiment_id}` +
        ` / ${ckpt.training_run_id}`;
      if (group !== lastGroup) {
        opts.push({
          label:
            `Exp: ${ckpt.training_experiment_id}` +
            ` / Run: ${ckpt.training_run_id}`,
          value: '',
        });
        lastGroup = group;
      }
      opts.push({ label: ckpt.id, value: ckpt.id });
    }
    return opts;
  };

  const checkpointOptions = $derived(
    buildCheckpointOptions(allCheckpoints),
  );

  const checkpointOptionsByNode = $derived<
    Record<string, Array<{ label: string; value: string }>>
  >({
    'control-model': checkpointOptions,
    'value-model': checkpointOptions,
  });

  let isTrainingActionDisabled = $derived(
    !isTrainingRunning && isTestingRunning,
  );
  let isTestingActionDisabled = $derived(
    !isTestingRunning && isTrainingRunning,
  );
  let selectedEditorTabId = $state('model-editor');
  let trainingLogs = $state<string[]>([]);
  type TrainingMetricType = TrainingMetrics['type'];
  type TrainingMetricsByType =
    Record<TrainingMetricType, MetricsData>;

  const createEmptyMetricsData = (): MetricsData => ({
    x_axis: [],
    y_axis: [],
  });

  const createEmptyTrainingMetricsByType =
    (): TrainingMetricsByType => ({
      MeanReward: createEmptyMetricsData(),
      MeanPolicyLoss: createEmptyMetricsData(),
      ValueLoss: createEmptyMetricsData(),
      MeanEntropy: createEmptyMetricsData(),
      PolicyGradientNorm: createEmptyMetricsData(),
      ValueGradientNorm: createEmptyMetricsData(),
    });

  let trainingMetricsByType =
    $state<TrainingMetricsByType>(
      createEmptyTrainingMetricsByType(),
    );

  const appendTrainingMetrics = (
    metrics: TrainingMetrics[],
  ): void => {
    const next: TrainingMetricsByType = {
      MeanReward: {
        x_axis: [
          ...trainingMetricsByType.MeanReward.x_axis,
        ],
        y_axis: [
          ...trainingMetricsByType.MeanReward.y_axis,
        ],
      },
      MeanPolicyLoss: {
        x_axis: [
          ...trainingMetricsByType.MeanPolicyLoss.x_axis,
        ],
        y_axis: [
          ...trainingMetricsByType.MeanPolicyLoss.y_axis,
        ],
      },
      ValueLoss: {
        x_axis: [
          ...trainingMetricsByType.ValueLoss.x_axis,
        ],
        y_axis: [
          ...trainingMetricsByType.ValueLoss.y_axis,
        ],
      },
      MeanEntropy: {
        x_axis: [
          ...trainingMetricsByType.MeanEntropy.x_axis,
        ],
        y_axis: [
          ...trainingMetricsByType.MeanEntropy.y_axis,
        ],
      },
      PolicyGradientNorm: {
        x_axis: [
          ...trainingMetricsByType
            .PolicyGradientNorm.x_axis,
        ],
        y_axis: [
          ...trainingMetricsByType
            .PolicyGradientNorm.y_axis,
        ],
      },
      ValueGradientNorm: {
        x_axis: [
          ...trainingMetricsByType
            .ValueGradientNorm.x_axis,
        ],
        y_axis: [
          ...trainingMetricsByType
            .ValueGradientNorm.y_axis,
        ],
      },
    };
    for (const metric of metrics) {
      const existing = next[metric.type];
      next[metric.type] = {
        x_axis: [
          ...existing.x_axis,
          ...metric.content.x_axis,
        ],
        y_axis: [
          ...existing.y_axis,
          ...metric.content.y_axis,
        ],
      };
    }
    trainingMetricsByType = next;
  };

  const appendTrainingLogs = (logs: string[]): void => {
    if (logs.length === 0) return;
    trainingLogs = [...trainingLogs, ...logs];
  };

  function openRewardEditor() {
    selectedEditorTabId = 'reward-editor';
  }

  // ── SSE stream openers ────────────────────────

  function openTrainingStream(
    expId: string,
    runId: string,
  ): void {
    trainingEventSource?.close();
    const url =
      '/api/training/experiments' +
      `/${expId}/runs/${runId}/updates`;
    trainingEventSource = new EventSource(url);
    trainingEventSource.onmessage = (event) => {
      try {
        const msg: ClientMessage = JSON.parse(
          event.data,
        );
        if (msg.type === 'TrainingUpdatesMessage') {
          for (const update of msg.content.updates) {
            appendTrainingMetrics(update.metrics);
            appendTrainingLogs(update.logs);
            if (update.checkpoints.length > 0) {
              runCheckpoints = [
                ...runCheckpoints,
                ...update.checkpoints,
              ];
            }
          }
        } else if (
          msg.type === 'StopTrainingMessage'
        ) {
          isTrainingRunning = false;
          trainingEventSource?.close();
          trainingEventSource = null;
        }
      } catch (e) {
        console.error('Training SSE parse error:', e);
      }
    };
    trainingEventSource.onerror = () => {
      isTrainingRunning = false;
      trainingEventSource?.close();
      trainingEventSource = null;
    };
  }

  function openSimulationStream(id: string): void {
    simulEventSource?.close();
    simulEventSource = new EventSource(
      `/api/testing/simulations/${id}/updates`,
    );
    simulEventSource.onmessage = (event) => {
      const bytes = Uint8Array.from(
        atob(event.data as string),
        (c) => c.charCodeAt(0),
      );
      const buffer = new DataView(bytes.buffer);
      let offset = 0;
      const messageType = buffer.getUint8(offset);
      offset += 1;
      if (messageType === 7) {
        const idLen = buffer.getUint16(offset, true);
        offset += 2;
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
      }
    };
    simulEventSource.onerror = () => {
      isTestingRunning = false;
      simulEventSource?.close();
      simulEventSource = null;
      currentSimulId = null;
    };
  }

  // ── Checkpoint actions ────────────────────────

  async function onCheckpointTypeSelected(
    nodeId: string,
  ): Promise<void> {
    pendingCheckpointNodeId = nodeId;
    try {
      const res = await fetch('/api/checkpoints');
      if (!res.ok) return;
      const data = await res.json() as {
        checkpoints: CheckpointReference[];
      };
      allCheckpoints = data.checkpoints;
    } catch (e) {
      console.error('fetchCheckpoints failed:', e);
    }
  }

  async function onCheckpointSelected(
    nodeId: string,
    checkpointId: string,
  ): Promise<void> {
    pendingCheckpointNodeId = nodeId;
    try {
      const res = await fetch(
        `/api/checkpoints/${checkpointId}`,
      );
      if (!res.ok) return;
      const data = await res.json() as {
        checkpoint: CheckpointReference;
      };
      const ckpt = data.checkpoint;
      if (nodeId && ckpt.control_model_params) {
        const {
          num_hidden_layers,
          hidden_layer_dim,
          output_dim,
          input_dim,
        } = ckpt.control_model_params;
        import('./workflows/workflowgraph').then(
          ({ TrainingGraphNodes }) => {
            const node = TrainingGraphNodes.find(
              (n) => n.id === nodeId,
            );
            if (node?.settings) {
              for (const s of node.settings) {
                if (s.key === 'num_hidden_layers')
                  s.value = num_hidden_layers;
                else if (s.key === 'hidden_layer_dim')
                  s.value = hidden_layer_dim;
                else if (s.key === 'output_dim')
                  s.value = output_dim;
                else if (s.key === 'input_dim')
                  s.value = input_dim;
              }
            }
          },
        );
      }
      pendingCheckpointNodeId = null;
    } catch (e) {
      console.error('fetchCheckpoint failed:', e);
    }
  }

  // ── Experiment / run actions ──────────────────

  async function createExperiment(): Promise<void> {
    try {
      const res = await fetch(
        '/api/training/experiments',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({}),
        },
      );
      if (!res.ok) return;
      const data = await res.json() as {
        experiment: TrainingExperiment;
      };
      activeExperiment = data.experiment;
      experiments = [...experiments, activeExperiment];
      activeRun = null;
    } catch (e) {
      console.error('createExperiment failed:', e);
    }
  }

  async function createRun(): Promise<void> {
    if (!activeExperiment) return;
    const eid =
      activeExperiment.training_experiment_id;
    try {
      const res = await fetch(
        `/api/training/experiments/${eid}/runs`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({}),
        },
      );
      if (!res.ok) return;
      const data = await res.json() as {
        run: TrainingExperimentRun;
      };
      activeRun = data.run;
      const curRuns = experimentRunsMap[eid] ?? [];
      experimentRunsMap = {
        ...experimentRunsMap,
        [eid]: [...curRuns, activeRun],
      };
    } catch (e) {
      console.error('createRun failed:', e);
    }
  }

  async function loadRun(
    expId: string,
    runId: string,
  ): Promise<void> {
    try {
      const base =
        `/api/training/experiments/${expId}` +
        `/runs/${runId}`;
      const [runRes, ckptRes] = await Promise.all([
        fetch(base),
        fetch(`${base}/checkpoints`),
      ]);
      if (!runRes.ok) return;
      const runData = await runRes.json() as {
        run: TrainingExperimentRun;
      };
      activeRun = runData.run;
      activeExperiment =
        experiments.find(
          (e) =>
            e.training_experiment_id === expId,
        ) ?? null;
      trainingMetricsByType =
        createEmptyTrainingMetricsByType();
      trainingLogs = [];
      runCheckpoints = [];
      if (ckptRes.ok) {
        const ckptData = await ckptRes.json() as {
          checkpoints: CheckpointReference[];
        };
        runCheckpoints = ckptData.checkpoints;
      }
    } catch (e) {
      console.error('loadRun failed:', e);
    }
  }

  function setCurrentExperiment(
    experiment: TrainingExperiment,
  ): void {
    activeExperiment = experiment;
    activeRun = null;
  }

  let editorTabs = $derived<TabbedTab[]>([
    ...(selectedMode === 'Testing'
      ? [
          {
            id: 'testing-graph',
            name: 'Testing setup',
            component: TestingGraphEditor,
          } satisfies TabbedTab,
        ]
      : [
          {
            id: 'training-experiments',
            name: 'Experiments',
            component: TrainingExperimentsTable,
            props: {
              experiments,
              experimentRunsMap,
              activeRunId: activeRun?.id,
              onLoadRun: loadRun,
              onSetCurrent: setCurrentExperiment,
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
        simulUpdateData,
      },
    },
    ...(selectedMode === 'Testing'
      ? []
      : [
          {
            id: 'training-graph',
            name: 'Training setup',
            component: TrainingGraphEditor,
            props: {
              onOpenRewardEditor: openRewardEditor,
              checkpointOptionsByNode,
              onCheckpointTypeSelected,
              onCheckpointSelected,
            },
          } satisfies TabbedTab,
          {
            id: 'reward-editor',
            name: 'Reward editor',
            component: RewardEditor,
          } satisfies TabbedTab,
          {
            id: 'training-runtime-dashboard',
            name: 'Training dashboard',
            component: TrainingRuntimeDashboard,
            props: {
              metricsByType: trainingMetricsByType,
            },
          } satisfies TabbedTab,
          {
            id: 'checkpoints',
            name: 'Checkpoints',
            component: CheckpointsViewer,
            props: {
              checkpoints: runCheckpoints,
            },
          } satisfies TabbedTab,
        ]),
  ]);

  let logTabs = $derived<TabbedTab[]>([
    {
      id: 'logs',
      name: 'Logs',
      component: Logger,
      props: {
        logs: trainingLogs,
      },
    },
  ]);

  let modeMenuContainer =
    $state<HTMLDivElement | undefined>(undefined);

  const toggleModeMenu = () => {
    showModeMenu = !showModeMenu;
  };

  const selectMode = (mode: string) => {
    selectedMode = mode;
    showModeMenu = false;
    selectedEditorTabId = 'model-editor';
  };

  // ── Model ─────────────────────────────────────

  const loadModelRuntime = async (): Promise<void> => {
    try {
      const res = await fetch('/api/models/default');
      if (!res.ok) return;
      const data = await res.json() as {
        id: string;
        model: WorldBody;
      };
      worldBody = data.model;
      simulUpdateData = undefined;
    } catch (e) {
      console.error('loadModelRuntime failed:', e);
    }
  };

  // ── Training ──────────────────────────────────

  const startTraining = async (): Promise<void> => {
    if (isTestingRunning) return;

    // Auto-create experiment if none is active
    if (!activeExperiment) {
      const res = await fetch(
        '/api/training/experiments',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({}),
        },
      );
      if (!res.ok) return;
      const d = await res.json() as {
        experiment: TrainingExperiment;
      };
      activeExperiment = d.experiment;
      experiments = [...experiments, activeExperiment];
      activeRun = null;
    }

    // Auto-create run if none is active
    if (!activeRun) {
      const eid =
        activeExperiment.training_experiment_id;
      const res = await fetch(
        `/api/training/experiments/${eid}/runs`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({}),
        },
      );
      if (!res.ok) return;
      const d = await res.json() as {
        run: TrainingExperimentRun;
      };
      activeRun = d.run;
      const curRuns =
        experimentRunsMap[eid] ?? [];
      experimentRunsMap = {
        ...experimentRunsMap,
        [eid]: [...curRuns, activeRun],
      };
    }

    const eid =
      activeExperiment.training_experiment_id;
    const rid = activeRun.id;
    trainingMetricsByType =
      createEmptyTrainingMetricsByType();
    trainingLogs = [];

    try {
      const startUrl =
        '/api/training/experiments' +
        `/${eid}/runs/${rid}/start`;
      const res = await fetch(startUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          training_graph: {
            training_job_controller:
              buildTrainingController(),
          },
        }),
      });
      if (!res.ok) return;
      isTrainingRunning = true;
      openTrainingStream(eid, rid);
    } catch (e) {
      console.error('startTraining failed:', e);
    }
  };

  const stopTraining = async (): Promise<void> => {
    if (!activeExperiment || !activeRun) return;
    const eid =
      activeExperiment.training_experiment_id;
    const rid = activeRun.id;
    try {
      await fetch(
        '/api/training/experiments' +
        `/${eid}/runs/${rid}/stop`,
        { method: 'POST' },
      );
    } catch (e) {
      console.error('stopTraining failed:', e);
    }
    isTrainingRunning = false;
    trainingEventSource?.close();
    trainingEventSource = null;
  };

  const toggleTrainingState =
    async (): Promise<void> => {
      if (isTrainingRunning) {
        await stopTraining();
        return;
      }
      await startTraining();
    };

  // ── Testing ───────────────────────────────────

  const startTesting = async (): Promise<void> => {
    if (isTrainingRunning) return;
    try {
      const res = await fetch(
        '/api/testing/simulations',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            simulation_params: buildSimulationParams(),
          }),
        },
      );
      if (!res.ok) return;
      const data = await res.json() as { id: string };
      currentSimulId = data.id;
      isTestingRunning = true;
      openSimulationStream(currentSimulId);
    } catch (e) {
      console.error('startTesting failed:', e);
    }
  };

  const stopTesting = async (): Promise<void> => {
    if (!currentSimulId) return;
    try {
      await fetch(
        `/api/testing/simulations/${currentSimulId}/stop`,
        { method: 'POST' },
      );
    } catch (e) {
      console.error('stopTesting failed:', e);
    }
    isTestingRunning = false;
    simulEventSource?.close();
    simulEventSource = null;
    currentSimulId = null;
  };

  const toggleTestingState =
    async (): Promise<void> => {
      if (isTestingRunning) {
        await stopTesting();
        return;
      }
      await startTesting();
    };

  // ── Global event handlers ─────────────────────

  const handleGlobalClick = (event: MouseEvent) => {
    if (!showModeMenu) return;
    if (
      modeMenuContainer &&
      !modeMenuContainer.contains(
        event.target as Node,
      )
    ) {
      showModeMenu = false;
    }
  };

  const handleGlobalKeyDown = (
    event: KeyboardEvent,
  ) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      showModeMenu = false;
    }
  };

  // ── Startup data load ─────────────────────────

  const loadInitialData = async (): Promise<void> => {
    try {
      const res = await fetch(
        '/api/training/experiments',
      );
      if (!res.ok) return;
      const data = await res.json() as {
        experiments: TrainingExperiment[];
      };
      experiments = data.experiments;
      const runsMap: Record<
        string,
        TrainingExperimentRun[]
      > = {};
      await Promise.all(
        data.experiments.map(async (exp) => {
          const eid = exp.training_experiment_id;
          const runsRes = await fetch(
            `/api/training/experiments/${eid}/runs`,
          );
          if (!runsRes.ok) return;
          const runsData = await runsRes.json() as {
            runs: TrainingExperimentRun[];
          };
          runsMap[eid] = runsData.runs;
        }),
      );
      experimentRunsMap = runsMap;
    } catch (e) {
      console.error('loadInitialData failed:', e);
    }
  };

  onMount(() => {
    loadModelRuntime().catch(console.error);
    loadInitialData().catch(console.error);
    document.addEventListener(
      'click',
      handleGlobalClick,
    );
    window.addEventListener(
      'keydown',
      handleGlobalKeyDown,
    );
    return () => {
      trainingEventSource?.close();
      simulEventSource?.close();
      document.removeEventListener(
        'click',
        handleGlobalClick,
      );
      window.removeEventListener(
        'keydown',
        handleGlobalKeyDown,
      );
    };
  });

  onDestroy(() => {
    trainingEventSource?.close();
    simulEventSource?.close();
    document.removeEventListener(
      'click',
      handleGlobalClick,
    );
    window.removeEventListener(
      'keydown',
      handleGlobalKeyDown,
    );
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
