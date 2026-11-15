<script lang="ts">
  import {
    Button,
    Breadcrumb,
    CardsView,
    CodeView,
    LineChart,
    MultiSelectDropdown,
    MultiValueSwitch,
    Properties,
    SearchBar,
    SingleSelectDropdown,
    TabSwitcher,
    Table,
    Toast,
  } from "#lib";
  import "./components-showcase.css";

  const runtimeOptions = ["Training", "Testing"];
  const cardThumbnail = "data:image/svg+xml," + encodeURIComponent(`
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 144">
      <rect width="360" height="144" fill="#17181d"/>
      <path d="M0 120h360M0 100h360M60 80l-40 64M120 80l-20 64
        M180 80v64M240 80l20 64M300 80l40 64"
        stroke="#303139" fill="none"/>
      <ellipse cx="180" cy="123" rx="55" ry="8" fill="#101013"/>
      <path d="M145 117h70M180 117V85l30-26-28-25"
        stroke="#9899a5" stroke-width="13" fill="none"
        stroke-linecap="round" stroke-linejoin="round"/>
      <g fill="#c18465" stroke="#24242a" stroke-width="3">
        <circle cx="180" cy="85" r="9"/>
        <circle cx="210" cy="59" r="9"/>
      </g>
      <path d="m182 34-12-6m12 6-4-13" stroke="#c18465"
        stroke-width="5" stroke-linecap="round"/>
    </svg>
  `);
  const codeExamples = {
    python: [
      "from dataclasses import dataclass",
      "",
      "@dataclass",
      "class TrainingConfig:",
      "    environment: str = \"Walker2d-v4\"",
      "    learning_rate: float = 0.0003",
      "    total_steps: int = 1_000_000",
      "",
      "config = TrainingConfig()",
      "print(f\"Training {config.environment}\")",
      "# A longer line to try the built-in wrap control:",
      "description = \"Train a walking policy with reproducible seeds, evaluation checkpoints, and periodic metric reporting.\"",
    ].join("\n"),
    yaml: [
      "workload:",
      "  name: walker-training",
      "  environment: Walker2d-v4",
      "  learning_rate: 0.0003",
      "  total_steps: 1000000",
      "  checkpoints:",
      "    enabled: true",
      "    interval: 10000",
    ].join("\n"),
  };
  let codeLanguage = $state("python");
  const codeLanguageOptions = [
    { label: "Python", value: "python" },
    { label: "YAML", value: "yaml" },
  ];
  const granularityOptions = ["Daily", "Weekly", "Monthly"];
  const metricOptions = ["Reward", "Loss", "Entropy", "KL"];
  const workloadViewOptions = ["All", "Training", "Testing"];
  const chartWindowOptions = ["7 days", "30 days", "90 days"];
  const chartXAxis = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const chartSeries = [
    {
      name: "Episode reward",
      values: [0.18, 0.24, 0.31, 0.38, 0.47, 0.52, 0.61],
    },
    {
      color: "#8e9de7",
      name: "Success rate",
      values: [0.11, 0.17, 0.23, 0.29, 0.36, 0.42, 0.51],
    },
  ];
  const modelOptions = [
    { label: "MuJoCo", value: "mujoco" },
    { label: "MJX", value: "mjx" },
    { label: "Isaac Lab", value: "isaac-lab" },
  ];
  const logLevelOptions = [
    { label: "Info", value: "info" },
    { label: "Warnings", value: "warnings" },
    { label: "Errors", value: "errors" },
  ];
  const workloadColumns = [
    { label: "Workload" },
    { label: "Environment" },
    {
      label: "Status",
      contentWidth: 64,
      button: {
        hasActiveState: false,
        icon: runningIcon,
        label: "Run workload",
        variant: "type-3" as const,
      },
    },
    { label: "Updated" },
  ];
  const workloadRows = [
    {
      cells: [
        { text: "TRN-1048" },
        { text: "Walker2d-v4" },
        { text: "Live" },
        { text: "2 minutes ago" },
      ],
    },
    {
      cells: [
        { text: "TRN-1039" },
        { text: "Humanoid-v5" },
        { text: "Queued — waiting for an available worker" },
        { text: "18 minutes ago" },
      ],
    },
    {
      cells: [
        { text: "EVAL-291" },
        { text: "Ant-v5" },
        { text: "Complete" },
        { text: "Yesterday" },
      ],
    },
  ] as const;
  const workloadRowControls = [
    {
      hasActiveState: false,
      icon: "view",
      label: "View",
      variant: "type-1",
    },
    {
      hasActiveState: false,
      icon: "stop",
      label: "Stop",
      variant: "type-4",
    },
  ] as const;

  let typeOneActive = $state(false);
  let typeTwoActive = $state(false);
  let typeThreeActive = $state(false);
  let typeFourActive = $state(false);
  let workloadType = $state("Training");
  let granularity = $state("Daily");
  let metric = $state("Reward");
  let workloadView = $state("All");
  let chartWindow = $state("7 days");
  let chartMetric = $state("reward");
  let model = $state("mujoco");
  let textModel = $state("mjx");
  let logLevels = $state(["info", "warnings"]);
  let borderedLogLevels = $state(["errors"]);
  let searchQuery = $state("");
  let toastOpen = $state(false);

  const showToast = (): void => {
    toastOpen = false;

    requestAnimationFrame(() => {
      toastOpen = true;
    });
  };
</script>

<svelte:head>
  <title>Component showcase</title>
  <meta
    content="Interactive Lagrangian button component examples."
    name="description"
  />
</svelte:head>

{#snippet runningIcon()}
  <svg
    aria-hidden="true"
    fill="none"
    stroke="currentColor"
    stroke-linecap="round"
    stroke-linejoin="round"
    stroke-width="1.75"
    viewBox="0 0 24 24"
  >
    <path d="m8 5 11 7-11 7Z" />
  </svg>
{/snippet}

<main class="component-showcase">
  <header class="component-showcase__header">
    <p class="component-showcase__eyebrow">Atomic components</p>
    <h1>Button</h1>
    <p>
      Select any example to inspect its persistent active state.
    </p>
  </header>

  <section aria-label="Button variants" class="button-showcase-grid">
    <article class="button-showcase-card">
      <div>
        <h2>Type-1</h2>
        <p>Text-only treatment in every state.</p>
      </div>
      <Button
        active={typeOneActive}
        aria-label="Toggle Type-1 selection"
        onclick={() => (typeOneActive = !typeOneActive)}
        variant="type-1"
      >
        <span>Type-1 action</span>
      </Button>
      <p class="button-showcase-card__state">
        {typeOneActive ? "Selected" : "Not selected"}
      </p>
    </article>

    <article class="button-showcase-card">
      <div>
        <h2>Type-2</h2>
        <p>Background appears only while selected.</p>
      </div>
      <Button
        active={typeTwoActive}
        aria-label="Toggle Type-2 selection"
        onclick={() => (typeTwoActive = !typeTwoActive)}
        variant="type-2"
      >
        <span>Type-2 action</span>
      </Button>
      <p class="button-showcase-card__state">
        {typeTwoActive ? "Selected" : "Not selected"}
      </p>
    </article>

    <article class="button-showcase-card">
      <div>
        <h2>Type-3</h2>
        <p>Background appears on hover and stays while selected.</p>
      </div>
      <Button
        active={typeThreeActive}
        aria-label="Toggle Type-3 selection"
        onclick={() => (typeThreeActive = !typeThreeActive)}
        variant="type-3"
      >
        <span>Type-3 action</span>
      </Button>
      <p class="button-showcase-card__state">
        {typeThreeActive ? "Selected" : "Not selected"}
      </p>
    </article>

    <article class="button-showcase-card">
      <div>
        <h2>Type-4</h2>
        <p>Background remains visible in every state.</p>
      </div>
      <Button
        active={typeFourActive}
        aria-label="Toggle Type-4 selection"
        onclick={() => (typeFourActive = !typeFourActive)}
        variant="type-4"
      >
        <span>Type-4 action</span>
      </Button>
      <p class="button-showcase-card__state">
        {typeFourActive ? "Selected" : "Not selected"}
      </p>
    </article>
  </section>

  <section
    aria-labelledby="card-heading"
    class="component-showcase__section"
  >
    <div class="component-showcase__section-header">
      <p class="component-showcase__eyebrow">Composite components</p>
      <h2 id="card-heading">Cards view</h2>
      <p>Cards resize and wrap to fit the available width.</p>
    </div>
    <CardsView items={[
      {
        id: "TRN-1048",
        progress: {
          value: 68,
          completionTime: "About 12 minutes remaining",
        },
        labels: ["PPO", "MuJoCo", "Walker2d-v4"],
        status: "Running",
        statusColor: "#6bbf97",
        subInfo: ["Started today, 10:21 AM", "Alex S."],
        subtitle: "Learning a stable walking policy with PPO.",
        thumbnail: cardThumbnail,
        thumbnailAlt: "Schematic robot arm on a simulation grid",
        thumbnailTitle: "walker2d.xml",
        title: "Locomotion baseline",
      },
      {
        id: "TST-2041",
        labels: ["Evaluation", "LIBERO"],
        status: "Completed",
        statusColor: "#8e9de7",
        subInfo: ["Started yesterday, 9:00 AM", "Alex S."],
        subtitle: "Evaluate the grasp policy on held-out tasks.",
        thumbnail: cardThumbnail,
        thumbnailAlt: "Schematic robot arm on a simulation grid",
        thumbnailTitle: "franka_panda.xml",
        title: "Panda policy evaluation",
      },
      {
        id: "TRN-1052",
        labels: ["SAC", "MuJoCo"],
        status: "Queued",
        statusColor: "#d4b36d",
        subInfo: ["Scheduled today, 2:30 PM", "Alex S."],
        subtitle: "Train a reaching policy with sparse rewards.",
        thumbnail: cardThumbnail,
        thumbnailAlt: "Schematic robot arm on a simulation grid",
        thumbnailTitle: "reach.xml",
        title: "Reaching experiment",
      },
    ]} />
  </section>

  <section
    aria-labelledby="properties-heading"
    class="component-showcase__section"
  >
    <div class="component-showcase__section-header">
      <p class="component-showcase__eyebrow">Composite components</p>
      <h2 id="properties-heading">Properties</h2>
      <p>Property values with optional links.</p>
    </div>
    <div class="properties-showcase">
      <Properties items={[
        { label: "Status", value: "In progress" },
        { label: "Workload ID", value: "TST-2041" },
        { label: "Type", value: "Testing" },
        { label: "Created by", value: "Alex S." },
        { label: "Started", value: "Today, 10:21 AM" },
        { label: "Compute", value: "1 × NVIDIA L40S" },
        { label: "Model", value: "franka_panda.xml" },
        { label: "Policy", value: "panda_v4.ckpt" },
        {
          label: "Documentation",
          value: "MuJoCo documentation",
          href: "https://mujoco.readthedocs.io/",
        },
      ]} />
    </div>
  </section>

  <section
    aria-labelledby="breadcrumb-heading"
    class="component-showcase__section"
  >
    <div class="component-showcase__section-header">
      <p class="component-showcase__eyebrow">Composite components</p>
      <h2 id="breadcrumb-heading">Breadcrumb</h2>
      <p>Type-1 buttons form a navigation trail.</p>
    </div>
    <Breadcrumb items={[
      { label: "Workspace", onclick: showToast },
      { label: "Workloads", onclick: showToast },
      { label: "TRN-1048" },
    ]} />
  </section>

  <section
    aria-labelledby="code-view-heading"
    class="component-showcase__section"
  >
    <div class="component-showcase__section-header">
      <p class="component-showcase__eyebrow">Composite components</p>
      <h2 id="code-view-heading">Code view</h2>
      <p>Highlighted code with copy and word-wrap controls.</p>
    </div>
    <CodeView
      code={codeLanguage === "yaml"
        ? codeExamples.yaml : codeExamples.python}
      language={codeLanguage}
      subtitle="Read only"
      title="Training configuration"
    >
      {#snippet controls()}
        <SingleSelectDropdown
          ariaLabel="Code language"
          bind:value={codeLanguage}
          options={codeLanguageOptions}
          variant="text"
        />
      {/snippet}
    </CodeView>
  </section>

  <section
    aria-labelledby="table-heading"
    class="component-showcase__section"
  >
    <div class="component-showcase__section-header">
      <p class="component-showcase__eyebrow">Composite components</p>
      <h2 id="table-heading">Table</h2>
      <p>Flexible data rows with header and row-level controls.</p>
    </div>

    <Table
      columns={workloadColumns}
      rowControls={workloadRowControls}
      rows={workloadRows}
      subtitle="3 workloads"
      title="Recent workloads"
    >
      {#snippet controls()}
        <SingleSelectDropdown
          ariaLabel="Filter workloads by simulation engine"
          bind:value={model}
          options={modelOptions}
          variant="text"
        />
        <Button hasActiveState={false} variant="type-4">
          New workload
        </Button>
      {/snippet}
    </Table>
  </section>

  <section
    aria-labelledby="charts-heading"
    class="component-showcase__section"
  >
    <div class="component-showcase__section-header">
      <p class="component-showcase__eyebrow">Composite components</p>
      <h2 id="charts-heading">Line chart</h2>
      <p>Responsive ECharts line rendering with composable controls.</p>
    </div>

    <LineChart
      singleSelectOptions={[
        { label: "Reward", value: "reward" },
        { label: "Success rate", value: "success-rate" },
      ]}
      bind:singleSelectValue={chartMetric}
      series={chartSeries}
      switchOptions={chartWindowOptions}
      bind:switchValue={chartWindow}
      title="Training performance"
      xAxis={chartXAxis}
    />
  </section>

  <section
    aria-labelledby="selection-controls-heading"
    class="component-showcase__section"
  >
    <div class="component-showcase__section-header">
      <p class="component-showcase__eyebrow">Atomic components</p>
      <h2 id="selection-controls-heading">Selection controls</h2>
      <p>
        Use the controls to inspect their selected, open, and keyboard states.
      </p>
    </div>

    <div class="selection-showcase-grid">
      <article class="selection-showcase-card">
        <div>
          <h3>Multi-value switch</h3>
          <p>Two values</p>
        </div>
        <MultiValueSwitch
          ariaLabel="Workload type"
          bind:value={workloadType}
          options={runtimeOptions}
        />
      </article>

      <article class="selection-showcase-card">
        <div>
          <h3>Tab switcher</h3>
          <p>Type-2 Buttons composed into a workload-view selector.</p>
        </div>
        <TabSwitcher
          ariaLabel="Workload view"
          bind:value={workloadView}
          options={workloadViewOptions}
        />
      </article>

      <article class="selection-showcase-card">
        <div>
          <h3>Multi-value switch</h3>
          <p>Three values</p>
        </div>
        <MultiValueSwitch
          ariaLabel="Usage granularity"
          bind:value={granularity}
          options={granularityOptions}
        />
      </article>

      <article class="selection-showcase-card">
        <div>
          <h3>Multi-value switch</h3>
          <p>Four values</p>
        </div>
        <MultiValueSwitch
          ariaLabel="Metric"
          bind:value={metric}
          options={metricOptions}
        />
      </article>

      <article class="selection-showcase-card">
        <div>
          <h3>Single-select dropdown</h3>
          <p>Bordered trigger: choose one simulation engine.</p>
        </div>
        <SingleSelectDropdown
          ariaLabel="Simulation engine"
          bind:value={model}
          options={modelOptions}
        />
      </article>

      <article class="selection-showcase-card">
        <div>
          <h3>Single-select dropdown</h3>
          <p>Text trigger: choose one simulation engine.</p>
        </div>
        <SingleSelectDropdown
          ariaLabel="Simulation engine text selector"
          bind:value={textModel}
          options={modelOptions}
          variant="text"
        />
      </article>

      <article class="selection-showcase-card">
        <div>
          <h3>Multi-select dropdown</h3>
          <p>Bordered trigger: choose one or more log levels.</p>
        </div>
        <MultiSelectDropdown
          ariaLabel="Bordered log levels"
          bind:values={borderedLogLevels}
          options={logLevelOptions}
          placeholder="All log levels"
        />
      </article>

      <article class="selection-showcase-card">
        <div>
          <h3>Multi-select dropdown</h3>
          <p>Text trigger: choose one or more log levels.</p>
        </div>
        <MultiSelectDropdown
          ariaLabel="Log levels"
          bind:values={logLevels}
          options={logLevelOptions}
          placeholder="All log levels"
          variant="text"
        />
      </article>
    </div>
  </section>

  <section
    aria-labelledby="feedback-controls-heading"
    class="component-showcase__section"
  >
    <div class="component-showcase__section-header">
      <p class="component-showcase__eyebrow">Atomic components</p>
      <h2 id="feedback-controls-heading">Search and feedback</h2>
      <p>Type a query or trigger a transient confirmation message.</p>
    </div>

    <div class="selection-showcase-grid">
      <article class="selection-showcase-card">
        <div>
          <h3>Search bar</h3>
          <p>Searches are owned by the consuming view.</p>
        </div>
        <SearchBar
          ariaLabel="Search workloads"
          bind:value={searchQuery}
          placeholder="Search workloads..."
        />
      </article>

      <article class="selection-showcase-card">
        <div>
          <h3>Toast</h3>
          <p>Dismisses automatically after 2.5 seconds.</p>
        </div>
        <Button
          hasActiveState={false}
          onclick={showToast}
          variant="type-4"
        >
          Show toast
        </Button>
      </article>
    </div>
  </section>
</main>

<Toast bind:open={toastOpen} message="Changes saved" />
