<script lang="ts">
  import {
    Button,
    MultiSelectDropdown,
    MultiValueSwitch,
    SearchBar,
    SingleSelectDropdown,
    Toast,
  } from "#lib";
  import "./components.css";

  const runtimeOptions = ["Training", "Testing"];
  const granularityOptions = ["Daily", "Weekly", "Monthly"];
  const metricOptions = ["Reward", "Loss", "Entropy", "KL"];
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

  let typeOneActive = $state(false);
  let typeTwoActive = $state(false);
  let typeThreeActive = $state(false);
  let typeFourActive = $state(false);
  let workloadType = $state("Training");
  let granularity = $state("Daily");
  let metric = $state("Reward");
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
