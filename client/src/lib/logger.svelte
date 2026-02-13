<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { Terminal } from "@xterm/xterm";

  const { logs = [] } = $props<{
    logs?: string[];
  }>();

  let terminalHost: HTMLDivElement;
  let terminalInstance: Terminal | null = null;
  let renderedLogCount = $state(0);

  onMount(() => {
    if (!terminalHost) {
      return;
    }
    terminalInstance = new Terminal({
      cols: 120,
      rows: 20,
      cursorBlink: false,
      theme: {
        background: '#000000',
        cursor: '#000000'
      }
    });
    terminalInstance.open(terminalHost);
    // terminalInstance.writeln("Logs ready.");
  });

  $effect(() => {
    logs;
    if (!terminalInstance) {
      return;
    }
    const nextLogs = logs.slice(renderedLogCount);
    for (const line of nextLogs) {
      terminalInstance.writeln(line);
    }
    renderedLogCount = logs.length;
  });

  onDestroy(() => {
    if (terminalInstance) {
      terminalInstance.dispose();
    }
  });
</script>

<div class="logger-container">
  <div class="logger-terminal" bind:this={terminalHost}></div>
</div>
