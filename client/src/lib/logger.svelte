<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { Terminal } from "@xterm/xterm";

  let terminalHost: HTMLDivElement;
  let terminalInstance: Terminal | null = null;

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
    terminalInstance.writeln("Logs ready.");
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
