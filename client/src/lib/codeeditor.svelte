<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import type { editor } from "monaco-editor";

  const { code = "", onCodeChange } = $props<{
    code?: string;
    onCodeChange?: (nextCode: string) => void;
  }>();

  let containerEl: HTMLDivElement | undefined = $state(undefined);
  let monacoEditor: editor.IStandaloneCodeEditor | undefined = undefined;

  onMount(async () => {
    if (!containerEl) {
      return;
    }

    const monaco = await import("monaco-editor");
    const editorInstance = monaco.editor.create(containerEl, {
      value: code,
      language: "rust",
      automaticLayout: true,
      minimap: { enabled: false },
      theme: "vs-dark",
      scrollBeyondLastLine: false,
      fontSize: 13,
    });
    monacoEditor = editorInstance;

    editorInstance.onDidChangeModelContent(() => {
      onCodeChange?.(editorInstance.getValue());
    });
  });

  onDestroy(() => {
    monacoEditor?.dispose();
    monacoEditor = undefined;
  });
</script>

<div class="code-editor-root" bind:this={containerEl}></div>
