<script lang="ts">
  import { onMount } from "svelte";
  import type { Snippet } from "svelte";
  import type { editor } from "monaco-editor";
  import Button from "../atomic/button.svelte";

  let {
    code,
    controls,
    language = "python",
    subtitle,
    title,
    wrap = $bindable(false),
  }: {
    code: string;
    controls?: Snippet;
    language?: string;
    subtitle?: string;
    title: string;
    wrap?: boolean;
  } = $props();

  let container: HTMLDivElement;
  let instance = $state<editor.IStandaloneCodeEditor>();
  let status = $state("");
  let copyTimer: ReturnType<typeof setTimeout>;

  onMount(() => {
    let disposed = false;
    let model: editor.ITextModel | undefined;

    const initialise = async (): Promise<void> => {
      try {
        const [monaco, worker] = await Promise.all([
          import("monaco-editor/editor/editor.api.js"),
          import(
            "monaco-editor/editor/editor.worker.js?worker"
          ),
          import(
            "monaco-editor/basic-languages/monaco.contribution.js"
          ),
        ]);

        if (disposed) return;

        Object.assign(globalThis, {
          MonacoEnvironment: {
            getWorker: () => new worker.default(),
          },
        });
        monaco.editor.defineTheme("lagrangian-dark", {
          base: "vs-dark",
          inherit: true,
          rules: [],
          colors: {
            "editor.background": "#151518",
            "editor.foreground": "#ededf0",
            "editorLineNumber.foreground": "#6e6e78",
            "editor.selectionBackground": "#393940",
          },
        });
        model = monaco.editor.createModel(code, language);
        instance = monaco.editor.create(container, {
          ariaLabel: `${title} code`,
          automaticLayout: true,
          fontFamily: "ui-monospace, SFMono-Regular, monospace",
          fontSize: 12,
          lineHeight: 20,
          minimap: { enabled: false },
          model,
          padding: { top: 16, bottom: 16 },
          readOnly: true,
          renderLineHighlight: "none",
          scrollBeyondLastLine: false,
          theme: "lagrangian-dark",
          wordWrap: wrap ? "on" : "off",
        });
      } catch {
        if (!disposed) status = "Unable to load code viewer.";
      }
    };

    void initialise();
    return () => {
      disposed = true;
      clearTimeout(copyTimer);
      instance?.dispose();
      model?.dispose();
    };
  });

  $effect(() => {
    if (instance && instance.getValue() !== code) {
      instance.setValue(code);
    }
  });

  $effect(() => {
    instance?.updateOptions({ wordWrap: wrap ? "on" : "off" });
  });

  $effect(() => {
    const model = instance?.getModel();
    const currentLanguage = language;
    let cancelled = false;

    if (model) {
      void import("monaco-editor/editor/editor.api.js")
        .then((monaco) => {
          if (!cancelled) {
            monaco.editor.setModelLanguage(model, currentLanguage);
          }
        });
    }
    return () => { cancelled = true; };
  });

  const copyCode = async (): Promise<void> => {
    clearTimeout(copyTimer);
    try {
      await navigator.clipboard.writeText(code);
      status = "Copied";
    } catch {
      status = "Copy failed. Select the code and copy manually.";
    }
    copyTimer = setTimeout(() => { status = ""; }, 2500);
  };
</script>

<section aria-label={title} class="lg-code-view">
  <header class="lg-code-view__header">
    <h3>{title}</h3>
    <div class="lg-code-view__header-end">
      {#if subtitle}<p>{subtitle}</p>{/if}
      <div class="lg-code-view__controls">
        {@render controls?.()}
        <Button
          active={wrap}
          onclick={() => (wrap = !wrap)}
          variant="type-3"
        >Wrap</Button>
        <Button
          hasActiveState={false}
          onclick={copyCode}
          variant="type-4"
        >Copy</Button>
      </div>
    </div>
  </header>
  <div aria-live="polite" class="lg-code-view__status">
    {status}
  </div>
  <div bind:this={container} class="lg-code-view__editor"></div>
</section>
