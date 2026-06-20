<script lang="ts">
  import { Anchor, Controls, Node, Svelvet } from "svelvet";
  import type {
    WorkflowEdge,
    WorkflowNode,
    WorkflowNodeSetting,
  } from "../workflows/workflowgraph";

  const {
    nodes = [],
    edges = [],
    onOpenRewardEditor,
    checkpointOptionsByNode,
    onCheckpointTypeSelected,
    onCheckpointSelected,
  } = $props<{
    nodes?: WorkflowNode[];
    edges?: WorkflowEdge[];
    onOpenRewardEditor?: () => void;
    checkpointOptionsByNode?: Record<string, Array<{ label: string; value: string }>>;
    onCheckpointTypeSelected?: (nodeId: string) => void;
    onCheckpointSelected?: (nodeId: string, checkpointId: string) => void;
  }>();

  let nodeSettingValues = $state<Record<string, Record<string, number | string>>>({});
  let edgeRevision = $state(0);

  $effect(() => {
    const nextValues: Record<string, Record<string, number | string>> = {};
    for (const node of nodes) {
      const settings = node.settings ?? [];
      if (!settings.length) {
        continue;
      }
      nextValues[node.id] = {};
      for (const setting of settings) {
        nextValues[node.id][setting.key] = setting.value;
      }
    }
    nodeSettingValues = nextValues;
  });

  const getSettingValue = (nodeId: string, setting: WorkflowNodeSetting): number | string => {
    return nodeSettingValues[nodeId]?.[setting.key] ?? setting.value;
  };

  const setSettingValue = (nodeId: string, setting: WorkflowNodeSetting, rawValue: string): void => {
    let value: number | string = rawValue;
    if ((setting.inputType ?? "number") === "number") {
      const parsedValue = Number.parseFloat(rawValue);
      value = Number.isNaN(parsedValue) ? setting.value : parsedValue;
    }
    nodeSettingValues[nodeId] = {
      ...(nodeSettingValues[nodeId] ?? {}),
      [setting.key]: value,
    };
    const targetNode = nodes.find((node: WorkflowNode) => node.id === nodeId);
    const targetSetting = targetNode?.settings?.find(
      (nodeSetting: WorkflowNodeSetting) => nodeSetting.key === setting.key
    );
    if (targetSetting) {
      targetSetting.value = value;
    }
    nodeSettingValues = { ...nodeSettingValues };
  };

  const getOutputConnections = (
    sourceNodeId: string,
    sourceOutputAnchorId: string
  ): Array<[string, string]> => {
    edgeRevision;
    return edges
      .filter(
        (edge: WorkflowEdge) =>
          edge.sourceNodeId === sourceNodeId &&
          edge.sourceOutputAnchor.id === sourceOutputAnchorId
      )
      .map((edge: WorkflowEdge) => [edge.targetNodeId, edge.targetInputAnchor.id]);
  };

  const normalizeNodeId = (id: string | number): string => {
    return String(id).replace(/^N-/, "");
  };

  const normalizeAnchorId = (id: string | number): string => {
    return String(id).replace(/^A-/, "").split("/")[0];
  };

  const handleAnchorConnection = (event: Event): void => {
    const customEvent = event as CustomEvent<{
      node: { id: string | number };
      anchor: { id: string | number; type?: string };
      connectedNode: { id: string | number };
      connectedAnchor: { id: string | number };
    }>;
    const detail = customEvent.detail;
    if (!detail) {
      return;
    }

    const sourceNodeId = normalizeNodeId(detail.node.id);
    const sourceAnchorId = normalizeAnchorId(detail.anchor.id);
    const targetNodeId = normalizeNodeId(detail.connectedNode.id);
    const targetAnchorId = normalizeAnchorId(detail.connectedAnchor.id);

    const exists = edges.some(
      (edge: WorkflowEdge) =>
        edge.sourceNodeId === sourceNodeId &&
        edge.targetNodeId === targetNodeId &&
        edge.sourceOutputAnchor.id === sourceAnchorId &&
        edge.targetInputAnchor.id === targetAnchorId
    );
    if (exists) {
      return;
    }

    edges.push({
      id: `${sourceNodeId}-${sourceAnchorId}-to-${targetNodeId}-${targetAnchorId}`,
      sourceNodeId,
      targetNodeId,
      sourceOutputAnchor: { id: sourceAnchorId, label: sourceAnchorId },
      targetInputAnchor: { id: targetAnchorId, label: targetAnchorId },
    });
    edgeRevision += 1;
  };

  const handleAnchorDisconnection = (event: Event): void => {
    const customEvent = event as CustomEvent<{
      node: { id: string | number };
      anchor: { id: string | number; type?: string };
      connectedNode: { id: string | number };
      connectedAnchor: { id: string | number };
    }>;
    const detail = customEvent.detail;
    if (!detail) {
      return;
    }

    const sourceNodeId = normalizeNodeId(detail.node.id);
    const sourceAnchorId = normalizeAnchorId(detail.anchor.id);
    const targetNodeId = normalizeNodeId(detail.connectedNode.id);
    const targetAnchorId = normalizeAnchorId(detail.connectedAnchor.id);

    const edgeIndex = edges.findIndex(
      (edge: WorkflowEdge) =>
        edge.sourceNodeId === sourceNodeId &&
        edge.targetNodeId === targetNodeId &&
        edge.sourceOutputAnchor.id === sourceAnchorId &&
        edge.targetInputAnchor.id === targetAnchorId
    );
    if (edgeIndex < 0) {
      return;
    }

    edges.splice(edgeIndex, 1);
    edgeRevision += 1;
  };
</script>

<div class="graph-editor">
  <Svelvet
    minimap
    theme="dark"
    zoom={0.75}
    fitView="resize"
  >
    {#each nodes as node}
      <Node
        id={node.id}
        label={node.label}
        bind:position={node.position}
        inputs={node.inputAnchors?.length ?? node.inputs ?? 1}
        outputs={node.outputAnchors?.length ?? node.outputs ?? 1}
      >
        <div class="workflow-node-shell">
          <div class="workflow-node-header">{node.label}</div>
          {#if (node.settings?.length ?? 0) > 0}
            <div class="workflow-node-settings">
              {#each node.settings ?? [] as setting}
                {@const isCheckpointMode =
                  (node.nodeType === 'ControlModel' || node.nodeType === 'ValueModel') &&
                  String(getSettingValue(node.id,
                    (node.settings ?? []).find((s: WorkflowNodeSetting) => s.key === 'type') ?? setting
                  )) === 'Checkpoint'}
                {@const isMlpField =
                  setting.key === 'num_hidden_layers' ||
                  setting.key === 'hidden_layer_dim' ||
                  setting.key === 'output_dim' ||
                  setting.key === 'input_dim'}
                {@const isCheckpointIdField = setting.key === 'checkpoint_id'}
                {#if isCheckpointIdField && !isCheckpointMode}
                  <!-- hide checkpoint_id when type is MLP -->
                {:else}
                  <label class="workflow-node-setting">
                    <span>{setting.label}</span>
                    {#if (setting.inputType ?? "number") === "select"}
                      {#if isCheckpointIdField}
                        <select
                          class="workflow-node-setting-select"
                          value={String(getSettingValue(node.id, setting))}
                          oninput={(event) => {
                            const val = (event.currentTarget as HTMLSelectElement).value;
                            setSettingValue(node.id, setting, val);
                            onCheckpointSelected?.(node.id, val);
                          }}
                        >
                          {#each (checkpointOptionsByNode?.[node.id] ?? []) as option}
                            <option value={option.value} disabled={option.value === ''}>{option.label}</option>
                          {/each}
                        </select>
                      {:else}
                        <select
                          class="workflow-node-setting-select"
                          disabled={setting.disabled ?? false}
                          value={String(getSettingValue(node.id, setting))}
                          oninput={(event) => {
                            const val = (event.currentTarget as HTMLSelectElement).value;
                            setSettingValue(node.id, setting, val);
                            if (setting.key === 'type' && val === 'Checkpoint') {
                              onCheckpointTypeSelected?.(node.id);
                            }
                          }}
                        >
                          {#each setting.options ?? [] as option}
                            <option value={option.value}>{option.label}</option>
                          {/each}
                        </select>
                      {/if}
                    {:else}
                      {#if setting.inputType === "link"}
                        <button
                          class="workflow-node-link-action"
                          type="button"
                          title="Open Reward editor"
                          aria-label="Open Reward editor"
                          onclick={() => onOpenRewardEditor?.()}
                        >
                          <span>{String(getSettingValue(node.id, setting))}</span>
                          <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                            <path d="M10.6 13.4a1 1 0 0 1 0-1.4l2.4-2.4a1 1 0 1 1 1.4 1.4L12 13.4a1 1 0 0 1-1.4 0zm-3.2 4.6a4 4 0 0 1 0-5.7l2.1-2.1a1 1 0 1 1 1.4 1.4l-2.1 2.1a2 2 0 1 0 2.8 2.8l2.1-2.1a1 1 0 1 1 1.4 1.4L13 18a4 4 0 0 1-5.7 0zm9.2-12a4 4 0 0 1 0 5.7l-2.1 2.1a1 1 0 1 1-1.4-1.4l2.1-2.1a2 2 0 1 0-2.8-2.8l-2.1 2.1a1 1 0 1 1-1.4-1.4L11 6a4 4 0 0 1 5.7 0z" />
                          </svg>
                        </button>
                      {:else}
                        <input
                          class="workflow-node-setting-input"
                          type="number"
                          disabled={(setting.disabled ?? false) || (isCheckpointMode && isMlpField)}
                          min={setting.min}
                          max={setting.max}
                          step={setting.step ?? 1}
                          value={getSettingValue(node.id, setting)}
                          oninput={(event) =>
                            setSettingValue(
                              node.id,
                              setting,
                              (event.currentTarget as HTMLInputElement).value
                            )}
                        />
                      {/if}
                    {/if}
                  </label>
                {/if}
              {/each}
            </div>
          {/if}
          {#if (node.inputAnchors?.length ?? 0) > 0}
            <div class="workflow-node-inputs">
              {#each node.inputAnchors ?? [] as anchor}
                <div class="workflow-node-input-item">
                  <Anchor id={anchor.id} input direction="west" />
                  <span>{anchor.label}</span>
                </div>
              {/each}
            </div>
          {/if}
          {#if (node.outputAnchors?.length ?? 0) > 0}
            <div class="workflow-node-outputs">
              {#each node.outputAnchors ?? [] as anchor}
                <div class="workflow-node-output-item">
                  <span>{anchor.label}</span>
                  <Anchor
                    id={anchor.id}
                    output
                    direction="east"
                    connections={getOutputConnections(node.id, anchor.id)}
                    on:connection={handleAnchorConnection}
                    on:disconnection={handleAnchorDisconnection}
                  />
                </div>
              {/each}
            </div>
          {/if}
          {#if
            (node.settings?.length ?? 0) === 0 &&
            (node.inputAnchors?.length ?? 0) === 0 &&
            (node.outputAnchors?.length ?? 0) === 0}
            <div class="workflow-node-default-body">{node.nodeType ?? node.label}</div>
          {/if}
        </div>
      </Node>
    {/each}
    <Controls horizontal slot="controls" />
  </Svelvet>
</div>
