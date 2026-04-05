<script lang="ts">
  import NodeGraphEditor from "./nodegrapheditor.svelte";
  import {
    TrainingGraphEdges,
    TrainingGraphNodes,
  } from "../workflows/workflowgraph";
  import type { WorkflowEdge, WorkflowNode } from "../workflows/workflowgraph";
  const {
    onOpenRewardEditor,
    checkpointOptionsByNode,
    onCheckpointTypeSelected,
    onCheckpointSelected,
  } = $props<{
    onOpenRewardEditor?: () => void;
    checkpointOptionsByNode?: Record<string, Array<{ label: string; value: string }>>;
    onCheckpointTypeSelected?: (nodeId: string) => void;
    onCheckpointSelected?: (nodeId: string, checkpointId: string) => void;
  }>();

  const trainingNodes: WorkflowNode[] = TrainingGraphNodes;
  const trainingEdges: WorkflowEdge[] = TrainingGraphEdges;
</script>

<div class="training-graph-editor">
  <NodeGraphEditor
    nodes={trainingNodes}
    edges={trainingEdges}
    onOpenRewardEditor={onOpenRewardEditor}
    checkpointOptionsByNode={checkpointOptionsByNode}
    onCheckpointTypeSelected={onCheckpointTypeSelected}
    onCheckpointSelected={onCheckpointSelected}
  />
</div>
