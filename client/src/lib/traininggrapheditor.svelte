<script lang="ts">
  import NodeGraphEditor from "./nodegrapheditor.svelte";
  import {
    TrainingGraphEdges,
    TrainingGraphNodes,
  } from "../commondef/workflowgraph";
  import type { WorkflowEdge, WorkflowNode } from "../commondef/workflowgraph";
  const { onOpenRewardEditor } = $props<{ onOpenRewardEditor?: () => void }>();

  const trainingNodes: WorkflowNode[] = TrainingGraphNodes;
  const trainingEdges: WorkflowEdge[] = TrainingGraphEdges;
</script>

<div class="training-graph-editor">
  <NodeGraphEditor
    nodes={trainingNodes}
    edges={trainingEdges}
    onOpenRewardEditor={onOpenRewardEditor}
  />
</div>
