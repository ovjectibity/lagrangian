<script lang="ts">
  import NodeGraphEditor from "./nodegrapheditor.svelte";
  import {
    TrainingGraphEdges,
    TrainingGraphNodes,
  } from "../common/workflowgraphtypes";
  import type { WorkflowEdge, WorkflowNode } from "../common/workflowgraphtypes";

  const trainingNodes: WorkflowNode[] = TrainingGraphNodes;
  const trainingEdges: WorkflowEdge[] = TrainingGraphEdges;
</script>

<div class="training-graph-editor">
  <NodeGraphEditor nodes={trainingNodes} edges={trainingEdges} />
</div>
