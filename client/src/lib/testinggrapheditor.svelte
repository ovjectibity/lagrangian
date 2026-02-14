<script lang="ts">
  import NodeGraphEditor from "./nodegrapheditor.svelte";
  import type { WorkflowEdge, WorkflowNode } from "../commondef/workflowgraph";

  const testingNodes: WorkflowNode[] = [];
  const testingEdges: WorkflowEdge[] = [];
</script>

<div class="testing-graph-editor">
  <NodeGraphEditor nodes={testingNodes} edges={testingEdges} />
</div>
