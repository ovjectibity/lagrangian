<script lang="ts">
  import NodeGraphEditor from "./nodegrapheditor.svelte";
  import { TestingGraphEdges, TestingGraphNodes } from "../commondef/workflowgraph";
  import type { WorkflowEdge, WorkflowNode } from "../commondef/workflowgraph";

  const testingNodes: WorkflowNode[] = TestingGraphNodes;
  const testingEdges: WorkflowEdge[] = TestingGraphEdges;
</script>

<div class="testing-graph-editor">
  <NodeGraphEditor nodes={testingNodes} edges={testingEdges} />
</div>
