<script lang="ts">
  import NodeGraphEditor from "./nodegrapheditor.svelte";
  import { TestingGraphEdges, TestingGraphNodes } from "../workflows/workflowgraph";
  import type { WorkflowEdge, WorkflowNode } from "../workflows/workflowgraph";

  const testingNodes: WorkflowNode[] = TestingGraphNodes;
  const testingEdges: WorkflowEdge[] = TestingGraphEdges;
</script>

<div class="testing-graph-editor">
  <NodeGraphEditor nodes={testingNodes} edges={testingEdges} />
</div>
