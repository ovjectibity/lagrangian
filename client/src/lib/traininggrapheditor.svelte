<script lang="ts">
  import NodeGraphEditor from "./nodegrapheditor.svelte";
  import type { WorkflowEdge, WorkflowNode } from "./nodegrapheditor.svelte";

  const trainingNodes: WorkflowNode[] = [
    {
      id: "training-method",
      label: "TrainingMethod",
      position: { x: 80, y: 90 },
      inputs: 0,
      outputs: 1,
    },
    {
      id: "control-model",
      label: "ControlModel",
      position: { x: 350, y: 230 },
      inputs: 1,
      outputs: 1,
    },
    {
      id: "value-model",
      label: "ValueModel",
      position: { x: 620, y: 350 },
      inputs: 1,
      outputs: 0,
    },
  ];

  const trainingEdges: WorkflowEdge[] = [];
</script>

<div class="training-graph-editor">
  <NodeGraphEditor nodes={trainingNodes} edges={trainingEdges} />
</div>
