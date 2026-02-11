<script lang="ts">
  import { Controls, Node, Svelvet } from "svelvet";

  export type WorkflowNode = {
    id: string;
    label: string;
    position: { x: number; y: number };
    inputs?: number;
    outputs?: number;
  };

  export type WorkflowEdge = {
    id: string;
    source: string;
    target: string;
  };

  const { nodes = [], edges = [] } = $props<{
    nodes?: WorkflowNode[];
    edges?: WorkflowEdge[];
  }>();
</script>

<div class="graph-editor">
  <Svelvet minimap theme="dark" zoom={0.75}>
    {#each nodes as node}
      <Node
        id={node.id}
        label={node.label}
        position={node.position}
        inputs={node.inputs ?? 1}
        outputs={node.outputs ?? 1}
      />
    {/each}
    <Controls horizontal slot="controls" />
  </Svelvet>
</div>
