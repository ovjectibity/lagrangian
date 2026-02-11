<script lang="ts">
  import CodeEditor from "./codeeditor.svelte";

  let rewardFunctionCode = $state(`fn get_reward(state: &[f64]) -> f64 {
    let _ = state;
    0.0
}`);
</script>

<div class="reward-editor">
  <CodeEditor
    code={rewardFunctionCode}
    onCodeChange={(nextCode) => {
      rewardFunctionCode = nextCode;
    }}
  />
</div>
