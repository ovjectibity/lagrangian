<script lang="ts">
  const {
    experimentName,
    runName,
  } = $props<{
    experimentName?: string;
    runName?: string;
  }>();
</script>

<div class="breadcrumb">
  <span class="breadcrumb-segment" class:breadcrumb-segment-placeholder={!experimentName}>
    {experimentName ?? 'Not set'}
  </span>
  <span class="breadcrumb-separator">›</span>
  <span class="breadcrumb-segment" class:breadcrumb-segment-placeholder={!runName}>
    {runName ?? 'Not set'}
  </span>
</div>
