<script lang="ts">
  let {
    id,
    labels = [],
    progress,
    status,
    statusColor = "#81818a",
    subInfo = [],
    subtitle,
    thumbnail,
    thumbnailAlt,
    thumbnailTitle,
    title,
  }: {
    id: string;
    labels?: readonly string[];
    progress?: {
      value: number;
      completionTime?: string;
    };
    status: string;
    statusColor?: string;
    subInfo?: readonly string[];
    subtitle: string;
    thumbnail: string;
    thumbnailAlt: string;
    thumbnailTitle?: string;
    title: string;
  } = $props();

  const progressValue = $derived(
    Math.min(100, Math.max(0, progress?.value ?? 0)),
  );
</script>

<article aria-label={title} class="lg-card">
  <div class="lg-card__thumbnail">
    <img alt={thumbnailAlt} loading="lazy" src={thumbnail} />
    {#if thumbnailTitle}
      <span class="lg-card__thumbnail-title">
        {thumbnailTitle}
      </span>
    {/if}
  </div>
  <div class="lg-card__body">
    <div class="lg-card__identity">
      <span class="lg-card__id">{id}</span>
      <span class="lg-card__status">
        <span
          aria-hidden="true"
          class="lg-card__status-dot"
          style:background-color={statusColor}
        ></span>
        {status}
      </span>
    </div>
    <h3>{title}</h3>
    <p class="lg-card__subtitle">{subtitle}</p>
    {#if labels.length}
      <ul aria-label="Labels" class="lg-card__labels">
        {#each labels as label}
          <li>{label}</li>
        {/each}
      </ul>
    {/if}
    {#if subInfo.length || progress}
      <div class="lg-card__footer">
    {#if subInfo.length}
      <ul aria-label="Additional information" class="lg-card__info">
        {#each subInfo as info}
          <li>{info}</li>
        {/each}
      </ul>
    {/if}
    {#if progress}
      <div class="lg-card__progress">
        <div class="lg-card__progress-label">
          <span>{progressValue}% complete</span>
          {#if progress.completionTime}
            <span>{progress.completionTime}</span>
          {/if}
        </div>
        <div
          aria-label={`${title} completion`}
          aria-valuemax="100"
          aria-valuemin="0"
          aria-valuenow={progressValue}
          class="lg-card__progress-track"
          role="progressbar"
        >
          <div
            class="lg-card__progress-fill"
            style:width={`${progressValue}%`}
          ></div>
        </div>
      </div>
    {/if}
      </div>
    {/if}
  </div>
</article>
