<script lang="ts">
  import arrowUpRight from "../assets/arrow-up-right.svg?raw";

  let {
    items,
    title = "Properties",
  }: {
    items: ReadonlyArray<{
      label: string;
      value: string;
      href?: string;
    }>;
    title?: string;
  } = $props();
</script>

<section aria-label={title} class="lg-properties">
  <h3>{title}</h3>
  <dl>
    {#each items as item}
      <div class="lg-properties__item">
        <dt>{item.label}</dt>
        <dd>
          {#if item.href}
            <a href={item.href}>
              <span>{item.value}</span>
              <span aria-hidden="true" class="lg-properties__arrow">
                {@html arrowUpRight}
              </span>
            </a>
          {:else}
            {item.value}
          {/if}
        </dd>
      </div>
    {/each}
  </dl>
</section>
