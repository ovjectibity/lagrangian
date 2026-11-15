<script lang="ts">
  import type { Snippet } from "svelte";
  import Button from "../atomic/button.svelte";

  let {
    columns,
    controls,
    rowControls = [],
    rows,
    subtitle,
    title,
  }: {
    columns: ReadonlyArray<{
      align?: "left" | "right";
      // Text space before the column button, in pixels.
      contentWidth?: number;
      label: string;
      button?: {
          active?: boolean;
          hasActiveState?: boolean;
          icon?: Snippet;
          label: string;
          onclick?: (rowIndex: number, event: MouseEvent) => void;
          variant?: "type-1" | "type-2" | "type-3" | "type-4";
      };
    }>;
    controls?: Snippet;
    // Shared definitions ensure controls apply uniformly to all rows.
    rowControls?: ReadonlyArray<{
        active?: boolean;
        hasActiveState?: boolean;
        icon: "play" | "stop" | "view";
        label: string;
        onclick?: (rowIndex: number, event: MouseEvent) => void;
        variant?: "type-1" | "type-2" | "type-3" | "type-4";
    }>;
    rows: ReadonlyArray<{
      cells: ReadonlyArray<{ text?: string }>;
    }>;
    subtitle?: string;
    title: string;
  } = $props();

</script>

<section aria-label={title} class="lg-table">
  <header class="lg-table__header">
    <h3>{title}</h3>
    {#if subtitle || controls}
      <div class="lg-table__header-end">
        {#if subtitle}
          <p>{subtitle}</p>
        {/if}
        {#if controls}
          <div class="lg-table__controls">
            {@render controls()}
          </div>
        {/if}
      </div>
    {/if}
  </header>

  <div class="lg-table__scroll">
    <table>
      <colgroup>
        {#each columns as _}
          <col />
        {/each}
      </colgroup>
      <thead>
        <tr>
          {#each columns as column}
            <th class:lg-table__cell--right={column.align === "right"}>
              {column.label}
            </th>
          {/each}
        </tr>
      </thead>
      <tbody>
        {#each rows as row, rowIndex}
          <tr>
            {#each columns as column, index}
              <td
                class:lg-table__cell--right={
                  columns[index]?.align === "right"
                }
              >
                <div
                  class="lg-table__cell-content"
                  style:--lg-table-content-width={
                    `${column.contentWidth ?? 64}px`
                  }
                  class:lg-table__cell-content--with-button={
                    !!column.button
                  }
                >
                  {#if row.cells[index]?.text}
                    <span
                      class="lg-table__cell-text"
                      title={row.cells[index].text}
                    >{row.cells[index].text}</span>
                  {/if}
                  {#if column.button}
                    <Button
                      active={column.button.active}
                      aria-label={column.button.label}
                      class="lg-table__column-button"
                      hasActiveState={column.button.hasActiveState}
                      onclick={(event) =>
                        column.button?.onclick?.(rowIndex, event)}
                      title={column.button.label}
                      variant={column.button.variant}
                    >
                      {#if column.button.icon}
                        {@render column.button.icon()}
                      {:else}
                        {column.button.label}
                      {/if}
                    </Button>
                  {/if}
            {#if index === columns.length - 1 && rowControls.length}
                <div class="lg-table__row-actions">
                <!--
                  Keep row controls icon-only so actions do not disrupt
                  column alignment. Use `label` for accessible text.
                -->
                {#each rowControls as control}
                  <Button
                    active={control.active}
                    aria-label={control.label}
                    hasActiveState={control.hasActiveState}
                    onclick={(event) =>
                      control.onclick?.(rowIndex, event)}
                    title={control.label}
                    variant={control.variant}
                  >
                    <svg
                      aria-hidden="true"
                      fill="none"
                      stroke="currentColor"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="1.75"
                      viewBox="0 0 24 24"
                    >
                      {#if control.icon === "view"}
                        <path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" />
                        <circle cx="12" cy="12" r="2.5" />
                      {:else if control.icon === "stop"}
                        <rect height="10" rx="1" width="10" x="7" y="7" />
                      {:else}
                        <path d="m8 5 11 7-11 7Z" />
                      {/if}
                    </svg>
                  </Button>
                {/each}
                </div>
            {/if}
                </div>
              </td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</section>
