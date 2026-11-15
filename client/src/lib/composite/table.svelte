<script lang="ts">
  import type { Snippet } from "svelte";
  import Button from "../atomic/button.svelte";

  let {
    columns,
    controls,
    rows,
    subtitle,
    title,
  }: {
    columns: ReadonlyArray<{
      align?: "left" | "right";
      label: string;
    }>;
    controls?: Snippet;
    rows: ReadonlyArray<{
      cells: ReadonlyArray<{
        button?: {
          active?: boolean;
          hasActiveState?: boolean;
          label: string;
          onclick?: (event: MouseEvent) => void;
          variant?: "type-1" | "type-2" | "type-3" | "type-4";
        };
        text?: string;
      }>;
      controls?: ReadonlyArray<{
        active?: boolean;
        hasActiveState?: boolean;
        icon: "play" | "stop" | "view";
        label: string;
        onclick?: (event: MouseEvent) => void;
        variant?: "type-1" | "type-2" | "type-3" | "type-4";
      }>;
    }>;
    subtitle?: string;
    title: string;
  } = $props();

  const hasRowControls = $derived(
    rows.some((row) => (row.controls?.length ?? 0) > 0),
  );
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
      <thead>
        <tr>
          {#each columns as column}
            <th class:lg-table__cell--right={column.align === "right"}>
              {column.label}
            </th>
          {/each}
          {#if hasRowControls}
            <th aria-label="Row actions"></th>
          {/if}
        </tr>
      </thead>
      <tbody>
        {#each rows as row}
          <tr>
            {#each row.cells as cell, index}
              <td
                class:lg-table__cell--right={
                  columns[index]?.align === "right"
                }
              >
                <div class="lg-table__cell-content">
                  {#if cell.text}
                    <span>{cell.text}</span>
                  {/if}
                  {#if cell.button}
                    <Button
                      active={cell.button.active}
                      hasActiveState={cell.button.hasActiveState}
                      onclick={cell.button.onclick}
                      variant={cell.button.variant}
                    >
                      {cell.button.label}
                    </Button>
                  {/if}
                </div>
              </td>
            {/each}
            {#if hasRowControls}
              <td class="lg-table__row-controls">
                <!--
                  Keep row controls icon-only so actions do not disrupt
                  column alignment. Use `label` for accessible text.
                -->
                {#each row.controls ?? [] as control}
                  <Button
                    active={control.active}
                    aria-label={control.label}
                    hasActiveState={control.hasActiveState}
                    onclick={control.onclick}
                    title={control.label}
                    variant={control.variant}
                  >
                    <svg aria-hidden="true" viewBox="0 0 24 24">
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
              </td>
            {/if}
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</section>
