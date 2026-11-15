<script lang="ts">
  import { LineChart as EChartsLineChart } from "echarts/charts";
  import {
    AriaComponent,
    GridComponent,
    LegendComponent,
    TooltipComponent,
  } from "echarts/components";
  import * as echarts from "echarts/core";
  import type { EChartsOption } from "echarts";
  import { CanvasRenderer } from "echarts/renderers";
  import MultiSelectDropdown from "../atomic/multi-select-dropdown.svelte";
  import MultiValueSwitch from "../atomic/multi-value-switch.svelte";
  import SingleSelectDropdown from "../atomic/single-select-dropdown.svelte";

  echarts.use([
    AriaComponent,
    CanvasRenderer,
    EChartsLineChart,
    GridComponent,
    LegendComponent,
    TooltipComponent,
  ]);

  let {
    multiSelectOptions,
    multiSelectValues = $bindable<string[]>([]),
    series,
    singleSelectOptions,
    singleSelectValue = $bindable<string>(),
    switchOptions,
    switchValue = $bindable<string>(),
    title,
    xAxis,
  }: {
    multiSelectOptions?: ReadonlyArray<{
      disabled?: boolean;
      label: string;
      value: string;
    }>;
    multiSelectValues?: string[];
    series: ReadonlyArray<{
      color?: string;
      name: string;
      values: readonly number[];
    }>;
    singleSelectOptions?: ReadonlyArray<{
      disabled?: boolean;
      label: string;
      value: string;
    }>;
    singleSelectValue?: string;
    switchOptions?: readonly string[];
    switchValue?: string;
    title: string;
    xAxis: readonly string[];
  } = $props();

  let chartElement = $state<HTMLDivElement>();

  const option = $derived.by((): EChartsOption => {
    const colors = ["#d7a280", "#8e9de7", "#6bbf97"];

    return {
      aria: {
        enabled: true,
      },
      color: series.map((line, index) => {
        return line.color ?? colors[index % colors.length];
      }),
      grid: {
        bottom: 32,
        left: 44,
        right: 16,
        top: 38,
      },
      legend: {
        itemHeight: 8,
        itemWidth: 8,
        right: 14,
        textStyle: {
          color: "#93949e",
          fontFamily: "Inter, system-ui, sans-serif",
          fontSize: 11,
        },
        top: 10,
      },
      series: series.map((line, index) => {
        const color = line.color ?? colors[index % colors.length];

        return {
          areaStyle: {
            color: index === 0 ? "rgba(215, 162, 128, 0.14)" : undefined,
          },
          data: [...line.values],
          lineStyle: {
            color,
            width: 2,
          },
          name: line.name,
          showSymbol: false,
          smooth: 0.18,
          symbol: "circle",
          type: "line",
        };
      }),
      tooltip: {
        axisPointer: {
          lineStyle: {
            color: "#53535d",
            type: "dashed",
          },
          type: "line",
        },
        backgroundColor: "#101013",
        borderColor: "#303036",
        borderWidth: 1,
        padding: [8, 10],
        textStyle: {
          color: "#ededf0",
          fontFamily: "Inter, system-ui, sans-serif",
          fontSize: 12,
        },
        trigger: "axis",
      },
      xAxis: {
        axisLabel: {
          color: "#81818a",
          fontFamily: "Inter, system-ui, sans-serif",
          fontSize: 11,
        },
        axisLine: {
          lineStyle: {
            color: "#303037",
          },
        },
        axisTick: {
          show: false,
        },
        boundaryGap: false,
        data: [...xAxis],
        type: "category",
      },
      yAxis: {
        axisLabel: {
          color: "#81818a",
          fontFamily: "Inter, system-ui, sans-serif",
          fontSize: 11,
        },
        axisLine: {
          show: false,
        },
        axisTick: {
          show: false,
        },
        splitLine: {
          lineStyle: {
            color: "#29292e",
            type: "dashed",
          },
        },
        type: "value",
      },
    };
  });

  $effect(() => {
    const chartOption = option;

    if (!chartElement) {
      return;
    }

    const chart = echarts.init(chartElement, undefined, {
      renderer: "canvas",
    });
    const observer = new ResizeObserver(() => {
      chart.resize();
    });

    chart.setOption(chartOption);
    observer.observe(chartElement);

    return () => {
      observer.disconnect();
      chart.dispose();
    };
  });
</script>

<section aria-label={title} class="lg-line-chart">
  <header class="lg-line-chart__header">
    <h3>{title}</h3>
    {#if switchOptions || singleSelectOptions || multiSelectOptions}
      <div class="lg-line-chart__controls">
        {#if switchOptions && switchValue !== undefined}
          <MultiValueSwitch
            ariaLabel={`${title} range`}
            bind:value={switchValue}
            options={switchOptions}
          />
        {/if}
        {#if singleSelectOptions && singleSelectValue !== undefined}
          <SingleSelectDropdown
            ariaLabel={`${title} metric`}
            bind:value={singleSelectValue}
            options={singleSelectOptions}
            variant="text"
          />
        {/if}
        {#if multiSelectOptions}
          <MultiSelectDropdown
            ariaLabel={`${title} series`}
            bind:values={multiSelectValues}
            options={multiSelectOptions}
            variant="text"
          />
        {/if}
      </div>
    {/if}
  </header>
  <div
    aria-label={`${title} line chart`}
    bind:this={chartElement}
    class="lg-line-chart__canvas"
    role="img"
  ></div>
</section>
