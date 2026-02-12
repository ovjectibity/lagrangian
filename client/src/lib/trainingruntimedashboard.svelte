<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { TRAINING_METRICS_CHARTS } from "../common/charts";
  import type { ECharts, EChartsOption } from "echarts";

  let chartContainers = $state<Array<HTMLDivElement | undefined>>([]);
  let chartInstances: ECharts[] = [];

  const makeSeriesData = (chartIdx: number): number[] => {
    return Array.from({ length: 20 }, (_, i) => {
      const t = i + 1;
      if (chartIdx === 0) {
        return Number((20 + t * 0.8 + Math.sin(t / 2) * 2).toFixed(3));
      }
      if (chartIdx === 1) {
        return Number((0.9 / (t + 1) + 0.05 * Math.sin(t / 3)).toFixed(4));
      }
      if (chartIdx === 2) {
        return Number((1.2 / (t + 1) + 0.03 * Math.cos(t / 2)).toFixed(4));
      }
      return Number((0.8 - t * 0.015 + 0.02 * Math.sin(t / 4)).toFixed(4));
    });
  };

  const makeXAxis = (): number[] => Array.from({ length: 20 }, (_, i) => i + 1);

  const buildOption = (
    chart: (typeof TRAINING_METRICS_CHARTS)[number],
    chartIdx: number
  ): EChartsOption => {
    return {
      backgroundColor: "#1f1f1f",
      title: {
        text: chart.metric_name,
        left: 12,
        top: 8,
        textStyle: { color: "#ffffff", fontSize: 12, fontWeight: 500 },
      },
      grid: {
        left: 52,
        right: 18,
        top: 42,
        bottom: 34,
      },
      xAxis: {
        type: "category",
        name: chart.x_axis,
        nameLocation: "middle",
        nameGap: 26,
        data: makeXAxis(),
        axisLine: { lineStyle: { color: "#6b6b6b" } },
        axisLabel: { color: "#c9c8c7", fontSize: 10 },
        nameTextStyle: { color: "#c9c8c7", fontSize: 10 },
      },
      yAxis: {
        type: "value",
        name: chart.y_axis,
        nameLocation: "middle",
        nameRotate: 90,
        nameGap: 42,
        axisLine: { lineStyle: { color: "#6b6b6b" } },
        splitLine: { lineStyle: { color: "#2f2f2f" } },
        axisLabel: { color: "#c9c8c7", fontSize: 10 },
        nameTextStyle: { color: "#c9c8c7", fontSize: 10 },
      },
      tooltip: {
        trigger: "axis",
      },
      series: [
        {
          type: "line",
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 2 },
          data: makeSeriesData(chartIdx),
        },
      ],
    };
  };

  const resizeCharts = (): void => {
    for (const chart of chartInstances) {
      chart.resize();
    }
  };

  onMount(async () => {
    const echarts = await import("echarts");
    chartInstances = [];

    TRAINING_METRICS_CHARTS.forEach((chart, idx) => {
      const target = chartContainers[idx];
      if (!target) {
        return;
      }
      const instance = echarts.init(target);
      instance.setOption(buildOption(chart, idx));
      chartInstances.push(instance);
    });

    window.addEventListener("resize", resizeCharts);
  });

  onDestroy(() => {
    window.removeEventListener("resize", resizeCharts);
    for (const chart of chartInstances) {
      chart.dispose();
    }
    chartInstances = [];
  });
</script>

<div class="training-runtime-dashboard">
  {#each TRAINING_METRICS_CHARTS as chart, idx}
    <div class="training-metric-chart-card">
      <div class="training-metric-chart" bind:this={chartContainers[idx]} data-metric={chart.metric_name}></div>
    </div>
  {/each}
</div>
