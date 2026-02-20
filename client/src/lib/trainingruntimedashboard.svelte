<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { TRAINING_METRICS_CHARTS } from "../commondef/charts";
  import type { MetricsData, TrainingMetrics } from "../commondef/training";
  import type { ECharts, EChartsOption } from "echarts";

  type TrainingMetricType = TrainingMetrics["type"];
  type TrainingMetricsByType = Record<TrainingMetricType, MetricsData>;

  const CHART_METRIC_ORDER: TrainingMetricType[] = [
    "MeanReward",
    "MeanEntropy",
    "ValueLoss",
    "MeanPolicyLoss",
    "PolicyGradientNorm",
    "ValueGradientNorm",
  ];

  const EMPTY_METRICS: TrainingMetricsByType = {
    MeanReward: { x_axis: [], y_axis: [] },
    MeanPolicyLoss: { x_axis: [], y_axis: [] },
    ValueLoss: { x_axis: [], y_axis: [] },
    MeanEntropy: { x_axis: [], y_axis: [] },
    PolicyGradientNorm: { x_axis: [], y_axis: [] },
    ValueGradientNorm: { x_axis: [], y_axis: [] },
  };

  const { metricsByType = EMPTY_METRICS } = $props<{
    metricsByType?: TrainingMetricsByType;
  }>();

  let chartContainers = $state<Array<HTMLDivElement | undefined>>([]);
  let chartInstances: Array<ECharts | undefined> = [];

  const getMetricDataForChart = (chartIdx: number): MetricsData => {
    const metricType = CHART_METRIC_ORDER[chartIdx];
    if (!metricType) {
      return { x_axis: [], y_axis: [] };
    }
    return metricsByType[metricType] ?? { x_axis: [], y_axis: [] };
  };

  const isMetricDisabled = (chartIdx: number): boolean => {
    const metricData = getMetricDataForChart(chartIdx);
    return metricData.x_axis.length === 0 || metricData.y_axis.length === 0;
  };

  const buildOption = (
    chart: (typeof TRAINING_METRICS_CHARTS)[number],
    chartIdx: number,
  ): EChartsOption => {
    const metricData = getMetricDataForChart(chartIdx);
    const hasData = !isMetricDisabled(chartIdx);

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
        data: metricData.x_axis,
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
        show: hasData,
        trigger: "axis",
      },
      series: [
        {
          type: "line",
          smooth: true,
          showSymbol: false,
          lineStyle: {
            width: 2,
            color: hasData ? "#6fa7ff" : "#666666",
          },
          data: metricData.y_axis,
        },
      ],
      graphic: hasData
        ? []
        : [
            {
              type: "text",
              left: "center",
              top: "middle",
              style: {
                text: "No data",
                fill: "#6b6b6b",
                fontSize: 11,
                fontWeight: 500,
              },
            },
          ],
    };
  };

  const resizeCharts = (): void => {
    for (const chart of chartInstances) {
      chart?.resize();
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
      chartInstances[idx] = instance;
    });

    window.addEventListener("resize", resizeCharts);
  });

  $effect(() => {
    metricsByType;
    TRAINING_METRICS_CHARTS.forEach((chart, idx) => {
      const instance = chartInstances[idx];
      if (!instance) {
        return;
      }
      instance.setOption(buildOption(chart, idx), true);
    });
  });

  onDestroy(() => {
    window.removeEventListener("resize", resizeCharts);
    for (const chart of chartInstances) {
      chart?.dispose();
    }
    chartInstances = [];
  });
</script>

<div class="training-runtime-dashboard">
  {#each TRAINING_METRICS_CHARTS as chart, idx}
    <div class="training-metric-chart-card" style:opacity={isMetricDisabled(idx) ? 0.72 : 1}>
      <div
        class="training-metric-chart"
        bind:this={chartContainers[idx]}
        data-metric={chart.metric_name}
        style:pointer-events={isMetricDisabled(idx) ? "none" : "auto"}
      ></div>
    </div>
  {/each}
</div>
