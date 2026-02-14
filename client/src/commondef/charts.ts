export type TrainingMetricsChart =
  | AvgRewardChart
  | MeanPolicyLossChart
  | ValueLossChart
  | MeanEntropyChart;

export interface AvgRewardChart {
  metric_name: "Mean reward";
  y_axis: "Mean reward";
  x_axis: "Epoch";
  type: "LineChart";
}

export interface MeanPolicyLossChart {
  metric_name: "Mean policy loss";
  y_axis: "Mean policy loss";
  x_axis: "Update step";
  type: "LineChart";
}

export interface ValueLossChart {
  metric_name: "Value loss";
  y_axis: "Value loss";
  x_axis: "Update step";
  type: "LineChart";
}

export interface MeanEntropyChart {
  metric_name: "Mean entropy";
  y_axis: "Mean entropy";
  x_axis: "Epoch";
  type: "LineChart";
}

export const TRAINING_METRICS_CHARTS: TrainingMetricsChart[] = [
  {
    metric_name: "Mean reward",
    y_axis: "Mean reward",
    x_axis: "Epoch",
    type: "LineChart",
  },
  {
    metric_name: "Mean entropy",
    y_axis: "Mean entropy",
    x_axis: "Epoch",
    type: "LineChart",
  },
  {
    metric_name: "Value loss",
    y_axis: "Value loss",
    x_axis: "Update step",
    type: "LineChart",
  },
  {
    metric_name: "Mean policy loss",
    y_axis: "Mean policy loss",
    x_axis: "Update step",
    type: "LineChart",
  },
];
