export type TrainingMetricsChart =
  | AvgRewardChart
  | MeanPolicyLossChart
  | ValueLossChart
  | MeanEntropyChart;

export interface AvgRewardChart {
  metric_name: "Avg Rewards";
  y_axis: "Avg Reward";
  x_axis: "Number of iterations";
  type: "LineChart";
}

export interface MeanPolicyLossChart {
  metric_name: "Mean policy loss";
  y_axis: "Mean policy loss";
  x_axis: "Number of epochs";
  type: "LineChart";
}

export interface ValueLossChart {
  metric_name: "Value loss";
  y_axis: "Value loss";
  x_axis: "Number of epochs";
  type: "LineChart";
}

export interface MeanEntropyChart {
  metric_name: "Mean entropy";
  y_axis: "Mean entropy";
  x_axis: "Number of epochs";
  type: "LineChart";
}

export const TRAINING_METRICS_CHARTS: TrainingMetricsChart[] = [
  {
    metric_name: "Avg Rewards",
    y_axis: "Avg Reward",
    x_axis: "Number of iterations",
    type: "LineChart",
  },
  {
    metric_name: "Mean entropy",
    y_axis: "Mean entropy",
    x_axis: "Number of epochs",
    type: "LineChart",
  },
  {
    metric_name: "Value loss",
    y_axis: "Value loss",
    x_axis: "Number of epochs",
    type: "LineChart",
  },
  {
    metric_name: "Mean policy loss",
    y_axis: "Mean policy loss",
    x_axis: "Number of epochs",
    type: "LineChart",
  },
];
