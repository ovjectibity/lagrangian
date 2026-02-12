export interface WorkflowNodeSetting {
  key: string;
  label: string;
  value: number | string;
  min?: number;
  max?: number;
  step?: number;
  inputType?: "number" | "select" | "link";
  options?: Array<{ label: string; value: string }>;
  disabled?: boolean;
}

export interface WorkflowNodeInputAnchor {
  id: string;
  label: string;
}

export interface WorkflowNodeOutputAnchor {
  id: string;
  label: string;
}

export interface WorkflowNode {
  id: string;
  label: string;
  position: { x: number; y: number };
  inputs?: number;
  outputs?: number;
  nodeType?: string;
  settings?: WorkflowNodeSetting[];
  inputAnchors?: WorkflowNodeInputAnchor[];
  outputAnchors?: WorkflowNodeOutputAnchor[];
}

export interface WorkflowEdge {
  id: string;
  sourceNodeId: string;
  targetNodeId: string;
  sourceOutputAnchor: WorkflowNodeOutputAnchor;
  targetInputAnchor: WorkflowNodeInputAnchor;
}

export const TrainingMethodNode: WorkflowNode = {
  id: "training-method",
  label: "TrainingMethod",
  nodeType: "TrainingMethod",
  position: { x: 80, y: 90 },
  outputs: 1,
  outputAnchors: [{ id: "training-method-output", label: "TrainingMethodOutput" }],
  settings: [
    {
      key: "type",
      label: "type",
      value: "PPO",
      inputType: "select",
      options: [{ label: "PPO", value: "PPO" }],
    },
    { key: "num_epochs", label: "num_epochs", value: 10, min: 1, step: 1 },
    { key: "num_updates", label: "num_updates", value: 10, min: 1, step: 1 },
    { key: "epsilon", label: "epsilon", value: 0.2, min: 0, max: 1, step: 0.01 },
    { key: "gamma", label: "gamma", value: 0.99, min: 0, max: 1, step: 0.01 },
    { key: "lambda", label: "lambda", value: 0.95, min: 0, max: 1, step: 0.01 },
  ],
  inputAnchors: [
    { id: "control_model", label: "ControlModel" },
    { id: "value_model", label: "ValueModel" },
    { id: "reward_function", label: "RewardFunction" },
  ],
};

export const ControlModelNode: WorkflowNode = {
  id: "control-model",
  label: "ControlModel",
  nodeType: "ControlModel",
  position: { x: 520, y: 110 },
  inputs: 1,
  outputs: 1,
  outputAnchors: [
    { id: "control-model-output", label: "ControlModelOutput" },
  ],
  settings: [
    {
      key: "type",
      label: "type",
      value: "FFN",
      inputType: "select",
      options: [{ label: "FFN", value: "FFN" }],
    },
    { key: "num_hidden_layers", label: "num_hidden_layers", value: 2, min: 1, step: 1 },
    { key: "hidden_layer_dim", label: "hidden_layer_dim", value: 128, min: 1, step: 1 },
    { key: "output_dim", label: "output_dim", value: 8, min: 1, step: 1 },
    { key: "input_dim", label: "input_dim", value: 8, min: 1, step: 1 },
  ],
};

export const ValueModelNode: WorkflowNode = {
  id: "value-model",
  label: "ValueModel",
  nodeType: "ValueModel",
  position: { x: 520, y: 300 },
  inputs: 1,
  outputs: 1,
  outputAnchors: [
    { id: "value-model-output", label: "ValueModelOutput" },
  ],
  settings: [
    {
      key: "type",
      label: "type",
      value: "FFN",
      inputType: "select",
      options: [{ label: "FFN", value: "FFN" }],
    },
    { key: "num_hidden_layers", label: "num_hidden_layers", value: 2, min: 1, step: 1 },
    { key: "hidden_layer_dim", label: "hidden_layer_dim", value: 128, min: 1, step: 1 },
    { key: "output_dim", label: "output_dim", value: 1, min: 1, step: 1, disabled: true },
    { key: "input_dim", label: "input_dim", value: 8, min: 1, step: 1 },
  ],
};

export const RewardFunctionNode: WorkflowNode = {
  id: "reward-function",
  label: "RewardFunction",
  nodeType: "RewardFunction",
  position: { x: 520, y: 480 },
  inputs: 0,
  outputs: 1,
  outputAnchors: [
    { id: "reward-function-output", label: "RewardFunctionOutput" },
  ],
  settings: [
    {
      key: "function_code",
      label: "Function code",
      value: "Edit in Reward editor",
      inputType: "link",
    },
  ],
};

export const TrainingGraphNodes: WorkflowNode[] = [
  TrainingMethodNode,
  ControlModelNode,
  ValueModelNode,
  RewardFunctionNode,
];

export const TrainingGraphEdges: WorkflowEdge[] = [
  {
    id: "control-model-to-training-method",
    sourceNodeId: "control-model",
    targetNodeId: "training-method",
    sourceOutputAnchor: { id: "control-model-output", label: "ControlModelOutput" },
    targetInputAnchor: { id: "control_model", label: "ControlModel" },
  },
  {
    id: "value-model-to-training-method",
    sourceNodeId: "value-model",
    targetNodeId: "training-method",
    sourceOutputAnchor: { id: "value-model-output", label: "ValueModelOutput" },
    targetInputAnchor: { id: "value_model", label: "ValueModel" },
  },
  {
    id: "reward-function-to-training-method",
    sourceNodeId: "reward-function",
    targetNodeId: "training-method",
    sourceOutputAnchor: { id: "reward-function-output", label: "RewardFunctionOutput" },
    targetInputAnchor: { id: "reward_function", label: "RewardFunction" },
  },
];
