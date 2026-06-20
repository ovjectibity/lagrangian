import type { SimulationParams } from '../commondef/testing';
import type {
  CheckpointFrequency,
  ComputeInstance,
  ControlModel,
  MLPModel,
  TrainingAccelerationMode,
  TrainingJobController,
  TrainingMethod,
  ValueModel,
} from '../commondef/training';

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
  position: { x: 780, y: 360 },
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
    { key: "num_trajs", label: "num_trajs", value: 10, min: 1, step: 1 },
    {
      key: "num_steps_per_traj",
      label: "num_steps_per_traj",
      value: 10000,
      min: 1,
      step: 1,
    },
    { key: "num_epochs", label: "num_epochs", value: 50, min: 1, step: 1 },
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
  position: { x: 180, y: 90 },
  inputs: 1,
  outputs: 1,
  outputAnchors: [
    { id: "control-model-output", label: "ControlModelOutput" },
  ],
  settings: [
    {
      key: "type",
      label: "type",
      value: "MLP",
      inputType: "select",
      options: [
        { label: "MLP", value: "MLP" },
        { label: "Checkpoint", value: "Checkpoint" },
      ],
    },
    { key: "num_hidden_layers", label: "num_hidden_layers", value: 3, min: 1, step: 1 },
    { key: "hidden_layer_dim", label: "hidden_layer_dim", value: 512, min: 1, step: 1 },
    { key: "output_dim", label: "output_dim", value: 14, min: 1, step: 1 },
    { key: "input_dim", label: "input_dim", value: 55, min: 1, step: 1 },
    {
      key: "checkpoint_id",
      label: "Checkpoint ID",
      value: "",
      inputType: "select",
      options: [],
    },
  ],
};

export const ValueModelNode: WorkflowNode = {
  id: "value-model",
  label: "ValueModel",
  nodeType: "ValueModel",
  position: { x: 180, y: 360 },
  inputs: 1,
  outputs: 1,
  outputAnchors: [
    { id: "value-model-output", label: "ValueModelOutput" },
  ],
  settings: [
    {
      key: "type",
      label: "type",
      value: "MLP",
      inputType: "select",
      options: [
        { label: "MLP", value: "MLP" },
        { label: "Checkpoint", value: "Checkpoint" },
      ],
    },
    { key: "num_hidden_layers", label: "num_hidden_layers", value: 3, min: 1, step: 1 },
    { key: "hidden_layer_dim", label: "hidden_layer_dim", value: 256, min: 1, step: 1 },
    { key: "output_dim", label: "output_dim", value: 1, min: 1, step: 1, disabled: true },
    { key: "input_dim", label: "input_dim", value: 55, min: 1, step: 1 },
    {
      key: "checkpoint_id",
      label: "Checkpoint ID",
      value: "",
      inputType: "select",
      options: [],
    },
  ],
};

export const RewardFunctionNode: WorkflowNode = {
  id: "reward-function",
  label: "RewardFunction",
  nodeType: "RewardFunction",
  position: { x: 180, y: 630 },
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

export const TrainingJobControllerNode: WorkflowNode = {
  id: "training-job-controller",
  label: "TrainingJobController",
  nodeType: "TrainingJobController",
  position: { x: 1150, y: 360 },
  inputs: 1,
  outputs: 0,
  inputAnchors: [
    { id: "training_method", label: "TrainingMethod" },
  ],
  settings: [
    {
      key: "acceleration_mode",
      label: "acceleration_mode",
      value: "AllOnServer",
      inputType: "select",
      options: [
        { label: "AllOnServer", value: "AllOnServer" },
        { label: "MLGPUSimulationCPU", value: "MLGPUSimulationCPU" },
        { label: "MLGPUSimulationGPU", value: "MLGPUSimulationGPU" },
        { label: "MLCPUSimulationCPU", value: "MLCPUSimulationCPU" },
      ],
    },
    {
      key: "compute_instance",
      label: "compute_instance",
      value: "AWSG4dn",
      inputType: "select",
      options: [{ label: "AWSG4dn", value: "AWSG4dn" }],
    },
    {
      key: "checkpoint_frequency_type",
      label: "checkpoint_frequency_type",
      value: "PostEpoch",
      inputType: "select",
      options: [
        { label: "PostEpoch", value: "PostEpoch" },
        { label: "PostUpdate", value: "PostUpdate" },
        { label: "LastOnly", value: "LastOnly" },
      ],
    },
    { key: "checkpoint_frequency_n", label: "checkpoint_frequency_n", value: 1, min: 1, step: 1 },
  ],
};

export const TrainingGraphNodes: WorkflowNode[] = [
  TrainingMethodNode,
  ControlModelNode,
  ValueModelNode,
  TrainingJobControllerNode,
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
    id: "training-method-to-training-job-controller",
    sourceNodeId: "training-method",
    targetNodeId: "training-job-controller",
    sourceOutputAnchor: {
      id: "training-method-output",
      label: "TrainingMethodOutput",
    },
    targetInputAnchor: { id: "training_method", label: "TrainingMethod" },
  },
  {
    id: "reward-function-to-training-method",
    sourceNodeId: "reward-function",
    targetNodeId: "training-method",
    sourceOutputAnchor: { id: "reward-function-output", label: "RewardFunctionOutput" },
    targetInputAnchor: { id: "reward_function", label: "RewardFunction" },
  },
];

export const SimulationParametersNode: WorkflowNode = {
  id: "simulation-parameters",
  label: "Simulation Parameters",
  nodeType: "SimulationParameters",
  position: { x: 780, y: 240 },
  inputs: 1,
  outputs: 1,
  inputAnchors: [{ id: "control_model", label: "ControlModel" }],
  outputAnchors: [{ id: "simulation-parameters-output", label: "SimulationParametersOutput" }],
  settings: [
    {
      key: "policy_id",
      label: "policy_id",
      value: "default-policy",
      inputType: "select",
      options: [{ label: "default-policy", value: "default-policy" }],
    },
    { key: "num_timesteps", label: "num_timesteps", value: 100, min: 1, step: 1 },
  ],
};

export const TestingControlModelNode: WorkflowNode = {
  id: "testing-control-model",
  label: "ControlModel",
  nodeType: "ControlModel",
  position: { x: 180, y: 240 },
  inputs: 1,
  outputs: 1,
  outputAnchors: [
    { id: "testing-control-model-output", label: "ControlModelOutput" },
  ],
  settings: [
    {
      key: "type",
      label: "type",
      value: "MLP",
      inputType: "select",
      options: [{ label: "MLP", value: "MLP" }],
    },
    { key: "num_hidden_layers", label: "num_hidden_layers", value: 3, min: 1, step: 1 },
    { key: "hidden_layer_dim", label: "hidden_layer_dim", value: 512, min: 1, step: 1 },
    { key: "output_dim", label: "output_dim", value: 14, min: 1, step: 1 },
    { key: "input_dim", label: "input_dim", value: 55, min: 1, step: 1 },
  ],
};

export const TestingGraphNodes: WorkflowNode[] = [
  SimulationParametersNode,
  TestingControlModelNode,
];

export const TestingGraphEdges: WorkflowEdge[] = [
  {
    id: "testing-control-model-to-simulation-parameters",
    sourceNodeId: "testing-control-model",
    targetNodeId: "simulation-parameters",
    sourceOutputAnchor: {
      id: "testing-control-model-output",
      label: "ControlModelOutput",
    },
    targetInputAnchor: { id: "control_model", label: "ControlModel" },
  },
];

export const buildSimulationParams = (): SimulationParams => {
  
  return {
    policy_id: 'default-policy',
    num_timesteps: 
      SimulationParametersNode.settings?.find(
        (node) => node.key === "num_timesteps")?.value as number,
    control_model: {
      type: 'MLP',
      content: {
        num_hidden_layers: 3,
        hidden_layer_dim: 128,
        output_dim: 58,
        input_dim: 58,
      },
    },
  };
};

export const getNodeSettingValue = (
  nodeId: string,
  settingKey: string,
): number | string | undefined => {
  const node = TrainingGraphNodes.find((workflowNode) => workflowNode.id === nodeId);
  return node?.settings?.find((setting) => setting.key === settingKey)?.value;
};

export const getNumericSetting = (
  nodeId: string,
  settingKey: string,
  fallbackValue: number,
): number => {
  const value = getNodeSettingValue(nodeId, settingKey);
  return typeof value === 'number' ? value : fallbackValue;
};

export const getStringSetting = (
  nodeId: string,
  settingKey: string,
  fallbackValue: string,
): string => {
  const value = getNodeSettingValue(nodeId, settingKey);
  return typeof value === 'string' ? value : fallbackValue;
};

export const buildMlpModel = (nodeId: string): MLPModel => {
  return {
    num_hidden_layers: getNumericSetting(nodeId, 'num_hidden_layers', 2),
    hidden_layer_dim: getNumericSetting(nodeId, 'hidden_layer_dim', 128),
    output_dim: getNumericSetting(nodeId, 'output_dim', 1),
    input_dim: getNumericSetting(nodeId, 'input_dim', 58),
  };
};

export const buildTrainingWorkflow = (): TrainingMethod => {
  return {
    type: 'PPO',
    content: {
      num_trajs: getNumericSetting('training-method', 'num_trajs', 10),
      num_steps_per_traj: getNumericSetting('training-method', 'num_steps_per_traj', 10000),
      num_epochs: getNumericSetting('training-method', 'num_epochs', 10),
      num_updates: getNumericSetting('training-method', 'num_updates', 10),
      epsilon: getNumericSetting('training-method', 'epsilon', 0.2),
      gamma: getNumericSetting('training-method', 'gamma', 0.99),
      lambda: getNumericSetting('training-method', 'lambda', 0.95),
      reward_function: {
        reward_function_code: getStringSetting(
          'reward-function',
          'function_code',
          '',
        ),
      },
      control_model: ((): ControlModel => {
        const t = getStringSetting('control-model', 'type', 'MLP');
        return t === 'Checkpoint'
          ? { type: 'Checkpoint', content: getStringSetting('control-model', 'checkpoint_id', '') }
          : { type: 'MLP', content: buildMlpModel('control-model') };
      })(),
      value_model: ((): ValueModel => {
        const t = getStringSetting('value-model', 'type', 'MLP');
        return t === 'Checkpoint'
          ? { type: 'Checkpoint', content: getStringSetting('value-model', 'checkpoint_id', '') }
          : { type: 'MLP', content: buildMlpModel('value-model') };
      })(),
    },
  };
};

export const buildTrainingController = (): TrainingJobController => {
  const freqType = getStringSetting('training-job-controller', 'checkpoint_frequency_type', 'PostEpoch');
  const freqN = getNumericSetting('training-job-controller', 'checkpoint_frequency_n', 1);
  return {
    checkpoint_frequency: freqType === 'LastOnly'
      ? { type: 'LastOnly' }
      : { type: freqType as 'PostEpoch' | 'PostUpdate', content: freqN },
    training_method: buildTrainingWorkflow(),
    acceleration_mode: getStringSetting(
      'training-job-controller',
      'acceleration_mode',
      'AllOnServer',
    ) as TrainingAccelerationMode,
    compute_instance: getStringSetting(
      'training-job-controller',
      'compute_instance',
      'AWSG4dn',
    ) as ComputeInstance,
  };
};