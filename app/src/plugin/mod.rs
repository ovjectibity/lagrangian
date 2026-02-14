use tch::Tensor;

pub mod mj_simulator;
pub mod ppo_trainer;

pub trait Simulator {
    fn reset_state(&self);
    fn nq(&self) -> i64;
    fn nu(&self) -> i64;
    fn nv(&self) -> i64;
    //Shape of u: [nu]
    fn forward_dyn(&self, controls: &Tensor);
    //Shape of acc: [nq]
    //Shape of output: [nv]
    fn inverse_dyn(&self, acc: &Tensor) -> Tensor;
    fn get_model_state(&self) -> Tensor;
    fn get_target_controls_from_target_positions(&self, target_pos: &Tensor) -> Tensor;
    fn get_target_controls_from_target_velocities(&self, target_vel: &Tensor) -> Tensor;
}

pub struct ControlModelInput {
    //Get the world state & model state for reward calculation + model input
    //Get sensor data
    //Generalised positions - nq
    //User controls
    //TODO: interface for the user to provide the reward calculations
    generalised_model_positions: Vec<f64>,
}

pub enum ControlModelOutput {
    TargetPositions,
    TargetVelocities,
    TargetTorques,
}

pub struct DirectActuation {
    actuation: Vec<f64>,
}

pub struct TargetPositions {
    positions: Vec<f64>,
}

pub trait PluginHandle {
    fn process_input(&self, input: ControlModelInput) -> ControlModelOutput;
}
