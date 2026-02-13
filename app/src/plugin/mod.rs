use tch::Tensor;

pub mod ppo_trainer;
pub mod mj_simulator;

pub trait Simulator {
    fn reset_state(&self);
    //Shape of u: [nu]
    fn forward_dyn(&self,controls: &Tensor);
    //Shape of acc: [nq]
    //Shape of output: [nv]
    fn inverse_dyn(&self,acc: &Tensor) -> Tensor;
    fn get_model_state(&self) -> Tensor;
    fn get_target_controls_from_target_positions(
        &self,target_pos: &Tensor) -> Tensor;
}

pub struct ControlModelInput {
    //Get the world state & model state for reward calculation + model input
    //Get sensor data
    //Generalised positions - nq
    //User controls
    //TODO: interface for the user to provide the reward calculations
    generalised_model_positions: Vec<f64>
}

pub enum ControlModelOutput {
    Direct(DirectActuation),
    //Inverse dynamics calculations to 
    //be handled by the simulator directly
    TargetPositions(TargetPositions)
}

pub struct DirectActuation {
    actuation: Vec<f64>
}

pub struct TargetPositions {
    positions: Vec<f64>
}

pub trait PluginHandle {
    fn process_input(&self, 
        input: ControlModelInput) -> ControlModelOutput;
}
