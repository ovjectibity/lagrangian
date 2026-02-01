pub mod manipulator;

pub trait SimulatorUtilities {
    //Get methods for forward dynamics steps
    fn forward();
}

pub struct ControlModelInput {
    //Get the world state & model state for reward calculation + model input
}

pub enum ControlOutput {
    Direct(Vec<f64>),
    //Inverse dynamics calculations to 
    //be handled by the simulator directly
    TargetPositions(Vec<f64>)
}

pub trait PluginHandle {
    fn create_plugin_handle() -> Self;
    fn output_control(&self, 
        input: ControlModelInput) -> ControlOutput; 
}