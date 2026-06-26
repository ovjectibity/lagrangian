use crate::simulator::mujoco;
use crate::simulator::Simulator;
use crate::simulator;

pub struct MjSimulator {
    model: *mut mujoco::mjModel,
    data: *mut mujoco::mjData,
}

// Safety: Access to the simulator is synchronized through a Mutex, so moving
// this owner between threads does not permit concurrent pointer use.
unsafe impl Send for MjSimulator {}

impl MjSimulator {
    pub fn new(model_path: String) -> Result<Self, String> {
        unsafe {
            if let Some((model, data)) = simulator::load_model(model_path, false) {
                mujoco::mj_forward(model, data);
                Ok(MjSimulator { model, data })
            } else {
                Err("Failed to load MuJoCo model".to_string())
            }
        }
    }
}

impl Drop for MjSimulator {
    fn drop(&mut self) {
        println!("Cleaning up model held by the simulator");
        unsafe {
            simulator::cleanup_model(self.model, self.data);
        }
    }
}

impl Simulator for MjSimulator {
    fn get_world_body(&self) -> Option<crate::commondef::model::WorldBody> {
        unsafe { simulator::create_model(self.model, self.data) }
    }
}
