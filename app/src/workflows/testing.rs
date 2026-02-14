use std::sync::{Arc, Mutex};

use rocket::tokio::sync::mpsc;
use tch::{Device, Kind, Tensor};

use crate::commondef::{testing::SimulationParams, workermessages::SimulatorThreadMessage};
use crate::plugin::{ControlPolicyModel, Simulator};

pub struct TestingWorkflow {
    simulator: Arc<Mutex<dyn Simulator + Send>>,
    model: Arc<Mutex<dyn ControlPolicyModel + Send>>,
    params: SimulationParams,
}

impl TestingWorkflow {
    pub fn new(
        simulator: Arc<Mutex<dyn Simulator + Send>>,
        model: Arc<Mutex<dyn ControlPolicyModel + Send>>,
        params: SimulationParams,
    ) -> Self {
        TestingWorkflow {
            simulator: simulator,
            model: model,
            params: params,
        }
    }

    pub fn run_test(&mut self, tx: mpsc::Sender<SimulatorThreadMessage>) {
        if let Ok(simulator) = self.simulator.lock() {
            simulator.reset_state();
        } else {
            eprintln!("run_test: failed to lock simulator for reset");
            return;
        }

        {
            if let Ok(simulator) = self.simulator.lock() {
                if let Some(world_body) = simulator.get_world_body() {
                    let _ = tx
                        .blocking_send(SimulatorThreadMessage::new_load_model_message(world_body));
                }
            } else {
                eprintln!("run_test: failed to lock simulator for load message");
                return;
            }
        }

        for _ in 0..self.params.num_timesteps {
            let controls = {
                if let Ok(simulator) = self.simulator.lock() {
                    let nu = simulator.nu();
                    Tensor::zeros([nu], (Kind::Double, Device::Cpu))
                } else {
                    eprintln!("run_test: failed to lock simulator for controls");
                    return;
                }
            };

            {
                if let Ok(simulator) = self.simulator.lock() {
                    simulator.forward_dyn(&controls);
                    let (xpos, xquat) = simulator.get_simul_update();
                    let _ = tx.blocking_send(SimulatorThreadMessage::new_simul_update_message(
                        xpos, xquat,
                    ));
                } else {
                    eprintln!("run_test: failed to lock simulator for step");
                    return;
                }
            }
        }
    }
}
