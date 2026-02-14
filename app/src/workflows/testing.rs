use std::sync::{Arc, Mutex};

use rocket::tokio::sync::mpsc;
use tch::{Device, Kind, Tensor};

use crate::commondef::{
    messages::{ModelForward, SimulUpdateMessage, StopSimulationMessage},
    testing::SimulationParams,
    workermessages::TestingTaskMessage,
};
use crate::plugin::{ControlPolicyModel, Simulator};

pub struct TestingWorkflow {
    id: String,
    simulator: Arc<Mutex<dyn Simulator + Send>>,
    model: Arc<Mutex<dyn ControlPolicyModel + Send>>,
    params: SimulationParams,
}

impl TestingWorkflow {
    pub fn new(
        id: String,
        simulator: Arc<Mutex<dyn Simulator + Send>>,
        model: Arc<Mutex<dyn ControlPolicyModel + Send>>,
        params: SimulationParams,
    ) -> Self {
        TestingWorkflow {
            id: id,
            simulator: simulator,
            model: model,
            params: params,
        }
    }

    fn send_stop(tx_test_t: &mpsc::Sender<TestingTaskMessage>, id: String) {
        let _ =
            tx_test_t.blocking_send(TestingTaskMessage::StopSimulation(StopSimulationMessage {
                id: id,
            }));
    }

    fn send_simul_update(
        tx_test_t: &mpsc::Sender<TestingTaskMessage>,
        id: String,
        xpos: Vec<f64>,
        xquat: Vec<f64>,
    ) {
        let _ =
            tx_test_t.blocking_send(TestingTaskMessage::SimulationUpdates(SimulUpdateMessage {
                id: id,
                updates: vec![ModelForward {
                    xpos: xpos,
                    xquat: xquat,
                }],
            }));
    }

    pub fn run_test(&mut self, tx_test_t: mpsc::Sender<TestingTaskMessage>) {
        if let Ok(simulator) = self.simulator.lock() {
            simulator.reset_state();
        } else {
            eprintln!("run_test: failed to lock simulator for reset");
            Self::send_stop(&tx_test_t, self.id.clone());
            return;
        }

        for _ in 0..self.params.num_timesteps {
            let controls = {
                if let Ok(simulator) = self.simulator.lock() {
                    let nu = simulator.nu();
                    Tensor::zeros([nu], (Kind::Double, Device::Cpu))
                } else {
                    eprintln!("run_test: failed to lock simulator for controls");
                    Self::send_stop(&tx_test_t, self.id.clone());
                    return;
                }
            };

            {
                if let Ok(simulator) = self.simulator.lock() {
                    simulator.forward_dyn(&controls);
                    let (xpos, xquat) = simulator.get_simul_update();
                    Self::send_simul_update(&tx_test_t, self.id.clone(), xpos, xquat);
                } else {
                    eprintln!("run_test: failed to lock simulator for step");
                    Self::send_stop(&tx_test_t, self.id.clone());
                    return;
                }
            }
        }
        Self::send_stop(&tx_test_t, self.id.clone());
    }
}
