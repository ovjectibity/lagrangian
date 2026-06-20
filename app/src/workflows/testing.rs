use std::sync::{Arc, Mutex};

use rocket::tokio::sync::mpsc;
use tch::{Device, Kind, Tensor};

use crate::commondef::{
    testing::{ModelForward, StopTestMessage, TestGraph, TestUpdateMessage},
    training::ControlModel,
    workermessages::TestTaskMessage,
};
use crate::plugin::ControlPolicyOutputMode;
use crate::plugin::{ControlPolicyModel, Simulator};

pub struct TestingWorkflow {
    id: String,
    simulator: Arc<Mutex<dyn Simulator + Send>>,
    model: Arc<Mutex<dyn ControlPolicyModel + Send>>,
    graph: TestGraph,
}

impl TestingWorkflow {
    pub fn new(
        id: String,
        simulator: Arc<Mutex<dyn Simulator + Send>>,
        model: Arc<Mutex<dyn ControlPolicyModel + Send>>,
        graph: TestGraph,
    ) -> Self {
        TestingWorkflow {
            id,
            simulator,
            model,
            graph,
        }
    }

    fn send_stop(tx_test_t: &mpsc::Sender<TestTaskMessage>, id: String) {
        let _ =
            tx_test_t.blocking_send(TestTaskMessage::StopTestMessage(StopTestMessage {
                id: id,
            }));
    }

    fn send_simul_update(
        tx_test_t: &mpsc::Sender<TestTaskMessage>,
        id: String,
        xpos: Vec<f64>,
        xquat: Vec<f64>,
    ) {
        let _ =
            tx_test_t.blocking_send(TestTaskMessage::TestUpdateMessage(TestUpdateMessage {
                id: id,
                updates: vec![vec![ModelForward {
                    xpos: xpos,
                    xquat: xquat,
                }]],
            }));
    }

    pub fn run_test(&mut self, tx_test_t: mpsc::Sender<TestTaskMessage>) {
        if let Ok(mut simulator) = self.simulator.lock()
            && let Ok(model) = self.model.lock()
        {
            simulator.reset_state();
            let action_output_size = simulator.nu();
            let num_timesteps = self.graph.test_job_controller.test_params.num_timesteps;
            let control_input_spec = match &self.graph.test_job_controller.test_params.control_model {
                ControlModel::VLA(vla) => vla.input_spec.clone(),
                ControlModel::FFN(ffn) => ffn.input_spec.clone(),
            };
            println!("Running test for {:?} timesteps", num_timesteps);
            for i in 0..num_timesteps {
                let parts = simulator.get_model_state(&control_input_spec);
                let wstate = model.assemble_control_state(&parts);
                let action_gaussian_params = model.get_policy_output(&wstate);
                let s_action_gaussian_params = action_gaussian_params
                    .split_with_sizes([action_output_size, action_output_size], 0);
                let action_gaussian_mean = s_action_gaussian_params
                    .get(0)
                    .expect("Couldn't extract action_gaussian_mean");
                let action_gaussian_stdev = s_action_gaussian_params
                    .get(1)
                    .expect("Couldn't extract action_gaussian_stddev")
                    .exp();
                let eps = Tensor::randn([action_output_size], (Kind::Double, Device::Cpu));
                let sampled_action = action_gaussian_mean + &eps * &action_gaussian_stdev;
                let controls = match model.get_output_mode() {
                    ControlPolicyOutputMode::TargetTorques => sampled_action.shallow_clone(),
                    ControlPolicyOutputMode::TargetPositions => {
                        simulator.modify_target_position(&sampled_action)
                    }
                    // ControlPolicyOutputMode::TargetPositions => {
                    //     simulator
                    //         .get_target_controls_from_target_positions(
                    //             &sampled_action
                    //         )
                    // }
                    ControlPolicyOutputMode::TargetVelocities => {
                        simulator.get_target_controls_from_target_velocities(&sampled_action)
                    }
                };
                simulator.forward_dyn(&controls);
                let (xpos, xquat) = simulator.get_simul_update();
                Self::send_simul_update(&tx_test_t, self.id.clone(), xpos, xquat);
            }
        } else {
            eprintln!("run_test: failed to lock simulator or model during test run");
            Self::send_stop(&tx_test_t, self.id.clone());
            return;
        }
        Self::send_stop(&tx_test_t, self.id.clone());
    }
}
