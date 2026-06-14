use std::sync::{Arc, Mutex};

use rocket::tokio::sync::mpsc;
use tch::{Device, Kind, Tensor};

use crate::commondef::{
    testing::{ModelForward, StopTestMessage, TestGraph, TestUpdateMessage},
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
            println!("Running test for {:?} timesteps", num_timesteps);
            for i in 0..num_timesteps {
                let wstate = simulator.get_model_state();
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
                let cstate = simulator.get_model_cartesian_state();
                simulator.forward_dyn(&controls);
                let cstate_next = simulator.get_model_cartesian_state();
                let contact_pairs = simulator.get_contact_body_index_pairs();
                let reward =
                    model.get_reward(&cstate, &cstate_next, &sampled_action, Some(&contact_pairs));
                println!("Got reward at run_test {:?} {:?}", i, reward);
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
