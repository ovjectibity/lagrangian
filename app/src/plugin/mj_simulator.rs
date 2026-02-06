use tch::{Kind, Tensor};

use crate::mujoco;
use crate::plugin::Simulator;
use crate::simulator;

pub struct MjSimulator {
    model: *mut mujoco::mjModel,
    data: *mut mujoco::mjData,
}

impl MjSimulator {
    pub fn new(model_path: String) -> Result<Self, String> {
        unsafe {
            if let Some((model, data)) = 
            simulator::load_model(model_path) {
                mujoco::mj_forward(model, data);
                Ok(MjSimulator { model, data })
            } else {
                Err("Failed to load MuJoCo model".to_string())
            }
        }
    }

    fn nq(&self) -> usize {
        unsafe { (*self.model).nq as usize }
    }

    fn nv(&self) -> usize {
        unsafe { (*self.model).nv as usize }
    }

    fn nu(&self) -> usize {
        unsafe { (*self.model).nu as usize }
    }

    fn qpos(&self) -> Vec<f64> {
        let nq = self.nq();
        unsafe {
            let qpos_ptr = (*self.data).qpos;
            std::slice::from_raw_parts(qpos_ptr, nq).to_vec()
        }
    }

    fn qvel(&self) -> Vec<f64> {
        let nv = self.nv();
        unsafe {
            let qvel_ptr = (*self.data).qvel;
            std::slice::from_raw_parts(qvel_ptr, nv).to_vec()
        }
    }

    fn is_torque_actuator(&self) -> bool {
        let nu = self.nu();
        if nu == 0 {
            return true;
        }
        unsafe {
            let biastype = (*self.model).actuator_biastype.add(0).read();
            biastype == mujoco::mjtBias__mjBIAS_NONE as i32
        }
    }
}

impl Drop for MjSimulator {
    fn drop(&mut self) {
        unsafe {
            simulator::cleanup_model(self.model, self.data);
        }
    }
}

impl Simulator for MjSimulator {
    fn forward_dyn(&self, controls: &Tensor) {
        let controls = controls.to_kind(Kind::Double);
        let ctrl_vals: Vec<f64> = 
            Vec::<f64>::try_from(controls)
            .expect("Expected controls to be a 1D f64 tensor");
        let nu = self.nu();
        if ctrl_vals.len() != nu {
            panic!(
                "Control dimension mismatch. Expected {}, got {}",
                nu,
                ctrl_vals.len()
            );
        }
        unsafe {
            let ctrl_ptr = (*self.data).ctrl;
            for (i, val) in ctrl_vals.iter().enumerate() {
                ctrl_ptr.add(i).write(*val);
            }
            simulator::run_dynamics_step(self.model, self.data);
        }
    }

    fn inverse_dyn(&self, acc: &Tensor) -> Tensor {
        let acc = acc.to_kind(Kind::Double);
        let acc_vals: Vec<f64> = 
            Vec::<f64>::try_from(acc)
            .expect("Expected acc to be a 1D f64 tensor");
        let nv = self.nv();
        if acc_vals.len() != nv {
            panic!(
                "Acceleration dimension mismatch. Expected {}, got {}",
                nv,
                acc_vals.len()
            );
        }
        unsafe {
            let qacc_ptr = (*self.data).qacc;
            for (i, val) in acc_vals.iter().enumerate() {
                qacc_ptr.add(i).write(*val);
            }
            mujoco::mj_inverse(self.model, self.data);
            let qfrc_ptr = (*self.data).qfrc_inverse;
            let qfrc = 
                std::slice::from_raw_parts(
                    qfrc_ptr, nv).to_vec();
            Tensor::from_slice(&qfrc)
        }
    } 

    fn get_model_state(&self) -> Tensor {
        let nq = self.nq();
        unsafe {
            let qpos_ptr = (*self.data).qpos;
            let qpos = 
                std::slice::from_raw_parts(
                    qpos_ptr, nq).to_vec();
            Tensor::from_slice(&qpos)
        }
    }

    fn get_target_controls_from_target_positions(
        &self,
        target_pos_diff: &Tensor,
    ) -> Tensor {
        let nu = self.nu();
        let nq = self.nq();
        let nv = self.nv();
        let target_pos_diff = 
            target_pos_diff.to_kind(Kind::Double);
        let target_diff_vals: Vec<f64> = 
            Vec::<f64>::try_from(target_pos_diff)
            .expect("Expected target_pos_diff to be a 1D f64 tensor");
        if target_diff_vals.len() != nq {
            panic!(
                "Target position diff dimension mismatch. Expected {}, got {}",
                nq,
                target_diff_vals.len()
            );
        }

        let qpos = self.qpos();
        let target_pos: Vec<f64> = qpos
            .iter()
            .zip(target_diff_vals.iter())
            .map(|(q, dq)| q + dq)
            .collect();

        if !self.is_torque_actuator() {
            let ctrl_vals: Vec<f64> = target_pos.into_iter().take(nu).collect();
            return Tensor::from_slice(&ctrl_vals);
        }

        let t = 0.1_f64;
        let qvel = self.qvel();
        if qvel.len() != nv {
            panic!(
                "Velocity dimension mismatch. Expected {}, got {}",
                nv,
                qvel.len()
            );
        }
        let qacc_des: Vec<f64> = target_diff_vals
            .iter()
            .zip(qvel.iter())
            .map(|(dq, v)| ((dq / t) - v) / t)
            .collect();
        let qacc_tensor = Tensor::from_slice(&qacc_des);
        let qfrc = self.inverse_dyn(&qacc_tensor);
        let qfrc_vals: Vec<f64> = Vec::<f64>::try_from(qfrc)
            .expect("Expected qfrc to be a 1D f64 tensor");
        let ctrl_vals: Vec<f64> = qfrc_vals.into_iter().take(nu).collect();
        Tensor::from_slice(&ctrl_vals)
    }
}
