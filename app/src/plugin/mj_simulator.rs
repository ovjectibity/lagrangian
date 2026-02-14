use tch::{Kind, Tensor};

use crate::mujoco;
use crate::plugin::Simulator;
use crate::simulator;

pub struct MjSimulator {
    model: *mut mujoco::mjModel,
    data: *mut mujoco::mjData,
}

// Safety: Access to the simulator in training is synchronized through a Mutex,
// so moving this owner between threads does not permit concurrent pointer use.
unsafe impl Send for MjSimulator {}

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

    pub fn qpos(&self) -> Vec<f64> {
        let nq = self.nq();
        unsafe {
            let qpos_ptr = (*self.data).qpos;
            std::slice::from_raw_parts(qpos_ptr, nq as usize).to_vec()
        }
    }

    pub fn qvel(&self) -> Vec<f64> {
        let nv = self.nv();
        unsafe {
            let qvel_ptr = (*self.data).qvel;
            std::slice::from_raw_parts(qvel_ptr, nv as usize).to_vec()
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

    // Returns (qpos_index, dof_index) per actuator in actuator order.
    fn get_actuator_joint_indices(&self) -> Vec<(usize, usize)> {
        let nu = self.nu() as usize;
        let nq = self.nq() as usize;
        let nv = self.nv() as usize;
        let mut indices: Vec<(usize, usize)> = Vec::with_capacity(nu);
        unsafe {
            let actuator_trnid_ptr = (*self.model).actuator_trnid;
            let jnt_qposadr_ptr = (*self.model).jnt_qposadr;
            let jnt_dofadr_ptr = (*self.model).jnt_dofadr;
            assert!(!actuator_trnid_ptr.is_null());
            assert!(!jnt_qposadr_ptr.is_null());
            assert!(!jnt_dofadr_ptr.is_null());
            for actuator_idx in 0..nu {
                // actuator_trnid is a 2-int tuple per actuator.
                let joint_id = actuator_trnid_ptr.add(actuator_idx * 2).read();
                if joint_id < 0 {
                    panic!(
                        "Actuator {} is not attached to a valid joint (joint_id={})",
                        actuator_idx, joint_id
                    );
                }
                let qpos_adr = jnt_qposadr_ptr.add(joint_id as usize).read();
                let dof_adr = jnt_dofadr_ptr.add(joint_id as usize).read();
                if qpos_adr < 0 || qpos_adr as usize >= nq {
                    panic!(
                        "Invalid qpos address {} for actuator {} (nq={})",
                        qpos_adr, actuator_idx, nq
                    );
                }
                if dof_adr < 0 || dof_adr as usize >= nv {
                    panic!(
                        "Invalid dof address {} for actuator {} (nv={})",
                        dof_adr, actuator_idx, nv
                    );
                }
                indices.push((qpos_adr as usize, dof_adr as usize));
            }
        }
        indices
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
    fn reset_state(&self) {
        unsafe {
            mujoco::mj_resetData(self.model, self.data);
            mujoco::mj_forward(self.model, self.data);
        }
    }

    fn nq(&self) -> i64 {
        unsafe { (*self.model).nq as i64 }
    }

    fn nv(&self) -> i64 {
        unsafe { (*self.model).nv as i64 }
    }

    fn nu(&self) -> i64 {
        unsafe { (*self.model).nu as i64 }
    }

    fn forward_dyn(&self, controls: &Tensor) {
        let controls = controls.to_kind(Kind::Double);
        let ctrl_vals: Vec<f64> =
            Vec::<f64>::try_from(controls).expect("Expected controls to be a 1D f64 tensor");
        let nu = self.nu();
        if ctrl_vals.len() != nu as usize {
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
            Vec::<f64>::try_from(acc).expect("Expected acc to be a 1D f64 tensor");
        let nv = self.nv();
        if acc_vals.len() != nv as usize {
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
            let qfrc = std::slice::from_raw_parts(qfrc_ptr, nv as usize).to_vec();
            Tensor::from_slice(&qfrc)
        }
    }

    fn get_model_state(&self) -> Tensor {
        let nq = self.nq();
        let nv = self.nv();
        unsafe {
            let qpos_ptr = (*self.data).qpos;
            let qvel_ptr = (*self.data).qvel;
            assert!(!qpos_ptr.is_null());
            assert!(!qvel_ptr.is_null());
            assert!(nq > 0);
            let mut qposvel = std::slice::from_raw_parts(qpos_ptr, nq as usize).to_vec();
            qposvel.extend(std::slice::from_raw_parts(qvel_ptr, nv as usize).to_vec());
            Tensor::from_slice(&qposvel)
        }
    }

    fn get_model_cartesian_state(&self) -> Tensor {
        unsafe {
            let nbody = (*self.model).nbody as usize;
            let xpos_ptr = (*self.data).xpos;
            let xquat_ptr = (*self.data).xquat;
            assert!(!xpos_ptr.is_null());
            assert!(!xquat_ptr.is_null());
            let mut xstate =
                std::slice::from_raw_parts(
                    xpos_ptr, nbody * 3).to_vec();
            xstate.extend(
                std::slice::from_raw_parts(
                    xquat_ptr, nbody * 4).to_vec(),
            );
            Tensor::from_slice(&xstate)
        }
    }

    fn get_world_body(&self) -> Option<crate::commondef::model::WorldBody> {
        unsafe { simulator::create_model(self.model, self.data) }
    }

    fn get_simul_update(&self) -> (Vec<f64>, Vec<f64>) {
        unsafe {
            let nbody = (*self.model).nbody as usize;
            let xpos = std::slice::from_raw_parts(
                (*self.data).xpos, nbody * 3).to_vec();
            let xquat = std::slice::from_raw_parts(
                (*self.data).xquat, nbody * 4).to_vec();
            (xpos, xquat)
        }
    }

    fn get_target_controls_from_target_positions(
        &self, target_pos_diff: &Tensor) -> Tensor {
        let nu = self.nu();
        let nv = self.nv();
        let target_pos_diff = target_pos_diff.to_kind(Kind::Double);
        let target_diff_vals: Vec<f64> = Vec::<f64>::try_from(
            target_pos_diff)
            .expect("Expected target_pos_diff to be a 1D f64 tensor");
        if target_diff_vals.len() != nu as usize {
            panic!(
                "Target position diff dimension mismatch. Expected {}, got {}",
                nu,
                target_diff_vals.len()
            );
        }

        let actuator_joint_indices = 
            self.get_actuator_joint_indices();
        if actuator_joint_indices.len() != nu as usize {
            panic!(
                "Actuator index mapping mismatch. Expected {}, got {}",
                nu,
                actuator_joint_indices.len()
            );
        }

        if !self.is_torque_actuator() {
            let qpos = self.qpos();
            let mut ctrl_vals: Vec<f64> = Vec::with_capacity(nu as usize);
            for (i, (qpos_idx, _)) in 
            actuator_joint_indices.iter().enumerate() {
                ctrl_vals.push(qpos[*qpos_idx] + target_diff_vals[i]);
            }
            return Tensor::from_slice(&ctrl_vals);
        }

        let t = 0.1_f64;
        let qvel = self.qvel();
        if qvel.len() != nv as usize {
            panic!(
                "Velocity dimension mismatch. Expected {}, got {}",
                nv,
                qvel.len()
            );
        }
        let mut qacc_des: Vec<f64> = vec![0.0_f64; nv as usize];
        for (i, (_, dof_idx)) in 
        actuator_joint_indices.iter().enumerate() {
            let dq = target_diff_vals[i];
            let v = qvel[*dof_idx];
            qacc_des[*dof_idx] = ((dq / t) - v) / t;
        }
        let qacc_tensor = Tensor::from_slice(&qacc_des);
        let qfrc = self.inverse_dyn(&qacc_tensor);
        let qfrc_vals: Vec<f64> =
            Vec::<f64>::try_from(qfrc).expect("Expected qfrc to be a 1D f64 tensor");
        let mut ctrl_vals: Vec<f64> = Vec::with_capacity(nu as usize);
        for (_, dof_idx) in actuator_joint_indices.iter() {
            ctrl_vals.push(qfrc_vals[*dof_idx]);
        }
        Tensor::from_slice(&ctrl_vals)
    }

    fn get_target_controls_from_target_velocities(&self, target_vel: &Tensor) -> Tensor {
        let nu = self.nu();
        let nv = self.nv();
        let target_vel = target_vel.to_kind(Kind::Double);
        let target_vel_vals: Vec<f64> =
            Vec::<f64>::try_from(target_vel).expect("Expected target_vel to be a 1D f64 tensor");
        if target_vel_vals.len() != nu as usize {
            panic!(
                "Target velocity dimension mismatch. Expected {}, got {}",
                nu,
                target_vel_vals.len()
            );
        }
        let actuator_joint_indices = 
            self.get_actuator_joint_indices();
        if actuator_joint_indices.len() != nu as usize {
            panic!(
                "Actuator index mapping mismatch. Expected {}, got {}",
                nu,
                actuator_joint_indices.len()
            );
        }

        if !self.is_torque_actuator() {
            let qpos = self.qpos();
            let t = 0.1_f64;
            let mut ctrl_vals: Vec<f64> = 
                Vec::with_capacity(nu as usize);
            for (i, (qpos_idx, _)) 
            in actuator_joint_indices.iter().enumerate() {
                let q_target = qpos[*qpos_idx] + target_vel_vals[i] * t;
                ctrl_vals.push(q_target);
            }
            return Tensor::from_slice(&ctrl_vals);
        }

        let qvel = self.qvel();
        if qvel.len() != nv as usize {
            panic!(
                "Velocity dimension mismatch. Expected {}, got {}",
                nv,
                qvel.len()
            );
        }

        let t = 0.1_f64;
        let mut qacc_des: Vec<f64> = vec![0.0_f64; nv as usize];
        for (i, (_, dof_idx)) in actuator_joint_indices.iter().enumerate() {
            let target_v = target_vel_vals[i];
            let v_curr = qvel[*dof_idx];
            qacc_des[*dof_idx] = (target_v - v_curr) / t;
        }
        let qacc_tensor = Tensor::from_slice(&qacc_des);
        let qfrc = self.inverse_dyn(&qacc_tensor);
        let qfrc_vals: Vec<f64> =
            Vec::<f64>::try_from(qfrc).expect("Expected qfrc to be a 1D f64 tensor");
        let mut ctrl_vals: Vec<f64> = Vec::with_capacity(nu as usize);
        for (_, dof_idx) in actuator_joint_indices.iter() {
            ctrl_vals.push(qfrc_vals[*dof_idx]);
        }
        Tensor::from_slice(&ctrl_vals)
    }
}
