use std::collections::HashMap;

use tch::{Kind, Tensor};

use crate::commondef::training::ModelInputSpec;
use crate::simulator::mujoco;
use crate::simulator::Simulator;
use crate::simulator;

pub struct MjSimulator {
    target_position: Vec<f64>,
    model: *mut mujoco::mjModel,
    data: *mut mujoco::mjData,
}

// Safety: Access to the simulator in training is synchronized through a Mutex,
// so moving this owner between threads does not permit concurrent pointer use.
unsafe impl Send for MjSimulator {}

impl MjSimulator {
    pub fn new(model_path: String) -> Result<Self, String> {
        unsafe {
            if let Some((model, data)) = simulator::load_model(model_path, false) {
                mujoco::mj_forward(model, data);
                let mut simulator = MjSimulator {
                    target_position: Vec::new(),
                    model,
                    data,
                };
                simulator.target_position = simulator.get_actuated_qpos_targets();
                Ok(simulator)
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

    fn clamp_controls_to_actuator_range(&self, controls: &[f64]) -> Vec<f64> {
        let nu = self.nu() as usize;
        if controls.len() != nu {
            panic!(
                "Control dimension mismatch. Expected {}, got {}",
                nu,
                controls.len()
            );
        }
        let mut clamped: Vec<f64> = Vec::with_capacity(nu);
        unsafe {
            let ctrllimited_ptr = (*self.model).actuator_ctrllimited;
            let ctrlrange_ptr = (*self.model).actuator_ctrlrange;
            assert!(!ctrlrange_ptr.is_null());
            for (i, val) in controls.iter().enumerate() {
                let is_limited = if ctrllimited_ptr.is_null() {
                    true
                } else {
                    ctrllimited_ptr.add(i).read() != 0
                };
                if !is_limited {
                    clamped.push(*val);
                    continue;
                }
                let min = ctrlrange_ptr.add(i * 2).read();
                let max = ctrlrange_ptr.add(i * 2 + 1).read();
                clamped.push(val.clamp(min, max));
            }
        }
        clamped
    }

    fn get_actuated_qpos_targets(&self) -> Vec<f64> {
        let nu = self.nu() as usize;
        let qpos = self.qpos();
        let actuator_joint_indices = self.get_actuator_joint_indices();
        if actuator_joint_indices.len() != nu {
            panic!(
                "Actuator index mapping mismatch. Expected {}, got {}",
                nu,
                actuator_joint_indices.len()
            );
        }
        let mut targets: Vec<f64> = Vec::with_capacity(nu);
        for (qpos_idx, _) in actuator_joint_indices.iter() {
            targets.push(qpos[*qpos_idx]);
        }
        self.clamp_controls_to_actuator_range(&targets)
    }

    fn get_qdof_values(&self) -> Vec<f64> {
        let nu = self.nu() as usize;
        let qpos = self.qpos();
        let mut result: Vec<f64> = Vec::new();
        unsafe {
            let jnt_type_ptr = (*self.model).jnt_type;
            let actuator_trnid_ptr = (*self.model).actuator_trnid;
            let jnt_qposadr_ptr = (*self.model).jnt_qposadr;
            assert!(!jnt_type_ptr.is_null() && !actuator_trnid_ptr.is_null() && !jnt_qposadr_ptr.is_null());
            for actuator_idx in 0..nu {
                let joint_id = actuator_trnid_ptr.add(actuator_idx * 2).read() as usize;
                let qpos_adr = jnt_qposadr_ptr.add(joint_id).read() as usize;
                let jnt_type = jnt_type_ptr.add(joint_id).read() as u32;
                let n_qpos = match jnt_type {
                    mujoco::mjtJoint__mjJNT_FREE => 7,
                    mujoco::mjtJoint__mjJNT_BALL => 4,
                    _ => 1, // hinge or slide
                };
                for j in 0..n_qpos {
                    result.push(qpos[qpos_adr + j]);
                }
            }
        }
        result
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
    fn modify_target_position(&mut self, target_pos_diff: &Tensor) -> Tensor {
        let nu = self.nu() as usize;
        let target_pos_diff = target_pos_diff.to_kind(Kind::Double);
        let target_diff_vals: Vec<f64> =
            Vec::<f64>::try_from(target_pos_diff).expect("Expected target_pos_diff 1D f64 tensor");
        if target_diff_vals.len() != nu {
            panic!(
                "Target position diff dimension mismatch. Expected {}, got {}",
                nu,
                target_diff_vals.len()
            );
        }
        if self.target_position.len() != nu {
            self.target_position = self.get_actuated_qpos_targets();
        }
        let updated_targets: Vec<f64> = self
            .target_position
            .iter()
            .zip(target_diff_vals.iter())
            .map(|(target, diff)| target + diff)
            .collect();
        self.target_position = self.clamp_controls_to_actuator_range(&updated_targets);
        Tensor::from_slice(&self.target_position)
    }

    fn reset_state(&mut self) {
        unsafe {
            mujoco::mj_resetData(self.model, self.data);
            mujoco::mj_forward(self.model, self.data);
            //Reset target_position here:
            self.target_position = self.get_actuated_qpos_targets();
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
            // println!("DEBUG: Applying these controls: {:?}",ctrl_vals);
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

    fn get_model_state(&self, input_spec: &ModelInputSpec) -> HashMap<String, Tensor> {
        let mut result: HashMap<String, Tensor> = HashMap::new();
        if input_spec.qdof_state {
            result.insert("qdof".to_string(), Tensor::from_slice(&self.get_qdof_values()));
        }
        if input_spec.qpos_state {
            result.insert("qpos".to_string(), Tensor::from_slice(&self.qpos()));
        }
        if input_spec.qvel_state {
            result.insert("qvel".to_string(), Tensor::from_slice(&self.qvel()));
        }
        if input_spec.xpos_state {
            let nbody = unsafe { (*self.model).nbody as usize };
            let xpos = unsafe { std::slice::from_raw_parts((*self.data).xpos, nbody * 3).to_vec() };
            result.insert("xpos".to_string(), Tensor::from_slice(&xpos));
        }
        if input_spec.xquat_state {
            let nbody = unsafe { (*self.model).nbody as usize };
            let xquat = unsafe { std::slice::from_raw_parts((*self.data).xquat, nbody * 4).to_vec() };
            result.insert("xquat".to_string(), Tensor::from_slice(&xquat));
        }
        if input_spec.contact_pairs {
            let pairs = self.get_contact_body_index_pairs();
            let flat: Vec<f64> = pairs.iter().flat_map(|(a, b)| [*a as f64, *b as f64]).collect();
            result.insert("contact_pairs".to_string(), Tensor::from_slice(&flat));
        }
        // all_cameras: Warp not available in Rust path — no "cameras" key
        if result.is_empty() {
            result.insert("qpos".to_string(), Tensor::from_slice(&self.qpos()));
            result.insert("qvel".to_string(), Tensor::from_slice(&self.qvel()));
        }
        result
    }

    fn get_model_cartesian_state(&self) -> Tensor {
        unsafe {
            let nbody = (*self.model).nbody as usize;
            let xpos_ptr = (*self.data).xpos;
            let xquat_ptr = (*self.data).xquat;
            assert!(!xpos_ptr.is_null());
            assert!(!xquat_ptr.is_null());
            let mut xstate = std::slice::from_raw_parts(xpos_ptr, nbody * 3).to_vec();
            xstate.extend(std::slice::from_raw_parts(xquat_ptr, nbody * 4).to_vec());
            Tensor::from_slice(&xstate)
        }
    }

    fn get_world_body(&self) -> Option<crate::commondef::model::WorldBody> {
        unsafe { simulator::create_model(self.model, self.data) }
    }

    fn get_simul_update(&self) -> (Vec<f64>, Vec<f64>) {
        unsafe {
            let nbody = (*self.model).nbody as usize;
            let xpos = std::slice::from_raw_parts((*self.data).xpos, nbody * 3).to_vec();
            let xquat = std::slice::from_raw_parts((*self.data).xquat, nbody * 4).to_vec();
            (xpos, xquat)
        }
    }

    fn get_contact_body_index_pairs(&self) -> Vec<(usize, usize)> {
        unsafe {
            let ncon = (*self.data).ncon as usize;
            let contact_ptr = (*self.data).contact;
            let geom_bodyid_ptr = (*self.model).geom_bodyid;
            if contact_ptr.is_null() || geom_bodyid_ptr.is_null() {
                return Vec::new();
            }
            let mut contact_pairs: Vec<(usize, usize)> = Vec::with_capacity(ncon);
            for i in 0..ncon {
                let contact = &*contact_ptr.add(i);
                if contact.geom1 < 0 || contact.geom2 < 0 {
                    continue;
                }
                let body1 = geom_bodyid_ptr.add(contact.geom1 as usize).read();
                let body2 = geom_bodyid_ptr.add(contact.geom2 as usize).read();
                if body1 < 0 || body2 < 0 {
                    continue;
                }
                contact_pairs.push((body1 as usize, body2 as usize));
            }
            contact_pairs
        }
    }

    fn get_target_controls_from_target_positions(&self, target_pos_diff: &Tensor) -> Tensor {
        let nu = self.nu();
        let nv = self.nv();
        let target_pos_diff = target_pos_diff.to_kind(Kind::Double);
        let target_diff_vals: Vec<f64> = Vec::<f64>::try_from(target_pos_diff)
            .expect("Expected target_pos_diff to be a 1D f64 tensor");
        if target_diff_vals.len() != nu as usize {
            panic!(
                "Target position diff dimension mismatch. Expected {}, got {}",
                nu,
                target_diff_vals.len()
            );
        }

        let actuator_joint_indices = self.get_actuator_joint_indices();
        if actuator_joint_indices.len() != nu as usize {
            panic!(
                "Actuator index mapping mismatch. Expected {}, got {}",
                nu,
                actuator_joint_indices.len()
            );
        }

        if !self.is_torque_actuator() {
            println!("DEBUG: Treating as a position actuator");
            let qpos = self.qpos();
            let mut ctrl_vals: Vec<f64> = Vec::with_capacity(nu as usize);
            for (i, (qpos_idx, _)) in actuator_joint_indices.iter().enumerate() {
                ctrl_vals.push(qpos[*qpos_idx] + target_diff_vals[i]);
            }
            let ctrl_vals = self.clamp_controls_to_actuator_range(&ctrl_vals);
            return Tensor::from_slice(&ctrl_vals);
        }
        println!("DEBUG: Treating as a torque actuator");

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
        for (i, (_, dof_idx)) in actuator_joint_indices.iter().enumerate() {
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
        let ctrl_vals = self.clamp_controls_to_actuator_range(&ctrl_vals);
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
        let actuator_joint_indices = self.get_actuator_joint_indices();
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
            let mut ctrl_vals: Vec<f64> = Vec::with_capacity(nu as usize);
            for (i, (qpos_idx, _)) in actuator_joint_indices.iter().enumerate() {
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
