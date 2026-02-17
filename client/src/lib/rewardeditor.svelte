<script lang="ts">
  import CodeEditor from "./codeeditor.svelte";

  let rewardFunctionCode = $state(`fn get_reward(&self, 
        cstate: &Tensor, 
        cstate_next: Option<&Tensor>, 
        action: &Tensor) -> f64 {
        let cube_pos = 
        |state: &Tensor| -> (f64, f64, f64) {
            let state = state.to_kind(tch::Kind::Double);
            let state_vals: Vec<f64> = Vec::<f64>::try_from(state)
                .expect("Expected cartesian state to be a 1D f64 tensor");
            if state_vals.len() % 7 != 0 {
                panic!(
                    "Expected cartesian state len to be 7*nbody, got {}",
                    state_vals.len()
                );
            }
            let nbody = state_vals.len() / 7;
            if self.target_cube_body_index >= nbody {
                panic!(
                    "target_cube body index {} out of bounds for nbody {}",
                    self.target_cube_body_index, nbody
                );
            }
            let cube_x_idx = self.target_cube_body_index * 3;
            (
                state_vals[cube_x_idx],
                state_vals[cube_x_idx + 1],
                state_vals[cube_x_idx + 2],
            )
        };

        let curr_state = cstate_next.unwrap_or(cstate);
        let (curr_x, curr_y, curr_z) = cube_pos(curr_state);
        let dx = curr_x - self.target_center_xy.0;
        let dy = curr_y - self.target_center_xy.1;
        let dz = curr_z - self.plane_z;
        let curr_dist = (dx * dx + dy * dy + dz * dz).sqrt();
        let r_dist = -curr_dist * self.w_dist;

        let r_prog = if cstate_next.is_some() {
            let (prev_x, prev_y, prev_z) = cube_pos(cstate);
            let prev_dx = prev_x - self.target_center_xy.0;
            let prev_dy = prev_y - self.target_center_xy.1;
            let prev_dz = prev_z - self.plane_z;
            let prev_dist = (prev_dx * prev_dx + prev_dy * prev_dy + prev_dz * prev_dz).sqrt();
            (prev_dist - curr_dist) * self.w_prog
        } else {
            0.0
        };

        let r_success = if curr_dist <= self.target_radius {
            self.w_success
        } else {
            0.0
        };

        let act_penalty = {
            let act_sq = action.pow_tensor_scalar(2.0).sum(tch::Kind::Double);
            let act_sq_v = f64::try_from(act_sq).expect("Expected action to be f64 tensor");
            -self.w_act * act_sq_v
        };

        r_dist + r_prog + r_success + act_penalty
    }`);
</script>

<div class="reward-editor">
  <CodeEditor
    code={rewardFunctionCode}
    onCodeChange={(nextCode) => {
      rewardFunctionCode = nextCode;
    }}
  />
</div>
