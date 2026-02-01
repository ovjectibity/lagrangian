// use rand::Rng;
// use tch::{nn::{self, Module, OptimizerConfig}, no_grad, Tensor};

pub struct ManipulatorModel {
    
}

// pub struct Trainer {

// }

// impl Trainer {
//     pub fn train_an_epoch_val_ppo(
//         &mut self,
//         trajs: &Vec<Trajectory>) {
//         // GD on the mean square loss b/w the value tensor & the rtg tensor to train
//         if let (Some(valnet),Some(val_optim)) = 
//             (&self.valnet,&mut self.val_optim) {
//             //POSSIBLE BUG: Check if the RTG tensor isn't backpropping grads to the pol
//             let cube_state_reps_t = Self::get_cube_state_reps_tensor(&trajs);
//             let rtg_t = Self::get_rtg_tensor(trajs);
//             for i in 0..self.num_epochs {
//                 info!("Training {} policy epoch for PPO valnet update",i);
//                 let move_probs_t = valnet.forward(&cube_state_reps_t).squeeze_dim(2);
//                 println!("Size of rtg_t & move_probs: {:?} {:?}",rtg_t.size(),move_probs_t.size());
//                 let mse_loss = rtg_t.mse_loss(&move_probs_t, 
//                     tch::Reduction::Mean);
//                 info!(
//                     "Val net mse_loss tensor: {:?} {:?}",
//                     mse_loss.size(),
//                     mse_loss.to_device(tch::Device::Cpu)
//                 );
//                 mse_loss.print();
//                 val_optim.zero_grad();
//                 mse_loss.backward();
//                 val_optim.step();
//             }
//         }
//     }
// }