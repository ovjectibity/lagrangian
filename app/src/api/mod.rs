pub mod checkpoints;
pub mod experiments;
pub mod models;
pub mod reward_validation;
pub mod testing;
pub mod training;


use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use rocket::tokio::sync::{
    Mutex as AsyncMutex,
    broadcast,
};
use rocket::tokio::task::JoinHandle;

pub struct ApiState {
    pub training_senders: AsyncMutex<
        HashMap<String, broadcast::Sender<String>>,
    >,
    pub training_stops: AsyncMutex<
        HashMap<
            String,
            (JoinHandle<()>, Arc<AtomicBool>),
        >,
    >,
    pub simul_senders: AsyncMutex<
        HashMap<String, broadcast::Sender<String>>,
    >,
    pub simul_tasks: AsyncMutex<
        HashMap<String, JoinHandle<()>>,
    >,
}

impl ApiState {
    pub fn new() -> Self {
        ApiState {
            training_senders: AsyncMutex::new(
                HashMap::new(),
            ),
            training_stops: AsyncMutex::new(
                HashMap::new(),
            ),
            simul_senders: AsyncMutex::new(
                HashMap::new(),
            ),
            simul_tasks: AsyncMutex::new(
                HashMap::new(),
            ),
        }
    }
}
