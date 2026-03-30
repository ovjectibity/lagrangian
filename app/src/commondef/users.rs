use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

#[derive(
    Debug, Clone, Serialize, Deserialize, FromRow,
)]
pub struct User {
    pub user_id: String,
    pub created_at: DateTime<Utc>,
    pub name: String,
    pub email: String,
}
