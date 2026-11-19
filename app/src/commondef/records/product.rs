use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use typeshare::typeshare;
use utoipa::ToSchema;

/// Project record in the product schema.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    FromRow,
    ToSchema,
)]
#[typeshare]
pub struct Project {
    pub project_id: String,
    pub user_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub name: String,
}

/// Role assigned to an account user.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    sqlx::Type,
    ToSchema,
)]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[typeshare]
pub enum UserRole {
    Admin,
}

/// User record in the account schema.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    FromRow,
    ToSchema,
)]
#[typeshare]
pub struct User {
    pub user_id: String,
    pub group_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub name: String,
    pub email: String,
    pub role: UserRole,
}

/// Lifecycle status of an account access key.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    sqlx::Type,
    ToSchema,
)]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[typeshare]
pub enum AccessKeyStatus {
    Expired,
    Active,
    Deleted,
}

/// Access-key record in the product schema.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    FromRow,
    ToSchema,
)]
#[typeshare]
pub struct AccessKey {
    pub access_key_id: String,
    pub access_key_value: String,
    pub user_id: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub status: AccessKeyStatus,
}

/// Authentication method configured for an account group.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    sqlx::Type,
    ToSchema,
)]
#[serde(rename_all = "snake_case")]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[typeshare]
pub enum LoginType {
    Direct,
    SsoGoogle,
    SsoGithub,
}

/// Group record in the account schema.
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    FromRow,
    ToSchema,
)]
#[typeshare]
pub struct Group {
    pub group_id: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub login_type: LoginType,
}
