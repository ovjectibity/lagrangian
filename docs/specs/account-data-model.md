# Account data model

Implement the account schema from the Notion WIP Data model in the existing
app/src/commondef/account.rs. Keep IDs as String, consistent with User,
and timestamps as chrono DateTime<Utc>.

- AccessKey: user_id, name, created_at, status.
- AccessKeyStatus: expired, active, deleted.
- User: user_id, group_id, created_at, updated_at, name, email, role.
- UserRole: admin.
- Group: group_id, name, created_at, updated_at, login_type.
- LoginType: direct, sso_google, sso_github.

Use native enums with snake_case serialization and SQL text mapping. Keep
types suitable for SQLx rows and future schema/type generation. Add only
documented fields; key material and additional access-key IDs are unspecified.
Database migrations and query changes are outside this step.
