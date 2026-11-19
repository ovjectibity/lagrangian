# Project data model

Add Project to app/src/commondef/records/product.rs following the current
commondef reorganisation and the Notion relational product.project table.

Fields: project_id and user_id as String, created_at and updated_at as the
existing Timestamp alias, and name as String. Match the module's serde,
SQLx FromRow, Schemars, Utoipa, and Typeshare derives.

Keep the legacy record::Project and existing SQL queries unchanged. Module
import repair and database migration are separate from this type addition.
