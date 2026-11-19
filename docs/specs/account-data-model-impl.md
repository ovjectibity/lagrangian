# Account data model implementation

1. Extend existing User/UserRole and add AccessKey/Group with native enums
   in commondef/account.rs, preserving the user's module rename.
2. Derive serde, SQLx, Schemars, Utoipa, and Typeshare support using existing
   dependencies and document public types.
3. Correct the existing ORM import from users to account, without changing
   SQL queries or applying migrations.
4. Format the touched type file and run cargo check from app.
