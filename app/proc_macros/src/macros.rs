//! Procedural macros for struct variants.

use proc_macro::TokenStream;

mod struct_variants;

/// Retains a struct and creates a `<snake_name>_variant!`
/// helper, or uses the helper name supplied as an argument.
///
/// Supports named fields without generics. Helpers are visible
/// within the current crate and may be imported across modules.
/// Field types resolve at the variant invocation site; import
/// their names there or use qualified paths in the source.
/// The dependency must be named `app_proc_macros`.
///
/// Variants declare their own derives. By default only field
/// documentation and conditional compilation attributes carry
/// over. Use `field_attributes: [serde, sqlx]` to copy additional
/// attributes by their first path segment. Added fields use the
/// attributes written on those fields.
///
/// Retained fields keep source order; additions are appended.
/// A removed name can be added again with a new definition;
/// it appears at the end. No conversions are generated.
///
/// ```
/// mod records {
///     #[app_proc_macros::variant_source]
///     pub struct Workload {
///         pub id: String,
///         pub graph: Vec<String>,
///         pub status: String,
///     }
/// }
///
/// use records::workload_variant;
/// workload_variant! {
///     #[derive(Debug)]
///     pub struct WorkloadRow {
///         remove: [graph],
///         add: { pub graph_id: String },
///     }
/// }
/// let row = WorkloadRow {
///     id: "run-1".into(),
///     status: "running".into(),
///     graph_id: "graph-1".into(),
/// };
/// assert_eq!(row.status, "running");
/// ```
///
/// ```
/// #[app_proc_macros::variant_source(user_fields)]
/// struct User { id: String, email: String }
/// user_fields! {
///     struct UserId { remove: [email] }
/// }
/// let user = UserId { id: "user-1".into() };
/// assert_eq!(user.id, "user-1");
/// ```
///
/// Unknown fields are rejected at compile time:
///
/// ```compile_fail
/// #[app_proc_macros::variant_source]
/// struct User { id: String }
/// user_variant! {
///     struct Invalid { remove: [email] }
/// }
/// ```
///
/// Adding a duplicate field is also rejected:
///
/// ```compile_fail
/// #[app_proc_macros::variant_source]
/// struct User { id: String }
/// user_variant! {
///     struct Invalid { add: { id: u32 } }
/// }
/// ```
#[proc_macro_attribute]
pub fn variant_source(
    args: TokenStream,
    item: TokenStream,
) -> TokenStream {
    struct_variants::source(args.into(), item.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Builds a struct from tokens supplied by a source helper.
///
/// Invoke the helper created by [`variant_source`] instead of
/// calling this implementation macro directly.
#[doc(hidden)]
#[proc_macro]
pub fn struct_variant(input: TokenStream) -> TokenStream {
    struct_variants::variant(input.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
