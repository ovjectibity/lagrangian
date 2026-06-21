//! Proc-macro helpers for Lagrangian's API layer.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as Ts2};
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Attribute, Ident, ItemFn, LitInt,
    LitStr, Token, Type,
};

// ── Argument types ────────────────────────────

struct RouteDocArgs {
    tag: LitStr,
    ok: Option<Type>,
    body: Option<Type>,
    multipart: bool,
    sse: bool,
    errors: Vec<u16>,
    params: Vec<(String, String)>,
}

impl Parse for RouteDocArgs {
    fn parse(
        input: ParseStream,
    ) -> syn::Result<Self> {
        let mut tag: Option<LitStr> = None;
        let mut ok: Option<Type> = None;
        let mut body: Option<Type> = None;
        let mut multipart = false;
        let mut sse = false;
        let mut errors: Vec<u16> = Vec::new();
        let mut params: Vec<(String, String)> =
            Vec::new();

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            match key.to_string().as_str() {
                "tag" => {
                    input.parse::<Token![=]>()?;
                    tag = Some(
                        input.parse::<LitStr>()?,
                    );
                }
                "ok" => {
                    input.parse::<Token![=]>()?;
                    ok =
                        Some(input.parse::<Type>()?);
                }
                "body" => {
                    input.parse::<Token![=]>()?;
                    body =
                        Some(input.parse::<Type>()?);
                }
                "multipart" => {
                    multipart = true;
                }
                "sse" => {
                    sse = true;
                }
                "errors" => {
                    let inner;
                    syn::parenthesized!(
                        inner in input
                    );
                    let nums: Punctuated<
                        LitInt, Token![,]
                    > = inner.parse_terminated(
                        LitInt::parse,
                        Token![,],
                    )?;
                    for n in nums {
                        errors.push(
                            n.base10_parse::<u16>()?,
                        );
                    }
                }
                "params" => {
                    let inner;
                    syn::parenthesized!(
                        inner in input
                    );
                    while !inner.is_empty() {
                        let name: Ident =
                            inner.parse()?;
                        inner
                            .parse::<Token![=]>()?;
                        let desc: LitStr =
                            inner.parse()?;
                        params.push((
                            name.to_string(),
                            desc.value(),
                        ));
                        if inner.peek(Token![,]) {
                            inner.parse::<
                                Token![,]
                            >()?;
                        }
                    }
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown `route_doc` \
                             argument: `{other}`"
                        ),
                    ));
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        let tag = tag.ok_or_else(|| {
            syn::Error::new(
                Span::call_site(),
                "route_doc requires \
                 `tag = \"...\"`",
            )
        })?;

        Ok(RouteDocArgs {
            tag,
            ok,
            body,
            multipart,
            sse,
            errors,
            params,
        })
    }
}

// ── Rocket route extraction ───────────────────

const METHODS: &[&str] = &[
    "get", "post", "put", "delete",
    "patch", "head", "options",
];

/// Parses the path string from a Rocket route
/// attribute, consuming any trailing tokens so
/// `parse_args` does not reject them.
struct RocketPath(LitStr);

impl Parse for RocketPath {
    fn parse(
        input: ParseStream,
    ) -> syn::Result<Self> {
        let lit: LitStr = input.parse()?;
        // drain format/data kwargs
        let _: proc_macro2::TokenStream =
            input.parse()?;
        Ok(RocketPath(lit))
    }
}

/// Finds the first Rocket HTTP method attribute
/// and returns `(method, openapi_path)`.
fn extract_rocket_route(
    attrs: &[Attribute],
) -> Option<(String, String)> {
    for attr in attrs {
        let seg = match attr
            .path()
            .segments
            .last()
        {
            Some(s) => s,
            None => continue,
        };
        let method = seg.ident.to_string();
        if !METHODS.contains(&method.as_str()) {
            continue;
        }
        let rocket_path = match attr
            .parse_args::<RocketPath>()
        {
            Ok(p) => p,
            Err(_) => continue,
        };
        let openapi_path = rocket_path
            .0
            .value()
            .replace('<', "{")
            .replace('>', "}")
            // strip leading _ from unused path params
            // e.g. {_experiment_id} → {experiment_id}
            .replace("{_", "{");
        return Some((method, openapi_path));
    }
    None
}

// ── Code generation ───────────────────────────

fn error_desc(code: u16) -> &'static str {
    match code {
        400 => "Bad request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not found",
        409 => "Conflict",
        422 => "Unprocessable entity",
        _ => "Error",
    }
}

/// Reads the Rocket route attribute on the
/// annotated function and generates the
/// corresponding `#[utoipa::path(...)]`.
///
/// # Required
/// * `tag = "Tag name"`
///
/// # Optional
/// * `ok = ResponseType` — 200 response body
/// * `body = RequestType` — JSON request body
/// * `multipart` — multipart/form-data upload
/// * `sse` — SSE stream response
/// * `errors(404, 422)` — extra error codes
/// * `params(id = "Description")` — path params
///
/// # Example
/// ```ignore
/// #[route_doc(tag = "Models", ok = ModelsResponse)]
/// #[get("/models")]
/// pub async fn list_models() -> ...
/// ```
#[proc_macro_attribute]
pub fn route_doc(
    args: TokenStream,
    item: TokenStream,
) -> TokenStream {
    let args =
        parse_macro_input!(args as RouteDocArgs);
    let func =
        parse_macro_input!(item as ItemFn);

    let (method_str, path) =
        match extract_rocket_route(&func.attrs) {
            Some(mp) => mp,
            None => {
                return syn::Error::new(
                    Span::call_site(),
                    "route_doc: no Rocket route \
                     attribute (#[get], #[post], \
                     …) found on this function",
                )
                .to_compile_error()
                .into();
            }
        };

    let tag = &args.tag;
    let method =
        Ident::new(&method_str, Span::call_site());

    // ── Responses ─────────────────────────────

    let mut resp: Vec<Ts2> = Vec::new();

    if args.sse {
        let sse_body: Ts2 =
            if let Some(ref ok_ty) = args.ok {
                quote! { #ok_ty }
            } else {
                quote! { String }
            };
        resp.push(quote! {
            (
                status = 200,
                description =
                    "Server-sent event stream",
                body = #sse_body,
                content_type = "text/event-stream",
            )
        });
    } else {
        if let Some(ref ok_ty) = args.ok {
            resp.push(quote! {
                (
                    status = 200,
                    description = "OK",
                    body = #ok_ty,
                )
            });
        }
        resp.push(quote! {
            (
                status = 500,
                description =
                    "Internal server error",
            )
        });
    }

    for code in &args.errors {
        let desc = error_desc(*code);
        resp.push(quote! {
            (status = #code, description = #desc)
        });
    }

    // ── Request body ──────────────────────────
    // multipart: utoipa requires `content = Type`
    // inside request_body(...); skip for now since
    // TempFile has no ToSchema impl.

    let req_body: Ts2 =
        if let Some(ref body_ty) = args.body {
            quote! { request_body = #body_ty, }
        } else {
            quote! {}
        };

    // ── Path params ───────────────────────────

    let param_ts: Vec<Ts2> = args
        .params
        .iter()
        .map(|(name, desc)| {
            quote! {
                (#name, Path, description = #desc)
            }
        })
        .collect();

    let params_section: Ts2 =
        if param_ts.is_empty() {
            quote! {}
        } else {
            quote! { params(#(#param_ts),*), }
        };

    // ── Emit ──────────────────────────────────

    quote! {
        #[utoipa::path(
            #method,
            path = #path,
            tag = #tag,
            #req_body
            #params_section
            responses(#(#resp),*)
        )]
        #func
    }
    .into()
}
