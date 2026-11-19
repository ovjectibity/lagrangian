//! Source helpers and explicit field transformations.

use std::collections::HashSet;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    Attribute, Field, Fields, Ident, ItemStruct, Token,
    Visibility, braced, bracketed,
};

fn named_fields(item: &ItemStruct) -> syn::Result<()> {
    if !item.generics.params.is_empty()
        || item.generics.where_clause.is_some()
    {
        return Err(syn::Error::new_spanned(
            &item.generics,
            "struct variants do not support generics yet",
        ));
    }
    if !matches!(item.fields, Fields::Named(_)) {
        return Err(syn::Error::new_spanned(
            item,
            "struct variants require named fields",
        ));
    }
    Ok(())
}

fn helper_name(name: &Ident) -> Ident {
    let letters: Vec<char> = name.to_string().chars().collect();
    let mut snake = String::new();
    for (index, letter) in letters.iter().enumerate() {
        let previous_lower =
            index > 0 && letters[index - 1].is_lowercase();
        let next_lower = letters
            .get(index + 1)
            .is_some_and(|next| next.is_lowercase());
        if letter.is_uppercase()
            && index > 0
            && (previous_lower || next_lower)
        {
            snake.push('_');
        }
        snake.extend(letter.to_lowercase());
    }
    format_ident!("{}_variant", snake)
}

pub(super) fn source(
    args: TokenStream,
    item: TokenStream,
) -> syn::Result<TokenStream> {
    let item: ItemStruct = syn::parse2(item)?;
    named_fields(&item)?;
    let helper = if args.is_empty() {
        helper_name(&item.ident)
    } else {
        syn::parse2::<Ident>(args)?
    };
    Ok(quote! {
        #item

        #[allow(unused_macros)]
        macro_rules! #helper {
            ($($variant:tt)*) => {
                ::app_proc_macros::struct_variant! {
                    { #item }
                    { $($variant)* }
                }
            };
        }

        #[allow(unused_imports)]
        pub(crate) use #helper;
    })
}

struct Variant {
    attrs: Vec<Attribute>,
    vis: Visibility,
    name: Ident,
    remove: Vec<Ident>,
    add: Vec<Field>,
    field_attributes: Vec<Ident>,
}

fn identifiers(input: ParseStream) -> syn::Result<Vec<Ident>> {
    let inner;
    bracketed!(inner in input);
    Ok(inner
        .parse_terminated(Ident::parse, Token![,])?
        .into_iter()
        .collect())
}

fn fields(input: ParseStream) -> syn::Result<Vec<Field>> {
    let inner;
    braced!(inner in input);
    Ok(inner
        .parse_terminated(Field::parse_named, Token![,])?
        .into_iter()
        .collect())
}

impl Parse for Variant {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis = input.parse()?;
        input.parse::<Token![struct]>()?;
        let name = input.parse()?;
        let mut result = Self {
            attrs,
            vis,
            name,
            remove: Vec::new(),
            add: Vec::new(),
            field_attributes: Vec::new(),
        };
        let inner;
        braced!(inner in input);
        let mut seen = HashSet::new();
        while !inner.is_empty() {
            let key: Ident = inner.parse()?;
            let label = key.to_string();
            if !seen.insert(label.clone()) {
                return Err(syn::Error::new(
                    key.span(),
                    "duplicate variant operation",
                ));
            }
            inner.parse::<Token![:]>()?;
            match label.as_str() {
                "remove" => {
                    result.remove = identifiers(&inner)?;
                }
                "add" => result.add = fields(&inner)?,
                "field_attributes" => {
                    result.field_attributes =
                        identifiers(&inner)?;
                }
                _ => {
                    return Err(syn::Error::new(
                        key.span(),
                        "expected remove, add, or field_attributes",
                    ));
                }
            }
            if !inner.is_empty() {
                inner.parse::<Token![,]>()?;
            }
        }
        Ok(result)
    }
}

struct VariantInput {
    source: ItemStruct,
    variant: Variant,
}

impl Parse for VariantInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let source;
        braced!(source in input);
        let variant;
        braced!(variant in input);
        let result = Self {
            source: source.parse()?,
            variant: variant.parse()?,
        };
        if !source.is_empty() || !variant.is_empty() {
            return Err(input.error("unexpected trailing tokens"));
        }
        Ok(result)
    }
}

fn field_name(field: &Field) -> syn::Result<&Ident> {
    field.ident.as_ref().ok_or_else(|| {
        syn::Error::new_spanned(field, "expected named field")
    })
}

fn validate_names(
    names: &[Ident],
    available: &HashSet<String>,
) -> syn::Result<()> {
    let mut seen = HashSet::new();
    for name in names {
        if !available.contains(&name.to_string()) {
            return Err(syn::Error::new(
                name.span(),
                format!("unknown source field `{name}`"),
            ));
        }
        if !seen.insert(name.to_string()) {
            return Err(syn::Error::new(
                name.span(),
                format!("duplicate field `{name}`"),
            ));
        }
    }
    Ok(())
}

pub(super) fn variant(
    input: TokenStream,
) -> syn::Result<TokenStream> {
    let VariantInput {
        mut source,
        variant,
    } = syn::parse2(input)?;
    named_fields(&source)?;
    let available: HashSet<String> = source
        .fields
        .iter()
        .map(field_name)
        .collect::<syn::Result<Vec<_>>>()?
        .into_iter()
        .map(ToString::to_string)
        .collect();
    validate_names(&variant.remove, &available)?;

    let mut output = Punctuated::<Field, Token![,]>::new();
    let mut names = HashSet::new();
    for mut field in source.fields {
        let name = field_name(&field)?;
        if variant.remove.contains(name) {
            continue;
        }
        names.insert(name.to_string());
        field.attrs.retain(|attr| {
            attr.path().segments.first().is_some_and(|seg| {
                matches!(
                    seg.ident.to_string().as_str(),
                    "doc" | "cfg" | "cfg_attr"
                ) || variant.field_attributes.contains(&seg.ident)
            })
        });
        output.push(field);
    }
    for field in variant.add {
        let name = field_name(&field)?;
        if !names.insert(name.to_string()) {
            return Err(syn::Error::new(
                name.span(),
                "added field duplicates a retained or added field",
            ));
        }
        output.push(field);
    }
    source.attrs = variant.attrs;
    source.vis = variant.vis;
    source.ident = variant.name;
    source.fields = Fields::Named(syn::FieldsNamed {
        brace_token: Default::default(),
        named: output,
    });
    Ok(quote! { #source })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(
        instructions: TokenStream,
    ) -> syn::Result<ItemStruct> {
        syn::parse2(variant(quote! {
            {
                struct Source {
                    /// Identifier.
                    #[serde(rename = "identifier")]
                    pub id: String,
                    pub graph: Vec<String>,
                    status: String,
                }
            }
            { #instructions }
        })?)
    }

    #[test]
    fn removes_and_adds() -> syn::Result<()> {
        let output = expand(quote! {
            #[derive(Debug)]
            pub struct Row {
                remove: [graph],
                add: { pub graph_id: String },
            }
        })?;
        assert_eq!(output.ident, "Row");
        assert!(matches!(output.vis, Visibility::Public(_)));
        let fields: Vec<_> = output.fields.iter().collect();
        let names: Vec<_> = fields
            .iter()
            .map(|field| {
                field_name(field).map(ToString::to_string)
            })
            .collect::<syn::Result<_>>()?;
        assert_eq!(names, ["id", "status", "graph_id"]);
        let status_type = &fields[1].ty;
        assert_eq!(quote!(#status_type).to_string(), "String");
        assert!(matches!(fields[1].vis, Visibility::Inherited));
        assert_eq!(fields[0].attrs.len(), 1);
        assert!(fields[0].attrs[0].path().is_ident("doc"));
        Ok(())
    }

    #[test]
    fn keeps_source_order_and_opted_in_attributes()
    -> syn::Result<()> {
        let output = expand(quote! {
            struct Selected {
                remove: [graph],
                field_attributes: [serde],
            }
        })?;
        let fields: Vec<_> = output.fields.iter().collect();
        assert_eq!(field_name(fields[0])?, "id");
        assert_eq!(field_name(fields[1])?, "status");
        assert_eq!(fields[0].attrs.len(), 2);
        Ok(())
    }

    #[test]
    fn rejects_invalid_operations() {
        let cases = [
            quote! { remove: [missing] },
            quote! { remove: [id, id] },
            quote! { keep: [id] },
            quote! { remove: [id], remove: [graph] },
            quote! { add: { id: u8 } },
            quote! { add: { extra: u8, extra: u16 } },
            quote! { replace: { id: u8 } },
            quote! { rename: [id] },
        ];
        for instructions in cases {
            assert!(
                expand(quote! {
                    struct Invalid { #instructions }
                })
                .is_err()
            );
        }
    }

    #[test]
    fn can_add_a_name_after_removing_it() -> syn::Result<()> {
        let output = expand(quote! {
            struct Row {
                remove: [id],
                add: { pub id: u32 },
            }
        })?;
        let fields: Vec<_> = output.fields.iter().collect();
        assert_eq!(field_name(fields[2])?, "id");
        let id_type = &fields[2].ty;
        assert_eq!(quote!(#id_type).to_string(), "u32");
        Ok(())
    }

    #[test]
    fn rejects_unsupported_sources() {
        for item in [
            quote! { struct Generic<T> { value: T } },
            quote! { struct Tuple(String); },
            quote! { struct Unit; },
        ] {
            assert!(source(TokenStream::new(), item).is_err());
        }
    }

    #[test]
    fn helper_names_handle_acronyms() {
        let name = format_ident!("APIResponse");
        assert_eq!(helper_name(&name), "api_response_variant");
    }
}
