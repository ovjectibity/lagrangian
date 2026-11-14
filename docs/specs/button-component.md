# Button component

## Goal

Provide one reusable Svelte 5 button component for the dashboard's atomic
controls. It must express the four approved visual-selection variants without
making selection ownership implicit.

## Contract

- Render a semantic native `button` and accept its normal button attributes.
- Render caller-provided text and/or an icon through its children snippet.
- Accept `variant` values `type-1` through `type-4`; default to `type-1`.
- Accept an `active` boolean; default to `false`.
- Accept `hasActiveState`; it defaults to `true`. When false, the component
  ignores `active` visually and omits `aria-pressed`.
- Expose selected toggle state to assistive technology with `aria-pressed`.
- Do not change `active` internally. The consuming control owns selection and
  updates `active` after its click handler runs when active state is enabled.

## Shared visual contract

- Controls are 32px high with 10px horizontal padding, 7px radius, 13px
  medium-weight text, and a 16px icon size.
- Values are calibrated from the dashboard prototype's breadcrumbs, tabs, and
  Copy JSON action. Active-hover content uses white for the extra highlight.
- All variants retain keyboard-visible focus and a disabled state.

## Variant state matrix

| Variant | Rest | Hover | Active | Active hover |
| --- | --- | --- | --- | --- |
| Type-1 | `#9999A4`, transparent | `#FFF`, transparent | `#D0D0D8`, transparent | `#FFF`, transparent |
| Type-2 | `#888891`, transparent | `#DEDEE5`, transparent | `#F0F0F2`, `#25252A` | `#FFF`, `#29292F` |
| Type-3 | `#9999A4`, transparent | `#DEDEE5`, `#25252A` | `#F0F0F2`, `#25252A` | `#FFF`, `#29292F` |
| Type-4 | `#D8D8DF`, `#202024` | `#FFF`, `#29292F` | `#EDEDF1`, `#25252A` | `#FFF`, `#29292F` |

## Non-goals

- This component does not include a tooltip, loading state, destructive
  treatment, or selection-group coordination.
