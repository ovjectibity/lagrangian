# Selection controls

## Goal

Add three reusable Svelte 5 selection controls following the dashboard
prototype: a single-choice multi-value switch, a single-select dropdown, and
a multi-select dropdown.

## Shared contract

- Each component is controlled through a bindable value owned by its parent.
- Controls are 32px high, use 12px control text, 7px corners, and visible
  keyboard focus.
- Switch tracks and open dropdown menus use jet-black interiors; dropdown
  triggers retain their prototype surface color.
- Dropdowns accept a `bordered` variant by default and a `text` variant. Text
  dropdowns are transparent and borderless at rest, exposing only the value
  label and chevron; they receive a grey surface on hover and while open.
- Popups use a shared 148px minimum width, independent of trigger variant, so
  text-only triggers retain compact controls while their menus have the same
  roomy width as bordered dropdowns.
- Dropdowns are custom controls, not native browser selects.
- Escape closes an open menu and returns focus to its trigger; Tab closes it.
- Arrow keys, Home, and End move focus between enabled menu choices.

## Multi-value switch

- Accept exactly two, three, or four string options and one selected value.
- Render a `radiogroup`; each option uses `role="radio"` and updates the
  selected value on click or keyboard activation.
- The track is jet black; the selected option uses the compact prototype
  selected surface.

## Single-select dropdown

- Accept labelled options, one selected value, and an accessible label.
- The trigger displays the selected option and chevron; choosing an item
  updates the value and closes the menu.
- The popup uses listbox/option semantics. Its selected row remains visibly
  selected.

## Multi-select dropdown

- Accept labelled options, selected string values, an accessible label, and
  an empty-state placeholder.
- The trigger shows the first selected label followed by `+N` for remaining
  selections; it shows the placeholder when none are selected.
- The popup stays open while choices are toggled. Items expose checkbox
  semantics and a checkmark when selected.

## Non-goals

- Menus do not support remote data, filtering, typeahead search, or portals.
- The controls do not manage application-level mutual exclusion beyond the
  selection data supplied by their parent.
