# Button showcase page

## Goal

Provide a dedicated `/components` page for visually and interactively checking
the four atomic Button variants before they are used in application screens.

## Behavior

- Show Type-1 through Type-4 in a consistent card layout.
- Each example starts unselected and toggles its controlled `active` state on
  click, so hover, selected, and selected-hover states can be inspected.
- Include a short description of each background rule and a small state label.
- Reuse the Button component; do not duplicate its styling or behavior.
- Keep the page responsive and keyboard-accessible.
- Keep showcase-only layout and presentation styles in
  `src/routes/components/components.css`, separate from shared application CSS.

## Non-goals

- This is a component reference page, not a production dashboard screen.
- It does not introduce routing navigation or a component documentation system.
