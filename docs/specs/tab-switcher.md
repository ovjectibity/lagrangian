# Tab switcher

## Goal

Add a composite Svelte 5 TabSwitcher built exclusively from the atomic
Type-2 Button component. It provides the compact workload-browser selection
mechanism without coupling the control to a specific page or tab panel.

## Contract

- Accept a string label for accessibility, two or more string options, and a
  bindable selected `value`.
- Render each option as a Type-2 Button with the selected option active.
- Update the selected value when an option is activated.
- Use a transparent, borderless 2px-inset track with the compact selected
  surface inherited from the Type-2 Button.
- Use a labelled group of native buttons; consuming views render the content
  associated with the selected value.

## Non-goals

- The component does not render or manage tab panels.
- It does not support disabled options, icons, overflow scrolling, or dynamic
  option creation in this initial version.
