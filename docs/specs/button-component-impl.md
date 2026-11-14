# Button component implementation plan

1. Add the approved behavior and accessibility contract to the feature spec.
2. Add `src/lib/button.svelte` using Svelte 5 runes and native button
   attributes, with controlled `active` and `variant` props.
3. Add the button state rules to `src/app.css`, keeping all component styling
   centralized and based on the documented design tokens.
4. Load the global stylesheet in the root layout.
5. Run Svelte type checking and a production build from `client`.
