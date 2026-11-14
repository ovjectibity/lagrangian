# Selection controls implementation plan

1. Add the interaction, accessibility, and visual contract to the feature
   specification.
2. Create controlled Svelte 5 components for the switch and two dropdowns,
   including bordered and text-only dropdown trigger variants.
3. Add shared control and popup styles to `client/src/app.css`.
4. Export the controls from `src/lib/index.ts`.
5. Extend the isolated `/components` showcase and its route-local stylesheet
   with interactive examples.
6. Run `npm run check` and `npm run build` from `client`.
