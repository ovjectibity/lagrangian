# Table component implementation plan

1. Create `src/lib/composite/table.svelte` with typed column, cell, and
   row-action data props plus a header controls snippet.
2. Use the existing Button component for optional cell actions and compact,
   icon-only row actions.
3. Add shared table styles to `client/src/app.css` and export the component
   through `src/lib/index.ts`.
4. Add a workload-style table example to the component showcase using table
   Buttons and existing select controls.
5. Run `npm run check` and `npm run build` from `client`.
