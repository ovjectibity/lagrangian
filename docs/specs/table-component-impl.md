# Table component implementation plan

1. Create `src/lib/composite/table.svelte` with column-level cell Buttons,
   shared table-level row actions, row text data, and header controls.
2. Use the existing Button component for optional cell actions and compact,
   icon-only row actions aligned right within the final cell.
3. Add shared table styles to `client/src/app.css` and export the component
   through `src/lib/index.ts`.
4. Add a workload-style table example to the component showcase using table
   Buttons and existing select controls.
5. Run `npm run check` and `npm run build` from `client`.
