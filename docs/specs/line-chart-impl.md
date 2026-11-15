# Line chart implementation plan

1. Add Apache ECharts as a client dependency.
2. Create `src/lib/composite/line-chart.svelte` with typed line data props,
   ECharts lifecycle handling, responsive resizing, and optional
   MultiValueSwitch, SingleSelectDropdown, and MultiSelectDropdown props.
3. Add shared chart panel styles to `client/src/app.css` and export the
   component through `src/lib/index.ts`.
4. Add a line-chart example to the component showcase with a MultiValueSwitch
   and SingleSelectDropdown in the chart header.
5. Run `npm run check` and `npm run build` from `client`.
