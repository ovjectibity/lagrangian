# Line chart

## Goal

Add a reusable ECharts-based LineChart composite that follows the dashboard's
dark chart language and supports the established selection controls.

## Contract

- Require a title, x-axis labels, and one or more named numeric line series.
- Render a responsive Apache ECharts canvas with dark grid lines, muted axis
  labels, muted-copper primary line treatment, a readable legend, and a dark
  tooltip.
- Support an optional MultiValueSwitch, SingleSelectDropdown, and
  MultiSelectDropdown in the chart header beside the title. The chart does not
  accept arbitrary controls or TabSwitcher.
- Observe container resizing and resize the ECharts instance accordingly.
- Use accessible chart labelling and ECharts' built-in aria support.

## Non-goals

- This first version supports only line series.
- It does not provide zooming, click-to-filter behavior, annotations, chart
  type selection, server data loading, or built-in control-state management.
