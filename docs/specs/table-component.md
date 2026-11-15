# Table component

## Goal

Add a reusable dark data-table composite for presenting arbitrary columns and
rows with dashboard-style controls and actions.

## Contract

- Render a left-aligned title with optional secondary text aligned right,
  immediately before any table-level controls.
- Accept a controls snippet so callers can compose existing Buttons,
  SingleSelectDropdowns, or MultiSelectDropdowns in the header.
- Accept any number of column definitions and rows.
- Configure optional cell Buttons on column definitions so every row in
  that column renders the same Button configuration.
- Reserve configurable `contentWidth` pixels (default 64px) for text before
  each column Button, followed by an 8px gap,
  so Buttons align across rows without drifting to the cell edge.
- Allow text to shrink when space is limited, truncate it with an ellipsis,
  and expose the complete value through its hover title.
- Configure row actions once at table level so every row has the same
  actions. Action callbacks receive the row index and click event.
- Render optional icon-only row-level Buttons aligned to the row's right
  edge, with at least an 8px gap from the final cell's content and without
  adding a separate actions column.
  Their accessible labels remain available to assistive technology.
- Preserve semantic table markup and allow horizontal scrolling on narrow
  viewports.

## Non-goals

- This version does not provide sorting, pagination, virtualisation,
  selectable rows, inline editing, or expandable row content.
