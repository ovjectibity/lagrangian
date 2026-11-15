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
- Render each cell's text and, optionally, one Button using any existing
  Button variant.
- Render optional icon-only row-level Buttons as the final cell in each row.
  Their accessible labels remain available to assistive technology.
- Preserve semantic table markup and allow horizontal scrolling on narrow
  viewports.

## Non-goals

- This version does not provide sorting, pagination, virtualisation,
  selectable rows, inline editing, or expandable row content.
