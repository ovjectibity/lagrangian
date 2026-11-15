# Card

Reusable dashboard card based on the workload browser prototype.

- Thumbnail supplied as an image URL with alternative text.
- Optional monospace thumbnail title overlaid at the top left.
- ID aligned left and status aligned right below the thumbnail.
- Title, subtitle, optional labels, and secondary information footer.
- Labels always follow the subtitle with consistent spacing; extra card
  height must not push them down.
- Sub-info and optional progress form a footer anchored to the card bottom.
- Status text may have a caller-specified dot colour.
- Labels and secondary information accept any number of strings.
- Optional progress accepts a percentage (clamped to 0–100) and an optional
  caller-provided completion-time label. Render below the secondary info,
  using a thin copper bar and accessible progress semantics. Hide the entire
  section when progress is omitted; do not calculate time estimates.
- Keep the card semantic and allow content to wrap without overflowing.
- Include examples in the component showcase.

Navigation, selection, and card actions are outside this first version.
