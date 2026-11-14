# Search bar and toast components

## Goal

Add reusable Svelte 5 SearchBar and Toast components using the dashboard
prototype's compact control and notification patterns.

## SearchBar contract

- Expose a bindable string `value`, an accessible label, and a placeholder.
- Render a semantic `type="search"` input with a 16px leading search icon.
- Use the shared 32px control height, dark input surface, and visible focus
  treatment from the prototype.

## Toast contract

- Expose bindable `open`, a `message`, and a configurable `duration` in
  milliseconds; duration defaults to 2500.
- Render centered at the bottom of the viewport with a jet-black surface and
  polite live announcement semantics.
- Automatically set `open` to false after the duration and reset that timer
  whenever the message changes while the toast is open.
- Include an accessible top-right dismissal control.
- Respect reduced-motion preferences.

## Non-goals

- This first Toast component has no severity variants, action button, or
  notification queue.
