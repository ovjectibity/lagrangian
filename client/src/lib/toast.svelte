<script lang="ts">
  let {
    duration = 2500,
    message,
    open = $bindable(false),
  }: {
    duration?: number;
    message: string;
    open?: boolean;
  } = $props();

  $effect(() => {
    const currentMessage = message;

    if (!open) {
      return;
    }

    const timeout = window.setTimeout(() => {
      open = false;
    }, duration);

    return () => {
      window.clearTimeout(timeout);
      void currentMessage;
    };
  });
</script>

<div
  aria-atomic="true"
  aria-hidden={!open}
  aria-live="polite"
  class:lg-toast--visible={open}
  class="lg-toast"
  inert={!open}
  role="status"
>
  <span>{message}</span>
  <button
    aria-label="Dismiss notification"
    class="lg-toast__dismiss"
    onclick={() => (open = false)}
    type="button"
  >
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="m6 6 12 12M18 6 6 18" />
    </svg>
  </button>
</div>
