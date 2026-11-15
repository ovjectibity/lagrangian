<script lang="ts">
  import Button from "../atomic/button.svelte";

  let {
    body,
    onPrimary,
    onSecondary,
    open = $bindable(false),
    primaryLabel = "Confirm",
    primaryIntent = "positive",
    secondaryLabel = "Cancel",
    title,
  }: {
    body: string;
    onPrimary?: () => void;
    onSecondary?: () => void;
    open?: boolean;
    primaryLabel?: string;
    primaryIntent?: "positive" | "negative";
    secondaryLabel?: string;
    title: string;
  } = $props();

  const id = $props.id();
  let dialog = $state<HTMLDialogElement>();

  $effect(() => {
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  });

  const confirm = (): void => {
    open = false;
    onPrimary?.();
  };

  const cancel = (): void => {
    open = false;
    onSecondary?.();
  };

  const dismissOutside = (event: MouseEvent): void => {
    if (!dialog || event.target !== dialog) return;

    const bounds = dialog.getBoundingClientRect();
    if (
      event.clientX < bounds.left ||
      event.clientX > bounds.right ||
      event.clientY < bounds.top ||
      event.clientY > bounds.bottom
    ) {
      cancel();
    }
  };
</script>

<dialog
  aria-describedby={`${id}-body`}
  aria-labelledby={`${id}-title`}
  bind:this={dialog}
  class="lg-user-popup"
  oncancel={cancel}
  onclick={dismissOutside}
  onclose={() => (open = false)}
>
  <header class="lg-user-popup__header">
    <h2 id={`${id}-title`}>{title}</h2>
  </header>
  <p class="lg-user-popup__body" id={`${id}-body`}>{body}</p>
  <footer class="lg-user-popup__actions">
    <Button
      hasActiveState={false}
      onclick={cancel}
      variant="type-3"
    >{secondaryLabel}</Button>
    <Button
      class={`lg-user-popup__primary--${primaryIntent}`}
      hasActiveState={false}
      onclick={confirm}
      variant="type-3"
    >
      {primaryLabel}
    </Button>
  </footer>
</dialog>
