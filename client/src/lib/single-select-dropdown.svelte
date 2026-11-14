<script lang="ts">
  import { tick } from "svelte";

  let {
    ariaLabel,
    options,
    value = $bindable(),
    variant = "bordered",
  }: {
    ariaLabel: string;
    options: ReadonlyArray<{
      disabled?: boolean;
      label: string;
      value: string;
    }>;
    value: string;
    variant?: "bordered" | "text";
  } = $props();

  let isOpen = $state(false);
  let menuElement = $state<HTMLDivElement>();
  let rootElement = $state<HTMLDivElement>();
  let triggerElement = $state<HTMLButtonElement>();

  const closeMenu = (restoreFocus = false): void => {
    isOpen = false;

    if (restoreFocus) {
      triggerElement?.focus();
    }
  };

  const focusSelectedOption = async (): Promise<void> => {
    await tick();

    const selectedOption = menuElement?.querySelector<HTMLButtonElement>(
      '[aria-selected="true"]:not(:disabled)',
    );
    const firstOption = menuElement?.querySelector<HTMLButtonElement>(
      'button:not(:disabled)',
    );

    (selectedOption ?? firstOption)?.focus();
  };

  const openMenu = (): void => {
    isOpen = true;
    void focusSelectedOption();
  };

  const toggleMenu = (): void => {
    if (isOpen) {
      closeMenu();
      return;
    }

    openMenu();
  };

  const selectOption = (optionValue: string): void => {
    value = optionValue;
    closeMenu(true);
  };

  const moveFocus = (event: KeyboardEvent): void => {
    const choices = Array.from(
      menuElement?.querySelectorAll<HTMLButtonElement>(
        "button:not(:disabled)",
      ) ?? [],
    );
    const currentIndex = choices.indexOf(
      document.activeElement as HTMLButtonElement,
    );
    const lastIndex = choices.length - 1;
    let nextIndex = currentIndex;

    if (event.key === "Home") {
      nextIndex = 0;
    }

    if (event.key === "End") {
      nextIndex = lastIndex;
    }

    if (event.key === "ArrowDown") {
      nextIndex = (currentIndex + 1 + choices.length) % choices.length;
    }

    if (event.key === "ArrowUp") {
      nextIndex = (currentIndex - 1 + choices.length) % choices.length;
    }

    choices[nextIndex]?.focus();
  };

  const onMenuKeydown = (event: KeyboardEvent): void => {
    if (event.key === "Escape") {
      event.preventDefault();
      closeMenu(true);
      return;
    }

    if (event.key === "Tab") {
      closeMenu();
      return;
    }

    if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      moveFocus(event);
    }
  };

  const onTriggerKeydown = (event: KeyboardEvent): void => {
    if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
      event.preventDefault();
      openMenu();
    }
  };

  const onWindowClick = (event: MouseEvent): void => {
    if (
      isOpen &&
      event.target instanceof Node &&
      !rootElement?.contains(event.target)
    ) {
      closeMenu();
    }
  };

  const selectedOption = $derived(
    options.find((option) => option.value === value),
  );
</script>

<svelte:window onclick={onWindowClick} />

<div bind:this={rootElement} class="lg-select-dropdown">
  <button
    aria-expanded={isOpen}
    aria-haspopup="listbox"
    aria-label={ariaLabel}
    bind:this={triggerElement}
    class={`lg-select-dropdown__trigger lg-select-dropdown__trigger--${variant}`}
    onclick={toggleMenu}
    onkeydown={onTriggerKeydown}
    type="button"
  >
    <span>{selectedOption?.label ?? "Select an option"}</span>
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="m6 9 6 6 6-6" />
    </svg>
  </button>

  {#if isOpen}
    <div
      aria-label={ariaLabel}
      bind:this={menuElement}
      class="lg-select-dropdown__menu"
      onkeydown={onMenuKeydown}
      role="listbox"
      tabindex="-1"
    >
      {#each options as option}
        <button
          aria-selected={option.value === value}
          class:lg-select-dropdown__option--selected={
            option.value === value
          }
          class="lg-select-dropdown__option"
          disabled={option.disabled}
          onclick={() => selectOption(option.value)}
          role="option"
          type="button"
        >
          {option.label}
        </button>
      {/each}
    </div>
  {/if}
</div>
