<script lang="ts">
  import { tick } from "svelte";

  let {
    ariaLabel,
    options,
    placeholder = "Select options",
    values = $bindable(),
    variant = "bordered",
  }: {
    ariaLabel: string;
    options: ReadonlyArray<{
      disabled?: boolean;
      label: string;
      value: string;
    }>;
    placeholder?: string;
    values: string[];
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

  const focusFirstOption = async (): Promise<void> => {
    await tick();

    const selectedOption = menuElement?.querySelector<HTMLButtonElement>(
      '[aria-checked="true"]:not(:disabled)',
    );
    const firstOption = menuElement?.querySelector<HTMLButtonElement>(
      "button:not(:disabled)",
    );

    (selectedOption ?? firstOption)?.focus();
  };

  const openMenu = (): void => {
    isOpen = true;
    void focusFirstOption();
  };

  const toggleMenu = (): void => {
    if (isOpen) {
      closeMenu();
      return;
    }

    openMenu();
  };

  const toggleOption = (optionValue: string): void => {
    if (values.includes(optionValue)) {
      values = values.filter((valueItem) => valueItem !== optionValue);
      return;
    }

    values = [...values, optionValue];
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

  const selectedOptions = $derived(
    options.filter((option) => values.includes(option.value)),
  );
  const selectedLabel = $derived(
    selectedOptions[0]?.label ?? placeholder,
  );
  const remainingSelectionCount = $derived(selectedOptions.length - 1);
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
    <span>{selectedLabel}</span>
    {#if remainingSelectionCount > 0}
      <span class="lg-select-dropdown__count">
        +{remainingSelectionCount}
      </span>
    {/if}
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
          aria-checked={values.includes(option.value)}
          class:lg-select-dropdown__option--selected={
            values.includes(option.value)
          }
          class="lg-select-dropdown__option"
          disabled={option.disabled}
          onclick={() => toggleOption(option.value)}
          role="checkbox"
          type="button"
        >
          <span>{option.label}</span>
          {#if values.includes(option.value)}
            <svg aria-hidden="true" viewBox="0 0 24 24">
              <path d="m5 12 4 4L19 6" />
            </svg>
          {/if}
        </button>
      {/each}
    </div>
  {/if}
</div>
