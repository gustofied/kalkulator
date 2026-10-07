import { gsap } from "gsap";

/** Follow streamed text without treating its growing height as reader input. */
export function createFollowScroller(element: HTMLElement) {
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const hover = matchMedia("(hover: hover) and (pointer: fine)");
  let following = true;
  let hovered = hover.matches && element.matches(":hover");
  let pointerHeld = false;
  let selected = false;
  let frame: number | null = null;
  let tween: gsap.core.Tween | null = null;

  const atEnd = () => element.scrollHeight - element.clientHeight - element.scrollTop <= 2;
  const stopMotion = () => {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    tween?.kill();
    tween = null;
  };
  const pause = () => {
    following = false;
    stopMotion();
  };
  const followLatest = () => {
    if (!following || hovered || pointerHeld || selected || frame !== null) return;
    frame = requestAnimationFrame(() => {
      frame = null;
      if (!following || hovered || pointerHeld || selected) return;
      const target = Math.max(0, element.scrollHeight - element.clientHeight);
      if (Math.abs(target - element.scrollTop) <= 1) return;
      tween?.kill();
      if (reducedMotion.matches) {
        element.scrollTop = target;
        tween = null;
        return;
      }
      tween = gsap.to(element, {
        scrollTop: target,
        duration: 0.2,
        ease: "power3.out",
        overwrite: true,
        onComplete: () => {
          tween = null;
          // New text may have arrived while this movement was in flight.
          followLatest();
        },
      });
    });
  };
  const resume = () => {
    following = true;
    followLatest();
  };

  element.addEventListener("pointerenter", (event) => {
    if (event.pointerType !== "mouse" || !hover.matches) return;
    hovered = true;
    stopMotion();
  });
  element.addEventListener("pointerleave", (event) => {
    if (event.pointerType !== "mouse") return;
    hovered = false;
    resume();
  });
  element.addEventListener("wheel", pause, { passive: true });
  element.addEventListener("pointerdown", () => {
    pointerHeld = true;
    pause();
  }, { passive: true });
  const releasePointer = (event: PointerEvent) => {
    if (!pointerHeld) return;
    pointerHeld = false;
    if ((event.pointerType === "mouse" && !hovered) || atEnd()) resume();
  };
  window.addEventListener("pointerup", releasePointer, { passive: true });
  window.addEventListener("pointercancel", releasePointer, { passive: true });

  document.addEventListener("selectionchange", () => {
    const selection = document.getSelection();
    const wasSelected = selected;
    selected = Boolean(selection && !selection.isCollapsed && (
      element.contains(selection.anchorNode) || element.contains(selection.focusNode)
    ));
    if (selected) stopMotion();
    else if (wasSelected) resume();
  });
  element.addEventListener("keydown", (event) => {
    if (event.key === "End") {
      event.preventDefault();
      stopMotion();
      element.scrollTop = element.scrollHeight;
      resume();
    } else if (event.key === "Home") {
      event.preventDefault();
      pause();
      element.scrollTop = 0;
    } else if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", " "].includes(event.key)) {
      pause();
    }
  });
  element.addEventListener("blur", resume);
  element.addEventListener("scroll", () => {
    // Reaching the end by touch or keyboard opts back into following.
    if (!following && atEnd() && !pointerHeld && !selected) resume();
  }, { passive: true });

  const observer = new ResizeObserver(followLatest);
  observer.observe(element);
  if (element.firstElementChild) observer.observe(element.firstElementChild);

  return {
    followLatest,
    reset() {
      stopMotion();
      following = true;
      pointerHeld = false;
      selected = false;
      hovered = hover.matches && element.matches(":hover");
      element.scrollTop = 0;
    },
  };
}
