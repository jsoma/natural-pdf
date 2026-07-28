/**
 * Progressive enhancement for npdf tab groups.
 *
 * The remark plugin (src/plugins/remark-npdf-tabs.mjs) renders each authored
 * tab group as:
 *
 *   <div class="npdf-tabs" data-npdf-tabs>
 *     <section class="npdf-tab" data-label="...">
 *       <p class="npdf-tab-label">...</p>
 *       ...content...
 *     </section>
 *     ...
 *   </div>
 *
 * Without JavaScript that renders as labeled stacked sections. This script
 * upgrades each group into an accessible tablist: one panel visible at a
 * time, ArrowLeft/ArrowRight/Home/End keyboard support, and label sync —
 * activating a tab labeled X also activates the tab labeled X in every other
 * group on the page (preserving MkDocs Material's content.tabs.link
 * behavior for the pdfplumber-vs-Natural-PDF comparison pages).
 */

interface TabGroup {
  tabs: HTMLButtonElement[];
  panels: HTMLElement[];
}

const groups: TabGroup[] = [];
let uid = 0;

function activate(group: TabGroup, index: number, options: { focus?: boolean; sync?: boolean }): void {
  group.tabs.forEach((tab, i) => {
    const selected = i === index;
    tab.setAttribute("aria-selected", selected ? "true" : "false");
    tab.tabIndex = selected ? 0 : -1;
    group.panels[i].hidden = !selected;
  });
  if (options.focus) {
    group.tabs[index].focus();
  }
  if (options.sync !== false) {
    const label = group.tabs[index].dataset.label;
    for (const other of groups) {
      if (other === group) continue;
      const match = other.tabs.findIndex((tab) => tab.dataset.label === label);
      if (match !== -1) {
        activate(other, match, { sync: false });
      }
    }
  }
}

function onKeydown(group: TabGroup, event: KeyboardEvent): void {
  const current = group.tabs.indexOf(event.currentTarget as HTMLButtonElement);
  if (current === -1) return;
  let next: number;
  switch (event.key) {
    case "ArrowLeft":
      next = (current - 1 + group.tabs.length) % group.tabs.length;
      break;
    case "ArrowRight":
      next = (current + 1) % group.tabs.length;
      break;
    case "Home":
      next = 0;
      break;
    case "End":
      next = group.tabs.length - 1;
      break;
    default:
      return;
  }
  event.preventDefault();
  activate(group, next, { focus: true });
}

function enhance(root: HTMLElement): void {
  const panels = Array.from(
    root.querySelectorAll<HTMLElement>(":scope > section.npdf-tab")
  );
  if (panels.length === 0) return;

  const groupId = `npdf-tabs-${uid++}`;
  const tablist = document.createElement("div");
  tablist.className = "npdf-tablist";
  tablist.setAttribute("role", "tablist");

  const group: TabGroup = { tabs: [], panels };

  panels.forEach((panel, i) => {
    const label = panel.dataset.label ?? `Tab ${i + 1}`;
    const tabId = `${groupId}-tab-${i}`;
    const panelId = `${groupId}-panel-${i}`;

    // The no-JS label paragraph is redundant once there is a tablist.
    const labelEl = panel.querySelector<HTMLElement>(":scope > .npdf-tab-label");
    if (labelEl) labelEl.hidden = true;

    const tab = document.createElement("button");
    tab.type = "button";
    tab.id = tabId;
    tab.textContent = label;
    tab.dataset.label = label;
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", panelId);
    tab.addEventListener("click", () => {
      activate(group, i, {});
    });
    tab.addEventListener("keydown", (event) => {
      onKeydown(group, event);
    });
    tablist.appendChild(tab);
    group.tabs.push(tab);

    panel.id = panelId;
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", tabId);
  });

  root.prepend(tablist);
  root.classList.add("npdf-tabs-enhanced");
  groups.push(group);
  activate(group, 0, { sync: false });
}

function init(): void {
  document
    .querySelectorAll<HTMLElement>("[data-npdf-tabs]")
    .forEach((root) => enhance(root));
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
