/** Look up an element by id, failing loudly if the page does not have it. */
export function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el;
}
