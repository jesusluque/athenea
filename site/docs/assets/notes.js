// Copyright (c) 2026 jesus luque.
// The paper's notes can be hidden: one switch on each paper page, remembered per browser.
document$.subscribe(() => {
  const notes = document.querySelectorAll(".md-typeset .nota");
  if (!notes.length) return;
  const es = document.documentElement.lang === "es";
  const key = "athenea-paper-notes";
  let hidden = false;
  try { hidden = localStorage.getItem(key) === "hidden"; } catch (e) {}
  const button = document.createElement("button");
  button.className = "nota-toggle md-button";
  const apply = () => {
    document.body.classList.toggle("notas-ocultas", hidden);
    button.textContent = hidden ? (es ? "Mostrar acotaciones" : "Show notes") : (es ? "Ocultar acotaciones" : "Hide notes");
  };
  button.addEventListener("click", () => {
    hidden = !hidden;
    try { localStorage.setItem(key, hidden ? "hidden" : "shown"); } catch (e) {}
    apply();
  });
  const h1 = document.querySelector(".md-typeset h1");
  if (h1) h1.after(button);
  apply();
});
