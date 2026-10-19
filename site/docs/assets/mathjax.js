// Copyright (c) 2026 jesus luque.
// MathJax for the paper's formulas, as pymdownx.arithmatex (generic) writes them.
window.MathJax = {
  loader: { load: ["[tex]/boldsymbol"] },
  tex: { packages: { "[+]": ["boldsymbol"] }, inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]], processEscapes: true, processEnvironments: true },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" }
};
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
