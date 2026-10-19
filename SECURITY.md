# Security

athenea reads USD stages, splat files (PLY, SPZ, SOG, `.athc`), EXR and texture files, and loads AOFX bundles and
OpenFX plugins. A malformed file that crashes the engine or makes it read or write out of bounds is a security
issue.

Report one privately to jesusluquemuriel@gmail.com, with the file or the steps that reproduce it, rather than in a
public issue. A report is acknowledged within a week.

Only the latest release and `main` receive fixes.
