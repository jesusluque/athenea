#!/bin/bash
# Copyright (c) 2026 jesus luque.
#
# The paper from its sections, in both languages: paper.md and paper.es.md,
# their HTML (pandoc, MathML, every resource embedded) and their PDF (headless
# Chrome). The table of contents pandoc writes goes inside the front's
# `#contents` block, after the title page, where the PDF expects it.
#
#   docs/paper/build.sh
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
SECTIONS=(00-front 01-introduction 02-engine 03-conversion 04-transfer 05-animation 06-lod 07-compression-player
          08-evaluation 09-limitations 10-references)
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

for lang in en es; do
    suffix=""; [ "$lang" = es ] && suffix=".es"
    md="paper${suffix}.md"; html="paper${suffix}.html"; pdf="paper.${lang}.pdf"
    python3 - "$md" "$suffix" "${SECTIONS[@]}" <<'PY'
import sys
out, suffix, sections = sys.argv[1], sys.argv[2], sys.argv[3:]
open(out, "w").write("\n\n".join(open(f + suffix + ".md").read().rstrip("\n") for f in sections) + "\n")
PY
    # The title is the front's first heading, without its attributes.
    title="$(grep -m1 '^# ' "00-front${suffix}.md" | sed -e 's/^# //' -e 's/ *{.*}$//')"
    pandoc "$md" -s --toc --mathml --embed-resources --css paper.css --metadata title="$title" -V lang="$lang" \
        -o "$WORK/$html"
    python3 - "$WORK/$html" "$html" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
nav = re.search(r'<nav id="TOC" role="doc-toc">.*?</nav>\n?', s, re.S)
s = s[:nav.start()] + s[nav.end():]
block = re.search(r'(<div id="contents">)(.*?)(</div>)', s, re.S)
s = s[:block.start()] + block.group(1) + nav.group(0) + block.group(3) + s[block.end():]
open(sys.argv[2], "w").write(s)
PY
    "$CHROME" --headless --disable-gpu --no-pdf-header-footer --user-data-dir="$WORK/chrome" \
        --virtual-time-budget=20000 --print-to-pdf="$WORK/$pdf" "file://$PWD/$html" 2>/dev/null
    cp "$WORK/$pdf" "$pdf"
    echo "$md, $html, $pdf"
done
