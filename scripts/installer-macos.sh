#!/bin/bash
# Copyright (c) 2026 jesus luque.
#
# The macOS installer: package.sh's folder as a .pkg that installs to
# /usr/local/athenea and links its programs into /usr/local/bin. Not signed with
# a Developer ID: Gatekeeper asks once (right click, Open).
#
#   scripts/installer-macos.sh <package folder> [out dir]
set -euo pipefail
pkgdir=$(cd "${1:?usage: installer-macos.sh <package folder> [out dir]}" && pwd)
out=$(mkdir -p "${2:-$(dirname "$pkgdir")}" && cd "${2:-$(dirname "$pkgdir")}" && pwd)
name=$(basename "$pkgdir")
version=$(echo "$name" | sed -E 's/^athenea-([0-9.]+)-.*/\1/')
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/root/usr/local" "$work/scripts"
cp -R "$pkgdir" "$work/root/usr/local/athenea"
cat > "$work/scripts/postinstall" <<'POST'
#!/bin/bash
mkdir -p /usr/local/bin
for program in athenea athenea-mcp; do
    [ -x /usr/local/athenea/bin/$program ] && ln -sf /usr/local/athenea/bin/$program /usr/local/bin/$program
done
exit 0
POST
chmod +x "$work/scripts/postinstall"
pkgbuild --root "$work/root" --scripts "$work/scripts" --identifier co.uk.lucab.athenea \
    --version "$version" --install-location / "$work/athenea.pkg" >/dev/null
productbuild --package "$work/athenea.pkg" "$out/$name.pkg" >/dev/null
echo "$out/$name.pkg"
