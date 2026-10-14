#!/usr/bin/env bash
# Render the whole course and publish the rendered site to GitHub Pages.
#
#   ./publish.sh            render everything, then publish
#   ./publish.sh --fast     publish what is already in _output (no re-render)
#
# Run this on the Mac (quarto and LaTeX live there). First-time setup is in
# PUBLISHING.md -- this script assumes the repo already has a GitHub remote
# and that Pages is set to serve the `gh-pages` branch.
set -euo pipefail
cd "$(dirname "$0")"

command -v quarto >/dev/null || { echo "quarto not found in PATH"; exit 1; }

if [[ "${1:-}" != "--fast" ]]; then
  echo "==> rendering (this takes a while: 9 decks, 8 notes, 8 exercise sheets, html + pdf)"
  quarto render
fi

echo "==> publishing _output to the gh-pages branch"
quarto publish gh-pages --no-render --no-prompt

echo
echo "Done. The site is at the URL printed above; it takes a minute or two"
echo "for GitHub to finish the deployment."
