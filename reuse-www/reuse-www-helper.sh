#!/usr/bin/env bash

set -e

print_help() {
  echo "Helper functions for reuse-www"
  echo ""
  echo "Usage: reuse-www-helper [ARGUMENTS]"
  echo ""
  echo "Arguments:"
  echo "  --help"
  echo "  --install-code-docs <REUSEIR_BUILD> <WEB_DST>"
}

install_docs() {
  REUSEIR_BUILD="$1"
  WEB_DST="$2"

  input_dir="$REUSEIR_BUILD/docs"
  output_dir="$WEB_DST/content/dialect/code_docs"
  mkdir -p $output_dir

  echo "Installing code docs from '$input_dir' into '$output_dir'"

  cat >"$output_dir/_index.md" <<EOF
---
title: "Code Documentation"
weight: 1
---
EOF

  find "$input_dir" -name "*.md" | while read file; do
    file=$(basename $file)
    title=${file#"ReuseIR"}
    title=${title%".md"}

    (echo "---" &&
     echo "title: \"$title\"" &&
     echo "---" &&
     cat "$input_dir/$file" | sed 's|\[TOC\]|<p/>{{< toc >}}|' ) > $output_dir/$file &&
    echo "Processed $file"
  done
}

if [[ "$#" == 0 || "$1" == "--help" ]]; then
  print_help
elif [[ "$1" == "--install-code-docs" ]]; then
  if [[ "$#" != 3 ]]; then
    echo "ERROR: --install-code-docs requires 2 arguments"
    exit 1
  fi
  install_docs "$2" "$3"
else
  echo "ERROR: Unknown argument(s) '$1'"
  exit 1
fi
