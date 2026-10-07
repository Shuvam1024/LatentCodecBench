#!/usr/bin/env bash
# Install the pinned BtbN ffmpeg build used by CI and by local metric runs.
# Ubuntu's ffmpeg package has libx264, libx265, and libsvtav1, and does not
# ship libvmaf. See docs/decisions/0003-pinned-ffmpeg.md.
set -euo pipefail

dest="${1:-/opt/ffmpeg}"
url="https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-06-13-06/ffmpeg-N-127222-g151814650f-linux64-gpl.tar.xz"
sha256="0223797ce0ece4b58a11af30cd69318bcdaf1fad395dbaeb163e1abc588706a0"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fsSL -o "$tmp/ffmpeg.tar.xz" "$url"
echo "${sha256}  ${tmp}/ffmpeg.tar.xz" | sha256sum -c -
tar -xJf "$tmp/ffmpeg.tar.xz" -C "$tmp"

src="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d -name 'ffmpeg-*')"
if [[ -z "$src" ]]; then
  echo "ffmpeg archive did not contain a top-level directory" >&2
  exit 1
fi

rm -rf "$dest"
mkdir -p "$(dirname "$dest")"
mv "$src" "$dest"
"$dest/bin/ffmpeg" -hide_banner -version | head -n 1
