#!/usr/bin/env bash
# Install the pinned BtbN ffmpeg build used by CI and by local metric runs.
# Ubuntu's ffmpeg package has libx264, libx265, and libsvtav1, and does not
# ship libvmaf. See docs/decisions/0003-pinned-ffmpeg.md.
set -euo pipefail

dest="${1:-/opt/ffmpeg}"
# Month-end release-branch build. BtbN drops daily builds after 14 days
# and keeps the last build of each month for two years. See ADR-0003.
url="https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-30-13-08/ffmpeg-n9.0.2-17-g2a571b6068-linux64-gpl-9.0.tar.xz"
sha256="68ee646831adaae2495618346f3bba94ff207ff83bbd34d643e7004730d66269"

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
