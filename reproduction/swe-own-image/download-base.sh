#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")"
BASE=/export/home/ext.luohaowen1/continuum
URL=https://cdimage.ubuntu.com/ubuntu-base/releases/22.04/release
ARCHIVE=ubuntu-base-22.04.5-base-amd64.tar.gz
FINGERPRINT=843938DF228D22F7B3742BC0D94AA3F0EFE21092
mkdir -p provenance/gnupg
chmod 700 provenance/gnupg
curl -fSL --connect-timeout 20 --max-time 60 "$URL/SHA256SUMS" -o provenance/SHA256SUMS
curl -fSL --connect-timeout 20 --max-time 60 "$URL/SHA256SUMS.gpg" -o provenance/SHA256SUMS.gpg
curl -fSL --connect-timeout 20 --max-time 60 \
    "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x$FINGERPRINT" \
    -o provenance/ubuntu-signing-key.asc
gpg --batch --homedir "$PWD/provenance/gnupg" --import provenance/ubuntu-signing-key.asc
gpg --batch --homedir "$PWD/provenance/gnupg" --status-fd 1 \
    --verify provenance/SHA256SUMS.gpg provenance/SHA256SUMS > provenance/signature-status.txt
grep -F "[GNUPG:] VALIDSIG $FINGERPRINT " provenance/signature-status.txt
if [ ! -f "$ARCHIVE" ]; then
    curl -fSL --connect-timeout 20 --max-time 300 "$URL/$ARCHIVE" -o "$ARCHIVE.part"
    mv "$ARCHIVE.part" "$ARCHIVE"
fi
echo "242cd8898b33ea806ef5f13b1076ed7c76f9f989d18384452f7166692438ff1a  $ARCHIVE" | sha256sum --check
grep -F " *$ARCHIVE" provenance/SHA256SUMS | sha256sum --check
INSTALLER=Miniforge3-26.7.2-0-Linux-x86_64.sh
if [ ! -f "$INSTALLER" ]; then
    cp "$BASE/reproduction/conda-setup/downloads/$INSTALLER" "$INSTALLER"
fi
echo "281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05  $INSTALLER" | sha256sum --check
printf '%s\n' "$URL/$ARCHIVE" \
    'https://github.com/conda-forge/miniforge/releases/download/26.7.2-0/Miniforge3-26.7.2-0-Linux-x86_64.sh' \
    > provenance/source-urls.txt
