#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECID=63168
RECID_DIR="${SCRIPT_DIR}/${RECID}"

printf 'Starting download for record %s\n' "${RECID}"
printf 'Dataset: QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8 (NANOAODSIM, 2016)\n'
printf 'Working directory: %s\n' "${SCRIPT_DIR}"

printf '\n------------------------------------------------------\n'

cd "${SCRIPT_DIR}"
cernopendata-client download-files --recid "${RECID}" --protocol http


ROOT_COUNT="$(find "${RECID_DIR}" -type f -name '*.root' 2>/dev/null | wc -l)"
TOTAL_SIZE="$(du -sh "${RECID_DIR}" 2>/dev/null | cut -f1)"
printf '\nDownload completed. ROOT files: %s; total size: %s\n' \
	"${ROOT_COUNT}" "${TOTAL_SIZE}"
