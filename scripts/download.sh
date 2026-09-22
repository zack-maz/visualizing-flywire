#!/usr/bin/env bash
# Downloads the FlyWire FAFB release 783 files the build needs into data/raw/.
# The synapse table (~2 GB) is not downloaded: neuropil_counts.py reads just three columns remotely.
set -euo pipefail
cd "$(dirname "$0")/.."
B="https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/fafb_783"
mkdir -p data/raw/neuropils

curl -sSfo data/raw/neuron_annotations.tsv \
  "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
curl -sSfo data/raw/fafb14_volume_raw.obj "$B/obj/fafb14_volume_raw.obj"

# Every neuropil mesh listed in the bucket.
curl -sSf "https://storage.googleapis.com/storage/v1/b/lee-lab_brain-and-nerve-cord-fly-connectome/o?prefix=compiled_data/fafb_783/obj/neuropils/&maxResults=500" \
  | grep -o 'fafb14_neuropil_[A-Za-z_]*_raw\.obj' | sort -u \
  | while read -r f; do
      name="${f#fafb14_neuropil_}"; name="${name%_raw.obj}"
      curl -sSfo "data/raw/neuropils/$name.obj" "$B/obj/neuropils/$f" &
    done
wait
echo "Downloaded $(ls data/raw/neuropils | wc -l | tr -d ' ') neuropil meshes"
