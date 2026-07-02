#!/usr/bin/env bash
# Replace symlinks under METRLA/ and PEMSBAY/ with real files (for git push).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_ROOT="/mnt/data728/linhui/9993/topo+Harmonic+GWN"
OSM_CACHE="/mnt/data728/linhui/9993/6-11/METRLA/osm_graph_cache.pkl"

copy_one() {
  local dst_dir=$1 name=$2 src=$3
  rm -f "$dst_dir/$name"
  cp -a "$src" "$dst_dir/$name"
  echo "copied $name -> $dst_dir/"
}

mkdir -p "$ROOT/METRLA" "$ROOT/PEMSBAY"

copy_one "$ROOT/METRLA" metr-la.h5           "$SRC_ROOT/METRLA/metr-la.h5"
copy_one "$ROOT/METRLA" adj_mx.pkl           "$SRC_ROOT/METRLA/adj_mx.pkl"
copy_one "$ROOT/METRLA" graph_sensor_locations.csv "$SRC_ROOT/METRLA/graph_sensor_locations.csv"
copy_one "$ROOT/METRLA" osm_graph_cache.pkl  "$OSM_CACHE"

copy_one "$ROOT/PEMSBAY" pems-bay.h5        "$SRC_ROOT/PEMSBAY/pems-bay.h5"
copy_one "$ROOT/PEMSBAY" adj_mx_bay.pkl      "$SRC_ROOT/PEMSBAY/adj_mx_bay.pkl"
copy_one "$ROOT/PEMSBAY" graph_sensor_locations_bay.csv "$SRC_ROOT/PEMSBAY/graph_sensor_locations_bay.csv"

echo "Done. METRLA + PEMSBAY are now real files under $ROOT"
