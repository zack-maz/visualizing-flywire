"""Count synapses per (neuron, role, neuropil, side) from the FlyWire 783 synapse table.

Each synapse counts once for its presynaptic neuron (role "out") and once for its postsynaptic
neuron (role "in"). Sum over role for totals.
Output: data/raw/neuron_neuropil_counts.parquet (root_id, role, neuropil, side, n)
"""
import os
import duckdb

SRC = "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/fafb_783/fafb_783_synapses.parquet"
OUT = "data/raw/neuron_neuropil_counts.parquet"
TMP = OUT + ".tmp"

con = duckdb.connect()
con.execute("SET enable_progress_bar = false; INSTALL httpfs; LOAD httpfs;")
con.execute(f"""
COPY (
  WITH s AS (SELECT pre, post, neuropil, side FROM read_parquet('{SRC}'))
  SELECT root_id, role, neuropil, side, count(*)::INTEGER AS n FROM (
    SELECT pre AS root_id, 'out' AS role, neuropil, side FROM s
    UNION ALL
    SELECT post AS root_id, 'in' AS role, neuropil, side FROM s
  ) GROUP BY ALL
) TO '{TMP}' (FORMAT parquet)
""")
os.replace(TMP, OUT)   # only replace the old file once the new one is complete
print(con.execute(f"SELECT role, sum(n)::BIGINT FROM '{OUT}' GROUP BY ALL ORDER BY 1").fetchall())
