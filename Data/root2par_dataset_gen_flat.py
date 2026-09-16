import uproot
import vector
import awkward as ak
import concurrent.futures
import os
import glob
import polars as pl
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get(
    "P2J_ROOT_DIR", os.path.join(SCRIPT_DIR, "Root_files", "63168")
)
FILE_GLOB = os.path.join(DATA_DIR, "*.root")
OUTPUT_PATH = os.path.join(
    SCRIPT_DIR, "dataset", "PJ_dataset_qcd_flat_15to7000.parquet"
)
SHARD_DIR = os.path.join(SCRIPT_DIR, "dataset", "_shards_flat")

PARTON_PT_MIN = 10.0
PARTON_PT_MAX = 8000.0
ETA_CUT = 2.4
DR_MATCH = 0.2

JET_ID_MIN = 2

PUID_MIN = 1
PUID_PT_MAX = 50.0

numworkers = os.cpu_count()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=numworkers)

vector.register_awkward()

root_files = sorted(glob.glob(FILE_GLOB))
if not root_files:
    raise FileNotFoundError(f"No .root files found in: {FILE_GLOB}")

print(f"Found {len(root_files)} ROOT file(s) to process.")

BRANCHES = [
    "event",
    "run",
    "luminosityBlock",
    "Jet_pt",
    "Jet_eta",
    "Jet_phi",
    "Jet_mass",
    "Jet_jetId",
    "Jet_puId",
    "Jet_area",
    "Jet_rawFactor",
    "Jet_nConstituents",
    "Jet_chHEF",
    "Jet_neHEF",
    "Jet_chEmEF",
    "Jet_neEmEF",
    "Jet_muEF",
    "Jet_nMuons",
    "Jet_qgl",
    "Jet_hadronFlavour",
    "Jet_partonFlavour",
    "Jet_bRegCorr",
    "Jet_bRegRes",
    "Jet_genJetIdx",
    "GenJet_pt",
    "GenJet_eta",
    "GenJet_phi",
    "GenJet_mass",
    "GenJet_hadronFlavour",
    "GenJet_partonFlavour",
    "GenPart_pt",
    "GenPart_eta",
    "GenPart_phi",
    "GenPart_mass",
    "GenPart_pdgId",
    "GenPart_status",
    "GenPart_statusFlags",
    "Generator_binvar",
    "PSWeight",
    "Generator_weight",
    "Pileup_nTrueInt",
    "fixedGridRhoFastjetAll",
]


def process_file(file_path: str) -> pl.DataFrame:
    fname = os.path.basename(file_path)
    try:
        with uproot.open(file_path, decomposition_executor=executor) as f:
            tree = f["Events"]
            data = tree.arrays(BRANCHES, decompression_executor=executor)
    except Exception as e:
        print(f"  [SKIP] {fname}: {e}")
        return pl.DataFrame()

    jets = ak.zip(
        {
            "pt": data.Jet_pt,
            "eta": data.Jet_eta,
            "phi": data.Jet_phi,
            "mass": data.Jet_mass,
            "jetId": data.Jet_jetId,
            "puId": data.Jet_puId,
            "area": data.Jet_area,
            "rawFactor": data.Jet_rawFactor,
            "nConstituents": data.Jet_nConstituents,
            "chHEF": data.Jet_chHEF,
            "neHEF": data.Jet_neHEF,
            "chEmEF": data.Jet_chEmEF,
            "neEmEF": data.Jet_neEmEF,
            "muEF": data.Jet_muEF,
            "nMuons": data.Jet_nMuons,
            "qgl": data.Jet_qgl,
            "hadronFlavour": data.Jet_hadronFlavour,
            "partonFlavour": data.Jet_partonFlavour,
            "bRegCorr": data.Jet_bRegCorr,
            "bRegRes": data.Jet_bRegRes,
            "genJetIdx": data.Jet_genJetIdx,
        },
        with_name="Momentum4D",
    )

    genjets = ak.zip(
        {
            "pt": data.GenJet_pt,
            "eta": data.GenJet_eta,
            "phi": data.GenJet_phi,
            "mass": data.GenJet_mass,
            "hadronFlavour": data.GenJet_hadronFlavour,
            "partonFlavour": data.GenJet_partonFlavour,
        },
        with_name="Momentum4D",
    )

    gen = ak.zip(
        {
            "pt": data.GenPart_pt,
            "eta": data.GenPart_eta,
            "phi": data.GenPart_phi,
            "mass": data.GenPart_mass,
            "pdgId": data.GenPart_pdgId,
        },
        with_name="Momentum4D",
    )

    sel_jets = jets[
        (abs(jets.eta) < ETA_CUT)
        & (jets.jetId >= JET_ID_MIN)
        & ((jets.puId >= PUID_MIN) | (jets.pt >= PUID_PT_MAX))
    ]

    is_parton_flavor = (
        (abs(gen.pdgId) == 1)
        | (abs(gen.pdgId) == 2)
        | (abs(gen.pdgId) == 3)
        | (abs(gen.pdgId) == 4)
        | (abs(gen.pdgId) == 5)
        | (gen.pdgId == 21)
    )
    is_outgoing_hard = data.GenPart_status == 23
    from_hard_process = (data.GenPart_statusFlags & (1 << 8)) > 0
    in_pt_window = (gen.pt > PARTON_PT_MIN) & (gen.pt < PARTON_PT_MAX)

    sel_partons = gen[
        is_parton_flavor & is_outgoing_hard & from_hard_process & in_pt_window
    ]

    pt_order = ak.argsort(sel_partons.pt, axis=1, ascending=False)
    sel_partons = sel_partons[pt_order[:, :2]]

    idx_jq = ak.argcartesian({"j": sel_jets, "q": sel_partons}, nested=True)
    pairs_jq = ak.cartesian({"j": sel_jets, "q": sel_partons}, nested=True)
    dR_jq = pairs_jq["j"].deltaR(pairs_jq["q"])
    best_jq = ak.argmin(dR_jq, axis=2, keepdims=True)
    j_best_dR = ak.fill_none(ak.flatten(dR_jq[best_jq], axis=2), 999.0)
    j_best_qidx = ak.fill_none(ak.flatten(idx_jq["q"][best_jq], axis=2), -1)

    idx_qj = ak.argcartesian({"q": sel_partons, "j": sel_jets}, nested=True)
    pairs_qj = ak.cartesian({"q": sel_partons, "j": sel_jets}, nested=True)
    dR_qj = pairs_qj["q"].deltaR(pairs_qj["j"])
    best_qj = ak.argmin(dR_qj, axis=2, keepdims=True)
    q_best_jidx = ak.fill_none(ak.flatten(idx_qj["j"][best_qj], axis=2), -1)

    jet_local_idx = ak.local_index(sel_jets, axis=1)
    safe_qidx = ak.where(j_best_qidx >= 0, j_best_qidx, 0)
    q_best_jidx_padded = ak.pad_none(q_best_jidx, 1, axis=1)
    gathered_jidx = q_best_jidx_padded[safe_qidx]
    mutual = (j_best_qidx >= 0) & ak.fill_none(gathered_jidx == jet_local_idx, False)

    match = mutual & (j_best_dR < DR_MATCH)

    mj = sel_jets[match]
    mq = sel_partons[j_best_qidx[match]]
    mdR = j_best_dR[match]

    n_genjet = ak.num(genjets, axis=1)
    gidx = mj.genJetIdx
    gj_valid = (gidx >= 0) & (gidx < n_genjet)
    safe_gidx = ak.where(gj_valid, gidx, 0)
    genjets_padded = ak.pad_none(genjets, 1, axis=1)
    gj = genjets_padded[safe_gidx]

    def flat(arr):
        return ak.to_numpy(ak.flatten(arr))

    def flat_masked_f32(values, valid):
        v = ak.to_numpy(ak.flatten(ak.where(valid, values, np.nan))).astype(np.float32)
        return v

    def flat_masked_i32(values, valid):
        v = ak.to_numpy(ak.flatten(ak.where(valid, values, -1))).astype(np.int32)
        return v

    n = int(ak.sum(ak.num(mj)))
    if n == 0:
        return pl.DataFrame()

    n_partons_per_evt = ak.num(sel_partons, axis=1)

    def broadcast_event_field(evt_field):
        return flat(ak.broadcast_arrays(evt_field, mj.pt)[0])

    df = pl.DataFrame(
        {
            "event": broadcast_event_field(data.event).astype(np.int64),
            "run": broadcast_event_field(data.run).astype(np.int32),
            "luminosityBlock": broadcast_event_field(data.luminosityBlock).astype(
                np.int32
            ),
            "pt_hat": broadcast_event_field(data.Generator_binvar).astype(np.float32),
            "generator_weight": broadcast_event_field(data.Generator_weight).astype(
                np.float32
            ),
            # PSWeight title: [0] ISR=2 FSR=1; [1] ISR=1 FSR=2; [2] ISR=0.5 FSR=1; [3] ISR=1 FSR=0.5
            "ps_isr_up": broadcast_event_field(data.PSWeight[:, 0]).astype(np.float32),
            "ps_fsr_up": broadcast_event_field(data.PSWeight[:, 1]).astype(np.float32),
            "ps_isr_down": broadcast_event_field(data.PSWeight[:, 2]).astype(
                np.float32
            ),
            "ps_fsr_down": broadcast_event_field(data.PSWeight[:, 3]).astype(
                np.float32
            ),
            "pileup_nTrueInt": broadcast_event_field(data.Pileup_nTrueInt).astype(
                np.float32
            ),
            "rho": broadcast_event_field(data.fixedGridRhoFastjetAll).astype(
                np.float32
            ),
            "jet_pt": flat(mj.pt).astype(np.float32),
            "jet_eta": flat(mj.eta).astype(np.float32),
            "jet_phi": flat(mj.phi).astype(np.float32),
            "jet_mass": flat(mj.mass).astype(np.float32),
            "jet_energy": flat(mj.E).astype(np.float32),
            "jet_px": flat(mj.px).astype(np.float32),
            "jet_py": flat(mj.py).astype(np.float32),
            "jet_pz": flat(mj.pz).astype(np.float32),
            "jet_jetId": flat(mj.jetId).astype(np.int32),
            "jet_puId": flat(mj.puId).astype(np.int32),
            "jet_area": flat(mj.area).astype(np.float32),
            "jet_rawFactor": flat(mj.rawFactor).astype(np.float32),
            "jet_nConstituents": flat(mj.nConstituents).astype(np.int32),
            "jet_chHEF": flat(mj.chHEF).astype(np.float32),
            "jet_neHEF": flat(mj.neHEF).astype(np.float32),
            "jet_chEmEF": flat(mj.chEmEF).astype(np.float32),
            "jet_neEmEF": flat(mj.neEmEF).astype(np.float32),
            "jet_muEF": flat(mj.muEF).astype(np.float32),
            "jet_nMuons": flat(mj.nMuons).astype(np.int32),
            "jet_qgl": flat(mj.qgl).astype(np.float32),
            "jet_hadronFlavour": flat(mj.hadronFlavour).astype(np.int32),
            "jet_partonFlavour": flat(mj.partonFlavour).astype(np.int32),
            "jet_bRegCorr": flat(mj.bRegCorr).astype(np.float32),
            "jet_bRegRes": flat(mj.bRegRes).astype(np.float32),
            "has_genjet_match": flat(gj_valid),
            "genjet_pt": flat_masked_f32(gj.pt, gj_valid),
            "genjet_eta": flat_masked_f32(gj.eta, gj_valid),
            "genjet_phi": flat_masked_f32(gj.phi, gj_valid),
            "genjet_mass": flat_masked_f32(gj.mass, gj_valid),
            "genjet_hadronFlavour": flat_masked_i32(gj.hadronFlavour, gj_valid),
            "genjet_partonFlavour": flat_masked_i32(gj.partonFlavour, gj_valid),
            "parton_px": flat(mq.px).astype(np.float32),
            "parton_py": flat(mq.py).astype(np.float32),
            "parton_pz": flat(mq.pz).astype(np.float32),
            "parton_energy": flat(mq.E).astype(np.float32),
            "parton_pt": flat(mq.pt).astype(np.float32),
            "parton_eta": flat(mq.eta).astype(np.float32),
            "parton_phi": flat(mq.phi).astype(np.float32),
            "parton_mass": flat(mq.mass).astype(np.float32),
            "parton_pdgId": flat(mq.pdgId).astype(np.int32),
            "delta_R": flat(mdR).astype(np.float32),
            "source_file": pl.Series([fname] * n),
        }
    )

    print(
        f"  {fname}: {n:,} matched parton-jet pairs "
        f"({int(ak.sum(n_partons_per_evt)):,} hard partons, "
        f"{len(data):,} events)"
    )
    return df


os.makedirs(SHARD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)

shards = []
total = 0
for i, fp in enumerate(root_files, 1):
    df = process_file(fp)
    if len(df) == 0:
        continue
    sp = os.path.join(SHARD_DIR, f"shard_{i:03d}.parquet")
    df.write_parquet(sp, compression="zstd", compression_level=3)
    shards.append(sp)
    total += len(df)
    del df
    print(f"  [{i}/{len(root_files)}] cumulative rows: {total:,}")

if not shards:
    raise RuntimeError("No matched pairs found across all files.")

lf = pl.scan_parquet(sorted(shards)).with_row_index("global_match_id")
try:
    lf.sink_parquet(OUTPUT_PATH, compression="zstd", compression_level=3)
except Exception as e:
    print(f"  streaming sink unavailable ({e}); falling back to in-memory collect")
    lf.collect().write_parquet(OUTPUT_PATH, compression="zstd", compression_level=3)

for sp in shards:
    os.remove(sp)
os.rmdir(SHARD_DIR)

n = pl.scan_parquet(OUTPUT_PATH).select(pl.len()).collect().item()
print(f"\nDone. Saved {n:,} parton-jet pairs to {OUTPUT_PATH}")
