# Emerging Jets — Data Exploration & Preprocessing

Pipeline dedicata **solo** a esplorazione dati e preprocessing (nessun training ML).

## Struttura
- `data/raw/`: HDF5 originali (`pp_output_test_background.h5`, `pp_output_test_signal.h5`, `pp_output_val.h5`)
- `data/processed/`: output preprocessati `.npz` + `preprocessing_stats.yaml`
- `configs/`: YAML di setup (`ejs_*.yaml`, `norm_dict.yaml`, `class_dict.yaml`)
- `src/inspect_h5.py`: ispezione HDF5 e statistiche feature
- `src/features.py`: selezione feature robuste da YAML + intersezioni tra file
- `src/visualize.py`: plot jet-level, track-level, scatter geometrico, heatmap
- `src/preprocess.py`: build raw arrays, cleaning, normalizzazione, salvataggio, validazione

## Jets vs Tracks
- `jets`: feature jet-level, shape `(N,)` con campi strutturati.
- `tracks`: feature track-level, shape `(N, 200)` con campi per traccia.
- `tracks['valid']`: maschera booleana che indica tracce reali/valide.

## Regole implementate
- Input separati da metadata/label/truth.
- `valid` usata solo come maschera (`track_mask`), mai input numerico.
- Esclusi dagli input: `isDisplaced`, `flavour_label`, `isTagged`, `salt_pdisp`, `mcEventWeight`, `eventNumber`, `truthOriginLabel`, `truthVertexIndex`, `VSIVertexIndex`, `truth_dark_pions`, `truth_stable_non_geant`.
- Feature finali = intersezione tra file + coerenza con YAML.

## Esecuzione
1. Copia i file HDF5 in `project/data/raw/`.
2. Ispezione veloce (Python shell/notebook):
   - `from src.inspect_h5 import inspect_h5_structure, compare_fields, summarize_feature`
3. Preprocessing completo:
   - `python project/src/preprocess.py --raw-dir project/data/raw --config-dir project/configs --out-dir project/data/processed`
   - opzionale: `--max-jets 200000`

## Output attesi
- `processed_background_test.npz` (con `y=0`)
- `processed_signal_test.npz` (con `y=1`)
- `processed_val.npz` (con `y` solo se deducibile da `isDisplaced`)
- `preprocessing_stats.yaml` (report NaN/inf + statistiche normalizzazione + feature selezionate)

Ogni file `.npz` contiene:
- `X_jets`, `X_tracks`, `track_mask`
- `jet_feature_names`, `track_feature_names`
- `meta_*` disponibili
- `y` se disponibile
