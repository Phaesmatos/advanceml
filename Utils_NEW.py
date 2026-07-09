"""
utils.py — Pipeline di preprocessing e valutazione per il progetto
           Emerging Jets Anomaly Detection.

Ogni notebook importa da qui, così la pipeline è definita una volta sola.

─────────────────────────────────────────────────────────────────────────────
GUIDA RAPIDA: quale preprocessing usare per quale modello?
─────────────────────────────────────────────────────────────────────────────

  AUTOENCODER (AE) / VAE con rappresentazione 'aggregate':
      → usa make_aggregate_embeddings()
      → input: vettore flat (N, 43) — media/max/std delle tracce per jet
      → perché: MLP si aspetta un vettore 1D per campione

  AUTOENCODER / VAE con rappresentazione sequenziale:
      → usa sort_by_pt() o sort_by_d0(), poi flatten
      → input: (N, 200*14) = (N, 2800)
      → perché: mantieni la struttura per-traccia, ma l'MLP la vede flat

  RNN / LSTM / GRU (modelli sequenziali):
      → usa sort_by_pt() o sort_by_d0() SENZA flatten
      → input: (N, T, F) — sequenza di T tracce con F feature
      → perché: questi modelli elaborano una traccia alla volta in ordine

  GNN (Graph Neural Network):
      → usa build_graphs() o build_graphs_impact()
      → input: lista di oggetti torch_geometric.data.Data
      → perché: ogni jet diventa un grafo, le tracce sono nodi,
                le connessioni (edge) codificano la prossimità spaziale

─────────────────────────────────────────────────────────────────────────────
"""

# ── Librerie standard Python ───────────────────────────────────────────────────
import os       # operazioni su file e cartelle (path, makedirs, ecc.)
import sys      # accesso al path di sistema (per importare moduli locali)

# ── Lettura file ───────────────────────────────────────────────────────────────
import h5py     # legge file HDF5 (formato standard al CERN per dataset grandi)
import yaml     # legge file di configurazione in formato YAML

# ── Numerica ───────────────────────────────────────────────────────────────────
import numpy as np

# ── Visualizzazione ────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns           # heatmap e plot statistici con stile migliorato

# ── Machine Learning (valutazione) ────────────────────────────────────────────
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors  # usato per costruire i grafi kNN

# ── Deep Learning ─────────────────────────────────────────────────────────────
import torch
from torch_geometric.data import Data   # struttura dati per grafi in PyTorch Geometric
from tqdm import tqdm                   # barra di progresso per loop lunghi


# ══════════════════════════════════════════════════════════════════════════════
# 0. PATH — adatta questi alle tue cartelle locali
# ══════════════════════════════════════════════════════════════════════════════

# Cartella principale del progetto (dove metti utils.py e i notebook)
PROJECT_DIR = r'C:\Users\dfraj\OneDrive\Desktop\Master\Adv_ML\Progetto\ej_anomaly'

# Cartella dei dati: su Leonardo era /leonardo_work/...,
# in locale mettiamo i file .h5 in Download (Windows: C:/Users/<nome>/Downloads)
DATA_DIR = r'C:\Users\dfraj\Downloads'

# File HDF5 con i dati grezzi
BKG_FILE = os.path.join(DATA_DIR, 'pp_output_test_background.h5')
SIG_FILE = os.path.join(DATA_DIR, 'pp_output_test_signal.h5')

# File YAML con le statistiche di normalizzazione (media e std per ogni feature).
# IMPORTANTE: questo file è FORNITO DAI SUPERVISORI insieme al dataset.
# NON va rigenerato né modificato: è parte della specifica del problema.
NORM_FILE = os.path.join(DATA_DIR, 'norm_dict.yaml')

# Cartella dove salvare i modelli addestrati e i plot
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
PLOTS_DIR  = os.path.join(PROJECT_DIR, 'plots')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE LISTS
# ══════════════════════════════════════════════════════════════════════════════

# Lista ordinata delle 14 feature di traccia usate come input dei modelli.
# L'ordine conta: la posizione in questa lista è l'indice della colonna
# nella matrice X di shape (N_jets, MAX_TRACKS, N_FEAT).
TRACK_FEATURES = [
    'pt_log1p',                          # log(1 + pT) della traccia [GeV] — coda pesante compressa
    'd0',                                # parametro d'impatto trasverso [mm]
    'z0SinTheta',                        # parametro d'impatto longitudinale proiettato [mm]
    'dphi',                              # differenza in phi rispetto all'asse del jet
    'deta',                              # differenza in eta rispetto all'asse del jet
    'chiSquared',                        # chi^2 del fit della traccia — qualità del tracking
    'radiusOfFirstHit',                  # raggio del primo hit nel detector [mm] — chiave per EJ!
    'qOverP',                            # carica / impulso — legato alla curvatura nel campo B
    'IP3D_signed_d0_significance',       # d0 / sigma(d0) con segno — significatività statistica
    'IP3D_signed_z0_significance',       # z0sinθ / sigma(z0sinθ) con segno
    'numberOfPixelHits',                 # hit nel Pixel Detector (layer interni)
    'numberOfInnermostPixelLayerHits',   # hit nel layer più interno (IBL) — assenti per displaced!
    'numberOfSCTHits',                   # hit nel Silicon Microstrip Tracker
    'numberOfSCTHoles',                  # layer SCT attraversati senza hit — indice di displaced vertex
]

# Numero totale di feature di traccia
N_FEAT = len(TRACK_FEATURES)   # = 14

# Indici utili per accedere a feature specifiche senza hardcodare numeri
PT_IDX      = TRACK_FEATURES.index('pt_log1p')
D0_IDX      = TRACK_FEATURES.index('d0')
DETA_IDX    = TRACK_FEATURES.index('deta')
DPHI_IDX    = TRACK_FEATURES.index('dphi')
IP3D_D0_IDX = TRACK_FEATURES.index('IP3D_signed_d0_significance')
IP3D_Z0_IDX = TRACK_FEATURES.index('IP3D_signed_z0_significance')


# ══════════════════════════════════════════════════════════════════════════════
# 2. CARICAMENTO DATI
# ══════════════════════════════════════════════════════════════════════════════

def _load_obj(path, group, fields, n_max):
    """
    Funzione interna (prefisso _ = non pensata per l'uso diretto).
    Carica `fields` dal gruppo `group` dentro il file HDF5 `path`.

    I file HDF5 possono avere due strutture diverse:
      - Structured Dataset: un unico array con colonne nominate (come una tabella)
        → accesso: ds['nome_colonna'][:]
      - Group standard: ogni feature è un Dataset separato
        → accesso: group['nome_feature'][:]

    Argomenti:
        path   : percorso al file .h5
        group  : nome del gruppo da leggere ('tracks' o 'jets')
        fields : lista di feature da caricare (None = carica tutto)
        n_max  : quante righe (jet) caricare (None = tutto)

    Restituisce:
        dizionario {nome_feature: array_numpy}
    """
    out = {}
    with h5py.File(path, 'r') as f:
        obj = f[group]
        # Distinguiamo i due casi strutturali
        if isinstance(obj, h5py.Dataset):
            available = list(obj.dtype.names) if obj.dtype.names else []
        else:
            available = list(obj.keys())
        # Prendi solo le feature richieste che esistono nel file
        use = [k for k in (fields or available) if k in available]
        if isinstance(obj, h5py.Dataset):
            # Structured dataset: slice rows FIRST, then access field.
            # Doing obj[k][:n_max] would load all 1M rows before slicing.
            sliced = obj[:n_max] if n_max else obj[:]
            for k in use:
                out[k] = np.array(sliced[k])
        else:
            for k in use:
                raw = obj[k]
                out[k] = np.array(raw[:n_max] if n_max else raw[:])
    return out


def load_tracks(path, features=None, n_max=None, strict=False):
    """
    Carica le feature di traccia dal file HDF5.

    Shape dell'output: ogni feature ha shape (N_jets, MAX_TRACKS).
    MAX_TRACKS = 200: ogni jet ha al massimo 200 slot per tracce.
    Gli slot vuoti (padding) hanno valid=0.

    Aggiunge automaticamente una maschera 'valid' se non è nel file:
    una traccia è valida se pt_log1p != 0 (le tracce di padding
    vengono riempite con zeri nel file HDF5).

    CONTROLLO FEATURE MANCANTI (importante):
      Se una feature richiesta NON esiste nel file (es. nome sbagliato), verrebbe
      saltata silenziosamente e, a valle, riempita di zeri — un bug subdolo che
      corrompe i dati senza errori. Per questo qui avvisiamo esplicitamente:
        - strict=False (default): stampa un WARNING ben visibile e prosegue
          (quelle colonne resteranno a zero in clean_tracks).
        - strict=True: solleva un'eccezione e si ferma.
      Cosi' un nome sbagliato diventa subito visibile invece di passare inosservato.

    Argomenti:
        path     : percorso al file .h5
        features : lista feature da caricare (None = TRACK_FEATURES)
        n_max    : quanti jet caricare (None = tutto)
        strict   : se True, errore in caso di feature mancanti (invece del warning)
    """
    features = features or TRACK_FEATURES
    with h5py.File(path, 'r') as f:
        obj = f['tracks']
        available = list(obj.dtype.names if isinstance(obj, h5py.Dataset) else obj.keys())
        keys = list(features) + (['valid'] if 'valid' in available else [])
        out = _load_obj(path, 'tracks', keys, n_max)

    # Controllo feature mancanti: confronta cosa avevi chiesto con cosa è uscito
    missing = [f for f in features if f not in out]
    if missing:
        msg = (f"[load_tracks] ATTENZIONE: {len(missing)} feature richieste NON trovate "
               f"nel file e verranno lasciate a ZERO: {missing}\n"
               f"  Feature disponibili nel file 'tracks': {available}")
        if strict:
            raise KeyError(msg)
        print("=" * 80)
        print(msg)
        print("=" * 80)

    # Se 'valid' non è nel file, lo costruiamo: una traccia è reale se pt != 0
    if 'valid' not in out and 'pt_log1p' in out:
        out['valid'] = (out['pt_log1p'] != 0).astype(np.float32)
    return out


def load_jets(path, features=None, n_max=None):
    """
    Carica le feature di jet dal file HDF5.
    Shape dell'output: ogni feature ha shape (N_jets,).
    """
    return _load_obj(path, 'jets', features, n_max)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING — step separati e ispezionabili
# ══════════════════════════════════════════════════════════════════════════════

"""
Il preprocessing è diviso in step espliciti e separati.
Questo permette di visualizzare le distribuzioni e le correlazioni
PRIMA e DOPO ogni trasformazione — fondamentale per capire cosa sta succedendo.

Flusso:
    1. load_tracks()       → dati grezzi (unità fisiche, NaN inclusi)
    2. clean_tracks()      → sostituisce NaN/Inf con la media del file
                             <- STAMPA CORRELAZIONI QUI (feature fisiche leggibili)
    3. zscore_normalise()  → z-score (x - mu)/sigma con i valori dei SUPERVISORI
                             + clip di sicurezza a ±CLIP_SIGMA in unità normalizzate
                             <- STAMPA CORRELAZIONI QUI (dopo normalizzazione)
    4. build_repr()        → costruisce la rappresentazione finale per il modello

══════════════════════════════════════════════════════════════════════════════
NOTA FONDAMENTALE SULLA NORMALIZZAZIONE (leggere prima di modificare!)
══════════════════════════════════════════════════════════════════════════════
La media e la deviazione standard di OGNI feature sono FORNITE DAI SUPERVISORI
nel file norm_dict.yaml, insieme al dataset. NON le calcoliamo noi e NON le
rigeneriamo: sono parte della specifica del problema.

Conseguenza pratica: la scala di normalizzazione è già decisa. Anche dove la
std sembra "gonfiata" dagli outlier (es. IP3D_signed_d0_significance ha std≈324
per via di code numeriche a ±29000), questa è la scala che i supervisori hanno
scelto, e ci atteniamo ad essa. Il nostro compito è solo APPLICARLA, non
ridiscuterla.

L'unica libertà che ci prendiamo è un CLIP DI SICUREZZA dopo la z-score, motivato
sotto in zscore_normalise(). Non tocca la scala: taglia solo artefatti numerici.
══════════════════════════════════════════════════════════════════════════════
"""

# Soglia del clip di sicurezza post-normalizzazione, in unità di sigma.
# Vedi la spiegazione dettagliata nel docstring di zscore_normalise().
CLIP_SIGMA = 10.0


def clean_tracks(track_dict, valid_mask):
    """
    STEP 1 del preprocessing: pulizia minimale dei dati grezzi.

    Cosa fa (e cosa NON fa):
      - Sostituisce SOLO NaN e Inf con la media della feature (presa dal file
        dei supervisori se disponibile, altrimenti dalla media delle tracce valide).
      - NON clippa, NON calcola statistiche di normalizzazione, NON sceglie
        percentili. La scala è decisa dai supervisori (vedi nota sopra).

    Perché così minimale?
      Le feature d0, z0SinTheta ecc. nel dataset sono GIA' in un range fisico
      pulito (d0 in ±300, z0 in ±500): chi ha preparato i dati le ha già tagliate.
      Non c'è nessun sentinella -999 e nessun outlier patologico da rimuovere a
      questo stadio. L'unico rischio reale sono i NaN/Inf sporadici, che gestiamo
      sostituendoli con la media così non propagano nei calcoli a valle.

    Argomenti:
        track_dict : dict {feat: (N, T) array} — dati grezzi
        valid_mask : (N, T) bool — True = traccia reale, False = padding

    Restituisce:
        X     : (N, T, N_FEAT) float32 — dati puliti (unità fisiche), padding azzerato
        mask  : (N, T) bool — invariata
    """
    N, T = valid_mask.shape
    X = np.zeros((N, T, N_FEAT), dtype=np.float32)

    # Proviamo a usare le medie dei supervisori per l'imputazione dei NaN.
    # Se il file non c'è, ripieghiamo sulla media delle tracce valide.
    try:
        feat_mean, _, _, _ = load_norm_stats()
        use_file_mean = True
    except Exception:
        use_file_mean = False

    for i, feat in enumerate(TRACK_FEATURES):
        arr = track_dict.get(feat)
        if arr is None:
            continue   # feature non presente nel file, lascia zeri

        a = arr.astype(np.float32)

        # Media da usare per imputare i NaN/Inf
        if use_file_mean:
            mean_for_impute = float(feat_mean[i])
        else:
            valid_vals = a[valid_mask]
            valid_vals = valid_vals[np.isfinite(valid_vals)]
            mean_for_impute = float(valid_vals.mean()) if valid_vals.size else 0.0

        # Sostituisci NaN/Inf con la media
        a[~np.isfinite(a)] = mean_for_impute

        # Azzera il padding (le tracce non reali non devono portare informazione)
        a[~valid_mask] = 0.0

        X[:, :, i] = a

    return X, valid_mask


def load_norm_stats(norm_file=NORM_FILE):
    """
    Carica le statistiche di normalizzazione FORNITE DAI SUPERVISORI dal file YAML.

    Il file contiene mean e std per ogni feature (non contiene limiti di clip:
    il clip lo aggiungiamo noi a valle, vedi zscore_normalise).

    Restituisce 4 array di shape (N_FEAT,):
        mean   : media di ogni feature (dai supervisori)
        std    : deviazione standard (dai supervisori)
        lo/hi  : limiti del clip di sicurezza IN UNITA' FISICHE, ricavati come
                 mean ± CLIP_SIGMA*std. Servono se vuoi clippare prima della
                 z-score; nella pipeline standard clippiamo DOPO (in unità sigma),
                 quindi questi sono forniti solo per comodità/ispezione.
    """
    with open(norm_file) as f:
        cfg = yaml.safe_load(f)
    track_norms = cfg.get('tracks', {})
    mean = np.array([track_norms[k]['mean'] for k in TRACK_FEATURES], dtype=np.float32)
    std  = np.array([track_norms[k]['std']  for k in TRACK_FEATURES], dtype=np.float32)
    std  = np.where(std < 1e-8, 1.0, std)   # protezione da std ≈ 0
    lo   = mean - CLIP_SIGMA * std
    hi   = mean + CLIP_SIGMA * std
    return mean, std, lo, hi


def zscore_normalise(X_clean, valid_mask, norm_file=NORM_FILE, clip_sigma=CLIP_SIGMA):
    """
    STEP 2 del preprocessing: normalizzazione z-score + clip di sicurezza.

    Formula:  x_norm = (x - mu) / sigma     con mu, sigma DAI SUPERVISORI
    poi:      x_norm = clip(x_norm, -clip_sigma, +clip_sigma)

    ══════════════════════════════════════════════════════════════════════════
    PERCHE' UN CLIP A ±10 SIGMA (e non meno, e non di più)?
    ══════════════════════════════════════════════════════════════════════════
    Dopo la z-score, alcune feature hanno code estreme. Il caso limite è
    IP3D_signed_d0_significance: il fit del parametro d'impatto a volte produce
    sigma(d0) quasi nullo, e il rapporto d0/sigma(d0) esplode fino a ±29000.
    Con mean≈11 e std≈324 (valori dei supervisori), un tale valore diventa:
        (29000 - 11) / 324 ≈ 90 sigma   →  artefatto numerico puro, NON fisica.

    Distinguiamo due popolazioni dopo la normalizzazione:
      • FISICA VERA (incluse le code displaced degli Emerging Jets):
        anche le tracce più displaced stanno entro pochi sigma. Esempio: una
        significatività fisica robusta di ~800 diventa (800-11)/324 ≈ 2.4 sigma.
        Le code fisiche realistiche restano comodamente sotto i ~5 sigma.
      • ARTEFATTI NUMERICI (fit mal condizionati): stanno a decine di sigma.

    Tra le due popolazioni c'è un GAP naturale (fisica < ~5σ, artefatti > ~30σ).
    Un clip a ±10 sigma cade nel mezzo del gap:
      - NON tocca NESSUNA traccia fisica reale, nemmeno le code displaced più
        estreme degli EJ — quindi NON distrugge il segnale che cerchiamo;
      - taglia SOLO gli artefatti numerici, che non portano informazione fisica
        e che a 90 sigma destabilizzerebbero il gradiente durante il training.

    Perché non più stretto (es. ±5σ)? Inizierebbe a mordere le code displaced
    fisiche degli EJ — cioè proprio il segnale. Male.
    Perché non più largo (es. ±20σ)? Lascerebbe passare artefatti che possono
    far oscillare la loss senza aggiungere nulla di utile.

    IMPORTANTE (metodologia): la soglia ±10σ è scelta A PRIORI su base fisica,
    NON ottimizzando l'AUC. Scegliere il clip guardando l'AUC sul set con segnale
    sarebbe un data leak (faresti tuning sul test set usando il segnale, vietato
    in anomaly detection). Un eventuale test di sensibilità (±8/±10/±12) va fatto
    SOLO sul background, verificando che la frazione di tracce clippate sia
    piccola e stabile — mai guardando la separazione EJ/QCD.

    Nota: il clip è in UNITA' NORMALIZZATE (sigma) e applicato in modo UNIFORME a
    tutte le feature. Non tocca mean/std (la scala dei supervisori resta intatta):
    è solo una rete di sicurezza numerica sui valori già normalizzati.
    ══════════════════════════════════════════════════════════════════════════

    Argomenti:
        X_clean    : (N, T, N_FEAT) — output di clean_tracks()
        valid_mask : (N, T) bool
        clip_sigma : soglia del clip di sicurezza (default CLIP_SIGMA = 10).
                     Passa clip_sigma=None per DISATTIVARE il clip e usare la
                     z-score pura (massima fedeltà ai dati grezzi). Utile se vuoi
                     prima vedere il comportamento senza clip e attivarlo solo
                     se il training mostra instabilità.

    Restituisce:
        X_norm : (N, T, N_FEAT) float32 — normalizzato (e clippato se clip_sigma),
                 padding a zero
    """
    mean, std, _, _ = load_norm_stats(norm_file)

    # z-score: broadcasting di mean/std (shape (N_FEAT,)) su X (N, T, N_FEAT)
    X_norm = (X_clean - mean) / std

    # Clip di sicurezza a ±clip_sigma (in unità normalizzate). Vedi spiegazione sopra.
    # Se clip_sigma is None, si salta del tutto (z-score pura).
    if clip_sigma is not None:
        X_norm = np.clip(X_norm, -clip_sigma, clip_sigma)

    # Ri-azzera il padding: la z-score sposta gli zero del padding (non sono veri
    # zeri fisici dopo la trasformazione), quindi li riportiamo a 0.
    X_norm[~valid_mask] = 0.0

    return X_norm.astype(np.float32)


def preprocess(path, n_max=None, norm_file=NORM_FILE, clip_sigma=CLIP_SIGMA, verbose=True):
    """
    Pipeline completa: load → clean → normalise (+ clip di sicurezza).
    Funzione di convenienza per i notebook che non hanno bisogno di ispezionare
    i passi intermedi.

    Usa SEMPRE le mean/std dei supervisori (norm_dict.yaml). Vedi le note in
    zscore_normalise() per il significato del clip a ±clip_sigma.

    Restituisce (X_norm, mask) dove X_norm è (N, T, N_FEAT) float32.
    """
    tracks = load_tracks(path, n_max=n_max)
    mask   = tracks['valid'].astype(bool)

    if verbose:
        n_valid = mask.sum()
        n_total = mask.size
        print(f'  Tracce valide: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)')

    X_clean, mask = clean_tracks(tracks, mask)
    X_norm = zscore_normalise(X_clean, mask, norm_file, clip_sigma)
    return X_norm, mask


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZZAZIONE — correlazioni e distribuzioni
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_for_corr(X, mask):
    """
    Funzione interna: costruisce un array 2D (N_jets, N_FEAT) prendendo
    la media delle feature su tutte le tracce valide di ogni jet.
    Serve per calcolare la matrice di correlazione a livello di jet.

    Non usiamo tutte le tracce perché avremmo (N*T, F) con T=200 righe
    per jet, ma la maggior parte sono padding — risulterebbe in una
    matrice di correlazione dominata dagli zeri del padding.

    ESCLUSIONE JET VUOTI:
      I jet senza tracce valide diventerebbero righe di tutti zeri (assenza di
      dati, non un valore fisico). Per coerenza con make_aggregate_embeddings e
      per non distorcere la statistica, li ESCLUDIAMO dal calcolo. Con pochissimi
      jet vuoti l'effetto è trascurabile, ma su file con molti jet vuoti gli zeri
      sposterebbero la correlazione: meglio toglierli sempre.
    """
    n_valid = mask.sum(axis=1)                       # (N,) tracce valide per jet
    keep    = n_valid > 0                            # tieni solo jet con >= 1 traccia
    Xk, mk  = X[keep], mask[keep]

    m = mk[:, :, None].astype(np.float32)                                  # (Nk, T, 1)
    n = np.maximum(mk.sum(axis=1, keepdims=True), 1).astype(np.float32)    # (Nk, 1)
    return ((Xk * m).sum(axis=1) / n).astype(np.float32)                   # (Nk, F)


def plot_correlation_matrix(X, mask, title='Matrice di correlazione',
                            label_bkg=None, label_sig=None,
                            X_sig=None, mask_sig=None,
                            save_path=None, figsize=(12, 10),
                            max_jets=10_000):
    """
    Stampa la matrice di correlazione di Pearson tra le feature di traccia.

    Puoi chiamarla in due modi:
      1) Solo background → mostra una heatmap
      2) Background + segnale → mostra due heatmap affiancate + differenza

    La matrice di correlazione ti dice se due feature sono linearmente
    legate: +1 = correlazione perfetta, -1 = anticorrelazione, 0 = indipendenti.
    Valori alti (|rho| > 0.7) indicano ridondanza — potresti rimuovere una feature.
    Valori diversi tra QCD e EJ indicano che quella coppia di feature discrimina il segnale.

    Argomenti:
        X, mask        : array (N, T, F) e maschera — background (obbligatori)
        title          : titolo del plot
        X_sig, mask_sig: se forniti, aggiunge pannello segnale e differenza
        save_path      : se fornito, salva il plot in quel percorso
        max_jets       : subsample a questo numero prima del calcolo (default 10k).
                         La correlazione di Pearson è stabile già con poche migliaia
                         di jet — non serve passare l'intero dataset.
    """
    # Subsample per evitare OOM su dataset grandi (la corr. è stabile con 10k jet)
    rng = np.random.default_rng(42)
    if len(X) > max_jets:
        idx = rng.choice(len(X), max_jets, replace=False)
        X, mask = X[idx], mask[idx]
    if X_sig is not None and len(X_sig) > max_jets:
        idx_s = rng.choice(len(X_sig), max_jets, replace=False)
        X_sig, mask_sig = X_sig[idx_s], mask_sig[idx_s]

    # Aggreghiamo le tracce → (N, F) con una riga per jet
    agg_bkg  = _aggregate_for_corr(X, mask)
    corr_bkg = np.corrcoef(agg_bkg.T)   # (F, F) — correlazione tra feature

    if X_sig is not None:
        agg_sig  = _aggregate_for_corr(X_sig, mask_sig)
        corr_sig  = np.corrcoef(agg_sig.T)
        corr_diff = corr_sig - corr_bkg

        fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*2, figsize[1]))
        panels = [
            (corr_bkg,  axes[0], label_bkg or 'QCD (background)',        'coolwarm', (-1,   1  )),
            (corr_sig,  axes[1], label_sig or 'Emerging Jets (signal)',   'coolwarm', (-1,   1  )),
            (corr_diff, axes[2], 'Differenza (EJ - QCD)',                 'RdBu_r',   (-0.4, 0.4)),
        ]
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        panels  = [(corr_bkg, ax, label_bkg or 'QCD (background)', 'coolwarm', (-1, 1))]

    for corr, ax, panel_title, cmap, (vmin, vmax) in panels:
        sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    xticklabels=TRACK_FEATURES,
                    yticklabels=TRACK_FEATURES,
                    linewidths=0.5, linecolor='white',
                    annot_kws={'size': 7})
        ax.set_title(panel_title, fontsize=12, pad=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0,  labelsize=8)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Plot salvato in: {save_path}')
    plt.show()


def plot_feature_distributions(X_before, mask_before,
                               X_after,  mask_after,
                               n_features=None, n_sample=50_000,
                               save_path=None):
    """
    Confronta le distribuzioni delle feature prima e dopo la normalizzazione.
    Utile per verificare che il preprocessing non introduca artefatti.

    Per ogni feature mostra:
      - Istogramma PRIMA (unità fisiche, dopo clip)
      - Istogramma DOPO (z-score, media≈0 std≈1)

    Argomenti:
        X_before : (N, T, F) dopo clean_tracks, prima di zscore
        X_after  : (N, T, F) dopo zscore_normalise
        n_sample : quante tracce campionare per il plot (usa un sottoinsieme
                   per velocità — le distribuzioni sono stabili con 50k tracce)
    """
    feats = TRACK_FEATURES[:n_features] if n_features else TRACK_FEATURES
    n = len(feats)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))

    for i, feat in enumerate(feats):
        # Estrae solo le tracce valide (non il padding) per questo campione
        vals_before = X_before[:n_sample][mask_before[:n_sample], i]
        vals_after  = X_after[:n_sample][mask_after[:n_sample],  i]

        ax_b = axes[i, 0]
        ax_a = axes[i, 1]

        ax_b.hist(vals_before, bins=80, color='steelblue', alpha=0.8, density=True)
        ax_b.set_title(f'{feat} — dopo clip', fontsize=9)
        ax_b.set_ylabel('densita')

        ax_a.hist(vals_after, bins=80, color='darkorange', alpha=0.8, density=True)
        ax_a.set_title(f'{feat} — dopo z-score', fontsize=9)
        ax_a.axvline(0, color='red', lw=1, ls='--', label='mu=0')
        ax_a.legend(fontsize=8)

    fig.suptitle('Distribuzioni feature: prima e dopo normalizzazione', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 5. RAPPRESENTAZIONI — da (N, T, F) al formato del modello
# ══════════════════════════════════════════════════════════════════════════════

def make_aggregate_embeddings(X, mask):
    """
    Costruisce una rappresentazione flat per jet: [mean, max, std] x N_FEAT + log(n_tracce).
    Shape output: (N, N_FEAT*3 + 1) = (N, 43)

    USO CONSIGLIATO PER:
      → Autoencoder MLP (AE), Variational Autoencoder (VAE) con input flat
      → Modelli che non hanno bisogno della struttura sequenziale delle tracce

    Come funziona:
      - Per ogni jet, per ogni feature, calcola media/massimo/std sulle T tracce valide
      - Aggiunge log(n_tracce_valide) come feature scalare del jet
      - Il risultato è un unico vettore per jet, indipendente dal numero di tracce

    Perché media/max/std?
      - La media cattura il valore tipico delle tracce del jet
      - Il massimo cattura la traccia più estrema (es. la traccia con |d0| maggiore)
      - La std cattura quanto le tracce variano tra loro (alta per EJ displaced)

    PROTEZIONE JET VUOTI:
      Alcuni jet (rarissimi: ~1 su 1 milione) hanno ZERO tracce valide.
      Per il calcolo del massimo sostituiamo il padding con un sentinella basso
      (-1e9) così non vince mai; ma in un jet TUTTO padding il massimo diventa
      -1e9, un valore mostruoso che, entrando nell'embedding, farebbe esplodere
      la loss del modello. Quindi, alla fine, AZZERIAMO l'embedding dei jet vuoti:
      zero è il valore neutro dopo la z-score (= la media), quindi un jet vuoto
      viene rappresentato come "perfettamente medio", innocuo per il training.
    """
    m     = mask[:, :, None].astype(np.float32)                              # (N, T, 1)
    n_valid = mask.sum(axis=1, keepdims=True).astype(np.float32)             # (N, 1) conteggio reale
    n     = np.maximum(n_valid, 1.0)                                         # evita /0

    feat_mean = (X * m).sum(axis=1) / n                                      # (N, F)
    feat_max  = np.where(mask[:, :, None], X, -1e9).max(axis=1)             # (N, F)
    sq_mean   = ((X * m) ** 2).sum(axis=1) / n
    feat_std  = np.sqrt(np.maximum(sq_mean - feat_mean ** 2, 0.0))          # (N, F)
    log_n     = np.log1p(n_valid)                                           # (N, 1)

    emb = np.concatenate([feat_mean, feat_max, feat_std, log_n], axis=1).astype(np.float32)

    # PROTEZIONE: azzera l'embedding dei jet senza tracce valide (altrimenti
    # feat_max conterrebbe -1e9 e manderebbe in pezzi il training).
    empty_jets = (n_valid[:, 0] == 0)
    if empty_jets.any():
        emb[empty_jets] = 0.0

    return emb


def sort_by_pt(X, mask):
    """
    Ordina le tracce di ogni jet per pT decrescente (tracce più energetiche prima).
    Le tracce di padding vengono sempre messe in fondo.
    Shape input/output: (N, T, F) e (N, T) — invariata.

    USO CONSIGLIATO PER:
      → RNN / LSTM / GRU: l'ordine fisico delle tracce per pT è il più naturale
      → AE sequenziale: se vuoi mantenere la struttura di traccia senza perderla nel flat
      → Qualunque modello che processa le tracce come sequenza ordinata

    Nota per EJ: le tracce displaced hanno tipicamente basso pT, quindi con questo
    ordinamento finiscono in fondo alla sequenza — meno visibili al modello.
    Considera sort_by_d0 se vuoi mettere le tracce anomale in primo piano.
    """
    pt            = X[:, :, PT_IDX].copy()
    pt[~mask]     = -1e9                        # padding → valore sentinella bassissimo
    order         = np.argsort(-pt, axis=1)     # argsort in ordine decrescente
    return (np.take_along_axis(X,    order[:, :, None], axis=1),
            np.take_along_axis(mask, order,             axis=1))


def sort_by_d0(X, mask):
    """
    Ordina le tracce di ogni jet per |d0| decrescente (tracce più displaced prima).
    Le tracce di padding vengono sempre messe in fondo.

    USO CONSIGLIATO PER:
      → AE / VAE sequenziale orientato alla firma EJ
      → RNN / LSTM / GRU quando la firma displaced è il segnale di interesse
      → Qualunque modello per cui vuoi che le tracce anomale siano 'visibili' subito

    Motivazione fisica:
      Le tracce degli Emerging Jets decadono lontano dal vertice primario →
      hanno |d0| grande (tipicamente >1 mm, contro <0.1 mm per tracce prompt QCD).
      Mettendo queste tracce PRIME nella sequenza, il modello (specie i GRU) le
      vede prima che il suo hidden state venga 'diluito' dalle tracce normali.
      Questo migliora la capacità del modello di catturare la firma displaced.
    """
    abs_d0        = np.abs(X[:, :, D0_IDX].copy())
    abs_d0[~mask] = -1e9
    order         = np.argsort(-abs_d0, axis=1)
    return (np.take_along_axis(X,    order[:, :, None], axis=1),
            np.take_along_axis(mask, order,             axis=1))


# ══════════════════════════════════════════════════════════════════════════════
# 6. GRAFI kNN — per Graph Neural Networks
# ══════════════════════════════════════════════════════════════════════════════

def build_knn_graph(x_jet, mask_jet, k=8):
    """
    Costruisce un grafo kNN per un singolo jet nello spazio (deta, dphi).

    USO CONSIGLIATO PER:
      → Graph Autoencoder (GAE), Graph VAE
      → GNN con aggregazione spaziale (GraphSAGE, GATConv, ecc.)
      → Qualunque modello che sfrutta la struttura spaziale del jet nel piano eta-phi

    Come funziona:
      - Ogni traccia valida diventa un nodo del grafo
      - Due nodi sono connessi se sono tra i k più vicini nello spazio (deta, dphi)
        (distanza euclidea nel piano eta-phi del rivelatore)
      - Gli edge attributes codificano la differenza di posizione (delta_eta, delta_phi)

    Perché (deta, dphi)?
      E' lo spazio naturale del calorimetro: le tracce dello stesso jet
      sono vicine in questo spazio, mentre tracce da jet diversi sono lontane.
      Per i GNN che vogliono catturare la struttura di sub-jet o di cluster
      di tracce displaced, questo è lo spazio più informativo.

    Argomenti:
        x_jet    : (T, F) array normalizzato per un jet
        mask_jet : (T,) bool — True = traccia valida
        k        : numero di vicini per ogni nodo

    Restituisce:
        torch_geometric.data.Data con:
          .x          : (n_valid, F) feature dei nodi
          .edge_index : (2, n_edges) indici sorgente/destinazione degli archi
          .edge_attr  : (n_edges, 2) attributi degli archi (delta_eta, delta_phi)
    """
    valid_idx = np.where(mask_jet)[0]
    n         = len(valid_idx)
    if n == 0:
        return None   # jet senza tracce valide: salta

    x_valid = x_jet[valid_idx]
    coords  = x_valid[:, [DETA_IDX, DPHI_IDX]]   # posizioni nel piano eta-phi
    k_eff   = min(k, n - 1)                        # non puoi avere più vicini dei nodi

    if k_eff == 0:
        # Jet con una sola traccia: self-loop triviale
        edge_index = torch.zeros((2, 1), dtype=torch.long)
        edge_attr  = torch.zeros((1, 2), dtype=torch.float32)
    else:
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        # indices[:, 0] è il nodo stesso (distanza = 0), lo escludiamo
        src        = np.repeat(np.arange(n), k_eff)
        dst        = indices[:, 1:].flatten()
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_attr  = torch.tensor(coords[dst] - coords[src], dtype=torch.float32)

    return Data(x=torch.tensor(x_valid, dtype=torch.float32),
                edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)


def build_knn_graph_impact(x_jet, mask_jet, k=8):
    """
    Come build_knn_graph, ma costruisce il grafo nello spazio (IP3D_d0_sig, IP3D_z0_sig).

    USO CONSIGLIATO PER:
      → Graph Autoencoder / GNN orientati alla firma displaced di EJ
      → Quando vuoi che gli edge codifichino compatibilità con un vertice secondario

    Motivazione fisica:
      Le tracce dallo stesso vertice displaced (dark pion decay) condividono
      valori simili di significatività d0 e z0. Costruire i k-NN in questo
      spazio connette preferenzialmente tracce che potrebbero venire dallo
      stesso vertice secondario — esattamente l'informazione che ATLAS usa
      per il tagging degli EJ.
      I jet QCD hanno tracce concentrate vicino a (0, 0) in questo spazio,
      mentre gli EJ hanno cluster di tracce lontani dall'origine.
    """
    valid_idx = np.where(mask_jet)[0]
    n         = len(valid_idx)
    if n == 0:
        return None

    x_valid = x_jet[valid_idx]
    coords  = x_valid[:, [IP3D_D0_IDX, IP3D_Z0_IDX]]
    k_eff   = min(k, n - 1)

    if k_eff == 0:
        edge_index = torch.zeros((2, 1), dtype=torch.long)
        edge_attr  = torch.zeros((1, 2), dtype=torch.float32)
    else:
        nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        src        = np.repeat(np.arange(n), k_eff)
        dst        = indices[:, 1:].flatten()
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        edge_attr  = torch.tensor(coords[dst] - coords[src], dtype=torch.float32)

    return Data(x=torch.tensor(x_valid, dtype=torch.float32),
                edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)


def build_graphs(X, mask, k=8):
    """Costruisce i grafi kNN (spazio eta-phi) per tutti i jet. Restituisce lista di Data."""
    return [g for i in tqdm(range(len(X)), desc='Building eta-phi graphs')
            if (g := build_knn_graph(X[i], mask[i], k)) is not None]


def build_graphs_impact(X, mask, k=8):
    """Costruisce i grafi kNN (spazio d0_sig-z0_sig) per tutti i jet."""
    return [g for i in tqdm(range(len(X)), desc='Building impact-space graphs')
            if (g := build_knn_graph_impact(X[i], mask[i], k)) is not None]


# ══════════════════════════════════════════════════════════════════════════════
# 7. VALUTAZIONE — ROC curve e AUC
# ══════════════════════════════════════════════════════════════════════════════

def compute_roc(scores_bkg, scores_sig):
    """
    Calcola la curva ROC e l'AUC dato un anomaly score per background e segnale.

    L'anomaly score deve essere tale che valori ALTI = più anomalo.
    Per un autoencoder, l'anomaly score è l'errore di ricostruzione MSE.

    Restituisce (fpr, tpr, roc_auc):
        fpr     : False Positive Rate = QCD classificato erroneamente come EJ
        tpr     : True Positive Rate  = EJ identificati correttamente
        roc_auc : area sotto la curva (AUC), da 0.5 (random) a 1.0 (perfetto)
    """
    labels = np.concatenate([np.zeros(len(scores_bkg)), np.ones(len(scores_sig))])
    scores = np.concatenate([scores_bkg, scores_sig])
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr, auc(fpr, tpr)


def plot_roc(fpr, tpr, roc_auc, label='', ax=None, save_path=None):
    """
    Visualizza la curva ROC.
    Può ricevere un ax esistente per sovrapporre più curve sullo stesso plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f'{label}  AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate (QCD mistag rate)')
    ax.set_ylabel('True Positive Rate (EJ efficiency)')
    ax.set_title('ROC curve — anomaly detection')
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    return ax
