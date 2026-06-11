# Emerging Jets - Preprocessing minimale e primo Autoencoder

## Panoramica

Questo progetto riguarda l'uso di metodi di anomaly detection non supervisionata per identificare **Emerging Jets** usando informazioni a livello di tracce.

L'idea generale e':

- usare jet QCD/background come comportamento "normale";
- addestrare un modello solo su questi jet;
- verificare se i jet di segnale, cioe' gli Emerging Jets, ottengono uno score di anomalia piu' alto.

In questa prima fase non e' stato ancora costruito un modello definitivo. Il lavoro svolto e' servito soprattutto a:

1. capire la struttura dei file `.h5`;
2. ispezionare le feature disponibili;
3. creare un dataset preprocessato minimale;
4. testare un primo autoencoder molto semplice;
5. verificare che servano feature piu' informative.

---

## File di input

I file originali sono:

```text
pp_output_val.h5
pp_output_test_background.h5
pp_output_test_signal.h5
```

Tutti e tre contengono almeno:

```text
jets
tracks
```

La struttura e':

```text
jets.shape   = (N,)
tracks.shape = (N, 200)
```

Quindi ogni elemento del dataset rappresenta un **jet**, e per ogni jet sono salvate fino a **200 tracce** associate.

Nel dataset `tracks`, il campo:

```text
valid
```

indica se una traccia e' reale oppure e' solo padding:

```text
valid = True   -> traccia reale
valid = False  -> padding
```

---

## Differenza tra i tre file

| File | Significato | Uso |
|---|---|---|
| `pp_output_val.h5` | campione di validation/background-like | usato come training set iniziale |
| `pp_output_test_background.h5` | campione test background/QCD | usato per confrontare lo score del background |
| `pp_output_test_signal.h5` | campione test signal/Emerging Jets | usato per confrontare lo score del segnale |

Il file signal contiene anche informazione truth-level, come:

```text
truth_dark_pions
truth_stable_non_geant
```

Queste informazioni non sono state usate come input del modello, perche' rappresentano verita' Monte Carlo e non feature realistiche da usare in anomaly detection.

---

## Feature escluse

Durante l'ispezione dei dataset sono state separate le feature fisiche vere da label, metadati e truth information.

Sono state escluse dagli input:

```text
isDisplaced
flavour_label
truthOriginLabel
truthVertexIndex
VSIVertexIndex
valid
mcEventWeight
eventNumber
isTagged
salt_pdisp
truth_dark_pions
truth_stable_non_geant
```

Motivo:

- `isDisplaced` e' una label del jet;
- `truthOriginLabel`, `truthVertexIndex`, `VSIVertexIndex` sono informazioni truth/label delle tracce;
- `valid` e' una maschera, non una feature numerica;
- `eventNumber`, `mcEventWeight` sono metadati;
- `isTagged` e `salt_pdisp` possono introdurre leakage perche' derivano da tagger o selezioni gia' esistenti;
- `truth_dark_pions` e `truth_stable_non_geant` sono informazioni Monte Carlo disponibili solo nel segnale.

---

## Preprocessing minimale

Per iniziare in modo semplice, e' stato costruito un dataset minimale usando solo due feature track-level:

```text
d0
z0SinTheta
```

Queste due variabili sono state scelte perche' descrivono quanto una traccia non punta bene al primary vertex, senza usare direttamente informazioni di incertezza.

In particolare:

- `d0` misura lo spostamento trasverso della traccia rispetto al primary vertex;
- `z0SinTheta` misura lo spostamento longitudinale/proiettato rispetto al primary vertex.

Sono state evitate inizialmente le variabili di significance, come:

```text
IP3D_signed_d0_significance
IP3D_signed_z0_significance
```

perche' includono informazioni legate all'incertezza, e i tutor avevano suggerito di partire senza queste quantita'.

---

## Passi di preprocessing

Per ogni file `.h5` sono stati eseguiti questi passaggi:

1. lettura di `tracks["d0"]`;
2. lettura di `tracks["z0SinTheta"]`;
3. lettura di `tracks["valid"]`;
4. normalizzazione di `d0` e `z0SinTheta` usando `norm_dict.yaml`;
5. sostituzione di eventuali valori `NaN` o `inf` con zero;
6. azzeramento delle tracce non valide;
7. salvataggio del dataset preprocessato in formato `.npz`.

La forma finale dei tensori e':

```text
X_tracks.shape   = (N, 200, 2)
track_mask.shape = (N, 200)
```

dove le due feature sono:

```text
["d0", "z0SinTheta"]
```

---

## File preprocessati generati

I dataset preprocessati sono stati salvati nella cartella:

```text
processed/
```

con i nomi:

```text
processed_minimal_val_d0_z0.npz
processed_minimal_test_background_d0_z0.npz
processed_minimal_test_signal_d0_z0.npz
```

Ogni file contiene:

```text
X_tracks
track_mask
track_feature_names
y
metadata disponibili
```

Per i file di test e' stata aggiunta una label semplice:

```text
background -> y = 0
signal     -> y = 1
```

---

## Primo autoencoder

E' stato poi testato un primo autoencoder molto semplice.

L'input per ogni jet e':

```text
200 tracce x 2 feature = 400 valori
```

Il modello appiattisce il tensore delle tracce, lo comprime in uno spazio latente e poi prova a ricostruire l'input.

La loss usata e' una **masked MSE**, cioe' un errore quadratico medio calcolato solo sulle tracce valide:

```text
loss = errore di ricostruzione sulle tracce con valid = True
```

Lo score di anomalia e' stato definito come:

```text
anomaly score = reconstruction error per jet
```

---

## Risultato attuale

Il primo autoencoder non funziona bene per distinguere background e signal.

Questo risultato non e' sorprendente, perche' il modello usa solo:

```text
d0
z0SinTheta
```

Queste due variabili contengono informazione fisica utile, ma probabilmente non sono sufficienti da sole per separare bene jet QCD e Emerging Jets con un autoencoder completamente semplice.

Quindi il risultato principale di questa fase e':

```text
la pipeline funziona,
ma le feature scelte sono troppo povere per ottenere una buona anomaly detection.
```

---

## Prossimi passi

Il prossimo passo sara' provare ad aggiungere nuove feature track-level, una alla volta o per piccoli gruppi, per capire quali migliorano la separazione tra background e signal.

Feature candidate:

```text
radiusOfFirstHit
numberOfPixelHits
numberOfSCTHits
numberOfInnermostPixelLayerHits
numberOfPixelSharedHits
numberOfSCTSharedHits
pt_log1p
dphi
deta
```

In una fase successiva si potranno testare anche le variabili legate alle incertezze o alle significance:

```text
IP3D_signed_d0_significance
IP3D_signed_z0_significance
z0SinThetaUncertainty
phiUncertainty
thetaUncertainty
qOverPUncertainty
```

Dopo aver migliorato la scelta delle feature, si potranno provare modelli piu' adatti, ad esempio:

```text
autoencoder piu' profondo
variational autoencoder
modelli set-based
transformer sulle tracce
graph neural network
```

---

## Stato del progetto

Al momento e' stata completata una prima pipeline end-to-end minimale:

```text
HDF5 raw
-> selezione feature d0/z0SinTheta
-> normalizzazione
-> masking delle tracce non valide
-> salvataggio dataset preprocessato
-> primo autoencoder
-> confronto background vs signal
```

Il codice e' quindi pronto per iterare sulle feature e sui modelli.
