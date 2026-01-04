# Séparation Vocale par U-Net

Implémentation d'un système de séparation vocale basé sur l'architecture U-Net, suivant les travaux de Jansson et al. (ISMIR 2017).

## Structure du Projet

```
├── model.py          # Architecture U-Net (PyTorch)
├── dataloader.py     # Stratégie de chargement des batchs
├── train.py          # Script d'entraînement
├── train_resume.py   # Reprise d'entraînement depuis checkpoint
├── run.ipynb         # Notebook d'inférence et calcul des métriques
├── tests/            # Résultats audio de séparation
└── STATS SDR SAR.xlsx # Métriques d'évaluation
```

## Description des Fichiers

### model.py
Architecture U-Net pour la séparation vocale :
- Encodeur : 6 couches convolutionnelles (stride 2, kernel 5×5)
- Décodeur : 6 couches de convolution transposée avec skip connections
- Activation finale : sigmoïde pour produire un masque ∈ [0,1]
- Entrée : spectrogramme (1, 512, 128)
- Sortie : masque de même dimension

### dataloader.py
Stratégie d'échantillonnage stratifié par piste :
- Chaque batch tire B pistes uniformément (avec remplacement)
- Pour chaque piste, une position temporelle aléatoire est sélectionnée
- Garantit une représentation équitable indépendamment de la durée des pistes
- Chargement par memory-mapping pour efficacité mémoire

### train.py
Script d'entraînement avec les paramètres suivants :
- Learning rate : 1e-4 (Adam optimizer)
- Batch size : 64
- Loss : L1 entre spectrogramme prédit et cible
- Détection automatique de plateau avec réduction du LR
- Sauvegarde des checkpoints

### run.ipynb
Notebook présentant :
- Pipeline d'inférence par fenêtre glissante
- Calcul des métriques SDR, SIR, SAR (mir_eval)
- Résultats sur le dataset d'évaluation
- Exemples de séparation sur deux chansons tests

## Paramètres Audio

| Paramètre | Valeur |
|-----------|--------|
| Sample rate | 8192 Hz |
| FFT size | 1024 |
| Hop length | 768 |
| Frame size | 128 frames (~11s) |

## Résultats

Évaluation sur dataset de test (pseudo-labels Demucs) :

| Métrique | Moyenne |
|----------|---------|
| SDR | ~10 dB |
| SIR | inf |
| SAR | ~10 dB |

## Usage

### Entraînement
```bash
python train.py
```

### Inférence
Voir `run.ipynb` pour des exemples de séparation et d'évaluation.

## Dépendances

- PyTorch
- librosa
- numpy
- mir_eval
- soundfile

## Référence

Jansson, A., et al. "Singing Voice Separation with Deep U-Net Convolutional Networks", ISMIR 2017.
