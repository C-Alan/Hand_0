# HandVision AI 🖐️🤖

Application web interactive de détection, suivi et analyse des mouvements de la main en temps réel, basée sur **React 19**, **TypeScript**, **Vite** et les modèles **MediaPipe Tasks-Vision** (Google).

L'application permet d'extraire les 21 points de repère (landmarks) de chaque main, d'isoler les contours via binarisation (threshold), et de classifier les gestes en direct grâce à la caméra.

---

## 📋 Prérequis

Avant de commencer, assurez-vous d'avoir installé sur votre machine :

- **Node.js** : version **18.x** ou supérieure (testé avec Node v20+ / v24+)
  - Téléchargement : [nodejs.org](https://nodejs.org/)
- **npm** (inclus avec Node.js) ou un autre gestionnaire de paquets compatible (**pnpm**, **yarn**)
- **Une webcam** fonctionnelle et connectée
- **Un navigateur web moderne** supportant WebAssembly (WASM) et WebGL/WebGPU (Google Chrome, Microsoft Edge, Mozilla Firefox, Safari)

---

## 📦 Installation des dépendances

1. Ouvrez un terminal dans le dossier racine du projet :
   ```bash
   cd c:\Users\LOU\Documents\CDAR\Hand\App_0
   ```

2. Installez toutes les dépendances listées dans `package.json` :
   ```bash
   npm install
   ```

### Dépendances principales installées :
- **`react` & `react-dom`** (v19) : Bibliothèque d'interface utilisateur
- **`@mediapipe/tasks-vision`** : Modèles d'intelligence artificielle et vision par ordinateur de Google (Hand Landmarker & WASM)
- **`lucide-react`** : Bibliothèque d'icônes
- **`vite`** & **`@vitejs/plugin-react`** : Outil de build et serveur de développement ultra-rapide
- **`typescript`** & typages associés : Typage statique

---

## 🚀 Lancement et exécution

### 1. Démarrer en mode développement

Pour lancer le serveur de développement local :

```bash
npm run dev
```

- Par défaut, l'application démarre sur : **[http://localhost:5173](http://localhost:5173)**
- Ouvrez cette adresse dans votre navigateur.
- **Important** : Acceptez l'autorisation d'accès à la webcam demandée par le navigateur.

---

### 2. Construire pour la production

Pour compiler le code TypeScript et générer les fichiers optimisés dans le dossier `dist/` :

```bash
npm run build
```

---

### 3. Prévisualiser le build de production

Pour tester localement le build généré dans le dossier `dist/` :

```bash
npm run preview
```

---

## 📁 Structure du projet

```text
App_0/
├── Model/                      # Modèle de secours hand_landmarker.task
├── public/                     # Fichiers statiques servis directement
│   ├── Model/
│   │   └── hand_landmarker.task  # Modèle MediaPipe Hand Landmarker
│   └── wasm/                   # Fichiers binaires WebAssembly MediaPipe
├── src/
│   ├── components/
│   │   └── HandTracking/
│   │       ├── HandTracking.tsx # Composant principal de détection et affichage
│   │       └── HandTracking.css # Styles du tracking et visualisations
│   ├── App.tsx                 # Composant racine avec navigation et modales
│   ├── main.tsx                # Point d'entrée React
│   └── vite-env.d.ts           # Déclarations TypeScript Vite
├── index.html                  # Page HTML racine
├── package.json                # Dépendances et scripts npm
├── tsconfig.json               # Configuration TypeScript
└── vite.config.ts              # Configuration du serveur Vite
```

---

## ⚙️ Fonctionnalités incluses

- **Affichage multi-écrans (3 ou 4 quadrants)** :
  - Flux direct avec affichage du squelette et des 21 repères anatomiques
  - Vue Niveaux de gris (Grayscale)
  - Vue Masque binaire (Threshold) avec seuillage réglable et inversion
  - Vue Détection des contours nets de la main
- **Reconnaissance de gestes en temps réel** (Main ouverte, Poing fermé, Signe de paix...)
- **Contrôles ajustables** :
  - Seuil de confiance de détection
  - Nombre maximal de mains simultanées (1 ou 2)
  - Seuil de binarisation (curseur 0 - 255)
  - Activation/Désactivation des points et du squelette
- **Export / Copie** :
  - Copie instantanée des coordonnées X, Y, Z des landmarks dans le presse-papier
- **Indicateurs de performance** :
  - Compteur de FPS en direct
  - Mesure de la latence de traitement (ms)
  - État de la caméra et du modèle IA

---

## 🛠️ Dépannage fréquent

1. **La caméra ne s'affiche pas / écran noir :**
   - Vérifiez que votre navigateur a la permission d'accéder à la webcam dans les paramètres du site (icône de cadenas dans la barre d'adresse).
   - Assurez-vous qu'aucune autre application n'utilise la caméra exclusivement.

2. **Erreur de chargement du modèle MediaPipe :**
   - Les fichiers requis sont situés dans `public/Model/` et `public/wasm/`. Si un problème réseau ou local survient, l'application bascule automatiquement sur les CDN distants de MediaPipe.

3. **Port 5173 déjà utilisé :**
   - Vous pouvez spécifier un autre port lors du lancement :
     ```bash
     npx vite --port 3000
     ```
