import React, { useState } from 'react';
import { HandTracking } from './components/HandTracking/HandTracking';
import { ChevronDown, X, BookOpen, Code2 } from 'lucide-react';

export const App: React.FC = () => {
  const [activeModal, setActiveModal] = useState<'documentation' | 'api' | null>(null);

  return (
    <div className="app-container">
      {/* Top Navbar */}
      <header className="app-navbar">
        <div className="navbar-brand">
          <svg
            className="brand-icon"
            viewBox="0 0 24 24"
            fill="currentColor"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M11 2a1 1 0 0 1 1 1v7h1V4a1 1 0 0 1 2 0v6h1V6a1 1 0 0 1 2 0v7.5c0 4.142-3.358 7.5-7.5 7.5S3 17.642 3 13.5V9a1 1 0 0 1 2 0v4.5a1 1 0 0 0 2 0V3a1 1 0 0 1 2 0v7h1V3a1 1 0 0 1 1-1z" />
          </svg>
          <span className="brand-title">HandVision AI</span>
        </div>

        <nav className="navbar-nav">
          <a
            href="#accueil"
            className={`nav-link ${activeModal === null ? 'active' : ''}`}
            onClick={(e) => {
              e.preventDefault();
              setActiveModal(null);
            }}
          >
            Accueil
          </a>
          <a
            href="#documentation"
            className={`nav-link ${activeModal === 'documentation' ? 'active' : ''}`}
            onClick={(e) => {
              e.preventDefault();
              setActiveModal('documentation');
            }}
          >
            Documentation
          </a>
          <a
            href="#api"
            className={`nav-link ${activeModal === 'api' ? 'active' : ''}`}
            onClick={(e) => {
              e.preventDefault();
              setActiveModal('api');
            }}
          >
            API
          </a>

          <div className="nav-profile">
            <img
              src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=80&q=80"
              alt="Profil utilisateur"
              className="profile-avatar"
              onError={(e) => {
                // Fallback if offline
                (e.currentTarget as HTMLImageElement).src =
                  'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="%233b82f6"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 4-6 8-6s8 2 8 6"/></svg>';
              }}
            />
            <ChevronDown className="profile-chevron" />
          </div>
        </nav>
      </header>

      {/* Main Content Dashboard */}
      <main>
        <HandTracking />
      </main>

      {/* Modal Documentation */}
      {activeModal === 'documentation' && (
        <div className="modal-overlay" onClick={() => setActiveModal(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <BookOpen size={20} color="#38bdf8" />
                <h3 className="modal-title">Documentation HandVision AI</h3>
              </div>
              <button className="modal-close-btn" onClick={() => setActiveModal(null)}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <div className="modal-section">
                <h4 className="modal-section-title">Modèle et Architecture</h4>
                <p>
                  HandVision AI s'appuie sur le modèle de réseau de neurones convolutifs et de régression de repères 3D <code>hand_landmarker.task</code> de Google MediaPipe. Il s'exécute directement dans le navigateur via WebAssembly et accélération WebGL/GPU.
                </p>
              </div>

              <div className="modal-section">
                <h4 className="modal-section-title">21 Repères Anatomiques (3D Landmarks)</h4>
                <p>Pour chaque main détectée, 21 points clés 3D sont renvoyés :</p>
                <div className="modal-code-block">
{`0: Poignet (Wrist)
1-4: Pouce (CMC, MCP, IP, TIP)
5-8: Index (MCP, PIP, DIP, TIP)
9-12: Majeur (MCP, PIP, DIP, TIP)
13-16: Annulaire (MCP, PIP, DIP, TIP)
17-20: Auriculaire (MCP, PIP, DIP, TIP)`}
                </div>
              </div>

              <div className="modal-section">
                <h4 className="modal-section-title">Classification Géométrique des Gestes</h4>
                <p>
                  Les pourcentages de confiance sont calculés en temps réel d'après les ratios de distance euclidienne entre les extrémités des doigts et la base de la paume, ainsi que l'écart angulaire en V entre l'index et le majeur.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Modal API */}
      {activeModal === 'api' && (
        <div className="modal-overlay" onClick={() => setActiveModal(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Code2 size={20} color="#38bdf8" />
                <h3 className="modal-title">Intégration API & Code</h3>
              </div>
              <button className="modal-close-btn" onClick={() => setActiveModal(null)}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <div className="modal-section">
                <h4 className="modal-section-title">Exemple d'appel du modèle</h4>
                <p>Initialisation et inférence continue à partir d'un flux vidéo HTML5 :</p>
                <div className="modal-code-block">
{`import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const vision = await FilesetResolver.forVisionTasks('/wasm');
const handLandmarker = await HandLandmarker.createFromOptions(vision, {
  baseOptions: {
    modelAssetPath: '/Model/hand_landmarker.task',
    delegate: 'GPU',
  },
  runningMode: 'VIDEO',
  numHands: 2,
  minHandDetectionConfidence: 0.75,
});

// Dans la boucle requestAnimationFrame :
const results = handLandmarker.detectForVideo(videoElement, performance.now());
console.log(results.landmarks);`}
                </div>
              </div>

              <div className="modal-section">
                <h4 className="modal-section-title">Format des résultats</h4>
                <p>
                  Chaque point contient des coordonnées normalisées : <code>x</code> [0, 1], <code>y</code> [0, 1], et <code>z</code> représentant la profondeur relative par rapport au poignet.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
