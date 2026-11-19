import React, { useEffect, useRef, useState, useCallback } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { Copy, Check, Camera, Pause, CameraOff, Eye, Sliders, Maximize2, Minimize2 } from 'lucide-react';
import './HandTracking.css';

// Hand landmark connections matching MediaPipe topology with palm fan mesh
const PALM_CONNECTIONS: [number, number][] = [
  [0, 1],
  [0, 5],
  [0, 9],
  [0, 13],
  [0, 17],
  [5, 9],
  [9, 13],
  [13, 17],
];

const FINGER_CONNECTIONS: [number, number][] = [
  // Thumb
  [1, 2], [2, 3], [3, 4],
  // Index
  [5, 6], [6, 7], [7, 8],
  // Middle
  [9, 10], [10, 11], [11, 12],
  // Ring
  [13, 14], [14, 15], [15, 16],
  // Pinky
  [17, 18], [18, 19], [19, 20],
];

interface GestureInfo {
  id: string;
  name: string;
  emoji: string;
  confidence: number;
}

interface CapturedLog {
  time: string;
  gesture: string;
  confidence: number;
  emoji: string;
}

interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

// Extraction des contours réels à partir des pixels de l'image binarisée (Threshold)
// Détecte la frontière où le blanc (main) rencontre le noir (fond), en n'affichant que la main isolée
function extractThresholdContours(
  threshData: Uint8ClampedArray,
  outputData: Uint8ClampedArray,
  width: number,
  height: number,
  roi: { minX: number; minY: number; maxX: number; maxY: number } | null
): number {
  const startX = roi ? Math.max(1, Math.floor(roi.minX)) : 1;
  const endX = roi ? Math.min(width - 2, Math.ceil(roi.maxX)) : width - 2;
  const startY = roi ? Math.max(1, Math.floor(roi.minY)) : 1;
  const endY = roi ? Math.min(height - 2, Math.ceil(roi.maxY)) : height - 2;

  const roiWidth = endX - startX + 1;
  const edgeBuffer = new Uint8Array(roiWidth * (endY - startY + 1));
  let edgeCount = 0;

  // 1. Détection des pixels de bordure (voisins noirs dans l'image threshold)
  for (let y = startY; y <= endY; y++) {
    const rowOffset = y * width;
    const localRow = (y - startY) * roiWidth;
    for (let x = startX; x <= endX; x++) {
      const idx = (rowOffset + x) * 4;
      if (threshData[idx] > 128) {
        // Pixel blanc (avant-plan threshold)
        const top = threshData[((y - 1) * width + x) * 4];
        const bottom = threshData[((y + 1) * width + x) * 4];
        const left = threshData[(rowOffset + (x - 1)) * 4];
        const right = threshData[(rowOffset + (x + 1)) * 4];

        if (top < 128 || bottom < 128 || left < 128 || right < 128) {
          edgeBuffer[localRow + (x - startX)] = 1;
          edgeCount++;
        } else {
          // Intérieur du masque de la main : teinte cyan haute visibilité
          outputData[idx] = 20;     // R
          outputData[idx + 1] = 120;// G
          outputData[idx + 2] = 220;// B
          outputData[idx + 3] = 255;
        }
      }
    }
  }

  // 2. Traçage du contour néon vert éclatant sur les bords détectés (épaisseur nette 2.5px)
  for (let y = startY; y <= endY; y++) {
    const rowOffset = y * width;
    const localRow = (y - startY) * roiWidth;
    for (let x = startX; x <= endX; x++) {
      if (edgeBuffer[localRow + (x - startX)] === 1) {
        for (let dy = -1; dy <= 1; dy++) {
          const py = y + dy;
          if (py < 0 || py >= height) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const px = x + dx;
            if (px < 0 || px >= width) continue;
            const targetIdx = (py * width + px) * 4;
            outputData[targetIdx] = 0;       // R
            outputData[targetIdx + 1] = 255; // G (Vert éclatant)
            outputData[targetIdx + 2] = 120; // B
            outputData[targetIdx + 3] = 255; // A
          }
        }
      }
    }
  }

  return edgeCount;
}

// Dessine le masque volumétrique complet de la main à partir des repères 3D
function drawAnatomicalHandMask(
  ctx: CanvasRenderingContext2D,
  points: { x: number; y: number }[],
  fingerWidth: number,
  palmWidth: number
) {
  ctx.save();
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = '#ffffff';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // 1. Polygone plein de la paume et éminence thénar
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  [1, 2, 5, 9, 13, 17, 0].forEach((idx) => {
    ctx.lineTo(points[idx].x, points[idx].y);
  });
  ctx.closePath();
  ctx.fill();

  // Épaisseur charnue de la paume
  ctx.lineWidth = palmWidth;
  PALM_CONNECTIONS.forEach(([i, j]) => {
    ctx.beginPath();
    ctx.moveTo(points[i].x, points[i].y);
    ctx.lineTo(points[j].x, points[j].y);
    ctx.stroke();
  });

  // Tissu interdigital / palm webbing
  ctx.lineWidth = fingerWidth * 1.35;
  ctx.beginPath();
  ctx.moveTo(points[1].x, points[1].y);
  ctx.lineTo(points[5].x, points[5].y);
  ctx.lineTo(points[9].x, points[9].y);
  ctx.lineTo(points[13].x, points[13].y);
  ctx.lineTo(points[17].x, points[17].y);
  ctx.stroke();

  // 2. Segments de chaque doigt avec épaisseur anatomique généreuse
  ctx.lineWidth = fingerWidth;
  FINGER_CONNECTIONS.forEach(([i, j]) => {
    ctx.beginPath();
    ctx.moveTo(points[i].x, points[i].y);
    ctx.lineTo(points[j].x, points[j].y);
    ctx.stroke();
  });

  // 3. Coussinets arrondis à chaque articulation et pulpe des doigts
  points.forEach((p, idx) => {
    const isTip = idx % 4 === 0 && idx > 0;
    const r = idx === 0 ? palmWidth * 0.75 : isTip ? fingerWidth * 0.75 : fingerWidth * 0.6;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.restore();
}

// Dessine le contour extérieur net et éclatant autour de la main entière
function drawHandContourOutline(
  ctx: CanvasRenderingContext2D,
  points: { x: number; y: number }[],
  fingerWidth: number
) {
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const fingers = [
    [1, 2, 3, 4],     // Pouce
    [5, 6, 7, 8],     // Index
    [9, 10, 11, 12],  // Majeur
    [13, 14, 15, 16], // Annulaire
    [17, 18, 19, 20], // Auriculaire
  ];

  // 1. Contour des doigts avec pulpe
  fingers.forEach((finger) => {
    ctx.beginPath();
    ctx.moveTo(points[finger[0]].x, points[finger[0]].y);
    for (let i = 1; i < finger.length; i++) {
      ctx.lineTo(points[finger[i]].x, points[finger[i]].y);
    }
    ctx.stroke();

    // Bout arrondi du doigt
    const tip = points[finger[finger.length - 1]];
    ctx.beginPath();
    ctx.arc(tip.x, tip.y, fingerWidth * 0.65, 0, Math.PI * 2);
    ctx.stroke();
  });

  // 2. Contour de la base de la paume et du poignet
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  ctx.lineTo(points[1].x, points[1].y);
  ctx.lineTo(points[5].x, points[5].y);
  ctx.lineTo(points[9].x, points[9].y);
  ctx.lineTo(points[13].x, points[13].y);
  ctx.lineTo(points[17].x, points[17].y);
  ctx.closePath();
  ctx.stroke();

  ctx.restore();
}

export const HandTracking: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // 3 Canvas Refs for the 3-Split Screen
  const canvasLandmarksRef = useRef<HTMLCanvasElement>(null);
  const canvasGrayscaleRef = useRef<HTMLCanvasElement>(null);
  const canvasMaskContoursRef = useRef<HTMLCanvasElement>(null);
  const thresholdOffscreenRef = useRef<HTMLCanvasElement | null>(null);

  const requestRef = useRef<number | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // States
  const [activeTab, setActiveTab] = useState<'control' | 'settings'>('control');
  const [isCameraActive, setIsCameraActive] = useState<boolean>(true);
  const isCameraActiveRef = useRef<boolean>(true);

  // Mode pour l'écran 4 : Landmarks + GrayScale ou Pure Threshold
  const [screen4Mode, setScreen4Mode] = useState<'landmarks-grayscale' | 'pure-threshold'>('landmarks-grayscale');
  const screen4ModeRef = useRef<'landmarks-grayscale' | 'pure-threshold'>('landmarks-grayscale');
  screen4ModeRef.current = screen4Mode;

  // Detection parameters
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.75);
  const [binaryThreshold, setBinaryThreshold] = useState<number>(115);
  const binaryThresholdRef = useRef<number>(115);
  binaryThresholdRef.current = binaryThreshold;
  const [invertThreshold, setInvertThreshold] = useState<boolean>(false);
  const invertThresholdRef = useRef<boolean>(false);
  invertThresholdRef.current = invertThreshold;

  const [maxHands, setMaxHands] = useState<number>(2);
  const [showSkeleton, setShowSkeleton] = useState<boolean>(true);
  const [showPoints, setShowPoints] = useState<boolean>(true);
  const [showContourMask, setShowContourMask] = useState<boolean>(true);
  const [captureGestures, setCaptureGestures] = useState<boolean>(false);

  // Real-time metrics
  const [fps, setFps] = useState<number>(0);
  const [latency, setLatency] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>('Initialisation du modèle MediaPipe...');
  const [handDetected, setHandDetected] = useState<boolean>(false);
  const [activeGesture, setActiveGesture] = useState<string | null>(null);
  const [landmarksData, setLandmarksData] = useState<string[]>([]);
  const [copied, setCopied] = useState<boolean>(false);
  const [maximizedQuadrant, setMaximizedQuadrant] = useState<1 | 2 | 3 | null>(null);
  const [contourPixelCount, setContourPixelCount] = useState<number>(0);

  // Real gestures with real computed confidences
  const [gesturesList, setGesturesList] = useState<GestureInfo[]>([
    { id: 'fist', name: 'Poing fermé', emoji: '✊', confidence: 0 },
    { id: 'open', name: 'Main ouverte', emoji: '✋', confidence: 0 },
    { id: 'peace', name: 'Signe de paix', emoji: '✌️', confidence: 0 },
  ]);

  isCameraActiveRef.current = isCameraActive;

  // Initialize MediaPipe HandLandmarker
  useEffect(() => {
    let isMounted = true;

    async function initMediaPipe() {
      try {
        setStatusMessage('Chargement du modèle MediaPipe Tasks-Vision...');

        let vision;
        try {
          vision = await FilesetResolver.forVisionTasks('/wasm');
        } catch {
          vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
          );
        }

        if (!isMounted) return;

        // Try local model first with GPU, then CPU, then CDN fallback
        const modelCandidates = [
          '/Model/hand_landmarker.task',
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        ];

        let landmarker: HandLandmarker | null = null;

        for (const modelPath of modelCandidates) {
          try {
            landmarker = await HandLandmarker.createFromOptions(vision, {
              baseOptions: {
                modelAssetPath: modelPath,
                delegate: 'GPU',
              },
              runningMode: 'VIDEO',
              numHands: maxHands,
              minHandDetectionConfidence: confidenceThreshold,
              minHandPresenceConfidence: confidenceThreshold,
              minTrackingConfidence: confidenceThreshold,
            });
            break;
          } catch {
            try {
              landmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                  modelAssetPath: modelPath,
                  delegate: 'CPU',
                },
                runningMode: 'VIDEO',
                numHands: maxHands,
                minHandDetectionConfidence: confidenceThreshold,
                minHandPresenceConfidence: confidenceThreshold,
                minTrackingConfidence: confidenceThreshold,
              });
              break;
            } catch {
              continue;
            }
          }
        }

        if (!landmarker) {
          throw new Error('Impossible de charger le modèle depuis les sources locales ou distantes.');
        }

        if (!isMounted) {
          landmarker.close();
          return;
        }

        handLandmarkerRef.current = landmarker;
        setStatusMessage('Prêt - En attente de détection');
      } catch (err) {
        console.error('Erreur chargement HandLandmarker:', err);
        setStatusMessage('Erreur : impossible de charger le modèle de détection');
      }
    }

    initMediaPipe();

    return () => {
      isMounted = false;
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
      }
    };
  }, [maxHands, confidenceThreshold]);

  // Stop and release hardware camera stream
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    [canvasLandmarksRef, canvasGrayscaleRef, canvasMaskContoursRef].forEach((ref) => {
      if (ref.current) {
        const ctx = ref.current.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, ref.current.width, ref.current.height);
      }
    });
    setIsCameraActive(false);
    isCameraActiveRef.current = false;
    setFps(0);
    setLatency(0);
    setHandDetected(false);
    setActiveGesture(null);
    setLandmarksData([]);
    setGesturesList([
      { id: 'fist', name: 'Poing fermé', emoji: '✊', confidence: 0 },
      { id: 'open', name: 'Main ouverte', emoji: '✋', confidence: 0 },
      { id: 'peace', name: 'Signe de paix', emoji: '✌️', confidence: 0 },
    ]);
    setStatusMessage('Caméra désactivée');
  }, []);

  // Request access and start hardware camera stream
  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatusMessage('Webcam non supportée par votre navigateur');
      return;
    }

    try {
      setStatusMessage('Démarrage de la caméra...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
        };
      }
      setIsCameraActive(true);
      isCameraActiveRef.current = true;
      setStatusMessage('Prêt - En attente de détection');
    } catch (err) {
      console.error('Erreur démarrage caméra:', err);
      setIsCameraActive(false);
      isCameraActiveRef.current = false;
      setStatusMessage('Accès caméra refusé ou indisponible');
    }
  }, []);

  const toggleCamera = useCallback(() => {
    if (isCameraActiveRef.current) {
      stopCamera();
    } else {
      startCamera();
    }
  }, [startCamera, stopCamera]);

  // Start Camera on mount, release on unmount
  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, [startCamera, stopCamera]);

  // Compute 100% REAL gesture confidences based on landmark 3D positions
  const analyzeRealGestures = useCallback((landmarks: LandmarkPoint[]) => {
    const dist3D = (p1: LandmarkPoint, p2: LandmarkPoint) => {
      return Math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
    };

    const wrist = landmarks[0];

    // Thumb extension ratio
    const thumbTipDist = dist3D(landmarks[4], wrist);
    const thumbMcpDist = dist3D(landmarks[2], wrist);
    const thumbExt = Math.min(1, Math.max(0, (thumbTipDist / (thumbMcpDist * 1.4) - 0.6) / 0.4));

    // Index extension ratio
    const indexTipDist = dist3D(landmarks[8], wrist);
    const indexMcpDist = dist3D(landmarks[5], wrist);
    const indexExt = Math.min(1, Math.max(0, (indexTipDist / indexMcpDist - 1.1) / 0.85));

    // Middle extension ratio
    const middleTipDist = dist3D(landmarks[12], wrist);
    const middleMcpDist = dist3D(landmarks[9], wrist);
    const middleExt = Math.min(1, Math.max(0, (middleTipDist / middleMcpDist - 1.1) / 0.85));

    // Ring extension ratio
    const ringTipDist = dist3D(landmarks[16], wrist);
    const ringMcpDist = dist3D(landmarks[13], wrist);
    const ringExt = Math.min(1, Math.max(0, (ringTipDist / ringMcpDist - 1.1) / 0.85));

    // Pinky extension ratio
    const pinkyTipDist = dist3D(landmarks[20], wrist);
    const pinkyMcpDist = dist3D(landmarks[17], wrist);
    const pinkyExt = Math.min(1, Math.max(0, (pinkyTipDist / pinkyMcpDist - 1.1) / 0.85));

    // 1. Main ouverte (Open Hand): all 5 fingers extended
    const rawOpen = thumbExt * 0.15 + indexExt * 0.25 + middleExt * 0.25 + ringExt * 0.2 + pinkyExt * 0.15;
    const openConfidence = Math.min(99, Math.max(2, Math.round(rawOpen * 100)));

    // 2. Poing fermé (Fist): all 4 fingers folded inward
    const rawFist = (1 - indexExt) * 0.3 + (1 - middleExt) * 0.3 + (1 - ringExt) * 0.25 + (1 - pinkyExt) * 0.15;
    const fistConfidence = Math.min(99, Math.max(2, Math.round(rawFist * 100)));

    // 3. Signe de paix (Peace sign / V): Index & Middle extended, Ring & Pinky folded, with spread between index & middle
    const indexMiddleSpread = Math.min(
      1,
      Math.max(0, dist3D(landmarks[8], landmarks[12]) / (dist3D(landmarks[5], landmarks[9]) * 1.5))
    );
    const rawPeace =
      (indexExt * 0.35 + middleExt * 0.35 + (1 - ringExt) * 0.15 + (1 - pinkyExt) * 0.15) *
      (0.6 + 0.4 * indexMiddleSpread);
    const peaceConfidence = Math.min(99, Math.max(2, Math.round(rawPeace * 100)));

    let topGestureId: string | null = null;
    const maxScore = Math.max(openConfidence, fistConfidence, peaceConfidence);
    if (maxScore >= 65) {
      if (maxScore === openConfidence) topGestureId = 'open';
      else if (maxScore === fistConfidence) topGestureId = 'fist';
      else if (maxScore === peaceConfidence) topGestureId = 'peace';
    }

    return {
      activeId: topGestureId,
      list: [
        { id: 'fist', name: 'Poing fermé', emoji: '✊', confidence: fistConfidence },
        { id: 'open', name: 'Main ouverte', emoji: '✋', confidence: openConfidence },
        { id: 'peace', name: 'Signe de paix', emoji: '✌️', confidence: peaceConfidence },
      ],
    };
  }, []);

  // 4-Quadrant Real-Time Detection & Rendering Loop
  useEffect(() => {
    let lastVideoTime = -1;
    let frameCounter = 0;
    let lastFpsTimestamp = performance.now();

    const renderLoop = () => {
      const video = videoRef.current;
      const cLandmarks = canvasLandmarksRef.current;
      const cGrayscale = canvasGrayscaleRef.current;
      const cMaskContours = canvasMaskContoursRef.current;

      if (
        isCameraActiveRef.current &&
        video &&
        cLandmarks &&
        cGrayscale &&
        cMaskContours &&
        video.readyState >= 2 &&
        handLandmarkerRef.current
      ) {
        const w = video.videoWidth || 640;
        const h = video.videoHeight || 360;

        // Synchronize all 3 canvas dimensions
        [cLandmarks, cGrayscale, cMaskContours].forEach((c) => {
          if (c.width !== w || c.height !== h) {
            c.width = w;
            c.height = h;
          }
        });

        const ctx1 = cLandmarks.getContext('2d');
        const ctx2 = cGrayscale.getContext('2d');
        const ctx4 = cMaskContours.getContext('2d', { willReadFrequently: true });

        if (ctx1 && ctx2 && ctx4) {
          if (video.currentTime !== lastVideoTime) {
            lastVideoTime = video.currentTime;

            // Inférence IA
            const t0 = performance.now();
            const results = handLandmarkerRef.current.detectForVideo(video, t0);
            const t1 = performance.now();
            const measuredLatency = Math.round(t1 - t0);
            setLatency(measuredLatency);

            // FPS
            frameCounter++;
            const now = performance.now();
            const elapsed = now - lastFpsTimestamp;
            if (elapsed >= 500) {
              const currentFps = Math.round((frameCounter * 1000) / elapsed);
              setFps(currentFps);
              frameCounter = 0;
              lastFpsTimestamp = now;
            }

            // === 1. ÉCRAN 1: LANDMARKS (MediaPipe) ===
            ctx1.clearRect(0, 0, w, h);
            ctx1.save();
            ctx1.scale(-1, 1);
            ctx1.drawImage(video, -w, 0, w, h);
            ctx1.restore();

            // === 2. ÉCRAN 2: GRAYSCALE (Filtre niveaux de gris fenêtre originale) ===
            ctx2.clearRect(0, 0, w, h);
            ctx2.save();
            ctx2.filter = 'grayscale(100%)';
            ctx2.scale(-1, 1);
            ctx2.drawImage(video, -w, 0, w, h);
            ctx2.restore();

            // === 3. ÉCRAN FORMES ISOLÉES DE LA MAIN (FOND 100% PUR) ===
            ctx4.clearRect(0, 0, w, h);
            ctx4.fillStyle = '#030712';
            ctx4.fillRect(0, 0, w, h);

            if (!offscreenCanvasRef.current) {
              offscreenCanvasRef.current = document.createElement('canvas');
            }
            const offCanvas = offscreenCanvasRef.current;
            if (offCanvas.width !== w || offCanvas.height !== h) {
              offCanvas.width = w;
              offCanvas.height = h;
            }
            const offCtx = offCanvas.getContext('2d');

            // Binarisation hors-écran uniquement si le mode "Contours Seuil" est sélectionné
            let threshImg: ImageData | null = null;
            if (screen4ModeRef.current === 'pure-threshold' && results?.landmarks?.length) {
              if (!thresholdOffscreenRef.current) {
                thresholdOffscreenRef.current = document.createElement('canvas');
              }
              const tCanvas = thresholdOffscreenRef.current;
              if (tCanvas.width !== w || tCanvas.height !== h) {
                tCanvas.width = w;
                tCanvas.height = h;
              }
              const tCtx = tCanvas.getContext('2d', { willReadFrequently: true });
              if (tCtx) {
                tCtx.clearRect(0, 0, w, h);
                tCtx.save();
                tCtx.scale(-1, 1);
                tCtx.drawImage(video, -w, 0, w, h);
                tCtx.restore();
                threshImg = tCtx.getImageData(0, 0, w, h);
                const d = threshImg.data;
                const th = binaryThresholdRef.current;
                const invert = invertThresholdRef.current;
                for (let i = 0; i < d.length; i += 4) {
                  const lum = (d[i] * 77 + d[i + 1] * 150 + d[i + 2] * 29) >> 8;
                  const isAbove = lum > th;
                  const bin = (invert ? !isAbove : isAbove) ? 255 : 0;
                  d[i] = bin;
                  d[i + 1] = bin;
                  d[i + 2] = bin;
                }
              }
            }

            // Traitement et rendu des mains
            if (results && results.landmarks && results.landmarks.length > 0) {
              setHandDetected(true);
              const handCount = results.landmarks.length;
              setStatusMessage(
                handCount === 1 ? 'Prêt - Main détectée' : `Prêt - ${handCount} mains détectées`
              );

              const primaryHand = results.landmarks[0];
              const realCoords = primaryHand.map(
                (p, i) =>
                  `Point ${i}: [${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)}]`
              );
              setLandmarksData(realCoords);

              const analyzed = analyzeRealGestures(primaryHand);
              setActiveGesture(analyzed.activeId);
              setGesturesList(analyzed.list);

              results.landmarks.forEach((landmarks, handIndex) => {
                let detectionConfidence = 95;
                if (results.handednesses?.[handIndex]?.[0]) {
                  detectionConfidence = Math.round(results.handednesses[handIndex][0].score * 100);
                }

                // Points en repère miroir horizontal
                const mirrorPoints = landmarks.map((p) => ({
                  x: (1 - p.x) * w,
                  y: p.y * h,
                }));

                // Calcul de la boîte englobante
                let minX = w, minY = h, maxX = 0, maxY = 0;
                mirrorPoints.forEach((p) => {
                  if (p.x < minX) minX = p.x;
                  if (p.x > maxX) maxX = p.x;
                  if (p.y < minY) minY = p.y;
                  if (p.y > maxY) maxY = p.y;
                });

                const padX = w * 0.04;
                const padY = h * 0.05;
                const boxX = Math.max(0, minX - padX);
                const boxY = Math.max(0, minY - padY);
                const boxW = Math.min(w - boxX, maxX - minX + padX * 2);
                const boxH = Math.min(h - boxY, maxY - minY + padY * 2);

                // --- RENDU QUADRANT 1 : LANDMARKS ---
                // Boîte englobante
                ctx1.save();
                ctx1.fillStyle = 'rgba(56, 189, 248, 0.12)';
                ctx1.fillRect(boxX, boxY, boxW, boxH);
                ctx1.strokeStyle = 'rgba(56, 189, 248, 0.6)';
                ctx1.lineWidth = 1.5;
                ctx1.strokeRect(boxX, boxY, boxW, boxH);

                // Label confiance
                const tagText = `Main (${detectionConfidence}%)`;
                ctx1.font = '500 13px Inter, sans-serif';
                const tagWidth = ctx1.measureText(tagText).width + 16;
                ctx1.fillStyle = 'rgba(15, 23, 42, 0.85)';
                ctx1.fillRect(boxX, Math.max(0, boxY - 26), tagWidth, 26);
                ctx1.fillStyle = '#ffffff';
                ctx1.fillText(tagText, boxX + 8, Math.max(18, boxY - 8));
                ctx1.restore();

                // Squelette
                if (showSkeleton) {
                  // Paume en étoile (Cyan)
                  ctx1.save();
                  ctx1.lineWidth = 2.5;
                  ctx1.strokeStyle = '#38bdf8';
                  ctx1.shadowColor = 'rgba(56, 189, 248, 0.6)';
                  ctx1.shadowBlur = 6;
                  PALM_CONNECTIONS.forEach(([i, j]) => {
                    ctx1.beginPath();
                    ctx1.moveTo(mirrorPoints[i].x, mirrorPoints[i].y);
                    ctx1.lineTo(mirrorPoints[j].x, mirrorPoints[j].y);
                    ctx1.stroke();
                  });

                  // Doigts (Vert néon)
                  ctx1.strokeStyle = '#4ade80';
                  ctx1.shadowColor = 'rgba(74, 222, 128, 0.6)';
                  FINGER_CONNECTIONS.forEach(([i, j]) => {
                    ctx1.beginPath();
                    ctx1.moveTo(mirrorPoints[i].x, mirrorPoints[i].y);
                    ctx1.lineTo(mirrorPoints[j].x, mirrorPoints[j].y);
                    ctx1.stroke();
                  });
                  ctx1.restore();
                }

                // Points 3D
                if (showPoints) {
                  ctx1.save();
                  mirrorPoints.forEach((p) => {
                    ctx1.beginPath();
                    ctx1.arc(p.x, p.y, 4.5, 0, 2 * Math.PI);
                    ctx1.fillStyle = '#22c55e';
                    ctx1.shadowColor = '#22c55e';
                    ctx1.shadowBlur = 8;
                    ctx1.fill();

                    ctx1.beginPath();
                    ctx1.arc(p.x, p.y, 1.5, 0, 2 * Math.PI);
                    ctx1.fillStyle = '#ffffff';
                    ctx1.shadowBlur = 0;
                    ctx1.fill();
                  });
                  ctx1.restore();
                }

                // --- RENDU QUADRANT 4 : REPRÉSENTATION HAUTE VISIBILITÉ DE LA MAIN ENTIÈRE ---
                if (showContourMask) {
                  if (screen4ModeRef.current === 'landmarks-grayscale') {
                    if (offCtx) {
                      offCtx.clearRect(0, 0, w, h);

                      // Calcul de dimensions volumétriques généreuses pour couvrir toute la chair
                      const palmDist = Math.hypot(
                        mirrorPoints[9].x - mirrorPoints[0].x,
                        mirrorPoints[9].y - mirrorPoints[0].y
                      );
                      const fingerW = Math.max(18, palmDist * 0.25);
                      const palmW = Math.max(30, palmDist * 0.42);

                      // 1. Aura lumineuse en arrière-plan pour détacher nettement la main du fond
                      ctx4.save();
                      ctx4.shadowColor = 'rgba(34, 197, 94, 0.9)';
                      ctx4.shadowBlur = 24;
                      ctx4.fillStyle = 'rgba(34, 197, 94, 0.18)';
                      drawAnatomicalHandMask(ctx4, mirrorPoints, fingerW + 8, palmW + 10);
                      ctx4.restore();

                      // 2. Dessiner le masque plein sur le canvas hors-écran
                      drawAnatomicalHandMask(offCtx, mirrorPoints, fingerW, palmW);

                      // 3. Découper la vidéo originale avec contraste et luminosité rehaussés
                      offCtx.save();
                      offCtx.globalCompositeOperation = 'source-in';
                      offCtx.filter = 'grayscale(100%) contrast(145%) brightness(120%)';
                      offCtx.scale(-1, 1);
                      offCtx.drawImage(video, -w, 0, w, h);
                      offCtx.restore();

                      // 4. Projeter la main éclatante en niveaux de gris sur l'écran 4
                      ctx4.drawImage(offCanvas, 0, 0);

                      // 5. Contour néon vert éclatant (épaisseur 3.5px avec lueur)
                      ctx4.save();
                      ctx4.strokeStyle = '#22c55e';
                      ctx4.shadowColor = '#22c55e';
                      ctx4.shadowBlur = 12;
                      ctx4.lineWidth = 3.5;
                      ctx4.lineCap = 'round';
                      ctx4.lineJoin = 'round';
                      drawHandContourOutline(ctx4, mirrorPoints, fingerW);
                      ctx4.restore();

                      // 6. Points de repère cyan éclatants aux bouts des doigts et au poignet
                      ctx4.save();
                      [0, 4, 8, 12, 16, 20].forEach((tipIdx) => {
                        const pt = mirrorPoints[tipIdx];
                        ctx4.beginPath();
                        ctx4.arc(pt.x, pt.y, 4.5, 0, Math.PI * 2);
                        ctx4.fillStyle = '#38bdf8';
                        ctx4.shadowColor = '#38bdf8';
                        ctx4.shadowBlur = 10;
                        ctx4.fill();

                        ctx4.beginPath();
                        ctx4.arc(pt.x, pt.y, 1.5, 0, Math.PI * 2);
                        ctx4.fillStyle = '#ffffff';
                        ctx4.shadowBlur = 0;
                        ctx4.fill();
                      });
                      ctx4.restore();

                      setContourPixelCount(21);
                    }
                  } else if (offCtx && threshImg) {
                    // Mode alternatif : détection de contours par pixels du threshold isolée sur la main
                    const maskImg = offCtx.createImageData(w, h);
                    const edges = extractThresholdContours(
                      threshImg.data,
                      maskImg.data,
                      w,
                      h,
                      {
                        minX: Math.max(0, boxX - 16),
                        minY: Math.max(0, boxY - 16),
                        maxX: Math.min(w, boxX + boxW + 16),
                        maxY: Math.min(h, boxY + boxH + 16),
                      }
                    );
                    offCtx.putImageData(maskImg, 0, 0);
                    ctx4.drawImage(offCanvas, 0, 0);
                    setContourPixelCount(edges);
                  }
                }
              });
            } else {
              setHandDetected(false);
              setStatusMessage('Prêt - En attente de détection');
              setActiveGesture(null);
              setLandmarksData([]);
              setContourPixelCount(0);
              setGesturesList([
                { id: 'fist', name: 'Poing fermé', emoji: '✊', confidence: 0 },
                { id: 'open', name: 'Main ouverte', emoji: '✋', confidence: 0 },
                { id: 'peace', name: 'Signe de paix', emoji: '✌️', confidence: 0 },
              ]);

              // Écran 4 : état d'attente épuré sans aucun bruit de fond
              ctx4.save();
              ctx4.fillStyle = 'rgba(148, 163, 184, 0.4)';
              ctx4.font = '500 13px Inter, sans-serif';
              ctx4.textAlign = 'center';
              ctx4.fillText('En attente de main (formes isolées)', w / 2, h / 2);
              ctx4.restore();
            }
          }
        }
      } else if (!isCameraActiveRef.current) {
        setFps(0);
        setLatency(0);
      }

      requestRef.current = requestAnimationFrame(renderLoop);
    };

    requestRef.current = requestAnimationFrame(renderLoop);

    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [showSkeleton, showPoints, showContourMask, analyzeRealGestures]);

  const handleCopyLandmarks = () => {
    if (landmarksData.length === 0) return;
    navigator.clipboard.writeText(landmarksData.join('\n'));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="dashboard-grid">
      {/* Colonne Gauche : Contrôles & Gestes */}
      <div className="left-column">
        <div className="dashboard-card">
          <div className="tabs-header">
            <button
              className={`tab-btn ${activeTab === 'control' ? 'active' : ''}`}
              onClick={() => setActiveTab('control')}
            >
              Accueil
            </button>
            <button
              className={`tab-btn ${activeTab === 'settings' ? 'active' : ''}`}
              onClick={() => setActiveTab('settings')}
            >
              Paramètres
            </button>
          </div>

          {activeTab === 'control' ? (
            <>
              <button
                className={`btn-action-primary ${isCameraActive ? 'active' : ''}`}
                onClick={toggleCamera}
                title={isCameraActive ? 'Arrêter la caméra' : 'Démarrer la caméra'}
              >
                {isCameraActive ? <Pause size={16} /> : <Camera size={16} />}
                {isCameraActive ? 'Arrêter la détection' : 'Démarrer la détection'}
              </button>

              <div className="setting-item">
                <div className="setting-header">
                  <span>Seuil de confiance</span>
                  <span className="setting-badge">{confidenceThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="slider-input"
                />
              </div>

              <div className="setting-item">
                <div className="setting-header">
                  <span>Seuil Binarisation (Threshold)</span>
                  <span className="setting-badge">{binaryThreshold}</span>
                </div>
                <input
                  type="range"
                  min="30"
                  max="220"
                  step="5"
                  value={binaryThreshold}
                  onChange={(e) => setBinaryThreshold(parseInt(e.target.value))}
                  className="slider-input"
                />
              </div>

              <div className="setting-item">
                <div className="setting-header">
                  <span>Nombre de mains max</span>
                  <span className="setting-badge">{maxHands}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="4"
                  step="1"
                  value={maxHands}
                  onChange={(e) => setMaxHands(parseInt(e.target.value))}
                  className="slider-input"
                />
              </div>

              <div className="checkbox-list">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={showSkeleton}
                    onChange={(e) => setShowSkeleton(e.target.checked)}
                  />
                  <span className="custom-checkbox">
                    <Check size={13} strokeWidth={3} />
                  </span>
                  Afficher le squelette
                </label>

                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={showPoints}
                    onChange={(e) => setShowPoints(e.target.checked)}
                  />
                  <span className="custom-checkbox">
                    <Check size={13} strokeWidth={3} />
                  </span>
                  Afficher les points
                </label>

                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={showContourMask}
                    onChange={(e) => setShowContourMask(e.target.checked)}
                  />
                  <span className="custom-checkbox">
                    <Check size={13} strokeWidth={3} />
                  </span>
                  Masque & contours de main
                </label>

                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={invertThreshold}
                    onChange={(e) => setInvertThreshold(e.target.checked)}
                  />
                  <span className="custom-checkbox">
                    <Check size={13} strokeWidth={3} />
                  </span>
                  Inverser le seuillage (fond clair)
                </label>

                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={captureGestures}
                    onChange={(e) => setCaptureGestures(e.target.checked)}
                  />
                  <span className="custom-checkbox">
                    <Check size={13} strokeWidth={3} />
                  </span>
                  Capture de gestes
                </label>
              </div>

              {/* Sélecteur de mode pour les formes isolées */}
              <div style={{ marginTop: '16px', borderTop: '1px solid var(--border-card)', paddingTop: '14px' }}>
                <div style={{ fontSize: '0.82rem', color: '#cbd5e1', marginBottom: '8px', fontWeight: 500 }}>
                  Représentation Formes Isolées :
                </div>
                <div className="mode-selector-group">
                  <button
                    className={`mode-btn ${screen4Mode === 'landmarks-grayscale' ? 'active' : ''}`}
                    onClick={() => setScreen4Mode('landmarks-grayscale')}
                    title="Isole la main entière grâce aux repères 3D et projette la vidéo en niveaux de gris"
                  >
                    Main (Landmarks+Gray)
                  </button>
                  <button
                    className={`mode-btn ${screen4Mode === 'pure-threshold' ? 'active' : ''}`}
                    onClick={() => setScreen4Mode('pure-threshold')}
                    title="Détection des contours uniquement basée sur les pixels du threshold"
                  >
                    Contours Seuil
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="settings-tab-content">
              <div className="setting-item">
                <div className="setting-header">
                  <span>Division d'écran</span>
                  <span className="setting-badge">3 flux simultanés</span>
                </div>
              </div>
              <div className="setting-item">
                <div className="setting-header">
                  <span>Modèle IA</span>
                  <span className="setting-badge">hand_landmarker.task</span>
                </div>
              </div>
              <div className="setting-item">
                <div className="setting-header">
                  <span>Accélération</span>
                  <span className="setting-badge">WebGL / GPU</span>
                </div>
              </div>
              <div className="setting-item">
                <div className="setting-header">
                  <span>Algorithme Contours</span>
                  <span className="setting-badge">Convex Hull 2D</span>
                </div>
              </div>
              <button
                className="btn-action-primary"
                onClick={() => setActiveTab('control')}
                style={{ marginTop: '16px' }}
              >
                Retour aux contrôles
              </button>
            </div>
          )}
        </div>

        {/* Gestes Détectés en temps réel */}
        <div className="dashboard-card">
          <h3 className="card-section-title">Gestes Détectés</h3>
          <div className="gestures-list">
            {gesturesList.map((g) => (
              <div
                key={g.id}
                className={`gesture-item ${activeGesture === g.id && handDetected ? 'active' : ''}`}
              >
                <span className="gesture-icon">{g.emoji}</span>
                <span className="gesture-name">{g.name}</span>
                <span className="gesture-pct">
                  {handDetected ? `(${g.confidence}%)` : `(0%)`}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Colonne Centrale : Division en 4 écrans + Guide */}
      <div className="center-column">
        {/* Élément vidéo source (invisible) */}
        <video
          ref={videoRef}
          style={{ display: 'none' }}
          playsInline
          autoPlay
          muted
        />

        {/* Grille des 3 écrans de traitement d'image */}
        <div className={`detection-grid-container ${maximizedQuadrant ? 'maximized' : ''}`}>
          {/* Écran 1 : Détection Landmarks */}
          <div className={`quadrant-box ${maximizedQuadrant && maximizedQuadrant !== 1 ? 'hidden' : ''}`}>
            <div className="quadrant-header">
              <span className="quadrant-tag">
                <Eye size={12} color="#38bdf8" /> 1. Détection Landmarks
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span className="quadrant-badge">MediaPipe</span>
                <button
                  className="quadrant-action-btn"
                  onClick={() => setMaximizedQuadrant(maximizedQuadrant === 1 ? null : 1)}
                  title={maximizedQuadrant === 1 ? 'Réduire' : 'Agrandir cet écran'}
                >
                  {maximizedQuadrant === 1 ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                </button>
              </div>
            </div>
            <div className="quadrant-content">
              <canvas ref={canvasLandmarksRef} className="quadrant-canvas" />
            </div>
          </div>

          {/* Écran 2 : Filtre GrayScale de la fenêtre d'origine */}
          <div className={`quadrant-box ${maximizedQuadrant && maximizedQuadrant !== 2 ? 'hidden' : ''}`}>
            <div className="quadrant-header">
              <span className="quadrant-tag">
                <Sliders size={12} color="#94a3b8" /> 2. Niveaux de Gris
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span className="quadrant-badge">GrayScale</span>
                <button
                  className="quadrant-action-btn"
                  onClick={() => setMaximizedQuadrant(maximizedQuadrant === 2 ? null : 2)}
                  title={maximizedQuadrant === 2 ? 'Réduire' : 'Agrandir cet écran'}
                >
                  {maximizedQuadrant === 2 ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                </button>
              </div>
            </div>
            <div className="quadrant-content">
              <canvas ref={canvasGrayscaleRef} className="quadrant-canvas" />
            </div>
          </div>

          {/* Écran 3 : Formes de la Main Isolée (Landmarks + GrayScale) ou Contours Seuil */}
          <div className={`quadrant-box quadrant-wide ${maximizedQuadrant && maximizedQuadrant !== 3 ? 'hidden' : ''}`}>
            <div className="quadrant-header">
              <span className="quadrant-tag">
                <Check size={12} color="#22c55e" />{' '}
                {screen4Mode === 'landmarks-grayscale'
                  ? 'Formes de la Main Isolée'
                  : 'Contours Filtre Seuil'}
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span className="quadrant-badge">
                  {handDetected
                    ? screen4Mode === 'landmarks-grayscale'
                      ? 'Formes Isolées'
                      : `${contourPixelCount} px contours`
                    : 'Attente'}
                </span>
                <button
                  className="quadrant-action-btn"
                  onClick={() => setMaximizedQuadrant(maximizedQuadrant === 3 ? null : 3)}
                  title={maximizedQuadrant === 3 ? 'Réduire' : 'Agrandir cet écran'}
                >
                  {maximizedQuadrant === 3 ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                </button>
              </div>
            </div>
            <div className="quadrant-content">
              <canvas ref={canvasMaskContoursRef} className="quadrant-canvas" />
            </div>
          </div>

          {/* Écran de veille quand la caméra est coupée */}
          {!isCameraActive && (
            <div className="camera-inactive-placeholder" style={{ position: 'absolute', inset: 0 }}>
              <div className="camera-off-icon-wrapper">
                <CameraOff size={38} className="camera-off-icon" />
              </div>
              <p className="camera-inactive-title">Caméra désactivée</p>
              <p className="camera-inactive-desc">
                Le capteur vidéo est arrêté et les 3 écrans de traitement sont en pause.
              </p>
              <button className="btn-activate-camera" onClick={toggleCamera}>
                <Camera size={16} /> Démarrer la détection
              </button>
            </div>
          )}
        </div>

        <div className="dashboard-card guide-card">
          <h4 className="guide-title">Guide d'utilisation</h4>
          <div className="guide-status">
            <span className={`status-dot ${!isCameraActive || !handDetected ? 'inactive' : ''}`} />
            <span>{statusMessage}</span>
          </div>
        </div>
      </div>

      {/* Colonne Droite : Données réelles FPS, Latence, Coordonnées */}
      <div className="right-column">
        <div className="dashboard-card metric-card">
          <div className="metric-header">
            <span className="metric-title">Fréquence d'images</span>
            <span className={`metric-dot ${fps > 0 ? '' : 'inactive'}`} />
          </div>
          <div className="metric-value">{fps} FPS</div>
        </div>

        <div className="dashboard-card metric-card">
          <div className="metric-header">
            <span className="metric-title">Temps de latence</span>
          </div>
          <div className="metric-value">{latency}ms</div>
        </div>

        <div className="dashboard-card terminal-card">
          <div className="terminal-header">
            <div className="mac-buttons">
              <span className="mac-dot red" />
              <span className="mac-dot yellow" />
              <span className="mac-dot green" />
            </div>
            <button
              className="copy-btn"
              onClick={handleCopyLandmarks}
              title="Copier les coordonnées réelles"
            >
              {copied ? <Check size={14} color="#22c55e" /> : <Copy size={14} />}
            </button>
          </div>
          <div className="terminal-body">
            {landmarksData.length > 0 ? (
              landmarksData.map((line, idx) => (
                <div key={idx} className="terminal-line">
                  <span className="point-idx">{line.split(':')[0]}:</span>
                  <span className="point-coords">{line.split(':')[1]}</span>
                </div>
              ))
            ) : (
              <div className="terminal-line">
                <span className="point-coords">En attente de repères réels...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default HandTracking;
