/**
 * AetherDraw Core Script - Refactored Modular Architecture (v3.0)
 * High-Performance Hand Gesture Drawing Canvas
 */

"use strict";

// ==========================================================================
// 1. POINT OBJECT POOL (GC Optimization)
// ==========================================================================
class PointPool {
    constructor(size = 2000) {
        this.pool = Array.from({ length: size }, () => ({ x: 0, y: 0, active: false }));
        this.index = 0;
    }

    acquire(x, y) {
        const pt = this.pool[this.index];
        pt.x = x;
        pt.y = y;
        pt.active = true;
        this.index = (this.index + 1) % this.pool.length;
        return pt;
    }

    releaseAll() {
        this.pool.forEach(pt => pt.active = false);
        this.index = 0;
    }
}

// ==========================================================================
// 2. POSITION & VELOCITY KALMAN FILTER (Jitter Smoothing & Latency Prediction)
// ==========================================================================
class PVKalmanFilter {
    constructor(processNoise = 0.05, measurementNoise = 4.0) {
        this.qPos = processNoise;
        this.qVel = processNoise * 0.5;
        this.r = measurementNoise;
        this.reset();
    }

    reset() {
        this.p = 0; // Position
        this.v = 0; // Velocity
        this.p00 = 1.0;
        this.p01 = 0.0;
        this.p11 = 1.0;
        this.initialized = false;
    }

    update(z, dt) {
        if (dt <= 0) dt = 0.016; // Clamp to ~60fps step if invalid
        
        if (!this.initialized) {
            this.p = z;
            this.v = 0;
            this.initialized = true;
            return this.p;
        }

        // 1. Predict
        const pPred = this.p + this.v * dt;
        const vPred = this.v;

        // Predict Covariance (P = F * P * F^T + Q)
        const p00Pred = this.p00 + 2 * dt * this.p01 + dt * dt * this.p11 + this.qPos;
        const p01Pred = this.p01 + dt * this.p11;
        const p11Pred = this.p11 + this.qVel;

        // 2. Update (Measurement)
        const y = z - pPred; // Innovation
        const s = p00Pred + this.r; // Innovation Covariance

        const k0 = p00Pred / s; // Kalman Gain 0
        const k1 = p01Pred / s; // Kalman Gain 1

        this.p = pPred + k0 * y;
        this.v = vPred + k1 * y;

        // Covariance Update (I - K H) * P
        this.p00 = (1 - k0) * p00Pred;
        this.p01 = (1 - k0) * p01Pred;
        this.p11 = -k1 * p01Pred + p11Pred;

        return this.p;
    }

    predict(seconds) {
        return this.p + this.v * seconds;
    }
}

// ==========================================================================
// 3. LOW-LATENCY CAMERA PIPELINE (Phase 2)
// ==========================================================================
class CameraPipeline {
    constructor(videoElement, onFrameCallback) {
        this.video = videoElement;
        this.onFrame = onFrameCallback;
        this.stream = null;
        this.active = false;
        
        this.videoWidth = 640;
        this.videoHeight = 480;
        this.targetFps = 60;
        
        // Latency metrics
        this.cameraLatency = 0;
        this.rVfcSupported = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
        this.isProcessingFrame = false;
    }

    async start(width = 1280, height = 720, fps = 60) {
        this.videoWidth = width;
        this.videoHeight = height;
        this.targetFps = fps;

        this.stop(); // Stop any active streams

        // Layered constraints from Premium down to basic video capture
        const constraintsList = [
            {
                video: {
                    width: { ideal: width },
                    height: { ideal: height },
                    frameRate: { ideal: fps },
                    facingMode: 'user'
                },
                audio: false
            },
            {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            },
            {
                video: true,
                audio: false
            }
        ];

        let lastError = null;
        for (const constraints of constraintsList) {
            try {
                this.stream = await navigator.mediaDevices.getUserMedia(constraints);
                break; // Stream acquired!
            } catch (e) {
                console.warn("Camera constraint set failed, trying fallback...", constraints, e);
                lastError = e;
            }
        }

        if (!this.stream) {
            console.error("All camera constraints failed:", lastError);
            throw lastError;
        }

        try {
            this.video.srcObject = this.stream;
            
            // Safe readyState synchronization to avoid metadata race conditions
            const playVideo = async () => {
                try {
                    await this.video.play();
                } catch (playError) {
                    console.warn("Muted video play promise rejected or aborted. Continuing anyway...", playError);
                }
            };

            if (this.video.readyState >= 1) {
                await playVideo();
            } else {
                await new Promise((resolve) => {
                    this.video.onloadedmetadata = async () => {
                        await playVideo();
                        resolve();
                    };
                });
            }

            this.active = true;
            this.setupLoop();
            return true;
        } catch (e) {
            console.error("Failed to start camera stream processing:", e);
            throw e;
        }
    }

    setupLoop() {
        const loop = async (now, metadata) => {
            if (!this.active) return;

            if (metadata) {
                // Precise hardware frame-receive-to-browser duration
                const receiveTime = metadata.receiveTime || now;
                this.cameraLatency = Math.max(0, performance.now() - receiveTime);
            }

            if (!this.isProcessingFrame) {
                this.isProcessingFrame = true;
                try {
                    await this.onFrame();
                } catch(e) {
                    console.error("Frame callback error:", e);
                } finally {
                    this.isProcessingFrame = false;
                }
            } // If busy, frame is skipped immediately to prevent main thread blocking

            if (this.rVfcSupported) {
                this.video.requestVideoFrameCallback(loop);
            } else {
                requestAnimationFrame(() => loop(performance.now()));
            }
        };

        if (this.rVfcSupported) {
            this.video.requestVideoFrameCallback(loop);
        } else {
            requestAnimationFrame(() => loop(performance.now()));
        }
    }

    stop() {
        this.active = false;
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
    }
}

// ==========================================================================
// 4. MEDIAPIPE TRACKER WRAPPER (Phase 3)
// ==========================================================================
class HandTracker {
    constructor(locateFileUrl, onResultsCallback) {
        this.hands = new Hands({
            locateFile: (file) => `${locateFileUrl}/${file}`
        });
        
        this.hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.75,
            minTrackingConfidence: 0.65
        });

        this.hands.onResults(onResultsCallback);
    }

    async sendFrame(videoElement) {
        await this.hands.send({ image: videoElement });
    }

    updateOptions(options) {
        this.hands.setOptions(options);
    }
}

// ==========================================================================
// 5. VECTOR DRAWING ENGINE (Phase 4)
// ==========================================================================
class Stroke {
    constructor(color, size, mode) {
        this.points = []; // Pre-allocated vector points
        this.color = color;
        this.size = size;
        this.mode = mode; // 'brush', 'neon', 'airbrush', 'laser', 'eraser'
        this.timestamp = Date.now();
        
        // Shape snapping data
        this.isShape = false;
        this.shapeType = null; // 'line', 'circle', 'rectangle', 'triangle'
        this.shapeData = null;
    }
}

class DrawingEngine {
    constructor(canvas, cacheCanvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        
        // Raster cache canvas for double buffering static strokes
        this.cacheCanvas = cacheCanvas;
        this.cacheCtx = cacheCanvas.getContext('2d');
        
        this.allStrokes = [];
        this.redoStack = [];
        this.activeStroke = null;
        this.shapeSnappingEnabled = true;

        this.laserStrokes = []; // Dynamic fading laser paths
    }

    resize(width, height) {
        // Redraw content on resize to prevent clearing canvas
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.canvas.width;
        tempCanvas.height = this.canvas.height;
        tempCanvas.getContext('2d').drawImage(this.canvas, 0, 0);

        this.canvas.width = width;
        this.canvas.height = height;
        this.cacheCanvas.width = width;
        this.cacheCanvas.height = height;

        this.ctx.drawImage(tempCanvas, 0, 0);
        this.cacheCtx.drawImage(tempCanvas, 0, 0);
        
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        this.cacheCtx.lineCap = 'round';
        this.cacheCtx.lineJoin = 'round';
    }

    startStroke(color, size, mode) {
        this.activeStroke = new Stroke(color, size, mode);
        this.redoStack = []; // Clear redo stack on new action
    }

    addPointToActive(x, y, pressure = 0.6) {
        if (!this.activeStroke) return;
        this.activeStroke.points.push({ x, y, pressure });
    }

    endStroke() {
        if (!this.activeStroke || this.activeStroke.points.length === 0) {
            this.activeStroke = null;
            return;
        }

        // Shape Detection Check
        if (this.shapeSnappingEnabled && this.activeStroke.mode !== 'eraser' && this.activeStroke.mode !== 'laser') {
            const detected = ShapeDetector.analyze(this.activeStroke.points);
            if (detected) {
                this.activeStroke.isShape = true;
                this.activeStroke.shapeType = detected.type;
                this.activeStroke.shapeData = detected;
            }
        }

        // Handle drawing category
        if (this.activeStroke.mode === 'laser') {
            this.laserStrokes.push(this.activeStroke);
        } else {
            this.allStrokes.push(this.activeStroke);
            // Draw permanently to raster cache
            this.drawStroke(this.cacheCtx, this.activeStroke);
        }

        this.activeStroke = null;
        this.commitCache();
    }

    undo() {
        if (this.allStrokes.length === 0) return;
        const popped = this.allStrokes.pop();
        this.redoStack.push(popped);
        this.redrawFullCache();
    }

    redo() {
        if (this.redoStack.length === 0) return;
        const popped = this.redoStack.pop();
        this.allStrokes.push(popped);
        this.redrawFullCache();
    }

    clear() {
        this.allStrokes = [];
        this.redoStack = [];
        this.laserStrokes = [];
        this.activeStroke = null;
        this.cacheCtx.clearRect(0, 0, this.cacheCanvas.width, this.cacheCanvas.height);
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    commitCache() {
        // Fast blit cache canvas to active drawing canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.cacheCanvas, 0, 0);
    }

    redrawFullCache() {
        this.cacheCtx.clearRect(0, 0, this.cacheCanvas.width, this.cacheCanvas.height);
        this.allStrokes.forEach(stroke => {
            this.drawStroke(this.cacheCtx, stroke);
        });
        this.commitCache();
    }

    // Main Draw Function (optimized vector drawing)
    drawStroke(ctx, stroke, isLowPerf = false) {
        if (stroke.points.length === 0) return;

        // Shape rendering branch
        if (stroke.isShape && stroke.shapeData) {
            this.drawShape(ctx, stroke, isLowPerf);
            return;
        }

        const pts = stroke.points;
        const color = stroke.color;
        const size = stroke.size;
        const mode = stroke.mode;

        // Airbrush uses dot cloud spraying
        if (mode === 'airbrush') {
            pts.forEach(p => this.drawAirbrushSpread(ctx, p.x, p.y, size, color));
            return;
        }

        // Decouple renderer for Catmull-Rom smooth paths
        this.drawCatmullRom(ctx, pts, color, size, mode, isLowPerf);
    }

    drawCatmullRom(ctx, points, color, size, mode, isLowPerf) {
        if (points.length < 2) return;

        ctx.save();
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Set baseline stroke style config
        this.applyStrokeStyle(ctx, color, size, mode, isLowPerf);

        if (points.length === 2) {
            ctx.beginPath();
            ctx.moveTo(points[0].x, points[0].y);
            ctx.lineTo(points[1].x, points[1].y);
            ctx.stroke();
            ctx.restore();
            return;
        }

        // Process points using Catmull-Rom spline curves
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);

        for (let i = 0; i < points.length - 1; i++) {
            const p0 = i === 0 ? points[0] : points[i - 1];
            const p1 = points[i];
            const p2 = points[i + 1];
            const p3 = i + 2 >= points.length ? points[points.length - 1] : points[i + 2];

            // 8 spline steps provide high smoothness at low computational cost
            const segments = 8;
            for (let tStep = 1; tStep <= segments; tStep++) {
                const t = tStep / segments;
                const t2 = t * t;
                const t3 = t2 * t;

                // Spline Basis Coefficients
                const f1 = -0.5 * t3 + t2 - 0.5 * t;
                const f2 = 1.5 * t3 - 2.5 * t2 + 1.0;
                const f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
                const f4 = 0.5 * t3 - 0.5 * t2;

                const x = p0.x * f1 + p1.x * f2 + p2.x * f3 + p3.x * f4;
                const y = p0.y * f1 + p1.y * f2 + p2.y * f3 + p3.y * f4;

                ctx.lineTo(x, y);
            }
        }

        // Draw normal and neon overlay cores
        if (mode === 'neon' && !isLowPerf) {
            // Neon Glow Pass 1: blur backdrop
            ctx.shadowBlur = size * 2.5;
            ctx.shadowColor = color;
            ctx.strokeStyle = color;
            ctx.lineWidth = size;
            ctx.stroke();

            // Neon Glow Pass 2: White central core
            ctx.shadowBlur = 0;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = Math.max(1.5, size * 0.25);
            ctx.stroke();
        } else if (mode === 'neon' && isLowPerf) {
            // Low perf Neon: opaque color backdrop with white core, no shadowBlur
            ctx.strokeStyle = color;
            ctx.lineWidth = size;
            ctx.globalAlpha = 0.45;
            ctx.stroke();
            
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = Math.max(1.5, size * 0.22);
            ctx.globalAlpha = 1.0;
            ctx.stroke();
        } else {
            ctx.stroke();
        }

        ctx.restore();
    }

    applyStrokeStyle(ctx, color, size, mode, isLowPerf) {
        if (mode === 'eraser') {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.strokeStyle = 'rgba(0,0,0,1)';
            ctx.lineWidth = size;
        } else if (mode === 'neon') {
            ctx.globalCompositeOperation = 'source-over';
            // Neon passes managed directly in line spline execution block
        } else {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = color;
            ctx.lineWidth = size;
            ctx.shadowBlur = 0;
        }
    }

    drawShape(ctx, stroke, isLowPerf) {
        const shape = stroke.shapeData;
        ctx.save();
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        this.applyStrokeStyle(ctx, stroke.color, stroke.size, stroke.mode, isLowPerf);

        ctx.beginPath();
        if (shape.type === 'circle') {
            ctx.arc(shape.center.x, shape.center.y, shape.radius, 0, 2 * Math.PI);
        } else if (shape.type === 'rectangle') {
            ctx.rect(shape.x, shape.y, shape.width, shape.height);
        } else if (shape.type === 'triangle') {
            ctx.moveTo(shape.points[0].x, shape.points[0].y);
            ctx.lineTo(shape.points[1].x, shape.points[1].y);
            ctx.lineTo(shape.points[2].x, shape.points[2].y);
            ctx.closePath();
        } else if (shape.type === 'line') {
            ctx.moveTo(shape.points[0].x, shape.points[0].y);
            ctx.lineTo(shape.points[1].x, shape.points[1].y);
        }

        // Draw stroke (manages neon blur logic)
        if (stroke.mode === 'neon' && !isLowPerf) {
            ctx.shadowBlur = stroke.size * 2.5;
            ctx.shadowColor = stroke.color;
            ctx.strokeStyle = stroke.color;
            ctx.lineWidth = stroke.size;
            ctx.stroke();

            ctx.shadowBlur = 0;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = Math.max(1.5, stroke.size * 0.25);
            ctx.stroke();
        } else if (stroke.mode === 'neon' && isLowPerf) {
            ctx.strokeStyle = stroke.color;
            ctx.lineWidth = stroke.size;
            ctx.globalAlpha = 0.45;
            ctx.stroke();

            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = Math.max(1.5, stroke.size * 0.22);
            ctx.globalAlpha = 1.0;
            ctx.stroke();
        } else {
            ctx.stroke();
        }

        ctx.restore();
    }

    drawAirbrushSpread(ctx, x, y, size, color) {
        ctx.save();
        ctx.fillStyle = color;
        // Dot count proportional to brush size
        const density = size * 2.2;
        for (let i = 0; i < density; i++) {
            const angle = Math.random() * Math.PI * 2;
            // Standard normal/Gaussian radius distribution (denser in center)
            const r = (Math.random() + Math.random() + Math.random()) / 3 * size * 1.6;
            const ptX = x + Math.cos(angle) * r;
            const ptY = y + Math.sin(angle) * r;
            ctx.fillRect(ptX, ptY, 1.2, 1.2);
        }
        ctx.restore();
    }

    // Update dynamic fading elements (Laser Pointer)
    updateAndDrawLaserStrokes(isLowPerf) {
        if (this.laserStrokes.length === 0) return;

        // Clean out dead laser strokes
        this.laserStrokes = this.laserStrokes.filter(stroke => {
            const age = Date.now() - stroke.timestamp;
            return age < 1200; // Laser pointer path has a lifetime of 1.2 seconds
        });

        // Blit permanent vectors, then draw translucent laser overlay
        this.commitCache();

        this.laserStrokes.forEach(stroke => {
            const age = Date.now() - stroke.timestamp;
            const alpha = Math.max(0, 1 - age / 1200);

            this.ctx.save();
            this.ctx.globalAlpha = alpha;
            this.drawStroke(this.ctx, stroke, isLowPerf);
            this.ctx.restore();
        });
    }

    generateSVGString(presetName) {
        let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${this.canvas.width} ${this.canvas.height}" width="100%" height="100%">`;
        
        // Background representation
        let fill = 'none';
        if (presetName === 'slate-grid') fill = '#080c14';
        else if (presetName === 'chalkboard') fill = '#1a302a';
        else if (presetName === 'white-paper') fill = '#fafbfc';
        
        if (fill !== 'none') {
            svg += `<rect width="100%" height="100%" fill="${fill}" />`;
        }

        this.allStrokes.forEach(stroke => {
            if (stroke.points.length < 2) return;

            let pathData = '';
            if (stroke.isShape) {
                const s = stroke.shapeData;
                if (s.type === 'circle') {
                    svg += `<circle cx="${s.center.x}" cy="${s.center.y}" r="${s.radius}" fill="none" stroke="${stroke.color}" stroke-width="${stroke.size}" stroke-linecap="round" />`;
                } else if (s.type === 'rectangle') {
                    svg += `<rect x="${s.x}" y="${s.y}" width="${s.width}" height="${s.height}" fill="none" stroke="${stroke.color}" stroke-width="${stroke.size}" stroke-linecap="round" />`;
                } else if (s.type === 'triangle') {
                    svg += `<polygon points="${s.points[0].x},${s.points[0].y} ${s.points[1].x},${s.points[1].y} ${s.points[2].x},${s.points[2].y}" fill="none" stroke="${stroke.color}" stroke-width="${stroke.size}" stroke-linejoin="round" />`;
                } else if (s.type === 'line') {
                    svg += `<line x1="${s.points[0].x}" y1="${s.points[0].y}" x2="${s.points[1].x}" y2="${s.points[1].y}" stroke="${stroke.color}" stroke-width="${stroke.size}" stroke-linecap="round" />`;
                }
                return;
            }

            // Normal path
            pathData = `M ${stroke.points[0].x.toFixed(1)} ${stroke.points[0].y.toFixed(1)} `;
            for (let i = 1; i < stroke.points.length; i++) {
                pathData += `L ${stroke.points[i].x.toFixed(1)} ${stroke.points[i].y.toFixed(1)} `;
            }

            if (stroke.mode === 'eraser') {
                const eraseColor = presetName === 'white-paper' ? '#fafbfc' : (presetName === 'slate-grid' ? '#080c14' : '#1a302a');
                svg += `<path d="${pathData}" fill="none" stroke="${eraseColor}" stroke-width="${stroke.size}" stroke-linecap="round" stroke-linejoin="round" />`;
            } else if (stroke.mode === 'neon') {
                svg += `<path d="${pathData}" fill="none" stroke="${stroke.color}" stroke-width="${stroke.size}" stroke-linecap="round" stroke-linejoin="round" opacity="0.65" filter="blur(${stroke.size * 0.35}px)" />`;
                svg += `<path d="${pathData}" fill="none" stroke="#ffffff" stroke-width="${Math.max(1.5, stroke.size * 0.25)}" stroke-linecap="round" stroke-linejoin="round" />`;
            } else {
                svg += `<path d="${pathData}" fill="none" stroke="${stroke.color}" stroke-width="${stroke.size}" stroke-linecap="round" stroke-linejoin="round" />`;
            }
        });

        svg += `</svg>`;
        return svg;
    }
}

// ==========================================================================
// 6. SHAPE SNAPPING ALGORITHM (Phase 10)
// ==========================================================================
class ShapeDetector {
    static analyze(points) {
        if (points.length < 12) return null;

        const start = points[0];
        const end = points[points.length - 1];
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const startEndDist = Math.hypot(dx, dy);

        // Path Length Calculation
        let pathLength = 0;
        for (let i = 1; i < points.length; i++) {
            pathLength += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
        }
        if (pathLength <= 0) return null;

        // Bounding Box
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        points.forEach(p => {
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        });

        const width = maxX - minX;
        const height = maxY - minY;
        const center = { x: minX + width / 2, y: minY + height / 2 };

        // 1. Straight Line Snap (high displacement ratio)
        if (startEndDist / pathLength > 0.94) {
            return { type: 'line', points: [start, end] };
        }

        // Closed shape check
        const isClosed = (startEndDist / pathLength < 0.22) || (startEndDist < Math.max(width, height) * 0.18);

        if (isClosed) {
            // Circle Check: variance of radii to center bounds
            let totalRadius = 0;
            const radii = points.map(p => {
                const r = Math.hypot(p.x - center.x, p.y - center.y);
                totalRadius += r;
                return r;
            });
            const avgRadius = totalRadius / points.length;

            let varSum = 0;
            radii.forEach(r => varSum += (r - avgRadius) ** 2);
            const stdDev = Math.sqrt(varSum / points.length);

            // Circle boundary check (deviation limit of 11%)
            if (stdDev / avgRadius < 0.11) {
                return { type: 'circle', center, radius: avgRadius };
            }

            // Path simplification using RDP to identify corners
            const simplified = this.rdp(points, Math.max(width, height) * 0.08);
            const corners = simplified.length - 1; // Start/End are duplicate nodes

            if (corners === 3) {
                return { type: 'triangle', points: simplified.slice(0, 3) };
            } else if (corners === 4) {
                return { type: 'rectangle', x: minX, y: minY, width, height };
            }
        }

        return null;
    }

    static rdp(points, epsilon) {
        if (points.length < 3) return points;

        let maxDist = 0;
        let index = 0;
        const end = points.length - 1;

        for (let i = 1; i < end; i++) {
            const dist = this.perpendicularDistance(points[i], points[0], points[end]);
            if (dist > maxDist) {
                maxDist = dist;
                index = i;
            }
        }

        if (maxDist > epsilon) {
            const left = this.rdp(points.slice(0, index + 1), epsilon);
            const right = this.rdp(points.slice(index), epsilon);
            return left.slice(0, left.length - 1).concat(right);
        } else {
            return [points[0], points[end]];
        }
    }

    static perpendicularDistance(pt, lineStart, lineEnd) {
        const dx = lineEnd.x - lineStart.x;
        const dy = lineEnd.y - lineStart.y;
        
        if (dx === 0 && dy === 0) {
            return Math.hypot(pt.x - lineStart.x, pt.y - lineStart.y);
        }

        const t = ((pt.x - lineStart.x) * dx + (pt.y - lineStart.y) * dy) / (dx * dx + dy * dy);
        let projection = { x: lineStart.x + t * dx, y: lineStart.y + t * dy };

        if (t < 0) projection = lineStart;
        else if (t > 1) projection = lineEnd;

        return Math.hypot(pt.x - projection.x, pt.y - projection.y);
    }
}

// ==========================================================================
// 7. REAL-TIME PARTICLE SYSTEM (Premium UX Trail)
// ==========================================================================
class ParticleSystem {
    constructor(maxParticles = 300) {
        this.particles = Array.from({ length: maxParticles }, () => ({
            x: 0,
            y: 0,
            vx: 0,
            vy: 0,
            radius: 0,
            alpha: 0,
            color: '',
            decay: 0,
            active: false
        }));
        this.emitIndex = 0;
    }

    emit(x, y, color, count = 2) {
        let spawned = 0;
        const len = this.particles.length;
        
        for (let i = 0; i < len && spawned < count; i++) {
            const idx = (this.emitIndex + i) % len;
            const p = this.particles[idx];
            if (!p.active) {
                p.x = x;
                p.y = y;
                p.vx = (Math.random() - 0.5) * 2.8;
                p.vy = (Math.random() - 0.5) * 2.8;
                p.radius = Math.random() * 3.5 + 1.2;
                p.alpha = 1.0;
                p.color = color;
                p.decay = Math.random() * 0.045 + 0.02;
                p.active = true;
                spawned++;
            }
        }
        
        this.emitIndex = (this.emitIndex + spawned) % len;
    }

    update() {
        this.particles.forEach(p => {
            if (p.active) {
                p.x += p.vx;
                p.y += p.vy;
                p.alpha -= p.decay;
                if (p.alpha <= 0) {
                    p.active = false;
                }
            }
        });
    }

    draw(ctx) {
        ctx.save();
        this.particles.forEach(p => {
            if (p.active) {
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, 2 * Math.PI);
                ctx.fillStyle = p.color;
                ctx.globalAlpha = p.alpha;
                ctx.fill();
            }
        });
        ctx.restore();
    }
}

// ==========================================================================
// 8. ADAPTIVE PERFORMANCE SYSTEM & MONITOR (Phase 7 & 12)
// ==========================================================================
class PerformanceMonitor {
    constructor(perfLevelBadge, statsElements) {
        this.badge = perfLevelBadge;
        this.stats = statsElements; // { fps, cam, track, draw, mem, res }
        
        this.fpsList = [];
        this.fpsCount = 0;
        this.lastFpsUpdate = 0;
        
        // Moving averages
        this.avgFps = 60;
        this.avgTrackTime = 15;
        this.avgCamTime = 10;
        this.avgDrawTime = 2;
    }

    tick(renderTime, camTime, trackTime, drawTime, resolutionString) {
        // Compute FPS
        const now = performance.now();
        this.fpsCount++;
        if (now - this.lastFpsUpdate >= 1000) {
            const fps = Math.min(60, Math.round((this.fpsCount * 1000) / (now - this.lastFpsUpdate)));
            this.fpsList.push(fps);
            if (this.fpsList.length > 5) this.fpsList.shift();
            this.avgFps = this.fpsList.reduce((a, b) => a + b, 0) / this.fpsList.length;
            this.fpsCount = 0;
            this.lastFpsUpdate = now;
        }

        // Exponential smoothing averages
        this.avgCamTime = this.avgCamTime * 0.9 + camTime * 0.1;
        this.avgTrackTime = this.avgTrackTime * 0.9 + trackTime * 0.1;
        this.avgDrawTime = this.avgDrawTime * 0.9 + drawTime * 0.1;

        // Memory usage (where supported)
        let memoryUsage = '--';
        if (window.performance && window.performance.memory) {
            memoryUsage = Math.round(window.performance.memory.usedJSHeapSize / (1024 * 1024));
        }

        // Update elements
        this.stats.fps.textContent = Math.round(this.avgFps);
        this.stats.cam.textContent = `${Math.round(this.avgCamTime)} ms`;
        this.stats.track.textContent = `${Math.round(this.avgTrackTime)} ms`;
        this.stats.draw.textContent = `${this.avgDrawTime.toFixed(1)} ms`;
        this.stats.mem.textContent = memoryUsage !== '--' ? `${memoryUsage} MB` : '--';
        this.stats.res.textContent = resolutionString;
    }

    updateBadge(levelName) {
        this.badge.textContent = levelName;
        this.badge.className = `hud-level-badge level-${levelName.toLowerCase()}`;
    }
}

class AdaptivePerformanceManager {
    constructor(monitor, onQualityChanged) {
        this.monitor = monitor;
        this.onQualityChanged = onQualityChanged;
        
        // Quality Level Indexes: 0 = Premium, 1 = High, 2 = Medium, 3 = Low
        this.qualityLevel = 0;
        this.levelNames = ["Premium", "High", "Medium", "Low"];
        this.lastAdjustment = 0;
    }

    check(fps, trackTime) {
        const now = performance.now();
        if (now - this.lastAdjustment < 4000) return; // Prevent hunting adjustments (cooldown)

        if (fps < 48 && this.qualityLevel < 3) {
            // Drop Quality
            this.qualityLevel++;
            this.adjust();
            this.lastAdjustment = now;
        } else if (fps > 55 && trackTime < 18 && this.qualityLevel > 0) {
            // Elevate Quality
            this.qualityLevel--;
            this.adjust();
            this.lastAdjustment = now;
        }
    }

    adjust() {
        const q = this.qualityLevel;
        this.monitor.updateBadge(this.levelNames[q]);
        
        // Map configurations to level indices
        const configs = [
            { width: 1280, height: 720, complexity: 1, skipFrames: false, lowPerfDraw: false }, // Premium
            { width: 960, height: 540, complexity: 1, skipFrames: false, lowPerfDraw: true },  // High (neon shadow disabled)
            { width: 640, height: 480, complexity: 0, skipFrames: false, lowPerfDraw: true },  // Medium (modelComplexity down)
            { width: 640, height: 480, complexity: 0, skipFrames: true, lowPerfDraw: true }    // Low (modelComplexity down, frame skip on)
        ];

        this.onQualityChanged(configs[q]);
    }
}

// ==========================================================================
// 9. UX CONTROLLER & AIR SNAP (Phase 9 & UX snapping)
// ==========================================================================
class UXController {
    constructor() {
        this.interactiveElements = [];
        this.snapTarget = null;
        this.snapDistance = 64; // Distance in pixels to snap cursor
        
        this.onboardingStep = 1;
        this.onbHandDetectedFrames = 0;
        this.onbPinchFrames = 0;
        this.onbProgressCircle = document.getElementById('onboarding-progress-circle');
    }

    cacheUIBounds() {
        this.interactiveElements = [];
        const queries = ['.mode-btn', '.color-btn', '.bg-btn', '.btn'];
        queries.forEach(query => {
            document.querySelectorAll(query).forEach(el => {
                const rect = el.getBoundingClientRect();
                this.interactiveElements.push({
                    element: el,
                    cx: rect.left + rect.width / 2,
                    cy: rect.top + rect.height / 2,
                    radius: Math.max(rect.width, rect.height) / 2
                });
            });
        });
    }

    processMagneticSnap(x, y) {
        // Clear old snapped state highlights
        if (this.snapTarget) {
            this.snapTarget.element.classList.remove('magnetic-target');
            this.snapTarget = null;
        }

        let nearest = null;
        let minDist = this.snapDistance;

        this.interactiveElements.forEach(item => {
            const dist = Math.hypot(item.cx - x, item.cy - y);
            if (dist < minDist) {
                minDist = dist;
                nearest = item;
            }
        });

        if (nearest) {
            this.snapTarget = nearest;
            nearest.element.classList.add('magnetic-target');
            
            // Snape coordinates using ease-out spring profile
            const scale = (1 - minDist / this.snapDistance) ** 2.2;
            return {
                x: x + (nearest.cx - x) * scale,
                y: y + (nearest.cy - y) * scale,
                isSnapped: true
            };
        }

        return { x, y, isSnapped: false };
    }

    triggerAirClick() {
        if (this.snapTarget) {
            this.snapTarget.element.click();
            
            // visual dynamic click trigger ripple effect on button
            const ripple = document.createElement('span');
            ripple.className = 'ripple-snapped';
            this.snapTarget.element.appendChild(ripple);
            setTimeout(() => ripple.remove(), 600);
        }
    }

    runOnboardingCalibration(trackerHandDetected, activePinch) {
        const overlay = document.getElementById('onboarding-overlay');
        const title = document.getElementById('onboarding-title');
        const desc = document.getElementById('onboarding-desc');
        const feedback = document.getElementById('onboarding-feedback');
        
        if (overlay.style.display === 'none') return;

        // Step 1: Detect Hand
        if (this.onboardingStep === 1) {
            feedback.textContent = trackerHandDetected ? "Hand visible! Hold still..." : "Searching for hand...";
            if (trackerHandDetected) {
                this.onbHandDetectedFrames++;
                const pct = Math.min(100, (this.onbHandDetectedFrames / 45) * 100);
                this.updateOnbProgress(pct);
                
                if (this.onbHandDetectedFrames >= 45) {
                    this.onboardingStep = 2;
                    this.updateOnbProgress(0);
                    document.getElementById('onb-dot-1').classList.remove('active');
                    document.getElementById('onb-dot-2').classList.add('active');
                    
                    title.textContent = "Gesture Calibrate";
                    desc.textContent = "Pinch thumb and index tip together to activate virtual brush.";
                }
            } else {
                this.onbHandDetectedFrames = Math.max(0, this.onbHandDetectedFrames - 2);
                this.updateOnbProgress(0);
            }
        }
        // Step 2: Calibrate Pinch
        else if (this.onboardingStep === 2) {
            feedback.textContent = activePinch ? "Pinch detected! Hold..." : "Pinch tips together";
            if (activePinch) {
                this.onbPinchFrames++;
                const pct = Math.min(100, (this.onbPinchFrames / 45) * 100);
                this.updateOnbProgress(pct);
                
                if (this.onbPinchFrames >= 45) {
                    this.onboardingStep = 3;
                    this.updateOnbProgress(0);
                    document.getElementById('onb-dot-2').classList.remove('active');
                    document.getElementById('onb-dot-3').classList.add('active');
                    
                    title.textContent = "Draw Practice";
                    desc.textContent = "Move your pinched hand to draw a calligraphic curve.";
                }
            } else {
                this.onbPinchFrames = Math.max(0, this.onbPinchFrames - 2);
                this.updateOnbProgress(0);
            }
        }
        // Step 3: Draw stroke practice
        else if (this.onboardingStep === 3) {
            feedback.textContent = activePinch ? "Drawing! Release to finish" : "Pinch and drag cursor";
            if (activePinch) {
                this.updateOnbProgress(50);
            } else if (this.onbPinchFrames > 10) { // If drew and released
                this.updateOnbProgress(100);
                feedback.textContent = "Calibration completed!";
                setTimeout(() => {
                    overlay.style.opacity = '0';
                    overlay.style.transform = 'translate(-50%, -45%) scale(0.95)';
                    setTimeout(() => overlay.style.display = 'none', 300);
                }, 500);
            }
        }
    }

    updateOnbProgress(pct) {
        if (!this.onbProgressCircle) return;
        const circumference = 2 * Math.PI * 40; // R=40
        const offset = circumference - (pct / 100) * circumference;
        this.onbProgressCircle.style.strokeDashoffset = offset;
    }
}

// ==========================================================================
// 10. COORDINATOR / APP ENTRY
// ==========================================================================
class AetherDrawApp {
    constructor() {
        this.pointPool = new PointPool();
        
        // Coordinates smoothers
        this.kalmanX = new PVKalmanFilter(0.06, 5.0);
        this.kalmanY = new PVKalmanFilter(0.06, 5.0);
        this.lastTrackTime = 0;

        // Web elements hooks
        this.videoEl = document.getElementById('webcam');
        this.drawingCanvas = document.getElementById('drawingCanvas');
        this.cursorCanvas = document.getElementById('cursorCanvas');
        this.overlayCanvas = document.getElementById('handOverlay');
        
        // Cache offscreen canvas for rendering
        this.cacheCanvas = document.createElement('canvas');
        this.drawingCtx = this.drawingCanvas.getContext('2d');
        this.cursorCtx = this.cursorCanvas.getContext('2d');
        this.overlayCtx = this.overlayCanvas.getContext('2d');

        // States
        this.currentColor = '#6366f1';
        this.currentBrushSize = 5;
        this.currentMode = 'brush';
        this.currentPreset = 'transparent-glass';
        this.isDrawing = false;
        this.handDetected = false;
        this.isCameraFullscreen = true;

        this.pinchGraceFrames = 0;
        this.MAX_GRACE_FRAMES = 3;

        // Managers
        this.camera = new CameraPipeline(this.videoEl, this.processCameraFrame.bind(this));
        this.tracker = new HandTracker('https://cdn.jsdelivr.net/npm/@mediapipe/hands', this.onHandTrackingResults.bind(this));
        this.drawEngine = new DrawingEngine(this.drawingCanvas, this.cacheCanvas);
        this.ux = new UXController();
        this.particles = new ParticleSystem();

        // Diagnostics
        const hudBadge = document.getElementById('hud-perf-level');
        const hudStats = {
            fps: document.getElementById('hud-fps'),
            cam: document.getElementById('hud-cam-latency'),
            track: document.getElementById('hud-track-latency'),
            draw: document.getElementById('hud-draw-latency'),
            mem: document.getElementById('hud-memory'),
            res: document.getElementById('hud-res')
        };
        this.perfMonitor = new PerformanceMonitor(hudBadge, hudStats);
        this.perfManager = new AdaptivePerformanceManager(this.perfMonitor, this.applyQualityAdjustment.bind(this));
        
        // Dynamic skip frame parameter for low-end configurations
        this.trackerFrameSkip = false;
        this.trackerFrameCounter = 0;

        this.lastInferenceTime = 0;
        this.lastDrawTickTime = 0;

        this.init();
    }

    async init() {
        document.body.classList.toggle('camera-bg-active', this.isCameraFullscreen);
        
        this.setupColorPalette();
        this.setupEventListeners();
        this.handleResize();
        this.ux.cacheUIBounds();
        
        // Set fallback UI handlers
        window.addEventListener('resize', () => {
            this.handleResize();
            this.ux.cacheUIBounds();
        });

        // Initialize webcam asynchronously so it doesn't block UI registration and ticker loop
        this.camera.start(1280, 720, 60).catch(e => {
            this.showCameraError();
        });

        // Onboarding overlay setup
        const skipOnboarding = document.getElementById('btn-skip-onboarding');
        skipOnboarding.addEventListener('click', () => {
            const overlay = document.getElementById('onboarding-overlay');
            overlay.style.opacity = '0';
            overlay.style.transform = 'translate(-50%, -45%) scale(0.95)';
            setTimeout(() => {
                overlay.style.display = 'none';
            }, 300);
        });

        // Start render ticker loop
        this.ticker();
    }

    handleResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        this.drawingCanvas.width = width;
        this.drawingCanvas.height = height;
        this.cursorCanvas.width = width;
        this.cursorCanvas.height = height;
        
        this.overlayCanvas.width = this.isCameraFullscreen ? width : 320;
        this.overlayCanvas.height = this.isCameraFullscreen ? height : 240;

        this.drawEngine.resize(width, height);
    }

    setupColorPalette() {
        const paletteContainer = document.getElementById('colorPalette');
        const customColorInput = document.getElementById('customColor');

        const colors = [
            '#f8fafc', '#64748b', '#0f172a',
            '#ef4444', '#f97316', '#f59e0b',
            '#84cc16', '#22c55e', '#10b981',
            '#06b6d4', '#3b82f6', '#6366f1',
            '#8b5cf6', '#d946ef', '#ec4899'
        ];

        colors.forEach(color => {
            const btn = document.createElement('button');
            btn.className = 'color-btn';
            btn.style.backgroundColor = color;
            btn.dataset.color = color;
            if (color === this.currentColor) btn.classList.add('active');

            btn.addEventListener('click', () => {
                document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentColor = color;
                customColorInput.value = color;
                if (this.currentMode === 'eraser') this.switchMode('brush');
            });
            paletteContainer.appendChild(btn);
        });

        customColorInput.addEventListener('input', (e) => {
            this.currentColor = e.target.value;
            document.querySelectorAll('.color-btn').forEach(b => b.classList.remove('active'));
            if (this.currentMode === 'eraser') this.switchMode('brush');
        });
    }

    setupEventListeners() {
        // Mode Selector Buttons
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => this.switchMode(btn.dataset.mode));
        });

        // Brush size slider
        const sizeInput = document.getElementById('brushSize');
        const sizeVal = document.getElementById('brushSizeValue');
        sizeInput.addEventListener('input', (e) => {
            this.currentBrushSize = parseInt(e.target.value);
            sizeVal.textContent = `${this.currentBrushSize}px`;
        });

        // Canvas presets
        document.querySelectorAll('.bg-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.bg-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const preset = btn.dataset.preset;
                this.drawingCanvas.className = '';
                this.drawingCanvas.classList.add(`bg-${preset}`);
                this.currentPreset = preset;
                this.drawEngine.redrawFullCache();
            });
        });

        // Toggle shapes checkbox
        const toggleShapes = document.getElementById('toggleShapes');
        toggleShapes.addEventListener('change', (e) => {
            this.drawEngine.shapeSnappingEnabled = e.target.checked;
        });

        // Actions
        document.getElementById('undoBtn').addEventListener('click', () => this.drawEngine.undo());
        document.getElementById('redoBtn').addEventListener('click', () => this.drawEngine.redo());
        document.getElementById('clearBtn').addEventListener('click', () => this.drawEngine.clear());
        document.getElementById('saveBtn').addEventListener('click', () => this.saveToPng());
        document.getElementById('saveSvgBtn').addEventListener('click', () => {
            const svgStr = this.drawEngine.generateSVGString(this.currentPreset);
            const blob = new Blob([svgStr], { type: 'image/svg+xml' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = `aetherdraw-${Date.now()}.svg`;
            link.click();
        });

        // UI Controls Panel Toggles
        document.getElementById('btn-toggle-video').addEventListener('click', () => {
            const wrapper = document.getElementById('video-wrapper');
            const isMinimized = wrapper.classList.toggle('minimized');
            document.getElementById('minimize-icon').style.display = isMinimized ? 'none' : 'block';
            document.getElementById('maximize-icon').style.display = isMinimized ? 'block' : 'none';
        });

        document.getElementById('btn-fullscreen-video').addEventListener('click', () => {
            this.isCameraFullscreen = !this.isCameraFullscreen;
            document.body.classList.toggle('camera-bg-active', this.isCameraFullscreen);
            document.getElementById('fullscreen-enter-icon').style.display = this.isCameraFullscreen ? 'none' : 'block';
            document.getElementById('fullscreen-exit-icon').style.display = this.isCameraFullscreen ? 'block' : 'none';
            this.handleResize();
        });

        document.getElementById('btn-close-instructions').addEventListener('click', () => {
            document.getElementById('instructions-card').style.display = 'none';
            document.getElementById('btn-help').style.display = 'flex';
        });

        document.getElementById('btn-help').addEventListener('click', () => {
            document.getElementById('instructions-card').style.display = 'flex';
            document.getElementById('btn-help').style.display = 'none';
        });
    }

    switchMode(mode) {
        this.currentMode = mode;
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
    }

    // Camera Frame Callback (runs at camera speed)
    async processCameraFrame() {
        if (this.trackerFrameSkip) {
            this.trackerFrameCounter++;
            if (this.trackerFrameCounter % 2 !== 0) return; // Skip frame inference in Low-end state
        }

        const trackerStart = performance.now();
        await this.tracker.sendFrame(this.videoEl);
        this.lastInferenceTime = performance.now() - trackerStart;
    }

    // MediaPipe Results (runs as fast as inference returns)
    onHandTrackingResults(results) {
        // Clear landmarks overlays
        this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);

        const hasHand = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
        this.ux.runOnboardingCalibration(hasHand, this.isDrawing);

        if (hasHand) {
            // Update tracking state indicators
            if (!this.handDetected) {
                this.handDetected = true;
                const badge = document.getElementById('badge-hand');
                badge.textContent = 'Active';
                badge.className = 'badge-status hand-detected';
            }

            const landmarks = results.multiHandLandmarks[0];
            this.drawSkeletonOverlay(landmarks);

            const indexFingerTip = landmarks[8];
            const thumbTip = landmarks[4];
            const wrist = landmarks[0];
            const indexKnuckle = landmarks[5];

            // Normalize coordinate calculations based on camera feed aspect crop
            const vWidth = this.camera.video.videoWidth || 640;
            const vHeight = this.camera.video.videoHeight || 480;

            let rawPt;
            if (this.isCameraFullscreen) {
                rawPt = this.mapLandmarksToScreen(indexFingerTip.x, indexFingerTip.y, vWidth, vHeight);
            } else {
                rawPt = {
                    x: (1 - indexFingerTip.x) * this.drawingCanvas.width,
                    y: indexFingerTip.y * this.drawingCanvas.height
                };
            }

            // Apply Kalman filter coordinates smoothing
            const now = performance.now();
            const dt = (now - this.lastTrackTime) / 1000;
            this.lastTrackTime = now;

            const filterX = this.kalmanX.update(rawPt.x, dt);
            const filterY = this.kalmanY.update(rawPt.y, dt);

            // Compute Motion Prediction based on camera & inference delay
            const delaySec = (this.camera.cameraLatency + this.lastInferenceTime) / 1000;
            const predX = this.kalmanX.predict(delaySec);
            const predY = this.kalmanY.predict(delaySec);

            // Snapping Cursor check
            const snapCoords = this.ux.processMagneticSnap(predX, predY);
            const targetX = snapCoords.x;
            const targetY = snapCoords.y;

            // Gesture engine relative pinch calculation (scale invariant)
            const handScale = Math.hypot(wrist.x - indexKnuckle.x, wrist.y - indexKnuckle.y);
            const pinchDist = Math.hypot(indexFingerTip.x - thumbTip.x, indexFingerTip.y - thumbTip.y);

            let isPinchGesture = false;
            if (handScale > 0.05) {
                isPinchGesture = (pinchDist / handScale) < 0.22;
            } else {
                isPinchGesture = pinchDist < 0.048; // Fallback
            }

            // Gesture open palm auto-eraser
            let isOpenPalm = false;
            const indexTip = landmarks[8], middleTip = landmarks[12], ringTip = landmarks[16], pinkyTip = landmarks[20];
            const fingerDistance = Math.hypot(indexTip.x - pinkyTip.x, indexTip.y - pinkyTip.y);
            if (handScale > 0.05 && (fingerDistance / handScale) > 1.1) {
                isOpenPalm = true;
            }

            if (isOpenPalm && this.currentMode !== 'eraser') {
                this.switchMode('eraser');
            } else if (isPinchGesture && this.currentMode === 'eraser' && isOpenPalm === false) {
                // Auto switch back to brush if pinch is active and palm is closed
                this.switchMode('brush');
            }

            // Debouncing grace frames
            let activePinch = false;
            if (isPinchGesture) {
                this.pinchGraceFrames = this.MAX_GRACE_FRAMES;
                activePinch = true;
            } else if (this.pinchGraceFrames > 0) {
                this.pinchGraceFrames--;
                activePinch = true; // Temporary drop grace
            }

            // Air clicks on snapping UI buttons
            if (activePinch && !this.isDrawing && snapCoords.isSnapped) {
                this.ux.triggerAirClick();
                // Sleep draw lock until pinch released
                this.isDrawing = false;
                return;
            }

            // Manage active drawing curves
            if (activePinch && !snapCoords.isSnapped) {
                const drawBadge = document.getElementById('badge-drawing');
                drawBadge.textContent = 'Drawing';
                drawBadge.className = 'badge-status drawing';

                if (!this.isDrawing) {
                    this.isDrawing = true;
                    this.drawEngine.startStroke(this.currentColor, this.currentBrushSize, this.currentMode);
                    this.drawEngine.addPointToActive(targetX, targetY);
                } else {
                    // Velocity calculation for calligraphy width
                    const velocity = Math.hypot(this.kalmanX.v, this.kalmanY.v);
                    const speedNormalized = Math.min(1.0, velocity / 1800);
                    // Fast = thin, Slow = thick
                    const pressure = Math.max(0.3, 1.0 - speedNormalized * 0.7);
                    
                    this.drawEngine.addPointToActive(targetX, targetY, pressure);

                    // Particle emitter trail (premium UX)
                    if (this.currentMode === 'neon') {
                        this.particles.emit(targetX, targetY, this.currentColor, 2);
                    } else if (this.currentMode === 'airbrush') {
                        this.particles.emit(targetX, targetY, 'rgba(255,255,255,0.15)', 1);
                    }
                }
            } else {
                const drawBadge = document.getElementById('badge-drawing');
                drawBadge.textContent = 'Idle';
                drawBadge.className = 'badge-status idle';

                if (this.isDrawing) {
                    this.isDrawing = false;
                    this.drawEngine.endStroke();
                }
            }

            // Store current cursor tracking coordinates
            this.activeCursorCoords = { x: targetX, y: targetY, activePinch };
        } else {
            // Hand Lost
            if (this.handDetected) {
                this.handDetected = false;
                const badge = document.getElementById('badge-hand');
                badge.textContent = 'Inactive';
                badge.className = 'badge-status no-hand';
                document.getElementById('badge-drawing').textContent = 'Idle';
                document.getElementById('badge-drawing').className = 'badge-status idle';
            }

            this.activeCursorCoords = null;
            if (this.isDrawing) {
                this.isDrawing = false;
                this.drawEngine.endStroke();
            }
            this.kalmanX.reset();
            this.kalmanY.reset();
        }
    }

    // High performance Ticker Loop (60 FPS rendering)
    ticker() {
        const tick = () => {
            const startDrawTick = performance.now();

            this.cursorCtx.clearRect(0, 0, this.cursorCanvas.width, this.cursorCanvas.height);

            // Render live temporary pointer or active strokes
            const isLowPerf = this.perfManager.qualityLevel > 0;
            
            // Render active draw stroke incrementally
            if (this.isDrawing && this.drawEngine.activeStroke) {
                this.drawEngine.commitCache(); // Draw cache static backdrop
                this.drawEngine.drawStroke(this.drawingCtx, this.drawEngine.activeStroke, isLowPerf);
            }

            // Update dynamic laser pointers (fade loops)
            this.drawEngine.updateAndDrawLaserStrokes(isLowPerf);

            // Update & Draw particles cursor trails
            this.particles.update();
            this.particles.draw(this.cursorCtx);

            // Renders cursor nodes
            if (this.activeCursorCoords) {
                this.drawCursorOverlay(this.cursorCtx, this.activeCursorCoords.x, this.activeCursorCoords.y, this.activeCursorCoords.activePinch);
            }

            const drawTickTime = performance.now() - startDrawTick;

            // Update performance monitor HUD diagnostics
            const fpsVal = Math.round(this.perfMonitor.avgFps);
            this.perfManager.check(fpsVal, this.lastInferenceTime);

            const resStr = `${this.camera.video.videoWidth || 0}x${this.camera.video.videoHeight || 0}`;
            this.perfMonitor.tick(
                performance.now() - this.lastDrawTickTime,
                this.camera.cameraLatency,
                this.lastInferenceTime,
                drawTickTime,
                resStr
            );

            this.lastDrawTickTime = performance.now();
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    }

    drawCursorOverlay(ctx, x, y, activePinch) {
        const color = activePinch ? '#f43f5e' : '#06b6d4';
        const rad = Math.max(8, this.currentBrushSize / 2);

        ctx.save();
        // Inner radius
        ctx.beginPath();
        ctx.arc(x, y, rad, 0, 2 * Math.PI);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.8;
        if (activePinch) {
            ctx.fillStyle = 'rgba(244, 63, 94, 0.08)';
            ctx.fill();
        }
        ctx.stroke();

        // Outer helper dashed ring
        ctx.beginPath();
        ctx.arc(x, y, rad + 6, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
        ctx.lineWidth = 1.0;
        ctx.setLineDash([4, 4]);
        ctx.stroke();

        // Center cross dot
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffffff';
        ctx.fill();

        ctx.restore();
    }

    drawSkeletonOverlay(landmarks) {
        const ctx = this.overlayCtx;
        ctx.save();
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.6)';
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        const points = landmarks.map(lm => {
            if (this.isCameraFullscreen) {
                const vWidth = this.camera.video.videoWidth || 640;
                const vHeight = this.camera.video.videoHeight || 480;
                return this.mapLandmarksToScreen(lm.x, lm.y, vWidth, vHeight);
            } else {
                return {
                    x: (1 - lm.x) * this.overlayCanvas.width,
                    y: lm.y * this.overlayCanvas.height
                };
            }
        });

        // Skeleton layout connections
        const connections = [
            [0, 1], [0, 5], [0, 17],
            [1, 2], [2, 3], [3, 4],
            [5, 6], [6, 7], [7, 8],
            [9, 10], [10, 11], [11, 12],
            [13, 14], [14, 15], [15, 16],
            [17, 18], [18, 19], [19, 20],
            [5, 9], [9, 13], [13, 17]
        ];

        ctx.beginPath();
        connections.forEach(([start, end]) => {
            ctx.moveTo(points[start].x, points[start].y);
            ctx.lineTo(points[end].x, points[end].y);
        });
        ctx.stroke();

        // Draw joint nodes
        points.forEach((pt, idx) => {
            ctx.beginPath();
            if (idx === 8 || idx === 4) {
                ctx.arc(pt.x, pt.y, 4.2, 0, 2 * Math.PI);
                ctx.fillStyle = '#f43f5e';
            } else {
                ctx.arc(pt.x, pt.y, 2.5, 0, 2 * Math.PI);
                ctx.fillStyle = '#06b6d4';
            }
            ctx.fill();
        });

        ctx.restore();
    }

    mapLandmarksToScreen(lmX, lmY, videoWidth, videoHeight) {
        const sWidth = window.innerWidth;
        const sHeight = window.innerHeight;

        const vRatio = videoWidth / videoHeight;
        const sRatio = sWidth / sHeight;

        let x = 0, y = 0;

        if (sRatio > vRatio) {
            const scale = sWidth / videoWidth;
            const scaledH = videoHeight * scale;
            const cropY = (scaledH - sHeight) / 2;
            x = (1 - lmX) * sWidth;
            y = lmY * scaledH - cropY;
        } else {
            const scale = sHeight / videoHeight;
            const scaledW = videoWidth * scale;
            const cropX = (scaledW - sWidth) / 2;
            x = (1 - lmX) * scaledW - cropX;
            y = lmY * sHeight;
        }

        return { x, y };
    }

    applyQualityAdjustment(config) {
        // Apply tracker model complexity and skipped frames params
        this.tracker.updateOptions({
            modelComplexity: config.complexity
        });
        this.trackerFrameSkip = config.skipFrames;
        // Dynamic camera resolution resizing is bypassed to avoid hardware-level stream interruptions and failures on the fly.
        // We rely on MediaPipe model complexity and frame skipping which are highly robust.
    }

    saveToPng() {
        const temp = document.createElement('canvas');
        temp.width = this.drawingCanvas.width;
        temp.height = this.drawingCanvas.height;
        const tempCtx = temp.getContext('2d');
        tempCtx.drawImage(this.drawingCanvas, 0, 0);

        const link = document.createElement('a');
        link.download = `aetherdraw-${Date.now()}.png`;
        link.href = temp.toDataURL('image/png');
        link.click();
    }

    showCameraError() {
        document.getElementById('badge-hand').textContent = 'Blocked';
        document.getElementById('badge-hand').className = 'badge-status no-hand';
        
        const error = document.createElement('div');
        error.className = 'camera-error-msg';
        error.innerHTML = `
            <svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="m1 1 22 22"></path>
                <path d="M21 21H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3m3-3h6l2 3h4a2 2 0 0 1 2 2v9.34"></path>
                <path d="M10.17 11.17a3 3 0 1 1 4.14 4.14"></path>
            </svg>
            <p>Camera access denied or device unavailable.<br>Please allow camera permissions and refresh.</p>
        `;
        document.getElementById('video-container').appendChild(error);
    }
}

// ==========================================================================
// 11. DIAGNOSTICS & BENCHMARKING SUITE (Phase 12 / Stage 11)
// ==========================================================================
class DiagnosticsBenchmark {
    static run(appInstance, durationMs = 5000) {
        if (!appInstance) {
            console.error("AetherDraw app instance not found.");
            return;
        }

        console.log("%c[AetherDraw Benchmark] Starting diagnostic run...", "color: #6366f1; font-weight: bold; font-size: 14px;");
        console.log(`Duration: ${durationMs}ms`);

        const results = {
            frames: 0,
            fpsSum: 0,
            camLatencySum: 0,
            inferenceTimeSum: 0,
            drawTimeSum: 0
        };

        // Mock coordinates for simulated circular drawing
        let angle = 0;
        const centerX = window.innerWidth / 2;
        const centerY = window.innerHeight / 2;
        const radius = 120;

        // Backup original camera loop state
        const wasActive = appInstance.camera.active;
        appInstance.camera.active = false; // Pause camera capture during mock run

        const benchmarkInterval = setInterval(() => {
            angle += 0.12;
            
            // Mock hand landmarks: wrist (0), thumb tip (4), index knuckle (5), index tip (8)
            const mockLandmarks = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
            
            // Setup physical scale coordinates
            mockLandmarks[0] = { x: 0.5, y: 0.75, z: 0 }; // wrist
            mockLandmarks[5] = { x: 0.5, y: 0.55, z: 0 }; // knuckle
            
            // Setup thumb & index tip close together to trigger pinch (drawing)
            mockLandmarks[4] = { x: 0.49, y: 0.4, z: 0 }; // thumb tip
            mockLandmarks[8] = { x: 0.51, y: 0.4, z: 0 }; // index tip (moves indexTip)

            // Override indexTip coordinate dynamically to simulate a circular draw path
            const normX = 0.5 + (Math.cos(angle) * radius) / window.innerWidth;
            const normY = 0.5 + (Math.sin(angle) * radius) / window.innerHeight;
            mockLandmarks[8] = { x: normX, y: normY, z: 0 };

            const startInference = performance.now();
            
            // Deliver mock landmarks to handler
            appInstance.onHandTrackingResults({
                multiHandLandmarks: [mockLandmarks]
            });
            
            const infTime = performance.now() - startInference;
            results.inferenceTimeSum += infTime;
            results.frames++;

            // Accumulate metrics from monitor HUD
            results.fpsSum += parseFloat(appInstance.perfMonitor.stats.fps.textContent) || 60;
            results.camLatencySum += parseFloat(appInstance.perfMonitor.stats.cam.textContent) || 0;
            results.drawTimeSum += parseFloat(appInstance.perfMonitor.stats.draw.textContent) || 0.5;
        }, 16); // 16ms interval runs at ~60 FPS

        setTimeout(() => {
            clearInterval(benchmarkInterval);
            
            // Restore camera capture
            appInstance.camera.active = wasActive;
            if (wasActive) appInstance.camera.setupLoop();

            const avgFps = results.fpsSum / results.frames;
            const avgCam = results.camLatencySum / results.frames;
            const avgInf = results.inferenceTimeSum / results.frames;
            const avgDraw = results.drawTimeSum / results.frames;
            const finalQuality = appInstance.perfMonitor.badge.textContent;
            let memUsage = '--';
            if (window.performance && window.performance.memory) {
                memUsage = `${Math.round(window.performance.memory.usedJSHeapSize / (1024 * 1024))} MB`;
            }

            console.log("%c[AetherDraw Benchmark] Run Complete!", "color: #10b981; font-weight: bold; font-size: 14px;");
            console.table({
                "Average FPS": avgFps.toFixed(1),
                "Average Camera Latency": `${avgCam.toFixed(1)} ms`,
                "Average Inference Time": `${avgInf.toFixed(1)} ms`,
                "Average Draw Thread Execution": `${avgDraw.toFixed(1)} ms`,
                "Final Quality Preset": finalQuality,
                "JS Heap Size": memUsage,
                "Total Frames Processed": results.frames
            });
            
            alert(`AetherDraw Diagnostics Benchmark Results:\n\n` +
                  `- Average FPS: ${avgFps.toFixed(1)}\n` +
                  `- Camera Latency: ${avgCam.toFixed(1)} ms\n` +
                  `- Inference time: ${avgInf.toFixed(1)} ms\n` +
                  `- Draw Thread time: ${avgDraw.toFixed(1)} ms\n` +
                  `- Quality Preset: ${finalQuality}\n` +
                  `- Heap Size: ${memUsage}\n\n` +
                  `Performance Level: ${avgFps > 55 ? "Excellent (Stable 60 FPS)" : "Adaptive Throttling Activated"}`);
        }, durationMs);
    }
}
window.runDiagnosticsBenchmark = (duration) => DiagnosticsBenchmark.run(window.aetherDraw, duration);

// Instantiate AetherDraw Application
window.addEventListener('DOMContentLoaded', () => {
    window.aetherDraw = new AetherDrawApp();
});
