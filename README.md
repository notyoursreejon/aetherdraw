# AetherDraw — High-Fidelity Hand Gesture Drawing Canvas (v2.0)

AetherDraw is a premium, developer-drafting styled web application that lets you draw on a transparent glass board using hand gestures tracked by your webcam. 

Version 2.0 introduces an advanced **high-precision tracking engine** and a gorgeous **glassmorphic design system** with vibrant neon details.

---

## 🚀 Key Upgrades in v2.0

### 1. Advanced Drawing Precision & Filters
*   **Adaptive Coordinate Smoothing (EMA)**: Uses a velocity-sensitive Exponential Moving Average filter. When drawing slowly, it heavily dampens noise for clean lines; when drawing quickly, it dynamically increases responsiveness to eliminate input latency.
*   **Quadratic Bezier Interpolation**: Coordinates are drawn using midpoint-to-midpoint quadratic Bezier curves (`quadraticCurveTo`), eliminating angular joint points and producing calligraphic lines.
*   **Distance-Invariant Pinch Calibration**: Calculates the physical size of your hand in the camera frame by tracking the wrist-to-knuckle distance. This scales the pinch threshold, ensuring precise tracking whether your hand is close to the webcam or far away.
*   **Gesture State Debouncer**: Implements a 3-frame grace buffer to prevent line breaking due to momentary detection drops.

### 2. Immersive Visual Modes
*   **Fullscreen Camera Overlay**: Scales the webcam feed to fill the viewport background at low opacity. Camera landmarks are mapped 1-to-1 to screen coordinates (accounting for aspect-ratio crops), allowing you to draw directly over your hand.
*   **Premium Background Presets**:
    *   **Glass Board (Default)**: Semi-transparent charcoal whiteboard with frosted glass blurs.
    *   **Drafting Slate Grid**: Deep slate blueprint background with neon blue gridlines.
    *   **Chalkboard**: Dusty green chalkboard texture with chalk-like visual feed.
    *   **White Grid Paper**: Clean white drafting grid paper.

### 3. Professional Creative Tools
*   **Neon Paint Brush**: Draws paths in two passes—a thick glowing shadow backdrop combined with a fine white core—to mimic glowing neon gas tubes.
*   **Transparent Pixel Eraser**: Erases persistent drawing pixels by switching the composite operation to `destination-out`.
*   **Curated Tech Palette**: Includes 15 curated colors alongside an integrated custom color globe.
*   **Robust Undo Stack**: Tracks 25 history states, allowing a clean single-click revert.

---

## 🛠️ How to Use

1.  **Open the app**: Load `index.html` on a secure context (`localhost` or `https://`).
2.  **Grant Camera Permission**: When prompted by your browser, allow camera access.
3.  **Position Hand**: Raise your hand in view of the webcam.
4.  **Drawing**:
    *   **Pinch** your thumb tip and index finger tip together to draw.
    *   **Release** the pinch to stop drawing.
5.  **Fullscreen Toggle**: Click the **Fullscreen** button in the camera box header to toggle between a floating corner camera stream and full-screen direct drafting overlay.
6.  **Backgrounds & Styles**: Switch canvas grid styles and brush thickness using the floating glassmorphic sidebar.

---

## 📐 Technical Architecture & Math

### Aspect-Ratio Video Crop Mapping
To map normalized webcam coordinates $(x_{norm}, y_{norm})$ to exact screen coordinates $(x_{screen}, y_{screen})$ under `object-fit: cover` (where the camera feed is scaled and cropped to fill the browser window):
$$\text{If } \text{ScreenRatio} > \text{VideoRatio}:$$
$$\text{Scale} = \frac{\text{ScreenWidth}}{\text{VideoWidth}}$$
$$y_{screen} = y_{norm} \times (\text{VideoHeight} \times \text{Scale}) - \frac{(\text{VideoHeight} \times \text{Scale}) - \text{ScreenHeight}}{2}$$

$$\text{If } \text{ScreenRatio} \le \text{VideoRatio}:$$
$$\text{Scale} = \frac{\text{ScreenHeight}}{\text{VideoHeight}}$$
$$x_{screen} = (1 - x_{norm}) \times (\text{VideoWidth} \times \text{Scale}) - \frac{(\text{VideoWidth} \times \text{Scale}) - \text{ScreenWidth}}{2}$$

### Dynamic Exponential Moving Average (EMA)
The smoothing factor $\alpha$ adjusts dynamically based on gesture velocity (pixel distance $d$ moved between frames):
$$\alpha = \max\left(\alpha_{min}, \min\left(\alpha_{max}, \alpha_{min} + \frac{d}{d_{threshold}} \times (\alpha_{max} - \alpha_{min})\right)\right)$$
$$\text{Smoothed}_{t} = \text{Smoothed}_{t-1} \times (1 - \alpha) + \text{Raw}_{t} \times \alpha$$

---

## 📂 File Structure

*   `index.html` - Premium glassmorphic structure with SVG icons and UI sections.
*   `style.css` - Custom styling tokens, grid patterns, and layout state animations.
*   `app.js` - Dynamic filters, bezier path geometry, and MediaPipe landmark listeners.

---

## 📄 License
Free to use and modify. Created for visual and gestural precision.
