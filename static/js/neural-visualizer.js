/**
 * @file neural-visualizer.js
 * @description Neural Heatmap Visualizer - A 3D pixelated grid visualization
 * that acts like an "LLM brain" providing visual feedback for AI activity states.
 * 
 * @overview
 * Creates an animated canvas-based blob with 3D depth effects that responds to LLM activity:
 * - Idle (blue): Gently flowing organic blob with occasional red sparks
 * - Memory (yellow): Loading/retrieving memories
 * - Thinking (red): Active generation with neuron firing effects
 * - Tool (purple): Executing external tools
 * 
 * @usage
 * The visualizer auto-initializes and attaches to `window.neuralViz`.
 * 
 * Control methods:
 * ```js
 * window.neuralViz.activate()            // Start thinking animation
 * window.neuralViz.deactivate()          // Fade back to idle (with delay)
 * window.neuralViz.setStatus('memory')   // Yellow: loading memories
 * window.neuralViz.setStatus('thinking') // Red: LLM generating
 * window.neuralViz.setStatus('tool')     // Purple: executing tool
 * window.neuralViz.setStatus('idle')     // Blue: back to idle
 * ```
 * 
 * Runtime tweaks (via console):
 * ```js
 * window.neuralViz.activity = 0.5              // Set activity level (0-1)
 * window.neuralViz.config.pulseSpeed = 0.03    // Adjust animation speed
 * window.neuralViz.config.sparkChance = 0.05   // More frequent idle sparks
 * window.neuralViz.config.depthIntensity = 0.5 // Adjust 3D depth effect
 * ```
 * 
 * @requires DOM elements: `.sidebar` and `.sidebar-footer` for positioning
 * @author Plasma LLM Assistant
 * @version 2.0.0
 */

class NeuralVisualizer {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.animationId = null;
        
        // Activity level: 0 = idle, 1 = fully active (thinking)
        this.activity = 0;
        this.targetActivity = 0;
        this.deactivateTimeout = null;
        
        // Status mode: 'idle', 'memory', 'thinking', 'tool', 'approval'
        this.status = 'idle';
        this.targetStatus = 'idle';
        
        // Approval flash state
        this.approvalFlashPhase = 0;
        
        // Time counter for animations
        this.time = 0;
        
        // ====================================================================
        // GRID SETTINGS - Controls the pixel grid resolution
        // ====================================================================
        this.gridCols = 45;      // Number of pixel columns (more = finer detail)
        this.gridRows = 45;      // Number of pixel rows (more = finer detail)
        this.pixels = [];        // Stores pixel data
        
        // ====================================================================
        // BLOB POSITION - Where the blob is centered
        // ====================================================================
        this.centerX = 0;        // Current center X (includes wander)
        this.centerY = 0;        // Current center Y (includes wander)
        this.baseCenterX = 0;    // Base center X (from sidebar position)
        this.baseCenterY = 0;    // Base center Y (from sidebar position)
        this.gridSize = 400;     // Total size of the grid in screen pixels
        
        // Wander animation phases (for slow drifting movement)
        this.wanderPhaseX = Math.random() * Math.PI * 2;
        this.wanderPhaseY = Math.random() * Math.PI * 2;
        
        // ====================================================================
        // NEURON FIRING - Random bursts when active (thinking)
        // ====================================================================
        this.firingNeurons = [];     // Active neuron fire points
        this.maxFiringNeurons = 10;   // Max simultaneous firing neurons
        
        // ====================================================================
        // IDLE SPARKS - Occasional red flashes when idle
        // ====================================================================
        this.idleSparks = [];        // Active spark points
        this.nextSparkTime = 0;
        
        // ====================================================================
        // CONFIGURATION - Tweak these values to adjust the visualization!
        // ====================================================================
        this.config = {
            // --- COLORS ---
            idleColor: { r: 45, g: 85, b: 145 },        // Blue when idle (RGB 0-255)
            activeColor: { r: 240, g: 80, b: 70 },      // Red when thinking (RGB 0-255)
            memoryColor: { r: 230, g: 180, b: 40 },     // Yellow when loading memory
            toolColor: { r: 160, g: 90, b: 220 },       // Purple when executing tools
            clusteringColor: { r: 0, g: 200, b: 180 },  // Teal when Torque Clustering runs
            introspectionColor: { r: 80, g: 60, b: 220 }, // Indigo for 3 AM introspection cycle
            decisionGateColor: { r: 245, g: 139, b: 40 }, // Orange when decision gate evaluates
            researchColor: { r: 50, g: 200, b: 120 },   // Green for hourly research/self-improve cycle
            approvalColor: { r: 255, g: 40, b: 40 },    // Bright red for approval flash
            approvalDimColor: { r: 80, g: 20, b: 20 },  // Dim red for approval flash
            sparkColor: { r: 255, g: 80, b: 80 },       // Color of idle sparks (RGB 0-255)
            
            // --- 3D DEPTH SETTINGS ---
            depthIntensity: 0.6,     // Overall 3D depth effect strength (0-1)
            lightAngle: -0.7,        // Light source angle in radians (top-left)
            lightElevation: 0.6,     // Light elevation (0 = horizon, 1 = overhead)
            ambientLight: 0.3,       // Minimum lighting (shadow darkness)
            specularIntensity: 0.4,  // Highlight brightness on peaks
            shadowOffset: 2,         // Pixel offset for drop shadow
            shadowOpacity: 0.3,      // Drop shadow opacity
            heightScale: 1.2,        // How much height affects pixel size
            perspectiveStrength: 0.15, // Perspective distortion amount
            
            // --- ANIMATION SPEEDS ---
            pulseSpeed: 0.02,        // Base pulse animation speed
            flowSpeed: 0.015,        // How fast the internal flow moves
            idleFlowSpeed: 0.012,    // Flow speed specifically for idle state
            
            // --- ACTIVITY TRANSITIONS ---
            activityDecay: 0.007,    // How fast it calms down (lower = slower fade)
            activityRise: 0.05,      // How fast it activates (higher = faster)
            sleepDelay: 2500,        // Ms to wait before starting to calm down
            
            // --- BLOB SIZE ---
            blobScale: 0.7,          // Size relative to sidebar (0.5 = 50%)
            
            // --- IDLE SPARKS ---
            sparkChance: 0.03,       // Chance per frame to spawn spark (0-1)
            
            // --- NEURON FIRING (active state) ---
            neuronFireRate: 0.15,     // Chance per frame to spawn neuron (0-1)
        };
        
        this.init();
    }
    
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    init() {
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'neural-visualizer';
        this.ctx = this.canvas.getContext('2d');
        
        document.body.insertBefore(this.canvas, document.body.firstChild);
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
        
        this.createGrid();
        this.animate();
    }
    
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.updateCenter();
    }
    
    // ========================================================================
    // POSITIONING - Where the blob appears on screen
    // ========================================================================
    updateCenter() {
        const sidebarFooter = document.querySelector('.sidebar-footer');
        const sidebar = document.querySelector('.sidebar');
        
        if (sidebarFooter && sidebar) {
            const sidebarRect = sidebar.getBoundingClientRect();
            const footerRect = sidebarFooter.getBoundingClientRect();
            
            this.baseCenterX = sidebarRect.left + sidebarRect.width / 2;
            this.gridSize = sidebarRect.width * 0.85;
            this.baseCenterY = footerRect.top - this.gridSize / 2 - 10;
        } else {
            this.baseCenterX = 140;
            this.baseCenterY = this.canvas.height - 150;
            this.gridSize = 200;
        }
        
        this.pixelSize = this.gridSize / this.gridCols;
    }
    
    // ========================================================================
    // WANDER - Slow drifting movement around center
    // ========================================================================
    updateWander() {
        this.wanderPhaseX += 0.0008;
        this.wanderPhaseY += 0.0006;
        
        const wanderAmount = this.gridSize * 0.05;
        
        const flowX = Math.sin(this.wanderPhaseX) * wanderAmount;
        const flowY = Math.sin(this.wanderPhaseY * 1.3) * wanderAmount * 0.8;
        
        const driftX = Math.sin(this.wanderPhaseX * 0.3) * wanderAmount * 0.4;
        const driftY = Math.cos(this.wanderPhaseY * 0.4) * wanderAmount * 0.3;
        
        this.centerX = this.baseCenterX + flowX + driftX;
        this.centerY = this.baseCenterY + flowY + driftY;
    }
    
    // ========================================================================
    // GRID CREATION - Sets up the pixel grid data
    // ========================================================================
    createGrid() {
        this.pixels = [];
        for (let row = 0; row < this.gridRows; row++) {
            this.pixels[row] = [];
            for (let col = 0; col < this.gridCols; col++) {
                const nx = (col - this.gridCols / 2) / (this.gridCols / 2);
                const ny = (row - this.gridRows / 2) / (this.gridRows / 2);
                
                const distFromCenter = Math.sqrt(nx * nx + ny * ny);
                const angle = Math.atan2(ny, nx);
                
                this.pixels[row][col] = {
                    distFromCenter,
                    angle,
                    noiseOffset: Math.random() * 1000,
                    intensity: 0,
                    height: 0,  // 3D height value for this pixel
                };
            }
        }
    }
    
    // ========================================================================
    // ACTIVATION CONTROL - Called by app.js when LLM starts/stops
    // ========================================================================
    activate() {
        if (this.deactivateTimeout) {
            clearTimeout(this.deactivateTimeout);
            this.deactivateTimeout = null;
        }
        this.targetActivity = 1;
        this.updateCenter();
    }
    
    deactivate() {
        if (this.deactivateTimeout) {
            clearTimeout(this.deactivateTimeout);
        }
        this.deactivateTimeout = setTimeout(() => {
            this.targetActivity = 0;
            this.targetStatus = 'idle';
            this.deactivateTimeout = null;
        }, this.config.sleepDelay);
    }
    
    // ========================================================================
    // STATUS CONTROL - Set different status modes with different colors
    // ========================================================================
    setStatus(status) {
        if (this.deactivateTimeout) {
            clearTimeout(this.deactivateTimeout);
            this.deactivateTimeout = null;
        }
        this.targetStatus = status;
        if (status !== 'idle') {
            this.targetActivity = 1;
        }
        this.updateCenter();
    }
    
    getStatusColor() {
        switch (this.status) {
            case 'memory':
                return this.config.memoryColor;
            case 'thinking':
                return this.config.activeColor;
            case 'tool':
                return this.config.toolColor;
            case 'clustering':
                // Slow pulse between teal shades to suggest reorganization
                const clusterPulse = (Math.sin(this.time * 1.5) + 1) / 2;
                return this.lerpColor(
                    { r: 0, g: 140, b: 130 },
                    this.config.clusteringColor,
                    clusterPulse
                );
            case 'introspection':
                // Slow deep pulse — the system is thinking for itself
                const introPulse = (Math.sin(this.time * 0.8) + 1) / 2;
                return this.lerpColor(
                    { r: 50, g: 40, b: 160 },
                    this.config.introspectionColor,
                    introPulse
                );
            case 'research':
                // Gentle green pulse — researching and learning
                const researchPulse = (Math.sin(this.time * 1.2) + 1) / 2;
                return this.lerpColor(
                    { r: 20, g: 130, b: 70 },
                    this.config.researchColor,
                    researchPulse
                );
            case 'decision_gate':
                // Quick amber pulse — the system is deciding what to do
                const gatePulse = (Math.sin(this.time * 3.0) + 1) / 2;
                return this.lerpColor(
                    { r: 200, g: 100, b: 20 },
                    this.config.decisionGateColor,
                    gatePulse
                );
            case 'approval':
                const flash = (Math.sin(this.approvalFlashPhase) + 1) / 2;
                return this.lerpColor(
                    this.config.approvalDimColor,
                    this.config.approvalColor,
                    flash
                );
            default:
                return this.config.idleColor;
        }
    }
    
    // ========================================================================
    // NOISE FUNCTION - Creates organic, flowing patterns
    // ========================================================================
    noise(x, y, t) {
        const n1 = Math.sin(x * 2.5 + t) * Math.cos(y * 2.5 + t * 0.7);
        const n2 = Math.sin(x * 1.3 - t * 0.5) * Math.sin(y * 1.8 + t * 0.3);
        const n3 = Math.cos(x * 3.1 + y * 2.1 + t * 0.4);
        return (n1 + n2 + n3) / 3;
    }
    
    // ========================================================================
    // COLOR INTERPOLATION - Blends between two colors
    // ========================================================================
    lerpColor(c1, c2, t) {
        t = Math.max(0, Math.min(1, t));
        return {
            r: Math.round(c1.r + (c2.r - c1.r) * t),
            g: Math.round(c1.g + (c2.g - c1.g) * t),
            b: Math.round(c1.b + (c2.b - c1.b) * t),
        };
    }
    
    // ========================================================================
    // 3D LIGHTING CALCULATION
    // ========================================================================
    calculateLighting(height, nx, ny, neighborHeights) {
        const { lightAngle, lightElevation, ambientLight, specularIntensity, depthIntensity } = this.config;
        
        // Calculate surface normal from height differences (gradient)
        const dhdx = (neighborHeights.right - neighborHeights.left) / 2;
        const dhdy = (neighborHeights.bottom - neighborHeights.top) / 2;
        
        // Normal vector (pointing up from surface)
        const normalX = -dhdx * depthIntensity;
        const normalY = -dhdy * depthIntensity;
        const normalZ = 1;
        const normalLen = Math.sqrt(normalX * normalX + normalY * normalY + normalZ * normalZ);
        
        // Light direction vector
        const lightX = Math.cos(lightAngle) * (1 - lightElevation);
        const lightY = Math.sin(lightAngle) * (1 - lightElevation);
        const lightZ = lightElevation;
        
        // Diffuse lighting (dot product of normal and light direction)
        const diffuse = Math.max(0, 
            (normalX * lightX + normalY * lightY + normalZ * lightZ) / normalLen
        );
        
        // Specular highlight (for shiny peaks)
        const viewZ = 1;
        const halfX = lightX;
        const halfY = lightY;
        const halfZ = (lightZ + viewZ) / 2;
        const halfLen = Math.sqrt(halfX * halfX + halfY * halfY + halfZ * halfZ);
        
        const specDot = Math.max(0,
            (normalX * halfX + normalY * halfY + normalZ * halfZ) / (normalLen * halfLen)
        );
        const specular = Math.pow(specDot, 16) * specularIntensity * height;
        
        // Combine lighting components
        const lighting = ambientLight + diffuse * (1 - ambientLight) + specular;
        
        return Math.min(1.3, Math.max(0, lighting));
    }
    
    // ========================================================================
    // HEIGHT MAP GENERATION - Creates 3D terrain from intensity
    // ========================================================================
    calculateHeight(pixel, baseIntensity, neuronInfluence, sparkInfluence) {
        // Base height from intensity (brighter = higher)
        let height = baseIntensity * 0.8;
        
        // Add dome shape (center is higher)
        const domeHeight = Math.max(0, 1 - pixel.distFromCenter * 1.2) * 0.4;
        height += domeHeight;
        
        // Neuron bursts create peaks
        height += neuronInfluence * 0.6;
        
        // Sparks create small bumps
        height += sparkInfluence.total * 0.3;
        
        // Add animated ripple for organic feel
        const ripple = Math.sin(pixel.distFromCenter * 8 - this.time * 2) * 0.1;
        height += ripple * (1 - pixel.distFromCenter) * this.activity;
        
        // Noise-based surface variation
        const surfaceNoise = this.noise(
            pixel.angle * 2,
            pixel.distFromCenter * 3,
            this.time * 0.5
        ) * 0.15;
        height += surfaceNoise;
        
        return Math.max(0, Math.min(1, height));
    }
    
    // ========================================================================
    // IDLE SPARKS - Random mini-bursts when idle
    // ========================================================================
    updateIdleSparks() {
        if (this.activity < 0.3 && Math.random() < this.config.sparkChance) {
            const nx = (Math.random() - 0.5) * 1.4;
            const ny = (Math.random() - 0.5) * 1.4;
            const dist = Math.sqrt(nx * nx + ny * ny);
            
            if (dist < 0.7) {
                this.idleSparks.push({
                    nx, ny,
                    intensity: 0.4 + Math.random() * 0.3,
                    radius: 0.08 + Math.random() * 0.1,
                    life: 1.0,
                    decay: 0.025 + Math.random() * 0.02,
                    phase: 0,
                });
            }
        }
        
        this.idleSparks = this.idleSparks.filter(spark => {
            spark.life -= spark.decay;
            spark.phase += 0.15;
            return spark.life > 0;
        });
    }
    
    getSparkInfluence(pixel) {
        let influence = 0;
        let maxInfluence = 0;
        
        for (const spark of this.idleSparks) {
            const nx = (pixel.col - this.gridCols / 2) / (this.gridCols / 2);
            const ny = (pixel.row - this.gridRows / 2) / (this.gridRows / 2);
            const dx = nx - spark.nx;
            const dy = ny - spark.ny;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < spark.radius) {
                const ripple = Math.sin(dist * 20 - spark.phase * 3) * 0.5 + 0.5;
                const falloff = 1 - (dist / spark.radius);
                const sparkEffect = spark.intensity * spark.life * falloff * ripple;
                influence += sparkEffect;
                maxInfluence = Math.max(maxInfluence, sparkEffect);
            }
        }
        return { total: Math.min(1, influence), max: maxInfluence };
    }
    
    // ========================================================================
    // NEURON FIRING - Random bursts when active (thinking)
    // ========================================================================
    updateFiringNeurons() {
        if (this.activity > 0.3 && Math.random() < this.config.neuronFireRate * this.activity) {
            if (this.firingNeurons.length < this.maxFiringNeurons) {
                const nx = (Math.random() - 0.5) * 1.4;
                const ny = (Math.random() - 0.5) * 1.4;
                const dist = Math.sqrt(nx * nx + ny * ny);
                
                if (dist < 0.9) {
                    this.firingNeurons.push({
                        nx, ny,
                        intensity: 0.8 + Math.random() * 0.2,
                        radius: 0.2 + Math.random() * 0.25,
                        life: 1.0,
                        decay: 0.006 + Math.random() * 0.008,
                        phase: Math.random() * Math.PI * 2,
                    });
                }
            }
        }
        
        this.firingNeurons = this.firingNeurons.filter(neuron => {
            neuron.life -= neuron.decay;
            neuron.phase += 0.08;
            return neuron.life > 0;
        });
    }
    
    getNeuronInfluence(pixel) {
        let influence = 0;
        for (const neuron of this.firingNeurons) {
            const nx = (pixel.col - this.gridCols / 2) / (this.gridCols / 2);
            const ny = (pixel.row - this.gridRows / 2) / (this.gridRows / 2);
            const dx = nx - neuron.nx;
            const dy = ny - neuron.ny;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < neuron.radius) {
                const ripple = Math.sin(dist * 15 - neuron.phase * 2) * 0.5 + 0.5;
                const falloff = 1 - (dist / neuron.radius);
                influence += neuron.intensity * neuron.life * falloff * ripple;
            }
        }
        return Math.min(1, influence);
    }
    
    // ========================================================================
    // MAIN ANIMATION LOOP
    // ========================================================================
    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.time += this.config.flowSpeed + (this.activity * 0.02);
        
        this.updateWander();
        
        if (this.activity < this.targetActivity) {
            this.activity = Math.min(this.targetActivity, this.activity + this.config.activityRise);
        } else if (this.activity > this.targetActivity) {
            this.activity = Math.max(this.targetActivity, this.activity - this.config.activityDecay);
        }
        
        this.status = this.targetStatus;
        
        if (this.status === 'approval') {
            this.approvalFlashPhase += 0.15;
        }
        
        this.frameCount = (this.frameCount || 0) + 1;
        if (this.frameCount % 30 === 0) {
            this.updateCenter();
        }
        
        this.updateIdleSparks();
        this.updateFiringNeurons();
        
        // First pass: calculate heights for all pixels
        this.calculateAllHeights();
        
        // Second pass: draw with 3D effects
        this.drawGrid3D();
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    // ========================================================================
    // CALCULATE ALL HEIGHTS - First pass for 3D rendering
    // ========================================================================
    calculateAllHeights() {
        for (let row = 0; row < this.gridRows; row++) {
            for (let col = 0; col < this.gridCols; col++) {
                const pixel = this.pixels[row][col];
                pixel.row = row;
                pixel.col = col;
                
                // Calculate blob mask
                const blobDistortion = this.noise(
                    pixel.angle * 2,
                    pixel.distFromCenter,
                    this.time * 0.8
                ) * 0.25;
                
                const morphWave = Math.sin(pixel.angle * 3 + this.time * 0.4) * 0.1;
                const morphWave2 = Math.sin(pixel.angle * 2 - this.time * 0.3) * 0.08;
                
                const distortedDist = pixel.distFromCenter - blobDistortion - morphWave - morphWave2;
                const blobMask = Math.max(0, 1 - distortedDist);
                pixel.blobMask = blobMask * blobMask;
                
                if (pixel.blobMask < 0.01) {
                    pixel.height = 0;
                    pixel.finalIntensity = 0;
                    continue;
                }
                
                // Calculate intensities
                const flowNoise = this.noise(
                    pixel.distFromCenter * 2 + pixel.noiseOffset * 0.01,
                    pixel.angle,
                    this.time
                );
                
                const idleBreath = (Math.sin(this.time * 0.8 + pixel.noiseOffset) + 1) / 2;
                const idleWave = (Math.sin(this.time * 1.5 - pixel.distFromCenter * 3 + pixel.angle * 2) + 1) / 2;
                const idleDrift = (Math.sin(this.time * 0.6 + pixel.angle * 3) + 1) / 2;
                const centerBoost = Math.max(0, 1 - pixel.distFromCenter * 2) * 0.3;
                
                const idleGlow = (1 - pixel.distFromCenter * 0.5) * (0.4 + idleBreath * 0.2 + idleWave * 0.25 + idleDrift * 0.15) + centerBoost;
                
                const sparkInfluence = this.getSparkInfluence(pixel);
                const neuronInfluence = this.getNeuronInfluence(pixel);
                
                const activeBase = (1 - pixel.distFromCenter * 0.4) * 0.6;
                const activeGlow = activeBase + neuronInfluence * 0.9;
                
                let baseIntensity = idleGlow * (1 - this.activity) + activeGlow * this.activity;
                baseIntensity += sparkInfluence.total * (1 - this.activity);
                
                const blobDistort = 0.5 + flowNoise * 0.3;
                pixel.finalIntensity = baseIntensity * blobDistort * pixel.blobMask;
                
                // Store for color calculation
                pixel.sparkInfluence = sparkInfluence;
                pixel.neuronInfluence = neuronInfluence;
                
                // Calculate 3D height
                pixel.height = this.calculateHeight(pixel, baseIntensity, neuronInfluence, sparkInfluence);
            }
        }
    }
    
    // ========================================================================
    // DRAW GRID 3D - Main rendering function with depth effects
    // ========================================================================
    drawGrid3D() {
        const startX = this.centerX - this.gridSize / 2;
        const startY = this.centerY - this.gridSize / 2;
        const { depthIntensity, shadowOffset, shadowOpacity, heightScale, perspectiveStrength } = this.config;
        
        // Draw from back to front for proper layering (painter's algorithm)
        for (let row = 0; row < this.gridRows; row++) {
            for (let col = 0; col < this.gridCols; col++) {
                const pixel = this.pixels[row][col];
                
                if (pixel.finalIntensity < 0.02) continue;
                
                // Get neighbor heights for lighting calculation
                const neighborHeights = {
                    left: col > 0 ? this.pixels[row][col - 1].height : pixel.height,
                    right: col < this.gridCols - 1 ? this.pixels[row][col + 1].height : pixel.height,
                    top: row > 0 ? this.pixels[row - 1][col].height : pixel.height,
                    bottom: row < this.gridRows - 1 ? this.pixels[row + 1][col].height : pixel.height,
                };
                
                // Calculate lighting
                const nx = (col - this.gridCols / 2) / (this.gridCols / 2);
                const ny = (row - this.gridRows / 2) / (this.gridRows / 2);
                const lighting = this.calculateLighting(pixel.height, nx, ny, neighborHeights);
                
                // Perspective: pixels closer to bottom appear larger (subtle)
                const perspectiveScale = 1 + (row / this.gridRows) * perspectiveStrength;
                
                // Height affects pixel size (taller = slightly larger)
                const heightBoost = 1 + pixel.height * heightScale * 0.15 * depthIntensity;
                
                // Base position
                let x = startX + col * this.pixelSize;
                let y = startY + row * this.pixelSize;
                
                // Height offset (pixels rise up based on height)
                const heightOffset = pixel.height * this.pixelSize * 0.5 * depthIntensity;
                y -= heightOffset;
                
                // Calculate final pixel size
                const gap = 1;
                const baseSize = this.pixelSize - gap;
                const size = baseSize * perspectiveScale * heightBoost;
                
                // Center the larger pixel
                const sizeOffset = (size - baseSize) / 2;
                x -= sizeOffset;
                
                // Determine base color
                let color;
                if (pixel.sparkInfluence.max > 0.1 && this.activity < 0.3) {
                    color = this.lerpColor(
                        this.config.idleColor,
                        this.config.sparkColor,
                        pixel.sparkInfluence.max * 0.6
                    );
                } else {
                    const statusColor = this.getStatusColor();
                    color = this.lerpColor(
                        this.config.idleColor,
                        statusColor,
                        this.activity
                    );
                    if (pixel.neuronInfluence > 0.05) {
                        color = this.lerpColor(
                            color,
                            this.config.activeColor,
                            pixel.neuronInfluence * 0.5
                        );
                    }
                }
                
                // Apply lighting to color
                const litColor = {
                    r: Math.min(255, Math.round(color.r * lighting)),
                    g: Math.min(255, Math.round(color.g * lighting)),
                    b: Math.min(255, Math.round(color.b * lighting)),
                };
                
                // Draw drop shadow (offset and darker)
                if (depthIntensity > 0 && pixel.height > 0.1) {
                    const shadowX = x + shadowOffset * pixel.height;
                    const shadowY = y + shadowOffset * pixel.height + heightOffset;
                    this.ctx.fillStyle = `rgba(0, 0, 0, ${shadowOpacity * pixel.height * pixel.blobMask})`;
                    this.ctx.fillRect(shadowX, shadowY, size * 0.9, size * 0.9);
                }
                
                // Draw main pixel with lighting
                this.ctx.fillStyle = `rgba(${litColor.r}, ${litColor.g}, ${litColor.b}, ${pixel.finalIntensity * 0.85})`;
                this.ctx.fillRect(x, y, size, size);
                
                // Specular highlight on peaks
                if (pixel.height > 0.4 && lighting > 0.8) {
                    const highlightIntensity = (pixel.height - 0.4) * (lighting - 0.8) * 2;
                    this.ctx.fillStyle = `rgba(255, 255, 255, ${highlightIntensity * 0.4})`;
                    this.ctx.fillRect(x + size * 0.2, y + size * 0.2, size * 0.4, size * 0.4);
                }
                
                // Edge highlight (rim lighting effect)
                if (pixel.distFromCenter > 0.5 && pixel.height > 0.2) {
                    const rimIntensity = (pixel.distFromCenter - 0.5) * pixel.height * 0.5;
                    this.ctx.fillStyle = `rgba(255, 255, 255, ${rimIntensity * 0.2})`;
                    this.ctx.fillRect(x, y, size, 1);
                }
                
                // Active state core glow
                if (this.activity > 0.5 && pixel.finalIntensity > 0.4) {
                    const coreIntensity = (this.activity - 0.5) * 2 * pixel.finalIntensity * 0.5;
                    this.ctx.fillStyle = `rgba(255, 255, 255, ${coreIntensity * 0.3})`;
                    this.ctx.fillRect(x + size * 0.25, y + size * 0.25, size * 0.5, size * 0.5);
                }
            }
        }
    }
    
    // ========================================================================
    // CLEANUP
    // ========================================================================
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }
}

// ============================================================================
// Create global instance - accessible via window.neuralViz
// ============================================================================
// You can control it from browser console:
//   window.neuralViz.activate()            - Start "thinking" animation
//   window.neuralViz.deactivate()          - Stop (with delay then fade)
//   window.neuralViz.setStatus('memory')   - Yellow: loading memories
//   window.neuralViz.setStatus('thinking') - Red: LLM generating
//   window.neuralViz.setStatus('tool')     - Purple: executing tool
//   window.neuralViz.setStatus('idle')     - Blue: back to idle
//   window.neuralViz.activity = 0.5        - Set activity level directly
//   window.neuralViz.config.depthIntensity = 0.8  - Adjust 3D depth
//   window.neuralViz.config.xxx            - Tweak any config value live
// ============================================================================
window.neuralViz = new NeuralVisualizer();

// Backwards compatibility alias
window.neuralBg = window.neuralViz;
