// ============================================================================
// Neural Background - Sample Animation Modes
// ============================================================================
// These are alternative animation styles you can try. To use one, copy the
// relevant methods into neural-bg.js and call them from animate()/drawGrid()
// ============================================================================

// ============================================================================
// SAMPLE 1: TOKEN STREAM
// ============================================================================
// Particles flow left-to-right like tokens being processed, with occasional
// "attention" connections that flash between distant particles.
// 
// Add these properties to constructor:
//   this.tokens = [];
//   this.attentionLinks = [];
//   this.tokenSpawnTimer = 0;
// 
// Call updateTokens() from animate() and add token rendering to drawGrid()
// ============================================================================

class TokenStreamSample {
    constructor(parent) {
        this.parent = parent;
        this.tokens = [];
        this.attentionLinks = [];
        this.maxTokens = 30;
    }
    
    update() {
        const activity = this.parent.activity;
        
        // Spawn new tokens from left edge
        if (activity > 0.2 && Math.random() < 0.1 * activity) {
            if (this.tokens.length < this.maxTokens) {
                this.tokens.push({
                    x: -0.5,  // Start at left edge
                    y: (Math.random() - 0.5) * 0.8,  // Random vertical position
                    speed: 0.008 + Math.random() * 0.012,
                    size: 0.03 + Math.random() * 0.02,
                    brightness: 0.7 + Math.random() * 0.3,
                    phase: Math.random() * Math.PI * 2,
                });
            }
        }
        
        // Update token positions
        this.tokens = this.tokens.filter(token => {
            token.x += token.speed * (0.5 + activity * 0.5);
            token.phase += 0.1;
            
            // Remove when off right edge
            return token.x < 0.6;
        });
        
        // Create attention links between random tokens
        if (activity > 0.5 && this.tokens.length > 3 && Math.random() < 0.05) {
            const t1 = this.tokens[Math.floor(Math.random() * this.tokens.length)];
            const t2 = this.tokens[Math.floor(Math.random() * this.tokens.length)];
            if (t1 !== t2) {
                this.attentionLinks.push({
                    from: { x: t1.x, y: t1.y },
                    to: { x: t2.x, y: t2.y },
                    life: 1.0,
                    decay: 0.03,
                });
            }
        }
        
        // Update attention links
        this.attentionLinks = this.attentionLinks.filter(link => {
            link.life -= link.decay;
            return link.life > 0;
        });
    }
    
    // Add to drawGrid to render tokens as bright spots
    getTokenInfluence(nx, ny) {
        let influence = 0;
        for (const token of this.tokens) {
            const dx = nx - token.x;
            const dy = ny - token.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < token.size * 2) {
                const pulse = Math.sin(token.phase) * 0.2 + 0.8;
                influence += token.brightness * pulse * Math.max(0, 1 - dist / (token.size * 2));
            }
        }
        return Math.min(1, influence);
    }
}


// ============================================================================
// SAMPLE 2: ATTENTION WAVES
// ============================================================================
// Concentric rings pulse outward, but occasionally "snap" to form connections
// to specific points (simulating attention heads focusing on key tokens).
// ============================================================================

class AttentionWavesSample {
    constructor(parent) {
        this.parent = parent;
        this.waves = [];
        this.focusPoints = [];
        this.maxWaves = 5;
    }
    
    update() {
        const activity = this.parent.activity;
        
        // Spawn waves from center when active
        if (activity > 0.3 && Math.random() < 0.02 * activity) {
            if (this.waves.length < this.maxWaves) {
                this.waves.push({
                    radius: 0,
                    maxRadius: 0.8 + Math.random() * 0.4,
                    speed: 0.01 + Math.random() * 0.01,
                    thickness: 0.05 + Math.random() * 0.05,
                    brightness: 0.6 + Math.random() * 0.4,
                    life: 1.0,
                });
            }
        }
        
        // Update waves
        this.waves = this.waves.filter(wave => {
            wave.radius += wave.speed;
            wave.life = 1 - (wave.radius / wave.maxRadius);
            return wave.radius < wave.maxRadius;
        });
        
        // Create focus points (attention targets)
        if (activity > 0.5 && Math.random() < 0.03) {
            this.focusPoints.push({
                x: (Math.random() - 0.5) * 1.2,
                y: (Math.random() - 0.5) * 1.2,
                life: 1.0,
                decay: 0.02,
                pulsePhase: 0,
            });
        }
        
        // Update focus points
        this.focusPoints = this.focusPoints.filter(point => {
            point.life -= point.decay;
            point.pulsePhase += 0.2;
            return point.life > 0;
        });
    }
    
    getWaveInfluence(nx, ny) {
        const dist = Math.sqrt(nx * nx + ny * ny);
        let influence = 0;
        
        for (const wave of this.waves) {
            const ringDist = Math.abs(dist - wave.radius);
            if (ringDist < wave.thickness) {
                influence += wave.brightness * wave.life * (1 - ringDist / wave.thickness);
            }
        }
        
        // Focus point pull - waves bend toward focus points
        for (const point of this.focusPoints) {
            const dx = nx - point.x;
            const dy = ny - point.y;
            const pointDist = Math.sqrt(dx * dx + dy * dy);
            if (pointDist < 0.15) {
                const pulse = Math.sin(point.pulsePhase) * 0.3 + 0.7;
                influence += point.life * pulse * (1 - pointDist / 0.15);
            }
        }
        
        return Math.min(1, influence);
    }
}


// ============================================================================
// SAMPLE 3: LAYER PROPAGATION
// ============================================================================
// Horizontal bands light up in sequence from top to bottom, like data flowing
// through transformer layers. Each "layer" processes then passes to next.
// ============================================================================

class LayerPropagationSample {
    constructor(parent) {
        this.parent = parent;
        this.layers = 8;  // Number of "transformer layers"
        this.layerStates = new Array(this.layers).fill(0);
        this.propagationPhase = 0;
        this.processingSpeed = 0.05;
    }
    
    update() {
        const activity = this.parent.activity;
        
        if (activity > 0.3) {
            this.propagationPhase += this.processingSpeed * activity;
            
            // Each layer lights up in sequence
            for (let i = 0; i < this.layers; i++) {
                // Staggered wave through layers
                const layerPhase = this.propagationPhase - (i * 0.3);
                const wave = Math.sin(layerPhase);
                
                // Layer is "processing" when wave is positive
                this.layerStates[i] = Math.max(0, wave) * activity;
            }
        } else {
            // Fade out all layers when idle
            for (let i = 0; i < this.layers; i++) {
                this.layerStates[i] *= 0.95;
            }
        }
    }
    
    getLayerInfluence(ny) {
        // ny is normalized -1 to 1, map to layer index
        const layerHeight = 2 / this.layers;
        const layerIndex = Math.floor((ny + 1) / layerHeight);
        
        if (layerIndex >= 0 && layerIndex < this.layers) {
            return this.layerStates[layerIndex];
        }
        return 0;
    }
}


// ============================================================================
// SAMPLE 4: EMBEDDING CLUSTERS
// ============================================================================
// Groups of pixels that move together as clusters, occasionally merging or
// splitting (like concepts in embedding space). More semantic/organic feel.
// ============================================================================

class EmbeddingClustersSample {
    constructor(parent) {
        this.parent = parent;
        this.clusters = [];
        this.maxClusters = 6;
    }
    
    update() {
        const activity = this.parent.activity;
        const time = this.parent.time;
        
        // Spawn clusters
        if (this.clusters.length < this.maxClusters && Math.random() < 0.01) {
            this.clusters.push({
                x: (Math.random() - 0.5) * 0.8,
                y: (Math.random() - 0.5) * 0.8,
                vx: (Math.random() - 0.5) * 0.002,
                vy: (Math.random() - 0.5) * 0.002,
                radius: 0.1 + Math.random() * 0.15,
                brightness: 0.5 + Math.random() * 0.5,
                life: 1.0,
                decay: 0.001 + Math.random() * 0.002,
                phase: Math.random() * Math.PI * 2,
            });
        }
        
        // Update clusters
        this.clusters = this.clusters.filter(cluster => {
            // Drift movement
            cluster.x += cluster.vx * (0.5 + activity);
            cluster.y += cluster.vy * (0.5 + activity);
            
            // Breathing
            cluster.phase += 0.02;
            
            // Boundary bounce
            if (Math.abs(cluster.x) > 0.6) cluster.vx *= -1;
            if (Math.abs(cluster.y) > 0.6) cluster.vy *= -1;
            
            // Slow decay
            cluster.life -= cluster.decay;
            
            return cluster.life > 0;
        });
        
        // Merge nearby clusters
        for (let i = 0; i < this.clusters.length; i++) {
            for (let j = i + 1; j < this.clusters.length; j++) {
                const c1 = this.clusters[i];
                const c2 = this.clusters[j];
                const dx = c1.x - c2.x;
                const dy = c1.y - c2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                
                if (dist < (c1.radius + c2.radius) * 0.5) {
                    // Merge into c1
                    c1.x = (c1.x + c2.x) / 2;
                    c1.y = (c1.y + c2.y) / 2;
                    c1.radius = Math.min(0.3, c1.radius + c2.radius * 0.3);
                    c1.brightness = Math.min(1, c1.brightness + 0.2);
                    c2.life = 0;  // Remove c2
                }
            }
        }
    }
    
    getClusterInfluence(nx, ny) {
        let influence = 0;
        
        for (const cluster of this.clusters) {
            const dx = nx - cluster.x;
            const dy = ny - cluster.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < cluster.radius) {
                const breathing = Math.sin(cluster.phase) * 0.15 + 0.85;
                const falloff = 1 - (dist / cluster.radius);
                influence += cluster.brightness * cluster.life * falloff * breathing;
            }
        }
        
        return Math.min(1, influence);
    }
}


// ============================================================================
// SAMPLE 5: CONTEXT WINDOW (Ring Buffer)
// ============================================================================
// Visualizes the context window as a circular buffer - new "tokens" appear
// at one position and old ones fade out as they're pushed out.
// ============================================================================

class ContextWindowSample {
    constructor(parent) {
        this.parent = parent;
        this.contextSize = 32;  // Number of "token slots"
        this.tokens = new Array(this.contextSize).fill(0);
        this.writeHead = 0;
        this.writeTimer = 0;
    }
    
    update() {
        const activity = this.parent.activity;
        
        // Decay all tokens slowly
        for (let i = 0; i < this.contextSize; i++) {
            this.tokens[i] *= 0.995;
        }
        
        // Add new tokens when active
        if (activity > 0.3) {
            this.writeTimer += activity * 0.1;
            
            if (this.writeTimer > 1) {
                this.writeTimer = 0;
                this.tokens[this.writeHead] = 0.8 + Math.random() * 0.2;
                this.writeHead = (this.writeHead + 1) % this.contextSize;
            }
        }
    }
    
    getContextInfluence(angle, dist) {
        // Map angle to token slot
        const normalizedAngle = (angle + Math.PI) / (2 * Math.PI);
        const slotIndex = Math.floor(normalizedAngle * this.contextSize) % this.contextSize;
        
        // Only show in a ring
        if (dist > 0.4 && dist < 0.8) {
            const ringFade = 1 - Math.abs(dist - 0.6) / 0.2;
            return this.tokens[slotIndex] * ringFade;
        }
        return 0;
    }
}


// ============================================================================
// SAMPLE 6: NEURAL PATHWAYS
// ============================================================================
// Simulates signal propagation through neural pathways - bright pulses travel
// along curved paths between nodes, like synapses firing.
// ============================================================================

class NeuralPathwaysSample {
    constructor(parent) {
        this.parent = parent;
        this.nodes = [];
        this.signals = [];
        this.maxNodes = 12;
        this.maxSignals = 20;
        
        // Initialize some fixed nodes
        this.initNodes();
    }
    
    initNodes() {
        for (let i = 0; i < this.maxNodes; i++) {
            const angle = (i / this.maxNodes) * Math.PI * 2;
            const radius = 0.3 + Math.random() * 0.3;
            this.nodes.push({
                x: Math.cos(angle) * radius,
                y: Math.sin(angle) * radius,
                connections: [],
                activity: 0,
            });
        }
        
        // Create random connections
        for (let i = 0; i < this.nodes.length; i++) {
            const numConnections = 2 + Math.floor(Math.random() * 2);
            for (let j = 0; j < numConnections; j++) {
                const target = Math.floor(Math.random() * this.nodes.length);
                if (target !== i) {
                    this.nodes[i].connections.push(target);
                }
            }
        }
    }
    
    update() {
        const activity = this.parent.activity;
        
        // Decay node activity
        for (const node of this.nodes) {
            node.activity *= 0.95;
        }
        
        // Spawn new signals
        if (activity > 0.3 && this.signals.length < this.maxSignals && Math.random() < 0.1 * activity) {
            const sourceIdx = Math.floor(Math.random() * this.nodes.length);
            const source = this.nodes[sourceIdx];
            
            if (source.connections.length > 0) {
                const targetIdx = source.connections[Math.floor(Math.random() * source.connections.length)];
                const target = this.nodes[targetIdx];
                
                this.signals.push({
                    fromX: source.x,
                    fromY: source.y,
                    toX: target.x,
                    toY: target.y,
                    progress: 0,
                    speed: 0.02 + Math.random() * 0.02,
                    brightness: 0.8 + Math.random() * 0.2,
                    targetNode: targetIdx,
                });
                
                source.activity = 1;
            }
        }
        
        // Update signals
        this.signals = this.signals.filter(signal => {
            signal.progress += signal.speed;
            
            // Activate target node when signal arrives
            if (signal.progress >= 1) {
                this.nodes[signal.targetNode].activity = 1;
                return false;
            }
            return true;
        });
    }
    
    getPathwayInfluence(nx, ny) {
        let influence = 0;
        
        // Node glow
        for (const node of this.nodes) {
            const dx = nx - node.x;
            const dy = ny - node.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < 0.08) {
                influence += (0.3 + node.activity * 0.7) * (1 - dist / 0.08);
            }
        }
        
        // Signal pulses along paths
        for (const signal of this.signals) {
            // Current position of signal
            const sx = signal.fromX + (signal.toX - signal.fromX) * signal.progress;
            const sy = signal.fromY + (signal.toY - signal.fromY) * signal.progress;
            
            const dx = nx - sx;
            const dy = ny - sy;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < 0.06) {
                influence += signal.brightness * (1 - dist / 0.06);
            }
        }
        
        return Math.min(1, influence);
    }
}


// ============================================================================
// USAGE EXAMPLE
// ============================================================================
// To integrate one of these into neural-bg.js:
// 
// 1. Add to constructor:
//    this.tokenStream = new TokenStreamSample(this);
// 
// 2. Add to animate():
//    this.tokenStream.update();
// 
// 3. Add to drawGrid() intensity calculation:
//    const tokenInfluence = this.tokenStream.getTokenInfluence(nx, ny);
//    finalIntensity += tokenInfluence * this.activity;
// ============================================================================

console.log("Neural background samples loaded. These are reference implementations.");
