/**
 * Memory Map — 3D star-map visualizer for Torque Clustering memory.
 *
 * Uses Three.js (r160, loaded via importmap) to render:
 *   - Individual memories as pulsing stars (coloured + sized by priority/cluster)
 *   - Cluster halos as translucent nebula spheres
 *   - Floating cluster-theme labels (HTML overlay, updated each frame)
 *   - OrbitControls for drag-rotate, scroll-zoom, right-drag pan
 *   - Raycaster for hover tooltips and click-to-inspect
 *
 * Exposes: window.memoryMap  (MemoryMapViz instance)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Colour helpers ─────────────────────────────────────────────────────────

/** Spread cluster hues evenly using the golden angle to maximise contrast. */
function clusterHue(index, _total) {
    const GOLDEN = 137.508;
    return ((index * GOLDEN) % 360) / 360;
}

// ── Main class ─────────────────────────────────────────────────────────────

class MemoryMapViz {
    constructor() {
        // Three.js core
        this.scene      = null;
        this.camera     = null;
        this.renderer   = null;
        this.controls   = null;
        this.container  = null;
        this.clock      = new THREE.Clock();

        // Scene objects
        this.starPoints    = null;   // THREE.Points — one point per memory
        this.sunPoints     = null;   // THREE.Points — one large point per cluster centroid
        this.orbitalLines  = null;   // THREE.LineSegments — memory → cluster connector lines
        this.orbitalParams = null;   // per-memory orbit data (parallel to memories array)
        this.nebulaMeshes  = [];     // THREE.Mesh[]  — cluster halos
        this.labelEls     = [];     // HTMLElement[] — cluster theme labels

        // Data
        this.data          = null;
        this.clusterColors = {};    // cluster_id → THREE.Color

        // Interaction
        this.raycaster        = new THREE.Raycaster();
        this.raycaster.params.Points = { threshold: 1.8 };
        this.pointer          = new THREE.Vector2(-999, -999);
        this.hoveredIdx       = -1;
        this.hoveredClusterId = null;   // cluster whose label is currently shown

        // DOM refs (set in mount())
        this.tooltip       = null;
        this.labelContainer = null;

        // State
        this.animId    = null;
        this.mounted   = false;
    }

    // ── Public API ───────────────────────────────────────────────────────────

    mount(container) {
        if (this.mounted) return;
        this.container    = container;
        this.tooltip      = document.getElementById('memory-map-tooltip');
        this.labelContainer = document.getElementById('memory-map-labels');

        this._initRenderer();
        this._initScene();
        this._addAmbientStars();
        this._startLoop();

        container.addEventListener('mousemove', e => this._onMouseMove(e));
        container.addEventListener('click',     e => this._onClick(e));
        container.addEventListener('mouseleave', () => this._hideTooltip());
        window.addEventListener('resize', () => this._onResize());

        this.mounted = true;
    }

    async loadAndRender() {
        const detail = document.getElementById('memory-map-detail');
        if (detail) detail.innerHTML = '<div class="mm-loading">Projecting memories…</div>';

        try {
            const res = await fetch('/api/memory/viz');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            this.data = await res.json();
        } catch (e) {
            if (detail) detail.innerHTML = `<div class="mm-error">Failed: ${e.message}</div>`;
            return;
        }

        this._buildViz();
    }

    destroy() {
        if (this.animId) cancelAnimationFrame(this.animId);
        this._clearMemoryObjects();
        if (this.renderer) {
            this.renderer.dispose();
            this.renderer.domElement?.remove();
        }
        window.removeEventListener('resize', () => this._onResize());
        this.mounted = false;
    }

    // ── Initialisation ───────────────────────────────────────────────────────

    _initRenderer() {
        const w = this.container.clientWidth  || 600;
        const h = this.container.clientHeight || 500;

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(w, h);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);

        this.camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 2000);
        this.camera.position.set(0, 30, 130);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping    = true;
        this.controls.dampingFactor    = 0.06;
        this.controls.autoRotate       = true;
        this.controls.autoRotateSpeed  = 0.35;
        this.controls.minDistance      = 5;
        this.controls.maxDistance      = 600;
        // Pause auto-rotate when user grabs the scene
        this.controls.addEventListener('start', () => {
            this.controls.autoRotate = false;
        });
    }

    _initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x060c18);
        this.scene.fog = new THREE.FogExp2(0x060c18, 0.005);
    }

    /** 3 000 tiny non-interactive background stars for a deep-space feel. */
    _addAmbientStars() {
        const n   = 3000;
        const pos = new Float32Array(n * 3);
        for (let i = 0; i < n * 3; i++) pos[i] = (Math.random() - 0.5) * 1200;

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        const mat = new THREE.PointsMaterial({
            size:        0.22,
            color:       0x9999cc,
            transparent: true,
            opacity:     0.32,
        });
        this.scene.add(new THREE.Points(geo, mat));
    }

    // ── Scene building ───────────────────────────────────────────────────────

    _buildViz() {
        this._clearMemoryObjects();

        const { memories, clusters } = this.data;

        if (!memories.length) {
            const detail = document.getElementById('memory-map-detail');
            if (detail) detail.innerHTML = '<div class="mm-empty">No memories yet — start chatting!</div>';
            return;
        }

        // Assign one hue per cluster
        clusters.forEach((c, i) => {
            this.clusterColors[c.id] = new THREE.Color().setHSL(
                clusterHue(i, clusters.length), 0.78, 0.60
            );
        });

        this._buildOrbitalLines(memories, clusters);  // drawn first (behind everything)
        this._buildNebulae(clusters);
        this._buildSuns(clusters);
        this._buildStars(memories);
        this._buildLabels(clusters);
        this._buildLegend(clusters);
        this._fitCamera(memories);

        const detail = document.getElementById('memory-map-detail');
        if (detail) detail.innerHTML =
            '<div class="mm-hint">Hover a star · click a cluster name or legend item to inspect</div>';
    }

    /** Build the Points mesh — one point per memory with per-vertex colour + size. */
    _buildStars(memories) {
        const n         = memories.length;
        const positions = new Float32Array(n * 3);
        const colors    = new Float32Array(n * 3);
        const sizes     = new Float32Array(n);
        const idxAttr   = new Float32Array(n);   // passed to shader for per-star phase offset

        memories.forEach((m, i) => {
            positions[i * 3]     = m.x;
            positions[i * 3 + 1] = m.y;
            positions[i * 3 + 2] = m.z;

            const col = (m.cluster_id && this.clusterColors[m.cluster_id])
                ? this.clusterColors[m.cluster_id]
                : new THREE.Color(0x667799);
            colors[i * 3]     = col.r;
            colors[i * 3 + 1] = col.g;
            colors[i * 3 + 2] = col.b;

            sizes[i]    = 1.4 + (m.priority || 3) * 0.72;
            idxAttr[i]  = i;
        });

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('color',    new THREE.BufferAttribute(colors,    3));
        geo.setAttribute('aSize',    new THREE.BufferAttribute(sizes,     1));
        geo.setAttribute('aIdx',     new THREE.BufferAttribute(idxAttr,   1));

        const mat = new THREE.ShaderMaterial({
            uniforms: {
                uTime:    { value: 0.0 },
                uTex:     { value: this._makeStarTexture() },
                uHovered: { value: -1.0 },
            },
            vertexShader: `
                attribute float aSize;
                attribute float aIdx;
                varying vec3  vColor;
                varying float vIdx;
                uniform float uTime;
                uniform float uHovered;

                void main() {
                    vColor = color;
                    vIdx   = aIdx;

                    // Individual twinkle via per-star phase offset
                    float pulse     = 1.0 + 0.13 * sin(uTime * 1.7 + aIdx * 0.83);
                    float isHovered = step(abs(aIdx - uHovered), 0.5);
                    float hoverSz   = mix(1.0, 2.8, isHovered);

                    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = aSize * pulse * hoverSz * (290.0 / -mvPos.z);
                    gl_Position  = projectionMatrix * mvPos;
                }
            `,
            fragmentShader: `
                uniform sampler2D uTex;
                uniform float     uHovered;
                varying vec3  vColor;
                varying float vIdx;

                void main() {
                    vec4 tex = texture2D(uTex, gl_PointCoord);
                    if (tex.a < 0.04) discard;

                    float isHovered = step(abs(vIdx - uHovered), 0.5);
                    vec3  col = mix(vColor, vec3(1.0), isHovered * 0.45);
                    gl_FragColor = vec4(col, tex.a);
                }
            `,
            transparent: true,
            depthWrite:  false,
            vertexColors: true,
        });

        this.starPoints = new THREE.Points(geo, mat);
        this.starPoints.userData.memories = memories;
        this.scene.add(this.starPoints);
    }

    /** Radial-gradient canvas texture for the star glow sprite. */
    _makeStarTexture() {
        const sz  = 128;
        const cvs = document.createElement('canvas');
        cvs.width = cvs.height = sz;
        const ctx = cvs.getContext('2d');
        const c   = sz / 2;
        const g   = ctx.createRadialGradient(c, c, 0, c, c, c);
        g.addColorStop(0,    'rgba(255,255,255,1.0)');
        g.addColorStop(0.12, 'rgba(255,255,255,0.95)');
        g.addColorStop(0.35, 'rgba(210,220,255,0.45)');
        g.addColorStop(0.65, 'rgba(160,180,255,0.10)');
        g.addColorStop(1,    'rgba(0,0,0,0)');
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, sz, sz);

        // Subtle 4-pointed flare cross
        ctx.strokeStyle = 'rgba(255,255,255,0.10)';
        ctx.lineWidth   = 0.8;
        ctx.beginPath();
        ctx.moveTo(c, 0); ctx.lineTo(c, sz);
        ctx.moveTo(0, c); ctx.lineTo(sz, c);
        ctx.stroke();

        return new THREE.CanvasTexture(cvs);
    }

    /** Translucent sphere meshes (inner core + outer wispy halo) per cluster. */
    _buildNebulae(clusters) {
        clusters.forEach(cluster => {
            if (!cluster.count) return;

            const col    = this.clusterColors[cluster.id] || new THREE.Color(0x334466);
            const radius = Math.max(4.5, Math.sqrt(cluster.mass || 1) * 3.8);
            const pos    = new THREE.Vector3(cluster.cx, cluster.cy, cluster.cz);

            // Inner dense halo
            const innerMesh = new THREE.Mesh(
                new THREE.SphereGeometry(radius, 20, 20),
                new THREE.MeshBasicMaterial({
                    color: col, transparent: true, opacity: 0.18,
                    side: THREE.DoubleSide, depthWrite: false,
                })
            );
            innerMesh.position.copy(pos);
            this.scene.add(innerMesh);
            this.nebulaMeshes.push({ mesh: innerMesh, outer: false });

            // Outer wispy halo (breathes in animation loop)
            const outerMesh = new THREE.Mesh(
                new THREE.SphereGeometry(radius * 1.7, 14, 14),
                new THREE.MeshBasicMaterial({
                    color: col, transparent: true, opacity: 0.06,
                    side: THREE.DoubleSide, depthWrite: false,
                })
            );
            outerMesh.position.copy(pos);
            this.scene.add(outerMesh);
            this.nebulaMeshes.push({ mesh: outerMesh, outer: true });
        });
    }

    /**
     * Builds orbital lines AND computes this.orbitalParams (parallel to memories array).
     * Each memory orbits its cluster centroid at its original UMAP radius; the orbital
     * plane is tilted uniquely per memory using a golden-angle seed for visual variety.
     */
    _buildOrbitalLines(memories, clusters) {
        const clusterPosMap = {};
        clusters.forEach(c => {
            if (c.count) clusterPosMap[c.id] = { cx: c.cx, cy: c.cy, cz: c.cz };
        });

        const posArr = [];
        const colArr = [];

        this.orbitalParams = memories.map((m, i) => {
            const cp = clusterPosMap[m.cluster_id];
            if (!cp) {
                // Unclassified memory — stays static
                return { static: true, x: m.x, y: m.y, z: m.z, lineVertexStart: -1 };
            }

            const dx = m.x - cp.cx;
            const dy = m.y - cp.cy;
            const dz = m.z - cp.cz;
            const radius = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1.5;

            // Deterministic tilted orbital-plane normal, unique per memory
            const sa = i * 2.39996;   // ≈ golden angle in radians
            const sb = i * 1.38361;
            let nx = Math.sin(sb) * Math.cos(sa);
            let ny = Math.cos(sb);
            let nz = Math.sin(sb) * Math.sin(sa);
            const nl = Math.sqrt(nx*nx + ny*ny + nz*nz) || 1;
            nx /= nl; ny /= nl; nz /= nl;

            // u = initial direction (centroid → memory), projected onto orbital plane
            let ux = dx / radius;
            let uy = dy / radius;
            let uz = dz / radius;
            const dot = ux*nx + uy*ny + uz*nz;
            ux -= dot*nx; uy -= dot*ny; uz -= dot*nz;
            const ul = Math.sqrt(ux*ux + uy*uy + uz*uz) || 1;
            ux /= ul; uy /= ul; uz /= ul;

            // v = n × u  (second axis, perpendicular in the orbital plane)
            const vx = ny*uz - nz*uy;
            const vy = nz*ux - nx*uz;
            const vz = nx*uy - ny*ux;

            // Kepler-ish: smaller orbits are faster
            const speed = 0.06 + 0.07 / (1 + radius * 0.06);

            const lineVertexStart = posArr.length / 3;
            posArr.push(m.x, m.y, m.z, cp.cx, cp.cy, cp.cz);
            const col = this.clusterColors[m.cluster_id] || new THREE.Color(0x334466);
            colArr.push(col.r, col.g, col.b, col.r, col.g, col.b);

            return {
                static: false,
                cx: cp.cx, cy: cp.cy, cz: cp.cz,
                radius, ux, uy, uz, vx, vy, vz,
                phase: 0,   // angle=0 → starts at original UMAP position
                speed,
                lineVertexStart,
            };
        });

        if (!posArr.length) return;

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(posArr), 3));
        geo.setAttribute('color',    new THREE.BufferAttribute(new Float32Array(colArr), 3));

        this.orbitalLines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
            transparent: true, opacity: 0.18, vertexColors: true, depthWrite: false,
        }));
        this.scene.add(this.orbitalLines);
    }

    /** Advance every memory along its elliptical orbit; update star + line buffers in-place. */
    _updateOrbits(t) {
        if (!this.starPoints || !this.orbitalParams) return;
        const starPos = this.starPoints.geometry.attributes.position;
        const linePos = this.orbitalLines?.geometry.attributes.position;

        this.orbitalParams.forEach((p, i) => {
            let x, y, z;
            if (p.static) {
                x = p.x; y = p.y; z = p.z;
            } else {
                const angle = t * p.speed + p.phase;
                const c = Math.cos(angle);
                const s = Math.sin(angle);
                x = p.cx + p.radius * (c * p.ux + s * p.vx);
                y = p.cy + p.radius * (c * p.uy + s * p.vy);
                z = p.cz + p.radius * (c * p.uz + s * p.vz);
            }
            starPos.setXYZ(i, x, y, z);
            // Move the memory-end of the orbital line; centroid-end stays fixed
            if (linePos && p.lineVertexStart >= 0) {
                linePos.setXYZ(p.lineVertexStart, x, y, z);
            }
        });

        starPos.needsUpdate = true;
        if (linePos) linePos.needsUpdate = true;
    }

    /** Large glowing points at each cluster centroid — the "suns". Size ∝ member count. */
    _buildSuns(clusters) {
        const valid = clusters.filter(c => c.count > 0);
        if (!valid.length) return;

        const n         = valid.length;
        const positions = new Float32Array(n * 3);
        const colors    = new Float32Array(n * 3);
        const sizes     = new Float32Array(n);
        const idxAttr   = new Float32Array(n);

        valid.forEach((cluster, i) => {
            positions[i * 3]     = cluster.cx;
            positions[i * 3 + 1] = cluster.cy;
            positions[i * 3 + 2] = cluster.cz;

            // Brighten the cluster colour for the sun
            const base = (this.clusterColors[cluster.id] || new THREE.Color(0x9999cc)).clone();
            colors[i * 3]     = Math.min(1, base.r + 0.30);
            colors[i * 3 + 1] = Math.min(1, base.g + 0.30);
            colors[i * 3 + 2] = Math.min(1, base.b + 0.30);

            // 20 members = max size; scale linearly from 6 → 18
            sizes[i]   = 6 + Math.min(cluster.count / 20, 1.0) * 12;
            idxAttr[i] = i;
        });

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('color',    new THREE.BufferAttribute(colors,    3));
        geo.setAttribute('aSize',    new THREE.BufferAttribute(sizes,     1));
        geo.setAttribute('aIdx',     new THREE.BufferAttribute(idxAttr,   1));

        this.sunPoints = new THREE.Points(geo, new THREE.ShaderMaterial({
            uniforms: {
                uTime: { value: 0.0 },
                uTex:  { value: this._makeStarTexture() },
            },
            vertexShader: `
                attribute float aSize;
                attribute float aIdx;
                varying vec3 vColor;
                uniform float uTime;
                void main() {
                    vColor = color;
                    // Slow, gentle sun-pulse
                    float pulse = 1.0 + 0.07 * sin(uTime * 0.55 + aIdx * 1.7);
                    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
                    gl_PointSize = aSize * pulse * (290.0 / -mvPos.z);
                    gl_Position  = projectionMatrix * mvPos;
                }
            `,
            fragmentShader: `
                uniform sampler2D uTex;
                varying vec3 vColor;
                void main() {
                    vec4 tex = texture2D(uTex, gl_PointCoord);
                    if (tex.a < 0.04) discard;
                    gl_FragColor = vec4(vColor, tex.a);
                }
            `,
            transparent: true, depthWrite: false, vertexColors: true,
        }));
        this.scene.add(this.sunPoints);
    }

    /** Create one HTML div per cluster, positioned via 3D→screen projection each frame. */
    _buildLabels(clusters) {
        this.labelEls.forEach(el => el.remove());
        this.labelEls = [];
        if (!this.labelContainer) return;

        clusters.forEach(cluster => {
            if (!cluster.count) return;
            const col = this.clusterColors[cluster.id];
            const el  = document.createElement('div');
            el.className = 'mm-cluster-label';
            if (col) {
                const hex = col.getHexString();
                el.style.color      = `#${hex}`;
                el.style.textShadow =
                    `0 0 8px rgba(${Math.round(col.r*255)},${Math.round(col.g*255)},${Math.round(col.b*255)},0.7)`;
            }
            // Truncate long themes to keep labels compact
            const theme = cluster.theme || 'Unknown';
            el.textContent        = theme.length > 26 ? theme.slice(0, 26) + '…' : theme;
            el.title              = theme;   // full text on native tooltip
            el.dataset.x          = cluster.cx;
            el.dataset.y          = cluster.cy;
            el.dataset.z          = cluster.cz;
            el.dataset.clusterId  = cluster.id;
            // Labels start hidden — revealed by proximity in _onMouseMove
            el.style.opacity      = '0';
            this.labelContainer.appendChild(el);
            this.labelEls.push(el);
        });
    }

    _buildLegend(clusters) {
        const legend = document.getElementById('memory-map-legend');
        if (!legend) return;
        legend.innerHTML = '<div class="mm-legend-title">Clusters</div>';

        [...clusters]
            .sort((a, b) => (b.mass || 0) - (a.mass || 0))
            .forEach(cluster => {
                const col = this.clusterColors[cluster.id];
                if (!col) return;
                const item = document.createElement('div');
                item.className = 'mm-legend-item';
                item.innerHTML = `
                    <span class="mm-legend-dot" style="background:#${col.getHexString()}"></span>
                    <span class="mm-legend-text">${this._esc(cluster.theme || 'Unknown')}</span>
                    <span class="mm-legend-count">${cluster.count}</span>
                `;
                item.addEventListener('click', () => this._focusCluster(cluster));
                legend.appendChild(item);
            });
    }

    _fitCamera(memories) {
        const box = new THREE.Box3();
        memories.forEach(m => box.expandByPoint(new THREE.Vector3(m.x, m.y, m.z)));
        const size   = box.getSize(new THREE.Vector3()).length();
        const center = box.getCenter(new THREE.Vector3());
        this.controls.target.copy(center);
        this.camera.position.set(
            center.x,
            center.y + size * 0.22,
            center.z + size * 0.88
        );
        this.controls.update();
    }

    // ── Cluster detail panel ─────────────────────────────────────────────────

    _focusCluster(cluster) {
        // Smoothly re-aim the camera
        this.controls.target.lerp(
            new THREE.Vector3(cluster.cx, cluster.cy, cluster.cz), 1.0
        );
        this.controls.update();

        const detail = document.getElementById('memory-map-detail');
        if (!detail || !this.data) return;

        const members = this.data.memories.filter(m => m.cluster_id === cluster.id);
        const col     = this.clusterColors[cluster.id];
        const hexCol  = col ? `#${col.getHexString()}` : '#aaa';

        detail.innerHTML = `
            <div class="mm-cluster-detail-title" style="color:${hexCol}">
                ${this._esc(cluster.theme || 'Unknown')}
            </div>
            <div class="mm-cluster-detail-meta">
                ${members.length} memories &middot; mass&nbsp;${(cluster.mass || 0).toFixed(1)}
            </div>
            <ul class="mm-cluster-memories">
                ${members.slice(0, 10).map(m => `
                    <li class="mm-memory-item">
                        <span class="mm-memory-pri p${m.priority || 3}"></span>
                        ${this._esc(m.summary.length > 85 ? m.summary.slice(0, 85) + '…' : m.summary)}
                    </li>
                `).join('')}
                ${members.length > 10
                    ? `<li class="mm-more">+${members.length - 10} more</li>`
                    : ''}
            </ul>
        `;
    }

    // ── Interaction ──────────────────────────────────────────────────────────

    _onMouseMove(e) {
        const rect = this.container.getBoundingClientRect();
        this.pointer.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
        this.pointer.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;

        if (!this.starPoints) return;

        this.raycaster.setFromCamera(this.pointer, this.camera);
        const hits = this.raycaster.intersectObject(this.starPoints);

        if (hits.length > 0) {
            const idx = hits[0].index;
            this.hoveredIdx = idx;
            this.starPoints.material.uniforms.uHovered.value = idx;

            const mem = this.starPoints.userData.memories[idx];
            if (mem && this.tooltip) {
                this.tooltip.innerHTML = `
                    <div class="mm-tip-summary">${this._esc(mem.summary)}</div>
                    <div class="mm-tip-meta">${this._esc(mem.category)} &middot; priority ${mem.priority}</div>
                `;
                this.tooltip.style.left = (e.clientX - rect.left + 16) + 'px';
                this.tooltip.style.top  = (e.clientY - rect.top  - 8)  + 'px';
                this.tooltip.classList.remove('hidden');
                this.container.style.cursor = 'pointer';
            }
        } else {
            this._hideTooltip();
        }

        // Show cluster label when mouse is near its centroid in screen space
        if (this.data?.clusters) {
            const rect2 = this.container.getBoundingClientRect();
            const mx    = e.clientX - rect2.left;
            const my    = e.clientY - rect2.top;
            let closestId   = null;
            let closestDist = 80; // px threshold

            this.data.clusters.forEach(cluster => {
                if (!cluster.count) return;
                const v = new THREE.Vector3(cluster.cx, cluster.cy, cluster.cz)
                    .project(this.camera);
                if (v.z > 1) return; // behind camera
                const sx   = (v.x * 0.5 + 0.5) * rect2.width;
                const sy   = (-v.y * 0.5 + 0.5) * rect2.height;
                const dist = Math.hypot(mx - sx, my - sy);
                if (dist < closestDist) { closestDist = dist; closestId = cluster.id; }
            });

            if (this.hoveredClusterId !== closestId) {
                this.hoveredClusterId = closestId;
                this.labelEls.forEach(el => {
                    el.style.opacity = el.dataset.clusterId === closestId ? '1' : '0';
                });
            }
        }
    }

    _onClick() {
        if (!this.starPoints || this.hoveredIdx < 0) return;
        const mem = this.starPoints.userData.memories[this.hoveredIdx];
        if (!mem?.cluster_id) return;
        const cluster = this.data.clusters.find(c => c.id === mem.cluster_id);
        if (cluster) this._focusCluster(cluster);
    }

    _hideTooltip() {
        if (this.tooltip) this.tooltip.classList.add('hidden');
        if (this.starPoints) this.starPoints.material.uniforms.uHovered.value = -1;
        this.hoveredIdx = -1;
        this.container.style.cursor = '';
        // Hide all cluster labels when mouse leaves the canvas
        if (this.hoveredClusterId !== null) {
            this.hoveredClusterId = null;
            this.labelEls.forEach(el => el.style.opacity = '0');
        }
    }

    // ── Animation loop ───────────────────────────────────────────────────────

    _startLoop() {
        const tick = () => {
            this.animId = requestAnimationFrame(tick);
            const t = this.clock.getElapsedTime();

            if (this.starPoints) {
                this.starPoints.material.uniforms.uTime.value = t;
            }
            if (this.sunPoints) {
                this.sunPoints.material.uniforms.uTime.value = t;
            }

            // Outer nebula halos breathe gently
            const breath = 1 + 0.028 * Math.sin(t * 0.45);
            this.nebulaMeshes.forEach(({ mesh, outer }) => {
                if (outer) mesh.scale.setScalar(breath);
            });

            this._updateOrbits(t);
            this.controls.update();
            this._updateLabels();
            this.renderer.render(this.scene, this.camera);
        };
        tick();
    }

    /** Project 3D cluster positions → screen coords and move HTML label divs. */
    _updateLabels() {
        if (!this.labelEls.length || !this.container) return;
        const rect = this.container.getBoundingClientRect();

        this.labelEls.forEach(el => {
            const v = new THREE.Vector3(
                parseFloat(el.dataset.x),
                parseFloat(el.dataset.y),
                parseFloat(el.dataset.z)
            ).project(this.camera);

            if (v.z > 1) { el.style.opacity = '0'; return; } // behind camera — always hide

            // Only update position; opacity is controlled by _onMouseMove proximity
            el.style.transform =
                `translate(${(v.x * 0.5 + 0.5) * rect.width}px, ${(-v.y * 0.5 + 0.5) * rect.height}px) translate(-50%, -50%)`;
        });
    }

    _onResize() {
        if (!this.container || !this.renderer) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    _clearMemoryObjects() {
        if (this.starPoints) {
            this.scene.remove(this.starPoints);
            this.starPoints.geometry.dispose();
            this.starPoints.material.dispose();
            this.starPoints = null;
        }
        if (this.sunPoints) {
            this.scene.remove(this.sunPoints);
            this.sunPoints.geometry.dispose();
            this.sunPoints.material.dispose();
            this.sunPoints = null;
        }
        if (this.orbitalLines) {
            this.scene.remove(this.orbitalLines);
            this.orbitalLines.geometry.dispose();
            this.orbitalLines.material.dispose();
            this.orbitalLines = null;
        }
        this.orbitalParams = null;
        this.nebulaMeshes.forEach(({ mesh }) => {
            this.scene.remove(mesh);
            mesh.geometry.dispose();
            mesh.material.dispose();
        });
        this.nebulaMeshes = [];
        this.labelEls.forEach(el => el.remove());
        this.labelEls     = [];
        this.clusterColors = {};
    }

    _esc(s) {
        return String(s)
            .replace(/&/g,  '&amp;')
            .replace(/</g,  '&lt;')
            .replace(/>/g,  '&gt;')
            .replace(/"/g,  '&quot;');
    }
}

// ── Singleton + wiring ────────────────────────────────────────────────────

const memoryMap = new MemoryMapViz();
window.memoryMap = memoryMap;

// Expose open/close to app.js via window (app.js loads before this module)
window._openMemoryMap = function () {
    const panel = document.getElementById('memory-map-panel');
    if (!panel) return;

    // Close any other open panels first
    window._closeAllPanels?.();

    // Slide in
    panel.classList.add('open');

    // Mount Three.js the first time (needs panel to be visible for correct size)
    if (!memoryMap.mounted) {
        requestAnimationFrame(() => {
            const canvas = document.getElementById('memory-map-canvas');
            memoryMap.mount(canvas);
            memoryMap.loadAndRender();
        });
    } else {
        memoryMap.loadAndRender();
    }
};

window._closeMemoryMap = function () {
    document.getElementById('memory-map-panel')?.classList.remove('open');
};

window._refreshMemoryMap = function () {
    if (memoryMap.mounted) memoryMap.loadAndRender();
};

// ── Drag-to-resize + maximize ─────────────────────────────────────────────

(function initPanelResize() {
    const panel   = document.getElementById('memory-map-panel');
    const resizer = panel?.querySelector('.memory-map-resizer');
    const maxBtn  = document.getElementById('memory-map-maximize-btn');
    if (!panel || !resizer) return;

    /** px width of the left sidebar (reads CSS var so it's always correct). */
    const sidebarW = () =>
        parseInt(getComputedStyle(document.documentElement)
            .getPropertyValue('--sidebar-width')) || 320;

    let maximized = false;

    // ── Drag handle ──────────────────────────────────────────────────────

    resizer.addEventListener('mousedown', e => {
        e.preventDefault();
        const startX = e.clientX;
        const startW = panel.offsetWidth;
        resizer.classList.add('dragging');
        document.body.style.cursor    = 'col-resize';
        document.body.style.userSelect = 'none';
        maximized = false;
        if (maxBtn) maxBtn.textContent = '⤢';

        const onMove = ev => {
            const maxW = window.innerWidth - sidebarW() - 8;
            const newW = Math.max(320, Math.min(maxW, startW + (ev.clientX - startX)));
            panel.style.width = newW + 'px';
            if (memoryMap.mounted) memoryMap._onResize();
        };

        const onUp = () => {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup',  onUp);
            resizer.classList.remove('dragging');
            document.body.style.cursor    = '';
            document.body.style.userSelect = '';
        };

        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup',  onUp);
    });

    // ── Maximize / restore button ────────────────────────────────────────

    if (maxBtn) {
        maxBtn.addEventListener('click', () => {
            if (!maximized) {
                panel.style.width = (window.innerWidth - sidebarW() - 8) + 'px';
                maxBtn.textContent = '⤡';
                maxBtn.title       = 'Restore';
                maximized = true;
            } else {
                panel.style.width  = ''; // revert to CSS min(700px, …)
                maxBtn.textContent = '⤢';
                maxBtn.title       = 'Maximize / Restore';
                maximized = false;
            }
            if (memoryMap.mounted) memoryMap._onResize();
        });
    }

    // Keep maximized state correct on window resize
    window.addEventListener('resize', () => {
        if (maximized) {
            panel.style.width = (window.innerWidth - sidebarW() - 8) + 'px';
            if (memoryMap.mounted) memoryMap._onResize();
        }
    });
})();
