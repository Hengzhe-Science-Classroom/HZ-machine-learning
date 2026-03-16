// === Chapter 8: Kernel Methods ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch08',
    number: 8,
    title: 'Kernel Methods',
    subtitle: 'Feature maps, reproducing kernels, and the kernel trick: working in high-dimensional spaces without ever entering them',
    sections: [
        // ========== SECTION 1: Feature Maps & Kernels ==========
        {
            id: 'ch08-sec01',
            title: 'Feature Maps & Kernels',
            content: `
<h2>8.1 Feature Maps &amp; Kernels</h2>

<div class="env-block intuition">
<strong>The Limitation of Linearity.</strong>
In Chapters 4 through 7, we built classifiers that separate data with hyperplanes. But many real datasets are not linearly separable in their original input space. The kernel trick lets us implicitly map data into a high-dimensional (even infinite-dimensional) feature space where a linear separator exists, without ever computing coordinates in that space.
</div>

<h3>Feature Maps</h3>

<p>A <em>feature map</em> is a function \\(\\phi: \\mathcal{X} \\to \\mathcal{F}\\) that transforms inputs from the original space \\(\\mathcal{X} \\subseteq \\mathbb{R}^d\\) into a (typically higher-dimensional) feature space \\(\\mathcal{F}\\).</p>

<div class="env-block example">
<strong>Example 8.1.1.</strong> For \\(\\mathbf{x} = (x_1, x_2)^\\top \\in \\mathbb{R}^2\\), define
\\[
\\phi(\\mathbf{x}) = (x_1^2,\\; \\sqrt{2}\\, x_1 x_2,\\; x_2^2)^\\top \\in \\mathbb{R}^3.
\\]
A dataset that forms concentric circles in \\(\\mathbb{R}^2\\) may become linearly separable in \\(\\mathbb{R}^3\\) after this mapping. The inner product in the feature space is
\\[
\\langle \\phi(\\mathbf{x}), \\phi(\\mathbf{z}) \\rangle = x_1^2 z_1^2 + 2 x_1 x_2 z_1 z_2 + x_2^2 z_2^2 = (\\mathbf{x}^\\top \\mathbf{z})^2.
\\]
</div>

<h3>The Kernel Function</h3>

<p>A <em>kernel function</em> \\(k: \\mathcal{X} \\times \\mathcal{X} \\to \\mathbb{R}\\) computes the inner product in the feature space directly:</p>

\\[
k(\\mathbf{x}, \\mathbf{z}) = \\langle \\phi(\\mathbf{x}),\\, \\phi(\\mathbf{z}) \\rangle_{\\mathcal{F}}.
\\]

<p>The power of this idea is that we never need to compute \\(\\phi(\\mathbf{x})\\) explicitly. If an algorithm depends on the data only through inner products \\(\\mathbf{x}_i^\\top \\mathbf{x}_j\\), we can replace each inner product with \\(k(\\mathbf{x}_i, \\mathbf{x}_j)\\) and operate implicitly in the feature space. This substitution is called the <strong>kernel trick</strong>.</p>

<div class="env-block definition">
<strong>Definition 8.1.2 (Kernel Trick).</strong> Given a learning algorithm whose solution depends on the data only through inner products \\(\\langle \\mathbf{x}_i, \\mathbf{x}_j \\rangle\\), we can "kernelize" it by replacing each inner product with \\(k(\\mathbf{x}_i, \\mathbf{x}_j)\\) for some kernel \\(k\\).
</div>

<div class="env-block remark">
<strong>Computational Savings.</strong> The polynomial feature map \\(\\phi\\) for degree-\\(p\\) polynomials in \\(d\\) dimensions produces \\(\\binom{d+p}{p}\\) features. For \\(d=100, p=5\\), that is over 96 million coordinates. But the polynomial kernel \\(k(\\mathbf{x},\\mathbf{z}) = (\\mathbf{x}^\\top\\mathbf{z} + c)^p\\) computes the same inner product in \\(O(d)\\) time.
</div>

<h3>Kernel Matrix (Gram Matrix)</h3>

<p>Given a dataset \\(\\{\\mathbf{x}_1, \\ldots, \\mathbf{x}_n\\}\\), the <em>kernel matrix</em> (or Gram matrix) is the \\(n \\times n\\) matrix</p>

\\[
\\mathbf{K}_{ij} = k(\\mathbf{x}_i, \\mathbf{x}_j).
\\]

<p>Since \\(\\mathbf{K}_{ij} = \\langle \\phi(\\mathbf{x}_i), \\phi(\\mathbf{x}_j) \\rangle\\), the kernel matrix is always symmetric and positive semi-definite.</p>

<div class="viz-placeholder" data-viz="viz-feature-map-3d"></div>
`,
            visualizations: [
                {
                    id: 'viz-feature-map-3d',
                    title: 'Feature Map: 2D to 3D',
                    description: 'Data on the left lives in 2D and is not linearly separable (concentric pattern). On the right, the feature map \\(\\phi(x_1,x_2) = (x_1^2, \\sqrt{2}\\,x_1 x_2, x_2^2)\\) lifts it to 3D, where a plane can separate the classes. Drag the rotation angle.',
                    setup(container, controls) {
                        const wrapper = document.createElement('div');
                        wrapper.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;justify-content:center;';
                        container.appendChild(wrapper);

                        const vizL = new VizEngine(wrapper, { width: 280, height: 280, scale: 50, originX: 140, originY: 140 });
                        const vizR = new VizEngine(wrapper, { width: 280, height: 280, scale: 50, originX: 140, originY: 160 });

                        let angle = 0.5;
                        VizEngine.createSlider(controls, 'Rotation', 0, 6.28, 0.5, 0.05, v => { angle = v; });

                        // Generate concentric circle data
                        const pts = [];
                        const rng = (seed) => { let s = seed; return () => { s = (s * 16807) % 2147483647; return s / 2147483647; }; };
                        const rand = rng(42);
                        for (let i = 0; i < 60; i++) {
                            const cls = i < 30 ? 0 : 1;
                            const r = cls === 0 ? 0.6 * rand() + 0.1 : 1.2 * rand() + 1.0;
                            const th = rand() * 2 * Math.PI;
                            pts.push({ x: r * Math.cos(th), y: r * Math.sin(th), cls });
                        }

                        function project3D(x, y, z, ang) {
                            const ca = Math.cos(ang), sa = Math.sin(ang);
                            const px = x * ca - z * sa;
                            const py = y;
                            return [px, py];
                        }

                        function draw() {
                            // Left: original 2D
                            vizL.clear(); vizL.drawGrid(1); vizL.drawAxes();
                            vizL.screenText('Input Space (x\u2081, x\u2082)', 140, 16, vizL.colors.white, 12);
                            for (const p of pts) {
                                const col = p.cls === 0 ? vizL.colors.blue : vizL.colors.orange;
                                vizL.drawPoint(p.x, p.y, col, null, 4);
                            }

                            // Right: 3D feature space
                            vizR.clear();
                            const ctx = vizR.ctx;
                            ctx.fillStyle = vizR.colors.bg;
                            ctx.fillRect(0, 0, vizR.width, vizR.height);
                            vizR.screenText('Feature Space \u03c6(x)', 140, 16, vizR.colors.white, 12);

                            // Draw 3D axes
                            const axLen = 2.2;
                            const axes = [
                                { v: [axLen, 0, 0], label: 'x\u2081\u00b2' },
                                { v: [0, axLen, 0], label: '\u221a2 x\u2081x\u2082' },
                                { v: [0, 0, axLen], label: 'x\u2082\u00b2' }
                            ];
                            for (const ax of axes) {
                                const [px, py] = project3D(ax.v[0], ax.v[1], ax.v[2], angle);
                                const [sx1, sy1] = vizR.toScreen(0, 0);
                                const [sx2, sy2] = vizR.toScreen(px, py);
                                ctx.strokeStyle = vizR.colors.axis; ctx.lineWidth = 1;
                                ctx.beginPath(); ctx.moveTo(sx1, sy1); ctx.lineTo(sx2, sy2); ctx.stroke();
                                ctx.fillStyle = vizR.colors.text; ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'center'; ctx.fillText(ax.label, sx2, sy2 - 8);
                            }

                            // Plot mapped points
                            const sortedPts = pts.map(p => {
                                const fx = p.x * p.x;
                                const fy = Math.sqrt(2) * p.x * p.y;
                                const fz = p.y * p.y;
                                const [px, py] = project3D(fx, fy, fz, angle);
                                return { px, py, cls: p.cls, depth: fx * Math.sin(angle) + fz * Math.cos(angle) };
                            }).sort((a, b) => a.depth - b.depth);

                            for (const sp of sortedPts) {
                                const col = sp.cls === 0 ? vizR.colors.blue : vizR.colors.orange;
                                vizR.drawPoint(sp.px, sp.py, col, null, 4);
                            }

                            // Draw separating plane hint
                            const planeY = 0.5;
                            const [psx1, psy1] = vizR.toScreen(...project3D(-2, planeY, -2, angle));
                            const [psx2, psy2] = vizR.toScreen(...project3D(2, planeY, -2, angle));
                            const [psx3, psy3] = vizR.toScreen(...project3D(2, planeY, 2, angle));
                            const [psx4, psy4] = vizR.toScreen(...project3D(-2, planeY, 2, angle));
                            ctx.fillStyle = vizR.colors.green + '18';
                            ctx.beginPath();
                            ctx.moveTo(psx1, psy1); ctx.lineTo(psx2, psy2);
                            ctx.lineTo(psx3, psy3); ctx.lineTo(psx4, psy4);
                            ctx.closePath(); ctx.fill();
                            ctx.strokeStyle = vizR.colors.green + '40'; ctx.lineWidth = 1; ctx.stroke();
                        }
                        vizL.animate(draw);
                        return { stopAnimation: () => vizL.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Verify that for \\(\\phi(\\mathbf{x}) = (x_1^2, \\sqrt{2}\\,x_1 x_2, x_2^2)\\), we have \\(\\langle \\phi(\\mathbf{x}), \\phi(\\mathbf{z}) \\rangle = (\\mathbf{x}^\\top \\mathbf{z})^2\\).',
                    hint: 'Expand both sides. On the left, compute the dot product of the three-component vectors. On the right, expand \\((x_1 z_1 + x_2 z_2)^2\\).',
                    solution: '\\(\\langle \\phi(\\mathbf{x}), \\phi(\\mathbf{z}) \\rangle = x_1^2 z_1^2 + 2 x_1 x_2 z_1 z_2 + x_2^2 z_2^2\\). Also \\((\\mathbf{x}^\\top \\mathbf{z})^2 = (x_1 z_1 + x_2 z_2)^2 = x_1^2 z_1^2 + 2 x_1 z_1 x_2 z_2 + x_2^2 z_2^2\\). These are identical.'
                },
                {
                    question: 'Why is it essential that the kernel matrix \\(\\mathbf{K}\\) be positive semi-definite? What would go wrong if it were not?',
                    hint: 'Think about what PSD means geometrically: \\(\\mathbf{K}\\) is a matrix of inner products. What does a negative eigenvalue imply about "distances" in the feature space?',
                    solution: 'PSD ensures that \\(k\\) corresponds to a valid inner product in some feature space. If \\(\\mathbf{K}\\) had a negative eigenvalue, there would exist a vector \\(\\mathbf{\\alpha}\\) with \\(\\mathbf{\\alpha}^\\top \\mathbf{K} \\mathbf{\\alpha} &lt; 0\\), meaning the "squared norm" \\(\\|\\sum_i \\alpha_i \\phi(\\mathbf{x}_i)\\|^2 &lt; 0\\), which is impossible for a valid inner product space. Optimization problems (e.g., SVM dual) could become non-convex.'
                },
                {
                    question: 'The SVM dual involves only inner products \\(\\mathbf{x}_i^\\top \\mathbf{x}_j\\). Explain how to kernelize it and what changes in the prediction function.',
                    hint: 'Replace each inner product with \\(k(\\mathbf{x}_i, \\mathbf{x}_j)\\). Then think about the decision function \\(f(\\mathbf{x}) = \\sum_i \\alpha_i y_i k(\\mathbf{x}_i, \\mathbf{x}) + b\\).',
                    solution: 'In the SVM dual, replace \\(\\mathbf{x}_i^\\top \\mathbf{x}_j\\) with \\(k(\\mathbf{x}_i, \\mathbf{x}_j)\\). The dual objective becomes \\(\\sum_i \\alpha_i - \\frac{1}{2} \\sum_{i,j} \\alpha_i \\alpha_j y_i y_j k(\\mathbf{x}_i, \\mathbf{x}_j)\\). The decision function is \\(f(\\mathbf{x}) = \\sum_i \\alpha_i y_i k(\\mathbf{x}_i, \\mathbf{x}) + b\\). Only support vectors (\\(\\alpha_i &gt; 0\\)) contribute, and we never compute \\(\\phi(\\mathbf{x})\\) explicitly.'
                }
            ]
        },

        // ========== SECTION 2: Mercer's Theorem ==========
        {
            id: 'ch08-sec02',
            title: "Mercer's Theorem",
            content: `
<h2>8.2 Mercer's Theorem</h2>

<div class="env-block intuition">
<strong>When Is a Function a Valid Kernel?</strong>
Not every function \\(k(\\mathbf{x}, \\mathbf{z})\\) corresponds to an inner product in some feature space. Mercer's theorem gives us a precise characterization: a function is a valid kernel if and only if it produces positive semi-definite Gram matrices for every possible dataset.
</div>

<h3>Positive Semi-Definite Kernels</h3>

<div class="env-block definition">
<strong>Definition 8.2.1 (Positive Semi-Definite Kernel).</strong> A function \\(k: \\mathcal{X} \\times \\mathcal{X} \\to \\mathbb{R}\\) is a <em>positive semi-definite</em> (PSD) kernel if for every finite set \\(\\{\\mathbf{x}_1, \\ldots, \\mathbf{x}_n\\} \\subset \\mathcal{X}\\), the Gram matrix \\(\\mathbf{K}\\) with entries \\(K_{ij} = k(\\mathbf{x}_i, \\mathbf{x}_j)\\) is positive semi-definite, i.e.,
\\[
\\sum_{i=1}^n \\sum_{j=1}^n c_i c_j\\, k(\\mathbf{x}_i, \\mathbf{x}_j) \\geq 0 \\quad \\text{for all } c_1, \\ldots, c_n \\in \\mathbb{R}.
\\]
</div>

<div class="env-block theorem">
<strong>Theorem 8.2.2 (Mercer's Theorem, Simplified).</strong> A continuous, symmetric function \\(k: \\mathcal{X} \\times \\mathcal{X} \\to \\mathbb{R}\\) (with \\(\\mathcal{X}\\) compact) is a valid kernel (i.e., corresponds to an inner product in some feature space) if and only if it is positive semi-definite.
</div>

<div class="env-block remark">
<strong>The Full Mercer Expansion.</strong> Under the conditions of the theorem, \\(k\\) admits a (possibly infinite) eigenfunction expansion:
\\[
k(\\mathbf{x}, \\mathbf{z}) = \\sum_{j=1}^{\\infty} \\lambda_j \\, \\psi_j(\\mathbf{x}) \\, \\psi_j(\\mathbf{z}),
\\]
where \\(\\lambda_j \\geq 0\\) are eigenvalues and \\(\\{\\psi_j\\}\\) are orthonormal eigenfunctions. The feature map is then \\(\\phi(\\mathbf{x}) = (\\sqrt{\\lambda_1}\\,\\psi_1(\\mathbf{x}),\\; \\sqrt{\\lambda_2}\\,\\psi_2(\\mathbf{x}),\\; \\ldots)\\).
</div>

<h3>Reproducing Kernel Hilbert Space (RKHS)</h3>

<p>Given a PSD kernel \\(k\\), there exists a unique Hilbert space \\(\\mathcal{H}_k\\) of functions, called the <em>reproducing kernel Hilbert space</em> (RKHS), with two key properties:</p>
<ol>
<li>For each \\(\\mathbf{x} \\in \\mathcal{X}\\), the function \\(k(\\mathbf{x}, \\cdot) \\in \\mathcal{H}_k\\).</li>
<li><strong>Reproducing property:</strong> \\(f(\\mathbf{x}) = \\langle f,\\, k(\\mathbf{x}, \\cdot) \\rangle_{\\mathcal{H}_k}\\) for all \\(f \\in \\mathcal{H}_k\\).</li>
</ol>

<p>The RKHS norm \\(\\|f\\|_{\\mathcal{H}_k}\\) measures function complexity. Regularizing with this norm (as in kernel ridge regression or SVM) penalizes "wiggly" functions, enforcing smoothness.</p>

<h3>Building New Kernels from Old</h3>

<p>PSD kernels are closed under several operations, making it easy to construct new valid kernels:</p>
<ul>
<li><strong>Sum:</strong> \\(k_1 + k_2\\) is PSD if both \\(k_1, k_2\\) are PSD.</li>
<li><strong>Product:</strong> \\(k_1 \\cdot k_2\\) is PSD (Schur product theorem).</li>
<li><strong>Scalar multiple:</strong> \\(\\alpha \\, k\\) is PSD for \\(\\alpha \\geq 0\\).</li>
<li><strong>Polynomial:</strong> \\(p(k)\\) is PSD if \\(p\\) has non-negative coefficients.</li>
<li><strong>Exponentiation:</strong> \\(e^{k}\\) is PSD.</li>
</ul>

<div class="viz-placeholder" data-viz="viz-kernel-matrix-heatmap"></div>
`,
            visualizations: [
                {
                    id: 'viz-kernel-matrix-heatmap',
                    title: 'Kernel Matrix Heatmap',
                    description: 'Visualize the Gram matrix \\(K_{ij} = k(\\mathbf{x}_i, \\mathbf{x}_j)\\) for 15 data points under three different kernels. Notice how the structure changes: RBF is smooth and localized, polynomial captures global interactions, and linear is just \\(\\mathbf{X}\\mathbf{X}^\\top\\).',
                    setup(container, controls) {
                        const wrapper = document.createElement('div');
                        wrapper.style.cssText = 'display:flex;gap:16px;flex-wrap:wrap;justify-content:center;';
                        container.appendChild(wrapper);

                        const n = 15;
                        const rng = (s) => { let v = s; return () => { v = (v * 16807) % 2147483647; return v / 2147483647; }; };
                        const rand = rng(123);
                        const data = [];
                        for (let i = 0; i < n; i++) data.push([rand() * 4 - 2, rand() * 4 - 2]);
                        // Sort by x for nicer heatmap structure
                        data.sort((a, b) => a[0] - b[0]);

                        let gamma = 0.5;
                        VizEngine.createSlider(controls, 'RBF \u03b3', 0.1, 5, 0.5, 0.1, v => { gamma = v; draw(); });

                        const kernels = {
                            'Linear': (a, b) => a[0]*b[0] + a[1]*b[1],
                            'Polynomial (d=3)': (a, b) => Math.pow(a[0]*b[0] + a[1]*b[1] + 1, 3),
                            'RBF': (a, b) => { const dx = a[0]-b[0], dy = a[1]-b[1]; return Math.exp(-gamma*(dx*dx+dy*dy)); }
                        };

                        function drawHeatmap(ctx, K, title, offX, offY, cellSize) {
                            let maxVal = 0;
                            for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) maxVal = Math.max(maxVal, Math.abs(K[i][j]));
                            if (maxVal < 1e-10) maxVal = 1;

                            ctx.fillStyle = '#c9d1d9'; ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.fillText(title, offX + n*cellSize/2, offY - 6);

                            for (let i = 0; i < n; i++) {
                                for (let j = 0; j < n; j++) {
                                    const val = K[i][j] / maxVal;
                                    const r = Math.round(88 + val * 120);
                                    const g = Math.round(166 + val * 80);
                                    const b = Math.round(255 * val);
                                    ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
                                    ctx.fillRect(offX + j * cellSize, offY + i * cellSize, cellSize - 1, cellSize - 1);
                                }
                            }
                        }

                        const canvas = document.createElement('canvas');
                        const totalW = 640, totalH = 260;
                        const dpr = window.devicePixelRatio || 1;
                        canvas.width = totalW * dpr; canvas.height = totalH * dpr;
                        canvas.style.width = totalW + 'px'; canvas.style.height = totalH + 'px';
                        wrapper.appendChild(canvas);
                        const ctx = canvas.getContext('2d');
                        ctx.scale(dpr, dpr);

                        function draw() {
                            ctx.fillStyle = '#0c0c20';
                            ctx.fillRect(0, 0, totalW, totalH);

                            const cellSize = 14;
                            const names = Object.keys(kernels);
                            const gap = (totalW - names.length * n * cellSize) / (names.length + 1);

                            names.forEach((name, ki) => {
                                const kfn = kernels[name];
                                const K = [];
                                for (let i = 0; i < n; i++) {
                                    K[i] = [];
                                    for (let j = 0; j < n; j++) K[i][j] = kfn(data[i], data[j]);
                                }
                                const offX = gap * (ki + 1) + ki * n * cellSize;
                                drawHeatmap(ctx, K, name, offX, 30, cellSize);
                            });
                        }
                        draw();
                        return { stopAnimation: () => {} };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Prove that if \\(k_1\\) and \\(k_2\\) are PSD kernels, then \\(k = k_1 + k_2\\) is also PSD.',
                    hint: 'Write the PSD condition \\(\\sum_{i,j} c_i c_j k(\\mathbf{x}_i, \\mathbf{x}_j) \\geq 0\\) and split the sum.',
                    solution: '\\(\\sum_{i,j} c_i c_j k(\\mathbf{x}_i, \\mathbf{x}_j) = \\sum_{i,j} c_i c_j k_1(\\mathbf{x}_i, \\mathbf{x}_j) + \\sum_{i,j} c_i c_j k_2(\\mathbf{x}_i, \\mathbf{x}_j) \\geq 0 + 0 = 0\\), since both \\(k_1\\) and \\(k_2\\) are PSD. Equivalently, the Gram matrix \\(\\mathbf{K} = \\mathbf{K}_1 + \\mathbf{K}_2\\), and the sum of PSD matrices is PSD.'
                },
                {
                    question: 'Explain the reproducing property \\(f(\\mathbf{x}) = \\langle f, k(\\mathbf{x}, \\cdot) \\rangle_{\\mathcal{H}_k}\\) in words. Why is it useful?',
                    hint: 'Think of \\(k(\\mathbf{x}, \\cdot)\\) as a "representer" of evaluation at \\(\\mathbf{x}\\).',
                    solution: 'The reproducing property says that evaluating a function \\(f\\) at a point \\(\\mathbf{x}\\) is the same as taking the inner product of \\(f\\) with the kernel function centered at \\(\\mathbf{x}\\). This means point evaluation is a continuous linear functional in \\(\\mathcal{H}_k\\), which ensures that convergence in RKHS norm implies pointwise convergence. It is the foundation of the representer theorem: the optimal solution to a regularized problem in the RKHS can be written as \\(f^*(\\cdot) = \\sum_{i=1}^n \\alpha_i k(\\mathbf{x}_i, \\cdot)\\).'
                },
                {
                    question: 'Show that the Gaussian RBF kernel \\(k(\\mathbf{x}, \\mathbf{z}) = \\exp(-\\gamma \\|\\mathbf{x} - \\mathbf{z}\\|^2)\\) corresponds to an infinite-dimensional feature map.',
                    hint: 'Expand the exponential as a Taylor series \\(e^t = \\sum_{n=0}^\\infty t^n / n!\\) and use the fact that each monomial in the expansion defines a finite-dimensional feature map whose product (by closure) is PSD.',
                    solution: 'Write \\(k(\\mathbf{x},\\mathbf{z}) = e^{-\\gamma\\|\\mathbf{x}\\|^2} \\cdot e^{2\\gamma \\mathbf{x}^\\top \\mathbf{z}} \\cdot e^{-\\gamma\\|\\mathbf{z}\\|^2}\\). The middle factor expands as \\(\\sum_{n=0}^\\infty \\frac{(2\\gamma)^n}{n!} (\\mathbf{x}^\\top \\mathbf{z})^n\\). Each term \\((\\mathbf{x}^\\top\\mathbf{z})^n\\) is a polynomial kernel with a finite feature map, but the infinite sum has infinitely many terms. The feature map has a component for every monomial of every degree, producing an infinite-dimensional vector. This can also be seen from the Mercer expansion: the eigenfunctions of the RBF kernel on \\(\\mathbb{R}^d\\) are Hermite functions, and there are countably infinitely many.'
                }
            ]
        },

        // ========== SECTION 3: Common Kernels ==========
        {
            id: 'ch08-sec03',
            title: 'Common Kernels',
            content: `
<h2>8.3 Common Kernels</h2>

<div class="env-block intuition">
<strong>Choosing the Right Kernel.</strong>
Different kernels encode different assumptions about the similarity of data points. Choosing a kernel is, in effect, choosing a hypothesis space. This section surveys the most widely used kernels and their properties.
</div>

<h3>Linear Kernel</h3>

\\[
k(\\mathbf{x}, \\mathbf{z}) = \\mathbf{x}^\\top \\mathbf{z}.
\\]

<p>The feature map is the identity: \\(\\phi(\\mathbf{x}) = \\mathbf{x}\\). A kernelized algorithm with the linear kernel is equivalent to its non-kernelized version. Useful as a baseline and when the data is (approximately) linearly separable.</p>

<h3>Polynomial Kernel</h3>

\\[
k(\\mathbf{x}, \\mathbf{z}) = (\\mathbf{x}^\\top \\mathbf{z} + c)^p, \\quad c \\geq 0,\\; p \\in \\mathbb{N}.
\\]

<p>The feature space includes all monomials up to degree \\(p\\). The constant \\(c\\) controls the relative weighting of lower-order vs. higher-order terms. When \\(c=0\\), only degree-\\(p\\) monomials appear (homogeneous polynomial kernel).</p>

<h3>Radial Basis Function (RBF / Gaussian) Kernel</h3>

\\[
k(\\mathbf{x}, \\mathbf{z}) = \\exp\\!\\left(-\\gamma \\|\\mathbf{x} - \\mathbf{z}\\|^2\\right), \\quad \\gamma &gt; 0.
\\]

<p>Also written with bandwidth \\(\\sigma\\): \\(\\gamma = 1/(2\\sigma^2)\\). This is the most popular kernel in practice. Key properties:</p>
<ul>
<li>\\(k(\\mathbf{x}, \\mathbf{x}) = 1\\) for all \\(\\mathbf{x}\\) (normalized).</li>
<li>\\(k(\\mathbf{x}, \\mathbf{z}) \\to 0\\) as \\(\\|\\mathbf{x} - \\mathbf{z}\\| \\to \\infty\\) (locality).</li>
<li>The feature space is infinite-dimensional.</li>
<li>Small \\(\\gamma\\): smooth, broad influence; large \\(\\gamma\\): sharp, local influence (risk of overfitting).</li>
</ul>

<h3>String and Structured Kernels</h3>

<p>Kernels can be defined on non-vectorial data. The <em>string kernel</em> counts common substrings of length \\(p\\) between two strings \\(s, t\\):</p>

\\[
k_p(s, t) = \\sum_{u \\in \\Sigma^p} \\#(u \\text{ in } s) \\cdot \\#(u \\text{ in } t),
\\]

<p>where \\(\\Sigma\\) is the alphabet. String kernels are widely used in bioinformatics (protein classification) and text categorization. Other structured kernels include <em>graph kernels</em>, <em>tree kernels</em>, and <em>Fisher kernels</em>.</p>

<div class="env-block remark">
<strong>Kernel Selection.</strong> In practice, one typically tries RBF first (it is a "universal kernel," meaning it can approximate any continuous function arbitrarily well). If domain knowledge suggests polynomial interactions, use the polynomial kernel. If the data is not vectorial, use an appropriate structured kernel. Cross-validation selects hyperparameters (\\(\\gamma\\), \\(c\\), \\(p\\)).
</div>

<div class="viz-placeholder" data-viz="viz-kernel-boundaries"></div>
`,
            visualizations: [
                {
                    id: 'viz-kernel-boundaries',
                    title: 'Decision Boundaries: Different Kernels',
                    description: 'Same XOR-like dataset, different kernel SVM decision boundaries. Toggle between kernels to compare. The RBF kernel captures the non-linear boundary, while the linear kernel fails.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 55, originX: null, originY: null });
                        viz.originX = viz.width / 2;
                        viz.originY = viz.height / 2;

                        // XOR-like dataset
                        const pts = [
                            {x:1,y:1,c:1},{x:1.3,y:0.8,c:1},{x:0.7,y:1.2,c:1},{x:0.9,y:0.9,c:1},{x:1.1,y:1.3,c:1},
                            {x:-1,y:-1,c:1},{x:-1.2,y:-0.8,c:1},{x:-0.8,y:-1.1,c:1},{x:-0.9,y:-1.3,c:1},{x:-1.1,y:-0.7,c:1},
                            {x:1,y:-1,c:0},{x:0.8,y:-1.2,c:0},{x:1.2,y:-0.9,c:0},{x:0.9,y:-0.8,c:0},{x:1.1,y:-1.1,c:0},
                            {x:-1,y:1,c:0},{x:-0.8,y:1.1,c:0},{x:-1.1,y:0.9,c:0},{x:-1.2,y:1.2,c:0},{x:-0.9,y:0.8,c:0}
                        ];

                        let kernelType = 'rbf';
                        const btnNames = ['Linear', 'Poly (d=2)', 'RBF'];
                        const btnKeys = ['linear', 'poly', 'rbf'];
                        const btns = [];
                        btnKeys.forEach((key, i) => {
                            const b = VizEngine.createButton(controls, btnNames[i], () => {
                                kernelType = key;
                                btns.forEach((bb, j) => bb.style.background = j === i ? '#58a6ff33' : '#1a1a40');
                            });
                            if (key === 'rbf') b.style.background = '#58a6ff33';
                            btns.push(b);
                        });

                        // Simple kernel-based classification using distance weighting
                        function kernelVal(a, b) {
                            if (kernelType === 'linear') return a.x*b.x + a.y*b.y;
                            if (kernelType === 'poly') return Math.pow(a.x*b.x + a.y*b.y + 1, 2);
                            const dx = a.x-b.x, dy = a.y-b.y;
                            return Math.exp(-1.5*(dx*dx+dy*dy));
                        }

                        function classify(px, py) {
                            // Kernel-weighted voting
                            let s0 = 0, s1 = 0;
                            const q = {x:px, y:py};
                            for (const p of pts) {
                                const kv = kernelVal(q, p);
                                if (p.c === 0) s0 += kv; else s1 += kv;
                            }
                            return s1 - s0;
                        }

                        function draw() {
                            viz.clear();
                            const ctx = viz.ctx;
                            const res = 4;
                            for (let sx = 0; sx < viz.width; sx += res) {
                                for (let sy = 0; sy < viz.height; sy += res) {
                                    const [mx, my] = viz.toMath(sx, sy);
                                    const val = classify(mx, my);
                                    if (val > 0) {
                                        ctx.fillStyle = viz.colors.blue + '18';
                                    } else {
                                        ctx.fillStyle = viz.colors.orange + '18';
                                    }
                                    ctx.fillRect(sx, sy, res, res);
                                }
                            }

                            // Draw decision boundary contour (val ~ 0)
                            for (let sx = 0; sx < viz.width; sx += 2) {
                                for (let sy = 0; sy < viz.height; sy += 2) {
                                    const [mx, my] = viz.toMath(sx, sy);
                                    const val = classify(mx, my);
                                    const [mx2] = viz.toMath(sx + 2, sy);
                                    const [, my2] = viz.toMath(sx, sy + 2);
                                    const vr = classify(mx2, my);
                                    const vd = classify(mx, my2);
                                    if ((val > 0) !== (vr > 0) || (val > 0) !== (vd > 0)) {
                                        ctx.fillStyle = viz.colors.white;
                                        ctx.fillRect(sx, sy, 2, 2);
                                    }
                                }
                            }

                            viz.drawGrid(1); viz.drawAxes();
                            for (const p of pts) {
                                const col = p.c === 1 ? viz.colors.blue : viz.colors.orange;
                                viz.drawPoint(p.x, p.y, col, null, 5);
                            }
                            viz.screenText('Kernel: ' + (kernelType === 'linear' ? 'Linear' : kernelType === 'poly' ? 'Polynomial (d=2)' : 'RBF (\u03b3=1.5)'), viz.width/2, 18, viz.colors.white, 13);
                        }
                        viz.animate(draw);
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that the polynomial kernel \\(k(\\mathbf{x},\\mathbf{z}) = (\\mathbf{x}^\\top\\mathbf{z}+1)^2\\) for \\(\\mathbf{x},\\mathbf{z}\\in\\mathbb{R}^2\\) corresponds to a 6-dimensional feature map. Write it out explicitly.',
                    hint: 'Expand \\((x_1 z_1 + x_2 z_2 + 1)^2\\) and identify each term with a component of \\(\\phi\\).',
                    solution: '\\((x_1 z_1 + x_2 z_2 + 1)^2 = x_1^2 z_1^2 + x_2^2 z_2^2 + 2x_1 x_2 z_1 z_2 + 2x_1 z_1 + 2x_2 z_2 + 1\\). So \\(\\phi(\\mathbf{x}) = (x_1^2,\\; x_2^2,\\; \\sqrt{2}x_1 x_2,\\; \\sqrt{2}x_1,\\; \\sqrt{2}x_2,\\; 1)^\\top\\), which is 6-dimensional. The \\(\\sqrt{2}\\) factors ensure the inner product \\(\\langle \\phi(\\mathbf{x}), \\phi(\\mathbf{z}) \\rangle\\) equals the kernel.'
                },
                {
                    question: 'Why is the RBF kernel called a "universal kernel"? What does this imply about the expressiveness of RBF-SVM?',
                    hint: 'A kernel is universal if its RKHS is dense in the space of continuous functions on any compact set.',
                    solution: 'The RBF kernel is universal because its RKHS is dense in \\(C(\\mathcal{X})\\), the space of continuous functions on any compact \\(\\mathcal{X}\\), under the supremum norm. This means that given enough data and the right regularization, an RBF-SVM can approximate any continuous decision boundary to arbitrary accuracy. In practice, this means RBF-SVM is very flexible, but the flip side is that it can overfit if \\(\\gamma\\) is too large or regularization is too weak.'
                },
                {
                    question: 'Explain intuitively why the linear kernel fails on XOR-type data, while the polynomial kernel of degree 2 succeeds.',
                    hint: 'Think about what the feature map of each kernel "sees." The linear kernel preserves the original coordinates; the polynomial kernel adds interaction terms.',
                    solution: 'XOR-type data has the pattern: \\((+,+) \\to \\text{class 1},\\; (-,-) \\to \\text{class 1},\\; (+,-) \\to \\text{class 0},\\; (-,+) \\to \\text{class 0}\\). The distinguishing feature is the <em>product</em> \\(x_1 x_2\\): it is positive for class 1 and negative for class 0. The linear kernel only "sees" \\(x_1\\) and \\(x_2\\) individually and cannot form the product term. The degree-2 polynomial kernel includes \\(x_1 x_2\\) as a feature (via the cross-term), so it can separate the classes with a hyperplane in feature space.'
                }
            ]
        },

        // ========== SECTION 4: Kernel PCA & Kernel Ridge Regression ==========
        {
            id: 'ch08-sec04',
            title: 'Kernel PCA & Kernel Ridge Regression',
            content: `
<h2>8.4 Kernel PCA &amp; Kernel Ridge Regression</h2>

<div class="env-block intuition">
<strong>Beyond Classification.</strong>
The kernel trick is not limited to SVMs. Any algorithm that depends on the data only through inner products can be kernelized. This section applies the kernel trick to two important problems: dimensionality reduction (PCA) and regression (ridge regression).
</div>

<h3>Kernel PCA</h3>

<p>Standard PCA finds directions of maximum variance in input space by computing eigenvectors of \\(\\mathbf{X}^\\top \\mathbf{X}\\). This only captures <em>linear</em> structure. Kernel PCA performs PCA in the feature space \\(\\mathcal{F}\\) induced by \\(\\phi\\), capturing non-linear patterns.</p>

<div class="env-block definition">
<strong>Definition 8.4.1 (Kernel PCA).</strong> Given centered kernel matrix \\(\\tilde{\\mathbf{K}}\\) with entries \\(\\tilde{K}_{ij} = \\langle \\tilde{\\phi}(\\mathbf{x}_i), \\tilde{\\phi}(\\mathbf{x}_j) \\rangle\\) (where \\(\\tilde{\\phi}\\) denotes centered features), the \\(k\\)-th principal component of a point \\(\\mathbf{x}\\) is
\\[
z_k(\\mathbf{x}) = \\sum_{i=1}^n \\alpha_i^{(k)} \\, \\tilde{k}(\\mathbf{x}_i, \\mathbf{x}),
\\]
where \\(\\boldsymbol{\\alpha}^{(k)}\\) is the \\(k\\)-th eigenvector of \\(\\tilde{\\mathbf{K}}\\), normalized so that \\(\\lambda_k (\\boldsymbol{\\alpha}^{(k)})^\\top \\boldsymbol{\\alpha}^{(k)} = 1\\).
</div>

<p><strong>Centering in feature space.</strong> Since we cannot compute the mean \\(\\bar{\\phi} = \\frac{1}{n}\\sum_i \\phi(\\mathbf{x}_i)\\) explicitly, we center the kernel matrix directly:</p>

\\[
\\tilde{\\mathbf{K}} = \\mathbf{K} - \\frac{1}{n} \\mathbf{1}_n \\mathbf{K} - \\frac{1}{n} \\mathbf{K} \\mathbf{1}_n + \\frac{1}{n^2} \\mathbf{1}_n \\mathbf{K} \\mathbf{1}_n,
\\]

<p>where \\(\\mathbf{1}_n\\) is the \\(n \\times n\\) matrix of all \\(1/n\\).</p>

<h3>Kernel Ridge Regression</h3>

<p>Ridge regression in feature space minimizes</p>

\\[
\\min_{\\mathbf{w}} \\sum_{i=1}^n (y_i - \\mathbf{w}^\\top \\phi(\\mathbf{x}_i))^2 + \\lambda \\|\\mathbf{w}\\|^2.
\\]

<p>By the representer theorem, the optimal \\(\\mathbf{w}\\) lies in the span of \\(\\{\\phi(\\mathbf{x}_1), \\ldots, \\phi(\\mathbf{x}_n)\\}\\), so we can write \\(\\mathbf{w} = \\sum_{i=1}^n \\alpha_i \\phi(\\mathbf{x}_i)\\). Substituting:</p>

\\[
\\boldsymbol{\\alpha} = (\\mathbf{K} + \\lambda \\mathbf{I})^{-1} \\mathbf{y}.
\\]

<p>Prediction at a new point \\(\\mathbf{x}_*\\) is</p>

\\[
\\hat{y}_* = \\sum_{i=1}^n \\alpha_i \\, k(\\mathbf{x}_i, \\mathbf{x}_*) = \\mathbf{k}_*^\\top (\\mathbf{K} + \\lambda \\mathbf{I})^{-1} \\mathbf{y},
\\]

<p>where \\(\\mathbf{k}_* = (k(\\mathbf{x}_1, \\mathbf{x}_*), \\ldots, k(\\mathbf{x}_n, \\mathbf{x}_*))^\\top\\).</p>

<div class="env-block remark">
<strong>Connection to Gaussian Processes.</strong> Kernel ridge regression is equivalent to the posterior mean of a Gaussian process with kernel \\(k\\) and noise variance \\(\\lambda\\). We will explore this connection in Chapter 15.
</div>

<div class="viz-placeholder" data-viz="viz-kernel-pca"></div>
`,
            visualizations: [
                {
                    id: 'viz-kernel-pca',
                    title: 'Kernel PCA vs. Linear PCA',
                    description: 'Concentric circles in 2D: linear PCA cannot separate the two classes because they share the same principal axes. Kernel PCA with an RBF kernel finds non-linear components that separate them. Toggle between the two methods.',
                    setup(container, controls) {
                        const wrapper = document.createElement('div');
                        wrapper.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;justify-content:center;';
                        container.appendChild(wrapper);

                        const vizL = new VizEngine(wrapper, { width: 280, height: 280, scale: 45, originX: 140, originY: 140 });
                        const vizR = new VizEngine(wrapper, { width: 280, height: 280, scale: 80, originX: 140, originY: 140 });

                        // Generate concentric circle data
                        const rng = (s) => { let v = s; return () => { v = (v * 16807) % 2147483647; return v / 2147483647; }; };
                        const rand = rng(99);
                        const pts = [];
                        for (let i = 0; i < 80; i++) {
                            const cls = i < 40 ? 0 : 1;
                            const r = cls === 0 ? 0.5 + rand() * 0.4 : 1.5 + rand() * 0.5;
                            const th = rand() * 2 * Math.PI;
                            const nx = (rand() - 0.5) * 0.15;
                            const ny = (rand() - 0.5) * 0.15;
                            pts.push({ x: r * Math.cos(th) + nx, y: r * Math.sin(th) + ny, cls });
                        }

                        let method = 'kernel';
                        const bLinear = VizEngine.createButton(controls, 'Linear PCA', () => { method = 'linear'; bLinear.style.background = '#58a6ff33'; bKernel.style.background = '#1a1a40'; });
                        const bKernel = VizEngine.createButton(controls, 'Kernel PCA (RBF)', () => { method = 'kernel'; bKernel.style.background = '#58a6ff33'; bLinear.style.background = '#1a1a40'; });
                        bKernel.style.background = '#58a6ff33';

                        // Simple kernel PCA approximation
                        function computeKernelPCA() {
                            const n = pts.length;
                            const gamma = 0.8;
                            // Build kernel matrix
                            const K = [];
                            for (let i = 0; i < n; i++) {
                                K[i] = [];
                                for (let j = 0; j < n; j++) {
                                    const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
                                    K[i][j] = Math.exp(-gamma * (dx*dx + dy*dy));
                                }
                            }
                            // Center kernel matrix
                            const rowMean = K.map(row => row.reduce((a,b) => a+b, 0) / n);
                            const totalMean = rowMean.reduce((a,b) => a+b, 0) / n;
                            const Kc = [];
                            for (let i = 0; i < n; i++) {
                                Kc[i] = [];
                                for (let j = 0; j < n; j++) {
                                    Kc[i][j] = K[i][j] - rowMean[i] - rowMean[j] + totalMean;
                                }
                            }
                            // Power iteration for top 2 eigenvectors
                            function powerIter(M, deflated) {
                                let v = Array.from({length: n}, () => rand() - 0.5);
                                for (let iter = 0; iter < 100; iter++) {
                                    let w = new Array(n).fill(0);
                                    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) w[i] += M[i][j] * v[j];
                                    if (deflated) {
                                        const dp = w.reduce((s, wi, i) => s + wi * deflated[i], 0);
                                        for (let i = 0; i < n; i++) w[i] -= dp * deflated[i];
                                    }
                                    const norm = Math.sqrt(w.reduce((s, wi) => s + wi*wi, 0));
                                    v = w.map(wi => wi / (norm || 1));
                                }
                                const lam = v.reduce((s, vi, i) => {
                                    let mv = 0; for (let j = 0; j < n; j++) mv += M[i][j] * v[j];
                                    return s + vi * mv;
                                }, 0);
                                return { vec: v, val: lam };
                            }
                            const e1 = powerIter(Kc, null);
                            const e2 = powerIter(Kc, e1.vec);
                            // Project each point
                            const proj = [];
                            for (let i = 0; i < n; i++) {
                                let z1 = 0, z2 = 0;
                                for (let j = 0; j < n; j++) {
                                    z1 += e1.vec[j] * Kc[i][j];
                                    z2 += e2.vec[j] * Kc[i][j];
                                }
                                proj.push({ z1, z2 });
                            }
                            return proj;
                        }

                        function computeLinearPCA() {
                            const n = pts.length;
                            const mx = pts.reduce((s, p) => s + p.x, 0) / n;
                            const my = pts.reduce((s, p) => s + p.y, 0) / n;
                            const proj = [];
                            for (const p of pts) {
                                proj.push({ z1: p.x - mx, z2: p.y - my });
                            }
                            return proj;
                        }

                        const kpcaProj = computeKernelPCA();
                        const lpcaProj = computeLinearPCA();

                        function draw() {
                            // Left: original data
                            vizL.clear(); vizL.drawGrid(1); vizL.drawAxes();
                            vizL.screenText('Input Space', 140, 16, vizL.colors.white, 12);
                            for (const p of pts) {
                                const col = p.cls === 0 ? vizL.colors.blue : vizL.colors.orange;
                                vizL.drawPoint(p.x, p.y, col, null, 3.5);
                            }

                            // Right: projected data
                            vizR.clear(); vizR.drawGrid(1); vizR.drawAxes();
                            const proj = method === 'kernel' ? kpcaProj : lpcaProj;
                            vizR.screenText(method === 'kernel' ? 'Kernel PCA (RBF)' : 'Linear PCA', 140, 16, vizR.colors.white, 12);

                            // Normalize for display
                            let maxAbs = 0;
                            for (const p of proj) maxAbs = Math.max(maxAbs, Math.abs(p.z1), Math.abs(p.z2));
                            const sc = maxAbs > 0.01 ? 1.2 / maxAbs : 1;

                            for (let i = 0; i < pts.length; i++) {
                                const col = pts[i].cls === 0 ? vizR.colors.blue : vizR.colors.orange;
                                vizR.drawPoint(proj[i].z1 * sc, proj[i].z2 * sc, col, null, 3.5);
                            }
                            vizR.screenText('PC1', 265, 145, vizR.colors.text, 10);
                            vizR.screenText('PC2', 145, 15, vizR.colors.text, 10);
                        }
                        vizL.animate(draw);
                        return { stopAnimation: () => vizL.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'State the representer theorem and explain why it makes kernel ridge regression computationally feasible.',
                    hint: 'The representer theorem says the optimal solution to a regularized problem in an RKHS lies in the span of the kernel functions evaluated at the training points.',
                    solution: '<strong>Representer Theorem:</strong> For a regularized empirical risk minimization problem \\(\\min_{f \\in \\mathcal{H}_k} \\frac{1}{n}\\sum_{i=1}^n L(y_i, f(\\mathbf{x}_i)) + \\lambda \\|f\\|_{\\mathcal{H}_k}^2\\), the optimal \\(f^*\\) can be written as \\(f^*(\\cdot) = \\sum_{i=1}^n \\alpha_i k(\\mathbf{x}_i, \\cdot)\\). This reduces an infinite-dimensional optimization over \\(\\mathcal{H}_k\\) to a finite \\(n\\)-dimensional optimization over \\(\\boldsymbol{\\alpha} \\in \\mathbb{R}^n\\). For kernel ridge regression, the closed form is \\(\\boldsymbol{\\alpha} = (\\mathbf{K} + \\lambda \\mathbf{I})^{-1}\\mathbf{y}\\), which requires \\(O(n^3)\\) time (matrix inversion) regardless of the feature space dimension.'
                },
                {
                    question: 'Explain why kernel PCA with a linear kernel is equivalent to standard PCA.',
                    hint: 'With the linear kernel, \\(\\mathbf{K} = \\mathbf{X}\\mathbf{X}^\\top\\). How do the eigenvalues/eigenvectors of \\(\\mathbf{X}\\mathbf{X}^\\top\\) relate to those of \\(\\mathbf{X}^\\top\\mathbf{X}\\)?',
                    solution: 'With the linear kernel, \\(K_{ij} = \\mathbf{x}_i^\\top \\mathbf{x}_j\\), so \\(\\mathbf{K} = \\mathbf{X}\\mathbf{X}^\\top\\). Standard PCA computes eigenvectors of \\(\\mathbf{X}^\\top \\mathbf{X}\\). By the SVD, if \\(\\mathbf{X} = \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^\\top\\), then \\(\\mathbf{X}^\\top\\mathbf{X} = \\mathbf{V}\\boldsymbol{\\Sigma}^2\\mathbf{V}^\\top\\) and \\(\\mathbf{X}\\mathbf{X}^\\top = \\mathbf{U}\\boldsymbol{\\Sigma}^2\\mathbf{U}^\\top\\). The non-zero eigenvalues are the same. The projections \\(\\mathbf{X}\\mathbf{v}_k\\) (standard PCA) equal \\(\\sigma_k \\mathbf{u}_k\\), which is proportional to the kernel PCA eigenvector. So the embeddings coincide.'
                },
                {
                    question: 'Why must we center the kernel matrix in kernel PCA? What goes wrong if we skip centering?',
                    hint: 'PCA assumes zero-mean data. In the feature space, the mean is \\(\\bar{\\phi} = \\frac{1}{n}\\sum_i \\phi(\\mathbf{x}_i)\\), which is generally nonzero.',
                    solution: 'PCA finds directions of maximum variance, which requires the data to be centered (mean-subtracted). In the feature space, the features \\(\\phi(\\mathbf{x}_i)\\) generally have a nonzero mean \\(\\bar{\\phi}\\). Without centering, the first "principal component" would partly capture the mean direction rather than the direction of maximum variance. Centering in feature space is done implicitly via \\(\\tilde{\\mathbf{K}} = \\mathbf{K} - \\frac{1}{n}\\mathbf{1}_n\\mathbf{K} - \\frac{1}{n}\\mathbf{K}\\mathbf{1}_n + \\frac{1}{n^2}\\mathbf{1}_n\\mathbf{K}\\mathbf{1}_n\\), which subtracts the feature-space mean from each mapped point without ever computing \\(\\phi\\) explicitly.'
                }
            ]
        }
    ]
});
