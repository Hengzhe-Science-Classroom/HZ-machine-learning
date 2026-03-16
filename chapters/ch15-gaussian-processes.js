// === Chapter 15: Gaussian Processes ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch15',
    number: 15,
    title: 'Gaussian Processes',
    subtitle: 'Non-parametric Bayesian regression through infinite-dimensional priors over functions',
    sections: [
        // ===================== Section 1: From Bayesian Regression to GPs =====================
        {
            id: 'ch15-sec01',
            title: 'From Bayesian Regression to GPs',
            content: `<h2>15.1 From Bayesian Regression to Gaussian Processes</h2>
<div class="env-block intuition"><div class="env-title">Intuition</div><div class="env-body">
<p>In Bayesian linear regression (Ch. 14), we placed a prior over a finite weight vector \\(\\mathbf{w}\\). But the real object of interest is the <em>function</em> \\(f(\\mathbf{x}) = \\mathbf{w}^\\top\\boldsymbol{\\phi}(\\mathbf{x})\\). What if we cut out the middleman and placed a prior directly on the function? A Gaussian process does exactly this: it defines a distribution over functions such that any finite collection of function values is jointly Gaussian.</p>
</div></div>

<h3>Weight-Space to Function-Space</h3>
<p>With prior \\(\\mathbf{w} \\sim \\mathcal{N}(\\mathbf{0}, \\alpha^{-1}\\mathbf{I})\\) and \\(f(\\mathbf{x}) = \\mathbf{w}^\\top\\boldsymbol{\\phi}(\\mathbf{x})\\), the induced distribution over function values at inputs \\(\\mathbf{x}_1,\\ldots,\\mathbf{x}_N\\) is</p>
\\[
\\begin{pmatrix} f(\\mathbf{x}_1) \\\\ \\vdots \\\\ f(\\mathbf{x}_N) \\end{pmatrix} \\sim \\mathcal{N}\\!\\left(\\mathbf{0},\\; \\alpha^{-1}\\boldsymbol{\\Phi}\\boldsymbol{\\Phi}^\\top\\right),
\\]
<p>where the \\((i,j)\\) entry of the covariance is \\(k(\\mathbf{x}_i,\\mathbf{x}_j) = \\alpha^{-1}\\boldsymbol{\\phi}(\\mathbf{x}_i)^\\top\\boldsymbol{\\phi}(\\mathbf{x}_j)\\). This is a <em>kernel function</em>.</p>

<div class="env-block definition"><div class="env-title">Definition 15.1.1 -- Gaussian Process</div><div class="env-body">
<p>A <strong>Gaussian process</strong> is a collection of random variables, any finite subset of which has a joint Gaussian distribution. A GP is fully specified by its mean function \\(m(\\mathbf{x})\\) and covariance (kernel) function \\(k(\\mathbf{x},\\mathbf{x}')\\):</p>
\\[f \\sim \\mathcal{GP}\\bigl(m(\\mathbf{x}),\\; k(\\mathbf{x},\\mathbf{x}')\\bigr).\\]
<p>This means for any \\(\\{\\mathbf{x}_1,\\ldots,\\mathbf{x}_N\\}\\), the vector \\((f(\\mathbf{x}_1),\\ldots,f(\\mathbf{x}_N))^\\top \\sim \\mathcal{N}(\\boldsymbol{\\mu}, \\mathbf{K})\\), where \\(\\mu_i = m(\\mathbf{x}_i)\\) and \\(K_{ij} = k(\\mathbf{x}_i,\\mathbf{x}_j)\\).</p>
</div></div>

<div class="env-block definition"><div class="env-title">Definition 15.1.2 -- Squared Exponential (RBF) Kernel</div><div class="env-body">
\\[k_{\\text{SE}}(\\mathbf{x},\\mathbf{x}') = \\sigma_f^2 \\exp\\!\\left(-\\frac{\\|\\mathbf{x}-\\mathbf{x}'\\|^2}{2\\ell^2}\\right),\\]
<p>where \\(\\sigma_f^2\\) is the signal variance and \\(\\ell\\) is the length-scale. Nearby inputs (relative to \\(\\ell\\)) produce highly correlated function values, enforcing smoothness.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark -- Infinite Basis Functions</div><div class="env-body">
<p>The RBF kernel corresponds to an inner product in an infinite-dimensional feature space (via Mercer's theorem). A GP with this kernel is equivalent to Bayesian linear regression with infinitely many basis functions, bypassing the need to choose a finite set.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-gp-prior-samples"></div>`,
            visualizations: [{
                id: 'viz-gp-prior-samples',
                title: 'Random Samples from a GP Prior',
                description: 'Each curve is a random function drawn from \\(\\mathcal{GP}(0, k_{\\text{SE}})\\). Adjust the length-scale \\(\\ell\\) to see smoother or wigglier samples. Click "Resample" for new draws.',
                setup(body, controls) {
                    const viz = new VizEngine(body, {width:680,height:360,scale:55,originX:340,originY:180});
                    const ctx = viz.ctx;
                    let ell = 1.0, sf2 = 1.0, nSamples = 5;
                    const nX = 80;
                    const xs = []; for (let i = 0; i <= nX; i++) xs.push(-5 + 10 * i / nX);
                    let samples = [];
                    const colors = [viz.colors.blue, viz.colors.teal, viz.colors.orange, viz.colors.purple, viz.colors.pink];

                    function rng() { let u=0,v=0; while(!u) u=Math.random(); while(!v) v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

                    function rbf(x1, x2) { return sf2 * Math.exp(-0.5 * (x1 - x2) * (x1 - x2) / (ell * ell)); }

                    function cholesky(A) {
                        const n = A.length, L = [];
                        for (let i = 0; i < n; i++) { L[i] = new Array(n).fill(0); }
                        for (let i = 0; i < n; i++) {
                            for (let j = 0; j <= i; j++) {
                                let s = A[i][j];
                                for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
                                L[i][j] = (i === j) ? Math.sqrt(Math.max(s, 1e-10)) : s / L[j][j];
                            }
                        }
                        return L;
                    }

                    function sampleGP() {
                        const n = xs.length;
                        const K = [];
                        for (let i = 0; i < n; i++) { K[i] = []; for (let j = 0; j < n; j++) K[i][j] = rbf(xs[i], xs[j]) + (i === j ? 1e-6 : 0); }
                        const L = cholesky(K);
                        samples = [];
                        for (let s = 0; s < nSamples; s++) {
                            const z = []; for (let i = 0; i < n; i++) z.push(rng());
                            const f = new Array(n).fill(0);
                            for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) f[i] += L[i][j] * z[j];
                            samples.push(f);
                        }
                    }

                    function draw() {
                        viz.clear(); viz.drawGrid(1); viz.drawAxes();
                        for (let s = 0; s < samples.length; s++) {
                            ctx.strokeStyle = colors[s % colors.length]; ctx.lineWidth = 2;
                            ctx.beginPath();
                            for (let i = 0; i <= nX; i++) {
                                const [sx, sy] = viz.toScreen(xs[i], samples[s][i]);
                                i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                            }
                            ctx.stroke();
                        }
                        viz.screenText('GP Prior Samples', 80, 20, viz.colors.white, 13, 'left');
                        viz.screenText('\u2113 = ' + ell.toFixed(1), viz.width - 80, 20, viz.colors.teal, 12);
                    }

                    VizEngine.createSlider(controls, '\u2113 (length-scale)', 0.2, 3.0, 1.0, 0.1, v => { ell = v; sampleGP(); draw(); });
                    VizEngine.createSlider(controls, '\u03c3\u00b2_f', 0.2, 3.0, 1.0, 0.1, v => { sf2 = v; sampleGP(); draw(); });
                    VizEngine.createButton(controls, 'Resample', () => { sampleGP(); draw(); });

                    sampleGP(); draw(); return viz;
                }
            }],
            exercises: [
                { question: 'Show that \\(k(\\mathbf{x},\\mathbf{x}\') = \\alpha^{-1}\\boldsymbol{\\phi}(\\mathbf{x})^\\top\\boldsymbol{\\phi}(\\mathbf{x}\')\\) is a valid (positive semi-definite) kernel.',
                  hint: 'Show that for any \\(\\mathbf{c} \\in \\mathbb{R}^N\\), \\(\\sum_{i,j} c_i c_j k(\\mathbf{x}_i,\\mathbf{x}_j) \\geq 0\\).',
                  solution: '\\(\\sum_{i,j} c_i c_j k(\\mathbf{x}_i,\\mathbf{x}_j) = \\alpha^{-1}\\sum_{i,j} c_i c_j \\boldsymbol{\\phi}(\\mathbf{x}_i)^\\top\\boldsymbol{\\phi}(\\mathbf{x}_j) = \\alpha^{-1}\\|\\sum_i c_i \\boldsymbol{\\phi}(\\mathbf{x}_i)\\|^2 \\geq 0\\). This holds for any \\(\\mathbf{c}\\), so \\(k\\) is positive semi-definite.' },
                { question: 'What happens to GP prior samples as the length-scale \\(\\ell \\to 0\\)? As \\(\\ell \\to \\infty\\)?',
                  hint: 'Consider the correlation between nearby points in each limit.',
                  solution: 'As \\(\\ell \\to 0\\), \\(k(x,x\') \\to \\sigma_f^2 \\delta_{xx\'}\\), so function values become independent. Samples look like white noise. As \\(\\ell \\to \\infty\\), \\(k(x,x\') \\to \\sigma_f^2\\) for all pairs, so all values are perfectly correlated. Samples become constant functions (drawn from \\(\\mathcal{N}(0,\\sigma_f^2)\\)).' },
                { question: 'A GP with a polynomial kernel \\(k(x,x\') = (1 + xx\')^d\\) is equivalent to Bayesian regression with what basis functions?',
                  hint: 'Expand the polynomial for \\(d=2\\).',
                  solution: 'For \\(d=2\\): \\((1+xx\')^2 = 1 + 2xx\' + (xx\')^2\\). This equals \\(\\boldsymbol{\\phi}(x)^\\top\\boldsymbol{\\phi}(x\')\\) with \\(\\boldsymbol{\\phi}(x) = (1,\\sqrt{2}x,x^2)^\\top\\). In general, the degree-\\(d\\) polynomial kernel corresponds to all monomials up to degree \\(d\\) (with appropriate scaling).' }
            ]
        },

        // ===================== Section 2: GP Regression =====================
        {
            id: 'ch15-sec02',
            title: 'GP Regression',
            content: `<h2>15.2 GP Regression</h2>
<div class="env-block intuition"><div class="env-title">Intuition</div><div class="env-body">
<p>Conditioning a GP on observed data is like threading a rubber sheet through observed points: the function must pass near the data, but away from observations the sheet can flex freely. The posterior mean interpolates the data; the posterior variance is small near observations and large far away.</p>
</div></div>

<h3>Posterior Derivation</h3>
<p>Given training data \\(\\mathbf{X} = \\{\\mathbf{x}_1,\\ldots,\\mathbf{x}_N\\}\\) with noisy observations \\(\\mathbf{y} = \\mathbf{f} + \\boldsymbol{\\epsilon}\\), \\(\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0},\\sigma_n^2\\mathbf{I})\\), and a test input \\(\\mathbf{x}_*\\), the joint prior is</p>
\\[
\\begin{pmatrix} \\mathbf{y} \\\\ f_* \\end{pmatrix} \\sim \\mathcal{N}\\!\\left(\\mathbf{0},\\; \\begin{pmatrix} \\mathbf{K} + \\sigma_n^2\\mathbf{I} & \\mathbf{k}_* \\\\ \\mathbf{k}_*^\\top & k_{**} \\end{pmatrix}\\right),
\\]
<p>where \\(K_{ij}=k(\\mathbf{x}_i,\\mathbf{x}_j)\\), \\((\\mathbf{k}_*)_i=k(\\mathbf{x}_i,\\mathbf{x}_*)\\), and \\(k_{**}=k(\\mathbf{x}_*,\\mathbf{x}_*)\\).</p>

<div class="env-block theorem"><div class="env-title">Theorem 15.2.1 -- GP Posterior</div><div class="env-body">
<p>The predictive distribution is Gaussian:</p>
\\[f_* \\mid \\mathbf{X},\\mathbf{y},\\mathbf{x}_* \\sim \\mathcal{N}(\\bar{f}_*,\\; \\mathbb{V}[f_*]),\\]
\\[\\bar{f}_* = \\mathbf{k}_*^\\top (\\mathbf{K}+\\sigma_n^2\\mathbf{I})^{-1}\\mathbf{y},\\qquad \\mathbb{V}[f_*] = k_{**} - \\mathbf{k}_*^\\top(\\mathbf{K}+\\sigma_n^2\\mathbf{I})^{-1}\\mathbf{k}_*.\\]
</div></div>

<div class="env-block proof"><div class="env-title">Proof</div><div class="env-body">
<p>This follows from the standard formula for the conditional of a joint Gaussian: if \\(\\begin{pmatrix}\\mathbf{a}\\\\\\mathbf{b}\\end{pmatrix} \\sim \\mathcal{N}\\left(\\mathbf{0}, \\begin{pmatrix}A & C\\\\C^\\top & B\\end{pmatrix}\\right)\\), then \\(\\mathbf{b}\\mid\\mathbf{a} \\sim \\mathcal{N}(C^\\top A^{-1}\\mathbf{a},\\; B - C^\\top A^{-1} C)\\). Set \\(\\mathbf{a}=\\mathbf{y}\\), \\(\\mathbf{b}=f_*\\), \\(A=\\mathbf{K}+\\sigma_n^2\\mathbf{I}\\), \\(C=\\mathbf{k}_*\\), \\(B=k_{**}\\).</p>
<div class="qed">&#8718;</div></div></div>

<div class="env-block remark"><div class="env-title">Remark -- Computational Cost</div><div class="env-body">
<p>The dominant cost is inverting \\(\\mathbf{K}+\\sigma_n^2\\mathbf{I}\\), which takes \\(O(N^3)\\). This limits vanilla GP regression to datasets of a few thousand points. Section 15.4 discusses scalable approximations.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-gp-regression"></div>`,
            visualizations: [{
                id: 'viz-gp-regression',
                title: 'GP Regression: Posterior Mean and Uncertainty',
                description: 'Click to add observations. The blue curve shows the posterior mean; shaded bands show \\(\\pm 2\\sigma\\). The GP interpolates data and reverts to the prior (wide bands) far from observations.',
                setup(body, controls) {
                    const viz = new VizEngine(body, {width:680,height:360,scale:55,originX:340,originY:180});
                    const ctx = viz.ctx;
                    let ell = 1.0, sf2 = 1.0, sn2 = 0.05;
                    let data = [];
                    const nTest = 120;
                    const xTest = []; for (let i = 0; i <= nTest; i++) xTest.push(-5 + 10 * i / nTest);

                    function rbf(x1, x2) { return sf2 * Math.exp(-0.5 * (x1-x2)*(x1-x2) / (ell*ell)); }

                    function solve(A, b) {
                        const n = A.length;
                        const aug = A.map((r,i) => [...r, b[i]]);
                        for (let i = 0; i < n; i++) {
                            let mx = i; for (let r = i+1; r < n; r++) if (Math.abs(aug[r][i]) > Math.abs(aug[mx][i])) mx = r;
                            [aug[i], aug[mx]] = [aug[mx], aug[i]];
                            if (Math.abs(aug[i][i]) < 1e-12) continue;
                            for (let r = i+1; r < n; r++) { const f = aug[r][i]/aug[i][i]; for (let c = i; c <= n; c++) aug[r][c] -= f * aug[i][c]; }
                        }
                        const x = new Array(n).fill(0);
                        for (let i = n-1; i >= 0; i--) { let s = aug[i][n]; for (let j = i+1; j < n; j++) s -= aug[i][j]*x[j]; x[i] = s / (aug[i][i] || 1e-12); }
                        return x;
                    }

                    function predict() {
                        const N = data.length;
                        if (N === 0) return xTest.map(() => ({mu: 0, v: sf2}));
                        // Build K + sn2*I
                        const Kn = [];
                        for (let i = 0; i < N; i++) { Kn[i] = []; for (let j = 0; j < N; j++) Kn[i][j] = rbf(data[i].x, data[j].x) + (i===j ? sn2 : 0); }
                        const y = data.map(d => d.y);
                        const alpha = solve(Kn, y);
                        return xTest.map(xs => {
                            const ks = data.map(d => rbf(d.x, xs));
                            let mu = 0; for (let i = 0; i < N; i++) mu += ks[i] * alpha[i];
                            // Variance: k** - ks^T K^-1 ks
                            const v_ks = solve(Kn, ks);
                            let v = rbf(xs, xs); for (let i = 0; i < N; i++) v -= ks[i] * v_ks[i];
                            return {mu, v: Math.max(v, 1e-8)};
                        });
                    }

                    function draw() {
                        viz.clear(); viz.drawGrid(1); viz.drawAxes();
                        const preds = predict();
                        // 2-sigma band
                        ctx.fillStyle = viz.colors.blue + '18'; ctx.beginPath();
                        for (let i = 0; i <= nTest; i++) { const [sx,sy] = viz.toScreen(xTest[i], preds[i].mu + 2*Math.sqrt(preds[i].v)); i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy); }
                        for (let i = nTest; i >= 0; i--) { const [sx,sy] = viz.toScreen(xTest[i], preds[i].mu - 2*Math.sqrt(preds[i].v)); ctx.lineTo(sx,sy); } ctx.fill();
                        // 1-sigma band
                        ctx.fillStyle = viz.colors.blue + '28'; ctx.beginPath();
                        for (let i = 0; i <= nTest; i++) { const [sx,sy] = viz.toScreen(xTest[i], preds[i].mu + Math.sqrt(preds[i].v)); i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy); }
                        for (let i = nTest; i >= 0; i--) { const [sx,sy] = viz.toScreen(xTest[i], preds[i].mu - Math.sqrt(preds[i].v)); ctx.lineTo(sx,sy); } ctx.fill();
                        // Mean
                        ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 2.5; ctx.beginPath();
                        for (let i = 0; i <= nTest; i++) { const [sx,sy] = viz.toScreen(xTest[i], preds[i].mu); i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy); } ctx.stroke();
                        // Data
                        for (const d of data) viz.drawPoint(d.x, d.y, viz.colors.orange, '', 5);
                        viz.screenText('N = ' + data.length, viz.width-80, 20, viz.colors.text, 12);
                    }

                    viz.canvas.addEventListener('click', function(e) {
                        const rect = viz.canvas.getBoundingClientRect();
                        const [mx, my] = viz.toMath(e.clientX - rect.left, e.clientY - rect.top);
                        data.push({x: mx, y: my}); draw();
                    });

                    VizEngine.createSlider(controls, '\u2113', 0.2, 3.0, 1.0, 0.1, v => {ell=v; draw();});
                    VizEngine.createSlider(controls, '\u03c3\u00b2_n', 0.01, 0.5, 0.05, 0.01, v => {sn2=v; draw();});
                    VizEngine.createButton(controls, 'Reset', () => {data=[]; draw();});
                    VizEngine.createButton(controls, 'Demo', () => {
                        const rng = () => {let u=0,v=0;while(!u)u=Math.random();while(!v)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
                        data=[];for(let i=0;i<6;i++){const x=-3+6*Math.random();data.push({x,y:Math.sin(x)+0.3*rng()});}draw();
                    });

                    draw(); return viz;
                }
            }],
            exercises: [
                { question: 'Show that the posterior mean \\(\\bar{f}_* = \\mathbf{k}_*^\\top(\\mathbf{K}+\\sigma_n^2\\mathbf{I})^{-1}\\mathbf{y}\\) is a linear combination of the observations \\(y_i\\).',
                  hint: 'Define \\(\\boldsymbol{\\alpha} = (\\mathbf{K}+\\sigma_n^2\\mathbf{I})^{-1}\\mathbf{y}\\) and write out the sum.',
                  solution: '\\(\\bar{f}_* = \\sum_{i=1}^N \\alpha_i k(\\mathbf{x}_i, \\mathbf{x}_*)\\), where \\(\\alpha_i\\) are the components of \\((\\mathbf{K}+\\sigma_n^2\\mathbf{I})^{-1}\\mathbf{y}\\). The prediction is a kernel-weighted sum of observations. This is the "representer theorem" for GPs.' },
                { question: 'Verify that the posterior variance equals the prior variance at test points far from all training data.',
                  hint: 'What is \\(\\mathbf{k}_*\\) when \\(\\mathbf{x}_*\\) is far from all training inputs?',
                  solution: 'For the RBF kernel, \\(k(\\mathbf{x}_i, \\mathbf{x}_*) \\to 0\\) as \\(\\|\\mathbf{x}_* - \\mathbf{x}_i\\| \\to \\infty\\). So \\(\\mathbf{k}_* \\to \\mathbf{0}\\) and \\(\\mathbb{V}[f_*] = k_{**} - \\mathbf{0}^\\top(\\cdot)^{-1}\\mathbf{0} = k_{**} = \\sigma_f^2\\), the prior variance.' },
                { question: 'In the noise-free case (\\(\\sigma_n^2 = 0\\)), show that the posterior mean interpolates the data exactly.',
                  hint: 'Evaluate \\(\\bar{f}_*\\) at a training input \\(\\mathbf{x}_* = \\mathbf{x}_j\\).',
                  solution: 'At \\(\\mathbf{x}_* = \\mathbf{x}_j\\), \\(\\mathbf{k}_*\\) is the \\(j\\)-th column of \\(\\mathbf{K}\\). So \\(\\bar{f}_j = \\mathbf{K}_{:,j}^\\top \\mathbf{K}^{-1}\\mathbf{y} = \\mathbf{e}_j^\\top \\mathbf{y} = y_j\\). The posterior mean exactly passes through every observation. The posterior variance at training points is also zero.' }
            ]
        },

        // ===================== Section 3: Kernel Choice =====================
        {
            id: 'ch15-sec03',
            title: 'Kernel Choice',
            content: `<h2>15.3 Kernel Choice</h2>
<div class="env-block intuition"><div class="env-title">Intuition</div><div class="env-body">
<p>The kernel encodes our assumptions about the function: how smooth it is, whether it is periodic, how quickly it varies. Choosing a kernel is the GP analog of choosing a model class. Different kernels produce qualitatively different function samples.</p>
</div></div>

<h3>Common Kernels</h3>

<div class="env-block definition"><div class="env-title">Definition 15.3.1 -- Mat&eacute;rn Kernel</div><div class="env-body">
<p>The Mat&eacute;rn kernel with smoothness parameter \\(\\nu\\) is</p>
\\[k_{\\text{Mat}}(r) = \\sigma_f^2 \\frac{2^{1-\\nu}}{\\Gamma(\\nu)}\\left(\\frac{\\sqrt{2\\nu}\\,r}{\\ell}\\right)^\\nu K_\\nu\\!\\left(\\frac{\\sqrt{2\\nu}\\,r}{\\ell}\\right),\\]
<p>where \\(r=\\|\\mathbf{x}-\\mathbf{x}'\\|\\) and \\(K_\\nu\\) is the modified Bessel function. Common special cases:</p>
<ul>
<li>\\(\\nu=1/2\\): \\(k(r)=\\sigma_f^2\\exp(-r/\\ell)\\) (Ornstein-Uhlenbeck; continuous but not differentiable)</li>
<li>\\(\\nu=3/2\\): \\(k(r)=\\sigma_f^2(1+\\frac{\\sqrt{3}r}{\\ell})\\exp(-\\frac{\\sqrt{3}r}{\\ell})\\) (once differentiable)</li>
<li>\\(\\nu=5/2\\): \\(k(r)=\\sigma_f^2(1+\\frac{\\sqrt{5}r}{\\ell}+\\frac{5r^2}{3\\ell^2})\\exp(-\\frac{\\sqrt{5}r}{\\ell})\\) (twice differentiable)</li>
<li>\\(\\nu\\to\\infty\\): recovers the RBF kernel (infinitely differentiable)</li>
</ul></div></div>

<div class="env-block definition"><div class="env-title">Definition 15.3.2 -- Periodic Kernel</div><div class="env-body">
\\[k_{\\text{per}}(x,x') = \\sigma_f^2 \\exp\\!\\left(-\\frac{2\\sin^2(\\pi|x-x'|/p)}{\\ell^2}\\right),\\]
<p>where \\(p\\) is the period. This kernel generates functions that repeat with period \\(p\\).</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark -- Kernel Composition</div><div class="env-body">
<p>Kernels can be composed: sums of kernels model additive structure, products model interactions. For example, \\(k_{\\text{SE}} \\times k_{\\text{per}}\\) produces locally periodic functions that decay away from the origin.</p>
</div></div>

<h3>Hyperparameter Learning</h3>
<p>Kernel hyperparameters \\(\\boldsymbol{\\theta} = (\\ell, \\sigma_f^2, \\sigma_n^2, \\ldots)\\) are selected by maximizing the log marginal likelihood:</p>
\\[\\ln p(\\mathbf{y}\\mid\\mathbf{X},\\boldsymbol{\\theta}) = -\\tfrac{1}{2}\\mathbf{y}^\\top(\\mathbf{K}_\\theta+\\sigma_n^2\\mathbf{I})^{-1}\\mathbf{y} - \\tfrac{1}{2}\\ln|\\mathbf{K}_\\theta+\\sigma_n^2\\mathbf{I}| - \\tfrac{N}{2}\\ln 2\\pi.\\]
<p>This is the same Bayesian model comparison principle from Ch. 14, now applied to continuous hyperparameters via gradient-based optimization.</p>

<div class="viz-placeholder" data-viz="viz-kernel-choice"></div>`,
            visualizations: [{
                id: 'viz-kernel-choice',
                title: 'GP Samples with Different Kernels',
                description: 'Toggle between RBF, Mat\u00e9rn-3/2, Mat\u00e9rn-1/2, and Periodic kernels. Adjust the length-scale to see its effect on sample smoothness. Click "Resample" for fresh draws.',
                setup(body, controls) {
                    const viz = new VizEngine(body, {width:680,height:360,scale:55,originX:340,originY:180});
                    const ctx = viz.ctx;
                    let ell = 1.0, sf2 = 1.0, kernelType = 'rbf', period = 2.0;
                    const nX = 100;
                    const xs = []; for (let i = 0; i <= nX; i++) xs.push(-5 + 10 * i / nX);
                    let samples = [];
                    const colors = [viz.colors.blue, viz.colors.teal, viz.colors.orange, viz.colors.purple, viz.colors.pink];

                    function rng() { let u=0,v=0; while(!u) u=Math.random(); while(!v) v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

                    function kernel(x1, x2) {
                        const r = Math.abs(x1 - x2);
                        if (kernelType === 'rbf') return sf2 * Math.exp(-0.5 * r * r / (ell * ell));
                        if (kernelType === 'matern32') { const s = Math.sqrt(3) * r / ell; return sf2 * (1 + s) * Math.exp(-s); }
                        if (kernelType === 'matern12') return sf2 * Math.exp(-r / ell);
                        if (kernelType === 'periodic') return sf2 * Math.exp(-2 * Math.sin(Math.PI * r / period) ** 2 / (ell * ell));
                        return sf2 * Math.exp(-0.5 * r * r / (ell * ell));
                    }

                    function cholesky(A) {
                        const n = A.length, L = [];
                        for (let i = 0; i < n; i++) L[i] = new Array(n).fill(0);
                        for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
                            let s = A[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
                            L[i][j] = (i === j) ? Math.sqrt(Math.max(s, 1e-8)) : s / L[j][j];
                        }
                        return L;
                    }

                    function sampleGP() {
                        const n = xs.length, K = [];
                        for (let i = 0; i < n; i++) { K[i] = []; for (let j = 0; j < n; j++) K[i][j] = kernel(xs[i], xs[j]) + (i===j ? 1e-6 : 0); }
                        const L = cholesky(K);
                        samples = [];
                        for (let s = 0; s < 4; s++) {
                            const z = []; for (let i = 0; i < n; i++) z.push(rng());
                            const f = new Array(n).fill(0);
                            for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) f[i] += L[i][j] * z[j];
                            samples.push(f);
                        }
                    }

                    function draw() {
                        viz.clear(); viz.drawGrid(1); viz.drawAxes();
                        for (let s = 0; s < samples.length; s++) {
                            ctx.strokeStyle = colors[s]; ctx.lineWidth = 2; ctx.beginPath();
                            for (let i = 0; i <= nX; i++) {
                                const [sx, sy] = viz.toScreen(xs[i], samples[s][i]);
                                i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                            }
                            ctx.stroke();
                        }
                        const names = {rbf:'RBF (SE)', matern32:'Mat\u00e9rn-3/2', matern12:'Mat\u00e9rn-1/2', periodic:'Periodic'};
                        viz.screenText(names[kernelType], 100, 20, viz.colors.white, 13, 'left');
                        viz.screenText('\u2113 = ' + ell.toFixed(1), viz.width-80, 20, viz.colors.teal, 12);
                    }

                    const kernels = ['rbf','matern32','matern12','periodic'];
                    const labels = ['RBF','Mat\u00e9rn-3/2','Mat\u00e9rn-1/2','Periodic'];
                    kernels.forEach((k, i) => {
                        const btn = VizEngine.createButton(controls, labels[i], () => { kernelType = k; sampleGP(); draw(); });
                        if (i === 0) btn.style.background = '#2a2a50';
                        btn.addEventListener('click', () => {
                            controls.querySelectorAll('button').forEach(b => b.style.background = '#1a1a40');
                            btn.style.background = '#2a2a50';
                        });
                    });
                    VizEngine.createSlider(controls, '\u2113', 0.2, 3.0, 1.0, 0.1, v => { ell = v; sampleGP(); draw(); });
                    VizEngine.createButton(controls, 'Resample', () => { sampleGP(); draw(); });

                    sampleGP(); draw(); return viz;
                }
            }],
            exercises: [
                { question: 'Show that the Mat\u00e9rn-\\(\\nu\\) kernel converges to the RBF kernel as \\(\\nu \\to \\infty\\).',
                  hint: 'Use the asymptotic expansion of \\(K_\\nu(z)\\) for large \\(\\nu\\): \\(\\frac{2^{1-\\nu}}{\\Gamma(\\nu)}(\\sqrt{2\\nu}z)^\\nu K_\\nu(\\sqrt{2\\nu}z) \\to \\exp(-z^2/2)\\).',
                  solution: 'As \\(\\nu \\to \\infty\\), the Mat\u00e9rn kernel satisfies \\(k_{\\text{Mat}}(r) \\to \\sigma_f^2 \\exp(-r^2/(2\\ell^2))\\), which is exactly the RBF kernel. The Mat\u00e9rn family thus interpolates between rough (\\(\\nu=1/2\\), exponential) and infinitely smooth (\\(\\nu\\to\\infty\\), Gaussian) functions.' },
                { question: 'If \\(k_1\\) and \\(k_2\\) are valid kernels, prove that \\(k_1 + k_2\\) and \\(k_1 \\cdot k_2\\) are also valid kernels.',
                  hint: 'A kernel is valid iff the Gram matrix is PSD for all input sets. Use properties of PSD matrices.',
                  solution: 'Sum: \\(\\mathbf{K}_{k_1+k_2} = \\mathbf{K}_{k_1} + \\mathbf{K}_{k_2}\\). Sum of PSD matrices is PSD. Product: \\(\\mathbf{K}_{k_1 \\cdot k_2} = \\mathbf{K}_{k_1} \\circ \\mathbf{K}_{k_2}\\) (Hadamard product). By the Schur product theorem, the Hadamard product of PSD matrices is PSD.' },
                { question: 'How does the length-scale \\(\\ell\\) affect the marginal likelihood? When is a short length-scale penalized?',
                  hint: 'Consider the complexity term \\(-\\frac{1}{2}\\ln|\\mathbf{K}+\\sigma_n^2\\mathbf{I}|\\).',
                  solution: 'A short \\(\\ell\\) makes \\(\\mathbf{K}\\) nearly diagonal (low correlation), so the model can fit any data pattern (good data fit) but the effective number of parameters is large (large \\(\\ln|\\mathbf{K}+\\sigma_n^2\\mathbf{I}|\\), heavy complexity penalty). The marginal likelihood penalizes unnecessarily short \\(\\ell\\), preferring the longest length-scale consistent with the data, embodying Occam\'s razor.' }
            ]
        },

        // ===================== Section 4: GP Classification & Scalability =====================
        {
            id: 'ch15-sec04',
            title: 'GP Classification & Scalability',
            content: `<h2>15.4 GP Classification &amp; Scalability</h2>
<div class="env-block intuition"><div class="env-title">Intuition</div><div class="env-body">
<p>For classification, we pass the GP output through a sigmoid to get class probabilities. Unlike regression, the posterior is no longer Gaussian, requiring approximations. For large datasets, we replace full \\(O(N^3)\\) computation with inducing-point methods that scale to millions of observations.</p>
</div></div>

<h3>GP Classification</h3>
<p>For binary classification with labels \\(y \\in \\{0, 1\\}\\), we model</p>
\\[p(y=1 \\mid f) = \\sigma(f), \\qquad f \\sim \\mathcal{GP}(0, k),\\]
<p>where \\(\\sigma(\\cdot)\\) is the logistic sigmoid. The likelihood \\(p(y\\mid f) = \\sigma(f)^y(1-\\sigma(f))^{1-y}\\) is not Gaussian, so the posterior \\(p(\\mathbf{f}\\mid\\mathbf{y})\\) is intractable.</p>

<div class="env-block definition"><div class="env-title">Definition 15.4.1 -- Laplace Approximation for GP Classification</div><div class="env-body">
<p>The Laplace approximation finds the mode \\(\\hat{\\mathbf{f}} = \\arg\\max_\\mathbf{f}\\, p(\\mathbf{f}\\mid\\mathbf{y})\\) by Newton's method, then approximates the posterior as</p>
\\[p(\\mathbf{f}\\mid\\mathbf{y}) \\approx \\mathcal{N}(\\mathbf{f};\\, \\hat{\\mathbf{f}},\\, (\\mathbf{K}^{-1}+\\mathbf{W})^{-1}),\\]
<p>where \\(\\mathbf{W} = -\\nabla\\nabla \\ln p(\\mathbf{y}\\mid\\mathbf{f})\\big|_{\\hat{\\mathbf{f}}}\\) is the diagonal matrix of negative second derivatives of the log-likelihood.</p>
</div></div>

<h3>Scalability: Inducing Points</h3>
<p>For \\(N\\) training points, exact GP inference costs \\(O(N^3)\\). Inducing-point methods reduce this by selecting \\(M \\ll N\\) <em>inducing points</em> \\(\\mathbf{Z} = \\{\\mathbf{z}_1,\\ldots,\\mathbf{z}_M\\}\\) and approximating the covariance as</p>
\\[\\mathbf{K} \\approx \\mathbf{K}_{NM} \\mathbf{K}_{MM}^{-1} \\mathbf{K}_{MN},\\]
<p>where \\(\\mathbf{K}_{NM}\\) is the \\(N \\times M\\) cross-covariance between training and inducing points.</p>

<div class="env-block definition"><div class="env-title">Definition 15.4.2 -- Sparse GP (FITC/VFE)</div><div class="env-body">
<p>The <strong>Fully Independent Training Conditional (FITC)</strong> approximation replaces the prior covariance with</p>
\\[\\mathbf{Q}_{\\text{FITC}} = \\mathbf{K}_{NM}\\mathbf{K}_{MM}^{-1}\\mathbf{K}_{MN} + \\text{diag}(\\mathbf{K} - \\mathbf{K}_{NM}\\mathbf{K}_{MM}^{-1}\\mathbf{K}_{MN}).\\]
<p>The <strong>Variational Free Energy (VFE)</strong> approach (Titsias, 2009) treats inducing point locations as variational parameters and optimizes them jointly with hyperparameters. Both reduce cost to \\(O(NM^2)\\).</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark -- Modern GP Scaling</div><div class="env-body">
<p>With GPU-accelerated frameworks (GPyTorch), stochastic variational inference, and structured kernel interpolation (KISS-GP), GPs can scale to datasets of millions. The key insight is that inducing points summarize the training data, and their locations can be learned.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-gp-classification"></div>`,
            visualizations: [{
                id: 'viz-gp-classification',
                title: 'GP Classification with Uncertainty',
                description: 'Click to add class-0 (left-click) or class-1 (right-click / shift-click) points. The color map shows the predicted probability \\(p(y=1)\\), with contours at 0.25, 0.5, 0.75. Bright regions indicate confident predictions.',
                setup(body, controls) {
                    const viz = new VizEngine(body, {width:680,height:360,scale:55,originX:340,originY:180});
                    const ctx = viz.ctx;
                    let ell = 1.0, sf2 = 1.5;
                    let data = []; // {x, y, label}

                    function rbf(x1,y1,x2,y2) { return sf2 * Math.exp(-0.5*((x1-x2)**2+(y1-y2)**2)/(ell*ell)); }
                    function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

                    function solve(A, b) {
                        const n = A.length;
                        const aug = A.map((r,i) => [...r, b[i]]);
                        for (let i = 0; i < n; i++) {
                            let mx = i; for (let r = i+1; r < n; r++) if (Math.abs(aug[r][i]) > Math.abs(aug[mx][i])) mx = r;
                            [aug[i], aug[mx]] = [aug[mx], aug[i]];
                            if (Math.abs(aug[i][i]) < 1e-12) continue;
                            for (let r = i+1; r < n; r++) { const f = aug[r][i]/aug[i][i]; for (let c = i; c <= n; c++) aug[r][c] -= f * aug[i][c]; }
                        }
                        const x = new Array(n).fill(0);
                        for (let i = n-1; i >= 0; i--) { let s = aug[i][n]; for (let j = i+1; j < n; j++) s -= aug[i][j]*x[j]; x[i] = s / (aug[i][i] || 1e-12); }
                        return x;
                    }

                    function laplacePredict() {
                        const N = data.length;
                        if (N === 0) return null;
                        // Build K
                        const K = [];
                        for (let i = 0; i < N; i++) { K[i] = []; for (let j = 0; j < N; j++) K[i][j] = rbf(data[i].x, data[i].y, data[j].x, data[j].y) + (i===j ? 1e-4 : 0); }
                        // Newton iterations for Laplace mode
                        let f = new Array(N).fill(0);
                        for (let iter = 0; iter < 8; iter++) {
                            const pi = f.map(sigmoid);
                            const W = pi.map(p => p * (1 - p) + 1e-6);
                            const grad = data.map((d, i) => d.label - pi[i]);
                            // Solve (K^-1 + W) delta = grad + W*f - K^-1*f
                            const Kinv_f = solve(K, f);
                            const rhs = grad.map((g, i) => g + W[i] * f[i] - Kinv_f[i]);
                            // Build K^-1 + diag(W)
                            const Kinv = [];
                            for (let i = 0; i < N; i++) {
                                const ei = new Array(N).fill(0); ei[i] = 1;
                                Kinv.push(solve(K, ei));
                            }
                            const A = Kinv.map((r, i) => r.map((v, j) => v + (i === j ? W[i] : 0)));
                            f = solve(A, rhs);
                        }
                        return { fHat: f, K };
                    }

                    function predictPoint(result, tx, ty) {
                        if (!result) return 0.5;
                        const N = data.length;
                        const ks = data.map(d => rbf(d.x, d.y, tx, ty));
                        const alpha = solve(result.K, result.fHat);
                        let mu = 0; for (let i = 0; i < N; i++) mu += ks[i] * alpha[i];
                        return sigmoid(mu);
                    }

                    function draw() {
                        viz.clear();
                        const result = laplacePredict();
                        // Heatmap
                        const res = 6;
                        for (let px = 0; px < viz.width; px += res) {
                            for (let py = 0; py < viz.height; py += res) {
                                const [mx, my] = viz.toMath(px + res/2, py + res/2);
                                const prob = predictPoint(result, mx, my);
                                const r = Math.round(50 + 180 * prob);
                                const b = Math.round(50 + 180 * (1 - prob));
                                const g = Math.round(50 + 80 * (1 - Math.abs(prob - 0.5) * 2));
                                ctx.fillStyle = 'rgb('+r+','+g+','+b+')';
                                ctx.fillRect(px, py, res, res);
                            }
                        }
                        // Contours at 0.25, 0.5, 0.75
                        viz.drawGrid(1);
                        ctx.strokeStyle = viz.colors.white + '55'; ctx.lineWidth = 1;
                        // Data points
                        for (const d of data) {
                            const color = d.label === 1 ? viz.colors.orange : viz.colors.blue;
                            viz.drawPoint(d.x, d.y, color, '', 6);
                            const [sx, sy] = viz.toScreen(d.x, d.y);
                            ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
                            ctx.beginPath(); ctx.arc(sx, sy, 6, 0, 2*Math.PI); ctx.stroke();
                        }
                        viz.screenText('Left-click: class 0 (blue)', 10, 20, viz.colors.blue, 11, 'left');
                        viz.screenText('Shift-click: class 1 (orange)', 10, 36, viz.colors.orange, 11, 'left');
                        viz.screenText('N = ' + data.length, viz.width - 60, 20, viz.colors.white, 12);
                    }

                    viz.canvas.addEventListener('click', function(e) {
                        const rect = viz.canvas.getBoundingClientRect();
                        const [mx, my] = viz.toMath(e.clientX - rect.left, e.clientY - rect.top);
                        const label = e.shiftKey ? 1 : 0;
                        data.push({x: mx, y: my, label}); draw();
                    });
                    viz.canvas.addEventListener('contextmenu', function(e) {
                        e.preventDefault();
                        const rect = viz.canvas.getBoundingClientRect();
                        const [mx, my] = viz.toMath(e.clientX - rect.left, e.clientY - rect.top);
                        data.push({x: mx, y: my, label: 1}); draw();
                    });

                    VizEngine.createSlider(controls, '\u2113', 0.3, 3.0, 1.0, 0.1, v => {ell=v; draw();});
                    VizEngine.createButton(controls, 'Reset', () => {data=[]; draw();});
                    VizEngine.createButton(controls, 'Demo', () => {
                        const rng = () => {let u=0,v=0;while(!u)u=Math.random();while(!v)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
                        data = [];
                        for (let i = 0; i < 8; i++) data.push({x: -1.5+rng()*0.6, y: 1+rng()*0.6, label: 0});
                        for (let i = 0; i < 8; i++) data.push({x: 1.5+rng()*0.6, y: -0.5+rng()*0.6, label: 1});
                        draw();
                    });

                    draw(); return viz;
                }
            }],
            exercises: [
                { question: 'Why can\'t we use exact Bayesian inference for GP classification, unlike GP regression?',
                  hint: 'Consider the form of the likelihood in classification vs. regression.',
                  solution: 'In regression, the likelihood \\(p(\\mathbf{y}\\mid\\mathbf{f})\\) is Gaussian. Combined with the Gaussian GP prior, the posterior is also Gaussian (conjugacy). In classification, \\(p(y\\mid f) = \\sigma(f)^y(1-\\sigma(f))^{1-y}\\) is not Gaussian. The product of this non-Gaussian likelihood with the Gaussian prior yields a non-Gaussian posterior, so we need approximations (Laplace, EP, or variational methods).' },
                { question: 'In the FITC approximation, why is the diagonal correction \\(\\text{diag}(\\mathbf{K} - \\mathbf{Q})\\) added?',
                  hint: 'Without it, the variance at each training point would be underestimated.',
                  solution: 'The low-rank approximation \\(\\mathbf{Q} = \\mathbf{K}_{NM}\\mathbf{K}_{MM}^{-1}\\mathbf{K}_{MN}\\) underestimates the diagonal of \\(\\mathbf{K}\\): \\(Q_{ii} \\leq K_{ii}\\) by positive semi-definiteness. Adding \\(\\text{diag}(\\mathbf{K}-\\mathbf{Q})\\) restores the correct marginal variance at each training point while keeping the off-diagonal low-rank structure. This prevents the approximation from being overconfident.' },
                { question: 'If we have \\(N=10^6\\) training points and \\(M=500\\) inducing points, what is the computational speedup of sparse GP over exact GP?',
                  hint: 'Compare \\(O(N^3)\\) with \\(O(NM^2)\\).',
                  solution: 'Exact: \\(O(N^3) = O(10^{18})\\). Sparse: \\(O(NM^2) = O(10^6 \\times 500^2) = O(2.5 \\times 10^{11})\\). The speedup factor is \\(10^{18}/(2.5 \\times 10^{11}) = 4 \\times 10^6\\), roughly six orders of magnitude. This makes the problem tractable on modern hardware.' }
            ]
        }
    ]
});
