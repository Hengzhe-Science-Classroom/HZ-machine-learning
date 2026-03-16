window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch03',
    number: 3,
    title: 'Bias-Variance & Model Selection',
    subtitle: 'Understanding Generalization, Choosing Models, and Controlling Complexity',
    sections: [

        // ===== SECTION 1: Bias-Variance Tradeoff =====
        {
            id: 'bias-variance',
            title: 'Bias-Variance Tradeoff',
            content: `
<div class="env-block intuition"><div class="env-title">The Central Tension in Learning</div><div class="env-body">
<p>Every predictive model faces a fundamental dilemma. A model that is too simple cannot capture the true pattern in the data; it <em>underfits</em>. A model that is too complex memorizes the idiosyncrasies of the training set and fails on new data; it <em>overfits</em>. The bias-variance decomposition formalizes this tension, revealing that prediction error is the sum of three irreducible components. Understanding this decomposition is the foundation of all model selection and regularization strategies.</p>
</div></div>

<h2>The Bias-Variance Decomposition</h2>

<p>Suppose the true data-generating process is \\(y = f(x) + \\epsilon\\), where \\(\\epsilon\\) is zero-mean noise with variance \\(\\sigma^2\\). We train a model \\(\\hat{f}(x)\\) on a random dataset \\(D\\). Different training sets yield different fitted models, so \\(\\hat{f}\\) is a random variable.</p>

<div class="env-block theorem"><div class="env-title">Theorem 3.1 (Bias-Variance Decomposition)</div><div class="env-body">
<p>For squared loss, the expected prediction error at a point \\(x\\) decomposes as:</p>
\\[\\mathbb{E}_D\\!\\left[(y - \\hat{f}(x))^2\\right] = \\underbrace{\\bigl(f(x) - \\mathbb{E}_D[\\hat{f}(x)]\\bigr)^2}_{\\text{Bias}^2} + \\underbrace{\\mathbb{E}_D\\!\\left[(\\hat{f}(x) - \\mathbb{E}_D[\\hat{f}(x)])^2\\right]}_{\\text{Variance}} + \\underbrace{\\sigma^2}_{\\text{Irreducible noise}}\\]
</div></div>

<div class="env-block proof"><div class="env-title">Derivation</div><div class="env-body">
<p>Let \\(\\bar{f}(x) = \\mathbb{E}_D[\\hat{f}(x)]\\). Since \\(\\epsilon\\) is independent of \\(\\hat{f}\\):</p>
\\[\\mathbb{E}[(y - \\hat{f})^2] = \\mathbb{E}[(f + \\epsilon - \\hat{f})^2] = \\mathbb{E}[(f - \\hat{f})^2] + \\sigma^2\\]
<p>Adding and subtracting \\(\\bar{f}\\) inside the first term and expanding:</p>
\\[\\mathbb{E}[(f - \\hat{f})^2] = (f - \\bar{f})^2 + \\mathbb{E}[(\\hat{f} - \\bar{f})^2]\\]
<p>The cross-term vanishes because \\(\\mathbb{E}[\\hat{f} - \\bar{f}] = 0\\). <span class="qed">\\(\\blacksquare\\)</span></p>
</div></div>

<h3>Interpreting Each Term</h3>

<ul>
  <li><strong>Bias\\(^2\\)</strong>: How far the <em>average</em> prediction is from the truth. High bias means the model class cannot represent \\(f\\), regardless of data.</li>
  <li><strong>Variance</strong>: How much predictions fluctuate across different training sets. High variance means the model is overly sensitive to the particular sample.</li>
  <li><strong>Irreducible noise</strong> \\(\\sigma^2\\): A lower bound on error that no model can beat.</li>
</ul>

<h3>Model Complexity and the U-Shaped Curve</h3>

<p>As model complexity increases (polynomial degree, number of parameters, tree depth):</p>
<ul>
  <li><strong>Bias decreases</strong>: more flexible models approximate \\(f(x)\\) more closely on average.</li>
  <li><strong>Variance increases</strong>: more flexible models are more sensitive to the training data.</li>
  <li><strong>Total test error</strong> follows a U-shape, first falling as bias drops, then rising as variance dominates.</li>
</ul>

<div class="viz-placeholder" data-viz="viz-bias-variance-curves"></div>

<div class="env-block definition"><div class="env-title">Definition 3.1 (Underfitting and Overfitting)</div><div class="env-body">
<p><strong>Underfitting</strong> occurs when a model is too simple to capture the underlying structure. Both training and test errors are high (high bias, low variance).</p>
<p><strong>Overfitting</strong> occurs when a model fits noise in the training data. Training error is low but test error is high (low bias, high variance).</p>
</div></div>

<div class="env-block example"><div class="env-title">Example 3.1 (Polynomial Regression)</div><div class="env-body">
<p>Fitting a degree-1 polynomial to a cubic target \\(f(x) = x^3 - 2x\\) yields high bias: the line cannot capture curvature. A degree-15 polynomial can fit any 16 points exactly, but oscillates wildly between them (high variance). A degree-3 polynomial is just right: low bias and controlled variance.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Double Descent</div><div class="env-body">
<p>In overparameterized models (e.g., deep networks), test error sometimes decreases again beyond the interpolation threshold, producing a <em>double descent</em> curve. This does not invalidate the decomposition; it reveals that implicit regularization from optimization can control variance even with more parameters than data points.</p>
</div></div>
`,
            visualizations: [
                {
                    id: 'viz-bias-variance-curves',
                    title: 'Bias-Variance Tradeoff: Error Decomposition',
                    description: 'Drag the complexity slider to see how bias\u00b2, variance, and total error change. The vertical dashed line marks the optimal complexity. Left panel shows polynomial fits on sample data; right panel shows the decomposition.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { width: 780, height: 370, scale: 1, originX: 0, originY: 0 });
                        const ctx = viz.ctx;
                        const W = viz.width, H = viz.height;
                        let complexity = 3;

                        function seededRandom(s) { let x = Math.sin(s) * 10000; return x - Math.floor(x); }
                        function trueF(x) { return Math.sin(1.8 * x) * 1.5 + 0.3 * x; }

                        const nTrain = 15;
                        const trainX = [], trainY = [];
                        for (let i = 0; i < nTrain; i++) {
                            const x = -2.5 + 5 * i / (nTrain - 1);
                            trainX.push(x);
                            trainY.push(trueF(x) + (seededRandom(42 + i * 7) - 0.5) * 1.8);
                        }

                        function polyFit(xs, ys, deg) {
                            const n = xs.length, d = deg + 1;
                            const X = xs.map(x => { const r = []; for (let j = 0; j < d; j++) r.push(Math.pow(x, j)); return r; });
                            const XtX = [], Xty = [];
                            for (let i = 0; i < d; i++) {
                                XtX.push([]);
                                let s = 0;
                                for (let k = 0; k < n; k++) s += X[k][i] * ys[k];
                                Xty.push(s);
                                for (let j = 0; j < d; j++) {
                                    let v = 0;
                                    for (let k = 0; k < n; k++) v += X[k][i] * X[k][j];
                                    XtX[i].push(v + (i === j ? 1e-8 : 0));
                                }
                            }
                            const A = XtX.map((r, i) => r.concat([Xty[i]]));
                            for (let c = 0; c < d; c++) {
                                let mr = c;
                                for (let r = c + 1; r < d; r++) if (Math.abs(A[r][c]) > Math.abs(A[mr][c])) mr = r;
                                [A[c], A[mr]] = [A[mr], A[c]];
                                if (Math.abs(A[c][c]) < 1e-14) continue;
                                for (let r = c + 1; r < d; r++) { const f = A[r][c] / A[c][c]; for (let j = c; j <= d; j++) A[r][j] -= f * A[c][j]; }
                            }
                            const co = new Array(d);
                            for (let i = d - 1; i >= 0; i--) { co[i] = A[i][d]; for (let j = i + 1; j < d; j++) co[i] -= A[i][j] * co[j]; co[i] /= A[i][i]; }
                            return co;
                        }
                        function polyEval(co, x) { let y = 0; for (let i = 0; i < co.length; i++) y += co[i] * Math.pow(x, i); return y; }

                        function computeErrors() {
                            const biasArr = [], varArr = [], totalArr = [];
                            const nSim = 30, nTest = 80;
                            for (let deg = 1; deg <= 12; deg++) {
                                const preds = [];
                                for (let t = 0; t < nTest; t++) preds.push([]);
                                for (let s = 0; s < nSim; s++) {
                                    const sy = trainX.map((x, i) => trueF(x) + (seededRandom(s * 100 + i * 13 + 3) - 0.5) * 1.8);
                                    const co = polyFit(trainX, sy, deg);
                                    for (let t = 0; t < nTest; t++) {
                                        const tx = -2.5 + 5 * t / (nTest - 1);
                                        preds[t].push(polyEval(co, tx));
                                    }
                                }
                                let bias2 = 0, vari = 0;
                                for (let t = 0; t < nTest; t++) {
                                    const tx = -2.5 + 5 * t / (nTest - 1);
                                    const mean = preds[t].reduce((a, b) => a + b, 0) / nSim;
                                    bias2 += Math.pow(trueF(tx) - mean, 2);
                                    let v2 = 0;
                                    for (let s = 0; s < nSim; s++) v2 += Math.pow(preds[t][s] - mean, 2);
                                    vari += v2 / nSim;
                                }
                                biasArr.push(bias2 / nTest);
                                varArr.push(vari / nTest);
                                totalArr.push(bias2 / nTest + vari / nTest + 0.27);
                            }
                            return { bias: biasArr, variance: varArr, total: totalArr };
                        }
                        const errs = computeErrors();

                        VizEngine.createSlider(controls, 'Degree', 1, 12, complexity, 1, v => { complexity = Math.round(v); draw(); });

                        function draw() {
                            viz.clear();
                            const leftW = W * 0.48, rightW = W - leftW;
                            const m = { top: 35, bottom: 35, left: 45, right: 15 };
                            const pW = leftW - m.left - m.right, pH = H - m.top - m.bottom;
                            const xMin = -3, xMax = 3, yMin = -5, yMax = 5;
                            const toL = (x, y) => [m.left + (x - xMin) / (xMax - xMin) * pW, m.top + (1 - (y - yMin) / (yMax - yMin)) * pH];

                            // Left grid
                            ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                            for (let gx = -2; gx <= 2; gx++) { const [px] = toL(gx, 0); ctx.beginPath(); ctx.moveTo(px, m.top); ctx.lineTo(px, m.top + pH); ctx.stroke(); }
                            for (let gy = -4; gy <= 4; gy += 2) { const [, py] = toL(0, gy); ctx.beginPath(); ctx.moveTo(m.left, py); ctx.lineTo(m.left + pW, py); ctx.stroke(); }
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(m.left, m.top + pH); ctx.lineTo(m.left + pW, m.top + pH); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + pH); ctx.stroke();

                            // True function
                            ctx.strokeStyle = viz.colors.green; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.beginPath();
                            for (let i = 0; i <= 200; i++) { const x = xMin + (xMax - xMin) * i / 200; const [px, py] = toL(x, trueF(x)); i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }
                            ctx.stroke(); ctx.setLineDash([]);

                            // Polynomial fit
                            const co = polyFit(trainX, trainY, complexity);
                            ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 2.5; ctx.beginPath();
                            for (let i = 0; i <= 300; i++) { const x = xMin + (xMax - xMin) * i / 300; let y = polyEval(co, x); y = Math.max(yMin - 1, Math.min(yMax + 1, y)); const [px, py] = toL(x, y); i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }
                            ctx.stroke();

                            // Data points
                            for (let i = 0; i < nTrain; i++) { const [px, py] = toL(trainX[i], trainY[i]); ctx.fillStyle = viz.colors.orange; ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill(); }

                            // Labels
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            ctx.fillText('Degree ' + complexity + ' fit', m.left + pW / 2, 18);
                            ctx.fillStyle = viz.colors.green; ctx.fillText('true f(x)', m.left + pW - 30, 18);

                            // === RIGHT PANEL: Error curves ===
                            const rLeft = leftW + 40, rW = rightW - 60, rTop = m.top, rH = pH;
                            const maxE = Math.max(...errs.total, ...errs.bias, ...errs.variance) * 1.15;
                            const toR = (d, e) => [rLeft + (d - 1) / 11 * rW, rTop + (1 - e / maxE) * rH];

                            ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                            for (let d = 1; d <= 12; d += 2) { const [px] = toR(d, 0); ctx.beginPath(); ctx.moveTo(px, rTop); ctx.lineTo(px, rTop + rH); ctx.stroke(); }
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(rLeft, rTop + rH); ctx.lineTo(rLeft + rW, rTop + rH); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(rLeft, rTop); ctx.lineTo(rLeft, rTop + rH); ctx.stroke();

                            // Curves
                            const drawCurve = (arr, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); arr.forEach((v, i) => { const [px, py] = toR(i + 1, v); i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); }); ctx.stroke(); };
                            drawCurve(errs.bias, viz.colors.teal);
                            drawCurve(errs.variance, viz.colors.orange);
                            drawCurve(errs.total, viz.colors.red);

                            // Noise floor
                            ctx.strokeStyle = viz.colors.text; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
                            const [, ny] = toR(1, 0.27); ctx.beginPath(); ctx.moveTo(rLeft, ny); ctx.lineTo(rLeft + rW, ny); ctx.stroke(); ctx.setLineDash([]);

                            // Current complexity marker
                            const [cx] = toR(complexity, 0);
                            ctx.strokeStyle = viz.colors.white; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
                            ctx.beginPath(); ctx.moveTo(cx, rTop); ctx.lineTo(cx, rTop + rH); ctx.stroke(); ctx.setLineDash([]);

                            // Optimal marker
                            let minIdx = 0; errs.total.forEach((v, i) => { if (v < errs.total[minIdx]) minIdx = i; });
                            const [ox] = toR(minIdx + 1, 0);
                            ctx.strokeStyle = viz.colors.green; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
                            ctx.beginPath(); ctx.moveTo(ox, rTop); ctx.lineTo(ox, rTop + rH); ctx.stroke(); ctx.setLineDash([]);

                            // Legend and axis labels
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            ctx.fillText('Model Complexity (degree)', rLeft + rW / 2, H - 5);
                            for (let d = 1; d <= 12; d += 2) { const [px] = toR(d, 0); ctx.fillText(d, px, rTop + rH + 14); }

                            ctx.textAlign = 'left'; ctx.font = '10px -apple-system,sans-serif';
                            const ly = rTop + 10;
                            ctx.fillStyle = viz.colors.teal; ctx.fillText('Bias\u00b2', rLeft + 4, ly);
                            ctx.fillStyle = viz.colors.orange; ctx.fillText('Variance', rLeft + 46, ly);
                            ctx.fillStyle = viz.colors.red; ctx.fillText('Total Error', rLeft + 104, ly);
                            ctx.fillStyle = viz.colors.text; ctx.fillText('\u03c3\u00b2 noise', rLeft + rW - 45, ny - 6);
                        }
                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Prove that the cross-term in the bias-variance derivation vanishes. That is, show \\(\\mathbb{E}_D\\!\\left[(f(x) - \\bar{f}(x))(\\bar{f}(x) - \\hat{f}(x))\\right] = 0\\) where \\(\\bar{f}(x) = \\mathbb{E}_D[\\hat{f}(x)]\\).',
                    hint: 'The term \\(f(x) - \\bar{f}(x)\\) is a constant with respect to the expectation over \\(D\\).',
                    solution: 'Since \\(f(x) - \\bar{f}(x)\\) does not depend on \\(D\\), it factors out: \\(\\mathbb{E}_D[(f - \\bar{f})(\\bar{f} - \\hat{f})] = (f - \\bar{f})\\,\\mathbb{E}_D[\\bar{f} - \\hat{f}] = (f - \\bar{f}) \\cdot 0 = 0\\), because \\(\\mathbb{E}_D[\\hat{f}] = \\bar{f}\\) by definition.'
                },
                {
                    question: 'A nearest-neighbor classifier with \\(k=1\\) has zero training error. Explain this in terms of bias and variance. What happens to the decomposition as \\(k \\to n\\)?',
                    hint: 'Think about what 1-NN predicts at a training point, and what happens when you average over all training points.',
                    solution: '<strong>\\(k=1\\):</strong> At each training point, the prediction equals the observed label, so training error is zero. Bias is low (the model can capture any local pattern), but variance is very high (the prediction depends entirely on one data point). <strong>\\(k \\to n\\):</strong> The prediction at every point converges to the global average \\(\\bar{y}\\). Variance drops to near zero, but bias is maximal because the model ignores all local structure. The optimal \\(k\\) balances these two extremes.'
                },
                {
                    question: 'For ridge regression \\(\\hat{\\boldsymbol{\\beta}}_\\lambda = (\\mathbf{X}^\\top \\mathbf{X} + \\lambda \\mathbf{I})^{-1}\\mathbf{X}^\\top \\mathbf{y}\\), show that increasing \\(\\lambda\\) increases bias and decreases variance.',
                    hint: 'Express \\(\\mathbb{E}[\\hat{\\boldsymbol{\\beta}}_\\lambda]\\) and \\(\\text{Cov}(\\hat{\\boldsymbol{\\beta}}_\\lambda)\\) in terms of \\(\\lambda\\) and the SVD of \\(\\mathbf{X}\\).',
                    solution: 'Using \\(\\mathbf{X} = \\mathbf{U}\\mathbf{D}\\mathbf{V}^\\top\\), we get \\(\\hat{\\boldsymbol{\\beta}}_\\lambda = \\sum_j \\frac{d_j^2}{d_j^2 + \\lambda} \\frac{\\mathbf{u}_j^\\top \\mathbf{y}}{d_j} \\mathbf{v}_j\\). Each coefficient is shrunk by the factor \\(d_j^2/(d_j^2 + \\lambda) &lt; 1\\), which biases \\(\\mathbb{E}[\\hat{\\boldsymbol{\\beta}}_\\lambda]\\) away from \\(\\boldsymbol{\\beta}\\). Meanwhile, \\(\\text{Cov}(\\hat{\\boldsymbol{\\beta}}_\\lambda) = \\sigma^2 (\\mathbf{X}^\\top \\mathbf{X} + \\lambda \\mathbf{I})^{-1} \\mathbf{X}^\\top \\mathbf{X} (\\mathbf{X}^\\top \\mathbf{X} + \\lambda \\mathbf{I})^{-1}\\). As \\(\\lambda\\) increases, each eigenvalue contribution \\(d_j^2/(d_j^2+\\lambda)^2\\) decreases, so variance decreases. This is the classic bias-variance tradeoff controlled by \\(\\lambda\\).'
                }
            ]
        },

        // ===== SECTION 2: Cross-Validation =====
        {
            id: 'cross-validation',
            title: 'Cross-Validation',
            content: `
<div class="env-block intuition"><div class="env-title">Why Cross-Validation?</div><div class="env-body">
<p>The bias-variance decomposition tells us that an optimal model complexity exists, but it requires knowing the true \\(f(x)\\), which we never have. Cross-validation provides a data-driven estimate of test error without needing a separate test set. The core idea is simple: repeatedly hold out a portion of the training data, fit the model on the rest, and evaluate on the held-out portion.</p>
</div></div>

<h2>K-Fold Cross-Validation</h2>

<div class="env-block definition"><div class="env-title">Definition 3.2 (K-Fold Cross-Validation)</div><div class="env-body">
<p>Partition the training set \\(\\{1, \\dots, n\\}\\) into \\(K\\) roughly equal-sized folds \\(F_1, \\dots, F_K\\). For each fold \\(k = 1, \\dots, K\\):</p>
<ol>
  <li>Train the model on all data <em>except</em> fold \\(F_k\\).</li>
  <li>Evaluate the prediction error on fold \\(F_k\\).</li>
</ol>
<p>The cross-validation estimate of test error is:</p>
\\[\\text{CV}(K) = \\frac{1}{n} \\sum_{k=1}^{K} \\sum_{i \\in F_k} L\\bigl(y_i,\\, \\hat{f}^{-k}(x_i)\\bigr)\\]
<p>where \\(\\hat{f}^{-k}\\) is the model trained without fold \\(k\\), and \\(L\\) is the loss function.</p>
</div></div>

<p>Common choices are \\(K = 5\\) or \\(K = 10\\). The visualization below shows how data cycles through the folds.</p>

<div class="viz-placeholder" data-viz="viz-kfold-animation"></div>

<h3>Leave-One-Out Cross-Validation (LOOCV)</h3>

<p>Setting \\(K = n\\) gives <strong>leave-one-out CV</strong>: each fold contains exactly one observation. The LOOCV estimate is:</p>
\\[\\text{CV}(n) = \\frac{1}{n} \\sum_{i=1}^{n} L\\bigl(y_i,\\, \\hat{f}^{-i}(x_i)\\bigr)\\]

<div class="env-block theorem"><div class="env-title">Theorem 3.2 (LOOCV for Linear Models)</div><div class="env-body">
<p>For linear regression with the hat matrix \\(\\mathbf{H} = \\mathbf{X}(\\mathbf{X}^\\top \\mathbf{X})^{-1}\\mathbf{X}^\\top\\), the LOOCV error has a closed form:</p>
\\[\\text{CV}(n) = \\frac{1}{n} \\sum_{i=1}^{n} \\left(\\frac{y_i - \\hat{y}_i}{1 - h_{ii}}\\right)^2\\]
<p>where \\(h_{ii}\\) is the \\(i\\)-th diagonal element of \\(\\mathbf{H}\\). This requires fitting the model only once.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Bias-Variance of CV</div><div class="env-body">
<p>LOOCV is approximately unbiased for estimating test error (each training set has \\(n-1\\) observations, almost the full dataset), but has high variance because the \\(n\\) training sets overlap heavily. \\(K\\)-fold CV with \\(K = 5\\) or 10 introduces a small upward bias (training on \\((K-1)/K\\) fraction of the data) but reduces variance. This is another bias-variance tradeoff, now in the model selection procedure itself.</p>
</div></div>

<h3>Stratified Cross-Validation</h3>

<p>For classification, we want each fold to preserve the class proportions of the full dataset. <strong>Stratified \\(K\\)-fold CV</strong> ensures that if class A comprises 30% of the data, each fold also contains approximately 30% class A. This reduces the variance of the CV estimate, especially when classes are imbalanced.</p>

<div class="env-block example"><div class="env-title">Example 3.2 (CV for Model Selection)</div><div class="env-body">
<p>To choose the polynomial degree \\(d\\) for regression, compute \\(\\text{CV}(10)\\) for \\(d = 1, 2, \\dots, 15\\). Select the degree \\(d^*\\) that minimizes the 10-fold CV error. This is an honest estimate of how well a degree-\\(d^*\\) polynomial trained on all \\(n\\) observations would perform on new data.</p>
</div></div>

<div class="env-block warning"><div class="env-title">The One-Standard-Error Rule</div><div class="env-body">
<p>Because CV estimates have variability, a common practice is to select the simplest model whose CV error is within one standard error of the minimum. This guards against selecting an unnecessarily complex model that happens to score marginally better due to sampling variation.</p>
</div></div>
`,
            visualizations: [
                {
                    id: 'viz-kfold-animation',
                    title: 'K-Fold Cross-Validation: Animated Data Splits',
                    description: 'Watch data points cycle through training (blue) and validation (orange) roles as the active fold changes. Adjust K to see how fold size changes.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { width: 700, height: 340, scale: 1, originX: 0, originY: 0 });
                        const ctx = viz.ctx;
                        const W = viz.width, H = viz.height;
                        let K = 5, speed = 1.0;
                        const n = 30;

                        function seededRandom(s) { let x = Math.sin(s) * 10000; return x - Math.floor(x); }
                        const data = [];
                        for (let i = 0; i < n; i++) {
                            data.push({ x: seededRandom(i * 17 + 5) * 0.8 + 0.1, y: seededRandom(i * 23 + 11) * 0.6 + 0.2 });
                        }

                        VizEngine.createSlider(controls, 'K', 2, 10, K, 1, v => { K = Math.round(v); });
                        VizEngine.createSlider(controls, 'Speed', 0.3, 3, speed, 0.1, v => { speed = v; });

                        function draw(t) {
                            viz.clear();
                            const period = 2500 / speed;
                            const activeFold = Math.floor((t % (period * K)) / period);
                            const progress = ((t % period) / period);

                            const m = { top: 50, bottom: 60, left: 40, right: 40 };
                            const pW = W - m.left - m.right, pH = H - m.top - m.bottom;

                            // Title
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            ctx.fillText(K + '-Fold CV  \u2014  Fold ' + (activeFold + 1) + ' / ' + K + ' as validation', W / 2, 24);

                            // Fold bars at bottom
                            const barY = H - 40, barH = 18, barGap = 4;
                            const barTotalW = pW;
                            const barW = (barTotalW - (K - 1) * barGap) / K;
                            for (let k = 0; k < K; k++) {
                                const bx = m.left + k * (barW + barGap);
                                ctx.fillStyle = k === activeFold ? viz.colors.orange + '99' : viz.colors.blue + '44';
                                ctx.beginPath();
                                ctx.roundRect(bx, barY, barW, barH, 3);
                                ctx.fill();
                                ctx.fillStyle = k === activeFold ? viz.colors.orange : viz.colors.blue;
                                ctx.font = '10px -apple-system,sans-serif'; ctx.textAlign = 'center';
                                ctx.fillText(k === activeFold ? 'Val' : 'Train', bx + barW / 2, barY + barH / 2 + 3.5);
                            }

                            // Data points
                            const foldSize = Math.ceil(n / K);
                            for (let i = 0; i < n; i++) {
                                const fold = Math.min(Math.floor(i / foldSize), K - 1);
                                const isVal = fold === activeFold;
                                const px = m.left + data[i].x * pW;
                                const py = m.top + data[i].y * pH;

                                const col = isVal ? viz.colors.orange : viz.colors.blue;
                                const r = isVal ? 6.5 : 5;

                                // Glow for validation
                                if (isVal) {
                                    const glowR = 6.5 + 3 * Math.sin(progress * Math.PI * 2);
                                    ctx.fillStyle = viz.colors.orange + '22';
                                    ctx.beginPath(); ctx.arc(px, py, glowR + 5, 0, Math.PI * 2); ctx.fill();
                                }

                                ctx.fillStyle = col;
                                ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill();
                                ctx.strokeStyle = col; ctx.lineWidth = 1.5;
                                ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.stroke();
                            }

                            // Legend
                            ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'left';
                            ctx.fillStyle = viz.colors.blue; ctx.beginPath(); ctx.arc(m.left + 10, m.top - 20, 5, 0, Math.PI * 2); ctx.fill();
                            ctx.fillText('Training', m.left + 22, m.top - 16);
                            ctx.fillStyle = viz.colors.orange; ctx.beginPath(); ctx.arc(m.left + 100, m.top - 20, 5, 0, Math.PI * 2); ctx.fill();
                            ctx.fillText('Validation', m.left + 112, m.top - 16);

                            // Stats
                            const nVal = Math.min(foldSize, n - activeFold * foldSize);
                            const nTr = n - nVal;
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'right';
                            ctx.fillText('Train: ' + nTr + '  |  Val: ' + nVal, W - m.right, m.top - 16);
                        }
                        viz.animate(draw);
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that for linear regression, the LOOCV shortcut \\(\\text{CV}(n) = \\frac{1}{n}\\sum_{i=1}^n \\left(\\frac{y_i - \\hat{y}_i}{1 - h_{ii}}\\right)^2\\) holds. (Hint: use the Sherman-Morrison-Woodbury formula to express \\(\\hat{f}^{-i}(x_i)\\).)',
                    hint: 'Removing observation \\(i\\) is a rank-1 update to \\(\\mathbf{X}^\\top\\mathbf{X}\\). Apply the matrix inversion lemma to express the leave-one-out prediction in terms of the full-data fit.',
                    solution: 'By Sherman-Morrison, \\((\\mathbf{X}^\\top\\mathbf{X} - \\mathbf{x}_i\\mathbf{x}_i^\\top)^{-1} = (\\mathbf{X}^\\top\\mathbf{X})^{-1} + \\frac{(\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{x}_i\\mathbf{x}_i^\\top(\\mathbf{X}^\\top\\mathbf{X})^{-1}}{1 - h_{ii}}\\). The leave-one-out prediction is \\(\\hat{f}^{-i}(x_i) = \\mathbf{x}_i^\\top \\hat{\\boldsymbol{\\beta}}^{-i}\\). After algebra, \\(y_i - \\hat{f}^{-i}(x_i) = \\frac{y_i - \\hat{y}_i}{1 - h_{ii}}\\), where \\(\\hat{y}_i = \\mathbf{x}_i^\\top \\hat{\\boldsymbol{\\beta}}\\) is the full-data prediction. Squaring and averaging gives the result.'
                },
                {
                    question: 'Explain why LOOCV has higher variance than 5-fold CV for estimating test error, even though it is nearly unbiased.',
                    hint: 'Consider the correlation between the \\(n\\) individual error estimates in LOOCV.',
                    solution: 'In LOOCV, the \\(n\\) training sets differ by only one observation, so they are nearly identical. The resulting models \\(\\hat{f}^{-1}, \\dots, \\hat{f}^{-n}\\) are highly correlated, making the individual error estimates \\((y_i - \\hat{f}^{-i}(x_i))^2\\) also highly correlated. The variance of the average of \\(n\\) positively correlated variables is larger than the variance of the average of \\(n\\) less correlated variables. In 5-fold CV, the training sets share only 3/4 of their data (rather than \\((n-2)/(n-1) \\approx 1\\)), so the fold-level error estimates are less correlated and the average has lower variance.'
                },
                {
                    question: 'You are building a classifier for a dataset with 95% class 0 and 5% class 1. Explain why stratified CV is important here, and describe what could go wrong without stratification.',
                    hint: 'Consider what happens if a fold ends up with zero class 1 observations.',
                    solution: 'With severe class imbalance, random (non-stratified) folds may contain very few or even zero class 1 examples. A fold with no positive examples cannot evaluate sensitivity/recall at all, and the model trained without that fold has even fewer positive examples to learn from. This makes the CV error estimate unstable and biased toward the majority class. Stratified CV preserves the 95/5 ratio in each fold, ensuring the model always sees both classes during training and evaluation, producing a more reliable error estimate.'
                }
            ]
        },

        // ===== SECTION 3: Information Criteria =====
        {
            id: 'information-criteria',
            title: 'Information Criteria',
            content: `
<div class="env-block intuition"><div class="env-title">Model Selection Without Refitting</div><div class="env-body">
<p>Cross-validation requires fitting the model multiple times, which can be expensive. Information criteria provide an alternative: estimate the out-of-sample prediction error from a <em>single</em> fit by adding a penalty for model complexity to the training loss. The penalty approximates the optimism, the gap between training error and test error, that arises from fitting and evaluating on the same data.</p>
</div></div>

<h2>Akaike Information Criterion (AIC)</h2>

<div class="env-block definition"><div class="env-title">Definition 3.3 (AIC)</div><div class="env-body">
<p>For a model with \\(p\\) estimated parameters and maximized log-likelihood \\(\\hat{\\ell}\\), the AIC is:</p>
\\[\\text{AIC} = -2\\hat{\\ell} + 2p\\]
<p>Select the model that <strong>minimizes</strong> AIC. The \\(2p\\) term penalizes complexity.</p>
</div></div>

<p>AIC arises from an asymptotic approximation to the Kullback-Leibler divergence between the true and fitted model. Under regularity conditions, the <em>optimism</em> (expected difference between training and test deviance) is approximately \\(2p\\), so AIC estimates the expected out-of-sample deviance.</p>

<div class="env-block theorem"><div class="env-title">Theorem 3.3 (AIC and LOOCV Equivalence)</div><div class="env-body">
<p>For linear regression with Gaussian errors, AIC is asymptotically equivalent to leave-one-out cross-validation. Both estimate the expected prediction error up to a constant.</p>
</div></div>

<h2>Bayesian Information Criterion (BIC)</h2>

<div class="env-block definition"><div class="env-title">Definition 3.4 (BIC)</div><div class="env-body">
<p>For \\(n\\) observations and \\(p\\) parameters:</p>
\\[\\text{BIC} = -2\\hat{\\ell} + p \\ln n\\]
<p>BIC replaces the \\(2p\\) penalty in AIC with \\(p \\ln n\\), which grows with the sample size.</p>
</div></div>

<p>BIC arises from a Laplace approximation to the marginal likelihood \\(p(\\text{data} \\mid \\text{model})\\). While AIC targets prediction error, BIC targets the true model. For \\(n \\geq 8\\), the BIC penalty \\(\\ln n > 2\\), so BIC penalizes complexity more heavily and tends to select simpler models than AIC.</p>

<div class="env-block remark"><div class="env-title">AIC vs. BIC: Different Goals</div><div class="env-body">
<p><strong>AIC</strong> is efficient: it minimizes prediction error but may overfit by selecting models with too many parameters. <strong>BIC</strong> is consistent: as \\(n \\to \\infty\\), it selects the true model (if it is among the candidates) with probability 1, but may underfit in small samples. Neither dominates the other; the choice depends on whether prediction accuracy or model identification is the goal.</p>
</div></div>

<h2>Minimum Description Length (MDL)</h2>

<p>The <strong>MDL principle</strong> (Rissanen, 1978) comes from information theory. It views model selection as data compression: the best model is the one that gives the shortest total description of the data, combining the cost of describing the model itself and the cost of describing the data given the model.</p>

\\[\\text{MDL} = \\underbrace{L(\\text{model})}_{\\text{model complexity}} + \\underbrace{L(\\text{data} \\mid \\text{model})}_{\\text{lack of fit}}\\]

<p>Under appropriate coding schemes, MDL coincides with BIC to leading order.</p>

<div class="viz-placeholder" data-viz="viz-aic-bic"></div>

<div class="env-block example"><div class="env-title">Example 3.3 (Comparing AIC and BIC)</div><div class="env-body">
<p>Consider polynomial regression with \\(n = 50\\) and true degree 3. AIC often selects degree 4 or 5 (slightly overfit, but good prediction). BIC more reliably selects degree 3 (the true model). With \\(n = 500\\), both converge to degree 3, but BIC does so faster due to the \\(\\ln n\\) penalty.</p>
</div></div>
`,
            visualizations: [
                {
                    id: 'viz-aic-bic',
                    title: 'AIC vs. BIC: Model Selection Curves',
                    description: 'Compare how AIC and BIC score models of increasing complexity. The markers show each criterion\'s selected model. Adjust sample size to see how BIC\'s penalty grows with n.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { width: 700, height: 350, scale: 1, originX: 0, originY: 0 });
                        const ctx = viz.ctx;
                        const W = viz.width, H = viz.height;
                        let nObs = 50;
                        const trueDeg = 3;

                        function seededRandom(s) { let x = Math.sin(s) * 10000; return x - Math.floor(x); }

                        VizEngine.createSlider(controls, 'n', 20, 200, nObs, 10, v => { nObs = Math.round(v); draw(); });

                        function computeCriteria(n) {
                            const maxDeg = 12;
                            const aic = [], bic = [];
                            // Simulate: RSS decreases with degree, levels off past true degree
                            for (let d = 1; d <= maxDeg; d++) {
                                // True signal captured by degree 3; beyond that, only noise reduction
                                const signalRSS = d >= trueDeg ? 0 : Math.pow(0.6, d) * 2.5;
                                const noiseRSS = 1.0 + 0.02 * (d - trueDeg) * (d > trueDeg ? 1 : 0);
                                const rss = signalRSS + noiseRSS + 0.05 * seededRandom(d * 31 + n);
                                const sigma2 = rss;
                                const p = d + 1;
                                const loglik = -n / 2 * Math.log(sigma2 / n) - n / 2;
                                aic.push(-2 * loglik + 2 * p);
                                bic.push(-2 * loglik + p * Math.log(n));
                            }
                            return { aic, bic };
                        }

                        function draw() {
                            viz.clear();
                            const { aic, bic } = computeCriteria(nObs);
                            const m = { top: 40, bottom: 45, left: 55, right: 30 };
                            const pW = W - m.left - m.right, pH = H - m.top - m.bottom;
                            const maxDeg = 12;

                            const allVals = [...aic, ...bic];
                            const yMin = Math.min(...allVals) - 5;
                            const yMax = Math.max(...allVals) + 5;

                            const toP = (d, v) => [m.left + (d - 1) / (maxDeg - 1) * pW, m.top + (1 - (v - yMin) / (yMax - yMin)) * pH];

                            // Grid
                            ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                            for (let d = 1; d <= maxDeg; d += 2) { const [px] = toP(d, yMin); ctx.beginPath(); ctx.moveTo(px, m.top); ctx.lineTo(px, m.top + pH); ctx.stroke(); }

                            // Axes
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(m.left, m.top + pH); ctx.lineTo(m.left + pW, m.top + pH); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + pH); ctx.stroke();

                            // X labels
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            for (let d = 1; d <= maxDeg; d++) { const [px] = toP(d, yMin); ctx.fillText(d, px, m.top + pH + 16); }
                            ctx.fillText('Model Complexity (degree)', m.left + pW / 2, H - 8);

                            // Y label
                            ctx.save(); ctx.translate(14, m.top + pH / 2); ctx.rotate(-Math.PI / 2);
                            ctx.fillText('Criterion Value', 0, 0); ctx.restore();

                            // Draw curves
                            const drawCurve = (arr, col, lbl) => {
                                ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.beginPath();
                                arr.forEach((v, i) => { const [px, py] = toP(i + 1, v); i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); });
                                ctx.stroke();
                                // Points
                                arr.forEach((v, i) => { const [px, py] = toP(i + 1, v); ctx.fillStyle = col; ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill(); });
                                // Min marker
                                let mi = 0; arr.forEach((v, i) => { if (v < arr[mi]) mi = i; });
                                const [mx, my] = toP(mi + 1, arr[mi]);
                                ctx.strokeStyle = col; ctx.lineWidth = 2;
                                ctx.beginPath(); ctx.arc(mx, my, 9, 0, Math.PI * 2); ctx.stroke();
                                ctx.fillStyle = col; ctx.font = 'bold 11px -apple-system,sans-serif'; ctx.textAlign = 'center';
                                ctx.fillText(lbl + ' \u2192 d=' + (mi + 1), mx, my - 15);
                            };

                            drawCurve(aic, viz.colors.blue, 'AIC');
                            drawCurve(bic, viz.colors.orange, 'BIC');

                            // True degree marker
                            const [tx] = toP(trueDeg, yMin);
                            ctx.strokeStyle = viz.colors.green; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
                            ctx.beginPath(); ctx.moveTo(tx, m.top); ctx.lineTo(tx, m.top + pH); ctx.stroke(); ctx.setLineDash([]);
                            ctx.fillStyle = viz.colors.green; ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillText('true d=3', tx, m.top - 6);

                            // Legend
                            ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'right';
                            ctx.fillStyle = viz.colors.text;
                            ctx.fillText('n = ' + nObs + '  |  BIC penalty = p\u00B7ln(' + nObs + ') = p\u00B7' + Math.log(nObs).toFixed(2), W - m.right, 20);
                        }
                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Derive the AIC penalty \\(2p\\) from the expected optimism of the training error. (Hint: for a linear model with \\(p\\) parameters and Gaussian errors, show that \\(\\mathbb{E}[\\text{train error}] \\approx \\mathbb{E}[\\text{test error}] - 2p\\sigma^2/n\\).)',
                    hint: 'Use the fact that for a linear model, \\(\\hat{y}_i = \\sum_j h_{ij} y_j\\), and compute \\(\\text{Cov}(y_i, \\hat{y}_i) = h_{ii} \\sigma^2\\). Sum over \\(i\\) and use \\(\\text{tr}(\\mathbf{H}) = p\\).',
                    solution: 'The optimism is \\(\\text{op} = \\frac{2}{n}\\sum_{i=1}^n \\text{Cov}(y_i, \\hat{y}_i) = \\frac{2}{n} \\sum_i h_{ii} \\sigma^2 = \\frac{2p\\sigma^2}{n}\\). So test error \\(\\approx\\) train error \\(+ 2p\\sigma^2/n\\). On the deviance scale (\\(-2\\ell\\)), this becomes an additive penalty of \\(2p\\), yielding \\(\\text{AIC} = -2\\hat{\\ell} + 2p\\).'
                },
                {
                    question: 'For \\(n = 100\\) observations, how much larger is the BIC penalty than the AIC penalty per parameter? At what sample size are they equal?',
                    hint: 'Compare \\(2p\\) (AIC) to \\(p\\ln n\\) (BIC).',
                    solution: 'BIC penalty per parameter is \\(\\ln(100) \\approx 4.61\\), while AIC penalty per parameter is 2. So BIC is \\(4.61/2 \\approx 2.3\\) times larger. They are equal when \\(\\ln n = 2\\), i.e., \\(n = e^2 \\approx 7.39\\). For any \\(n \\geq 8\\), BIC penalizes more heavily.'
                },
                {
                    question: 'A researcher fits models with \\(p = 1, 2, \\ldots, 20\\) parameters. She selects the model by AIC, then reports the AIC-selected model\'s training error as the test error estimate. What is wrong with this procedure?',
                    hint: 'The AIC was computed over many candidate models. Does the minimum AIC over 20 models still estimate the test error of the selected model?',
                    solution: 'This is a form of selection bias. AIC estimates the test error of a <em>single</em> pre-specified model. When we take the minimum over 20 models, we are selecting the model that happened to look best, which biases the AIC-selected value downward. The correct approach is to either (1) report the AIC value itself (which already accounts for complexity) as the error estimate, or (2) use a separate held-out test set to evaluate the selected model, or (3) use nested cross-validation (inner loop for model selection, outer loop for error estimation).'
                }
            ]
        },

        // ===== SECTION 4: Regularization as Model Selection =====
        {
            id: 'regularization-model-selection',
            title: 'Regularization as Model Selection',
            content: `
<div class="env-block intuition"><div class="env-title">Continuous Model Selection</div><div class="env-body">
<p>Traditional model selection chooses among a discrete set of models (degree 1, 2, 3, ...). Regularization provides a <em>continuous</em> alternative: instead of choosing which parameters to include, we shrink all parameters toward zero by a controlled amount. The regularization strength \\(\\lambda\\) smoothly interpolates between the null model (\\(\\lambda = \\infty\\)) and the full model (\\(\\lambda = 0\\)). The <em>regularization path</em>, showing how coefficients evolve as \\(\\lambda\\) varies, reveals the relative importance of each feature.</p>
</div></div>

<h2>Ridge Regression (\\(L_2\\) Penalty)</h2>

<div class="env-block definition"><div class="env-title">Definition 3.5 (Ridge Regression)</div><div class="env-body">
<p>Ridge regression solves:</p>
\\[\\hat{\\boldsymbol{\\beta}}_\\lambda^{\\text{ridge}} = \\arg\\min_{\\boldsymbol{\\beta}} \\left\\{ \\|\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}\\|^2 + \\lambda \\|\\boldsymbol{\\beta}\\|^2 \\right\\}\\]
<p>The closed-form solution is \\(\\hat{\\boldsymbol{\\beta}}_\\lambda = (\\mathbf{X}^\\top\\mathbf{X} + \\lambda \\mathbf{I})^{-1}\\mathbf{X}^\\top\\mathbf{y}\\).</p>
</div></div>

<h3>Effective Degrees of Freedom</h3>

<p>For OLS, the number of parameters \\(p\\) measures model complexity. For ridge, the <em>effective degrees of freedom</em> generalize this notion:</p>

<div class="env-block definition"><div class="env-title">Definition 3.6 (Effective Degrees of Freedom)</div><div class="env-body">
\\[\\text{df}(\\lambda) = \\text{tr}\\bigl(\\mathbf{X}(\\mathbf{X}^\\top\\mathbf{X} + \\lambda\\mathbf{I})^{-1}\\mathbf{X}^\\top\\bigr) = \\sum_{j=1}^{p} \\frac{d_j^2}{d_j^2 + \\lambda}\\]
<p>where \\(d_1, \\dots, d_p\\) are the singular values of \\(\\mathbf{X}\\). When \\(\\lambda = 0\\), \\(\\text{df} = p\\). As \\(\\lambda \\to \\infty\\), \\(\\text{df} \\to 0\\).</p>
</div></div>

<h2>Lasso Regression (\\(L_1\\) Penalty)</h2>

<div class="env-block definition"><div class="env-title">Definition 3.7 (Lasso)</div><div class="env-body">
<p>The Lasso (Tibshirani, 1996) solves:</p>
\\[\\hat{\\boldsymbol{\\beta}}_\\lambda^{\\text{lasso}} = \\arg\\min_{\\boldsymbol{\\beta}} \\left\\{ \\|\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}\\|^2 + \\lambda \\|\\boldsymbol{\\beta}\\|_1 \\right\\}\\]
<p>Unlike ridge, the \\(L_1\\) penalty drives some coefficients exactly to zero, performing <strong>variable selection</strong>.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Geometry of Ridge vs. Lasso</div><div class="env-body">
<p>The ridge constraint region \\(\\|\\boldsymbol{\\beta}\\|^2 \\leq t\\) is a sphere; the lasso constraint \\(\\|\\boldsymbol{\\beta}\\|_1 \\leq t\\) is a diamond. The diamond has corners on the coordinate axes, so the elliptical contours of the RSS are more likely to touch a corner, setting a coefficient exactly to zero. This geometric argument explains why lasso produces sparse solutions while ridge does not.</p>
</div></div>

<h3>The Regularization Path</h3>

<p>Plotting \\(\\hat{\\beta}_j(\\lambda)\\) as a function of \\(\\lambda\\) (or equivalently, \\(\\text{df}(\\lambda)\\)) gives the <em>regularization path</em>. For ridge, all paths are smooth curves shrinking toward zero. For lasso, coefficients enter the model one at a time as \\(\\lambda\\) decreases, and the paths are piecewise linear.</p>

<div class="viz-placeholder" data-viz="viz-reg-path"></div>

<h3>Choosing \\(\\lambda\\) by Cross-Validation</h3>

<p>The regularization parameter \\(\\lambda\\) is a <em>hyperparameter</em> that controls model complexity. It is typically chosen by cross-validation: compute the CV error for a grid of \\(\\lambda\\) values and select the \\(\\lambda\\) that minimizes CV error (or apply the one-standard-error rule).</p>

<div class="env-block theorem"><div class="env-title">Theorem 3.4 (Ridge GCV)</div><div class="env-body">
<p>For ridge regression, the generalized cross-validation (GCV) criterion is:</p>
\\[\\text{GCV}(\\lambda) = \\frac{1}{n} \\frac{\\|\\mathbf{y} - \\mathbf{X}\\hat{\\boldsymbol{\\beta}}_\\lambda\\|^2}{\\bigl(1 - \\text{df}(\\lambda)/n\\bigr)^2}\\]
<p>GCV approximates LOOCV and avoids computing individual leverage values.</p>
</div></div>

<div class="env-block example"><div class="env-title">Example 3.4 (Ridge vs. Lasso in Practice)</div><div class="env-body">
<p>With 50 features, only 5 of which are truly relevant, lasso typically outperforms ridge because it zeroes out the 45 irrelevant features. But if all features contribute weakly, ridge outperforms because the \\(L_1\\) penalty's aggressive zeroing discards useful signal. The elastic net \\(\\lambda_1\\|\\boldsymbol{\\beta}\\|_1 + \\lambda_2\\|\\boldsymbol{\\beta}\\|^2\\) combines both penalties and often works well in practice.</p>
</div></div>
`,
            visualizations: [
                {
                    id: 'viz-reg-path',
                    title: 'Regularization Path with CV Error',
                    description: 'Left: coefficient paths as regularization strength varies (log scale). Right: CV error curve. Drag the lambda slider to see which coefficients are active at each regularization level.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { width: 780, height: 370, scale: 1, originX: 0, originY: 0 });
                        const ctx = viz.ctx;
                        const W = viz.width, H = viz.height;
                        let logLam = 0.0;
                        const p = 8;
                        const trueCoeffs = [3.0, -2.5, 2.0, -1.5, 0.0, 0.0, 0.0, 0.0];
                        const colors = ['#58a6ff', '#3fb9a0', '#f0883e', '#bc8cff', '#f85149', '#d29922', '#f778ba', '#3fb950'];
                        const nLam = 80;
                        const logLamMin = -3, logLamMax = 3;

                        // Simulate ridge paths
                        function ridgePaths() {
                            const paths = [];
                            for (let li = 0; li < nLam; li++) {
                                const ll = logLamMin + (logLamMax - logLamMin) * li / (nLam - 1);
                                const lam = Math.pow(10, ll);
                                const coeffs = trueCoeffs.map((b, j) => {
                                    const dj = 3.0 - j * 0.3;
                                    return b * (dj * dj) / (dj * dj + lam);
                                });
                                paths.push({ ll, coeffs });
                            }
                            return paths;
                        }

                        // Simulate CV error
                        function cvErrors() {
                            const errs = [];
                            for (let li = 0; li < nLam; li++) {
                                const ll = logLamMin + (logLamMax - logLamMin) * li / (nLam - 1);
                                const lam = Math.pow(10, ll);
                                // Bias increases with lambda, variance decreases
                                let bias2 = 0, vari = 0;
                                for (let j = 0; j < p; j++) {
                                    const dj = 3.0 - j * 0.3;
                                    const shrink = dj * dj / (dj * dj + lam);
                                    bias2 += trueCoeffs[j] * trueCoeffs[j] * (1 - shrink) * (1 - shrink);
                                    vari += shrink * shrink * 0.5;
                                }
                                errs.push(bias2 + vari + 0.8);
                            }
                            return errs;
                        }

                        const paths = ridgePaths();
                        const cvErr = cvErrors();

                        VizEngine.createSlider(controls, 'log\u2081\u2080\u03bb', logLamMin, logLamMax, logLam, 0.1, v => { logLam = v; draw(); });

                        function draw() {
                            viz.clear();
                            const leftW = W * 0.52, rightW = W - leftW;
                            const m = { top: 35, bottom: 40, left: 50, right: 15 };

                            // === LEFT: Coefficient paths ===
                            const pW = leftW - m.left - m.right, pH = H - m.top - m.bottom;
                            const cMax = 3.5, cMin = -3.0;

                            const toL = (ll, c) => [m.left + (ll - logLamMin) / (logLamMax - logLamMin) * pW, m.top + (1 - (c - cMin) / (cMax - cMin)) * pH];

                            // Grid
                            ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                            for (let g = logLamMin; g <= logLamMax; g++) { const [px] = toL(g, 0); ctx.beginPath(); ctx.moveTo(px, m.top); ctx.lineTo(px, m.top + pH); ctx.stroke(); }
                            for (let g = -2; g <= 3; g++) { const [, py] = toL(0, g); ctx.beginPath(); ctx.moveTo(m.left, py); ctx.lineTo(m.left + pW, py); ctx.stroke(); }

                            // Zero line
                            const [, zy] = toL(0, 0);
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(m.left, zy); ctx.lineTo(m.left + pW, zy); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + pH); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(m.left, m.top + pH); ctx.lineTo(m.left + pW, m.top + pH); ctx.stroke();

                            // Draw paths
                            for (let j = 0; j < p; j++) {
                                ctx.strokeStyle = colors[j]; ctx.lineWidth = 1.8; ctx.beginPath();
                                paths.forEach((pt, i) => { const [px, py] = toL(pt.ll, pt.coeffs[j]); i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); });
                                ctx.stroke();
                            }

                            // Current lambda marker
                            const [lx] = toL(logLam, 0);
                            ctx.strokeStyle = viz.colors.white; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
                            ctx.beginPath(); ctx.moveTo(lx, m.top); ctx.lineTo(lx, m.top + pH); ctx.stroke(); ctx.setLineDash([]);

                            // Current coefficients as dots
                            const lam = Math.pow(10, logLam);
                            for (let j = 0; j < p; j++) {
                                const dj = 3.0 - j * 0.3;
                                const c = trueCoeffs[j] * (dj * dj) / (dj * dj + lam);
                                const [px, py] = toL(logLam, c);
                                ctx.fillStyle = colors[j]; ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
                            }

                            // Labels
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            ctx.fillText('log\u2081\u2080(\u03bb)', m.left + pW / 2, H - 8);
                            ctx.fillText('Ridge Coefficient Paths', m.left + pW / 2, 18);
                            for (let g = logLamMin; g <= logLamMax; g++) { const [px] = toL(g, 0); ctx.fillText(g, px, m.top + pH + 14); }

                            // Right label for feature legend
                            ctx.textAlign = 'left'; ctx.font = '9px -apple-system,sans-serif';
                            const featureNames = ['\u03b2\u2081', '\u03b2\u2082', '\u03b2\u2083', '\u03b2\u2084', '\u03b2\u2085', '\u03b2\u2086', '\u03b2\u2087', '\u03b2\u2088'];
                            for (let j = 0; j < p; j++) {
                                const dj = 3.0 - j * 0.3;
                                const c = trueCoeffs[j] * (dj * dj) / (dj * dj + Math.pow(10, logLamMin));
                                const [, py] = toL(logLamMin, c);
                                ctx.fillStyle = colors[j]; ctx.fillText(featureNames[j], m.left - 18, py + 3);
                            }

                            // === RIGHT: CV error ===
                            const rLeft = leftW + 35, rW = rightW - 55, rTop = m.top, rH = pH;
                            const eMin = Math.min(...cvErr) * 0.9, eMax = Math.max(...cvErr) * 1.1;
                            const toR = (ll, e) => [rLeft + (ll - logLamMin) / (logLamMax - logLamMin) * rW, rTop + (1 - (e - eMin) / (eMax - eMin)) * rH];

                            ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                            for (let g = logLamMin; g <= logLamMax; g++) { const [px] = toR(g, 0); ctx.beginPath(); ctx.moveTo(px, rTop); ctx.lineTo(px, rTop + rH); ctx.stroke(); }

                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(rLeft, rTop + rH); ctx.lineTo(rLeft + rW, rTop + rH); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(rLeft, rTop); ctx.lineTo(rLeft, rTop + rH); ctx.stroke();

                            // CV curve
                            ctx.strokeStyle = viz.colors.red; ctx.lineWidth = 2.5; ctx.beginPath();
                            cvErr.forEach((e, i) => { const ll = logLamMin + (logLamMax - logLamMin) * i / (nLam - 1); const [px, py] = toR(ll, e); i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py); });
                            ctx.stroke();

                            // Optimal lambda
                            let minI = 0; cvErr.forEach((e, i) => { if (e < cvErr[minI]) minI = i; });
                            const optLL = logLamMin + (logLamMax - logLamMin) * minI / (nLam - 1);
                            const [ox, oy] = toR(optLL, cvErr[minI]);
                            ctx.fillStyle = viz.colors.green; ctx.beginPath(); ctx.arc(ox, oy, 6, 0, Math.PI * 2); ctx.fill();
                            ctx.strokeStyle = viz.colors.green; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
                            ctx.beginPath(); ctx.moveTo(ox, rTop); ctx.lineTo(ox, rTop + rH); ctx.stroke(); ctx.setLineDash([]);

                            // Current lambda on CV plot
                            const [cx2] = toR(logLam, eMin);
                            ctx.strokeStyle = viz.colors.white; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
                            ctx.beginPath(); ctx.moveTo(cx2, rTop); ctx.lineTo(cx2, rTop + rH); ctx.stroke(); ctx.setLineDash([]);

                            // Labels
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            ctx.fillText('log\u2081\u2080(\u03bb)', rLeft + rW / 2, H - 8);
                            ctx.fillText('CV Error', rLeft + rW / 2, 18);
                            for (let g = logLamMin; g <= logLamMax; g++) { const [px] = toR(g, 0); ctx.fillText(g, px, rTop + rH + 14); }

                            ctx.fillStyle = viz.colors.green; ctx.font = '10px -apple-system,sans-serif';
                            ctx.fillText('\u03bb* = ' + Math.pow(10, optLL).toFixed(2), ox, oy - 12);

                            // df(lambda)
                            let df = 0;
                            for (let j = 0; j < p; j++) { const dj = 3.0 - j * 0.3; df += (dj * dj) / (dj * dj + lam); }
                            ctx.fillStyle = viz.colors.teal; ctx.textAlign = 'center'; ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('df(\u03bb) = ' + df.toFixed(1), rLeft + rW / 2, rTop + rH + 30);
                        }
                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that as \\(\\lambda \\to 0\\), \\(\\text{df}(\\lambda) \\to p\\), and as \\(\\lambda \\to \\infty\\), \\(\\text{df}(\\lambda) \\to 0\\). Interpret this in terms of model complexity.',
                    hint: 'Examine the formula \\(\\text{df}(\\lambda) = \\sum_j d_j^2 / (d_j^2 + \\lambda)\\) in each limit.',
                    solution: 'When \\(\\lambda = 0\\), each term is \\(d_j^2/d_j^2 = 1\\) (assuming \\(d_j > 0\\)), so \\(\\text{df} = p\\): the model uses all \\(p\\) degrees of freedom, equivalent to OLS. When \\(\\lambda \\to \\infty\\), each term \\(d_j^2/(d_j^2 + \\lambda) \\to 0\\), so \\(\\text{df} \\to 0\\): the model shrinks all coefficients to zero, equivalent to the null model. Intermediate \\(\\lambda\\) values give fractional degrees of freedom, capturing the idea that regularization partially constrains each parameter.'
                },
                {
                    question: 'Explain geometrically why the Lasso produces sparse solutions (coefficients exactly zero) while Ridge does not.',
                    hint: 'Consider the constraint regions \\(\\|\\boldsymbol{\\beta}\\|_1 \\leq t\\) (diamond) and \\(\\|\\boldsymbol{\\beta}\\|_2 \\leq t\\) (circle) in 2D, and where the RSS contour ellipses first touch these regions.',
                    solution: 'The RSS contours are ellipses centered at the OLS solution \\(\\hat{\\boldsymbol{\\beta}}^{\\text{OLS}}\\). As we shrink the constraint budget \\(t\\), the smallest ellipse touching the constraint region gives the regularized solution. For the \\(L_1\\) diamond, the corners lie on the coordinate axes (where one coefficient is zero), and ellipses generically first touch a corner, producing a zero coefficient. For the \\(L_2\\) circle, the first contact point lies on the smooth boundary, which almost never falls on a coordinate axis. Hence ridge shrinks coefficients toward, but not exactly to, zero.'
                },
                {
                    question: 'Derive the soft-thresholding operator for the Lasso in the orthonormal design case (\\(\\mathbf{X}^\\top\\mathbf{X} = \\mathbf{I}\\)). That is, show \\(\\hat{\\beta}_j^{\\text{lasso}} = \\text{sign}(\\hat{\\beta}_j^{\\text{OLS}})\\,(|\\hat{\\beta}_j^{\\text{OLS}}| - \\lambda/2)_+\\).',
                    hint: 'When \\(\\mathbf{X}^\\top\\mathbf{X} = \\mathbf{I}\\), the Lasso objective separates into \\(p\\) independent scalar problems. Solve each by taking the subdifferential and setting it to zero.',
                    solution: 'With orthonormal design, the objective becomes \\(\\sum_j [(\\hat{\\beta}_j^{\\text{OLS}} - \\beta_j)^2 + \\lambda|\\beta_j|]\\). Each coordinate decouples. For a single coordinate, minimize \\(g(\\beta) = (\\hat{\\beta} - \\beta)^2 + \\lambda|\\beta|\\). The subdifferential at \\(\\beta\\) is \\(-2(\\hat{\\beta} - \\beta) + \\lambda \\cdot \\text{sign}(\\beta)\\) for \\(\\beta \\neq 0\\), and \\(-2\\hat{\\beta} + \\lambda[-1,1]\\) at \\(\\beta = 0\\). Setting 0 in the subdifferential: if \\(|\\hat{\\beta}| \\leq \\lambda/2\\), the solution is \\(\\beta^* = 0\\). If \\(\\hat{\\beta} > \\lambda/2\\), then \\(\\beta^* = \\hat{\\beta} - \\lambda/2\\). If \\(\\hat{\\beta} &lt; -\\lambda/2\\), then \\(\\beta^* = \\hat{\\beta} + \\lambda/2\\). This is the soft-thresholding operator \\(S_{\\lambda/2}(\\hat{\\beta})\\).'
                }
            ]
        }
    ]
});
