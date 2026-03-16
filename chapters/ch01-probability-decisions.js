// === Chapter 1: Probability & Statistical Decision Theory ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch01',
    number: 1,
    title: 'Probability & Statistical Decision Theory',
    subtitle: 'Bayes\' theorem, loss functions, bias-variance tradeoff, dimensionality, and the limits of learning',
    sections: [
        // ========== SECTION 1: Probability Review ==========
        {
            id: 'ch01-sec01',
            title: 'Probability Review',
            content: `
<h2>1.1 Probability Review</h2>

<div class="env-block intuition">
<strong>Why Start Here?</strong> Every machine learning algorithm makes decisions under uncertainty. Classification assigns labels to inputs that may overlap between classes. Regression predicts continuous targets corrupted by noise. Probability theory gives us the language to reason precisely about this uncertainty, and Bayes' theorem tells us how to update beliefs given evidence.
</div>

<h3>Joint, Marginal, and Conditional Probability</h3>

<p>For two random variables \\(X\\) and \\(Y\\), the <em>joint distribution</em> \\(p(x,y)\\) specifies the probability of both events. From the joint we recover:</p>

<ul>
<li><strong>Marginal:</strong> \\(p(x) = \\sum_y p(x,y)\\) (or \\(\\int p(x,y)\\,dy\\) for continuous variables)</li>
<li><strong>Conditional:</strong> \\(p(y \\mid x) = p(x,y) / p(x)\\), provided \\(p(x) &gt; 0\\)</li>
</ul>

<p>The <em>chain rule</em> factorizes any joint: \\(p(x,y) = p(y \\mid x)\\,p(x) = p(x \\mid y)\\,p(y)\\).</p>

<h3>Bayes' Theorem</h3>

<div class="env-block theorem">
<strong>Theorem 1.1.1 (Bayes' Rule).</strong> For hypothesis \\(H\\) and observed data \\(D\\):
\\[
p(H \\mid D) = \\frac{p(D \\mid H)\\,p(H)}{p(D)}, \\quad p(D) = \\sum_{H'} p(D \\mid H')\\,p(H').
\\]
The terms are: \\(p(H)\\) the <em>prior</em>, \\(p(D \\mid H)\\) the <em>likelihood</em>, \\(p(H \\mid D)\\) the <em>posterior</em>, and \\(p(D)\\) the <em>evidence</em> (normalizing constant).
</div>

<div class="env-block remark">
<strong>ML Connection.</strong> In classification, \\(H\\) is a class label \\(C_k\\) and \\(D\\) is a feature vector \\(\\mathbf{x}\\). The Bayes-optimal classifier picks the class maximizing \\(p(C_k \\mid \\mathbf{x}) \\propto p(\\mathbf{x} \\mid C_k)\\,p(C_k)\\). This is the foundation of generative classifiers (Ch 6) and Bayesian methods (Ch 14).
</div>

<h3>Law of Total Probability</h3>

<p>If \\(\\{H_1, \\ldots, H_K\\}\\) partition the hypothesis space, then for any event \\(D\\):</p>

\\[
p(D) = \\sum_{k=1}^{K} p(D \\mid H_k)\\,p(H_k).
\\]

<p>This "mixing" formula appears constantly: mixture models (Ch 13), marginal likelihoods in Bayesian inference, and ensemble predictions.</p>

<h3>Common Distributions</h3>

<div class="env-block definition">
<strong>Definition 1.1.2 (Key Distributions).</strong>
<ul>
<li><strong>Bernoulli:</strong> \\(p(x) = \\theta^x(1-\\theta)^{1-x}\\) for \\(x \\in \\{0,1\\}\\). Mean \\(\\theta\\), variance \\(\\theta(1-\\theta)\\).</li>
<li><strong>Gaussian:</strong> \\(p(x) = (2\\pi\\sigma^2)^{-1/2}\\exp(-(x-\\mu)^2/2\\sigma^2)\\). The Central Limit Theorem makes it ubiquitous.</li>
<li><strong>Categorical:</strong> \\(p(x=k) = \\pi_k\\) for \\(k \\in \\{1,\\ldots,K\\}\\), with \\(\\sum_k \\pi_k = 1\\). Softmax outputs parameterize this.</li>
<li><strong>Beta:</strong> \\(p(\\theta) \\propto \\theta^{\\alpha-1}(1-\\theta)^{\\beta-1}\\) on \\([0,1]\\). Conjugate prior for Bernoulli.</li>
</ul>
</div>

<div class="viz-placeholder" data-viz="viz-bayes-area"></div>
`,
            visualizations: [
                {
                    id: 'viz-bayes-area',
                    title: 'Bayes\' Theorem: Prior to Posterior',
                    description: 'Adjust the prior \\(p(H)\\) and likelihood \\(p(D \\mid H)\\) to see how the posterior \\(p(H \\mid D)\\) is computed. The bar areas show the numerator and denominator of Bayes\' rule.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 40, originX: 60, originY: null });
                        viz.originY = viz.height - 40;
                        let prior = 0.3, likH = 0.9, likNotH = 0.2;
                        VizEngine.createSlider(controls, 'p(H)', 0.01, 0.99, prior, 0.01, v => { prior = v; });
                        VizEngine.createSlider(controls, 'p(D|H)', 0.01, 0.99, likH, 0.01, v => { likH = v; });
                        VizEngine.createSlider(controls, 'p(D|\u00acH)', 0.01, 0.99, likNotH, 0.01, v => { likNotH = v; });

                        function draw() {
                            viz.clear();
                            const ctx = viz.ctx;
                            const W = viz.width, H = viz.height;
                            const m = 60, top = 30;
                            const barW = (W - 2 * m - 80) / 4;
                            const maxH = H - top - 60;

                            // Compute Bayes
                            const pDH = likH * prior;
                            const pDnH = likNotH * (1 - prior);
                            const pD = pDH + pDnH;
                            const posterior = pDH / pD;

                            const bars = [
                                { label: 'p(H)', val: prior, color: viz.colors.blue },
                                { label: 'p(D|H)', val: likH, color: viz.colors.teal },
                                { label: 'p(D)', val: pD, color: viz.colors.orange },
                                { label: 'p(H|D)', val: posterior, color: viz.colors.green }
                            ];

                            bars.forEach((b, i) => {
                                const x = m + i * (barW + 20);
                                const bH = b.val * maxH;
                                const y = H - 40 - bH;
                                ctx.fillStyle = b.color + '55';
                                ctx.fillRect(x, y, barW, bH);
                                ctx.strokeStyle = b.color;
                                ctx.lineWidth = 2;
                                ctx.strokeRect(x, y, barW, bH);
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = 'bold 12px -apple-system,sans-serif';
                                ctx.textAlign = 'center';
                                ctx.fillText(b.label, x + barW / 2, H - 22);
                                ctx.fillStyle = b.color;
                                ctx.font = '13px -apple-system,sans-serif';
                                ctx.fillText(b.val.toFixed(3), x + barW / 2, y - 8);
                            });

                            // Formula text
                            ctx.fillStyle = viz.colors.white;
                            ctx.font = '13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText(
                                'p(H|D) = p(D|H)\u00b7p(H) / p(D) = ' +
                                likH.toFixed(2) + '\u00d7' + prior.toFixed(2) + ' / ' + pD.toFixed(3) +
                                ' = ' + posterior.toFixed(3),
                                W / 2, 18
                            );

                            // Decomposition of p(D) bar
                            const x3 = m + 2 * (barW + 20);
                            const hDH = pDH * maxH;
                            const hDnH = pDnH * maxH;
                            ctx.fillStyle = viz.colors.teal + '44';
                            ctx.fillRect(x3 + 2, H - 40 - hDH - hDnH, barW - 4, hDH);
                            ctx.fillStyle = viz.colors.red + '44';
                            ctx.fillRect(x3 + 2, H - 40 - hDnH, barW - 4, hDnH);
                        }
                        viz.animate(draw);
                        return { stopAnimation: () => viz.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'A disease has prevalence \\(p(D) = 0.001\\). A test has sensitivity \\(p(+|D) = 0.99\\) and false positive rate \\(p(+|\\neg D) = 0.05\\). Compute \\(p(D|+)\\). Why is the result counterintuitive?',
                    hint: 'First compute \\(p(+) = p(+|D)p(D) + p(+|\\neg D)p(\\neg D)\\).',
                    solution: '\\(p(+) = 0.99(0.001) + 0.05(0.999) = 0.05094\\). So \\(p(D|+) = 0.00099/0.05094 \\approx 1.94\\%\\). Despite high sensitivity, the posterior is low because false positives from the large healthy population swamp the true positives. This is the <em>base-rate fallacy</em>.'
                },
                {
                    question: 'Show that Bayes\' rule follows from the two ways of writing the chain rule: \\(p(x,y) = p(x|y)p(y) = p(y|x)p(x)\\).',
                    hint: 'Equate the two expressions and solve for \\(p(x|y)\\).',
                    solution: 'From \\(p(y|x)p(x) = p(x|y)p(y)\\), divide both sides by \\(p(y)\\): \\(p(x|y) = p(y|x)p(x)/p(y)\\). This is Bayes\' rule.'
                },
                {
                    question: 'If \\(X \\sim \\text{Beta}(\\alpha, \\beta)\\) is the prior for a coin\'s bias and we observe \\(h\\) heads and \\(t\\) tails, show the posterior is \\(\\text{Beta}(\\alpha+h, \\beta+t)\\).',
                    hint: 'Write the likelihood as \\(\\theta^h(1-\\theta)^t\\) and multiply by the Beta prior.',
                    solution: 'The posterior is \\(p(\\theta|\\text{data}) \\propto \\theta^h(1-\\theta)^t \\cdot \\theta^{\\alpha-1}(1-\\theta)^{\\beta-1} = \\theta^{\\alpha+h-1}(1-\\theta)^{\\beta+t-1}\\), which is the kernel of \\(\\text{Beta}(\\alpha+h, \\beta+t)\\). This is the conjugate prior property.'
                }
            ]
        },

        // ========== SECTION 2: Expected Loss & Risk ==========
        {
            id: 'ch01-sec02',
            title: 'Expected Loss & Risk',
            content: `
<h2>1.2 Expected Loss & Risk</h2>

<div class="env-block intuition">
<strong>From Probability to Decisions.</strong> Probability tells us what might happen; <em>decision theory</em> tells us what to do about it. The central idea: define a loss function measuring the cost of errors, then choose the action that minimizes expected loss.
</div>

<h3>Loss Functions</h3>

<p>A <em>loss function</em> \\(L(y, \\hat{y})\\) quantifies the penalty for predicting \\(\\hat{y}\\) when the true value is \\(y\\). Common choices:</p>

<ul>
<li><strong>0-1 loss</strong> (classification): \\(L(y, \\hat{y}) = \\mathbb{1}[y \\neq \\hat{y}]\\). Every misclassification costs 1.</li>
<li><strong>Squared loss</strong> (regression): \\(L(y, \\hat{y}) = (y - \\hat{y})^2\\). Penalizes large errors quadratically.</li>
<li><strong>Absolute loss:</strong> \\(L(y, \\hat{y}) = |y - \\hat{y}|\\). More robust to outliers than squared loss.</li>
</ul>

<h3>Expected Loss (Risk)</h3>

<div class="env-block definition">
<strong>Definition 1.2.1 (Risk).</strong> The <em>risk</em> (expected loss) of a decision rule \\(\\hat{y}(\\mathbf{x})\\) is
\\[
R(\\hat{y}) = \\mathbb{E}_{(\\mathbf{x},y)}\\big[L(y, \\hat{y}(\\mathbf{x}))\\big] = \\int L(y, \\hat{y}(\\mathbf{x}))\\,p(\\mathbf{x}, y)\\,d\\mathbf{x}\\,dy.
\\]
The <em>Bayes-optimal</em> decision rule minimizes risk over all measurable functions.
</div>

<h3>Bayes Optimal Classifier</h3>

<div class="env-block theorem">
<strong>Theorem 1.2.2 (Bayes Classifier).</strong> Under 0-1 loss, the risk-minimizing classifier assigns each \\(\\mathbf{x}\\) to the class with highest posterior probability:
\\[
\\hat{y}^*(\\mathbf{x}) = \\arg\\max_k\\, p(C_k \\mid \\mathbf{x}).
\\]
The resulting error rate, called the <em>Bayes error rate</em>, is \\(R^* = \\mathbb{E}[1 - \\max_k p(C_k \\mid \\mathbf{x})]\\). No classifier can achieve lower error.
</div>

<div class="env-block remark">
<strong>Proof sketch.</strong> Write the risk as \\(R = \\int \\sum_k \\mathbb{1}[\\hat{y}(\\mathbf{x}) \\neq k]\\,p(C_k \\mid \\mathbf{x})\\,p(\\mathbf{x})\\,d\\mathbf{x}\\). For each \\(\\mathbf{x}\\), the integrand is minimized by choosing \\(\\hat{y}\\) to be the class with largest \\(p(C_k \\mid \\mathbf{x})\\), giving pointwise loss \\(1 - \\max_k p(C_k \\mid \\mathbf{x})\\).
</div>

<h3>Decision Boundaries</h3>

<p>For two classes with Gaussian class-conditionals \\(p(\\mathbf{x} \\mid C_k) = \\mathcal{N}(\\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)\\), the Bayes decision boundary is the set \\(\\{\\mathbf{x} : p(C_1 \\mid \\mathbf{x}) = p(C_2 \\mid \\mathbf{x})\\}\\). When both classes share the same covariance \\(\\boldsymbol{\\Sigma}\\), this boundary is <strong>linear</strong> (a hyperplane). Unequal covariances produce quadratic boundaries.</p>

<div class="viz-placeholder" data-viz="viz-bayes-boundary"></div>
`,
            visualizations: [
                {
                    id: 'viz-bayes-boundary',
                    title: 'Bayes-Optimal Decision Boundary',
                    description: 'Two 1D Gaussians with equal variance. Drag the means to see the optimal decision boundary move. The shaded regions show classification errors (Bayes error rate).',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 50, originX: null, originY: null });
                        viz.originX = viz.width / 2;
                        viz.originY = viz.height * 0.72;
                        let mu1 = -1.5, mu2 = 1.5, sigma = 1.0, pi1 = 0.5;
                        VizEngine.createSlider(controls, '\u03bc\u2081', -4, 1, mu1, 0.1, v => { mu1 = v; });
                        VizEngine.createSlider(controls, '\u03bc\u2082', -1, 4, mu2, 0.1, v => { mu2 = v; });
                        VizEngine.createSlider(controls, '\u03c3', 0.3, 2.5, sigma, 0.1, v => { sigma = v; });
                        VizEngine.createSlider(controls, 'p(C\u2081)', 0.05, 0.95, pi1, 0.05, v => { pi1 = v; });

                        function gauss(x, mu) {
                            return Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
                        }
                        function shadeError(ctx, xMin, xMax, steps, boundary, mu, w, color, side) {
                            ctx.beginPath();
                            let started = false;
                            for (let i = 0; i <= steps; i++) {
                                const x = xMin + (xMax - xMin) * i / steps;
                                const inRegion = side === 'right' ? x >= boundary : x <= boundary;
                                if (inRegion) {
                                    const [sx, sy] = viz.toScreen(x, w * gauss(x, mu));
                                    if (!started) { const [sb] = viz.toScreen(side === 'right' ? boundary : xMin, 0); ctx.moveTo(sb, viz.toScreen(0,0)[1]); started = true; }
                                    ctx.lineTo(sx, sy);
                                }
                            }
                            if (started) {
                                const endX = side === 'right' ? xMax : boundary;
                                ctx.lineTo(viz.toScreen(endX, 0)[0], viz.toScreen(0,0)[1]);
                                ctx.closePath(); ctx.fillStyle = color + '30'; ctx.fill();
                            }
                        }
                        function draw() {
                            viz.clear(); viz.drawGrid(); viz.drawAxes();
                            const ctx = viz.ctx;
                            const xMin = -6, xMax = 6, steps = 300;
                            const pi2 = 1 - pi1;
                            const dmu = mu2 - mu1;
                            let boundary = (mu1 + mu2) / 2;
                            if (Math.abs(dmu) > 1e-8) boundary += sigma * sigma / dmu * Math.log(pi2 / pi1);

                            shadeError(ctx, xMin, xMax, steps, boundary, mu1, pi1, viz.colors.blue, 'right');
                            shadeError(ctx, xMin, xMax, steps, boundary, mu2, pi2, viz.colors.red, 'left');

                            // Draw class-conditional densities (weighted by prior)
                            [{ mu: mu1, col: viz.colors.blue, lbl: 'p(x|C\u2081)\u00b7p(C\u2081)' },
                             { mu: mu2, col: viz.colors.red, lbl: 'p(x|C\u2082)\u00b7p(C\u2082)' }].forEach((cls, ci) => {
                                const w = ci === 0 ? pi1 : pi2;
                                ctx.beginPath();
                                for (let i = 0; i <= steps; i++) {
                                    const x = xMin + (xMax - xMin) * i / steps;
                                    const [sx, sy] = viz.toScreen(x, w * gauss(x, cls.mu));
                                    i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                                }
                                ctx.strokeStyle = cls.col; ctx.lineWidth = 2.5; ctx.stroke();
                            });

                            // Decision boundary line
                            const [bx, by1] = viz.toScreen(boundary, 0);
                            const [, by2] = viz.toScreen(boundary, 0.5);
                            ctx.setLineDash([6, 4]); ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 2;
                            ctx.beginPath(); ctx.moveTo(bx, by1); ctx.lineTo(bx, by2); ctx.stroke();
                            ctx.setLineDash([]);

                            // Compute Bayes error numerically
                            let bayesErr = 0;
                            const dx = (xMax - xMin) / steps;
                            for (let i = 0; i <= steps; i++) {
                                const x = xMin + dx * i;
                                const p1 = pi1 * gauss(x, mu1);
                                const p2 = pi2 * gauss(x, mu2);
                                bayesErr += Math.min(p1, p2) * dx;
                            }

                            // Labels
                            viz.screenText('Decision boundary at x=' + boundary.toFixed(2) +
                                '    Bayes error = ' + (bayesErr * 100).toFixed(1) + '%',
                                viz.width / 2, 16, viz.colors.white, 12);
                            viz.screenText('C\u2081', viz.width * 0.25, viz.height - 10, viz.colors.blue, 13);
                            viz.screenText('C\u2082', viz.width * 0.75, viz.height - 10, viz.colors.red, 13);
                        }
                        viz.animate(draw);
                        return { stopAnimation: () => viz.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Under squared loss \\(L(y, \\hat{y}) = (y - \\hat{y})^2\\), show that the Bayes-optimal prediction is the conditional mean \\(\\hat{y}^*(\\mathbf{x}) = \\mathbb{E}[Y \\mid \\mathbf{x}]\\).',
                    hint: 'Write the conditional risk \\(\\mathbb{E}[(Y - \\hat{y})^2 \\mid \\mathbf{x}]\\), expand, and minimize over \\(\\hat{y}\\).',
                    solution: '\\(\\mathbb{E}[(Y-a)^2 | \\mathbf{x}] = \\mathbb{E}[Y^2|\\mathbf{x}] - 2a\\mathbb{E}[Y|\\mathbf{x}] + a^2\\). Taking derivative w.r.t. \\(a\\) and setting to zero: \\(-2\\mathbb{E}[Y|\\mathbf{x}] + 2a = 0\\), so \\(a^* = \\mathbb{E}[Y|\\mathbf{x}]\\). The residual risk is \\(\\text{Var}(Y|\\mathbf{x})\\), the irreducible noise.'
                },
                {
                    question: 'Under absolute loss \\(L(y, \\hat{y}) = |y - \\hat{y}|\\), the optimal prediction is the <em>conditional median</em>. Why does this make the prediction more robust to outliers than the conditional mean?',
                    hint: 'Think about what happens to the mean vs. median when a single observation is moved far away.',
                    solution: 'The median minimizes \\(\\mathbb{E}[|Y - a| \\mid \\mathbf{x}]\\). Unlike the mean, which weights each observation linearly (and thus is pulled by extreme values), the median depends only on the <em>rank order</em> of observations. Moving one point to infinity changes the mean but not the median (as long as it stays on the same side).'
                },
                {
                    question: 'For a two-class problem with unequal misclassification costs \\(L_{12}\\) (cost of predicting class 2 when truth is 1) and \\(L_{21}\\), show the optimal decision boundary is at \\(p(C_1|\\mathbf{x})/p(C_2|\\mathbf{x}) = L_{21}/L_{12}\\).',
                    hint: 'Write the expected loss for predicting each class and equate them.',
                    solution: 'Predict \\(C_1\\) when \\(L_{21}\\,p(C_2|\\mathbf{x}) &lt; L_{12}\\,p(C_1|\\mathbf{x})\\), i.e., the expected cost of a wrong \\(C_1\\) prediction exceeds that of a wrong \\(C_2\\). Rearranging: \\(p(C_1|\\mathbf{x})/p(C_2|\\mathbf{x}) &gt; L_{21}/L_{12}\\). At equality we have the boundary. When costs are equal, this reduces to the standard Bayes classifier.'
                }
            ]
        },

        // ========== SECTION 3: Bias-Variance Decomposition ==========
        {
            id: 'ch01-sec03',
            title: 'Bias-Variance Decomposition',
            content: `
<h2>1.3 Bias-Variance Decomposition</h2>

<div class="env-block intuition">
<strong>The Fundamental Tradeoff.</strong> A model that is too simple misses real patterns (high bias). A model that is too complex chases noise in the training data (high variance). The bias-variance decomposition makes this tradeoff precise.
</div>

<h3>Setup</h3>

<p>Suppose the true relationship is \\(y = f(\\mathbf{x}) + \\varepsilon\\), where \\(\\varepsilon\\) is zero-mean noise with variance \\(\\sigma^2\\). Given a training set \\(\\mathcal{D}\\), we learn an estimator \\(\\hat{f}_{\\mathcal{D}}(\\mathbf{x})\\). The expectation \\(\\mathbb{E}_{\\mathcal{D}}\\) is over different training sets of the same size.</p>

<div class="env-block theorem">
<strong>Theorem 1.3.1 (Bias-Variance Decomposition).</strong> Under squared loss, the expected prediction error at a point \\(\\mathbf{x}\\) decomposes as
\\[
\\mathbb{E}_{\\mathcal{D}}\\big[(y - \\hat{f}_{\\mathcal{D}}(\\mathbf{x}))^2\\big] = \\underbrace{\\big(f(\\mathbf{x}) - \\mathbb{E}_{\\mathcal{D}}[\\hat{f}_{\\mathcal{D}}(\\mathbf{x})]\\big)^2}_{\\text{bias}^2} + \\underbrace{\\mathbb{E}_{\\mathcal{D}}\\big[(\\hat{f}_{\\mathcal{D}}(\\mathbf{x}) - \\mathbb{E}_{\\mathcal{D}}[\\hat{f}_{\\mathcal{D}}(\\mathbf{x})])^2\\big]}_{\\text{variance}} + \\underbrace{\\sigma^2}_{\\text{noise}}.
\\]
</div>

<div class="env-block remark">
<strong>Proof.</strong> Let \\(\\bar{f} = \\mathbb{E}_{\\mathcal{D}}[\\hat{f}]\\). Since \\(y = f + \\varepsilon\\) and \\(\\varepsilon\\) is independent of \\(\\hat{f}\\):
\\[
\\mathbb{E}[(y - \\hat{f})^2] = \\mathbb{E}[(f + \\varepsilon - \\hat{f})^2] = \\mathbb{E}[(f - \\hat{f})^2] + \\sigma^2
\\]
(the cross term \\(2\\mathbb{E}[\\varepsilon(f - \\hat{f})]\\) vanishes by independence). Now add and subtract \\(\\bar{f}\\):
\\[
\\mathbb{E}[(f - \\hat{f})^2] = \\mathbb{E}[(f - \\bar{f} + \\bar{f} - \\hat{f})^2] = (f - \\bar{f})^2 + \\mathbb{E}[(\\bar{f} - \\hat{f})^2]
\\]
since the cross term \\(2(f - \\bar{f})\\mathbb{E}[\\bar{f} - \\hat{f}] = 0\\) by definition of \\(\\bar{f}\\).
</div>

<h3>Interpretation</h3>

<ul>
<li><strong>Bias</strong> measures systematic error: how far the average model is from the truth. High bias means underfitting.</li>
<li><strong>Variance</strong> measures sensitivity to the training data: how much the model changes across different samples. High variance means overfitting.</li>
<li><strong>Noise</strong> (\\(\\sigma^2\\)) is irreducible. No model can do better than this.</li>
</ul>

<p>Model complexity controls the tradeoff. Simple models (few parameters) have high bias, low variance. Complex models (many parameters) have low bias, high variance. The optimal complexity minimizes total error.</p>

<div class="viz-placeholder" data-viz="viz-bias-variance"></div>
`,
            visualizations: [
                {
                    id: 'viz-bias-variance',
                    title: 'Bias-Variance Tradeoff',
                    description: 'Polynomial fits of different degrees to noisy samples from \\(f(x) = \\sin(\\pi x)\\). Low degree = high bias. High degree = high variance. Click "Resample" to draw new data and see variance in action.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 60, originX: null, originY: null });
                        viz.originX = viz.width * 0.12;
                        viz.originY = viz.height * 0.55;
                        let degree = 1, nSamples = 15, noiseLevel = 0.3;
                        let seed = 1;
                        const fits = []; // store multiple fits to show variance

                        VizEngine.createSlider(controls, 'Degree', 1, 12, degree, 1, v => { degree = Math.round(v); fits.length = 0; seed = 1; });
                        VizEngine.createSlider(controls, 'Noise \u03c3', 0.05, 0.8, noiseLevel, 0.05, v => { noiseLevel = v; fits.length = 0; seed = 1; });
                        VizEngine.createButton(controls, 'Resample', () => {
                            seed++;
                            if (fits.length > 8) fits.shift();
                            fits.push(generateFit());
                        });

                        function pseudoRand(s) {
                            return ((Math.sin(s * 127.1) * 43758.5453) % 1 + 1) % 1;
                        }
                        function boxMuller(s) {
                            const u1 = Math.max(pseudoRand(s), 1e-10);
                            const u2 = pseudoRand(s + 0.5);
                            return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                        }
                        function trueF(x) { return Math.sin(Math.PI * x); }

                        function generateFit() {
                            const xs = [], ys = [];
                            for (let i = 0; i < nSamples; i++) {
                                const x = pseudoRand(seed * 100 + i * 7.3) * 2 - 0.2;
                                const y = trueF(x) + noiseLevel * boxMuller(seed * 200 + i * 13.7);
                                xs.push(x); ys.push(y);
                            }
                            // Fit polynomial via normal equations
                            const d = Math.min(degree, nSamples - 1);
                            const X = xs.map(x => {
                                const row = [];
                                for (let j = 0; j <= d; j++) row.push(Math.pow(x, j));
                                return row;
                            });
                            // (X^T X)^{-1} X^T y via simple Gaussian elimination
                            const n = d + 1;
                            const XtX = Array.from({ length: n }, (_, i) =>
                                Array.from({ length: n }, (_, j) =>
                                    xs.reduce((s, x, k) => s + Math.pow(x, i) * Math.pow(x, j), 0)));
                            const Xty = Array.from({ length: n }, (_, i) =>
                                xs.reduce((s, x, k) => s + Math.pow(x, i) * ys[k], 0));
                            // Augmented matrix
                            const A = XtX.map((row, i) => [...row, Xty[i]]);
                            for (let col = 0; col < n; col++) {
                                let maxR = col;
                                for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[maxR][col])) maxR = r;
                                [A[col], A[maxR]] = [A[maxR], A[col]];
                                if (Math.abs(A[col][col]) < 1e-12) continue;
                                const piv = A[col][col];
                                for (let j = col; j <= n; j++) A[col][j] /= piv;
                                for (let r = 0; r < n; r++) {
                                    if (r === col) continue;
                                    const f = A[r][col];
                                    for (let j = col; j <= n; j++) A[r][j] -= f * A[col][j];
                                }
                            }
                            const coeffs = A.map(row => row[n]);
                            return { xs, ys, coeffs, d };
                        }
                        // Initial fit
                        fits.push(generateFit());

                        function evalPoly(coeffs, x) {
                            let v = 0;
                            for (let i = 0; i < coeffs.length; i++) v += coeffs[i] * Math.pow(x, i);
                            return v;
                        }

                        function draw() {
                            viz.clear(); viz.drawGrid(); viz.drawAxes();
                            const ctx = viz.ctx;
                            const xMin = -0.3, xMax = 2.1, steps = 200;

                            // True function
                            ctx.beginPath();
                            for (let i = 0; i <= steps; i++) {
                                const x = xMin + (xMax - xMin) * i / steps;
                                const [sx, sy] = viz.toScreen(x, trueF(x));
                                i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                            }
                            ctx.strokeStyle = viz.colors.white; ctx.lineWidth = 2; ctx.stroke();

                            // Previous fits (faded)
                            fits.forEach((fit, fi) => {
                                const alpha = fi === fits.length - 1 ? 'cc' : '44';
                                const lw = fi === fits.length - 1 ? 2.5 : 1;
                                ctx.beginPath();
                                for (let i = 0; i <= steps; i++) {
                                    const x = xMin + (xMax - xMin) * i / steps;
                                    let y = evalPoly(fit.coeffs, x);
                                    y = Math.max(-3, Math.min(3, y));
                                    const [sx, sy] = viz.toScreen(x, y);
                                    i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                                }
                                ctx.strokeStyle = viz.colors.teal + alpha; ctx.lineWidth = lw; ctx.stroke();
                            });

                            // Data points of latest fit
                            const latest = fits[fits.length - 1];
                            if (latest) {
                                latest.xs.forEach((x, i) => {
                                    viz.drawPoint(x, latest.ys[i], viz.colors.orange, null, 3);
                                });
                            }

                            // Labels
                            viz.screenText('White = true f(x)    Teal = polynomial fit (degree ' + degree + ')    Orange = data',
                                viz.width / 2, 16, viz.colors.text, 11);
                            viz.screenText('Resample multiple times to see variance across datasets',
                                viz.width / 2, viz.height - 10, viz.colors.text, 11);
                        }
                        viz.animate(draw);
                        return { stopAnimation: () => viz.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'For a constant predictor \\(\\hat{f}(\\mathbf{x}) = c\\) (ignoring the input), what are the bias and variance? What is the optimal \\(c\\)?',
                    hint: 'The variance is zero since the model does not depend on the training data. The bias depends on how \\(c\\) relates to \\(f(\\mathbf{x})\\).',
                    solution: 'Variance = 0 (no randomness in the model). Bias\\(^2\\) at point \\(\\mathbf{x}\\) is \\((f(\\mathbf{x}) - c)^2\\). Integrating over \\(\\mathbf{x}\\), the optimal \\(c\\) is \\(\\mathbb{E}[f(\\mathbf{x})]\\), the overall mean of the true function. This gives maximum bias, zero variance.'
                },
                {
                    question: 'Show that for the \\(k\\)-nearest neighbors estimator, as \\(k \\to n\\) (all training points), variance decreases to zero but bias increases. What happens as \\(k \\to 1\\)?',
                    hint: 'At \\(k=n\\), the prediction is the global training average. At \\(k=1\\), it is the value of the nearest training point.',
                    solution: 'At \\(k=n\\): \\(\\hat{f}(\\mathbf{x}) = \\bar{y}\\) for all \\(\\mathbf{x}\\), which has variance \\(\\sigma^2/n \\to 0\\) but bias \\(f(\\mathbf{x}) - \\mathbb{E}[\\bar{y}]\\). At \\(k=1\\): the model interpolates the training data, giving low bias (approximately \\(f(\\mathbf{x})\\) plus noise) but high variance (\\(\\approx \\sigma^2\\) from a single noisy observation). This illustrates the bias-variance tradeoff controlled by \\(k\\).'
                },
                {
                    question: 'Suppose we average \\(M\\) independently trained models, each with bias \\(b\\) and variance \\(v\\). What are the bias and variance of the averaged model? How does this connect to bagging (Ch 10)?',
                    hint: 'Averaging does not change the mean (bias unchanged). Variance of the average of independent quantities scales as \\(1/M\\).',
                    solution: 'The averaged model has bias \\(b\\) (unchanged) and variance \\(v/M\\). This is the core idea behind <em>bagging</em> (bootstrap aggregating): by training \\(M\\) models on different bootstrap samples and averaging, we reduce variance without increasing bias. Random forests extend this further by decorrelating the models.'
                }
            ]
        },

        // ========== SECTION 4: The Curse of Dimensionality ==========
        {
            id: 'ch01-sec04',
            title: 'The Curse of Dimensionality',
            content: `
<h2>1.4 The Curse of Dimensionality</h2>

<div class="env-block intuition">
<strong>High Dimensions Are Strange.</strong> In high-dimensional spaces, our low-dimensional intuitions break down catastrophically. Data becomes sparse, distances concentrate, and the volume of space grows so fast that exponentially more data is needed to maintain the same density.
</div>

<h3>Volume Explosion</h3>

<p>Consider the unit hypercube \\([0,1]^d\\). To capture a fraction \\(r\\) of the data range along each axis, we need a sub-cube with side length \\(r\\). This sub-cube contains a fraction \\(r^d\\) of the total volume. To capture 10% of the volume in \\(d=10\\) dimensions, we need a side length of \\(0.1^{1/10} = 0.794\\), covering 79.4% of the range in each dimension.</p>

<div class="env-block theorem">
<strong>Theorem 1.4.1 (Volume Concentration).</strong> For uniformly distributed points in \\([0,1]^d\\), the median distance to the nearest neighbor scales as
\\[
d_{\\text{NN}} \\sim \\left(\\frac{1}{n}\\right)^{1/d}
\\]
where \\(n\\) is the number of data points. As \\(d\\) grows, \\(d_{\\text{NN}} \\to 1\\) unless \\(n\\) grows exponentially in \\(d\\).
</div>

<h3>Distance Concentration</h3>

<p>In high dimensions, distances between points concentrate. For i.i.d. uniform points in \\([0,1]^d\\), the ratio of the farthest to nearest neighbor distance approaches 1:</p>

\\[
\\frac{\\max_{i} \\|\\mathbf{x}_i - \\mathbf{x}_0\\|}{\\min_{i} \\|\\mathbf{x}_i - \\mathbf{x}_0\\|} \\xrightarrow{d \\to \\infty} 1.
\\]

<p>This means nearest-neighbor methods degrade: when all points are roughly equidistant, the "nearest" neighbor carries little information.</p>

<h3>Implications for ML</h3>

<ul>
<li><strong>Sample complexity:</strong> To maintain a fixed density of points, sample size must grow as \\(n = O(m^d)\\) where \\(m\\) is the number of points per axis.</li>
<li><strong>Feature selection:</strong> Irrelevant features add dimensions without information, harming all distance-based methods.</li>
<li><strong>Blessing of structure:</strong> Real data often lies on low-dimensional manifolds. Algorithms that exploit this (PCA, manifold learning) circumvent the curse.</li>
</ul>

<div class="viz-placeholder" data-viz="viz-curse-dim"></div>
`,
            visualizations: [
                {
                    id: 'viz-curse-dim',
                    title: 'The Curse of Dimensionality',
                    description: 'Left: 50 random points in 1D, 2D, and 3D (projected), showing how space empties. Right: average nearest-neighbor distance vs. dimension for 100 points.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 40, originX: 30, originY: null });
                        viz.originY = viz.height / 2;
                        let nPoints = 50;
                        VizEngine.createSlider(controls, 'n points', 10, 200, nPoints, 10, v => { nPoints = Math.round(v); });

                        function pseudoRand(s) {
                            return ((Math.sin(s * 127.1) * 43758.5453) % 1 + 1) % 1;
                        }

                        function draw() {
                            viz.clear();
                            const ctx = viz.ctx;
                            const W = viz.width, H = viz.height;
                            const panelW = W / 2 - 20;

                            // === Left panel: scatter in 1D, 2D, 3D ===
                            ctx.fillStyle = viz.colors.text;
                            ctx.font = 'bold 11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';

                            const dims = [1, 2, 3];
                            const panelH = (H - 40) / 3;

                            dims.forEach((d, di) => {
                                const y0 = 20 + di * panelH;
                                const boxW = panelW - 30;
                                const boxH = panelH - 15;

                                // Background
                                ctx.fillStyle = viz.colors.grid;
                                ctx.fillRect(15, y0, boxW, boxH);
                                ctx.strokeStyle = viz.colors.axis;
                                ctx.lineWidth = 1;
                                ctx.strokeRect(15, y0, boxW, boxH);

                                // Label
                                ctx.fillStyle = viz.colors.white;
                                ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillText(d + 'D', 18, y0 + 13);

                                // Points
                                for (let i = 0; i < nPoints; i++) {
                                    const coords = [];
                                    for (let j = 0; j < d; j++) {
                                        coords.push(pseudoRand(i * 17.3 + j * 31.7 + d * 59));
                                    }
                                    let px, py;
                                    if (d === 1) {
                                        px = 15 + coords[0] * boxW;
                                        py = y0 + boxH / 2;
                                    } else if (d === 2) {
                                        px = 15 + coords[0] * boxW;
                                        py = y0 + coords[1] * boxH;
                                    } else {
                                        // 3D projected
                                        px = 15 + (coords[0] + 0.3 * coords[2]) * boxW * 0.7 + boxW * 0.1;
                                        py = y0 + (coords[1] + 0.3 * coords[2]) * boxH * 0.7 + boxH * 0.1;
                                    }
                                    ctx.fillStyle = viz.colors.teal;
                                    ctx.beginPath();
                                    ctx.arc(px, py, d === 1 ? 3 : 2.5, 0, Math.PI * 2);
                                    ctx.fill();
                                }
                            });

                            // === Right panel: NN distance vs dimension ===
                            const rx = W / 2 + 20, rw = panelW - 20, rh = H - 50, ry = 25;

                            ctx.fillStyle = viz.colors.grid;
                            ctx.fillRect(rx, ry, rw, rh);
                            ctx.strokeStyle = viz.colors.axis;
                            ctx.strokeRect(rx, ry, rw, rh);

                            ctx.fillStyle = viz.colors.white;
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Avg NN distance vs. dimension (n=' + nPoints + ')', rx + rw / 2, 18);

                            const maxDim = 20;
                            const nnDists = [];
                            for (let dim = 1; dim <= maxDim; dim++) {
                                // Theoretical expected NN distance for uniform in [0,1]^d
                                // Approximation: Gamma(1+1/d) / (n * V_d)^(1/d) ~ (1/n)^(1/d) * c_d
                                // Simple: E[NN] ~ (1/n)^(1/d) for unit cube
                                const nnDist = Math.pow(1 / nPoints, 1 / dim);
                                nnDists.push(nnDist);
                            }

                            const maxNND = 1.05;
                            ctx.fillStyle = viz.colors.text; ctx.font = '10px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            for (let d = 5; d <= maxDim; d += 5) { const px = rx + (d / maxDim) * rw; ctx.fillText(d, px, ry + rh + 12); }
                            ctx.fillText('dim', rx + rw / 2, ry + rh + 24);
                            ctx.textAlign = 'right';
                            for (let v = 0; v <= 1; v += 0.2) {
                                const py = ry + rh - (v / maxNND) * rh;
                                ctx.fillText(v.toFixed(1), rx - 4, py + 3);
                                ctx.strokeStyle = viz.colors.grid; ctx.beginPath(); ctx.moveTo(rx, py); ctx.lineTo(rx + rw, py); ctx.stroke();
                            }

                            ctx.beginPath();
                            nnDists.forEach((d, i) => { const px = rx + ((i+1)/maxDim)*rw, py = ry+rh-(d/maxNND)*rh; i === 0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py); });
                            ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 2.5; ctx.stroke();
                            nnDists.forEach((d, i) => { const px = rx+((i+1)/maxDim)*rw, py = ry+rh-(d/maxNND)*rh; ctx.fillStyle = viz.colors.orange; ctx.beginPath(); ctx.arc(px,py,3,0,Math.PI*2); ctx.fill(); });
                            ctx.setLineDash([4,4]); ctx.strokeStyle = viz.colors.red+'88'; ctx.lineWidth = 1;
                            const refY = ry+rh-(1.0/maxNND)*rh;
                            ctx.beginPath(); ctx.moveTo(rx,refY); ctx.lineTo(rx+rw,refY); ctx.stroke(); ctx.setLineDash([]);
                            ctx.fillStyle = viz.colors.red; ctx.textAlign = 'left';
                            ctx.fillText('all equidistant', rx+rw-70, refY-6);
                        }
                        viz.animate(draw);
                        return { stopAnimation: () => viz.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'In a unit hypercube \\([0,1]^d\\), what fraction of the volume lies in the "shell" between radius 1 and \\(1-\\epsilon\\) from a corner, for small \\(\\epsilon\\)? Compute this for \\(d = 100\\) and \\(\\epsilon = 0.01\\).',
                    hint: 'The inscribed hypersphere has radius 1, but the volume ratio of sphere to cube shrinks with \\(d\\). Think about how volume concentrates near the boundary.',
                    solution: 'Consider the ball of radius 1 centered at a corner. The volume of the ball of radius \\(r\\) in the positive orthant scales as \\(r^d\\). So the fraction in the shell is \\(1 - (1-\\epsilon)^d\\). For \\(d=100, \\epsilon=0.01\\): \\(1 - 0.99^{100} \\approx 1 - 0.366 = 63.4\\%\\). Nearly two-thirds of the volume is in a thin shell, illustrating extreme volume concentration.'
                },
                {
                    question: 'If you need \\(m = 10\\) training points per axis to get good coverage in 1D, how many total points do you need in \\(d = 10\\) dimensions? What does this imply for practical ML?',
                    hint: 'On a grid, the total number of points is \\(m^d\\).',
                    solution: '\\(10^{10} = 10\\) billion points. This exponential scaling is exactly the curse of dimensionality. In practice, we cannot fill high-dimensional spaces with data, so we must rely on structural assumptions (smoothness, sparsity, low-dimensional manifolds) to make learning feasible.'
                },
                {
                    question: 'Why does the curse of dimensionality affect \\(k\\)-nearest neighbors more severely than linear regression?',
                    hint: 'Think about the implicit assumptions each method makes and how many effective parameters each has.',
                    solution: 'KNN makes no assumptions about \\(f\\); it is a nonparametric method whose effective number of parameters grows with \\(n\\). It requires local density of points to estimate \\(f(\\mathbf{x})\\), which fails in high \\(d\\). Linear regression assumes \\(f\\) is linear, requiring only \\(d+1\\) parameters regardless of \\(n\\). This strong assumption acts as a regularizer, making it robust to high dimensions (as long as the assumption holds). The tradeoff: KNN has low bias but suffers from high-dimensional variance; linear regression has bias when \\(f\\) is nonlinear but tightly controlled variance.'
                }
            ]
        },

        // ========== SECTION 5: No Free Lunch Theorem ==========
        {
            id: 'ch01-sec05',
            title: 'No Free Lunch Theorem',
            content: `
<h2>1.5 No Free Lunch Theorem</h2>

<div class="env-block intuition">
<strong>No Universal Winner.</strong> A natural question after studying different algorithms: is there one that always works best? The No Free Lunch (NFL) theorem gives a definitive answer: <em>no</em>. Averaged over all possible problems, every algorithm performs the same. Superior performance on one class of problems must come at the cost of inferior performance on another.
</div>

<h3>Formal Statement</h3>

<div class="env-block theorem">
<strong>Theorem 1.5.1 (No Free Lunch, Wolpert & Macready 1997).</strong> Let \\(\\mathcal{A}_1\\) and \\(\\mathcal{A}_2\\) be any two learning algorithms. Let \\(E_f\\) denote the expected off-training-set error under target function \\(f\\). Then, averaged uniformly over all possible target functions \\(f: \\mathcal{X} \\to \\mathcal{Y}\\):
\\[
\\sum_f E_f(\\mathcal{A}_1) = \\sum_f E_f(\\mathcal{A}_2).
\\]
No algorithm has lower average error than any other when averaged over all problems.
</div>

<h3>What NFL Means (and What It Does Not)</h3>

<ul>
<li><strong>Does mean:</strong> Every algorithm has a class of problems where it fails. Claims of universal superiority are unfounded.</li>
<li><strong>Does mean:</strong> The choice of algorithm encodes assumptions about the problem. Good ML practice is about matching algorithms to problem structure.</li>
<li><strong>Does not mean:</strong> All algorithms are equal on a <em>specific</em> problem. For real-world problems (which have structure), some algorithms vastly outperform others.</li>
<li><strong>Does not mean:</strong> Algorithm comparison is pointless. It means we should compare on problems of interest, not in the abstract.</li>
</ul>

<h3>Inductive Bias</h3>

<p>Every learning algorithm has an <em>inductive bias</em>: the set of assumptions it makes about the target function beyond what the training data provides.</p>

<div class="env-block definition">
<strong>Definition 1.5.2 (Inductive Bias).</strong> Examples of inductive biases:
<ul>
<li><strong>Linear regression:</strong> the target is a linear function of features.</li>
<li><strong>Decision trees:</strong> the target can be approximated by axis-aligned splits.</li>
<li><strong>KNN:</strong> nearby points have similar labels (smoothness assumption).</li>
<li><strong>SVMs:</strong> classes are separable with a large margin in some feature space.</li>
<li><strong>Neural networks:</strong> the target has a hierarchical, compositional structure.</li>
</ul>
An algorithm succeeds when its inductive bias matches the true structure of the problem.
</div>

<div class="env-block remark">
<strong>Practical Takeaway.</strong> NFL motivates the ML workflow: (1) understand the problem structure, (2) choose algorithms whose inductive biases match, (3) validate empirically. Model selection (Ch 3), cross-validation, and ensemble methods (Ch 10) are all responses to the reality that no single model dominates.
</div>

<div class="viz-placeholder" data-viz="viz-no-free-lunch"></div>
`,
            visualizations: [
                {
                    id: 'viz-no-free-lunch',
                    title: 'No Free Lunch: Different Problems Favor Different Algorithms',
                    description: 'Two datasets where different algorithms win. Linear boundary (left) favors linear models; nonlinear boundary (right) favors flexible models. Toggle to see each algorithm\'s decision regions.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 40, originX: 30, originY: null });
                        viz.originY = viz.height / 2;
                        let showAlg = 0; // 0=linear, 1=nonlinear
                        VizEngine.createButton(controls, 'Linear Model', () => { showAlg = 0; });
                        VizEngine.createButton(controls, 'Nonlinear Model', () => { showAlg = 1; });

                        function pseudoRand(s) {
                            return ((Math.sin(s * 127.1) * 43758.5453) % 1 + 1) % 1;
                        }
                        function boxMuller(s) {
                            const u1 = Math.max(pseudoRand(s), 1e-10);
                            const u2 = pseudoRand(s + 0.5);
                            return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                        }

                        // Dataset 1: linearly separable
                        const data1 = [];
                        for (let i = 0; i < 60; i++) {
                            const x = pseudoRand(i * 7 + 1) * 2 - 1;
                            const y = pseudoRand(i * 11 + 3) * 2 - 1;
                            const label = (0.8 * x + 0.6 * y + 0.1 * boxMuller(i * 5) > 0) ? 1 : 0;
                            data1.push({ x, y, label });
                        }

                        // Dataset 2: XOR-like (nonlinear)
                        const data2 = [];
                        for (let i = 0; i < 60; i++) {
                            const x = pseudoRand(i * 13 + 7) * 2 - 1;
                            const y = pseudoRand(i * 17 + 11) * 2 - 1;
                            const label = ((x * y > 0) ? 1 : 0);
                            data2.push({ x, y, label });
                        }

                        function linearPredict(x, y) {
                            // Simple linear decision: sign(0.8x + 0.6y)
                            return (0.8 * x + 0.6 * y) > 0 ? 1 : 0;
                        }
                        function nonlinearPredict(x, y) {
                            // XOR-like: sign(x*y)
                            return (x * y) > 0 ? 1 : 0;
                        }

                        function drawPanel(data, predict, px0, py0, pw, ph, title, isGood) {
                            const ctx = viz.ctx;
                            // Decision region shading
                            const res = 4;
                            for (let ix = 0; ix < pw; ix += res) {
                                for (let iy = 0; iy < ph; iy += res) {
                                    const x = (ix / pw) * 2 - 1;
                                    const y = ((ph - iy) / ph) * 2 - 1;
                                    const pred = predict(x, y);
                                    ctx.fillStyle = pred === 1 ? viz.colors.blue + '18' : viz.colors.red + '18';
                                    ctx.fillRect(px0 + ix, py0 + iy, res, res);
                                }
                            }

                            // Border
                            ctx.strokeStyle = isGood ? viz.colors.green : viz.colors.red;
                            ctx.lineWidth = 2;
                            ctx.strokeRect(px0, py0, pw, ph);

                            // Data points
                            data.forEach(p => {
                                const sx = px0 + (p.x + 1) / 2 * pw;
                                const sy = py0 + (1 - (p.y + 1) / 2) * ph;
                                const pred = predict(p.x, p.y);
                                const correct = pred === p.label;
                                ctx.fillStyle = p.label === 1 ? viz.colors.blue : viz.colors.red;
                                ctx.beginPath(); ctx.arc(sx, sy, 3.5, 0, Math.PI * 2); ctx.fill();
                                if (!correct) {
                                    ctx.strokeStyle = viz.colors.yellow;
                                    ctx.lineWidth = 1.5;
                                    ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2); ctx.stroke();
                                }
                            });

                            // Accuracy
                            let correct = 0;
                            data.forEach(p => { if (predict(p.x, p.y) === p.label) correct++; });
                            const acc = (correct / data.length * 100).toFixed(0);

                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText(title, px0 + pw / 2, py0 - 8);
                            ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillStyle = isGood ? viz.colors.green : viz.colors.red;
                            ctx.fillText('Acc: ' + acc + '%', px0 + pw / 2, py0 + ph + 14);
                        }

                        function draw() {
                            viz.clear();
                            const ctx = viz.ctx;
                            const W = viz.width, H = viz.height;
                            const pw = (W - 60) / 2 - 10;
                            const ph = H - 70;
                            const py0 = 30;

                            const predict = showAlg === 0 ? linearPredict : nonlinearPredict;
                            const algName = showAlg === 0 ? 'Linear Model' : 'Nonlinear (XOR) Model';

                            ctx.fillStyle = viz.colors.white;
                            ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            ctx.fillText('Algorithm: ' + algName + '    (yellow rings = errors)', W / 2, 16);

                            // Panel 1: linear data
                            const good1 = showAlg === 0;
                            drawPanel(data1, predict, 20, py0, pw, ph, 'Linear Data', good1);

                            // Panel 2: XOR data
                            const good2 = showAlg === 1;
                            drawPanel(data2, predict, W / 2 + 10, py0, pw, ph, 'XOR Data', good2);
                        }
                        viz.animate(draw);
                        return { stopAnimation: () => viz.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'If NFL says no algorithm is universally best, why does deep learning seem to dominate so many benchmarks? Does this contradict the theorem?',
                    hint: 'Think about what "averaged over all problems" means vs. the distribution of problems we actually care about.',
                    solution: 'No contradiction. NFL averages over <em>all possible</em> target functions, including pathological ones no one encounters. Real-world data has structure (smoothness, compositionality, symmetry) that deep networks exploit well. Deep learning dominates on the <em>distribution of problems humans care about</em>, not on all conceivable functions. On random mappings, deep learning has no advantage.'
                },
                {
                    question: 'Give a concrete example of two datasets where (a) a linear model beats a decision tree, and (b) a decision tree beats a linear model.',
                    hint: 'Think about what each model can and cannot represent.',
                    solution: '(a) \\(y = 2x_1 + 3x_2 + \\varepsilon\\): a linear model fits exactly, while a decision tree must approximate the linear boundary with axis-aligned splits, requiring many splits and suffering high variance. (b) \\(y = \\mathbb{1}[x_1 &gt; 0 \\text{ and } x_2 &gt; 0]\\): a single tree with two splits captures this perfectly, while a linear model cannot represent the interaction.'
                },
                {
                    question: 'The NFL theorem assumes a uniform distribution over target functions. Why is this assumption crucial, and what happens if we restrict the function class?',
                    hint: 'Under a uniform prior, every function is equally likely, which removes all structure. Restricting the class introduces structure.',
                    solution: 'Under a uniform distribution, for every function \\(f\\) where algorithm \\(\\mathcal{A}\\) succeeds, there exists a "mirror" function where it fails by exactly the same amount (the function that agrees with the data but behaves oppositely elsewhere). Restricting the function class (e.g., to smooth functions, or functions with bounded complexity) breaks this symmetry. With a non-uniform prior, algorithms whose inductive bias matches the prior will outperform those that do not. This is why domain knowledge matters.'
                }
            ]
        }
    ]
});
