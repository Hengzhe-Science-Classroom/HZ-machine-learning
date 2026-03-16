// === Chapter 18: Variational Inference ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch18',
    number: 18,
    title: 'Variational Inference',
    subtitle: 'Turning intractable Bayesian posteriors into optimization problems via the ELBO, mean-field factorization, and stochastic gradients',
    sections: [
        // ===================== Section 1: The Variational Objective =====================
        {
            id: 'ch18-sec01',
            title: 'The Variational Objective',
            content: `<h2>18.1 The Variational Objective</h2>

<div class="env-block intuition">
<div class="env-title">Why Variational Inference?</div>
<div class="env-body"><p>In Bayesian inference, we seek the posterior \\(p(\\mathbf{z} \\mid \\mathbf{x})\\) over latent variables \\(\\mathbf{z}\\) given observed data \\(\\mathbf{x}\\). By Bayes' theorem,
\\[p(\\mathbf{z} \\mid \\mathbf{x}) = \\frac{p(\\mathbf{x} \\mid \\mathbf{z})\\, p(\\mathbf{z})}{p(\\mathbf{x})}.\\]
The denominator \\(p(\\mathbf{x}) = \\int p(\\mathbf{x} \\mid \\mathbf{z})\\, p(\\mathbf{z})\\, d\\mathbf{z}\\) is the <strong>evidence</strong> (or marginal likelihood). For most interesting models, this integral is intractable: the latent space is high-dimensional and the integrand has no closed form. Variational inference sidesteps this by converting the inference problem into an <em>optimization</em> problem.</p></div>
</div>

<h3>KL Divergence</h3>

<p>To measure how well an approximation \\(q(\\mathbf{z})\\) matches the true posterior \\(p(\\mathbf{z} \\mid \\mathbf{x})\\), we use the <strong>Kullback-Leibler divergence</strong>:</p>

<div class="env-block definition">
<div class="env-title">Definition 18.1.1 (KL Divergence)</div>
<div class="env-body"><p>The KL divergence from distribution \\(q\\) to distribution \\(p\\) is
\\[\\mathrm{KL}(q \\| p) = \\int q(\\mathbf{z}) \\log \\frac{q(\\mathbf{z})}{p(\\mathbf{z})} \\, d\\mathbf{z} = \\mathbb{E}_{q}\\!\\left[\\log \\frac{q(\\mathbf{z})}{p(\\mathbf{z})}\\right].\\]
Key properties: (1) \\(\\mathrm{KL}(q \\| p) \\geq 0\\) (Gibbs' inequality), (2) \\(\\mathrm{KL}(q \\| p) = 0\\) if and only if \\(q = p\\) a.e., (3) it is <em>not</em> symmetric: \\(\\mathrm{KL}(q \\| p) \\neq \\mathrm{KL}(p \\| q)\\) in general.</p></div>
</div>

<h3>Deriving the ELBO</h3>

<p>We want to minimize \\(\\mathrm{KL}(q(\\mathbf{z}) \\| p(\\mathbf{z} \\mid \\mathbf{x}))\\), but this requires the intractable posterior. The key decomposition is:</p>

\\[\\log p(\\mathbf{x}) = \\underbrace{\\mathbb{E}_{q}\\!\\left[\\log \\frac{p(\\mathbf{x}, \\mathbf{z})}{q(\\mathbf{z})}\\right]}_{\\mathcal{L}(q) \\;=\\; \\text{ELBO}} + \\mathrm{KL}(q(\\mathbf{z}) \\| p(\\mathbf{z} \\mid \\mathbf{x})).\\]

<p>Since \\(\\log p(\\mathbf{x})\\) is a constant with respect to \\(q\\), maximizing the ELBO is equivalent to minimizing the KL divergence.</p>

<div class="env-block theorem">
<div class="env-title">Theorem 18.1.2 (Evidence Lower Bound)</div>
<div class="env-body"><p>For any distribution \\(q(\\mathbf{z})\\),
\\[\\log p(\\mathbf{x}) \\geq \\mathcal{L}(q) = \\mathbb{E}_{q}[\\log p(\\mathbf{x}, \\mathbf{z})] - \\mathbb{E}_{q}[\\log q(\\mathbf{z})].\\]
Equivalently,
\\[\\mathcal{L}(q) = \\mathbb{E}_{q}[\\log p(\\mathbf{x} \\mid \\mathbf{z})] - \\mathrm{KL}(q(\\mathbf{z}) \\| p(\\mathbf{z})).\\]
The first term encourages \\(q\\) to place mass where the likelihood is high; the second keeps \\(q\\) close to the prior. The bound is tight when \\(q = p(\\mathbf{z} \\mid \\mathbf{x})\\).</p></div>
</div>

<div class="env-block remark">
<div class="env-title">Remark (Forward vs. Reverse KL)</div>
<div class="env-body"><p>VI minimizes the <em>reverse</em> KL \\(\\mathrm{KL}(q \\| p)\\), which is <strong>mode-seeking</strong>: when \\(q\\) has less capacity than \\(p\\), it tends to concentrate on one mode rather than spread across all modes. The <em>forward</em> KL \\(\\mathrm{KL}(p \\| q)\\) is <strong>mass-covering</strong>, forcing \\(q\\) to cover all regions where \\(p\\) is nonzero. This asymmetry has significant implications for approximation quality in multimodal posteriors.</p></div>
</div>

<div class="viz-placeholder" data-viz="viz-elbo-kl"></div>`,

            visualizations: [
                {
                    id: 'viz-elbo-kl',
                    title: 'True Posterior vs. Variational Approximation',
                    description: 'The blue curve shows a bimodal true posterior. The orange Gaussian is the variational approximation \\(q\\). Drag the mean or adjust the width to minimize KL divergence. Notice how the reverse KL makes \\(q\\) lock onto one mode.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, {scale: 50, originX: null, originY: null});
                        viz.originX = viz.width / 2;
                        viz.originY = viz.height * 0.78;
                        var ctx = viz.ctx;
                        var qMu = 1.5, qSig = 0.8;

                        VizEngine.createSlider(controls, 'q mean', -4, 4, qMu, 0.1, function(v) { qMu = v; });
                        VizEngine.createSlider(controls, 'q std', 0.2, 3, qSig, 0.1, function(v) { qSig = v; });

                        function truePosterior(x) {
                            var g1 = Math.exp(-0.5 * ((x + 1.5) / 0.7) * ((x + 1.5) / 0.7)) / (0.7 * Math.sqrt(2 * Math.PI));
                            var g2 = Math.exp(-0.5 * ((x - 1.5) / 0.7) * ((x - 1.5) / 0.7)) / (0.7 * Math.sqrt(2 * Math.PI));
                            return 0.5 * g1 + 0.5 * g2;
                        }
                        function qDist(x) {
                            return Math.exp(-0.5 * ((x - qMu) / qSig) * ((x - qMu) / qSig)) / (qSig * Math.sqrt(2 * Math.PI));
                        }

                        function computeKL() {
                            var kl = 0, dx = 0.02;
                            for (var x = -6; x <= 6; x += dx) {
                                var q = qDist(x), p = truePosterior(x);
                                if (q > 1e-10 && p > 1e-10) kl += q * Math.log(q / p) * dx;
                            }
                            return Math.max(0, kl);
                        }

                        function drawCurve(fn, color, fill) {
                            ctx.beginPath();
                            var steps = 300, xMin = -5, xMax = 5;
                            var sx0 = viz.toScreen(xMin, 0);
                            ctx.moveTo(sx0[0], sx0[1]);
                            for (var i = 0; i <= steps; i++) {
                                var x = xMin + (xMax - xMin) * i / steps;
                                var sc = viz.toScreen(x, fn(x));
                                ctx.lineTo(sc[0], sc[1]);
                            }
                            if (fill) {
                                ctx.lineTo(viz.toScreen(xMax, 0)[0], viz.toScreen(xMax, 0)[1]);
                                ctx.closePath();
                                ctx.fillStyle = color + '20';
                                ctx.fill();
                            }
                            ctx.beginPath();
                            for (var i = 0; i <= steps; i++) {
                                var x = xMin + (xMax - xMin) * i / steps;
                                var sc = viz.toScreen(x, fn(x));
                                i === 0 ? ctx.moveTo(sc[0], sc[1]) : ctx.lineTo(sc[0], sc[1]);
                            }
                            ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.stroke();
                        }

                        function draw() {
                            viz.clear(); viz.drawGrid(); viz.drawAxes();
                            drawCurve(truePosterior, viz.colors.blue, true);
                            drawCurve(qDist, viz.colors.orange, true);
                            var kl = computeKL();
                            viz.screenText('p(z|x) = true posterior', 120, 20, viz.colors.blue, 13, 'left');
                            viz.screenText('q(z) = variational approx', 120, 38, viz.colors.orange, 13, 'left');
                            viz.screenText('KL(q || p) = ' + kl.toFixed(3), viz.width - 20, 20, viz.colors.white, 14, 'right');
                            if (kl < 0.05) {
                                viz.screenText('Excellent fit!', viz.width - 20, 40, viz.colors.green, 12, 'right');
                            }
                        }
                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Derive the ELBO decomposition starting from \\(\\mathrm{KL}(q(\\mathbf{z}) \\| p(\\mathbf{z} \\mid \\mathbf{x}))\\). Show that \\(\\log p(\\mathbf{x}) = \\mathcal{L}(q) + \\mathrm{KL}(q \\| p(\\mathbf{z}|\\mathbf{x}))\\).',
                    hint: 'Expand \\(\\log p(\\mathbf{z} \\mid \\mathbf{x}) = \\log p(\\mathbf{x}, \\mathbf{z}) - \\log p(\\mathbf{x})\\) inside the KL definition.',
                    solution: '\\(\\mathrm{KL}(q \\| p(\\mathbf{z}|\\mathbf{x})) = \\mathbb{E}_q[\\log q(\\mathbf{z}) - \\log p(\\mathbf{z}|\\mathbf{x})] = \\mathbb{E}_q[\\log q(\\mathbf{z}) - \\log p(\\mathbf{x},\\mathbf{z}) + \\log p(\\mathbf{x})] = -\\mathcal{L}(q) + \\log p(\\mathbf{x})\\). Rearranging: \\(\\log p(\\mathbf{x}) = \\mathcal{L}(q) + \\mathrm{KL}(q \\| p(\\mathbf{z}|\\mathbf{x})) \\geq \\mathcal{L}(q)\\), since KL is non-negative.'
                },
                {
                    question: 'Explain intuitively why the reverse KL \\(\\mathrm{KL}(q \\| p)\\) is mode-seeking while the forward KL \\(\\mathrm{KL}(p \\| q)\\) is mass-covering.',
                    hint: 'Consider what happens when \\(q(z) > 0\\) but \\(p(z) \\approx 0\\) in the reverse KL, versus when \\(p(z) > 0\\) but \\(q(z) \\approx 0\\) in the forward KL.',
                    solution: 'In reverse KL \\(\\mathbb{E}_q[\\log(q/p)]\\), if \\(q(z) > 0\\) where \\(p(z) \\approx 0\\), the ratio \\(q/p \\to \\infty\\) creating a huge penalty. So \\(q\\) avoids placing mass where \\(p\\) is small, concentrating on a single mode. In forward KL \\(\\mathbb{E}_p[\\log(p/q)]\\), if \\(p(z) > 0\\) where \\(q(z) \\approx 0\\), the penalty is large. So \\(q\\) must cover all modes of \\(p\\), spreading mass broadly. This makes forward KL mass-covering and reverse KL mode-seeking.'
                },
                {
                    question: 'Show that the ELBO can be rewritten as \\(\\mathcal{L}(q) = \\mathbb{E}_q[\\log p(\\mathbf{x}|\\mathbf{z})] - \\mathrm{KL}(q(\\mathbf{z}) \\| p(\\mathbf{z}))\\). Interpret each term.',
                    hint: 'Factor \\(p(\\mathbf{x}, \\mathbf{z}) = p(\\mathbf{x}|\\mathbf{z})p(\\mathbf{z})\\) in the ELBO expression.',
                    solution: '\\(\\mathcal{L}(q) = \\mathbb{E}_q[\\log p(\\mathbf{x},\\mathbf{z})] - \\mathbb{E}_q[\\log q(\\mathbf{z})] = \\mathbb{E}_q[\\log p(\\mathbf{x}|\\mathbf{z}) + \\log p(\\mathbf{z})] - \\mathbb{E}_q[\\log q(\\mathbf{z})] = \\mathbb{E}_q[\\log p(\\mathbf{x}|\\mathbf{z})] - \\mathbb{E}_q[\\log q(\\mathbf{z}) - \\log p(\\mathbf{z})] = \\mathbb{E}_q[\\log p(\\mathbf{x}|\\mathbf{z})] - \\mathrm{KL}(q\\|p(\\mathbf{z}))\\). The first term is the expected log-likelihood (reconstruction quality). The second is a regularizer keeping \\(q\\) close to the prior.'
                }
            ]
        },

        // ===================== Section 2: Mean-Field Approximation =====================
        {
            id: 'ch18-sec02',
            title: 'Mean-Field Approximation',
            content: `<h2>18.2 Mean-Field Approximation</h2>

<div class="env-block intuition">
<div class="env-title">The Simplest Variational Family</div>
<div class="env-body"><p>The ELBO framework works for any choice of variational family \\(\\mathcal{Q}\\). The most common and simplest choice is the <strong>mean-field</strong> family, where we assume that \\(q\\) fully factorizes across latent dimensions. This eliminates all posterior correlations but makes optimization tractable via coordinate ascent.</p></div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 18.2.1 (Mean-Field Variational Family)</div>
<div class="env-body"><p>The <strong>mean-field family</strong> restricts the variational distribution to a fully factorized form:
\\[q(\\mathbf{z}) = \\prod_{j=1}^{d} q_j(z_j).\\]
Each factor \\(q_j\\) has its own parameters (e.g., mean and variance if Gaussian). The approximation ignores all correlations between latent variables. Despite this strong assumption, mean-field VI is often remarkably effective in practice.</p></div>
</div>

<h3>Coordinate Ascent Variational Inference (CAVI)</h3>

<p>Under the mean-field assumption, we can optimize each factor \\(q_j\\) while holding the others fixed. This leads to the <strong>CAVI</strong> algorithm.</p>

<div class="env-block theorem">
<div class="env-title">Theorem 18.2.2 (Optimal Mean-Field Factor)</div>
<div class="env-body"><p>Holding all factors \\(q_{k \\neq j}\\) fixed, the optimal \\(q_j^*(z_j)\\) maximizing the ELBO is
\\[\\log q_j^*(z_j) = \\mathbb{E}_{q_{-j}}[\\log p(\\mathbf{x}, \\mathbf{z})] + \\text{const},\\]
where \\(\\mathbb{E}_{q_{-j}}\\) denotes expectation over all latent variables except \\(z_j\\). The constant ensures \\(q_j^*\\) normalizes to 1.</p></div>
</div>

<div class="env-block example">
<div class="env-title">Example (Gaussian with Unknown Mean and Precision)</div>
<div class="env-body"><p>Consider data \\(x_1, \\ldots, x_n \\sim \\mathcal{N}(\\mu, \\tau^{-1})\\) with conjugate priors \\(\\mu \\sim \\mathcal{N}(\\mu_0, \\tau_0^{-1})\\) and \\(\\tau \\sim \\text{Gamma}(a_0, b_0)\\). The true posterior \\(p(\\mu, \\tau \\mid \\mathbf{x})\\) couples \\(\\mu\\) and \\(\\tau\\). Mean-field VI uses \\(q(\\mu, \\tau) = q_\\mu(\\mu)\\, q_\\tau(\\tau)\\). Applying the optimal factor formula:
\\[q_\\mu^*(\\mu) = \\mathcal{N}(\\mu_n, \\tau_n^{-1}), \\quad q_\\tau^*(\\tau) = \\text{Gamma}(a_n, b_n),\\]
where the parameters depend on expectations under the other factor. CAVI alternates between updating \\(q_\\mu\\) and \\(q_\\tau\\) until convergence.</p></div>
</div>

<h3>Convergence Properties</h3>

<p>CAVI is guaranteed to increase the ELBO at each step (or leave it unchanged). Since the ELBO is bounded above by \\(\\log p(\\mathbf{x})\\), CAVI converges to a local optimum. However, like EM, it is sensitive to initialization and may find different local optima from different starting points.</p>

<div class="env-block remark">
<div class="env-title">Remark (Limitations of Mean-Field)</div>
<div class="env-body"><p>The factorization assumption \\(q(\\mathbf{z}) = \\prod_j q_j(z_j)\\) cannot capture posterior correlations. For strongly correlated posteriors (e.g., a banana-shaped distribution), the mean-field approximation underestimates posterior variance and may give misleading uncertainty estimates. Structured variational families that allow some dependencies (e.g., block-diagonal covariance) offer a middle ground between tractability and expressiveness.</p></div>
</div>

<div class="viz-placeholder" data-viz="viz-mean-field"></div>`,

            visualizations: [
                {
                    id: 'viz-mean-field',
                    title: 'Mean-Field VI on a 2D Posterior',
                    description: 'The blue ellipse represents a correlated 2D Gaussian posterior. The orange rectangle shows the factorized mean-field approximation \\(q(z_1)q(z_2)\\). Watch CAVI converge as it alternates updating each factor. Adjust the correlation to see how mean-field struggles with strong dependencies.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, {scale: 60, originX: null, originY: null});
                        viz.originX = viz.width / 2;
                        viz.originY = viz.height / 2;
                        var ctx = viz.ctx;
                        var rho = 0.7, running = false, step = 0;
                        var trueMu = [0, 0], trueSig1 = 1.0, trueSig2 = 0.8;
                        var qMu1 = -1.5, qMu2 = 1.5, qSig1 = 0.5, qSig2 = 0.5;
                        var maxSteps = 30;

                        VizEngine.createSlider(controls, 'Correlation \u03c1', -0.95, 0.95, rho, 0.05, function(v) {
                            rho = v; step = 0; qMu1 = -1.5; qMu2 = 1.5; qSig1 = 0.5; qSig2 = 0.5;
                        });
                        VizEngine.createButton(controls, 'Run CAVI', function() { running = true; step = 0; qMu1 = -1.5; qMu2 = 1.5; qSig1 = 0.5; qSig2 = 0.5; });
                        VizEngine.createButton(controls, 'Reset', function() { running = false; step = 0; qMu1 = -1.5; qMu2 = 1.5; qSig1 = 0.5; qSig2 = 0.5; });

                        function caviStep() {
                            // Update q1: optimal mean and variance given q2
                            qMu1 = trueMu[0] + rho * (trueSig1 / trueSig2) * (qMu2 - trueMu[1]);
                            qSig1 = trueSig1 * Math.sqrt(1 - rho * rho);
                            // Update q2: optimal mean and variance given q1
                            qMu2 = trueMu[1] + rho * (trueSig2 / trueSig1) * (qMu1 - trueMu[0]);
                            qSig2 = trueSig2 * Math.sqrt(1 - rho * rho);
                        }

                        var lastStepTime = 0;
                        function draw(t) {
                            if (running && step < maxSteps && t - lastStepTime > 300) {
                                caviStep();
                                step++;
                                lastStepTime = t;
                                if (step >= maxSteps) running = false;
                            }
                            viz.clear(); viz.drawGrid(); viz.drawAxes();

                            // Draw true posterior ellipse
                            var angle = 0.5 * Math.atan2(2 * rho * trueSig1 * trueSig2, trueSig1 * trueSig1 - trueSig2 * trueSig2);
                            var cos = Math.cos(angle), sin = Math.sin(angle);
                            var a2 = trueSig1 * trueSig1, b2 = trueSig2 * trueSig2, ab = rho * trueSig1 * trueSig2;
                            var v1 = 0.5 * (a2 + b2) + Math.sqrt(0.25 * (a2 - b2) * (a2 - b2) + ab * ab);
                            var v2 = 0.5 * (a2 + b2) - Math.sqrt(0.25 * (a2 - b2) * (a2 - b2) + ab * ab);
                            var r1 = Math.sqrt(Math.max(v1, 0.01)), r2 = Math.sqrt(Math.max(v2, 0.01));

                            for (var s = 3; s >= 1; s--) {
                                var alpha = s === 3 ? '15' : s === 2 ? '25' : '40';
                                ctx.save();
                                var sc = viz.toScreen(trueMu[0], trueMu[1]);
                                ctx.translate(sc[0], sc[1]);
                                ctx.rotate(-angle);
                                ctx.beginPath();
                                ctx.ellipse(0, 0, r1 * s * viz.scale, r2 * s * viz.scale, 0, 0, Math.PI * 2);
                                ctx.fillStyle = viz.colors.blue + alpha;
                                ctx.fill();
                                ctx.strokeStyle = viz.colors.blue;
                                ctx.lineWidth = s === 1 ? 2 : 0.5;
                                ctx.stroke();
                                ctx.restore();
                            }

                            // Draw mean-field approximation (axis-aligned rectangle/ellipse)
                            for (var s = 3; s >= 1; s--) {
                                var alpha = s === 3 ? '15' : s === 2 ? '25' : '40';
                                ctx.save();
                                var sc2 = viz.toScreen(qMu1, qMu2);
                                ctx.translate(sc2[0], sc2[1]);
                                ctx.beginPath();
                                ctx.ellipse(0, 0, qSig1 * s * viz.scale, qSig2 * s * viz.scale, 0, 0, Math.PI * 2);
                                ctx.fillStyle = viz.colors.orange + alpha;
                                ctx.fill();
                                ctx.strokeStyle = viz.colors.orange;
                                ctx.lineWidth = s === 1 ? 2 : 0.5;
                                ctx.stroke();
                                ctx.restore();
                            }

                            // Labels
                            viz.screenText('True posterior p(z|x)', 15, 20, viz.colors.blue, 13, 'left');
                            viz.screenText('Mean-field q(z\u2081)q(z\u2082)', 15, 38, viz.colors.orange, 13, 'left');
                            viz.screenText('Step: ' + step + '  \u03c1=' + rho.toFixed(2), viz.width - 15, 20, viz.colors.white, 13, 'right');
                            viz.screenText('q: \u03bc=(' + qMu1.toFixed(2) + ',' + qMu2.toFixed(2) + ')  \u03c3=(' + qSig1.toFixed(2) + ',' + qSig2.toFixed(2) + ')', viz.width - 15, 38, viz.colors.text, 11, 'right');
                        }
                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Derive the optimal mean-field factor \\(q_j^*(z_j)\\) by taking the functional derivative of the ELBO with respect to \\(q_j\\) and applying a Lagrange multiplier for the normalization constraint.',
                    hint: 'Write the ELBO as \\(\\mathcal{L} = \\int q_j(z_j) \\left[ \\mathbb{E}_{q_{-j}}[\\log p(\\mathbf{x}, \\mathbf{z})] \\right] dz_j - \\int q_j(z_j) \\log q_j(z_j)\\, dz_j + \\text{const}\\).',
                    solution: 'Isolating terms involving \\(q_j\\): \\(\\mathcal{L}[q_j] = \\int q_j(z_j) \\mathbb{E}_{q_{-j}}[\\log p(\\mathbf{x},\\mathbf{z})]\\, dz_j - \\int q_j(z_j) \\log q_j(z_j)\\, dz_j + C\\). Adding the constraint \\(\\lambda(\\int q_j\\, dz_j - 1)\\), the functional derivative is \\(\\mathbb{E}_{q_{-j}}[\\log p(\\mathbf{x},\\mathbf{z})] - \\log q_j(z_j) - 1 + \\lambda = 0\\). Thus \\(\\log q_j^*(z_j) = \\mathbb{E}_{q_{-j}}[\\log p(\\mathbf{x},\\mathbf{z})] + \\text{const}\\), where the constant absorbs \\(\\lambda - 1\\) and ensures normalization.'
                },
                {
                    question: 'For a bivariate Gaussian posterior with correlation \\(\\rho = 0.9\\), what fraction of the true posterior variance in each dimension does the mean-field approximation capture?',
                    hint: 'In the mean-field Gaussian case, the marginal variance of each factor is \\(\\sigma_j^2(1 - \\rho^2)\\).',
                    solution: 'The mean-field approximation of each factor has variance \\(\\sigma_j^2(1 - \\rho^2) = \\sigma_j^2(1 - 0.81) = 0.19\\sigma_j^2\\). This is only 19% of the true marginal variance \\(\\sigma_j^2\\). The mean-field approximation dramatically underestimates uncertainty when correlations are strong. At \\(\\rho = 0.9\\), it captures less than one-fifth of the true variance.'
                },
                {
                    question: 'Prove that CAVI monotonically increases the ELBO. Under what condition does the ELBO remain unchanged at a CAVI step?',
                    hint: 'Each CAVI step optimizes the ELBO over one factor while fixing the rest. Think about what "optimal" means.',
                    solution: 'At each step, we replace \\(q_j\\) with the \\(q_j^*\\) that maximizes \\(\\mathcal{L}\\) over the choice of \\(q_j\\), holding all other factors fixed. Since the old \\(q_j\\) is a feasible choice, the new ELBO is \\(\\geq\\) the old ELBO. The ELBO remains unchanged only if \\(q_j\\) was already at its optimum, i.e., \\(q_j = q_j^*\\). Since this holds for all \\(j\\) and the ELBO is bounded above by \\(\\log p(\\mathbf{x})\\), the algorithm converges to a fixed point (local maximum of the ELBO).'
                }
            ]
        },

        // ===================== Section 3: Stochastic Variational Inference =====================
        {
            id: 'ch18-sec03',
            title: 'Stochastic Variational Inference',
            content: `<h2>18.3 Stochastic Variational Inference</h2>

<div class="env-block intuition">
<div class="env-title">Scaling VI to Large Data</div>
<div class="env-body"><p>CAVI requires passing through the entire dataset at each iteration, making it impractical for large-scale problems. <strong>Stochastic Variational Inference</strong> (SVI) uses noisy but unbiased gradient estimates from mini-batches, combined with the <strong>reparameterization trick</strong> to backpropagate through random samples. This is the engine behind Variational Autoencoders (VAEs) and modern Bayesian deep learning.</p></div>
</div>

<h3>The Reparameterization Trick</h3>

<p>To compute gradients of the ELBO with respect to variational parameters \\(\\phi\\), we need to differentiate through the expectation \\(\\mathbb{E}_{q_\\phi(\\mathbf{z})}[f(\\mathbf{z})]\\). The problem is that the distribution we sample from depends on \\(\\phi\\).</p>

<div class="env-block theorem">
<div class="env-title">Theorem 18.3.1 (Reparameterization Trick)</div>
<div class="env-body"><p>If \\(q_\\phi(\\mathbf{z})\\) can be expressed via a deterministic transformation \\(\\mathbf{z} = g_\\phi(\\boldsymbol{\\epsilon})\\) where \\(\\boldsymbol{\\epsilon} \\sim p(\\boldsymbol{\\epsilon})\\) is a fixed base distribution, then
\\[\\nabla_\\phi \\mathbb{E}_{q_\\phi}[f(\\mathbf{z})] = \\mathbb{E}_{p(\\boldsymbol{\\epsilon})}[\\nabla_\\phi f(g_\\phi(\\boldsymbol{\\epsilon}))].\\]
For the Gaussian case \\(q_\\phi = \\mathcal{N}(\\boldsymbol{\\mu}, \\text{diag}(\\boldsymbol{\\sigma}^2))\\), we write \\(\\mathbf{z} = \\boldsymbol{\\mu} + \\boldsymbol{\\sigma} \\odot \\boldsymbol{\\epsilon}\\) with \\(\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\). This moves the randomness into \\(\\boldsymbol{\\epsilon}\\) and makes \\(\\mathbf{z}\\) a differentiable function of \\(\\phi = (\\boldsymbol{\\mu}, \\boldsymbol{\\sigma})\\).</p></div>
</div>

<h3>SVI Algorithm</h3>

<div class="env-block definition">
<div class="env-title">Algorithm (Stochastic Variational Inference)</div>
<div class="env-body"><p>Given model \\(p(\\mathbf{x}, \\mathbf{z})\\) and variational family \\(q_\\phi(\\mathbf{z})\\):
<ol>
<li>Initialize variational parameters \\(\\phi\\).</li>
<li>Repeat until convergence:
    <ol style="list-style-type: lower-alpha;">
    <li>Sample mini-batch \\(\\mathbf{x}_B\\) from the data.</li>
    <li>Sample \\(\\boldsymbol{\\epsilon}^{(1)}, \\ldots, \\boldsymbol{\\epsilon}^{(L)} \\sim p(\\boldsymbol{\\epsilon})\\) and compute \\(\\mathbf{z}^{(l)} = g_\\phi(\\boldsymbol{\\epsilon}^{(l)})\\).</li>
    <li>Estimate the ELBO gradient: \\(\\hat{\\nabla}_\\phi \\mathcal{L} \\approx \\frac{1}{L} \\sum_{l=1}^L \\nabla_\\phi \\left[\\log p(\\mathbf{x}_B, \\mathbf{z}^{(l)}) - \\log q_\\phi(\\mathbf{z}^{(l)})\\right]\\).</li>
    <li>Update \\(\\phi \\leftarrow \\phi + \\rho_t \\hat{\\nabla}_\\phi \\mathcal{L}\\) (using Adam or another optimizer).</li>
    </ol>
</li>
</ol></p></div>
</div>

<div class="env-block remark">
<div class="env-title">Remark (Variance Reduction)</div>
<div class="env-body"><p>The Monte Carlo gradient estimate can have high variance, especially with a single sample (\\(L=1\\)). Several techniques reduce variance: (1) using the analytic KL term when available (as in the Gaussian prior case), (2) control variates, (3) importance weighting. In the VAE setting, using a single sample per data point (\\(L=1\\)) often works well because the mini-batch itself provides averaging.</p></div>
</div>

<div class="viz-placeholder" data-viz="viz-svi-elbo"></div>`,

            visualizations: [
                {
                    id: 'viz-svi-elbo',
                    title: 'ELBO Optimization via SVI',
                    description: 'Watch the ELBO increase over SVI iterations. The noisy line shows stochastic gradient updates with mini-batch noise. Adjust the learning rate and batch size to see their effect on convergence speed and stability.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, {scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var lr = 0.05, batchSize = 32;
                        var running = false, elboHistory = [], iteration = 0;
                        var trueElbo = -1.2;

                        VizEngine.createSlider(controls, 'Learning rate', 0.001, 0.2, lr, 0.001, function(v) { lr = v; });
                        VizEngine.createSlider(controls, 'Batch size', 1, 128, batchSize, 1, function(v) { batchSize = Math.round(v); });
                        VizEngine.createButton(controls, 'Run SVI', function() { running = true; elboHistory = []; iteration = 0; });
                        VizEngine.createButton(controls, 'Reset', function() { running = false; elboHistory = []; iteration = 0; });

                        var lastTime = 0;
                        function draw(t) {
                            if (running && iteration < 300 && t - lastTime > 30) {
                                // Simulate ELBO: converges to trueElbo with noise inversely proportional to batch size
                                var progress = 1 - Math.exp(-lr * 3 * iteration);
                                var noise = (Math.random() - 0.5) * 2 / Math.sqrt(batchSize) * 0.4;
                                var elbo = -5 + (trueElbo + 5) * progress + noise;
                                elboHistory.push(elbo);
                                iteration++;
                                lastTime = t;
                            }

                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            // Plot area
                            var pad = {left: 65, right: 20, top: 35, bottom: 45};
                            var pw = W - pad.left - pad.right;
                            var ph = H - pad.top - pad.bottom;

                            // Axes
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + ph); ctx.lineTo(pad.left + pw, pad.top + ph); ctx.stroke();

                            // Y axis labels
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
                            var yMin = -6, yMax = 0;
                            for (var yv = yMin; yv <= yMax; yv += 1) {
                                var yy = pad.top + ph - (yv - yMin) / (yMax - yMin) * ph;
                                ctx.fillText(yv.toFixed(0), pad.left - 8, yy);
                                ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                                ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(pad.left + pw, yy); ctx.stroke();
                            }

                            // X axis labels
                            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                            var xMax = 300;
                            for (var xv = 0; xv <= xMax; xv += 50) {
                                var xx = pad.left + (xv / xMax) * pw;
                                ctx.fillText(xv, xx, pad.top + ph + 6);
                            }
                            ctx.fillText('Iteration', pad.left + pw / 2, pad.top + ph + 25);
                            ctx.save(); ctx.translate(15, pad.top + ph / 2); ctx.rotate(-Math.PI / 2);
                            ctx.textAlign = 'center'; ctx.fillText('ELBO', 0, 0); ctx.restore();

                            // Draw true ELBO line
                            var trueY = pad.top + ph - (trueElbo - yMin) / (yMax - yMin) * ph;
                            ctx.setLineDash([6, 4]); ctx.strokeStyle = viz.colors.green; ctx.lineWidth = 1.5;
                            ctx.beginPath(); ctx.moveTo(pad.left, trueY); ctx.lineTo(pad.left + pw, trueY); ctx.stroke();
                            ctx.setLineDash([]);
                            ctx.fillStyle = viz.colors.green; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'left';
                            ctx.fillText('log p(x) bound', pad.left + pw - 90, trueY - 10);

                            // Draw ELBO trace
                            if (elboHistory.length > 1) {
                                ctx.beginPath();
                                for (var i = 0; i < elboHistory.length; i++) {
                                    var ex = pad.left + (i / xMax) * pw;
                                    var ey = pad.top + ph - (elboHistory[i] - yMin) / (yMax - yMin) * ph;
                                    ey = Math.max(pad.top, Math.min(pad.top + ph, ey));
                                    i === 0 ? ctx.moveTo(ex, ey) : ctx.lineTo(ex, ey);
                                }
                                ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 1.8; ctx.stroke();
                            }

                            // Title
                            viz.screenText('Stochastic Variational Inference', W / 2, 16, viz.colors.white, 14);
                            if (elboHistory.length > 0) {
                                var last = elboHistory[elboHistory.length - 1];
                                viz.screenText('Current ELBO: ' + last.toFixed(3), W - pad.right - 5, pad.top + 8, viz.colors.orange, 12, 'right');
                            }
                        }
                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Show that the reparameterization trick gives an unbiased gradient estimate. That is, verify \\(\\mathbb{E}_{p(\\boldsymbol{\\epsilon})}[\\nabla_\\phi f(g_\\phi(\\boldsymbol{\\epsilon}))] = \\nabla_\\phi \\mathbb{E}_{q_\\phi}[f(\\mathbf{z})]\\).',
                    hint: 'Use the change of variables \\(\\mathbf{z} = g_\\phi(\\boldsymbol{\\epsilon})\\) and the fact that \\(p(\\boldsymbol{\\epsilon})\\) does not depend on \\(\\phi\\).',
                    solution: 'By the change of variables, \\(\\mathbb{E}_{q_\\phi}[f(\\mathbf{z})] = \\mathbb{E}_{p(\\boldsymbol{\\epsilon})}[f(g_\\phi(\\boldsymbol{\\epsilon}))]\\). Since \\(p(\\boldsymbol{\\epsilon})\\) does not depend on \\(\\phi\\), we can interchange the gradient and expectation: \\(\\nabla_\\phi \\mathbb{E}_{p(\\boldsymbol{\\epsilon})}[f(g_\\phi(\\boldsymbol{\\epsilon}))] = \\mathbb{E}_{p(\\boldsymbol{\\epsilon})}[\\nabla_\\phi f(g_\\phi(\\boldsymbol{\\epsilon}))]\\) (under mild regularity conditions on \\(f\\) and \\(g_\\phi\\)). Each sample \\(\\boldsymbol{\\epsilon}^{(l)}\\) provides an unbiased single-sample estimate of this gradient.'
                },
                {
                    question: 'Why does increasing the mini-batch size reduce the variance of the ELBO gradient estimate? What is the relationship between batch size \\(B\\) and gradient variance?',
                    hint: 'Think about the variance of a sample mean of \\(B\\) i.i.d. random variables.',
                    solution: 'The mini-batch gradient is an average of \\(B\\) i.i.d. per-sample gradient estimates. By the properties of sample means, \\(\\text{Var}[\\hat{\\nabla}] = \\text{Var}[\\nabla_i] / B\\), so the variance decreases as \\(1/B\\). Doubling the batch size halves the gradient variance, yielding smoother convergence. However, there are diminishing returns: the computational cost scales linearly with \\(B\\) while the variance reduction is sub-linear.'
                },
                {
                    question: 'In the VAE setting, the ELBO is often written as \\(\\mathbb{E}_{q_\\phi(\\mathbf{z}|\\mathbf{x})}[\\log p_\\theta(\\mathbf{x}|\\mathbf{z})] - \\mathrm{KL}(q_\\phi(\\mathbf{z}|\\mathbf{x}) \\| p(\\mathbf{z}))\\). When both \\(q_\\phi\\) and \\(p(\\mathbf{z})\\) are Gaussian, derive the closed-form KL term.',
                    hint: 'Use the formula for KL between two Gaussians: \\(\\mathrm{KL}(\\mathcal{N}(\\boldsymbol{\\mu}, \\text{diag}(\\boldsymbol{\\sigma}^2)) \\| \\mathcal{N}(\\mathbf{0}, \\mathbf{I}))\\).',
                    solution: 'For \\(q = \\mathcal{N}(\\boldsymbol{\\mu}, \\text{diag}(\\boldsymbol{\\sigma}^2))\\) and \\(p = \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\): \\(\\mathrm{KL}(q \\| p) = \\frac{1}{2} \\sum_{j=1}^d (\\mu_j^2 + \\sigma_j^2 - \\log \\sigma_j^2 - 1)\\). This has a nice interpretation: \\(\\mu_j^2\\) penalizes means far from zero, \\(\\sigma_j^2 - \\log \\sigma_j^2 - 1 \\geq 0\\) penalizes variances far from 1. The closed form avoids Monte Carlo estimation of this term, reducing gradient variance.'
                }
            ]
        },

        // ===================== Section 4: VI vs MCMC =====================
        {
            id: 'ch18-sec04',
            title: 'VI vs MCMC',
            content: `<h2>18.4 VI vs MCMC</h2>

<div class="env-block intuition">
<div class="env-title">Two Philosophies of Approximate Inference</div>
<div class="env-body"><p>Variational Inference and Markov Chain Monte Carlo (MCMC, Chapter 17) are the two main approaches to approximate Bayesian inference. They make fundamentally different trade-offs: VI is fast but provides a biased approximation, while MCMC is asymptotically exact but can be slow. Understanding when to use each is critical for applied Bayesian work.</p></div>
</div>

<h3>Comparison Summary</h3>

<table style="width:100%;border-collapse:collapse;margin:1em 0;">
<tr style="border-bottom:2px solid #30363d;">
<th style="text-align:left;padding:6px 10px;color:#c9d1d9;">Criterion</th>
<th style="text-align:left;padding:6px 10px;color:#58a6ff;">Variational Inference</th>
<th style="text-align:left;padding:6px 10px;color:#3fb950;">MCMC</th>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:6px 10px;color:#8b949e;">Nature</td>
<td style="padding:6px 10px;color:#c9d1d9;">Optimization</td>
<td style="padding:6px 10px;color:#c9d1d9;">Sampling</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:6px 10px;color:#8b949e;">Guarantee</td>
<td style="padding:6px 10px;color:#c9d1d9;">Lower bound on \\(\\log p(\\mathbf{x})\\)</td>
<td style="padding:6px 10px;color:#c9d1d9;">Asymptotically exact</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:6px 10px;color:#8b949e;">Speed</td>
<td style="padding:6px 10px;color:#c9d1d9;">Fast (gradient-based)</td>
<td style="padding:6px 10px;color:#c9d1d9;">Slow (sequential sampling)</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:6px 10px;color:#8b949e;">Scalability</td>
<td style="padding:6px 10px;color:#c9d1d9;">Scales to large data (SVI)</td>
<td style="padding:6px 10px;color:#c9d1d9;">Difficult for large \\(N\\)</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:6px 10px;color:#8b949e;">Multimodality</td>
<td style="padding:6px 10px;color:#c9d1d9;">Tends to miss modes (reverse KL)</td>
<td style="padding:6px 10px;color:#c9d1d9;">Can explore multiple modes</td>
</tr>
<tr style="border-bottom:1px solid #21262d;">
<td style="padding:6px 10px;color:#8b949e;">Convergence</td>
<td style="padding:6px 10px;color:#c9d1d9;">Easy to assess (ELBO)</td>
<td style="padding:6px 10px;color:#c9d1d9;">Hard to diagnose</td>
</tr>
<tr>
<td style="padding:6px 10px;color:#8b949e;">Uncertainty</td>
<td style="padding:6px 10px;color:#c9d1d9;">Often underestimates</td>
<td style="padding:6px 10px;color:#c9d1d9;">Calibrated (if converged)</td>
</tr>
</table>

<div class="env-block theorem">
<div class="env-title">Theorem 18.4.1 (VI Underestimates Variance)</div>
<div class="env-body"><p>Let \\(p(\\mathbf{z} \\mid \\mathbf{x})\\) be the true posterior and \\(q^*(\\mathbf{z})\\) the optimal mean-field approximation minimizing \\(\\mathrm{KL}(q \\| p)\\). Then for each component \\(j\\),
\\[\\text{Var}_{q^*}(z_j) \\leq \\text{Var}_{p}(z_j).\\]
The mean-field approximation <em>always</em> underestimates marginal posterior variances. This is a direct consequence of the reverse KL penalty structure: \\(q\\) concentrates to avoid placing mass where \\(p\\) is small.</p></div>
</div>

<h3>When to Use Each</h3>

<div class="env-block remark">
<div class="env-title">Practical Guidelines</div>
<div class="env-body"><p><strong>Prefer VI when:</strong> (1) the dataset is large and speed matters, (2) you need point estimates or approximate posteriors for downstream tasks, (3) the model is amenable to reparameterization (continuous latent variables), (4) you are willing to accept approximation bias for computational savings.</p>
<p><strong>Prefer MCMC when:</strong> (1) posterior accuracy is paramount (e.g., scientific inference), (2) the posterior is multimodal and you need to capture all modes, (3) the dataset is small enough that MCMC is feasible, (4) you need calibrated uncertainty quantification.</p>
<p><strong>Hybrid approaches:</strong> Use VI to initialize MCMC (warm start), or use MCMC diagnostics to validate VI approximations. Normalizing flows extend VI to more expressive variational families that can capture complex posteriors.</p></div>
</div>

<div class="viz-placeholder" data-viz="viz-vi-vs-mcmc"></div>`,

            visualizations: [
                {
                    id: 'viz-vi-vs-mcmc',
                    title: 'VI vs MCMC: Speed-Accuracy Trade-off',
                    description: 'Side-by-side comparison: VI (left) quickly finds a Gaussian approximation to a bimodal posterior, while MCMC (right) slowly explores both modes with samples. Watch how VI converges fast but misses a mode, while MCMC eventually captures the full distribution.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, {scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var halfW = Math.floor(W / 2) - 10;
                        var running = false, iteration = 0;

                        // True posterior: bimodal
                        function truePosterior(x) {
                            var g1 = Math.exp(-0.5 * Math.pow((x + 1.8) / 0.6, 2)) / (0.6 * 2.507);
                            var g2 = Math.exp(-0.5 * Math.pow((x - 1.8) / 0.6, 2)) / (0.6 * 2.507);
                            return 0.5 * g1 + 0.5 * g2;
                        }

                        // VI state
                        var viMu = 0, viSig = 2.0;
                        // MCMC state
                        var mcmcSamples = [], mcmcCurrent = 0;

                        VizEngine.createButton(controls, 'Run Both', function() { running = true; iteration = 0; viMu = 0; viSig = 2.0; mcmcSamples = []; mcmcCurrent = 0; });
                        VizEngine.createButton(controls, 'Reset', function() { running = false; iteration = 0; viMu = 0; viSig = 2.0; mcmcSamples = []; mcmcCurrent = 0; });

                        function viStep() {
                            // Gradient ascent on ELBO: simulate convergence to one mode
                            var gradMu = 0, gradSig = 0, dx = 0.05;
                            for (var x = -5; x <= 5; x += dx) {
                                var q = Math.exp(-0.5 * Math.pow((x - viMu) / viSig, 2)) / (viSig * 2.507);
                                var p = truePosterior(x);
                                if (q > 1e-12) {
                                    var score = (x - viMu) / (viSig * viSig);
                                    gradMu += q * (Math.log(p) - Math.log(q)) * score * dx;
                                    var scoreSig = ((x - viMu) * (x - viMu) / (viSig * viSig * viSig) - 1 / viSig);
                                    gradSig += q * (Math.log(p) - Math.log(q)) * scoreSig * dx;
                                }
                            }
                            viMu += 0.3 * gradMu;
                            viSig += 0.1 * gradSig;
                            viSig = Math.max(0.2, Math.min(3, viSig));
                        }

                        function mcmcStep() {
                            // Metropolis-Hastings
                            var proposal = mcmcCurrent + (Math.random() - 0.5) * 1.5;
                            var pCurr = truePosterior(mcmcCurrent);
                            var pProp = truePosterior(proposal);
                            if (Math.random() < Math.min(1, pProp / (pCurr + 1e-15))) {
                                mcmcCurrent = proposal;
                            }
                            mcmcSamples.push(mcmcCurrent);
                        }

                        function drawPanel(offsetX, title, drawFn) {
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(offsetX, 0, halfW, H);
                            ctx.clip();
                            drawFn(offsetX);
                            ctx.restore();
                            // Border
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.strokeRect(offsetX + 1, 1, halfW - 2, H - 2);
                            // Title
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.fillText(title, offsetX + halfW / 2, 20);
                        }

                        function plotCurve(offsetX, fn, color, xRange, yRange) {
                            var pad = {left: 30, right: 10, top: 35, bottom: 30};
                            var pw = halfW - pad.left - pad.right;
                            var ph = H - pad.top - pad.bottom;
                            ctx.beginPath();
                            var steps = 200;
                            for (var i = 0; i <= steps; i++) {
                                var x = xRange[0] + (xRange[1] - xRange[0]) * i / steps;
                                var y = fn(x);
                                var sx = offsetX + pad.left + (x - xRange[0]) / (xRange[1] - xRange[0]) * pw;
                                var sy = pad.top + ph - (y - yRange[0]) / (yRange[1] - yRange[0]) * ph;
                                sy = Math.max(pad.top, Math.min(pad.top + ph, sy));
                                i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                            }
                            ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
                        }

                        var lastTime = 0;
                        function draw(t) {
                            if (running && iteration < 500 && t - lastTime > 20) {
                                viStep();
                                for (var s = 0; s < 3; s++) mcmcStep();
                                iteration++;
                                lastTime = t;
                            }

                            ctx.fillStyle = viz.colors.bg; ctx.fillRect(0, 0, W, H);

                            var xRange = [-5, 5], yRange = [0, 0.7];
                            var pad = {left: 30, right: 10, top: 35, bottom: 30};

                            // VI panel
                            drawPanel(0, 'Variational Inference', function(ox) {
                                plotCurve(ox, truePosterior, viz.colors.blue, xRange, yRange);
                                var qFn = function(x) {
                                    return Math.exp(-0.5 * Math.pow((x - viMu) / viSig, 2)) / (viSig * 2.507);
                                };
                                plotCurve(ox, qFn, viz.colors.orange, xRange, yRange);
                                ctx.fillStyle = viz.colors.blue; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'left';
                                ctx.fillText('True p(z|x)', ox + 35, H - 10);
                                ctx.fillStyle = viz.colors.orange;
                                ctx.fillText('q(z)', ox + 130, H - 10);
                            });

                            // MCMC panel
                            drawPanel(halfW + 20, 'MCMC (Metropolis)', function(ox) {
                                plotCurve(ox, truePosterior, viz.colors.blue, xRange, yRange);
                                // Draw histogram of samples
                                if (mcmcSamples.length > 10) {
                                    var bins = 40, counts = new Array(bins).fill(0);
                                    var bw = (xRange[1] - xRange[0]) / bins;
                                    for (var i = 0; i < mcmcSamples.length; i++) {
                                        var bi = Math.floor((mcmcSamples[i] - xRange[0]) / bw);
                                        if (bi >= 0 && bi < bins) counts[bi]++;
                                    }
                                    var maxC = Math.max.apply(null, counts);
                                    if (maxC > 0) {
                                        var pw2 = halfW - pad.left - pad.right;
                                        var ph2 = H - pad.top - pad.bottom;
                                        for (var b = 0; b < bins; b++) {
                                            var bx = ox + pad.left + (b / bins) * pw2;
                                            var bh = (counts[b] / maxC) * yRange[1];
                                            var by = pad.top + ph2 - (bh / yRange[1]) * ph2;
                                            var bwPx = pw2 / bins;
                                            ctx.fillStyle = viz.colors.green + '55';
                                            ctx.fillRect(bx, by, bwPx - 1, pad.top + ph2 - by);
                                        }
                                    }
                                }
                                ctx.fillStyle = viz.colors.blue; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'left';
                                ctx.fillText('True p(z|x)', ox + 35, H - 10);
                                ctx.fillStyle = viz.colors.green;
                                ctx.fillText('Samples (' + mcmcSamples.length + ')', ox + 130, H - 10);
                            });

                            viz.screenText('Iteration: ' + iteration, W / 2, H - 10, viz.colors.text, 11);
                        }
                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],

            exercises: [
                {
                    question: 'A researcher has a small dataset (\\(N = 50\\)) and needs accurate posterior credible intervals for hypothesis testing. Should they use VI or MCMC? Justify your recommendation.',
                    hint: 'Consider calibration of uncertainty and dataset size.',
                    solution: 'MCMC is the better choice here. With only 50 data points, MCMC is computationally feasible (the likelihood is cheap to evaluate). More importantly, the researcher needs <em>accurate</em> credible intervals for hypothesis testing. VI (especially mean-field) systematically underestimates posterior variance, producing credible intervals that are too narrow and may lead to overconfident conclusions. MCMC provides asymptotically exact samples from the posterior, yielding well-calibrated credible intervals if run long enough with proper convergence diagnostics.'
                },
                {
                    question: 'Suppose a mean-field VI approximation gives posterior variance 0.04 for a parameter, while HMC gives posterior variance 0.25. Compute the approximate correlation the mean-field is missing, assuming the true posterior is bivariate Gaussian.',
                    hint: 'Recall that mean-field variance for a bivariate Gaussian is \\(\\sigma^2(1 - \\rho^2)\\).',
                    solution: 'If the true marginal variance is 0.25 and the mean-field gives \\(\\sigma^2(1-\\rho^2) = 0.04\\), then \\(1-\\rho^2 = 0.04/0.25 = 0.16\\), so \\(\\rho^2 = 0.84\\), giving \\(|\\rho| = 0.917\\). The mean-field is missing a very strong correlation of about 0.92 between the two parameters. This illustrates how factorized approximations can severely underestimate uncertainty when parameters are strongly correlated.'
                },
                {
                    question: 'Describe how you would use VI to initialize MCMC for a complex posterior. What are the advantages of this hybrid approach?',
                    hint: 'Think about burn-in time and the purpose of initialization in MCMC.',
                    solution: 'Run VI first to obtain an approximate posterior \\(q^*\\). Then initialize MCMC chains at samples drawn from \\(q^*\\) (or at the mode of \\(q^*\\)). Advantages: (1) MCMC burn-in is drastically reduced because chains start near the typical set, (2) the proposal distribution can be informed by the VI covariance, improving acceptance rates, (3) VI is fast and scales to the data, while MCMC refines the approximation for accuracy. This combines the speed of VI for exploration with the exactness of MCMC for the final posterior estimate.'
                }
            ]
        }
    ]
});
