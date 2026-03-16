window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch13',
    number: 13,
    title: 'Expectation-Maximization',
    subtitle: 'A general algorithm for maximum likelihood in latent variable models, from Gaussian mixtures to incomplete data',
    sections: [
        // ===================== Section 1: Gaussian Mixture Models =====================
        {
            id: 'ch13-sec01',
            title: 'Gaussian Mixture Models',
            content: `<h2>Gaussian Mixture Models</h2>

                <div class="env-block intuition">
                    <div class="env-title">Why Mixtures?</div>
                    <div class="env-body"><p>A single Gaussian can only model unimodal, elliptical clusters. Real data often has multiple clusters, skewness, or multimodal structure. A <strong>Gaussian mixture model</strong> (GMM) represents the data distribution as a weighted sum of several Gaussians, each capturing one "component" or cluster. The challenge is that we do not observe which component generated each data point; this is a <strong>latent variable</strong>.</p></div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Definition (Gaussian Mixture Model)</div>
                    <div class="env-body"><p>A <strong>Gaussian mixture model</strong> with \\(K\\) components defines the density
                    \\[p(\\mathbf{x} \\mid \\boldsymbol{\\theta}) = \\sum_{k=1}^{K} \\pi_k \\, \\mathcal{N}(\\mathbf{x} \\mid \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)\\]
                    where \\(\\pi_k \\geq 0\\) are the <strong>mixing weights</strong> (with \\(\\sum_k \\pi_k = 1\\)), \\(\\boldsymbol{\\mu}_k \\in \\mathbb{R}^D\\) are the component means, and \\(\\boldsymbol{\\Sigma}_k \\in \\mathbb{R}^{D \\times D}\\) are the component covariance matrices. The full parameter set is \\(\\boldsymbol{\\theta} = \\{\\pi_k, \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k\\}_{k=1}^K\\).</p></div>
                </div>

                <h3>Latent Variable Interpretation</h3>

                <p>Introduce a latent (unobserved) variable \\(z_i \\in \\{1, \\dots, K\\}\\) for each data point \\(\\mathbf{x}_i\\), indicating which component generated it. The generative process is:</p>
                <ol>
                    <li>Draw the component assignment: \\(z_i \\sim \\text{Categorical}(\\pi_1, \\dots, \\pi_K)\\)</li>
                    <li>Draw the observation: \\(\\mathbf{x}_i \\mid z_i = k \\sim \\mathcal{N}(\\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)\\)</li>
                </ol>

                <p>Marginalizing over \\(z_i\\) recovers the mixture density: \\(p(\\mathbf{x}_i) = \\sum_k p(z_i = k) \\, p(\\mathbf{x}_i \\mid z_i = k) = \\sum_k \\pi_k \\, \\mathcal{N}(\\mathbf{x}_i \\mid \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)\\).</p>

                <div class="env-block remark">
                    <div class="env-title">Remark (Why Not Just Maximize Likelihood Directly?)</div>
                    <div class="env-body"><p>The log-likelihood is \\(\\ell(\\boldsymbol{\\theta}) = \\sum_{i=1}^N \\log \\sum_{k=1}^K \\pi_k \\, \\mathcal{N}(\\mathbf{x}_i \\mid \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)\\). The log-sum structure prevents us from obtaining closed-form MLEs. Taking derivatives and setting them to zero yields coupled equations that cannot be solved analytically. This is precisely the problem that EM solves.</p></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-gmm-components"></div>`,

            visualizations: [
                {
                    id: 'viz-gmm-components',
                    title: 'Gaussian Mixture Model: Component Densities',
                    description: 'Adjust K and mixing weights to see how individual Gaussian components (colored curves) combine into the mixture density (white curve).',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var K = 3;
                        var spread = 1.0;

                        VizEngine.createSlider(controls, 'K', 1, 5, 3, 1, function(v) { K = Math.round(v); draw(); });
                        VizEngine.createSlider(controls, 'Separation', 0.3, 2.0, 1.0, 0.1, function(v) { spread = v; draw(); });

                        // Pre-define component params for K=1..5
                        var compColors = ['#58a6ff', '#3fb950', '#f0883e', '#bc8cff', '#f778ba'];
                        var baseMeans = [-3, -1.2, 0.8, 2.5, 4.0];
                        var baseSigmas = [0.8, 0.6, 0.9, 0.7, 0.5];

                        function gaussian(x, mu, sigma) {
                            var z = (x - mu) / sigma;
                            return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg; ctx.fillRect(0, 0, W, H);
                            var m = {top: 30, right: 20, bottom: 40, left: 40};
                            var pw = W - m.left - m.right, ph = H - m.top - m.bottom;
                            // x range
                            var xMin = -6, xMax = 6;
                            var nPts = 300;
                            // Compute densities
                            var mixture = new Array(nPts);
                            var components = [];
                            for (var k = 0; k < K; k++) components.push(new Array(nPts));
                            var piK = 1.0 / K;
                            for (var i = 0; i < nPts; i++) {
                                var x = xMin + (xMax - xMin) * i / (nPts - 1);
                                mixture[i] = 0;
                                for (var k = 0; k < K; k++) {
                                    var mu = baseMeans[k] * spread;
                                    var sig = baseSigmas[k];
                                    var val = piK * gaussian(x, mu, sig);
                                    components[k][i] = val;
                                    mixture[i] += val;
                                }
                            }
                            var maxY = 0;
                            for (var i = 0; i < nPts; i++) { if (mixture[i] > maxY) maxY = mixture[i]; }
                            maxY *= 1.15;
                            // Axes
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(m.left, m.top + ph); ctx.lineTo(m.left + pw, m.top + ph); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + ph); ctx.stroke();
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            ctx.fillText('x', m.left + pw / 2, H - 8);
                            // Draw component curves (filled)
                            for (var k = 0; k < K; k++) {
                                ctx.fillStyle = compColors[k] + '22';
                                ctx.strokeStyle = compColors[k] + 'aa'; ctx.lineWidth = 1.5;
                                ctx.beginPath();
                                for (var i = 0; i < nPts; i++) {
                                    var px = m.left + (i / (nPts - 1)) * pw;
                                    var py = m.top + ph - (components[k][i] / maxY) * ph;
                                    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                                }
                                // Close to baseline for fill
                                ctx.lineTo(m.left + pw, m.top + ph); ctx.lineTo(m.left, m.top + ph);
                                ctx.closePath(); ctx.fill();
                                // Stroke on top
                                ctx.beginPath();
                                for (var i = 0; i < nPts; i++) {
                                    var px = m.left + (i / (nPts - 1)) * pw;
                                    var py = m.top + ph - (components[k][i] / maxY) * ph;
                                    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                                }
                                ctx.stroke();
                            }
                            // Mixture curve
                            ctx.strokeStyle = viz.colors.white; ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var i = 0; i < nPts; i++) {
                                var px = m.left + (i / (nPts - 1)) * pw;
                                var py = m.top + ph - (mixture[i] / maxY) * ph;
                                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                            }
                            ctx.stroke();
                            // Legend
                            ctx.font = '12px -apple-system,sans-serif'; ctx.textAlign = 'left';
                            ctx.fillStyle = viz.colors.white; ctx.fillText('Mixture (sum)', m.left + 10, m.top + 16);
                            for (var k = 0; k < K; k++) {
                                ctx.fillStyle = compColors[k]; ctx.fillText('Component ' + (k + 1) + ' (\u03C0=' + (1/K).toFixed(2) + ')', m.left + 10, m.top + 34 + k * 16);
                            }
                        }
                        draw();
                        return { stopAnimation: function() {} };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Write down the complete data log-likelihood for a GMM when the latent variables \\(z_1, \\dots, z_N\\) are observed. Why is this much simpler than the incomplete data log-likelihood?',
                    hint: 'With \\(z_i\\) observed, each term in the likelihood involves only one Gaussian, not a sum over \\(K\\) Gaussians.',
                    solution: 'The complete data log-likelihood is \\(\\ell_c(\\boldsymbol{\\theta}) = \\sum_{i=1}^N \\bigl[\\log \\pi_{z_i} + \\log \\mathcal{N}(\\mathbf{x}_i \\mid \\boldsymbol{\\mu}_{z_i}, \\boldsymbol{\\Sigma}_{z_i})\\bigr]\\). This decomposes into separate terms for each component and admits closed-form MLEs: \\(\\hat{\\boldsymbol{\\mu}}_k = \\sum_{i:z_i=k} \\mathbf{x}_i / N_k\\), \\(\\hat{\\boldsymbol{\\Sigma}}_k = \\sum_{i:z_i=k}(\\mathbf{x}_i - \\hat{\\boldsymbol{\\mu}}_k)(\\mathbf{x}_i - \\hat{\\boldsymbol{\\mu}}_k)^\\top / N_k\\), \\(\\hat{\\pi}_k = N_k / N\\). The incomplete-data log-likelihood \\(\\sum_i \\log \\sum_k \\pi_k \\mathcal{N}(\\mathbf{x}_i \\mid \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)\\) has the log-sum structure that couples all parameters.'
                },
                {
                    question: 'A 1D GMM with \\(K=2\\) has parameters \\(\\pi_1 = 0.6\\), \\(\\mu_1 = 0\\), \\(\\sigma_1 = 1\\), \\(\\pi_2 = 0.4\\), \\(\\mu_2 = 3\\), \\(\\sigma_2 = 0.5\\). Compute \\(p(x = 1)\\).',
                    hint: 'Evaluate each component density at \\(x=1\\) and take the weighted sum.',
                    solution: '\\(p(x=1) = 0.6 \\cdot \\mathcal{N}(1 \\mid 0, 1) + 0.4 \\cdot \\mathcal{N}(1 \\mid 3, 0.25)\\). We get \\(\\mathcal{N}(1 \\mid 0, 1) = \\frac{1}{\\sqrt{2\\pi}}e^{-0.5} \\approx 0.2420\\) and \\(\\mathcal{N}(1 \\mid 3, 0.25) = \\frac{1}{\\sqrt{2\\pi \\cdot 0.25}}e^{-\\frac{(1-3)^2}{2 \\cdot 0.25}} = \\frac{1}{\\sqrt{0.5\\pi}}e^{-8} \\approx 2.68 \\times 10^{-4}\\). So \\(p(x=1) \\approx 0.6(0.2420) + 0.4(0.000268) \\approx 0.1453\\).'
                },
                {
                    question: 'Show that a GMM with \\(K\\) components and unconstrained covariances is a universal density approximator: given any continuous density \\(f\\) on a compact domain, \\(\\sup_x |f(x) - p_K(x)| \\to 0\\) as \\(K \\to \\infty\\).',
                    hint: 'Use the fact that Gaussian kernels can approximate any continuous function uniformly on compact sets (by choosing small enough bandwidths).',
                    solution: 'This follows from the density of Gaussian kernels in \\(C_0(\\mathbb{R}^D)\\). By placing a narrow Gaussian (\\(\\sigma_k \\to 0\\)) at each of many points sampling the support of \\(f\\), the mixture approximates \\(f\\) as a Riemann sum of kernel evaluations. More formally, by the Stone-Weierstrass theorem applied to the algebra generated by Gaussian functions (which separates points and does not vanish), any continuous density on a compact set can be uniformly approximated. The weights \\(\\pi_k \\propto f(\\boldsymbol{\\mu}_k) \\cdot \\Delta V_k\\) discretize the integral. This is a nonconstructive result; in practice, finite \\(K\\) gives a finite approximation error.'
                }
            ]
        },

        // ===================== Section 2: The EM Algorithm =====================
        {
            id: 'ch13-sec02',
            title: 'The EM Algorithm',
            content: `<h2>The EM Algorithm</h2>

                <div class="env-block intuition">
                    <div class="env-title">The Key Insight</div>
                    <div class="env-body"><p>If we knew the latent variables \\(z_i\\), maximum likelihood would be easy (just compute sample means and covariances per cluster). If we knew the parameters \\(\\boldsymbol{\\theta}\\), computing the posterior over \\(z_i\\) would be easy (just apply Bayes' rule). EM alternates between these two steps, using each to improve the other.</p></div>
                </div>

                <div class="env-block definition">
                    <div class="env-title">Definition (EM Algorithm for GMMs)</div>
                    <div class="env-body"><p>Initialize parameters \\(\\boldsymbol{\\theta}^{(0)}\\). Then repeat until convergence:</p>
                    <p><strong>E-step (Expectation):</strong> Compute the <strong>responsibilities</strong> (posterior probabilities of component assignments):
                    \\[\\gamma_{ik}^{(t)} = p(z_i = k \\mid \\mathbf{x}_i, \\boldsymbol{\\theta}^{(t)}) = \\frac{\\pi_k^{(t)} \\, \\mathcal{N}(\\mathbf{x}_i \\mid \\boldsymbol{\\mu}_k^{(t)}, \\boldsymbol{\\Sigma}_k^{(t)})}{\\sum_{j=1}^K \\pi_j^{(t)} \\, \\mathcal{N}(\\mathbf{x}_i \\mid \\boldsymbol{\\mu}_j^{(t)}, \\boldsymbol{\\Sigma}_j^{(t)})}\\]</p>
                    <p><strong>M-step (Maximization):</strong> Update parameters using the responsibilities as soft counts:
                    \\[N_k = \\sum_{i=1}^N \\gamma_{ik}, \\quad \\boldsymbol{\\mu}_k^{(t+1)} = \\frac{1}{N_k}\\sum_{i=1}^N \\gamma_{ik} \\, \\mathbf{x}_i\\]
                    \\[\\boldsymbol{\\Sigma}_k^{(t+1)} = \\frac{1}{N_k}\\sum_{i=1}^N \\gamma_{ik}(\\mathbf{x}_i - \\boldsymbol{\\mu}_k^{(t+1)})(\\mathbf{x}_i - \\boldsymbol{\\mu}_k^{(t+1)})^\\top, \\quad \\pi_k^{(t+1)} = \\frac{N_k}{N}\\]</p></div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">Remark (Soft vs. Hard Assignments)</div>
                    <div class="env-body"><p>Unlike K-means, which makes <em>hard</em> assignments (each point belongs to exactly one cluster), EM makes <em>soft</em> assignments via the responsibilities \\(\\gamma_{ik} \\in [0, 1]\\). A point near the boundary between two clusters will have substantial responsibility for both. In the limit where all covariances shrink to zero, EM reduces to K-means.</p></div>
                </div>

                <p>The M-step update formulas are "soft" versions of the K-means updates. The mean \\(\\boldsymbol{\\mu}_k\\) is the responsibility-weighted average of the data, and \\(N_k\\) is the effective number of points assigned to component \\(k\\).</p>

                <div class="viz-placeholder" data-viz="viz-em-steps"></div>`,

            visualizations: [
                {
                    id: 'viz-em-steps',
                    title: 'EM for 2D GMM: Watch the Gaussians Converge',
                    description: 'Click "Step" to run one EM iteration. The ellipses show component covariances; colors indicate responsibilities.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {scale: 40});
                        var ctx = viz.ctx;
                        var iteration = 0;

                        // Generate 2D data from 3 true clusters
                        var seed = 77;
                        function srand() { seed = (seed * 16807 + 0) % 2147483647; return (seed - 1) / 2147483646; }
                        function randn() { var u = srand(), v = srand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

                        var data = [];
                        var trueMu = [[-2, 1.5], [1.5, -1], [2.5, 2]];
                        var trueN = [25, 30, 20];
                        for (var k = 0; k < 3; k++) {
                            for (var i = 0; i < trueN[k]; i++) {
                                data.push([trueMu[k][0] + randn() * 0.6, trueMu[k][1] + randn() * 0.6]);
                            }
                        }
                        var N = data.length;
                        var K = 3;
                        var compCol = ['#58a6ff', '#3fb950', '#f0883e'];

                        // Initialize GMM params (poor init to show convergence)
                        var mu = [[0, 0], [1, 1], [-1, -1]];
                        var sigma = [[0.8, 0.8], [0.8, 0.8], [0.8, 0.8]]; // diagonal covariances (sx, sy)
                        var pi = [1/3, 1/3, 1/3];
                        var gamma = [];
                        for (var i = 0; i < N; i++) gamma.push([1/3, 1/3, 1/3]);

                        function gauss2d(x, y, mx, my, sx, sy) {
                            var zx = (x - mx) / sx, zy = (y - my) / sy;
                            return Math.exp(-0.5 * (zx * zx + zy * zy)) / (2 * Math.PI * sx * sy);
                        }

                        function eStep() {
                            for (var i = 0; i < N; i++) {
                                var total = 0;
                                for (var k = 0; k < K; k++) {
                                    gamma[i][k] = pi[k] * gauss2d(data[i][0], data[i][1], mu[k][0], mu[k][1], sigma[k][0], sigma[k][1]);
                                    total += gamma[i][k];
                                }
                                for (var k = 0; k < K; k++) gamma[i][k] /= (total + 1e-300);
                            }
                        }

                        function mStep() {
                            for (var k = 0; k < K; k++) {
                                var Nk = 0;
                                for (var i = 0; i < N; i++) Nk += gamma[i][k];
                                var mx = 0, my = 0;
                                for (var i = 0; i < N; i++) { mx += gamma[i][k] * data[i][0]; my += gamma[i][k] * data[i][1]; }
                                mu[k] = [mx / Nk, my / Nk];
                                var sx = 0, sy = 0;
                                for (var i = 0; i < N; i++) {
                                    sx += gamma[i][k] * (data[i][0] - mu[k][0]) * (data[i][0] - mu[k][0]);
                                    sy += gamma[i][k] * (data[i][1] - mu[k][1]) * (data[i][1] - mu[k][1]);
                                }
                                sigma[k] = [Math.sqrt(sx / Nk + 0.01), Math.sqrt(sy / Nk + 0.01)];
                                pi[k] = Nk / N;
                            }
                        }

                        VizEngine.createButton(controls, 'EM Step', function() {
                            eStep(); mStep(); iteration++; draw();
                        });
                        VizEngine.createButton(controls, 'Reset', function() {
                            mu = [[0, 0], [1, 1], [-1, -1]];
                            sigma = [[0.8, 0.8], [0.8, 0.8], [0.8, 0.8]];
                            pi = [1/3, 1/3, 1/3];
                            for (var i = 0; i < N; i++) gamma[i] = [1/3, 1/3, 1/3];
                            iteration = 0; draw();
                        });

                        function draw() {
                            viz.clear(); viz.drawGrid(); viz.drawAxes();
                            // Draw ellipses for each component (2-sigma)
                            for (var k = 0; k < K; k++) {
                                viz.drawEllipse(mu[k][0], mu[k][1], sigma[k][0] * 2, sigma[k][1] * 2, 0, compCol[k] + '15', compCol[k] + '88');
                                viz.drawPoint(mu[k][0], mu[k][1], compCol[k], '\u03BC' + (k + 1), 6);
                            }
                            // Draw data points colored by max responsibility
                            for (var i = 0; i < N; i++) {
                                var maxK = 0;
                                for (var k = 1; k < K; k++) { if (gamma[i][k] > gamma[i][maxK]) maxK = k; }
                                // Blend based on responsibility
                                var alpha = Math.max(0.4, gamma[i][maxK]);
                                var hex = Math.round(alpha * 255).toString(16);
                                if (hex.length < 2) hex = '0' + hex;
                                viz.drawPoint(data[i][0], data[i][1], compCol[maxK] + hex, null, 3);
                            }
                            // Info
                            ctx.font = 'bold 13px -apple-system,sans-serif'; ctx.textAlign = 'left';
                            ctx.fillStyle = viz.colors.white;
                            ctx.fillText('Iteration: ' + iteration, 10, 20);
                            ctx.font = '11px -apple-system,sans-serif';
                            for (var k = 0; k < K; k++) {
                                ctx.fillStyle = compCol[k];
                                ctx.fillText('\u03C0' + (k+1) + '=' + pi[k].toFixed(2) + '  \u03BC=(' + mu[k][0].toFixed(1) + ',' + mu[k][1].toFixed(1) + ')', 10, 38 + k * 16);
                            }
                        }
                        eStep(); draw();
                        return { stopAnimation: function() {} };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Derive the M-step update for \\(\\boldsymbol{\\mu}_k\\) by differentiating the expected complete-data log-likelihood \\(Q(\\boldsymbol{\\theta}, \\boldsymbol{\\theta}^{(t)}) = \\sum_{i=1}^N \\sum_{k=1}^K \\gamma_{ik} [\\log \\pi_k + \\log \\mathcal{N}(\\mathbf{x}_i \\mid \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)]\\) with respect to \\(\\boldsymbol{\\mu}_k\\).',
                    hint: 'The only term depending on \\(\\boldsymbol{\\mu}_k\\) is \\(-\\frac{1}{2}(\\mathbf{x}_i - \\boldsymbol{\\mu}_k)^\\top \\boldsymbol{\\Sigma}_k^{-1}(\\mathbf{x}_i - \\boldsymbol{\\mu}_k)\\).',
                    solution: 'Differentiating \\(Q\\) with respect to \\(\\boldsymbol{\\mu}_k\\): \\(\\frac{\\partial Q}{\\partial \\boldsymbol{\\mu}_k} = \\sum_{i=1}^N \\gamma_{ik} \\boldsymbol{\\Sigma}_k^{-1}(\\mathbf{x}_i - \\boldsymbol{\\mu}_k) = \\mathbf{0}\\). Multiplying by \\(\\boldsymbol{\\Sigma}_k\\): \\(\\sum_i \\gamma_{ik}(\\mathbf{x}_i - \\boldsymbol{\\mu}_k) = \\mathbf{0}\\), so \\(\\boldsymbol{\\mu}_k = \\frac{\\sum_i \\gamma_{ik} \\mathbf{x}_i}{\\sum_i \\gamma_{ik}} = \\frac{1}{N_k}\\sum_i \\gamma_{ik} \\mathbf{x}_i\\).'
                },
                {
                    question: 'In a 1D GMM with \\(K=2\\), the current parameters are \\(\\mu_1 = 0, \\sigma_1 = 1, \\pi_1 = 0.5\\) and \\(\\mu_2 = 4, \\sigma_2 = 1, \\pi_2 = 0.5\\). Compute the responsibility \\(\\gamma_{i1}\\) for a data point \\(x_i = 1\\).',
                    hint: 'Apply Bayes\' rule: \\(\\gamma_{i1} = \\frac{\\pi_1 \\mathcal{N}(x_i \\mid \\mu_1, \\sigma_1^2)}{\\pi_1 \\mathcal{N}(x_i \\mid \\mu_1, \\sigma_1^2) + \\pi_2 \\mathcal{N}(x_i \\mid \\mu_2, \\sigma_2^2)}\\).',
                    solution: '\\(\\mathcal{N}(1 \\mid 0, 1) = \\frac{1}{\\sqrt{2\\pi}}e^{-0.5} \\approx 0.2420\\). \\(\\mathcal{N}(1 \\mid 4, 1) = \\frac{1}{\\sqrt{2\\pi}}e^{-4.5} \\approx 0.0044\\). So \\(\\gamma_{i1} = \\frac{0.5 \\times 0.2420}{0.5 \\times 0.2420 + 0.5 \\times 0.0044} = \\frac{0.1210}{0.1210 + 0.0022} \\approx 0.982\\). The point at \\(x=1\\) is almost entirely assigned to component 1, as expected since it is much closer to \\(\\mu_1 = 0\\).'
                },
                {
                    question: 'Explain the relationship between K-means and EM for GMMs. Under what conditions does EM reduce to K-means?',
                    hint: 'Consider what happens when all covariances are \\(\\sigma^2 \\mathbf{I}\\) and \\(\\sigma \\to 0\\).',
                    solution: 'K-means is a special case of EM for GMMs where (1) all components have equal, isotropic covariance \\(\\sigma^2 \\mathbf{I}\\), (2) all mixing weights are equal \\(\\pi_k = 1/K\\), and (3) \\(\\sigma \\to 0\\). In this limit, the responsibilities become hard assignments: \\(\\gamma_{ik} \\to 1\\) for the nearest cluster and \\(0\\) otherwise (since the Gaussian becomes a delta function). The M-step for \\(\\boldsymbol{\\mu}_k\\) becomes the hard cluster mean. The E-step becomes nearest-centroid assignment. Thus K-means is "hard EM" for a homoscedastic GMM.'
                }
            ]
        },

        // ===================== Section 3: EM Convergence & ELBO =====================
        {
            id: 'ch13-sec03',
            title: 'EM Convergence & ELBO',
            content: `<h2>EM Convergence &amp; ELBO</h2>

                <h3>Why Does EM Work?</h3>

                <p>The remarkable property of EM is that each iteration is guaranteed to increase (or maintain) the log-likelihood. This is not obvious from the algorithm description; the proof relies on Jensen's inequality and the concept of the <strong>evidence lower bound</strong> (ELBO).</p>

                <div class="env-block theorem">
                    <div class="env-title">Theorem (Monotonic Likelihood Increase)</div>
                    <div class="env-body"><p>For any latent variable model with observed data \\(\\mathbf{X}\\), latent variables \\(\\mathbf{Z}\\), and parameters \\(\\boldsymbol{\\theta}\\), the EM algorithm satisfies
                    \\[\\ell(\\boldsymbol{\\theta}^{(t+1)}) \\geq \\ell(\\boldsymbol{\\theta}^{(t)})\\]
                    where \\(\\ell(\\boldsymbol{\\theta}) = \\log p(\\mathbf{X} \\mid \\boldsymbol{\\theta})\\) is the (incomplete-data) log-likelihood.</p></div>
                </div>

                <h3>The Evidence Lower Bound</h3>

                <p>For any distribution \\(q(\\mathbf{Z})\\) over the latent variables, the log-likelihood decomposes as:</p>
                \\[\\ell(\\boldsymbol{\\theta}) = \\mathcal{L}(q, \\boldsymbol{\\theta}) + \\text{KL}(q \\| p(\\mathbf{Z} \\mid \\mathbf{X}, \\boldsymbol{\\theta}))\\]
                <p>where the ELBO is</p>
                \\[\\mathcal{L}(q, \\boldsymbol{\\theta}) = \\sum_{\\mathbf{Z}} q(\\mathbf{Z}) \\log \\frac{p(\\mathbf{X}, \\mathbf{Z} \\mid \\boldsymbol{\\theta})}{q(\\mathbf{Z})} = \\mathbb{E}_q[\\log p(\\mathbf{X}, \\mathbf{Z} \\mid \\boldsymbol{\\theta})] + H(q)\\]
                <p>and \\(\\text{KL}(q \\| p) \\geq 0\\). Since the KL divergence is non-negative, \\(\\mathcal{L}(q, \\boldsymbol{\\theta}) \\leq \\ell(\\boldsymbol{\\theta})\\): the ELBO is indeed a lower bound on the evidence.</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (EM as Coordinate Ascent on the ELBO)</div>
                    <div class="env-body"><p>EM can be interpreted as coordinate ascent on \\(\\mathcal{L}(q, \\boldsymbol{\\theta})\\):</p>
                    <ul>
                        <li><strong>E-step:</strong> Fix \\(\\boldsymbol{\\theta}^{(t)}\\) and maximize \\(\\mathcal{L}\\) over \\(q\\). The optimal \\(q^*\\) is the true posterior \\(q^*(\\mathbf{Z}) = p(\\mathbf{Z} \\mid \\mathbf{X}, \\boldsymbol{\\theta}^{(t)})\\), which sets \\(\\text{KL} = 0\\) and makes the bound tight.</li>
                        <li><strong>M-step:</strong> Fix \\(q = q^*\\) and maximize \\(\\mathcal{L}\\) over \\(\\boldsymbol{\\theta}\\). This is equivalent to maximizing \\(Q(\\boldsymbol{\\theta}, \\boldsymbol{\\theta}^{(t)}) = \\mathbb{E}_{q^*}[\\log p(\\mathbf{X}, \\mathbf{Z} \\mid \\boldsymbol{\\theta})]\\).</li>
                    </ul>
                    <p>Each step increases \\(\\mathcal{L}\\), which in turn increases \\(\\ell\\).</p></div>
                </div>

                <h3>Local Optima</h3>

                <p>EM guarantees convergence to a <em>local</em> maximum (or saddle point) of the likelihood, not the global maximum. Different initializations can lead to different solutions. In practice, one runs EM multiple times with different random initializations and selects the result with the highest final log-likelihood.</p>

                <div class="viz-placeholder" data-viz="viz-em-convergence"></div>

                <div class="env-block remark">
                    <div class="env-title">Remark (Convergence Rate)</div>
                    <div class="env-body"><p>EM converges linearly, with rate determined by the fraction of "missing information." When the latent variables carry little information about the parameters (well-separated clusters), EM converges fast. When the components overlap heavily and the responsibilities are uncertain, EM can be very slow. Acceleration techniques such as Aitken's method or SQUAREM can speed convergence.</p></div>
                </div>`,

            visualizations: [
                {
                    id: 'viz-em-convergence',
                    title: 'Log-Likelihood Over EM Iterations',
                    description: 'Watch the log-likelihood increase monotonically as EM iterates. Click "Run EM" to see convergence. Note the rapid early gains followed by slow refinement.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;

                        // Generate data and run EM, recording log-likelihood
                        var seed = 55;
                        function srand() { seed = (seed * 16807 + 0) % 2147483647; return (seed - 1) / 2147483646; }
                        function randn() { var u = srand(), v = srand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

                        var data = [];
                        var tMu = [[-2, 1], [2, -0.5], [0, 2.5]];
                        for (var k = 0; k < 3; k++) { for (var i = 0; i < 30; i++) data.push([tMu[k][0] + randn() * 0.7, tMu[k][1] + randn() * 0.7]); }
                        var N = data.length, K = 3;

                        var lls = [];
                        var maxIter = 0;
                        var running = false;
                        var runInterval = null;

                        // GMM state
                        var mu, sig, pi, gam;

                        function initGMM() {
                            mu = [[srand() * 4 - 2, srand() * 4 - 2], [srand() * 4 - 2, srand() * 4 - 2], [srand() * 4 - 2, srand() * 4 - 2]];
                            sig = [[1, 1], [1, 1], [1, 1]];
                            pi = [1/3, 1/3, 1/3];
                            gam = []; for (var i = 0; i < N; i++) gam.push([1/3, 1/3, 1/3]);
                            lls = [computeLL()];
                            maxIter = 0;
                        }

                        function gauss2d(x, y, mx, my, sx, sy) {
                            var zx = (x - mx) / sx, zy = (y - my) / sy;
                            return Math.exp(-0.5 * (zx * zx + zy * zy)) / (2 * Math.PI * sx * sy);
                        }

                        function computeLL() {
                            var ll = 0;
                            for (var i = 0; i < N; i++) {
                                var s = 0;
                                for (var k = 0; k < K; k++) s += pi[k] * gauss2d(data[i][0], data[i][1], mu[k][0], mu[k][1], sig[k][0], sig[k][1]);
                                ll += Math.log(s + 1e-300);
                            }
                            return ll;
                        }

                        function emStep() {
                            // E-step
                            for (var i = 0; i < N; i++) {
                                var tot = 0;
                                for (var k = 0; k < K; k++) { gam[i][k] = pi[k] * gauss2d(data[i][0], data[i][1], mu[k][0], mu[k][1], sig[k][0], sig[k][1]); tot += gam[i][k]; }
                                for (var k = 0; k < K; k++) gam[i][k] /= (tot + 1e-300);
                            }
                            // M-step
                            for (var k = 0; k < K; k++) {
                                var Nk = 0; for (var i = 0; i < N; i++) Nk += gam[i][k];
                                var mx = 0, my = 0;
                                for (var i = 0; i < N; i++) { mx += gam[i][k] * data[i][0]; my += gam[i][k] * data[i][1]; }
                                mu[k] = [mx / Nk, my / Nk];
                                var sx = 0, sy = 0;
                                for (var i = 0; i < N; i++) { sx += gam[i][k] * (data[i][0] - mu[k][0]) * (data[i][0] - mu[k][0]); sy += gam[i][k] * (data[i][1] - mu[k][1]) * (data[i][1] - mu[k][1]); }
                                sig[k] = [Math.sqrt(sx / Nk + 0.01), Math.sqrt(sy / Nk + 0.01)];
                                pi[k] = Nk / N;
                            }
                            maxIter++;
                            lls.push(computeLL());
                        }

                        VizEngine.createButton(controls, 'Run EM', function() {
                            if (running) return;
                            running = true;
                            runInterval = setInterval(function() {
                                emStep(); draw();
                                if (maxIter >= 40) { clearInterval(runInterval); running = false; }
                            }, 120);
                        });
                        VizEngine.createButton(controls, 'Reset', function() {
                            if (runInterval) clearInterval(runInterval);
                            running = false; seed = Math.floor(Math.random() * 10000);
                            initGMM(); draw();
                        });

                        function draw() {
                            ctx.fillStyle = viz.colors.bg; ctx.fillRect(0, 0, W, H);
                            var m = {top: 40, right: 30, bottom: 45, left: 70};
                            var pw = W - m.left - m.right, ph = H - m.top - m.bottom;
                            // Axes
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + ph); ctx.lineTo(m.left + pw, m.top + ph); ctx.stroke();
                            ctx.fillStyle = viz.colors.text; ctx.font = '12px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            ctx.fillText('EM iteration', m.left + pw / 2, H - 8);
                            ctx.save(); ctx.translate(16, m.top + ph / 2); ctx.rotate(-Math.PI / 2);
                            ctx.fillText('Log-likelihood', 0, 0); ctx.restore();
                            if (lls.length < 2) { ctx.fillStyle = viz.colors.text; ctx.textAlign = 'center'; ctx.fillText('Click "Run EM" to start', W / 2, H / 2); return; }
                            var minLL = lls[0], maxLL = lls[lls.length - 1];
                            var range = maxLL - minLL;
                            if (range < 1) range = 1;
                            minLL -= range * 0.1; maxLL += range * 0.05;
                            range = maxLL - minLL;
                            var maxT = Math.max(lls.length - 1, 10);
                            // Plot curve
                            ctx.strokeStyle = viz.colors.teal; ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var t = 0; t < lls.length; t++) {
                                var px = m.left + (t / maxT) * pw;
                                var py = m.top + ph - ((lls[t] - minLL) / range) * ph;
                                if (t === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                            }
                            ctx.stroke();
                            // Dots
                            for (var t = 0; t < lls.length; t++) {
                                var px = m.left + (t / maxT) * pw;
                                var py = m.top + ph - ((lls[t] - minLL) / range) * ph;
                                ctx.fillStyle = viz.colors.teal;
                                ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
                            }
                            // Current value
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            ctx.fillText('Iteration ' + maxIter + ': LL = ' + lls[lls.length - 1].toFixed(2), W / 2, 22);
                            // Y-axis ticks
                            ctx.fillStyle = viz.colors.text; ctx.font = '10px -apple-system,sans-serif'; ctx.textAlign = 'right';
                            for (var f = 0; f <= 4; f++) {
                                var val = minLL + (f / 4) * range;
                                var yy = m.top + ph - (f / 4) * ph;
                                ctx.fillText(val.toFixed(1), m.left - 6, yy + 3);
                            }
                        }

                        initGMM(); draw();
                        return { stopAnimation: function() { if (runInterval) clearInterval(runInterval); } };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Derive the ELBO decomposition: show that \\(\\log p(\\mathbf{X} \\mid \\boldsymbol{\\theta}) = \\mathcal{L}(q, \\boldsymbol{\\theta}) + \\text{KL}(q \\| p(\\mathbf{Z} \\mid \\mathbf{X}, \\boldsymbol{\\theta}))\\) for any distribution \\(q(\\mathbf{Z})\\).',
                    hint: 'Start from \\(\\log p(\\mathbf{X} \\mid \\boldsymbol{\\theta}) = \\log p(\\mathbf{X}, \\mathbf{Z} \\mid \\boldsymbol{\\theta}) - \\log p(\\mathbf{Z} \\mid \\mathbf{X}, \\boldsymbol{\\theta})\\) and take expectation under \\(q(\\mathbf{Z})\\).',
                    solution: 'Take \\(\\mathbb{E}_q\\) of both sides of \\(\\log p(\\mathbf{X}) = \\log p(\\mathbf{X}, \\mathbf{Z}) - \\log p(\\mathbf{Z} \\mid \\mathbf{X})\\). The left side is constant in \\(\\mathbf{Z}\\), so it stays \\(\\log p(\\mathbf{X})\\). The right side gives \\(\\mathbb{E}_q[\\log p(\\mathbf{X}, \\mathbf{Z})] - \\mathbb{E}_q[\\log p(\\mathbf{Z} \\mid \\mathbf{X})]\\). Add and subtract \\(\\mathbb{E}_q[\\log q(\\mathbf{Z})]\\): \\(\\log p(\\mathbf{X}) = \\underbrace{\\mathbb{E}_q[\\log p(\\mathbf{X},\\mathbf{Z})] - \\mathbb{E}_q[\\log q(\\mathbf{Z})]}_\\text{ELBO} + \\underbrace{\\mathbb{E}_q[\\log q(\\mathbf{Z})] - \\mathbb{E}_q[\\log p(\\mathbf{Z}\\mid\\mathbf{X})]}_\\text{KL}\\).'
                },
                {
                    question: 'Explain why setting \\(q(\\mathbf{Z}) = p(\\mathbf{Z} \\mid \\mathbf{X}, \\boldsymbol{\\theta}^{(t)})\\) in the E-step makes the ELBO tight (i.e., equal to the log-likelihood).',
                    hint: 'What is the KL divergence of a distribution with itself?',
                    solution: 'When \\(q = p(\\mathbf{Z} \\mid \\mathbf{X}, \\boldsymbol{\\theta}^{(t)})\\), the KL divergence \\(\\text{KL}(q \\| p(\\mathbf{Z} \\mid \\mathbf{X}, \\boldsymbol{\\theta}^{(t)})) = 0\\), since the divergence of a distribution with itself is zero. Therefore \\(\\mathcal{L}(q, \\boldsymbol{\\theta}^{(t)}) = \\ell(\\boldsymbol{\\theta}^{(t)})\\): the bound is tight at the current parameter value. The M-step then increases \\(\\mathcal{L}\\) over \\(\\boldsymbol{\\theta}\\), which must also increase \\(\\ell\\) since \\(\\ell \\geq \\mathcal{L}\\).'
                },
                {
                    question: 'A GMM with \\(K=3\\) is initialized with all means at the data centroid. After running EM, you find that one component has \\(\\pi_k \\approx 0\\) and the other two capture the data. What happened, and how might you address it?',
                    hint: 'Think about the symmetry of the initialization and the fact that EM can only find local optima.',
                    solution: 'When all means start at the same point, the initial symmetry is fragile. Small numerical perturbations cause one or two components to "win" and attract most of the data, while a component that moves to a low-density region gets \\(N_k \\to 0\\) and its mixing weight \\(\\pi_k \\to 0\\). This is a symptom of local optima. Remedies include: (1) random initialization from different starting points; (2) K-means++ initialization for the means; (3) running EM multiple times and selecting the best log-likelihood; (4) split-and-merge heuristics that can break out of poor local optima.'
                }
            ]
        },

        // ===================== Section 4: EM Beyond GMMs =====================
        {
            id: 'ch13-sec04',
            title: 'EM Beyond GMMs',
            content: `<h2>EM Beyond GMMs</h2>

                <div class="env-block intuition">
                    <div class="env-title">A General Framework</div>
                    <div class="env-body"><p>The EM algorithm applies to <em>any</em> latent variable model, not just Gaussian mixtures. Whenever the log-likelihood is hard to optimize directly but the <em>complete-data</em> log-likelihood (with latent variables observed) is tractable, EM provides a principled iterative approach.</p></div>
                </div>

                <h3>The General EM Algorithm</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (General EM)</div>
                    <div class="env-body"><p>Given observed data \\(\\mathbf{X}\\), latent variables \\(\\mathbf{Z}\\), and a model \\(p(\\mathbf{X}, \\mathbf{Z} \\mid \\boldsymbol{\\theta})\\):</p>
                    <p><strong>E-step:</strong> Compute the expected complete-data log-likelihood under the current posterior:
                    \\[Q(\\boldsymbol{\\theta}, \\boldsymbol{\\theta}^{(t)}) = \\mathbb{E}_{p(\\mathbf{Z} \\mid \\mathbf{X}, \\boldsymbol{\\theta}^{(t)})}[\\log p(\\mathbf{X}, \\mathbf{Z} \\mid \\boldsymbol{\\theta})]\\]</p>
                    <p><strong>M-step:</strong> Maximize the Q-function over \\(\\boldsymbol{\\theta}\\):
                    \\[\\boldsymbol{\\theta}^{(t+1)} = \\arg\\max_{\\boldsymbol{\\theta}} Q(\\boldsymbol{\\theta}, \\boldsymbol{\\theta}^{(t)})\\]</p></div>
                </div>

                <h3>EM for Missing Data</h3>

                <p>A natural application of EM is handling <strong>missing data</strong>. Suppose we observe \\(\\mathbf{X}_{\\text{obs}}\\) and some entries \\(\\mathbf{X}_{\\text{miss}}\\) are missing. If we assume data is missing at random (MAR), we can treat \\(\\mathbf{X}_{\\text{miss}}\\) as latent variables.</p>

                <div class="env-block example">
                    <div class="env-title">Example (EM for Missing Data in a Multivariate Gaussian)</div>
                    <div class="env-body"><p>Suppose \\(\\mathbf{x}_i \\sim \\mathcal{N}(\\boldsymbol{\\mu}, \\boldsymbol{\\Sigma})\\) and some entries of \\(\\mathbf{x}_i\\) are unobserved. The E-step computes the conditional expectation of each missing entry given the observed entries:
                    \\[\\mathbb{E}[x_{ij}^{\\text{miss}} \\mid \\mathbf{x}_i^{\\text{obs}}, \\boldsymbol{\\mu}, \\boldsymbol{\\Sigma}] = \\mu_j + \\boldsymbol{\\Sigma}_{j, \\text{obs}} \\boldsymbol{\\Sigma}_{\\text{obs,obs}}^{-1}(\\mathbf{x}_i^{\\text{obs}} - \\boldsymbol{\\mu}_{\\text{obs}})\\]
                    The M-step re-estimates \\(\\boldsymbol{\\mu}\\) and \\(\\boldsymbol{\\Sigma}\\) from the "filled-in" data (plus the appropriate second-moment corrections).</p></div>
                </div>

                <h3>EM for Hidden Markov Models</h3>

                <p>In a <strong>hidden Markov model</strong> (HMM), the latent variables are the hidden states \\(z_1, \\dots, z_T\\) forming a Markov chain. The E-step uses the <strong>forward-backward algorithm</strong> (Baum-Welch) to compute posterior marginals \\(\\gamma_t(k) = p(z_t = k \\mid \\mathbf{x}_{1:T})\\) and pairwise marginals \\(\\xi_t(j,k) = p(z_t = j, z_{t+1} = k \\mid \\mathbf{x}_{1:T})\\). The M-step updates the transition matrix, emission distributions, and initial state distribution using these sufficient statistics.</p>

                <h3>Other Applications</h3>

                <div class="env-block remark">
                    <div class="env-title">Remark (Breadth of EM)</div>
                    <div class="env-body"><p>EM appears throughout machine learning and statistics:</p>
                    <ul>
                        <li><strong>Factor analysis:</strong> latent factors \\(\\mathbf{z}\\) generate observed features \\(\\mathbf{x} = \\mathbf{W}\\mathbf{z} + \\boldsymbol{\\epsilon}\\).</li>
                        <li><strong>Probabilistic PCA:</strong> a special case of factor analysis where the noise is isotropic; EM recovers the PCA solution.</li>
                        <li><strong>Mixture of experts:</strong> a gating network selects among expert models; the expert assignments are latent.</li>
                        <li><strong>Topic models (pLSA):</strong> document-word co-occurrences are explained by latent topics.</li>
                    </ul></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-em-missing"></div>`,

            visualizations: [
                {
                    id: 'viz-em-missing',
                    title: 'EM for Missing Data: Filled-In Values',
                    description: 'A 2D dataset where some values are missing (shown as hollow circles on the axes). EM fills in the missing values using the current model. Watch how the imputed values (crosses) converge as EM iterates.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {scale: 30});
                        var ctx = viz.ctx;
                        var iteration = 0;

                        var seed = 99;
                        function srand() { seed = (seed * 16807 + 0) % 2147483647; return (seed - 1) / 2147483646; }
                        function randn() { var u = srand(), v = srand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

                        // Generate complete data
                        var trueMu = [1.0, 0.5];
                        var trueCorr = 0.7;
                        var complete = [];
                        for (var i = 0; i < 40; i++) {
                            var z1 = randn(), z2 = randn();
                            var x1 = trueMu[0] + z1 * 1.5;
                            var x2 = trueMu[1] + (trueCorr * z1 + Math.sqrt(1 - trueCorr * trueCorr) * z2) * 1.5;
                            complete.push([x1, x2]);
                        }
                        // Introduce missingness: ~25% of x2 missing
                        var missing = []; // indices where x2 is missing
                        for (var i = 0; i < complete.length; i++) {
                            if (srand() < 0.25) missing.push(i);
                        }

                        // EM state for multivariate Gaussian
                        var mu = [0, 0];
                        var cov = [[1, 0], [0, 1]]; // [[s11, s12], [s21, s22]]
                        var imputed = []; // imputed x2 for missing entries
                        for (var i = 0; i < missing.length; i++) imputed.push(0);

                        function emIteration() {
                            // E-step: fill in missing x2 values
                            for (var m = 0; m < missing.length; m++) {
                                var idx = missing[m];
                                var x1 = complete[idx][0];
                                // E[x2 | x1] = mu2 + cov12/cov11 * (x1 - mu1)
                                imputed[m] = mu[1] + (cov[0][1] / (cov[0][0] + 1e-10)) * (x1 - mu[0]);
                            }
                            // M-step: re-estimate mu and cov from filled data
                            var N = complete.length;
                            var sx = 0, sy = 0;
                            for (var i = 0; i < N; i++) {
                                sx += complete[i][0];
                                var mIdx = missing.indexOf(i);
                                sy += mIdx >= 0 ? imputed[mIdx] : complete[i][1];
                            }
                            mu = [sx / N, sy / N];
                            var s11 = 0, s12 = 0, s22 = 0;
                            for (var i = 0; i < N; i++) {
                                var x1 = complete[i][0];
                                var mIdx = missing.indexOf(i);
                                var x2 = mIdx >= 0 ? imputed[mIdx] : complete[i][1];
                                s11 += (x1 - mu[0]) * (x1 - mu[0]);
                                s12 += (x1 - mu[0]) * (x2 - mu[1]);
                                s22 += (x2 - mu[1]) * (x2 - mu[1]);
                                // For missing entries, add conditional variance correction
                                if (mIdx >= 0) {
                                    var condVar = cov[1][1] - cov[0][1] * cov[0][1] / (cov[0][0] + 1e-10);
                                    s22 += Math.max(condVar, 0.01);
                                }
                            }
                            cov = [[s11 / N, s12 / N], [s12 / N, s22 / N]];
                            iteration++;
                        }

                        VizEngine.createButton(controls, 'EM Step', function() { emIteration(); draw(); });
                        VizEngine.createButton(controls, 'Run 10 Steps', function() { for (var i = 0; i < 10; i++) emIteration(); draw(); });
                        VizEngine.createButton(controls, 'Reset', function() {
                            mu = [0, 0]; cov = [[1, 0], [0, 1]];
                            for (var i = 0; i < imputed.length; i++) imputed[i] = 0;
                            iteration = 0; draw();
                        });

                        function draw() {
                            viz.clear(); viz.drawGrid(); viz.drawAxes();
                            // Draw observed points
                            for (var i = 0; i < complete.length; i++) {
                                if (missing.indexOf(i) >= 0) continue;
                                viz.drawPoint(complete[i][0], complete[i][1], viz.colors.blue + 'cc', null, 3.5);
                            }
                            // Draw missing points: x1 observed, x2 missing
                            for (var m = 0; m < missing.length; m++) {
                                var idx = missing[m];
                                var x1 = complete[idx][0];
                                // Hollow circle on x-axis for observed x1
                                var sx = viz.originX + x1 * viz.scale;
                                var sy = viz.originY;
                                ctx.strokeStyle = viz.colors.red; ctx.lineWidth = 1.5;
                                ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.stroke();
                                // Cross for imputed value
                                var impX = x1, impY = imputed[m];
                                var sp = viz.toScreen(impX, impY);
                                ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 2;
                                ctx.beginPath(); ctx.moveTo(sp[0] - 5, sp[1] - 5); ctx.lineTo(sp[0] + 5, sp[1] + 5); ctx.stroke();
                                ctx.beginPath(); ctx.moveTo(sp[0] + 5, sp[1] - 5); ctx.lineTo(sp[0] - 5, sp[1] + 5); ctx.stroke();
                                // True value (faint)
                                viz.drawPoint(complete[idx][0], complete[idx][1], viz.colors.green + '44', null, 3);
                                // Line from imputed to true
                                viz.drawSegment(impX, impY, complete[idx][0], complete[idx][1], viz.colors.green + '33', 0.8, true);
                            }
                            // Draw estimated mean
                            viz.drawPoint(mu[0], mu[1], viz.colors.white, '\u03BC', 5);
                            // Legend
                            ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'left';
                            ctx.fillStyle = viz.colors.blue; ctx.fillText('\u25CF Observed', 10, 20);
                            ctx.fillStyle = viz.colors.orange; ctx.fillText('\u00D7 Imputed (EM)', 10, 36);
                            ctx.fillStyle = viz.colors.green; ctx.fillText('\u25CB True (hidden)', 10, 52);
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 12px -apple-system,sans-serif';
                            ctx.fillText('Iteration: ' + iteration, viz.width - 130, 20);
                            ctx.font = '10px -apple-system,sans-serif'; ctx.fillStyle = viz.colors.text;
                            ctx.fillText('\u03BC=(' + mu[0].toFixed(2) + ', ' + mu[1].toFixed(2) + ')', viz.width - 130, 36);
                            ctx.fillText('\u03C1=' + (cov[0][1] / Math.sqrt(cov[0][0] * cov[1][1] + 1e-10)).toFixed(3), viz.width - 130, 50);
                        }
                        draw();
                        return { stopAnimation: function() {} };
                    }
                }
            ],

            exercises: [
                {
                    question: 'In a factor analysis model \\(\\mathbf{x} = \\mathbf{W}\\mathbf{z} + \\boldsymbol{\\mu} + \\boldsymbol{\\epsilon}\\) with \\(\\mathbf{z} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})\\) and \\(\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\boldsymbol{\\Psi})\\) (diagonal), what are the latent variables, and what does the E-step compute?',
                    hint: 'The latent variables are \\(\\mathbf{z}\\). The E-step needs the posterior \\(p(\\mathbf{z} \\mid \\mathbf{x}, \\mathbf{W}, \\boldsymbol{\\Psi})\\), which is Gaussian.',
                    solution: 'The latent variables are the factors \\(\\mathbf{z} \\in \\mathbb{R}^d\\). The E-step computes the posterior \\(p(\\mathbf{z} \\mid \\mathbf{x}, \\mathbf{W}, \\boldsymbol{\\Psi}) = \\mathcal{N}(\\mathbf{z} \\mid \\boldsymbol{\\mu}_z, \\boldsymbol{\\Sigma}_z)\\) where \\(\\boldsymbol{\\Sigma}_z = (\\mathbf{I} + \\mathbf{W}^\\top \\boldsymbol{\\Psi}^{-1}\\mathbf{W})^{-1}\\) and \\(\\boldsymbol{\\mu}_z = \\boldsymbol{\\Sigma}_z \\mathbf{W}^\\top \\boldsymbol{\\Psi}^{-1}(\\mathbf{x} - \\boldsymbol{\\mu})\\). These are the sufficient statistics (\\(\\mathbb{E}[\\mathbf{z}]\\) and \\(\\mathbb{E}[\\mathbf{z}\\mathbf{z}^\\top]\\)) needed for the M-step.'
                },
                {
                    question: 'Explain why EM for missing data under a Gaussian model does <em>not</em> simply replace missing values with the unconditional mean \\(\\mu_j\\). What additional information does the conditional imputation use?',
                    hint: 'The conditional expectation \\(\\mathbb{E}[x_j^{\\text{miss}} \\mid \\mathbf{x}^{\\text{obs}}]\\) uses the correlation between observed and missing variables.',
                    solution: 'Simple mean imputation (\\(\\hat{x}_j = \\mu_j\\)) ignores the relationship between the missing variable and the observed variables. The EM conditional imputation uses the formula \\(\\mathbb{E}[x_j \\mid \\mathbf{x}^{\\text{obs}}] = \\mu_j + \\boldsymbol{\\Sigma}_{j,\\text{obs}}\\boldsymbol{\\Sigma}_{\\text{obs,obs}}^{-1}(\\mathbf{x}^{\\text{obs}} - \\boldsymbol{\\mu}_{\\text{obs}})\\), which is a regression of the missing variable on the observed variables. This exploits the correlation structure, producing more accurate imputations. Furthermore, the M-step variance correction (adding back the conditional variance) ensures that the covariance estimate is not deflated by the imputation.'
                },
                {
                    question: 'Describe the Baum-Welch algorithm for HMMs as an instance of EM. What are the E-step and M-step?',
                    hint: 'The forward-backward algorithm computes the posterior over hidden states given observations and current parameters.',
                    solution: 'In an HMM with states \\(z_t\\), observations \\(x_t\\), transition matrix \\(\\mathbf{A}\\), emission distributions \\(p(x_t \\mid z_t)\\), and initial distribution \\(\\boldsymbol{\\pi}\\): <br><strong>E-step:</strong> Run the forward-backward algorithm to compute \\(\\gamma_t(k) = p(z_t = k \\mid x_{1:T})\\) and \\(\\xi_t(j,k) = p(z_t = j, z_{t+1} = k \\mid x_{1:T})\\). These are the expected sufficient statistics. <br><strong>M-step:</strong> Update \\(\\hat{\\pi}_k = \\gamma_1(k)\\), \\(\\hat{A}_{jk} = \\sum_{t=1}^{T-1} \\xi_t(j,k) / \\sum_{t=1}^{T-1} \\gamma_t(j)\\), and the emission parameters using the \\(\\gamma_t(k)\\) as soft state assignments. The algorithm is guaranteed to increase the log-likelihood at each step.'
                }
            ]
        }
    ]
});
