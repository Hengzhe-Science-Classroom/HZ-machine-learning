// === Chapter 17: Sampling & MCMC ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch17',
    number: 17,
    title: 'Sampling & MCMC',
    subtitle: 'Approximating intractable integrals and distributions through stochastic simulation',
    sections: [
        // ========== SECTION 1: Monte Carlo Methods ==========
        {
            id: 'ch17-sec01',
            title: 'Monte Carlo Methods',
            content: `
<h2>17.1 Monte Carlo Methods</h2>

<div class="env-block intuition"><div class="env-title">Intuition -- Replacing Integrals with Averages</div><div class="env-body">
Many quantities in machine learning involve integrals that have no closed form: posterior expectations, marginal likelihoods, partition functions. Monte Carlo methods replace these integrals with sample averages. If we can draw samples \\(x^{(1)}, \\ldots, x^{(N)}\\) from a distribution \\(p(x)\\), then \\(\\mathbb{E}_p[f(x)] \\approx \\frac{1}{N}\\sum_{i=1}^N f(x^{(i)})\\), and the law of large numbers guarantees convergence.
</div></div>

<h3>The Monte Carlo Estimator</h3>

<div class="env-block definition"><div class="env-title">Definition 17.1.1 -- Monte Carlo Estimator</div><div class="env-body">
Given i.i.d. samples \\(x^{(1)}, \\ldots, x^{(N)} \\sim p(x)\\), the <em>Monte Carlo estimator</em> of \\(I = \\mathbb{E}_p[f(x)] = \\int f(x)\\,p(x)\\,dx\\) is
\\[
\\hat{I}_N = \\frac{1}{N} \\sum_{i=1}^{N} f(x^{(i)}).
\\]
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 17.1.2 -- Properties of the MC Estimator</div><div class="env-body">
<ol>
<li><strong>Unbiased</strong>: \\(\\mathbb{E}[\\hat{I}_N] = I\\).</li>
<li><strong>Consistent</strong>: \\(\\hat{I}_N \\xrightarrow{a.s.} I\\) as \\(N \\to \\infty\\) (SLLN).</li>
<li><strong>Variance</strong>: \\(\\mathrm{Var}(\\hat{I}_N) = \\frac{\\mathrm{Var}_p(f)}{N}\\). The error decreases as \\(O(1/\\sqrt{N})\\), independent of dimension.</li>
</ol>
</div></div>

<p>The dimension-free convergence rate is Monte Carlo's great advantage over deterministic quadrature, which typically suffers the curse of dimensionality.</p>

<h3>Monte Carlo Integration by Random Sampling</h3>

<p>A simple application: estimate the area under a curve \\(g(x)\\) on \\([a, b]\\). If we sample \\(x^{(i)} \\sim \\mathrm{Uniform}(a, b)\\), then</p>
\\[
\\int_a^b g(x)\\,dx = (b - a)\\,\\mathbb{E}_{\\mathrm{Unif}}[g(x)] \\approx \\frac{b - a}{N} \\sum_{i=1}^N g(x^{(i)}).
\\]

<h3>Importance Sampling</h3>

<p>What if we cannot sample from \\(p\\) directly? Importance sampling uses a <em>proposal distribution</em> \\(q(x)\\) instead.</p>

<div class="env-block definition"><div class="env-title">Definition 17.1.3 -- Importance Sampling</div><div class="env-body">
\\[
\\mathbb{E}_p[f(x)] = \\int f(x) \\frac{p(x)}{q(x)} q(x)\\,dx = \\mathbb{E}_q\\!\\left[f(x)\\,w(x)\\right], \\quad w(x) = \\frac{p(x)}{q(x)}.
\\]
The IS estimator is \\(\\hat{I} = \\frac{1}{N}\\sum_{i=1}^N f(x^{(i)})\\,w(x^{(i)})\\) with \\(x^{(i)} \\sim q\\).
</div></div>

<div class="env-block remark"><div class="env-title">Remark -- Self-Normalized IS</div><div class="env-body">
When \\(p\\) is known only up to a constant (as with Bayesian posteriors), use <em>self-normalized</em> importance sampling:
\\[
\\hat{I} = \\frac{\\sum_i w(x^{(i)}) f(x^{(i)})}{\\sum_i w(x^{(i)})}.
\\]
This is biased but consistent, and avoids needing the normalization constant \\(Z\\).
</div></div>

<div class="viz-placeholder" data-viz="viz-mc-integration"></div>
`,
            visualizations: [
                {
                    id: 'viz-mc-integration',
                    title: 'Monte Carlo Integration',
                    description: 'Estimate the area under \\(f(x) = \\sin(\\pi x)\\) on \\([0, 1]\\) by dropping random points. Points below the curve are hits (blue); above are misses (gray). The estimate converges to the true area \\(2/\\pi \\approx 0.6366\\).',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 1, originX: 0, originY: 0 });
                        const ctx = viz.ctx, W = viz.width, H = viz.height;
                        const margin = { l: 60, r: 30, t: 30, b: 50 };
                        const pw = W - margin.l - margin.r, ph = H - margin.t - margin.b;
                        const f = x => Math.sin(Math.PI * x);
                        const trueArea = 2 / Math.PI;
                        let points = [], hits = 0, estimates = [];

                        function toP(x, y) { return [margin.l + x * pw, margin.t + (1 - y) * ph]; }

                        function addPoints(n) {
                            for (let i = 0; i < n; i++) {
                                const x = Math.random(), y = Math.random();
                                const hit = y <= f(x);
                                if (hit) hits++;
                                points.push({ x, y, hit });
                                if (points.length % 5 === 0) estimates.push(hits / points.length);
                            }
                        }

                        function draw() {
                            viz.clear();
                            // Axes
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(margin.l, margin.t); ctx.lineTo(margin.l, margin.t + ph);
                            ctx.lineTo(margin.l + pw, margin.t + ph); ctx.stroke();
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            for (let x = 0; x <= 1; x += 0.2) {
                                const [px, py] = toP(x, 0);
                                ctx.fillText(x.toFixed(1), px, py + 15);
                            }
                            ctx.textAlign = 'right';
                            for (let y = 0; y <= 1; y += 0.2) {
                                const [px, py] = toP(0, y);
                                ctx.fillText(y.toFixed(1), px - 8, py + 4);
                            }

                            // Draw curve
                            ctx.beginPath();
                            for (let i = 0; i <= 200; i++) {
                                const x = i / 200;
                                const [px, py] = toP(x, f(x));
                                i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
                            }
                            ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 2.5; ctx.stroke();

                            // Fill under curve
                            ctx.beginPath(); ctx.moveTo(...toP(0, 0));
                            for (let i = 0; i <= 200; i++) { const x = i/200; ctx.lineTo(...toP(x, f(x))); }
                            ctx.lineTo(...toP(1, 0)); ctx.closePath();
                            ctx.fillStyle = viz.colors.orange + '15'; ctx.fill();

                            // Draw points
                            for (const p of points) {
                                const [px, py] = toP(p.x, p.y);
                                ctx.fillStyle = p.hit ? viz.colors.blue : '#ffffff22';
                                ctx.beginPath(); ctx.arc(px, py, 1.5, 0, Math.PI * 2); ctx.fill();
                            }

                            // Stats
                            const est = points.length > 0 ? hits / points.length : 0;
                            const err = Math.abs(est - trueArea);
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'left';
                            ctx.fillText('N = ' + points.length, margin.l + 10, margin.t + 18);
                            ctx.fillStyle = viz.colors.teal; ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillText('Estimate: ' + est.toFixed(4), margin.l + 10, margin.t + 36);
                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillText('True: ' + trueArea.toFixed(4), margin.l + 10, margin.t + 52);
                            ctx.fillStyle = viz.colors.yellow;
                            ctx.fillText('Error: ' + err.toFixed(4), margin.l + 10, margin.t + 68);

                            // Convergence mini-plot
                            if (estimates.length > 1) {
                                const cpx = W - 160, cpy = margin.t + 10, cpw = 130, cph = 60;
                                ctx.fillStyle = '#0c0c20cc'; ctx.fillRect(cpx, cpy, cpw, cph);
                                ctx.strokeStyle = '#30363d'; ctx.strokeRect(cpx, cpy, cpw, cph);
                                // True line
                                const ty = cpy + cph * (1 - (trueArea - 0.4) / 0.5);
                                ctx.setLineDash([3, 3]); ctx.strokeStyle = viz.colors.orange;
                                ctx.beginPath(); ctx.moveTo(cpx, ty); ctx.lineTo(cpx + cpw, ty); ctx.stroke();
                                ctx.setLineDash([]);
                                // Estimate line
                                ctx.beginPath(); ctx.strokeStyle = viz.colors.teal; ctx.lineWidth = 1.5;
                                for (let i = 0; i < estimates.length; i++) {
                                    const ex = cpx + (i / (estimates.length - 1)) * cpw;
                                    const ey = cpy + cph * (1 - (estimates[i] - 0.4) / 0.5);
                                    i === 0 ? ctx.moveTo(ex, ey) : ctx.lineTo(ex, ey);
                                }
                                ctx.stroke();
                                ctx.fillStyle = viz.colors.text; ctx.font = '9px -apple-system,sans-serif';
                                ctx.textAlign = 'center'; ctx.fillText('Convergence', cpx + cpw / 2, cpy + cph + 12);
                            }
                        }

                        VizEngine.createButton(controls, '+10', () => { addPoints(10); draw(); });
                        VizEngine.createButton(controls, '+100', () => { addPoints(100); draw(); });
                        VizEngine.createButton(controls, '+1000', () => { addPoints(1000); draw(); });
                        VizEngine.createButton(controls, 'Reset', () => { points = []; hits = 0; estimates = []; draw(); });

                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that the MC estimator has variance \\(\\mathrm{Var}(\\hat{I}_N) = \\mathrm{Var}_p(f) / N\\). Why is the \\(O(1/\\sqrt{N})\\) rate dimension-free?',
                    hint: 'Use the formula for variance of a sample mean of i.i.d. variables.',
                    solution: 'Since \\(\\hat{I}_N = \\frac{1}{N}\\sum_i f(x^{(i)})\\) with i.i.d. \\(x^{(i)} \\sim p\\), \\(\\mathrm{Var}(\\hat{I}_N) = \\frac{1}{N^2}\\sum_i \\mathrm{Var}(f(x^{(i)})) = \\frac{\\mathrm{Var}_p(f)}{N}\\). The standard error is \\(\\sigma/\\sqrt{N}\\). This rate depends only on \\(N\\) and the integrand variance, not on the dimension of \\(x\\). Deterministic quadrature in \\(d\\) dimensions typically needs \\(O(\\epsilon^{-d})\\) points for accuracy \\(\\epsilon\\), but MC always needs \\(O(\\epsilon^{-2})\\).'
                },
                {
                    question: 'Derive the optimal proposal distribution \\(q^*(x)\\) for importance sampling that minimizes \\(\\mathrm{Var}_q[f(x)\\,w(x)]\\).',
                    hint: 'The optimal proposal is \\(q^*(x) \\propto |f(x)|\\,p(x)\\).',
                    solution: 'We want to minimize \\(\\mathrm{Var}_q[fw] = \\mathbb{E}_q[(fw)^2] - I^2\\). Since \\(I^2\\) is constant, minimize \\(\\int f(x)^2 p(x)^2/q(x)\\,dx\\). By Cauchy-Schwarz or Lagrange multipliers, the minimum occurs at \\(q^*(x) = |f(x)|p(x)/\\int|f(x)|p(x)dx\\). If \\(f \\geq 0\\), then \\(q^* = fp/I\\), and the variance becomes zero (every sample gives the same weighted value). In practice, \\(q^*\\) itself requires knowing \\(I\\), so the result is theoretical but guides the choice of \\(q\\) to concentrate mass where \\(|f|p\\) is large.'
                },
                {
                    question: 'Explain why importance sampling can fail catastrophically in high dimensions even when \\(q\\) seems reasonable.',
                    hint: 'Think about the effective sample size and weight degeneracy.',
                    solution: 'In high dimensions, even mild mismatch between \\(p\\) and \\(q\\) causes extreme weight variation. The effective sample size \\(N_{\\mathrm{eff}} = (\\sum_i w_i)^2 / \\sum_i w_i^2\\) can collapse to \\(O(1)\\) even with millions of samples. This happens because the typical set of \\(p\\) and \\(q\\) become nearly disjoint in high dimensions. A few samples receive nearly all the weight, making the estimate dominated by a single point with huge variance. This is sometimes called "weight degeneracy" and motivates the use of MCMC instead.'
                }
            ]
        },

        // ========== SECTION 2: Metropolis-Hastings ==========
        {
            id: 'ch17-sec02',
            title: 'Metropolis-Hastings',
            content: `
<h2>17.2 Metropolis-Hastings</h2>

<div class="env-block intuition"><div class="env-title">Intuition -- A Random Walk That Learns the Landscape</div><div class="env-body">
Imagine exploring a mountain range at night with a flashlight that only illuminates your feet. At each step, you propose a random direction. If the new spot is higher (more probable), you move there. If it is lower, you sometimes move there anyway (with probability proportional to the height ratio). Over time, you spend more time on peaks than in valleys, and your trajectory samples the landscape proportional to elevation.
</div></div>

<h3>The Algorithm</h3>

<div class="env-block definition"><div class="env-title">Definition 17.2.1 -- Metropolis-Hastings Algorithm</div><div class="env-body">
Given a target distribution \\(p(x)\\) (known up to a constant) and a proposal distribution \\(q(x' \\mid x)\\):
<ol>
<li>Initialize \\(x^{(0)}\\).</li>
<li>For \\(t = 1, 2, \\ldots\\):<br>
   (a) Propose \\(x' \\sim q(x' \\mid x^{(t-1)})\\).<br>
   (b) Compute acceptance ratio \\(\\alpha = \\min\\!\\left(1,\\; \\frac{p(x')\\,q(x^{(t-1)} \\mid x')}{p(x^{(t-1)})\\,q(x' \\mid x^{(t-1)})}\\right)\\).<br>
   (c) Accept \\(x^{(t)} = x'\\) with probability \\(\\alpha\\); otherwise \\(x^{(t)} = x^{(t-1)}\\).
</li>
</ol>
</div></div>

<p>The key insight: we only need the <em>ratio</em> \\(p(x')/p(x^{(t-1)})\\), so the normalization constant \\(Z\\) cancels. This is what makes MH applicable to Bayesian posteriors and MRFs.</p>

<h3>Detailed Balance</h3>

<div class="env-block theorem"><div class="env-title">Theorem 17.2.2 -- Detailed Balance</div><div class="env-body">
The MH chain satisfies <em>detailed balance</em> with respect to \\(p\\):
\\[
p(x)\\,T(x' \\mid x) = p(x')\\,T(x \\mid x'),
\\]
where \\(T(x' \\mid x) = q(x' \\mid x)\\,\\alpha(x, x')\\) is the transition kernel. Detailed balance implies \\(p\\) is a stationary distribution. Under mild regularity conditions (irreducibility, aperiodicity), the chain converges to \\(p\\).
</div></div>

<h3>Choosing the Proposal</h3>

<p>Common choices:</p>
<ul>
<li><strong>Random walk MH</strong>: \\(q(x' \\mid x) = \\mathcal{N}(x, \\sigma^2 I)\\). The scale \\(\\sigma\\) controls the step size. Too small: slow exploration. Too large: frequent rejections.</li>
<li><strong>Independent MH</strong>: \\(q(x' \\mid x) = q(x')\\), independent of the current state. Works well when \\(q \\approx p\\).</li>
</ul>

<div class="env-block remark"><div class="env-title">Remark -- Optimal Acceptance Rate</div><div class="env-body">
For random walk MH targeting a \\(d\\)-dimensional Gaussian, the optimal acceptance rate is approximately 0.234 (Roberts, Gelman, and Gilks, 1997). The optimal proposal scale is \\(\\sigma \\approx 2.38/\\sqrt{d}\\).
</div></div>

<div class="viz-placeholder" data-viz="viz-mh-sampling"></div>
`,
            visualizations: [
                {
                    id: 'viz-mh-sampling',
                    title: 'Metropolis-Hastings Sampling',
                    description: 'Watch an MH chain explore a 2D target distribution (mixture of two Gaussians). Accepted steps are shown in blue, rejected proposals in red. Adjust the proposal scale \\(\\sigma\\) to see how it affects mixing.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 60, originX: 280, originY: 195 });
                        const ctx = viz.ctx, W = viz.width, H = viz.height;
                        // Target: mixture of 2 Gaussians
                        function logTarget(x, y) {
                            const g1 = -0.5 * ((x - 1.5) ** 2 + (y - 1) ** 2) / 0.25;
                            const g2 = -0.5 * ((x + 1) ** 2 + (y - 0.5) ** 2) / 0.4;
                            return Math.log(0.4 * Math.exp(g1) + 0.6 * Math.exp(g2) + 1e-30);
                        }
                        let sigma = 0.5, chain = [{ x: 0, y: 0 }], rejected = [];
                        const rng = () => { let u=0,v=0; while(!u) u=Math.random(); while(!v) v=Math.random();
                            return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };

                        function step(n) {
                            for (let i = 0; i < n; i++) {
                                const cur = chain[chain.length - 1];
                                const xp = cur.x + sigma * rng(), yp = cur.y + sigma * rng();
                                const logA = logTarget(xp, yp) - logTarget(cur.x, cur.y);
                                if (Math.log(Math.random()) < logA) {
                                    chain.push({ x: xp, y: yp });
                                } else {
                                    rejected.push({ x: xp, y: yp });
                                    chain.push({ x: cur.x, y: cur.y });
                                }
                            }
                        }

                        function draw() {
                            viz.clear();
                            // Draw target density as heatmap
                            const res = 4;
                            for (let px = 0; px < W; px += res) {
                                for (let py = 0; py < H; py += res) {
                                    const [mx, my] = viz.toMath(px, py);
                                    const v = Math.exp(logTarget(mx, my));
                                    const intensity = Math.min(255, Math.floor(v * 600));
                                    ctx.fillStyle = 'rgba(30,' + Math.floor(intensity * 0.6) + ',' + intensity + ',0.5)';
                                    ctx.fillRect(px, py, res, res);
                                }
                            }
                            // Draw rejected (faint red)
                            const maxShow = 500;
                            const rejStart = Math.max(0, rejected.length - maxShow);
                            ctx.fillStyle = '#f8514933';
                            for (let i = rejStart; i < rejected.length; i++) {
                                const [sx, sy] = viz.toScreen(rejected[i].x, rejected[i].y);
                                ctx.beginPath(); ctx.arc(sx, sy, 1.5, 0, Math.PI * 2); ctx.fill();
                            }
                            // Draw chain
                            const start = Math.max(0, chain.length - maxShow);
                            if (chain.length > 1) {
                                ctx.strokeStyle = viz.colors.blue + '44'; ctx.lineWidth = 0.5;
                                ctx.beginPath();
                                const [sx0, sy0] = viz.toScreen(chain[start].x, chain[start].y);
                                ctx.moveTo(sx0, sy0);
                                for (let i = start + 1; i < chain.length; i++) {
                                    const [sx, sy] = viz.toScreen(chain[i].x, chain[i].y);
                                    ctx.lineTo(sx, sy);
                                }
                                ctx.stroke();
                            }
                            ctx.fillStyle = viz.colors.blue;
                            for (let i = start; i < chain.length; i++) {
                                const [sx, sy] = viz.toScreen(chain[i].x, chain[i].y);
                                ctx.beginPath(); ctx.arc(sx, sy, 2, 0, Math.PI * 2); ctx.fill();
                            }
                            // Current position
                            if (chain.length > 0) {
                                const c = chain[chain.length - 1];
                                const [sx, sy] = viz.toScreen(c.x, c.y);
                                ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 2;
                                ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2); ctx.stroke();
                            }
                            // Stats
                            const nRej = rejected.length, nAcc = chain.length - 1;
                            const rate = nAcc > 0 ? (1 - nRej / (nAcc + nRej)) : 0;
                            ctx.fillStyle = viz.colors.white; ctx.font = '12px -apple-system,sans-serif'; ctx.textAlign = 'left';
                            ctx.fillText('Samples: ' + (chain.length - 1), 10, 18);
                            ctx.fillText('Accept rate: ' + (rate * 100).toFixed(1) + '%', 10, 34);
                            ctx.fillText('\u03C3 = ' + sigma.toFixed(2), 10, 50);
                        }

                        VizEngine.createSlider(controls, '\u03C3', 0.05, 3.0, sigma, 0.05, v => { sigma = v; });
                        VizEngine.createButton(controls, '+10', () => { step(10); draw(); });
                        VizEngine.createButton(controls, '+100', () => { step(100); draw(); });
                        VizEngine.createButton(controls, '+500', () => { step(500); draw(); });
                        VizEngine.createButton(controls, 'Reset', () => { chain = [{ x: 0, y: 0 }]; rejected = []; draw(); });

                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Prove that the MH acceptance ratio ensures detailed balance: \\(p(x)T(x\'\\mid x) = p(x\')T(x\\mid x\')\\).',
                    hint: 'Write out \\(T(x\'\\mid x) = q(x\'\\mid x)\\alpha(x,x\')\\) and consider two cases: \\(p(x\')q(x\\mid x\') \\geq p(x)q(x\'\\mid x)\\) and vice versa.',
                    solution: 'WLOG assume \\(p(x\')q(x\\mid x\') \\geq p(x)q(x\'\\mid x)\\). Then \\(\\alpha(x,x\') = \\frac{p(x\')q(x\\mid x\')}{p(x)q(x\'\\mid x)}\\) and \\(\\alpha(x\',x) = 1\\). So \\(p(x)T(x\'\\mid x) = p(x)q(x\'\\mid x)\\frac{p(x\')q(x\\mid x\')}{p(x)q(x\'\\mid x)} = p(x\')q(x\\mid x\') = p(x\')T(x\\mid x\')\\). By symmetry, the other case also holds. This establishes detailed balance, which implies \\(p\\) is stationary: \\(\\int p(x)T(x\'\\mid x)dx = p(x\')\\).'
                },
                {
                    question: 'Why does a very large proposal scale \\(\\sigma\\) lead to a low acceptance rate? Why does a very small \\(\\sigma\\) also perform poorly?',
                    hint: 'Think about the geometry of the target in high dimensions.',
                    solution: 'Large \\(\\sigma\\): proposals land far from the current point, likely in low-density regions, so \\(p(x\')/p(x)\\) is small and most proposals are rejected. The chain barely moves. Small \\(\\sigma\\): nearly all proposals are accepted (nearby points have similar density), but each step is tiny. The chain takes many correlated steps to traverse the distribution, making it slow to explore. The optimal tradeoff in \\(d\\) dimensions gives an acceptance rate of about 23.4% and step size \\(\\sigma \\approx 2.38/\\sqrt{d}\\).'
                },
                {
                    question: 'For the symmetric random walk proposal \\(q(x\'\\mid x) = q(x\\mid x\')\\), show that MH reduces to the Metropolis algorithm with \\(\\alpha = \\min(1, p(x\')/p(x))\\).',
                    hint: 'Symmetry means the proposal ratio cancels.',
                    solution: 'The MH acceptance ratio is \\(\\alpha = \\min(1, \\frac{p(x\')q(x\\mid x\')}{p(x)q(x\'\\mid x)})\\). When \\(q\\) is symmetric, \\(q(x\\mid x\') = q(x\'\\mid x)\\), so the ratio simplifies to \\(\\alpha = \\min(1, p(x\')/p(x))\\). This is the original Metropolis algorithm (1953). The proposal cancels because proposing \\(x\' \\to x\\) is equally likely as \\(x \\to x\'\\), so the only asymmetry comes from the target density.'
                }
            ]
        },

        // ========== SECTION 3: Gibbs Sampling ==========
        {
            id: 'ch17-sec03',
            title: 'Gibbs Sampling',
            content: `
<h2>17.3 Gibbs Sampling</h2>

<div class="env-block intuition"><div class="env-title">Intuition -- One Variable at a Time</div><div class="env-body">
Instead of proposing moves in all dimensions simultaneously (like MH), Gibbs sampling updates one variable at a time, sampling from its <em>full conditional</em> distribution holding all others fixed. This avoids the problem of tuning a proposal distribution, since we sample from the exact conditional. The price: we need to know the conditional distributions in closed form.
</div></div>

<h3>The Algorithm</h3>

<div class="env-block definition"><div class="env-title">Definition 17.3.1 -- Gibbs Sampler</div><div class="env-body">
Given a joint distribution \\(p(x_1, \\ldots, x_d)\\), the Gibbs sampler iterates:
<ol>
<li>For \\(i = 1, 2, \\ldots, d\\):</li>
<li style="margin-left:2em;">Sample \\(x_i^{(t)} \\sim p(x_i \\mid x_1^{(t)}, \\ldots, x_{i-1}^{(t)}, x_{i+1}^{(t-1)}, \\ldots, x_d^{(t-1)})\\).</li>
</ol>
Each full sweep through all variables counts as one iteration.
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 17.3.2 -- Gibbs as Special Case of MH</div><div class="env-body">
Gibbs sampling is a special case of Metropolis-Hastings where the proposal for coordinate \\(i\\) is \\(q(x_i' \\mid \\mathbf{x}) = p(x_i' \\mid \\mathbf{x}_{-i})\\). The acceptance probability is always 1.
</div></div>

<div class="env-block proof"><div class="env-title">Proof</div><div class="env-body">
The MH ratio for coordinate \\(i\\) is
\\[
\\frac{p(\\mathbf{x}')\\,q(x_i \\mid \\mathbf{x}')}{p(\\mathbf{x})\\,q(x_i' \\mid \\mathbf{x})} = \\frac{p(x_i' \\mid \\mathbf{x}_{-i})p(\\mathbf{x}_{-i}) \\cdot p(x_i \\mid \\mathbf{x}_{-i})}{p(x_i \\mid \\mathbf{x}_{-i})p(\\mathbf{x}_{-i}) \\cdot p(x_i' \\mid \\mathbf{x}_{-i})} = 1.
\\]
<div class="qed">&#8718;</div>
</div></div>

<h3>Blocked Gibbs Sampling</h3>

<p>When variables are strongly correlated, updating one at a time mixes slowly. <em>Blocked</em> (or "collapsed") Gibbs samples groups of variables jointly from their conditional, improving mixing at the cost of more complex conditionals.</p>

<h3>Application to Bayesian Models</h3>

<p>Gibbs sampling is the workhorse of Bayesian computation. If a model uses conjugate priors, each full conditional has a known parametric form, making Gibbs straightforward. Examples include Gaussian mixture models, latent Dirichlet allocation, and Bayesian linear regression.</p>

<div class="viz-placeholder" data-viz="viz-gibbs-2d"></div>
`,
            visualizations: [
                {
                    id: 'viz-gibbs-2d',
                    title: 'Gibbs Sampler on a 2D Gaussian',
                    description: 'Watch the Gibbs sampler alternate between horizontal (sample \\(x_1 \\mid x_2\\)) and vertical (sample \\(x_2 \\mid x_1\\)) moves on a correlated bivariate Gaussian. Adjust the correlation \\(\\rho\\) to see how it affects mixing.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 55, originX: 280, originY: 195 });
                        const ctx = viz.ctx, W = viz.width, H = viz.height;
                        let rho = 0.8;
                        let chain = [{ x: -2, y: -2 }], segments = [];
                        const rng = () => { let u=0,v=0; while(!u) u=Math.random(); while(!v) v=Math.random();
                            return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };

                        function gibbsStep() {
                            const cur = chain[chain.length - 1];
                            // Sample x | y: mean = rho*y, var = 1-rho^2
                            const mx = rho * cur.y, sx = Math.sqrt(1 - rho * rho);
                            const nx = mx + sx * rng();
                            segments.push({ x1: cur.x, y1: cur.y, x2: nx, y2: cur.y, dir: 'h' });
                            // Sample y | x: mean = rho*x, var = 1-rho^2
                            const my = rho * nx, sy = Math.sqrt(1 - rho * rho);
                            const ny = my + sy * rng();
                            segments.push({ x1: nx, y1: cur.y, x2: nx, y2: ny, dir: 'v' });
                            chain.push({ x: nx, y: ny });
                        }

                        function logP(x, y) {
                            return -0.5 / (1 - rho * rho) * (x * x - 2 * rho * x * y + y * y);
                        }

                        function draw() {
                            viz.clear();
                            // Draw target density
                            const res = 5;
                            for (let px = 0; px < W; px += res) {
                                for (let py = 0; py < H; py += res) {
                                    const [mx, my] = viz.toMath(px, py);
                                    const v = Math.exp(logP(mx, my));
                                    const c = Math.min(255, Math.floor(v * 400));
                                    ctx.fillStyle = 'rgba(30,' + Math.floor(c * 0.4) + ',' + c + ',0.4)';
                                    ctx.fillRect(px, py, res, res);
                                }
                            }
                            // Draw segments
                            const maxSeg = 200;
                            const sStart = Math.max(0, segments.length - maxSeg);
                            for (let i = sStart; i < segments.length; i++) {
                                const s = segments[i];
                                const [sx1, sy1] = viz.toScreen(s.x1, s.y1);
                                const [sx2, sy2] = viz.toScreen(s.x2, s.y2);
                                const alpha = 0.2 + 0.6 * (i - sStart) / (segments.length - sStart);
                                ctx.strokeStyle = s.dir === 'h' ? viz.colors.orange : viz.colors.teal;
                                ctx.globalAlpha = alpha; ctx.lineWidth = 1;
                                ctx.beginPath(); ctx.moveTo(sx1, sy1); ctx.lineTo(sx2, sy2); ctx.stroke();
                            }
                            ctx.globalAlpha = 1;
                            // Draw chain points
                            const cStart = Math.max(0, chain.length - 100);
                            for (let i = cStart; i < chain.length; i++) {
                                const [sx, sy] = viz.toScreen(chain[i].x, chain[i].y);
                                ctx.fillStyle = viz.colors.blue;
                                ctx.beginPath(); ctx.arc(sx, sy, 2, 0, Math.PI * 2); ctx.fill();
                            }
                            // Current
                            if (chain.length > 0) {
                                const c = chain[chain.length - 1];
                                const [sx, sy] = viz.toScreen(c.x, c.y);
                                ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 2;
                                ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2); ctx.stroke();
                            }
                            // Stats
                            ctx.fillStyle = viz.colors.white; ctx.font = '12px -apple-system,sans-serif'; ctx.textAlign = 'left';
                            ctx.fillText('Iterations: ' + (chain.length - 1), 10, 18);
                            ctx.fillText('\u03C1 = ' + rho.toFixed(2), 10, 34);
                            ctx.fillStyle = viz.colors.orange; ctx.fillText('\u2014 sample x|y', 10, H - 24);
                            ctx.fillStyle = viz.colors.teal; ctx.fillText('\u2014 sample y|x', 10, H - 10);
                        }

                        VizEngine.createSlider(controls, '\u03C1', -0.99, 0.99, rho, 0.05, v => {
                            rho = v; chain = [{ x: -2, y: -2 }]; segments = []; draw();
                        });
                        VizEngine.createButton(controls, '+1 step', () => { gibbsStep(); draw(); });
                        VizEngine.createButton(controls, '+10', () => { for (let i = 0; i < 10; i++) gibbsStep(); draw(); });
                        VizEngine.createButton(controls, '+100', () => { for (let i = 0; i < 100; i++) gibbsStep(); draw(); });
                        VizEngine.createButton(controls, 'Reset', () => { chain = [{ x: -2, y: -2 }]; segments = []; draw(); });

                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that the Gibbs update for coordinate \\(x_i\\) always has acceptance probability 1 in the MH framework.',
                    hint: 'Substitute the full conditional as the proposal into the MH ratio.',
                    solution: 'Proposal: \\(q(x_i\'\\mid \\mathbf{x}) = p(x_i\'\\mid \\mathbf{x}_{-i})\\). MH ratio: \\(\\frac{p(x_i\',\\mathbf{x}_{-i})\\,p(x_i\\mid \\mathbf{x}_{-i})}{p(x_i,\\mathbf{x}_{-i})\\,p(x_i\'\\mid \\mathbf{x}_{-i})} = \\frac{p(x_i\'\\mid \\mathbf{x}_{-i})p(\\mathbf{x}_{-i})\\cdot p(x_i\\mid \\mathbf{x}_{-i})}{p(x_i\\mid \\mathbf{x}_{-i})p(\\mathbf{x}_{-i})\\cdot p(x_i\'\\mid \\mathbf{x}_{-i})} = 1\\). Every proposal is accepted.'
                },
                {
                    question: 'Explain why Gibbs sampling mixes slowly when variables are highly correlated. What is the remedy?',
                    hint: 'Consider the conditional distribution of \\(x_1 \\mid x_2\\) when \\(\\rho \\to 1\\).',
                    solution: 'When \\(\\rho \\to 1\\), the conditional \\(p(x_1 \\mid x_2) = \\mathcal{N}(\\rho x_2, 1-\\rho^2)\\) has very small variance. Each Gibbs step barely moves \\(x_1\\) away from \\(\\rho x_2\\), causing the sampler to take tiny steps along the ridge of the density. The autocorrelation time scales as \\(O(1/(1-\\rho^2))\\). Remedies: (1) blocked Gibbs, sampling \\((x_1, x_2)\\) jointly, (2) reparameterize to decorrelate variables, or (3) use a different algorithm like HMC.'
                },
                {
                    question: 'Derive the full conditional distributions for a Gibbs sampler on the bivariate Gaussian \\(\\begin{pmatrix}x_1\\\\x_2\\end{pmatrix} \\sim \\mathcal{N}\\!\\left(\\mathbf{0}, \\begin{pmatrix}1&\\rho\\\\\\rho&1\\end{pmatrix}\\right)\\).',
                    hint: 'The conditional of a multivariate Gaussian is Gaussian with known formulas.',
                    solution: '\\(p(x_1 \\mid x_2) = \\mathcal{N}(\\rho x_2,\\; 1 - \\rho^2)\\) and \\(p(x_2 \\mid x_1) = \\mathcal{N}(\\rho x_1,\\; 1 - \\rho^2)\\). These follow from the standard conditional Gaussian formula: if \\(\\mathbf{x} \\sim \\mathcal{N}(\\mathbf{0}, \\Sigma)\\), then \\(x_i \\mid x_j \\sim \\mathcal{N}(\\Sigma_{ij}\\Sigma_{jj}^{-1}x_j,\\; \\Sigma_{ii} - \\Sigma_{ij}^2/\\Sigma_{jj})\\). With \\(\\Sigma_{11}=\\Sigma_{22}=1\\) and \\(\\Sigma_{12}=\\rho\\), the conditional mean is \\(\\rho x_j\\) and variance is \\(1-\\rho^2\\).'
                }
            ]
        },

        // ========== SECTION 4: Diagnostics & Advanced Methods ==========
        {
            id: 'ch17-sec04',
            title: 'Diagnostics & Advanced Methods',
            content: `
<h2>17.4 Diagnostics & Advanced Methods</h2>

<div class="env-block intuition"><div class="env-title">Intuition -- How Do We Know the Chain Has Converged?</div><div class="env-body">
MCMC guarantees convergence <em>asymptotically</em>, but in practice we run a finite chain. How do we know when to stop? And how do we know the samples are representative? Diagnostics tools, such as trace plots, autocorrelation, and the \\(\\hat{R}\\) statistic, help us assess whether the chain has "mixed" well enough to trust the results.
</div></div>

<h3>Burn-in and Mixing</h3>

<p>The initial samples from an MCMC chain are typically discarded as <em>burn-in</em>, since the chain has not yet reached its stationary distribution. After burn-in, we want the chain to <em>mix</em>: it should explore the full support of the target, visiting all modes and not getting stuck.</p>

<div class="env-block definition"><div class="env-title">Definition 17.4.1 -- Effective Sample Size</div><div class="env-body">
The <em>effective sample size</em> (ESS) accounts for autocorrelation:
\\[
\\mathrm{ESS} = \\frac{N}{1 + 2\\sum_{k=1}^{\\infty} \\rho_k},
\\]
where \\(\\rho_k\\) is the lag-\\(k\\) autocorrelation of the chain. A well-mixing chain has low autocorrelation and ESS close to \\(N\\).
</div></div>

<h3>The \\(\\hat{R}\\) Statistic</h3>

<div class="env-block definition"><div class="env-title">Definition 17.4.2 -- Gelman-Rubin \\(\\hat{R}\\)</div><div class="env-body">
Run \\(M\\) independent chains, each of length \\(N\\). Let \\(B\\) be the between-chain variance and \\(W\\) the within-chain variance. Then
\\[
\\hat{R} = \\sqrt{\\frac{(N-1)/N \\cdot W + B/N}{W}}.
\\]
Values near 1 indicate convergence; \\(\\hat{R} \\gt 1.1\\) suggests the chains have not yet converged.
</div></div>

<h3>Hamiltonian Monte Carlo</h3>

<p>Standard MH with random walk proposals scales poorly with dimension. <em>Hamiltonian Monte Carlo</em> (HMC) uses gradient information to make proposals that travel far in parameter space while maintaining high acceptance rates.</p>

<div class="env-block definition"><div class="env-title">Definition 17.4.3 -- HMC Algorithm</div><div class="env-body">
Introduce auxiliary momentum \\(\\mathbf{p} \\sim \\mathcal{N}(\\mathbf{0}, M)\\) and define the Hamiltonian \\(H(\\mathbf{x}, \\mathbf{p}) = -\\log p(\\mathbf{x}) + \\frac{1}{2}\\mathbf{p}^T M^{-1} \\mathbf{p}\\). Each iteration:
<ol>
<li>Resample momentum: \\(\\mathbf{p} \\sim \\mathcal{N}(\\mathbf{0}, M)\\).</li>
<li>Simulate Hamiltonian dynamics for \\(L\\) leapfrog steps with step size \\(\\epsilon\\).</li>
<li>Accept/reject with MH ratio \\(\\alpha = \\min(1, \\exp(-H(\\mathbf{x}', \\mathbf{p}') + H(\\mathbf{x}, \\mathbf{p})))\\).</li>
</ol>
</div></div>

<div class="env-block remark"><div class="env-title">Remark -- Why HMC Works Well</div><div class="env-body">
Hamiltonian dynamics preserves the total energy \\(H\\), so the acceptance probability would be 1 with exact integration. The leapfrog integrator introduces small errors, keeping acceptance high. Unlike random walk MH, HMC proposals can traverse the entire distribution in a single step, leading to nearly independent samples. The cost is requiring \\(\\nabla \\log p(\\mathbf{x})\\), which is available for differentiable models (e.g., via automatic differentiation in Stan or PyMC).
</div></div>

<div class="viz-placeholder" data-viz="viz-diagnostics"></div>
`,
            visualizations: [
                {
                    id: 'viz-diagnostics',
                    title: 'Trace Plots: Good vs Bad Mixing',
                    description: 'Compare trace plots of a well-mixing chain (small autocorrelation) versus a poorly-mixing chain (high autocorrelation / stuck in a mode). The autocorrelation function (ACF) is shown on the right.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 1, originX: 0, originY: 0 });
                        const ctx = viz.ctx, W = viz.width, H = viz.height;
                        const rng = () => { let u=0,v=0; while(!u) u=Math.random(); while(!v) v=Math.random();
                            return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };

                        function genChain(rho, N, stuck) {
                            const chain = [rng()];
                            for (let i = 1; i < N; i++) {
                                if (stuck && Math.random() < 0.02) {
                                    chain.push(chain[i-1] + (Math.random() < 0.5 ? 3 : -3));
                                } else {
                                    chain.push(rho * chain[i - 1] + Math.sqrt(1 - rho * rho) * rng());
                                }
                            }
                            return chain;
                        }

                        function acf(chain, maxLag) {
                            const n = chain.length, mu = chain.reduce((a,b)=>a+b,0)/n;
                            const v = chain.reduce((a,x)=>a+(x-mu)**2,0)/n;
                            const result = [];
                            for (let k = 0; k <= maxLag; k++) {
                                let s = 0;
                                for (let i = 0; i < n - k; i++) s += (chain[i] - mu) * (chain[i + k] - mu);
                                result.push(s / (n * v));
                            }
                            return result;
                        }

                        let N = 500;
                        let goodChain = genChain(0.1, N, false);
                        let badChain = genChain(0.97, N, true);

                        function drawTrace(chain, ox, oy, w, h, title, color) {
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 11px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.fillText(title, ox + w / 2, oy - 6);
                            ctx.strokeStyle = '#30363d'; ctx.lineWidth = 0.5;
                            ctx.strokeRect(ox, oy, w, h);
                            // Find range
                            let mn = Infinity, mx = -Infinity;
                            for (const v of chain) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
                            const pad = (mx - mn) * 0.1 || 1; mn -= pad; mx += pad;
                            // Draw trace
                            ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1;
                            for (let i = 0; i < chain.length; i++) {
                                const px = ox + (i / (chain.length - 1)) * w;
                                const py = oy + h - ((chain[i] - mn) / (mx - mn)) * h;
                                i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
                            }
                            ctx.stroke();
                        }

                        function drawACF(chain, ox, oy, w, h, title, color) {
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 11px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.fillText(title, ox + w / 2, oy - 6);
                            ctx.strokeStyle = '#30363d'; ctx.lineWidth = 0.5;
                            ctx.strokeRect(ox, oy, w, h);
                            const maxLag = 40, ac = acf(chain, maxLag);
                            const barW = w / (maxLag + 1) * 0.8;
                            // CI line
                            const ci = 1.96 / Math.sqrt(chain.length);
                            const ciY = oy + h - ci * h;
                            ctx.setLineDash([3,3]); ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(ox, ciY); ctx.lineTo(ox + w, ciY); ctx.stroke();
                            ctx.setLineDash([]);
                            // Bars
                            for (let k = 0; k <= maxLag; k++) {
                                const bx = ox + (k / (maxLag + 1)) * w;
                                const bh = Math.abs(ac[k]) * h;
                                const by = ac[k] >= 0 ? oy + h - bh : oy + h;
                                ctx.fillStyle = color + '99';
                                ctx.fillRect(bx, by, barW, bh);
                            }
                        }

                        function draw() {
                            viz.clear();
                            const tw = W * 0.52, th = (H - 80) / 2 - 20;
                            const aw = W - tw - 30, ah = th;
                            drawTrace(goodChain, 10, 25, tw, th, 'Good Mixing (low \u03C1)', viz.colors.teal);
                            drawTrace(badChain, 10, th + 60, tw, th, 'Poor Mixing (high \u03C1, mode-hopping)', viz.colors.red);
                            drawACF(goodChain, tw + 25, 25, aw, ah, 'ACF (Good)', viz.colors.teal);
                            drawACF(badChain, tw + 25, th + 60, aw, ah, 'ACF (Poor)', viz.colors.red);

                            // ESS
                            const acGood = acf(goodChain, 40), acBad = acf(badChain, 40);
                            let sumG = 0, sumB = 0;
                            for (let k = 1; k < acGood.length; k++) { if (acGood[k] < 0) break; sumG += acGood[k]; }
                            for (let k = 1; k < acBad.length; k++) { if (acBad[k] < 0.05) break; sumB += acBad[k]; }
                            const essG = Math.round(N / (1 + 2 * sumG)), essB = Math.round(N / (1 + 2 * sumB));
                            ctx.fillStyle = viz.colors.white; ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            ctx.fillText('ESS (Good): ' + essG + '/' + N + '    ESS (Poor): ' + essB + '/' + N, W / 2, H - 8);
                        }

                        VizEngine.createButton(controls, 'Regenerate', () => {
                            goodChain = genChain(0.1, N, false);
                            badChain = genChain(0.97, N, true);
                            draw();
                        });
                        VizEngine.createSlider(controls, 'N', 100, 2000, N, 100, v => {
                            N = Math.round(v);
                            goodChain = genChain(0.1, N, false);
                            badChain = genChain(0.97, N, true);
                            draw();
                        });

                        draw();
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that the effective sample size of an AR(1) process \\(x_t = \\rho x_{t-1} + \\epsilon_t\\) is approximately \\(N(1 - \\rho)/(1 + \\rho)\\) for large \\(N\\).',
                    hint: 'For AR(1), \\(\\rho_k = \\rho^k\\). Sum the geometric series.',
                    solution: 'For AR(1), \\(\\rho_k = \\rho^k\\). Then \\(1 + 2\\sum_{k=1}^\\infty \\rho^k = 1 + 2\\rho/(1-\\rho) = (1+\\rho)/(1-\\rho)\\). So \\(\\mathrm{ESS} = N(1-\\rho)/(1+\\rho)\\). For \\(\\rho = 0.95\\), \\(\\mathrm{ESS} \\approx N/39\\): we need 39 MCMC samples for each effective independent sample. For \\(\\rho = 0.1\\), \\(\\mathrm{ESS} \\approx 0.82N\\), nearly independent.'
                },
                {
                    question: 'Explain the leapfrog integrator used in HMC and why it preserves the symplectic structure.',
                    hint: 'The leapfrog update is: half-step momentum, full-step position, half-step momentum.',
                    solution: 'The leapfrog integrator performs: (1) \\(\\mathbf{p}_{t+\\epsilon/2} = \\mathbf{p}_t - (\\epsilon/2)\\nabla U(\\mathbf{x}_t)\\), (2) \\(\\mathbf{x}_{t+\\epsilon} = \\mathbf{x}_t + \\epsilon M^{-1}\\mathbf{p}_{t+\\epsilon/2}\\), (3) \\(\\mathbf{p}_{t+\\epsilon} = \\mathbf{p}_{t+\\epsilon/2} - (\\epsilon/2)\\nabla U(\\mathbf{x}_{t+\\epsilon})\\). It is a composition of three shear maps, each with unit Jacobian determinant. Therefore the overall map is volume-preserving (symplectic) and time-reversible. Volume preservation means no Jacobian correction in the MH step; time-reversibility ensures detailed balance. The energy error is \\(O(\\epsilon^2)\\) per step, giving high acceptance for small \\(\\epsilon\\).'
                },
                {
                    question: 'Describe three practical diagnostics for assessing MCMC convergence and explain what each reveals.',
                    hint: 'Think about trace plots, autocorrelation, and multi-chain statistics.',
                    solution: '(1) <strong>Trace plots</strong>: plot \\(x^{(t)}\\) vs \\(t\\). A well-mixing chain looks like white noise around the stationary mean; a poorly-mixing chain shows trends, sticky periods, or slow drifts. (2) <strong>Autocorrelation function (ACF)</strong>: plot \\(\\rho_k\\) vs lag \\(k\\). A well-mixing chain has ACF that drops to zero quickly; high autocorrelation indicates the chain is taking correlated steps and ESS is low. (3) <strong>\\(\\hat{R}\\) (Gelman-Rubin)</strong>: compare between-chain and within-chain variance across multiple independent chains. \\(\\hat{R} \\approx 1\\) means chains have converged to the same distribution; \\(\\hat{R} > 1.1\\) suggests insufficient convergence, possibly due to multimodality or slow mixing.'
                }
            ]
        }
    ]
});
