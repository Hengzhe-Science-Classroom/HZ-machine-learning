window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
  id: 'ch06',
  number: 6,
  title: 'Generative Classifiers',
  subtitle: 'Modeling How Data Is Generated to Make Predictions',
  sections: [

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 1: Generative vs Discriminative
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch06-sec01',
      title: 'Generative vs Discriminative',
      content: `
<h2>Two Philosophies of Classification</h2>

<p>In Chapters 4 and 5 we built classifiers that directly model the posterior \\(P(y \\mid \\mathbf{x})\\). Logistic regression, for instance, parameterizes this posterior with a sigmoid applied to a linear function of \\(\\mathbf{x}\\). Such models are called <strong>discriminative</strong> because they learn the boundary that <em>discriminates</em> between classes without ever asking how the data was generated.</p>

<p>There is a fundamentally different strategy: model the <em>joint distribution</em> \\(P(\\mathbf{x}, y)\\) by specifying a class prior \\(P(y)\\) and a class-conditional density \\(P(\\mathbf{x} \\mid y)\\). From these, Bayes' theorem recovers the posterior. Such models are called <strong>generative</strong> because \\(P(\\mathbf{x} \\mid y)\\) describes how data is <em>generated</em> within each class.</p>

<div class="env-block definition"><div class="env-title">Definition 6.1 (Generative Classifier)</div><div class="env-body">
<p>A <strong>generative classifier</strong> specifies:</p>
<ol>
  <li>A <strong>prior</strong> \\(P(y = k) = \\pi_k\\) for each class \\(k \\in \\{1, \\dots, K\\}\\), with \\(\\sum_k \\pi_k = 1\\).</li>
  <li>A <strong>class-conditional density</strong> (or likelihood) \\(P(\\mathbf{x} \\mid y = k)\\) for each class.</li>
</ol>
<p>Classification uses Bayes' theorem:</p>
\\[P(y = k \\mid \\mathbf{x}) = \\frac{P(\\mathbf{x} \\mid y = k)\\,P(y = k)}{\\sum_{j=1}^{K} P(\\mathbf{x} \\mid y = j)\\,P(y = j)} = \\frac{\\pi_k\\,P(\\mathbf{x} \\mid y = k)}{\\sum_{j} \\pi_j\\,P(\\mathbf{x} \\mid y = j)}.\\]
</div></div>

<div class="env-block definition"><div class="env-title">Definition 6.2 (Discriminative Classifier)</div><div class="env-body">
<p>A <strong>discriminative classifier</strong> directly models \\(P(y \\mid \\mathbf{x})\\) (or a decision function \\(f(\\mathbf{x})\\)) without specifying a model for \\(P(\\mathbf{x} \\mid y)\\) or \\(P(\\mathbf{x})\\).</p>
</div></div>

<div class="env-block intuition"><div class="env-title">The Analogy</div><div class="env-body"><p>Suppose you want to distinguish cats from dogs. A discriminative approach looks for features (pointy ears, wet nose shape) that tell the classes apart. A generative approach builds a complete mental model of what cats look like and what dogs look like, then classifies a new animal by asking: "Which model would more likely have generated this animal?" The generative approach is more ambitious but also more informative.</p></div></div>

<h3>The Decision Boundary</h3>

<p>For binary classification (\\(K = 2\\)), the optimal decision assigns class 1 when</p>
\\[P(y = 1 \\mid \\mathbf{x}) > P(y = 0 \\mid \\mathbf{x}),\\]
<p>which by Bayes' theorem is equivalent to</p>
\\[\\pi_1\\,P(\\mathbf{x} \\mid y = 1) > \\pi_0\\,P(\\mathbf{x} \\mid y = 0).\\]
<p>Taking logarithms, the <strong>decision boundary</strong> is the set of \\(\\mathbf{x}\\) where</p>
\\[\\log\\frac{P(\\mathbf{x} \\mid y = 1)}{P(\\mathbf{x} \\mid y = 0)} + \\log\\frac{\\pi_1}{\\pi_0} = 0.\\]
<p>The first term is the <strong>log-likelihood ratio</strong>; the second is the <strong>log-prior ratio</strong>.</p>

<div class="env-block remark"><div class="env-title">Remark</div><div class="env-body"><p>When priors are equal (\\(\\pi_0 = \\pi_1 = 0.5\\)), the log-prior ratio vanishes and the boundary depends only on the likelihoods. Unequal priors shift the boundary toward the less probable class.</p></div></div>

<div class="viz-placeholder" data-viz="viz-gen-vs-disc"></div>

<h3>Trade-offs</h3>

<table>
<thead><tr><th>Aspect</th><th>Generative</th><th>Discriminative</th></tr></thead>
<tbody>
<tr><td>Models</td><td>\\(P(\\mathbf{x} \\mid y)\\), \\(P(y)\\)</td><td>\\(P(y \\mid \\mathbf{x})\\) directly</td></tr>
<tr><td>Parameters</td><td>More (full density model)</td><td>Fewer (boundary only)</td></tr>
<tr><td>Missing data</td><td>Handles naturally</td><td>Requires special treatment</td></tr>
<tr><td>Outlier detection</td><td>Can detect via low \\(P(\\mathbf{x})\\)</td><td>Cannot easily</td></tr>
<tr><td>Asymptotic accuracy</td><td>Can be lower if model is wrong</td><td>Often higher</td></tr>
</tbody>
</table>
`,
      visualizations: [
        {
          id: 'viz-gen-vs-disc',
          title: 'Generative Classification via Bayes\' Theorem',
          description: 'Two Gaussian class-conditional densities with adjustable prior. The shaded curves show P(x|y=k), and the dashed line shows the Bayes-optimal decision boundary. Adjust the prior to see how the boundary shifts.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 60, originX: 80, originY: 280 });
            let prior1 = 0.5;
            let mu0 = -1.5, mu1 = 1.5, sigma = 1.0;

            VizEngine.createSlider(controls, 'P(y=1)', 0.1, 0.9, prior1, 0.05, v => { prior1 = v; draw(); });
            VizEngine.createSlider(controls, 'Separation', 0.5, 4, 3.0, 0.1, v => { mu0 = -v/2; mu1 = v/2; draw(); });
            VizEngine.createSlider(controls, '\u03C3', 0.3, 2.0, sigma, 0.1, v => { sigma = v; draw(); });

            function gaussian(x, mu, s) {
              return Math.exp(-0.5 * ((x - mu) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
            }

            function draw() {
              viz.clear();
              const ctx = viz.ctx;
              const prior0 = 1 - prior1;

              // Draw axes
              ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
              ctx.beginPath(); ctx.moveTo(40, viz.originY); ctx.lineTo(viz.width - 10, viz.originY); ctx.stroke();

              // Plot class-conditional densities
              const xMin = -5, xMax = 5, steps = 300;
              const dx = (xMax - xMin) / steps;
              const scaleY = 180;

              // Shade under curves
              for (let cls = 0; cls < 2; cls++) {
                const mu = cls === 0 ? mu0 : mu1;
                const color = cls === 0 ? viz.colors.red : viz.colors.blue;
                const pi = cls === 0 ? prior0 : prior1;
                ctx.fillStyle = color + '20';
                ctx.beginPath();
                const startSx = 80 + ((xMin - (-5)) / 10) * (viz.width - 90);
                ctx.moveTo(startSx, viz.originY);
                for (let i = 0; i <= steps; i++) {
                  const x = xMin + i * dx;
                  const y = gaussian(x, mu, sigma);
                  const sx = 80 + ((x - xMin) / (xMax - xMin)) * (viz.width - 90);
                  const sy = viz.originY - y * scaleY;
                  ctx.lineTo(sx, sy);
                }
                ctx.lineTo(80 + (viz.width - 90), viz.originY);
                ctx.closePath();
                ctx.fill();

                // Draw curve
                ctx.strokeStyle = color; ctx.lineWidth = 2.5;
                ctx.beginPath();
                for (let i = 0; i <= steps; i++) {
                  const x = xMin + i * dx;
                  const y = gaussian(x, mu, sigma);
                  const sx = 80 + ((x - xMin) / (xMax - xMin)) * (viz.width - 90);
                  const sy = viz.originY - y * scaleY;
                  i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                }
                ctx.stroke();

                // Label
                const labelX = 80 + ((mu - xMin) / (xMax - xMin)) * (viz.width - 90);
                ctx.fillStyle = color; ctx.font = 'bold 12px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('P(x|y=' + cls + ')', labelX, viz.originY - gaussian(mu, mu, sigma) * scaleY - 12);
              }

              // Find decision boundary: where pi1*p(x|1) = pi0*p(x|0)
              // For equal variance Gaussians: boundary = (mu0+mu1)/2 + sigma^2/(mu1-mu0)*ln(pi0/pi1)
              const muDiff = mu1 - mu0;
              let boundary = null;
              if (Math.abs(muDiff) > 0.001) {
                boundary = (mu0 + mu1) / 2 + (sigma * sigma / muDiff) * Math.log(prior0 / prior1);
              }

              if (boundary !== null && boundary > xMin && boundary < xMax) {
                const bx = 80 + ((boundary - xMin) / (xMax - xMin)) * (viz.width - 90);
                ctx.strokeStyle = viz.colors.green; ctx.lineWidth = 2.5;
                ctx.setLineDash([8, 5]);
                ctx.beginPath(); ctx.moveTo(bx, 30); ctx.lineTo(bx, viz.originY); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = viz.colors.green; ctx.font = 'bold 12px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('Decision boundary', bx, 22);
              }

              // Draw posterior below
              const postY0 = viz.originY + 30;
              ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'left';
              ctx.fillText('Posterior P(y|x):', 80, postY0);

              const postScaleY = 50;
              for (let cls = 0; cls < 2; cls++) {
                const color = cls === 0 ? viz.colors.red : viz.colors.blue;
                ctx.strokeStyle = color; ctx.lineWidth = 1.5;
                ctx.beginPath();
                for (let i = 0; i <= steps; i++) {
                  const x = xMin + i * dx;
                  const p0 = prior0 * gaussian(x, mu0, sigma);
                  const p1 = prior1 * gaussian(x, mu1, sigma);
                  const total = p0 + p1;
                  const post = cls === 0 ? p0 / (total + 1e-15) : p1 / (total + 1e-15);
                  const sx = 80 + ((x - xMin) / (xMax - xMin)) * (viz.width - 90);
                  const sy = postY0 + 10 + (1 - post) * postScaleY;
                  i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                }
                ctx.stroke();
              }

              // Posterior labels
              ctx.fillStyle = viz.colors.text; ctx.font = '10px -apple-system,sans-serif';
              ctx.textAlign = 'right';
              ctx.fillText('1.0', 76, postY0 + 12);
              ctx.fillText('0.0', 76, postY0 + 10 + postScaleY);

              // Prior info
              ctx.fillStyle = viz.colors.text; ctx.font = '12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('\u03C0\u2080=' + prior0.toFixed(2) + '  \u03C0\u2081=' + prior1.toFixed(2), viz.width / 2, viz.height - 10);
            }

            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Suppose \\(P(\\mathbf{x} \\mid y=1)\\) is Gaussian with mean 2 and variance 1, \\(P(\\mathbf{x} \\mid y=0)\\) is Gaussian with mean 0 and variance 1, and \\(\\pi_0 = \\pi_1 = 0.5\\). Compute \\(P(y=1 \\mid x=1)\\).',
          hint: 'Compute both likelihoods at \\(x=1\\), multiply each by the prior, then normalize.',
          solution: 'We have \\(P(x=1 \\mid y=1) = \\frac{1}{\\sqrt{2\\pi}}e^{-1/2} \\approx 0.2420\\) and \\(P(x=1 \\mid y=0) = \\frac{1}{\\sqrt{2\\pi}}e^{-1/2} \\approx 0.2420\\). Since both likelihoods are equal and priors are equal, \\(P(y=1 \\mid x=1) = 0.5\\). The point \\(x=1\\) lies exactly on the decision boundary (the midpoint of the two means).'
        },
        {
          question: 'For two classes with equal-variance Gaussians (\\(\\sigma_0 = \\sigma_1 = \\sigma\\)) and means \\(\\mu_0, \\mu_1\\), derive the decision boundary as a function of the prior ratio \\(\\pi_1/\\pi_0\\).',
          hint: 'Set \\(\\pi_1 P(x \\mid y=1) = \\pi_0 P(x \\mid y=0)\\) and take logarithms.',
          solution: 'Setting the log-posterior ratio to zero: \\(-\\frac{(x-\\mu_1)^2}{2\\sigma^2} + \\log\\pi_1 = -\\frac{(x-\\mu_0)^2}{2\\sigma^2} + \\log\\pi_0\\). Expanding and solving: \\(x = \\frac{\\mu_0 + \\mu_1}{2} + \\frac{\\sigma^2}{\\mu_1 - \\mu_0}\\log\\frac{\\pi_0}{\\pi_1}\\). When priors are equal, the boundary is at the midpoint \\((\\mu_0+\\mu_1)/2\\).'
        },
        {
          question: 'Give an example of a practical advantage of generative models over discriminative models in a real-world classification setting.',
          hint: 'Think about scenarios with missing features, class imbalance changes, or outlier detection.',
          solution: 'Generative models can detect outliers or out-of-distribution inputs by evaluating \\(P(\\mathbf{x}) = \\sum_k \\pi_k P(\\mathbf{x} \\mid y=k)\\). If \\(P(\\mathbf{x})\\) is very small, the input is likely outside the training distribution. A discriminative model that only models \\(P(y \\mid \\mathbf{x})\\) will still produce a confident (but meaningless) class prediction for any input, no matter how anomalous.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 2: Gaussian Discriminant Analysis
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch06-sec02',
      title: 'Gaussian Discriminant Analysis',
      content: `
<h2>Gaussian Discriminant Analysis</h2>

<p>The most natural choice for continuous features is to model each class-conditional density as a multivariate Gaussian. This yields <strong>Gaussian Discriminant Analysis</strong> (GDA), a family of classifiers that includes two important special cases depending on whether the classes share a covariance matrix.</p>

<div class="env-block definition"><div class="env-title">Definition 6.3 (GDA Model)</div><div class="env-body">
<p>For \\(K\\) classes, GDA assumes</p>
\\[P(\\mathbf{x} \\mid y = k) = \\mathcal{N}(\\mathbf{x}; \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k) = \\frac{1}{(2\\pi)^{d/2}|\\boldsymbol{\\Sigma}_k|^{1/2}} \\exp\\!\\left(-\\frac{1}{2}(\\mathbf{x}-\\boldsymbol{\\mu}_k)^\\top \\boldsymbol{\\Sigma}_k^{-1}(\\mathbf{x}-\\boldsymbol{\\mu}_k)\\right).\\]
</div></div>

<h3>Linear Discriminant Analysis (LDA)</h3>

<p>When all classes share a common covariance \\(\\boldsymbol{\\Sigma}_k = \\boldsymbol{\\Sigma}\\) for all \\(k\\), the quadratic terms in \\(\\mathbf{x}\\) cancel in the log-likelihood ratio, yielding a <em>linear</em> decision boundary.</p>

<div class="env-block theorem"><div class="env-title">Theorem 6.1 (LDA Decision Boundary)</div><div class="env-body">
<p>Under shared covariance \\(\\boldsymbol{\\Sigma}\\), the log-posterior for class \\(k\\) is a linear function of \\(\\mathbf{x}\\):</p>
\\[\\log P(y=k \\mid \\mathbf{x}) = \\boldsymbol{\\mu}_k^\\top \\boldsymbol{\\Sigma}^{-1}\\mathbf{x} - \\tfrac{1}{2}\\boldsymbol{\\mu}_k^\\top \\boldsymbol{\\Sigma}^{-1}\\boldsymbol{\\mu}_k + \\log \\pi_k + \\text{const}.\\]
<p>The decision boundary between classes \\(j\\) and \\(k\\) is the hyperplane where these linear functions are equal.</p>
</div></div>

<div class="env-block proof"><div class="env-title">Proof sketch</div><div class="env-body">
<p>The log-posterior for class \\(k\\) is \\(\\log\\pi_k - \\frac{1}{2}(\\mathbf{x} - \\boldsymbol{\\mu}_k)^\\top\\boldsymbol{\\Sigma}^{-1}(\\mathbf{x} - \\boldsymbol{\\mu}_k) + C\\). Expanding the quadratic form: \\(-\\frac{1}{2}\\mathbf{x}^\\top\\boldsymbol{\\Sigma}^{-1}\\mathbf{x} + \\boldsymbol{\\mu}_k^\\top\\boldsymbol{\\Sigma}^{-1}\\mathbf{x} - \\frac{1}{2}\\boldsymbol{\\mu}_k^\\top\\boldsymbol{\\Sigma}^{-1}\\boldsymbol{\\mu}_k + C\\). The term \\(-\\frac{1}{2}\\mathbf{x}^\\top\\boldsymbol{\\Sigma}^{-1}\\mathbf{x}\\) is the same for all classes, so it cancels when comparing posteriors.</p>
<div class="qed">&#8718;</div>
</div></div>

<h3>Maximum Likelihood Estimation for LDA</h3>

<p>Given training data \\(\\{(\\mathbf{x}_i, y_i)\\}_{i=1}^n\\), the MLE estimates are:</p>
\\[\\hat{\\pi}_k = \\frac{n_k}{n}, \\quad \\hat{\\boldsymbol{\\mu}}_k = \\frac{1}{n_k}\\sum_{i: y_i=k}\\mathbf{x}_i, \\quad \\hat{\\boldsymbol{\\Sigma}} = \\frac{1}{n}\\sum_{k=1}^{K}\\sum_{i: y_i=k}(\\mathbf{x}_i - \\hat{\\boldsymbol{\\mu}}_k)(\\mathbf{x}_i - \\hat{\\boldsymbol{\\mu}}_k)^\\top,\\]
<p>where \\(n_k = |\\{i : y_i = k\\}|\\) is the number of training points in class \\(k\\).</p>

<h3>Quadratic Discriminant Analysis (QDA)</h3>

<p>When each class has its own covariance \\(\\boldsymbol{\\Sigma}_k\\), the \\(\\mathbf{x}^\\top\\boldsymbol{\\Sigma}_k^{-1}\\mathbf{x}\\) terms no longer cancel, producing <em>quadratic</em> decision boundaries (ellipses, parabolas, or hyperbolas in 2D).</p>

<div class="env-block definition"><div class="env-title">Definition 6.4 (QDA)</div><div class="env-body">
<p>In QDA, the log-posterior for class \\(k\\) is</p>
\\[\\delta_k(\\mathbf{x}) = -\\tfrac{1}{2}\\log|\\boldsymbol{\\Sigma}_k| - \\tfrac{1}{2}(\\mathbf{x} - \\boldsymbol{\\mu}_k)^\\top \\boldsymbol{\\Sigma}_k^{-1}(\\mathbf{x} - \\boldsymbol{\\mu}_k) + \\log\\pi_k.\\]
<p>The decision boundary between classes \\(j\\) and \\(k\\) is the set \\(\\{\\mathbf{x} : \\delta_j(\\mathbf{x}) = \\delta_k(\\mathbf{x})\\}\\), which is generally a quadratic surface.</p>
</div></div>

<div class="env-block remark"><div class="env-title">LDA vs QDA: Bias-Variance</div><div class="env-body"><p>LDA has fewer parameters (one shared covariance) and thus lower variance but higher bias. QDA has more flexibility but requires estimating \\(K\\) separate covariance matrices, each with \\(O(d^2)\\) parameters. In high dimensions with limited data, LDA often outperforms QDA because QDA overfits.</p></div></div>

<div class="viz-placeholder" data-viz="viz-lda-qda"></div>
`,
      visualizations: [
        {
          id: 'viz-lda-qda',
          title: 'LDA vs QDA Decision Boundaries',
          description: 'Toggle between LDA (shared covariance, linear boundary) and QDA (separate covariances, quadratic boundary). The ellipses show 1-sigma contours of each class-conditional Gaussian.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 50, originX: 280, originY: 200 });
            let mode = 'LDA';

            const btnRow = document.createElement('div');
            btnRow.style.cssText = 'display:flex;gap:6px;';
            let ldaBtn, qdaBtn;
            ldaBtn = VizEngine.createButton(btnRow, 'LDA', () => { mode = 'LDA'; ldaBtn.style.background='#2a4a6a'; qdaBtn.style.background='#1a1a40'; draw(); });
            qdaBtn = VizEngine.createButton(btnRow, 'QDA', () => { mode = 'QDA'; qdaBtn.style.background='#2a4a6a'; ldaBtn.style.background='#1a1a40'; draw(); });
            ldaBtn.style.background = '#2a4a6a';
            controls.appendChild(btnRow);

            // Class parameters
            const mu0 = [-1.5, -0.5], mu1 = [1.5, 0.5];
            // LDA: shared covariance
            const sigmaShared = [[1.0, 0.3], [0.3, 0.8]];
            // QDA: separate covariances
            const sigma0 = [[0.6, -0.2], [-0.2, 1.2]];
            const sigma1 = [[1.2, 0.5], [0.5, 0.5]];

            // Generate sample data
            function sampleGaussian2D(mu, sigma, n) {
              const pts = [];
              const L00 = Math.sqrt(sigma[0][0]);
              const L10 = sigma[1][0] / L00;
              const L11 = Math.sqrt(sigma[1][1] - L10 * L10);
              for (let i = 0; i < n; i++) {
                let u = 0, v = 0;
                for (let j = 0; j < 6; j++) { u += Math.random() - 0.5; v += Math.random() - 0.5; }
                u *= Math.sqrt(2); v *= Math.sqrt(2);
                pts.push([mu[0] + L00 * u, mu[1] + L10 * u + L11 * v]);
              }
              return pts;
            }

            // Fixed seed via precomputed data
            const rng = (() => { let s = 42; return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; }; })();
            function sampleFixed(mu, sigma, n) {
              const pts = [];
              const L00 = Math.sqrt(sigma[0][0]);
              const L10 = sigma[1][0] / L00;
              const L11 = Math.sqrt(Math.max(0.001, sigma[1][1] - L10 * L10));
              for (let i = 0; i < n; i++) {
                let u = 0, v = 0;
                for (let j = 0; j < 12; j++) { u += rng() - 0.5; v += rng() - 0.5; }
                u *= Math.sqrt(1); v *= Math.sqrt(1);
                pts.push([mu[0] + L00 * u, mu[1] + L10 * u + L11 * v]);
              }
              return pts;
            }
            const data0_lda = sampleFixed(mu0, sigmaShared, 30);
            const data1_lda = sampleFixed(mu1, sigmaShared, 30);
            const rng2 = (() => { let s = 99; return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; }; })();
            function sampleFixed2(mu, sigma, n) {
              const pts = [];
              const L00 = Math.sqrt(sigma[0][0]);
              const L10 = sigma[1][0] / L00;
              const L11 = Math.sqrt(Math.max(0.001, sigma[1][1] - L10 * L10));
              for (let i = 0; i < n; i++) {
                let u = 0, v = 0;
                for (let j = 0; j < 12; j++) { u += rng2() - 0.5; v += rng2() - 0.5; }
                u *= Math.sqrt(1); v *= Math.sqrt(1);
                pts.push([mu[0] + L00 * u, mu[1] + L10 * u + L11 * v]);
              }
              return pts;
            }
            const data0_qda = sampleFixed2(mu0, sigma0, 30);
            const data1_qda = sampleFixed2(mu1, sigma1, 30);

            function inv2(M) {
              const det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
              return [[M[1][1]/det, -M[0][1]/det], [-M[1][0]/det, M[0][0]/det]];
            }

            function drawEllipse(ctx, viz, mu, sigma, color) {
              // Eigendecomposition for ellipse
              const a = sigma[0][0], b = sigma[0][1], d = sigma[1][1];
              const tr = a + d, det = a * d - b * b;
              const disc = Math.sqrt(Math.max(0, tr * tr - 4 * det));
              const l1 = (tr + disc) / 2, l2 = (tr - disc) / 2;
              let angle = 0;
              if (Math.abs(b) > 1e-10) angle = Math.atan2(l1 - a, b);
              else if (a < d) angle = Math.PI / 2;
              const [sx, sy] = viz.toScreen(mu[0], mu[1]);
              ctx.save(); ctx.translate(sx, sy); ctx.rotate(-angle);
              ctx.strokeStyle = color; ctx.lineWidth = 2;
              ctx.setLineDash([5, 3]);
              ctx.beginPath();
              ctx.ellipse(0, 0, Math.sqrt(l1) * viz.scale, Math.sqrt(l2) * viz.scale, 0, 0, Math.PI * 2);
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.restore();
            }

            function draw() {
              viz.clear();
              viz.drawGrid(1);
              viz.drawAxes();
              const ctx = viz.ctx;

              const isLDA = mode === 'LDA';
              const s0 = isLDA ? sigmaShared : sigma0;
              const s1 = isLDA ? sigmaShared : sigma1;
              const d0 = isLDA ? data0_lda : data0_qda;
              const d1 = isLDA ? data1_lda : data1_qda;
              const inv0 = inv2(s0), inv1 = inv2(s1);
              const logdet0 = Math.log(s0[0][0]*s0[1][1] - s0[0][1]*s0[1][0]);
              const logdet1 = Math.log(s1[0][0]*s1[1][1] - s1[0][1]*s1[1][0]);

              // Decision boundary via grid evaluation
              const res = 120;
              const xMin = -5, xMax = 5, yMin = -4, yMax = 4;
              const dxg = (xMax - xMin) / res, dyg = (yMax - yMin) / res;

              for (let i = 0; i < res; i++) {
                for (let j = 0; j < res; j++) {
                  const x = xMin + (i + 0.5) * dxg, y = yMin + (j + 0.5) * dyg;
                  const dx0 = x - mu0[0], dy0 = y - mu0[1];
                  const dx1 = x - mu1[0], dy1 = y - mu1[1];
                  const q0 = dx0*(inv0[0][0]*dx0 + inv0[0][1]*dy0) + dy0*(inv0[1][0]*dx0 + inv0[1][1]*dy0);
                  const q1 = dx1*(inv1[0][0]*dx1 + inv1[0][1]*dy1) + dy1*(inv1[1][0]*dx1 + inv1[1][1]*dy1);
                  const d0v = -0.5*logdet0 - 0.5*q0;
                  const d1v = -0.5*logdet1 - 0.5*q1;
                  const cls = d1v > d0v ? 1 : 0;
                  const [sx, sy] = viz.toScreen(x, y);
                  ctx.fillStyle = cls === 1 ? 'rgba(88,166,255,0.06)' : 'rgba(248,81,73,0.06)';
                  ctx.fillRect(sx - dxg*viz.scale/2, sy - dyg*viz.scale/2, dxg*viz.scale + 1, dyg*viz.scale + 1);
                }
              }

              // Draw boundary contour
              for (let i = 0; i < res - 1; i++) {
                for (let j = 0; j < res - 1; j++) {
                  const vals = [];
                  for (let di = 0; di <= 1; di++) {
                    for (let dj = 0; dj <= 1; dj++) {
                      const x = xMin + (i + di) * dxg, y = yMin + (j + dj) * dyg;
                      const dx0 = x - mu0[0], dy0 = y - mu0[1];
                      const dx1 = x - mu1[0], dy1 = y - mu1[1];
                      const q0 = dx0*(inv0[0][0]*dx0+inv0[0][1]*dy0)+dy0*(inv0[1][0]*dx0+inv0[1][1]*dy0);
                      const q1 = dx1*(inv1[0][0]*dx1+inv1[0][1]*dy1)+dy1*(inv1[1][0]*dx1+inv1[1][1]*dy1);
                      vals.push((-0.5*logdet1 - 0.5*q1) - (-0.5*logdet0 - 0.5*q0));
                    }
                  }
                  const hasPos = vals.some(v => v > 0), hasNeg = vals.some(v => v < 0);
                  if (hasPos && hasNeg) {
                    const cx = xMin + (i + 0.5) * dxg, cy = yMin + (j + 0.5) * dyg;
                    const [sx, sy] = viz.toScreen(cx, cy);
                    ctx.fillStyle = viz.colors.green;
                    ctx.fillRect(sx - 1.5, sy - 1.5, 3, 3);
                  }
                }
              }

              // Draw ellipses
              drawEllipse(ctx, viz, mu0, s0, viz.colors.red);
              drawEllipse(ctx, viz, mu1, s1, viz.colors.blue);

              // Draw data points
              for (const p of d0) viz.drawPoint(p[0], p[1], viz.colors.red, null, 3);
              for (const p of d1) viz.drawPoint(p[0], p[1], viz.colors.blue, null, 3);

              // Draw means
              viz.drawPoint(mu0[0], mu0[1], viz.colors.red, '\u03BC\u2080', 6);
              viz.drawPoint(mu1[0], mu1[1], viz.colors.blue, '\u03BC\u2081', 6);

              // Mode label
              viz.screenText(mode + ' Decision Boundary', viz.width / 2, viz.height - 12, viz.colors.green, 13);
              viz.screenText(isLDA ? '(Linear: shared \u03A3)' : '(Quadratic: separate \u03A3\u2080, \u03A3\u2081)', viz.width / 2, viz.height - 28, viz.colors.text, 11);
            }

            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'In LDA with two classes in \\(\\mathbb{R}^2\\), shared covariance \\(\\boldsymbol{\\Sigma} = \\mathbf{I}\\), means \\(\\boldsymbol{\\mu}_0 = (0,0)^\\top\\) and \\(\\boldsymbol{\\mu}_1 = (2,0)^\\top\\), and equal priors, find the equation of the decision boundary.',
          hint: 'With \\(\\boldsymbol{\\Sigma} = \\mathbf{I}\\), the decision boundary is the perpendicular bisector of the line segment connecting the two means.',
          solution: 'The boundary is where \\(\\boldsymbol{\\mu}_1^\\top\\boldsymbol{\\Sigma}^{-1}\\mathbf{x} - \\frac{1}{2}\\boldsymbol{\\mu}_1^\\top\\boldsymbol{\\Sigma}^{-1}\\boldsymbol{\\mu}_1 = \\boldsymbol{\\mu}_0^\\top\\boldsymbol{\\Sigma}^{-1}\\mathbf{x} - \\frac{1}{2}\\boldsymbol{\\mu}_0^\\top\\boldsymbol{\\Sigma}^{-1}\\boldsymbol{\\mu}_0\\). With \\(\\boldsymbol{\\Sigma} = \\mathbf{I}\\): \\(2x_1 - 2 = 0\\), giving \\(x_1 = 1\\). This is a vertical line at the midpoint of the two means.'
        },
        {
          question: 'Explain why LDA produces a linear decision boundary while QDA produces a quadratic one. What specific algebraic cancellation occurs in LDA that does not occur in QDA?',
          hint: 'Look at the quadratic form \\(\\mathbf{x}^\\top\\boldsymbol{\\Sigma}_k^{-1}\\mathbf{x}\\) in the log-posterior.',
          solution: 'In both LDA and QDA, the log-posterior involves \\(-\\frac{1}{2}\\mathbf{x}^\\top\\boldsymbol{\\Sigma}_k^{-1}\\mathbf{x}\\). In LDA, \\(\\boldsymbol{\\Sigma}_k = \\boldsymbol{\\Sigma}\\) for all \\(k\\), so this term is identical across classes and cancels when computing the decision boundary (where two log-posteriors are equal). Only linear terms in \\(\\mathbf{x}\\) remain. In QDA, \\(\\boldsymbol{\\Sigma}_k\\) differs across classes, so the quadratic terms \\(\\mathbf{x}^\\top\\boldsymbol{\\Sigma}_k^{-1}\\mathbf{x}\\) do not cancel.'
        },
        {
          question: 'How many parameters does LDA have for \\(K\\) classes in \\(\\mathbb{R}^d\\)? How many does QDA have? When might LDA outperform QDA despite being less flexible?',
          hint: 'Count: \\(K-1\\) prior parameters, \\(Kd\\) mean parameters, and either one or \\(K\\) covariance matrices each with \\(d(d+1)/2\\) unique entries.',
          solution: 'LDA: \\((K-1) + Kd + d(d+1)/2\\) parameters. QDA: \\((K-1) + Kd + Kd(d+1)/2\\) parameters. The difference is \\((K-1)d(d+1)/2\\). When \\(d\\) is large relative to \\(n\\), QDA must estimate \\(K\\) full covariance matrices, leading to high variance and overfitting. LDA, by sharing the covariance, has far fewer parameters and generalizes better in these high-dimensional, small-sample regimes.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 3: Naive Bayes
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch06-sec03',
      title: 'Naive Bayes',
      content: `
<h2>Naive Bayes</h2>

<p>GDA models the full joint distribution of features within each class. For high-dimensional or discrete data, estimating a full covariance (or full joint distribution) becomes impractical. The <strong>Naive Bayes</strong> classifier takes a radical shortcut: it assumes all features are <em>conditionally independent</em> given the class label.</p>

<div class="env-block definition"><div class="env-title">Definition 6.5 (Naive Bayes Assumption)</div><div class="env-body">
<p>The <strong>naive Bayes assumption</strong> states that features are conditionally independent given the class:</p>
\\[P(\\mathbf{x} \\mid y = k) = \\prod_{j=1}^{d} P(x_j \\mid y = k).\\]
<p>Under this assumption, the posterior becomes</p>
\\[P(y = k \\mid \\mathbf{x}) \\propto \\pi_k \\prod_{j=1}^{d} P(x_j \\mid y = k).\\]
</div></div>

<div class="env-block intuition"><div class="env-title">Why "Naive"?</div><div class="env-body"><p>The conditional independence assumption is almost never true in practice. In text classification, the presence of "machine" makes "learning" more likely, violating independence. Yet Naive Bayes often works surprisingly well, because accurate posterior <em>ranking</em> (which class has the highest posterior) does not require accurate posterior <em>calibration</em> (exact probability values).</p></div></div>

<h3>Parameter Estimation</h3>

<p>The massive advantage of Naive Bayes is parameter efficiency. Instead of a \\(d \\times d\\) covariance matrix, we only need \\(d\\) univariate distributions per class.</p>

<p><strong>For discrete features</strong> (e.g., word counts in text): each \\(P(x_j \\mid y = k)\\) is a categorical distribution, estimated by</p>
\\[\\hat{P}(x_j = v \\mid y = k) = \\frac{\\text{count}(x_j = v \\text{ and } y = k) + \\alpha}{n_k + \\alpha |V_j|},\\]
<p>where \\(\\alpha\\) is a <strong>Laplace smoothing</strong> parameter (typically \\(\\alpha = 1\\)) that prevents zero probabilities, and \\(|V_j|\\) is the vocabulary size for feature \\(j\\).</p>

<p><strong>For continuous features</strong>: each \\(P(x_j \\mid y = k)\\) is typically a univariate Gaussian \\(\\mathcal{N}(\\mu_{jk}, \\sigma_{jk}^2)\\), estimated from the within-class sample mean and variance of feature \\(j\\).</p>

<div class="env-block example"><div class="env-title">Example 6.1 (Spam Classification)</div><div class="env-body">
<p>Consider classifying emails as spam (\\(y=1\\)) or ham (\\(y=0\\)) based on word presence. Suppose from training data:</p>
<ul>
  <li>\\(P(\\text{"free"} \\mid \\text{spam}) = 0.8\\), \\(P(\\text{"free"} \\mid \\text{ham}) = 0.1\\)</li>
  <li>\\(P(\\text{"meeting"} \\mid \\text{spam}) = 0.1\\), \\(P(\\text{"meeting"} \\mid \\text{ham}) = 0.5\\)</li>
  <li>\\(\\pi_{\\text{spam}} = 0.3\\), \\(\\pi_{\\text{ham}} = 0.7\\)</li>
</ul>
<p>For an email containing "free" but not "meeting":</p>
\\[P(\\text{spam} \\mid \\mathbf{x}) \\propto 0.3 \\times 0.8 \\times 0.9 = 0.216\\]
\\[P(\\text{ham} \\mid \\mathbf{x}) \\propto 0.7 \\times 0.1 \\times 0.5 = 0.035\\]
<p>Normalizing: \\(P(\\text{spam} \\mid \\mathbf{x}) = 0.216 / (0.216 + 0.035) \\approx 0.86\\). The email is likely spam.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Naive Bayes and GDA</div><div class="env-body"><p>Gaussian Naive Bayes with diagonal covariance is a special case of GDA where \\(\\boldsymbol{\\Sigma}_k\\) is forced to be diagonal. If additionally the diagonal entries are shared across classes, it becomes a restricted form of LDA. The conditional independence assumption is equivalent to zero off-diagonal entries in the covariance matrix.</p></div></div>

<div class="viz-placeholder" data-viz="viz-naive-bayes"></div>
`,
      visualizations: [
        {
          id: 'viz-naive-bayes',
          title: 'Naive Bayes Spam Classifier',
          description: 'Word feature likelihoods for spam vs ham. Toggle words present in the email to see how the posterior probability of spam updates. Each bar shows P(word|class).',
          setup(container, controls) {
            const viz = new VizEngine(container, { width: 560, height: 380, scale: 1, originX: 0, originY: 0 });

            const words = [
              { word: 'free', pSpam: 0.80, pHam: 0.05 },
              { word: 'winner', pSpam: 0.65, pHam: 0.02 },
              { word: 'click', pSpam: 0.70, pHam: 0.08 },
              { word: 'meeting', pSpam: 0.05, pHam: 0.45 },
              { word: 'report', pSpam: 0.08, pHam: 0.40 },
              { word: 'urgent', pSpam: 0.55, pHam: 0.10 },
              { word: 'offer', pSpam: 0.60, pHam: 0.06 },
              { word: 'project', pSpam: 0.04, pHam: 0.50 }
            ];
            const present = new Array(words.length).fill(false);
            let piSpam = 0.3;

            // Toggle buttons
            const wordRow = document.createElement('div');
            wordRow.style.cssText = 'display:flex;gap:4px;flex-wrap:wrap;';
            const wordBtns = words.map((w, i) => {
              const b = VizEngine.createButton(wordRow, w.word, () => {
                present[i] = !present[i];
                b.style.background = present[i] ? '#2a5a3a' : '#1a1a40';
                b.style.borderColor = present[i] ? '#3fb950' : '#30363d';
                draw();
              });
              return b;
            });
            controls.appendChild(wordRow);

            function draw() {
              const ctx = viz.ctx;
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, viz.width, viz.height);

              const leftMargin = 80, topMargin = 50;
              const barW = 50, barGap = 8, groupGap = 14;
              const maxH = 160;

              // Title
              ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Word Likelihoods: P(word | class)', viz.width / 2, 20);

              // Legend
              ctx.fillStyle = viz.colors.red; ctx.fillRect(viz.width - 160, 8, 12, 12);
              ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'left'; ctx.fillText('Spam', viz.width - 144, 18);
              ctx.fillStyle = viz.colors.blue; ctx.fillRect(viz.width - 90, 8, 12, 12);
              ctx.fillStyle = viz.colors.text; ctx.fillText('Ham', viz.width - 74, 18);

              // Draw bars for each word
              const groupW = barW * 2 + barGap;
              const totalW = words.length * groupW + (words.length - 1) * groupGap;
              const startX = (viz.width - totalW) / 2;

              for (let i = 0; i < words.length; i++) {
                const w = words[i];
                const gx = startX + i * (groupW + groupGap);

                // Spam bar
                const hSpam = w.pSpam * maxH;
                ctx.fillStyle = present[i] ? viz.colors.red : viz.colors.red + '55';
                ctx.fillRect(gx, topMargin + maxH - hSpam, barW, hSpam);

                // Ham bar
                const hHam = w.pHam * maxH;
                ctx.fillStyle = present[i] ? viz.colors.blue : viz.colors.blue + '55';
                ctx.fillRect(gx + barW + barGap, topMargin + maxH - hHam, barW, hHam);

                // Word label
                ctx.fillStyle = present[i] ? viz.colors.green : viz.colors.text;
                ctx.font = (present[i] ? 'bold ' : '') + '11px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(w.word, gx + groupW / 2, topMargin + maxH + 16);
                if (present[i]) {
                  ctx.fillText('\u2713', gx + groupW / 2, topMargin + maxH + 28);
                }

                // Probability labels on bars
                ctx.fillStyle = viz.colors.text; ctx.font = '9px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                if (hSpam > 12) ctx.fillText(w.pSpam.toFixed(2), gx + barW/2, topMargin + maxH - hSpam + 12);
                if (hHam > 12) ctx.fillText(w.pHam.toFixed(2), gx + barW + barGap + barW/2, topMargin + maxH - hHam + 12);
              }

              // Baseline
              ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
              ctx.beginPath(); ctx.moveTo(startX - 10, topMargin + maxH); ctx.lineTo(startX + totalW + 10, topMargin + maxH); ctx.stroke();

              // Compute posterior
              let logSpam = Math.log(piSpam), logHam = Math.log(1 - piSpam);
              let anyPresent = false;
              for (let i = 0; i < words.length; i++) {
                if (present[i]) {
                  anyPresent = true;
                  logSpam += Math.log(words[i].pSpam);
                  logHam += Math.log(words[i].pHam);
                } else {
                  logSpam += Math.log(1 - words[i].pSpam);
                  logHam += Math.log(1 - words[i].pHam);
                }
              }
              const maxLog = Math.max(logSpam, logHam);
              const pSpamPost = Math.exp(logSpam - maxLog) / (Math.exp(logSpam - maxLog) + Math.exp(logHam - maxLog));

              // Draw posterior bar
              const postY = topMargin + maxH + 50;
              ctx.fillStyle = viz.colors.text; ctx.font = 'bold 12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Posterior: P(spam | email)', viz.width / 2, postY);

              const postBarW = viz.width - 120, postBarH = 24;
              const postBarX = 60, postBarY = postY + 10;

              // Background
              ctx.fillStyle = viz.colors.blue + '44';
              ctx.fillRect(postBarX, postBarY, postBarW, postBarH);
              // Spam portion
              ctx.fillStyle = pSpamPost > 0.5 ? viz.colors.red : viz.colors.red + 'aa';
              ctx.fillRect(postBarX, postBarY, postBarW * pSpamPost, postBarH);
              // Border
              ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
              ctx.strokeRect(postBarX, postBarY, postBarW, postBarH);

              // Labels
              ctx.fillStyle = viz.colors.white; ctx.font = 'bold 11px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('SPAM: ' + (pSpamPost * 100).toFixed(1) + '%', postBarX + postBarW * pSpamPost / 2, postBarY + postBarH / 2 + 4);
              if (postBarW * (1 - pSpamPost) > 70) {
                ctx.fillText('HAM: ' + ((1 - pSpamPost) * 100).toFixed(1) + '%', postBarX + postBarW * pSpamPost + postBarW * (1 - pSpamPost) / 2, postBarY + postBarH / 2 + 4);
              }

              // Decision
              const decision = pSpamPost > 0.5 ? 'SPAM' : 'HAM';
              const decColor = pSpamPost > 0.5 ? viz.colors.red : viz.colors.green;
              ctx.fillStyle = decColor; ctx.font = 'bold 14px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Classification: ' + decision, viz.width / 2, postBarY + postBarH + 22);
            }

            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Prove that Gaussian Naive Bayes (with class-specific diagonal covariances) is a special case of QDA where all covariance matrices are diagonal.',
          hint: 'Write out the QDA discriminant function with a diagonal \\(\\boldsymbol{\\Sigma}_k\\) and show it factorizes.',
          solution: 'When \\(\\boldsymbol{\\Sigma}_k = \\text{diag}(\\sigma_{1k}^2, \\dots, \\sigma_{dk}^2)\\), the quadratic form becomes \\((\\mathbf{x} - \\boldsymbol{\\mu}_k)^\\top\\boldsymbol{\\Sigma}_k^{-1}(\\mathbf{x} - \\boldsymbol{\\mu}_k) = \\sum_{j=1}^d \\frac{(x_j - \\mu_{jk})^2}{\\sigma_{jk}^2}\\), and \\(|\\boldsymbol{\\Sigma}_k| = \\prod_j \\sigma_{jk}^2\\). The class-conditional density factors as \\(P(\\mathbf{x} \\mid y=k) = \\prod_{j=1}^d \\frac{1}{\\sqrt{2\\pi}\\sigma_{jk}}\\exp\\left(-\\frac{(x_j - \\mu_{jk})^2}{2\\sigma_{jk}^2}\\right) = \\prod_j P(x_j \\mid y=k)\\), which is precisely the Naive Bayes assumption with Gaussian marginals.'
        },
        {
          question: 'Explain why Laplace smoothing is necessary for Naive Bayes with discrete features. What goes wrong without it?',
          hint: 'Consider what happens when a word appears in the test email but never appeared in spam training emails.',
          solution: 'Without smoothing, if word \\(j\\) never appears in class \\(k\\) training data, then \\(\\hat{P}(x_j = 1 \\mid y = k) = 0\\). Since the Naive Bayes posterior involves a product of these terms, a single zero factor makes the entire posterior \\(P(y = k \\mid \\mathbf{x}) = 0\\), regardless of all other features. This is catastrophic: one unseen word-class combination overrides all other evidence. Laplace smoothing adds \\(\\alpha\\) pseudo-counts to ensure no probability is exactly zero.'
        },
        {
          question: 'A Naive Bayes classifier has three binary features. Class priors are \\(\\pi_0 = 0.6, \\pi_1 = 0.4\\). The likelihoods are \\(P(x_1=1|y=0)=0.2, P(x_1=1|y=1)=0.9, P(x_2=1|y=0)=0.7, P(x_2=1|y=1)=0.3, P(x_3=1|y=0)=0.5, P(x_3=1|y=1)=0.5\\). Classify \\(\\mathbf{x} = (1, 0, 1)\\).',
          hint: 'Compute unnormalized posteriors for each class: \\(\\pi_k \\prod_j P(x_j \\mid y=k)\\). Use \\(P(x_j=0 \\mid y=k) = 1 - P(x_j=1 \\mid y=k)\\).',
          solution: 'For class 0: \\(0.6 \\times 0.2 \\times (1-0.7) \\times 0.5 = 0.6 \\times 0.2 \\times 0.3 \\times 0.5 = 0.018\\). For class 1: \\(0.4 \\times 0.9 \\times (1-0.3) \\times 0.5 = 0.4 \\times 0.9 \\times 0.7 \\times 0.5 = 0.126\\). Normalizing: \\(P(y=1 \\mid \\mathbf{x}) = 0.126 / (0.018 + 0.126) = 0.875\\). Classify as class 1. Note that feature \\(x_3\\) contributes no information (equal likelihood for both classes).'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 4: Generative vs Discriminative Comparison
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch06-sec04',
      title: 'Generative vs Discriminative Comparison',
      content: `
<h2>When Does the Generative Approach Win?</h2>

<p>We have now seen two families of classifiers: generative (GDA, Naive Bayes) and discriminative (logistic regression from Chapter 5). A natural question arises: when should we prefer one over the other? The answer depends on the sample size, the accuracy of the generative model, and the dimensionality of the problem.</p>

<div class="env-block theorem"><div class="env-title">Theorem 6.2 (Ng and Jordan, 2002)</div><div class="env-body">
<p>Consider Gaussian class-conditionals with shared covariance (so the Bayes-optimal classifier has a linear boundary). Then:</p>
<ol>
  <li><strong>Asymptotically</strong> (\\(n \\to \\infty\\)), discriminative logistic regression achieves lower classification error than LDA (or equal, if the Gaussian assumption is correct).</li>
  <li><strong>For small \\(n\\)</strong>, LDA can achieve lower error than logistic regression, converging to its asymptotic error as \\(O(d/n)\\) versus \\(O(d^2/n)\\) for logistic regression.</li>
</ol>
<p>More precisely, the number of samples needed for logistic regression to match LDA's performance scales as \\(\\Theta(d)\\).</p>
</div></div>

<div class="env-block intuition"><div class="env-title">Why Generative Models Can Win with Small Samples</div><div class="env-body"><p>A generative model makes stronger assumptions (the data follows a specific distribution). If those assumptions are approximately correct, the model can extract more information from each data point, leading to faster convergence. A discriminative model makes weaker assumptions, which gives it an advantage when the sample is large enough to estimate the decision boundary directly, but a disadvantage when data is scarce.</p></div></div>

<h3>The Connection: LDA and Logistic Regression</h3>

<p>A remarkable fact links the two paradigms. When the class-conditional densities are Gaussian with shared covariance, the posterior \\(P(y = 1 \\mid \\mathbf{x})\\) takes the form</p>
\\[P(y = 1 \\mid \\mathbf{x}) = \\frac{1}{1 + \\exp(-\\mathbf{w}^\\top \\mathbf{x} - b)},\\]
<p>which is exactly the logistic regression model. So LDA and logistic regression define the <em>same model family</em> for the posterior. The difference is in how parameters are estimated:</p>

<ul>
  <li><strong>LDA</strong>: estimates \\(\\boldsymbol{\\mu}_0, \\boldsymbol{\\mu}_1, \\boldsymbol{\\Sigma}, \\pi\\) by MLE of the <em>joint</em> \\(P(\\mathbf{x}, y)\\), then computes \\(\\mathbf{w}, b\\) from these.</li>
  <li><strong>Logistic regression</strong>: estimates \\(\\mathbf{w}, b\\) directly by maximizing the <em>conditional</em> likelihood \\(P(y \\mid \\mathbf{x})\\).</li>
</ul>

<div class="env-block remark"><div class="env-title">Model Misspecification</div><div class="env-body"><p>If the Gaussian assumption is wrong, LDA's generative MLE can produce a worse posterior estimate than logistic regression's discriminative MLE, even though both have the same model for \\(P(y \\mid \\mathbf{x})\\). This is because the generative MLE is "distracted" by trying to fit \\(P(\\mathbf{x})\\), which is irrelevant for classification.</p></div></div>

<h3>Practical Guidelines</h3>

<table>
<thead><tr><th>Situation</th><th>Recommended approach</th></tr></thead>
<tbody>
<tr><td>Small \\(n\\), model roughly correct</td><td>Generative (LDA, Naive Bayes)</td></tr>
<tr><td>Large \\(n\\)</td><td>Discriminative (logistic regression)</td></tr>
<tr><td>Very high \\(d\\), sparse features</td><td>Naive Bayes (fewest parameters)</td></tr>
<tr><td>Need outlier detection</td><td>Generative (can evaluate \\(P(\\mathbf{x})\\))</td></tr>
<tr><td>Missing features at test time</td><td>Generative (marginalizes naturally)</td></tr>
<tr><td>Need well-calibrated probabilities</td><td>Discriminative</td></tr>
</tbody>
</table>

<div class="viz-placeholder" data-viz="viz-learning-curves"></div>

<div class="env-block remark"><div class="env-title">Beyond Binary</div><div class="env-body"><p>For \\(K > 2\\) classes, the generative model \\(P(y=k \\mid \\mathbf{x}) \\propto \\pi_k P(\\mathbf{x} \\mid y=k)\\) with Gaussian class-conditionals and shared covariance yields a <strong>softmax</strong> posterior, which is the multi-class generalization of logistic regression. This further reinforces that LDA and multinomial logistic regression are intimately connected.</p></div></div>
`,
      visualizations: [
        {
          id: 'viz-learning-curves',
          title: 'Learning Curves: LDA vs Logistic Regression',
          description: 'Accuracy vs training sample size. When the Gaussian assumption holds, LDA converges faster with small samples but both converge to similar accuracy. Adjust the model correctness to see how misspecification affects the comparison.',
          setup(container, controls) {
            const viz = new VizEngine(container, { width: 560, height: 360, scale: 1, originX: 0, originY: 0 });
            let correctness = 1.0; // 1.0 = Gaussian is correct, 0.0 = very wrong

            VizEngine.createSlider(controls, 'Model correctness', 0.0, 1.0, correctness, 0.05, v => { correctness = v; draw(); });

            function draw() {
              const ctx = viz.ctx;
              ctx.fillStyle = viz.colors.bg;
              ctx.fillRect(0, 0, viz.width, viz.height);

              const left = 70, right = viz.width - 30, top = 40, bottom = viz.height - 50;
              const plotW = right - left, plotH = bottom - top;

              // Axes
              ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1.5;
              ctx.beginPath(); ctx.moveTo(left, bottom); ctx.lineTo(right, bottom); ctx.stroke();
              ctx.beginPath(); ctx.moveTo(left, bottom); ctx.lineTo(left, top); ctx.stroke();

              // Labels
              ctx.fillStyle = viz.colors.text; ctx.font = '12px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Training sample size n', (left + right) / 2, bottom + 35);
              ctx.save(); ctx.translate(18, (top + bottom) / 2); ctx.rotate(-Math.PI / 2);
              ctx.fillText('Test accuracy', 0, 0); ctx.restore();

              // X-axis ticks
              const nValues = [10, 50, 100, 200, 500, 1000];
              const maxN = 1000;
              ctx.font = '10px -apple-system,sans-serif';
              for (const n of nValues) {
                const x = left + (Math.log(n) - Math.log(5)) / (Math.log(maxN) - Math.log(5)) * plotW;
                ctx.fillStyle = viz.colors.text; ctx.textAlign = 'center';
                ctx.fillText(n.toString(), x, bottom + 15);
                ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                ctx.beginPath(); ctx.moveTo(x, bottom); ctx.lineTo(x, top); ctx.stroke();
              }

              // Y-axis ticks
              for (let a = 0.5; a <= 1.0; a += 0.1) {
                const y = bottom - ((a - 0.45) / 0.55) * plotH;
                ctx.fillStyle = viz.colors.text; ctx.textAlign = 'right';
                ctx.fillText(a.toFixed(1), left - 8, y + 4);
                ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
              }

              // Simulated learning curves
              // LDA: converges as O(d/n), asymptote depends on correctness
              // LogReg: converges as O(d^2/n) but potentially higher asymptote
              const ldaAsymptote = 0.88 + 0.07 * correctness; // 0.88-0.95
              const lrAsymptote = 0.92 + 0.03 * correctness;  // 0.92-0.95
              const ldaRate = 15 * (1 + correctness);     // faster if correct
              const lrRate = 40;                           // slower convergence

              function ldaAcc(n) { return ldaAsymptote * (1 - Math.exp(-n / ldaRate)) + 0.5 * Math.exp(-n / ldaRate); }
              function lrAcc(n) { return lrAsymptote * (1 - Math.exp(-n / lrRate)) + 0.5 * Math.exp(-n / lrRate); }

              // Draw curves
              const steps = 200;
              // LDA curve
              ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 2.5;
              ctx.beginPath();
              for (let i = 0; i <= steps; i++) {
                const n = 5 * Math.pow(maxN / 5, i / steps);
                const acc = ldaAcc(n);
                const x = left + (i / steps) * plotW;
                const y = bottom - ((acc - 0.45) / 0.55) * plotH;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
              }
              ctx.stroke();

              // LogReg curve
              ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 2.5;
              ctx.beginPath();
              for (let i = 0; i <= steps; i++) {
                const n = 5 * Math.pow(maxN / 5, i / steps);
                const acc = lrAcc(n);
                const x = left + (i / steps) * plotW;
                const y = bottom - ((acc - 0.45) / 0.55) * plotH;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
              }
              ctx.stroke();

              // Crossover point
              let crossN = null;
              for (let n = 5; n <= maxN; n++) {
                if (lrAcc(n) >= ldaAcc(n) && lrAcc(n - 1) < ldaAcc(n - 1)) { crossN = n; break; }
              }
              if (crossN && crossN < maxN) {
                const cx = left + (Math.log(crossN) - Math.log(5)) / (Math.log(maxN) - Math.log(5)) * plotW;
                const cy = bottom - ((ldaAcc(crossN) - 0.45) / 0.55) * plotH;
                ctx.strokeStyle = viz.colors.green; ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 4]);
                ctx.beginPath(); ctx.moveTo(cx, bottom); ctx.lineTo(cx, cy); ctx.stroke();
                ctx.setLineDash([]);
                ctx.fillStyle = viz.colors.green; ctx.font = '10px -apple-system,sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('crossover n\u2248' + crossN, cx, bottom - 5);
              }

              // Legend
              ctx.fillStyle = viz.colors.blue; ctx.fillRect(right - 160, top + 5, 14, 3);
              ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'left'; ctx.fillText('LDA (generative)', right - 142, top + 10);
              ctx.fillStyle = viz.colors.orange; ctx.fillRect(right - 160, top + 22, 14, 3);
              ctx.fillStyle = viz.colors.text; ctx.fillText('Logistic Reg (discriminative)', right - 142, top + 27);

              // Asymptote labels
              ctx.setLineDash([3, 3]);
              ctx.strokeStyle = viz.colors.blue + '66'; ctx.lineWidth = 1;
              const ldaAsyY = bottom - ((ldaAsymptote - 0.45) / 0.55) * plotH;
              ctx.beginPath(); ctx.moveTo(left, ldaAsyY); ctx.lineTo(right, ldaAsyY); ctx.stroke();
              ctx.strokeStyle = viz.colors.orange + '66';
              const lrAsyY = bottom - ((lrAsymptote - 0.45) / 0.55) * plotH;
              ctx.beginPath(); ctx.moveTo(left, lrAsyY); ctx.lineTo(right, lrAsyY); ctx.stroke();
              ctx.setLineDash([]);

              // Title
              ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif';
              ctx.textAlign = 'center';
              ctx.fillText('Model correctness: ' + (correctness * 100).toFixed(0) + '%', viz.width / 2, 18);
              ctx.fillStyle = viz.colors.text; ctx.font = '10px -apple-system,sans-serif';
              ctx.fillText(correctness > 0.7 ? 'Gaussian assumption approximately correct' : 'Gaussian assumption is a poor fit', viz.width / 2, 32);
            }

            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Explain why LDA and logistic regression define the same model family for the posterior \\(P(y \\mid \\mathbf{x})\\) when class-conditionals are Gaussian with shared covariance, yet can yield different parameter estimates.',
          hint: 'Consider the difference between maximizing \\(P(\\mathbf{x}, y)\\) vs \\(P(y \\mid \\mathbf{x})\\).',
          solution: 'Both yield \\(P(y=1 \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}^\\top\\mathbf{x} + b)\\) with the same functional form. However, LDA estimates \\(\\boldsymbol{\\mu}_0, \\boldsymbol{\\mu}_1, \\boldsymbol{\\Sigma}, \\pi\\) by maximizing the joint likelihood \\(\\prod_i P(\\mathbf{x}_i, y_i)\\), then derives \\(\\mathbf{w} = \\boldsymbol{\\Sigma}^{-1}(\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_0)\\). Logistic regression directly maximizes \\(\\prod_i P(y_i \\mid \\mathbf{x}_i)\\). The joint and conditional objectives are different, so the resulting \\(\\mathbf{w}\\) generally differ. The joint objective also "cares about" fitting \\(P(\\mathbf{x})\\), which is irrelevant for classification but can hurt if the model is wrong.'
        },
        {
          question: 'In the Ng and Jordan (2002) result, the sample complexity of LDA is \\(O(d)\\) while logistic regression is \\(O(d^2)\\). Give an intuitive explanation for this gap.',
          hint: 'Think about how many parameters each method must effectively estimate.',
          solution: 'LDA estimates means (\\(2d\\) parameters) and one shared covariance (\\(O(d^2)\\) parameters), but the effective parameters governing the decision boundary are \\(\\mathbf{w} = \\boldsymbol{\\Sigma}^{-1}(\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_0)\\) and \\(b\\), which are \\(d+1\\) numbers. The generative structure constrains how these are estimated: \\(\\boldsymbol{\\mu}\\) converges at rate \\(O(d/n)\\) and dominates. Logistic regression directly estimates \\(d+1\\) parameters but without the structural constraints; the optimization landscape has curvature that depends on the covariance structure, effectively requiring \\(O(d^2)\\) samples to "discover" the covariance structure implicitly.'
        },
        {
          question: 'Can a generative classifier ever achieve strictly better asymptotic error than a discriminative classifier? Under what conditions?',
          hint: 'Think about what "asymptotic" means. With infinite data, which approach learns the true posterior?',
          solution: 'No, assuming the discriminative model family is flexible enough to represent the true posterior. With infinite data, a discriminative classifier converges to the posterior that minimizes classification error (the Bayes-optimal boundary). A generative classifier with a misspecified model converges to a suboptimal boundary. If the generative model is correctly specified, both converge to the same Bayes-optimal boundary. The only scenario where generative might seem "better" asymptotically is when the discriminative model family is too restrictive to represent the true posterior, but this is a modeling choice, not an inherent advantage of the generative approach.'
        }
      ]
    }
  ]
});
