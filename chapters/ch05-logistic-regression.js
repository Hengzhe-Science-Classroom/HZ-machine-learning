window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
  id: 'ch05',
  number: 5,
  title: 'Logistic Regression',
  subtitle: 'Probabilistic Classification via the Sigmoid, Maximum Likelihood, and Gradient Descent',
  sections: [

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 1: The Logistic Function
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch05-sec01',
      title: 'The Logistic Function',
      content: `
<h2>The Logistic Function</h2>

<p>The perceptron makes hard binary decisions: a point is either class +1 or class -1, with no notion of uncertainty. In many applications, we want a <strong>probability</strong>: how confident is the classifier? Logistic regression achieves this by replacing the step function with the <strong>logistic (sigmoid) function</strong>.</p>

<div class="env-block definition"><div class="env-title">Definition 5.1 (Logistic/Sigmoid Function)</div><div class="env-body">
<p>The <strong>logistic function</strong> (or <strong>sigmoid</strong>) is</p>
\\[\\sigma(z) = \\frac{1}{1 + e^{-z}} = \\frac{e^z}{1 + e^z}.\\]
<p>It maps \\(\\mathbb{R} \\to (0, 1)\\), with \\(\\sigma(0) = 0.5\\), \\(\\lim_{z \\to \\infty} \\sigma(z) = 1\\), and \\(\\lim_{z \\to -\\infty} \\sigma(z) = 0\\).</p>
</div></div>

<h3>Key Properties</h3>

<p>The sigmoid has several elegant properties that make it ideal for probabilistic classification:</p>
<ol>
  <li><strong>Symmetry:</strong> \\(\\sigma(-z) = 1 - \\sigma(z)\\). This means \\(p(y=0 \\mid \\mathbf{x}) = 1 - \\sigma(z) = \\sigma(-z)\\).</li>
  <li><strong>Derivative:</strong> \\(\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))\\). The derivative is maximized at \\(z=0\\), where uncertainty is greatest.</li>
  <li><strong>Log-odds (logit):</strong> The inverse of the sigmoid is the <strong>logit</strong> function: \\(\\text{logit}(p) = \\ln \\frac{p}{1-p}\\). If \\(p = \\sigma(z)\\), then \\(z = \\text{logit}(p)\\).</li>
</ol>

<div class="env-block intuition"><div class="env-title">From Odds to Probabilities</div><div class="env-body"><p>Logistic regression models the <strong>log-odds</strong> as a linear function of the features: \\(\\ln \\frac{p(y=1|\\mathbf{x})}{p(y=0|\\mathbf{x})} = \\mathbf{w}^\\top \\mathbf{x} + b\\). Applying the sigmoid converts this linear score into a probability. The decision boundary occurs where the log-odds equal zero (probability 0.5), which is exactly the hyperplane \\(\\mathbf{w}^\\top \\mathbf{x} + b = 0\\).</p></div></div>

<div class="viz-placeholder" data-viz="viz-sigmoid-threshold"></div>

<div class="env-block definition"><div class="env-title">Definition 5.2 (Logistic Regression Model)</div><div class="env-body">
<p>The <strong>logistic regression</strong> model defines the class-conditional probability</p>
\\[p(y = 1 \\mid \\mathbf{x}; \\mathbf{w}, b) = \\sigma(\\mathbf{w}^\\top \\mathbf{x} + b) = \\frac{1}{1 + \\exp(-(\\mathbf{w}^\\top \\mathbf{x} + b))}.\\]
<p>A new point is classified as \\(y = 1\\) if \\(p(y=1 \\mid \\mathbf{x}) \\geq \\tau\\) for some threshold \\(\\tau\\) (typically \\(\\tau = 0.5\\)).</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark</div><div class="env-body"><p>Despite its name, logistic regression is a <em>classification</em> algorithm, not a regression algorithm. The "regression" refers to the fact that we are regressing the log-odds on the features. The outputs are probabilities, not continuous values.</p></div></div>

<div class="env-block example"><div class="env-title">Example 5.1</div><div class="env-body">
<p>With \\(\\mathbf{w} = (2, -1)^\\top\\) and \\(b = 0\\), the predicted probability for \\(\\mathbf{x} = (1, 1)^\\top\\) is \\(\\sigma(2 \\cdot 1 + (-1) \\cdot 1 + 0) = \\sigma(1) = 1/(1+e^{-1}) \\approx 0.731\\). The model assigns about 73% confidence to class 1.</p>
</div></div>
`,
      visualizations: [
        {
          id: 'viz-sigmoid-threshold',
          title: 'Sigmoid Function with Classification Threshold',
          description: 'The sigmoid curve transforms any real number into a probability. Adjust the threshold slider to change the classification cutoff. Points above the threshold are classified as class 1 (blue region), below as class 0 (red region).',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 40, originX: 280, originY: 200 });
            const ctx = viz.ctx;

            let threshold = 0.5;
            let wScale = 1.0;

            VizEngine.createSlider(controls, 'Threshold', 0.1, 0.9, threshold, 0.05, v => { threshold = v; draw(); });
            VizEngine.createSlider(controls, 'Steepness', 0.2, 3.0, wScale, 0.1, v => { wScale = v; draw(); });

            function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

            function draw() {
              viz.clear();

              // Custom axes for sigmoid plot
              // x-axis: z from -6 to 6; y-axis: sigma from 0 to 1
              const xMin = -6, xMax = 6;
              const plotLeft = 50, plotRight = viz.width - 30;
              const plotTop = 30, plotBottom = viz.height - 50;
              const pw = plotRight - plotLeft, ph = plotBottom - plotTop;

              function toPlot(z, s) {
                return [plotLeft + (z - xMin) / (xMax - xMin) * pw,
                        plotBottom - s * ph];
              }

              // Grid
              ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
              for (let z = -6; z <= 6; z += 2) {
                const [px] = toPlot(z, 0);
                ctx.beginPath(); ctx.moveTo(px, plotTop); ctx.lineTo(px, plotBottom); ctx.stroke();
              }
              for (let s = 0; s <= 1; s += 0.25) {
                const [, py] = toPlot(0, s);
                ctx.beginPath(); ctx.moveTo(plotLeft, py); ctx.lineTo(plotRight, py); ctx.stroke();
              }

              // Shade classification regions
              const threshZ = -Math.log(1/threshold - 1) / wScale;
              const [thPx] = toPlot(threshZ, 0);
              ctx.fillStyle = 'rgba(88,166,255,0.08)';
              ctx.fillRect(thPx, plotTop, plotRight - thPx, ph);
              ctx.fillStyle = 'rgba(248,81,73,0.08)';
              ctx.fillRect(plotLeft, plotTop, thPx - plotLeft, ph);

              // Axes
              ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1.5;
              ctx.beginPath(); ctx.moveTo(plotLeft, plotBottom); ctx.lineTo(plotRight, plotBottom); ctx.stroke();
              ctx.beginPath(); ctx.moveTo(plotLeft, plotBottom); ctx.lineTo(plotLeft, plotTop); ctx.stroke();

              // Labels
              ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              for (let z = -6; z <= 6; z += 2) {
                const [px] = toPlot(z, 0);
                ctx.fillText(z.toString(), px, plotBottom + 5);
              }
              ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              for (let s = 0; s <= 1; s += 0.25) {
                const [, py] = toPlot(0, s);
                ctx.fillText(s.toFixed(2), plotLeft - 5, py);
              }

              ctx.fillStyle = viz.colors.white; ctx.font = '13px -apple-system,sans-serif';
              ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              ctx.fillText('z = w\u1d40x + b', (plotLeft + plotRight) / 2, plotBottom + 22);
              ctx.save(); ctx.translate(15, (plotTop + plotBottom) / 2);
              ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
              ctx.fillText('\u03c3(z)', 0, 0); ctx.restore();

              // Sigmoid curve
              ctx.strokeStyle = viz.colors.teal; ctx.lineWidth = 2.5;
              ctx.beginPath();
              for (let px = plotLeft; px <= plotRight; px++) {
                const z = xMin + (px - plotLeft) / pw * (xMax - xMin);
                const s = sigmoid(wScale * z);
                const py = plotBottom - s * ph;
                px === plotLeft ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
              }
              ctx.stroke();

              // Threshold line
              const [, thPy] = toPlot(0, threshold);
              ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 1.5;
              ctx.setLineDash([6, 4]);
              ctx.beginPath(); ctx.moveTo(plotLeft, thPy); ctx.lineTo(plotRight, thPy); ctx.stroke();
              ctx.setLineDash([]);
              ctx.beginPath(); ctx.moveTo(thPx, plotTop); ctx.lineTo(thPx, plotBottom); ctx.stroke();

              // Labels
              ctx.fillStyle = viz.colors.yellow; ctx.font = '12px -apple-system,sans-serif';
              ctx.textAlign = 'left'; ctx.fillText('\u03c4 = ' + threshold.toFixed(2), plotRight + 3, thPy);
              ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
              ctx.fillText('z* = ' + threshZ.toFixed(2), thPx, plotTop - 4);

              // Region labels
              ctx.font = 'bold 14px -apple-system,sans-serif';
              ctx.fillStyle = viz.colors.blue;
              ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText('y = 1', (thPx + plotRight) / 2, plotTop + 15);
              ctx.fillStyle = viz.colors.red;
              ctx.fillText('y = 0', (plotLeft + thPx) / 2, plotTop + 15);

              // Midpoint dot
              const [midPx, midPy] = toPlot(0, sigmoid(0));
              ctx.fillStyle = viz.colors.white;
              ctx.beginPath(); ctx.arc(midPx, midPy, 4, 0, Math.PI * 2); ctx.fill();
              ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'left'; ctx.fillText('(0, 0.5)', midPx + 6, midPy);
            }
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Prove that \\(\\sigma(-z) = 1 - \\sigma(z)\\).',
          hint: 'Start from the definition \\(\\sigma(-z) = 1/(1+e^z)\\) and show it equals \\(1 - 1/(1+e^{-z})\\).',
          solution: '\\(\\sigma(-z) = \\frac{1}{1+e^{z}}\\). Meanwhile, \\(1 - \\sigma(z) = 1 - \\frac{1}{1+e^{-z}} = \\frac{1+e^{-z}-1}{1+e^{-z}} = \\frac{e^{-z}}{1+e^{-z}} = \\frac{1}{e^z + 1} = \\frac{1}{1+e^z} = \\sigma(-z)\\).'
        },
        {
          question: 'Show that \\(\\sigma\'(z) = \\sigma(z)(1 - \\sigma(z))\\). At what value of \\(z\\) is this derivative maximized?',
          hint: 'Write \\(\\sigma(z) = (1+e^{-z})^{-1}\\) and use the chain rule.',
          solution: 'Let \\(\\sigma = (1+e^{-z})^{-1}\\). Then \\(\\sigma\' = -(-e^{-z})(1+e^{-z})^{-2} = \\frac{e^{-z}}{(1+e^{-z})^2} = \\frac{1}{1+e^{-z}} \\cdot \\frac{e^{-z}}{1+e^{-z}} = \\sigma(1-\\sigma)\\). This is maximized when \\(\\sigma = 0.5\\), i.e., \\(z = 0\\), giving \\(\\sigma\'(0) = 0.25\\). The gradient is steepest where the model is most uncertain.'
        },
        {
          question: 'If a logistic regression model outputs \\(p(y=1 \\mid \\mathbf{x}) = 0.8\\), what are the odds and log-odds of class 1?',
          hint: 'Odds = \\(p/(1-p)\\). Log-odds = \\(\\ln(\\text{odds})\\).',
          solution: 'Odds \\(= 0.8/0.2 = 4\\). The model says class 1 is four times more likely than class 0. Log-odds \\(= \\ln(4) \\approx 1.386\\). This is the raw score \\(z = \\mathbf{w}^\\top \\mathbf{x} + b\\) that produced this probability.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 2: Maximum Likelihood for Logistic Regression
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch05-sec02',
      title: 'Maximum Likelihood for Logistic Regression',
      content: `
<h2>Maximum Likelihood for Logistic Regression</h2>

<p>Unlike linear regression, there is no closed-form solution for the optimal parameters in logistic regression. We must define a loss function and optimize it iteratively. The natural choice is <strong>maximum likelihood estimation</strong>.</p>

<h3>The Likelihood Function</h3>

<p>Given training data \\(\\{(\\mathbf{x}_i, y_i)\\}_{i=1}^n\\) with \\(y_i \\in \\{0, 1\\}\\), the probability of observing label \\(y_i\\) under the logistic model is</p>
\\[p(y_i \\mid \\mathbf{x}_i; \\boldsymbol{\\theta}) = \\sigma(z_i)^{y_i}(1 - \\sigma(z_i))^{1-y_i},\\]
<p>where \\(z_i = \\mathbf{w}^\\top \\mathbf{x}_i + b\\) and \\(\\boldsymbol{\\theta} = (\\mathbf{w}, b)\\). Assuming independence, the likelihood is</p>
\\[\\mathcal{L}(\\boldsymbol{\\theta}) = \\prod_{i=1}^n \\sigma(z_i)^{y_i}(1 - \\sigma(z_i))^{1-y_i}.\\]

<div class="env-block definition"><div class="env-title">Definition 5.3 (Negative Log-Likelihood / Cross-Entropy Loss)</div><div class="env-body">
<p>The <strong>negative log-likelihood</strong>, also known as the <strong>binary cross-entropy</strong>, is</p>
\\[\\text{NLL}(\\boldsymbol{\\theta}) = -\\ell(\\boldsymbol{\\theta}) = -\\sum_{i=1}^n \\left[y_i \\ln \\sigma(z_i) + (1-y_i) \\ln(1 - \\sigma(z_i))\\right].\\]
<p>Maximum likelihood estimation minimizes the NLL.</p>
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 5.1 (Convexity of the NLL)</div><div class="env-body">
<p>The negative log-likelihood for logistic regression is a <strong>convex function</strong> of \\(\\boldsymbol{\\theta}\\). Therefore, any local minimum is a global minimum.</p>
</div></div>

<div class="env-block proof"><div class="env-title">Proof sketch</div><div class="env-body">
<p>The NLL can be written as \\(\\text{NLL} = \\sum_i [-y_i z_i + \\ln(1 + e^{z_i})]\\). The function \\(g(z) = \\ln(1+e^z)\\) (the softplus function) has second derivative \\(g''(z) = \\sigma(z)(1-\\sigma(z)) &gt; 0\\) for all \\(z\\), so \\(g\\) is strictly convex. Since \\(z_i\\) is linear in \\(\\boldsymbol{\\theta}\\), the composition is convex. The NLL is a sum of convex functions, hence convex.</p>
<div class="qed">&#8718;</div>
</div></div>

<div class="env-block intuition"><div class="env-title">Why No Closed Form?</div><div class="env-body"><p>In linear regression, the normal equations \\(\\mathbf{X}^\\top \\mathbf{X} \\boldsymbol{\\theta} = \\mathbf{X}^\\top \\mathbf{y}\\) yield a closed-form solution because the loss is quadratic. In logistic regression, the sigmoid introduces a nonlinear transformation of \\(z\\). Setting the gradient of the NLL to zero yields a system of nonlinear equations with no algebraic solution. However, convexity guarantees that iterative methods like gradient descent will find the global optimum.</p></div></div>

<div class="viz-placeholder" data-viz="viz-log-likelihood-surface"></div>

<div class="env-block remark"><div class="env-title">Remark</div><div class="env-body"><p>The cross-entropy loss has a natural information-theoretic interpretation. It measures the average number of extra bits needed to encode labels from the true distribution using the model's predicted distribution. Minimizing cross-entropy is equivalent to minimizing the KL divergence from the true distribution to the model.</p></div></div>
`,
      visualizations: [
        {
          id: 'viz-log-likelihood-surface',
          title: 'Negative Log-Likelihood Surface',
          description: 'The contour plot shows the NLL as a function of two parameters (w, b) for a small dataset. The green trajectory shows gradient descent converging to the minimum. The surface is convex, so there is a unique minimum.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 30, originX: 280, originY: 220 });
            const ctx = viz.ctx;

            // Small dataset for 1D logistic regression: z = w*x + b
            const data = [
              {x:-2, y:0}, {x:-1.5, y:0}, {x:-1, y:0}, {x:-0.3, y:0},
              {x:0.3, y:1}, {x:1, y:1}, {x:1.5, y:1}, {x:2.5, y:1}
            ];

            function sigmoid(z) { return 1/(1+Math.exp(-Math.max(-20,Math.min(20,z)))); }
            function nll(w, b) {
              let L = 0;
              for (const p of data) {
                const z = w*p.x + b;
                const s = sigmoid(z);
                L -= p.y * Math.log(s+1e-15) + (1-p.y) * Math.log(1-s+1e-15);
              }
              return L;
            }
            function grad(w, b) {
              let gw = 0, gb = 0;
              for (const p of data) {
                const z = w*p.x + b;
                const s = sigmoid(z);
                const err = s - p.y;
                gw += err * p.x;
                gb += err;
              }
              return [gw, gb];
            }

            // Gradient descent trajectory
            let trajectory = [];
            let animStep = 0;
            let running = false, animId = null;

            function runGD() {
              trajectory = [];
              let w = -4, b = 3;
              const lr = 0.15;
              for (let i = 0; i < 60; i++) {
                trajectory.push({w, b, nll: nll(w, b)});
                const [gw, gb] = grad(w, b);
                w -= lr * gw;
                b -= lr * gb;
              }
              trajectory.push({w, b, nll: nll(w, b)});
            }
            runGD();

            VizEngine.createButton(controls, 'Run GD', () => {
              runGD(); animStep = 0; running = true;
              (function tick() {
                if (!running || animStep >= trajectory.length) { running = false; return; }
                animStep++;
                draw();
                animId = setTimeout(tick, 80);
              })();
            });
            VizEngine.createButton(controls, 'Reset', () => {
              running = false; if (animId) clearTimeout(animId);
              animStep = trajectory.length; draw();
            });

            function draw() {
              viz.clear();

              // Plot axes: w on x, b on y
              const wMin = -5, wMax = 6, bMin = -4, bMax = 5;
              const plotL = 40, plotR = viz.width - 20, plotT = 20, plotB = viz.height - 40;
              const pw = plotR - plotL, ph = plotB - plotT;
              function toP(w, b) {
                return [plotL + (w - wMin)/(wMax - wMin)*pw, plotB - (b - bMin)/(bMax - bMin)*ph];
              }

              // Contour plot
              const step = 4;
              let minNLL = Infinity, maxNLL = -Infinity;
              const grid = [];
              for (let px = plotL; px < plotR; px += step) {
                const row = [];
                for (let py = plotT; py < plotB; py += step) {
                  const w = wMin + (px - plotL)/pw * (wMax - wMin);
                  const b = bMin + (plotB - py)/ph * (bMax - bMin);
                  const v = nll(w, b);
                  row.push(v);
                  if (v < minNLL) minNLL = v;
                  if (v > maxNLL) maxNLL = v;
                }
                grid.push(row);
              }
              maxNLL = Math.min(maxNLL, minNLL + 30);

              let ci = 0;
              for (let px = plotL; px < plotR; px += step) {
                let cj = 0;
                for (let py = plotT; py < plotB; py += step) {
                  const v = grid[ci][cj];
                  const t = Math.min(1, (v - minNLL) / (maxNLL - minNLL));
                  const r = Math.round(12 + t * 30);
                  const g = Math.round(12 + (1-t) * 60);
                  const bl = Math.round(32 + (1-t) * 100);
                  ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + bl + ')';
                  ctx.fillRect(px, py, step, step);
                  cj++;
                }
                ci++;
              }

              // Contour lines
              const levels = [minNLL + 1, minNLL + 2, minNLL + 4, minNLL + 8, minNLL + 15];
              ctx.strokeStyle = '#ffffff22'; ctx.lineWidth = 0.8;
              for (const lev of levels) {
                for (let px = plotL; px < plotR - step; px += step) {
                  const colI = (px - plotL) / step;
                  for (let py = plotT; py < plotB - step; py += step) {
                    const rowI = (py - plotT) / step;
                    if (colI >= grid.length - 1 || rowI >= grid[0].length - 1) continue;
                    const v00 = grid[colI][rowI], v10 = grid[colI+1]?.[rowI];
                    if (v00 === undefined || v10 === undefined) continue;
                    if ((v00 - lev) * (v10 - lev) < 0) {
                      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px+step, py); ctx.stroke();
                    }
                    const v01 = grid[colI][rowI+1];
                    if (v01 !== undefined && (v00 - lev) * (v01 - lev) < 0) {
                      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, py+step); ctx.stroke();
                    }
                  }
                }
              }

              // Axes
              ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
              ctx.beginPath(); ctx.moveTo(plotL, plotB); ctx.lineTo(plotR, plotB); ctx.stroke();
              ctx.beginPath(); ctx.moveTo(plotL, plotB); ctx.lineTo(plotL, plotT); ctx.stroke();

              ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              for (let w = -4; w <= 6; w += 2) {
                const [px] = toP(w, bMin);
                ctx.fillText(w.toString(), px, plotB + 4);
              }
              ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              for (let b = -4; b <= 4; b += 2) {
                const [, py] = toP(wMin, b);
                ctx.fillText(b.toString(), plotL - 4, py);
              }

              ctx.fillStyle = viz.colors.white; ctx.font = '13px -apple-system,sans-serif';
              ctx.textAlign = 'center'; ctx.fillText('w', (plotL+plotR)/2, plotB + 22);
              ctx.save(); ctx.translate(12, (plotT+plotB)/2); ctx.rotate(-Math.PI/2);
              ctx.textAlign = 'center'; ctx.fillText('b', 0, 0); ctx.restore();

              // GD trajectory
              const showN = Math.min(animStep, trajectory.length);
              if (showN > 1) {
                ctx.strokeStyle = viz.colors.green; ctx.lineWidth = 2;
                ctx.beginPath();
                for (let i = 0; i < showN; i++) {
                  const [px, py] = toP(trajectory[i].w, trajectory[i].b);
                  i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
                }
                ctx.stroke();
              }
              if (showN > 0) {
                const last = trajectory[showN - 1];
                const [px, py] = toP(last.w, last.b);
                ctx.fillStyle = viz.colors.green;
                ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI*2); ctx.fill();
                viz.screenText('NLL = ' + last.nll.toFixed(2) + '  w = ' + last.w.toFixed(2) + '  b = ' + last.b.toFixed(2), viz.width/2, viz.height - 10, viz.colors.green, 11);
              }

              // Start marker
              const [spx, spy] = toP(trajectory[0].w, trajectory[0].b);
              ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 2;
              ctx.beginPath(); ctx.arc(spx, spy, 7, 0, Math.PI*2); ctx.stroke();
            }

            animStep = trajectory.length;
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Write the negative log-likelihood for a single data point \\((\\mathbf{x}, y=1)\\) and simplify. What happens to this loss as the model becomes more confident (\\(\\sigma(z) \\to 1\\))?',
          hint: 'Substitute \\(y=1\\) into the general NLL formula.',
          solution: 'For \\(y=1\\): \\(\\text{NLL} = -\\ln \\sigma(z) = -\\ln \\frac{1}{1+e^{-z}} = \\ln(1+e^{-z})\\). As \\(z \\to +\\infty\\), \\(\\sigma(z) \\to 1\\) and \\(\\text{NLL} \\to 0\\): no loss for a correct, confident prediction. As \\(z \\to -\\infty\\), \\(\\sigma(z) \\to 0\\) and \\(\\text{NLL} \\to +\\infty\\): infinite penalty for a confidently wrong prediction.'
        },
        {
          question: 'Show that the NLL can be rewritten as \\(\\text{NLL} = \\sum_i [\\ln(1 + e^{z_i}) - y_i z_i]\\). (This form is numerically more stable.)',
          hint: 'Use \\(\\ln \\sigma(z) = z - \\ln(1+e^z)\\) and \\(\\ln(1-\\sigma(z)) = -\\ln(1+e^z)\\).',
          solution: 'Starting from \\(-[y \\ln \\sigma + (1-y) \\ln(1-\\sigma)]\\): note \\(\\ln \\sigma(z) = -\\ln(1+e^{-z}) = z - \\ln(1+e^z)\\) and \\(\\ln(1-\\sigma(z)) = \\ln(e^{-z}/(1+e^{-z})) = -z - \\ln(1+e^{-z}) = -\\ln(1+e^z)\\). Substituting: \\(-[y(z - \\ln(1+e^z)) + (1-y)(-\\ln(1+e^z))] = -[yz - y\\ln(1+e^z) - \\ln(1+e^z) + y\\ln(1+e^z)] = -yz + \\ln(1+e^z) = \\ln(1+e^z) - yz\\).'
        },
        {
          question: 'Explain why the convexity of the NLL is important for practical optimization. What would happen if the loss surface had multiple local minima?',
          hint: 'Consider what gradient descent guarantees for convex vs. non-convex functions.',
          solution: 'Convexity guarantees that every local minimum is a global minimum, so any gradient-based optimizer will find the best possible parameters regardless of initialization. If the NLL were non-convex (as in neural networks), gradient descent could get stuck in suboptimal local minima or saddle points, and the solution would depend on initialization. Convexity also ensures that the Hessian is positive semi-definite, enabling second-order methods like Newton\'s method to converge quickly.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 3: Gradient Descent for Logistic Regression
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch05-sec03',
      title: 'Gradient Descent for Logistic Regression',
      content: `
<h2>Gradient Descent for Logistic Regression</h2>

<p>Since the NLL has no closed-form minimum, we use <strong>gradient descent</strong> to iteratively move toward the optimum. The key step is deriving the gradient.</p>

<h3>Gradient Derivation</h3>

<div class="env-block theorem"><div class="env-title">Theorem 5.2 (Gradient of the NLL)</div><div class="env-body">
<p>The gradient of the negative log-likelihood with respect to the parameters is</p>
\\[\\frac{\\partial \\text{NLL}}{\\partial \\mathbf{w}} = \\sum_{i=1}^n (\\sigma(z_i) - y_i)\\,\\mathbf{x}_i, \\quad \\frac{\\partial \\text{NLL}}{\\partial b} = \\sum_{i=1}^n (\\sigma(z_i) - y_i),\\]
<p>where \\(z_i = \\mathbf{w}^\\top \\mathbf{x}_i + b\\). In matrix form: \\(\\nabla_\\mathbf{w} \\text{NLL} = \\mathbf{X}^\\top(\\boldsymbol{\\sigma} - \\mathbf{y})\\).</p>
</div></div>

<div class="env-block proof"><div class="env-title">Proof</div><div class="env-body">
<p>Consider one term of the NLL: \\(L_i = -y_i \\ln \\sigma(z_i) - (1-y_i)\\ln(1-\\sigma(z_i))\\). Using \\(\\sigma' = \\sigma(1-\\sigma)\\):</p>
\\[\\frac{\\partial L_i}{\\partial z_i} = -y_i \\frac{\\sigma'(z_i)}{\\sigma(z_i)} + (1-y_i)\\frac{\\sigma'(z_i)}{1-\\sigma(z_i)} = -y_i(1-\\sigma(z_i)) + (1-y_i)\\sigma(z_i) = \\sigma(z_i) - y_i.\\]
<p>By the chain rule, \\(\\frac{\\partial L_i}{\\partial \\mathbf{w}} = (\\sigma(z_i) - y_i)\\mathbf{x}_i\\) and \\(\\frac{\\partial L_i}{\\partial b} = \\sigma(z_i) - y_i\\). Summing over all \\(i\\) gives the result.</p>
<div class="qed">&#8718;</div>
</div></div>

<div class="env-block intuition"><div class="env-title">Interpreting the Gradient</div><div class="env-body"><p>The gradient has a beautiful form: the error \\(\\sigma(z_i) - y_i\\) is the difference between the predicted probability and the true label. Each data point "pulls" the weight vector proportionally to this error, scaled by its feature vector \\(\\mathbf{x}_i\\). Points that are correctly classified with high confidence contribute almost zero gradient; misclassified points or uncertain points drive the updates.</p></div></div>

<h3>The Update Rule</h3>

<div class="env-block definition"><div class="env-title">Algorithm 5.1 (Gradient Descent for Logistic Regression)</div><div class="env-body">
<p>Initialize \\(\\mathbf{w} = \\mathbf{0}\\), \\(b = 0\\). Repeat until convergence:</p>
\\[\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta \\sum_{i=1}^n (\\sigma(z_i) - y_i)\\,\\mathbf{x}_i, \\quad b \\leftarrow b - \\eta \\sum_{i=1}^n (\\sigma(z_i) - y_i),\\]
<p>where \\(\\eta &gt; 0\\) is the learning rate.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Comparison with Perceptron</div><div class="env-body"><p>The perceptron update is \\(\\mathbf{w} \\leftarrow \\mathbf{w} + y_i \\mathbf{x}_i\\) for misclassified points only (\\(y_i \\in \\{+1,-1\\}\\)). The logistic regression update is \\(\\mathbf{w} \\leftarrow \\mathbf{w} - (\\sigma(z_i) - y_i)\\mathbf{x}_i\\) for <em>all</em> points. The key difference is that logistic regression uses the "soft" error \\(\\sigma(z_i) - y_i \\in (-1, 1)\\), giving a smoother, differentiable objective.</p></div></div>

<div class="viz-placeholder" data-viz="viz-gd-boundary"></div>
`,
      visualizations: [
        {
          id: 'viz-gd-boundary',
          title: 'Decision Boundary During Gradient Descent',
          description: 'Watch the decision boundary evolve as gradient descent optimizes the logistic regression parameters. The boundary rotates and shifts until it best separates the two classes. Use the learning rate slider to see how step size affects convergence.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 45 });
            const ctx = viz.ctx;

            function sigmoid(z) { return 1/(1+Math.exp(-Math.max(-20,Math.min(20,z)))); }
            function randn() { let u=0,v=0; while(!u) u=Math.random(); while(!v) v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

            const data = [];
            for (let i = 0; i < 15; i++) {
              data.push({x: 1.5+randn()*0.8, y: 1.5+randn()*0.8, c: 1});
              data.push({x: -1.5+randn()*0.8, y: -1.0+randn()*0.8, c: 0});
            }

            let w1 = 0, w2 = 0, b = 0;
            let lr = 0.1, iteration = 0;
            let running = false, animId = null;

            const lrSlider = VizEngine.createSlider(controls, 'Learning rate', 0.01, 0.5, lr, 0.01, v => { lr = v; });

            VizEngine.createButton(controls, 'Step', () => { running = false; gdStep(); draw(); });
            VizEngine.createButton(controls, 'Run', () => {
              if (running) return;
              running = true;
              (function tick() {
                if (!running) return;
                gdStep(); draw();
                animId = setTimeout(tick, 60);
              })();
            });
            VizEngine.createButton(controls, 'Stop', () => { running = false; if (animId) clearTimeout(animId); });
            VizEngine.createButton(controls, 'Reset', () => {
              running = false; if (animId) clearTimeout(animId);
              w1 = 0; w2 = 0; b = 0; iteration = 0; draw();
            });

            function gdStep() {
              let gw1 = 0, gw2 = 0, gb = 0;
              for (const p of data) {
                const z = w1*p.x + w2*p.y + b;
                const s = sigmoid(z);
                const err = s - p.c;
                gw1 += err * p.x;
                gw2 += err * p.y;
                gb += err;
              }
              w1 -= lr * gw1;
              w2 -= lr * gw2;
              b -= lr * gb;
              iteration++;
            }

            function draw() {
              viz.clear(); viz.drawGrid(); viz.drawAxes();
              const norm = Math.sqrt(w1*w1 + w2*w2);

              if (norm > 0.01) {
                const perpX = -w2/norm, perpY = w1/norm;
                const cx = -b*w1/(w1*w1+w2*w2), cy = -b*w2/(w1*w1+w2*w2);

                // Shade probability regions
                const regionStep = 6;
                for (let px = 0; px < viz.width; px += regionStep) {
                  for (let py = 0; py < viz.height; py += regionStep) {
                    const [mx, my] = viz.toMath(px, py);
                    const prob = sigmoid(w1*mx + w2*my + b);
                    if (prob > 0.5) {
                      ctx.fillStyle = 'rgba(88,166,255,' + ((prob-0.5)*0.2).toFixed(3) + ')';
                    } else {
                      ctx.fillStyle = 'rgba(248,81,73,' + ((0.5-prob)*0.2).toFixed(3) + ')';
                    }
                    ctx.fillRect(px, py, regionStep, regionStep);
                  }
                }

                viz.drawGrid(); viz.drawAxes();
                viz.drawLine(cx-perpX*12, cy-perpY*12, cx+perpX*12, cy+perpY*12, viz.colors.green, 2.5);
              }

              // Data points
              let nll = 0;
              for (const p of data) {
                const z = w1*p.x + w2*p.y + b;
                const s = sigmoid(z);
                nll -= p.c * Math.log(s+1e-15) + (1-p.c) * Math.log(1-s+1e-15);
                const col = p.c === 1 ? viz.colors.blue : viz.colors.red;
                const [sx, sy] = viz.toScreen(p.x, p.y);
                ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI*2);
                ctx.fillStyle = col; ctx.fill();
              }

              const acc = data.filter(p => {
                const s = sigmoid(w1*p.x + w2*p.y + b);
                return (s >= 0.5 ? 1 : 0) === p.c;
              }).length;

              viz.screenText('Iter: ' + iteration + '  NLL: ' + nll.toFixed(2) + '  Acc: ' + acc + '/' + data.length, viz.width/2, viz.height-14, viz.colors.text, 12);
              viz.screenText('w = (' + w1.toFixed(2) + ', ' + w2.toFixed(2) + ')  b = ' + b.toFixed(2), viz.width/2, 16, viz.colors.teal, 12);
            }
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'For a single data point \\((\\mathbf{x}, y=1)\\) with current prediction \\(\\sigma(z) = 0.3\\), compute the gradient update direction for \\(\\mathbf{w}\\). Is the weight vector pushed toward or away from \\(\\mathbf{x}\\)?',
          hint: 'The gradient contribution is \\((\\sigma(z) - y)\\mathbf{x}\\). The update subtracts this from \\(\\mathbf{w}\\).',
          solution: 'The gradient is \\((0.3 - 1)\\mathbf{x} = -0.7\\mathbf{x}\\). The update is \\(\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta(-0.7)\\mathbf{x} = \\mathbf{w} + 0.7\\eta\\mathbf{x}\\). The weight is pushed <em>toward</em> \\(\\mathbf{x}\\), increasing the score \\(z = \\mathbf{w}^\\top \\mathbf{x} + b\\), which raises \\(\\sigma(z)\\) toward the true label 1.'
        },
        {
          question: 'Show that the Hessian of the NLL is \\(\\mathbf{H} = \\mathbf{X}^\\top \\mathbf{D} \\mathbf{X}\\) where \\(\\mathbf{D} = \\text{diag}(\\sigma(z_i)(1-\\sigma(z_i)))\\). Explain why this confirms convexity.',
          hint: 'Differentiate the gradient \\(\\mathbf{X}^\\top(\\boldsymbol{\\sigma} - \\mathbf{y})\\) with respect to \\(\\mathbf{w}\\) again.',
          solution: 'The gradient is \\(\\nabla = \\mathbf{X}^\\top(\\boldsymbol{\\sigma} - \\mathbf{y})\\). Since \\(\\partial \\sigma(z_i)/\\partial \\mathbf{w} = \\sigma(z_i)(1-\\sigma(z_i))\\mathbf{x}_i^\\top\\), the Hessian is \\(\\mathbf{H} = \\sum_i \\sigma_i(1-\\sigma_i)\\mathbf{x}_i\\mathbf{x}_i^\\top = \\mathbf{X}^\\top \\mathbf{D} \\mathbf{X}\\). Since \\(0 &lt; \\sigma_i(1-\\sigma_i) \\leq 0.25\\), \\(\\mathbf{D}\\) is positive definite. For any \\(\\mathbf{v} \\neq 0\\): \\(\\mathbf{v}^\\top \\mathbf{H} \\mathbf{v} = \\sum_i d_i(\\mathbf{x}_i^\\top \\mathbf{v})^2 \\geq 0\\), confirming positive semi-definiteness and hence convexity.'
        },
        {
          question: 'In stochastic gradient descent (SGD), we update using one random data point at a time instead of the full sum. Write the SGD update for logistic regression and explain its computational advantage.',
          hint: 'Replace the sum over all \\(i\\) with a single randomly chosen sample.',
          solution: 'Pick a random index \\(i\\). The SGD update is \\(\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta(\\sigma(z_i) - y_i)\\mathbf{x}_i\\), \\(b \\leftarrow b - \\eta(\\sigma(z_i) - y_i)\\). Each update costs \\(O(d)\\) instead of \\(O(nd)\\) for full-batch GD. For large datasets, SGD can make many fast, noisy updates in the time batch GD makes one precise update, often converging faster in practice. The noise also helps escape shallow local minima in non-convex settings (though logistic regression is convex).'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 4: Regularized Logistic Regression
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch05-sec04',
      title: 'Regularized Logistic Regression',
      content: `
<h2>Regularized Logistic Regression</h2>

<p>When the data is linearly separable, the maximum likelihood solution for logistic regression sends \\(\\|\\mathbf{w}\\| \\to \\infty\\): the sigmoid becomes a step function, producing probability estimates of exactly 0 or 1. This is a form of <strong>overfitting</strong>. Regularization prevents this by penalizing large weights.</p>

<div class="env-block definition"><div class="env-title">Definition 5.4 (L2-Regularized Logistic Regression)</div><div class="env-body">
<p>The regularized objective adds an L2 penalty to the NLL:</p>
\\[\\mathcal{J}(\\boldsymbol{\\theta}) = \\text{NLL}(\\boldsymbol{\\theta}) + \\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2 = -\\sum_{i=1}^n \\left[y_i \\ln \\sigma(z_i) + (1-y_i) \\ln(1-\\sigma(z_i))\\right] + \\frac{\\lambda}{2}\\sum_{j=1}^d w_j^2,\\]
<p>where \\(\\lambda &gt; 0\\) is the <strong>regularization strength</strong>. The bias \\(b\\) is typically not regularized.</p>
</div></div>

<h3>Effect of Regularization</h3>

<p>The penalty \\(\\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2\\) pulls the weights toward zero. Larger \\(\\lambda\\) produces a simpler model with a smoother decision boundary; smaller \\(\\lambda\\) allows the model to fit the data more closely at the risk of overfitting.</p>

<div class="env-block theorem"><div class="env-title">Theorem 5.3 (Regularized Gradient)</div><div class="env-body">
<p>The gradient of the regularized objective is</p>
\\[\\frac{\\partial \\mathcal{J}}{\\partial \\mathbf{w}} = \\sum_{i=1}^n (\\sigma(z_i) - y_i)\\mathbf{x}_i + \\lambda \\mathbf{w}, \\quad \\frac{\\partial \\mathcal{J}}{\\partial b} = \\sum_{i=1}^n (\\sigma(z_i) - y_i).\\]
<p>The regularization term adds a "weight decay" of \\(\\lambda \\mathbf{w}\\) to the gradient.</p>
</div></div>

<div class="env-block intuition"><div class="env-title">Bayesian Interpretation</div><div class="env-body"><p>L2 regularization is equivalent to placing a <strong>Gaussian prior</strong> \\(p(\\mathbf{w}) = \\mathcal{N}(\\mathbf{0}, \\lambda^{-1}\\mathbf{I})\\) on the weights and performing <strong>MAP estimation</strong>. Minimizing \\(\\text{NLL} + \\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2\\) is the same as maximizing \\(\\ln p(\\mathbf{y} \\mid \\mathbf{X}, \\mathbf{w}) + \\ln p(\\mathbf{w})\\). The regularization strength \\(\\lambda\\) reflects how strongly we believe the weights should be small before seeing the data.</p></div></div>

<div class="viz-placeholder" data-viz="viz-regularization-effect"></div>

<h3>Multi-class: Softmax Regression</h3>

<p>Logistic regression generalizes to \\(K\\) classes via the <strong>softmax function</strong>:</p>
\\[p(y = k \\mid \\mathbf{x}) = \\frac{\\exp(\\mathbf{w}_k^\\top \\mathbf{x} + b_k)}{\\sum_{j=1}^K \\exp(\\mathbf{w}_j^\\top \\mathbf{x} + b_j)}.\\]
<p>The loss becomes the <strong>categorical cross-entropy</strong>:</p>
\\[\\text{NLL} = -\\sum_{i=1}^n \\ln p(y = y_i \\mid \\mathbf{x}_i) = -\\sum_{i=1}^n \\left[\\mathbf{w}_{y_i}^\\top \\mathbf{x}_i + b_{y_i} - \\ln \\sum_{j=1}^K \\exp(\\mathbf{w}_j^\\top \\mathbf{x}_i + b_j)\\right].\\]

<div class="env-block remark"><div class="env-title">Remark</div><div class="env-body"><p>For \\(K = 2\\), softmax regression reduces to binary logistic regression. Setting \\(\\mathbf{w}_1 = \\mathbf{0}\\) (reference class), we get \\(p(y=2 \\mid \\mathbf{x}) = \\sigma(\\mathbf{w}_2^\\top \\mathbf{x} + b_2)\\). The two formulations are equivalent up to a re-parameterization.</p></div></div>

<div class="env-block example"><div class="env-title">Example 5.2 (Choosing \\(\\lambda\\))</div><div class="env-body">
<p>In practice, \\(\\lambda\\) is chosen by <strong>cross-validation</strong>. A common search range is \\(\\lambda \\in \\{10^{-4}, 10^{-3}, \\dots, 10^2\\}\\) on a logarithmic scale. Too small \\(\\lambda\\) overfits; too large \\(\\lambda\\) underfits (decision boundary becomes nearly flat). The optimal \\(\\lambda\\) balances model complexity and data fit.</p>
</div></div>
`,
      visualizations: [
        {
          id: 'viz-regularization-effect',
          title: 'Effect of Regularization on Decision Boundary',
          description: 'Adjust the regularization strength lambda. With lambda = 0 the boundary fits the training data tightly (and weights may be large). Increasing lambda pushes weights toward zero, smoothing the boundary. The right panel shows the weight magnitudes.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 45 });
            const ctx = viz.ctx;

            function sigmoid(z) { return 1/(1+Math.exp(-Math.max(-20,Math.min(20,z)))); }
            function randn() { let u=0,v=0; while(!u) u=Math.random(); while(!v) v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

            const data = [];
            for (let i = 0; i < 12; i++) {
              data.push({x: 1.2+randn()*0.7, y: 1.5+randn()*0.7, c: 1});
              data.push({x: -1.2+randn()*0.7, y: -1.0+randn()*0.7, c: 0});
            }
            // Add a few "tricky" points
            data.push({x:0.1, y:0.2, c:1});
            data.push({x:-0.1, y:-0.3, c:0});

            let lambda = 0;

            function fitModel(lam) {
              let w1 = 0, w2 = 0, b = 0;
              const lr = 0.05;
              for (let iter = 0; iter < 300; iter++) {
                let gw1 = 0, gw2 = 0, gb = 0;
                for (const p of data) {
                  const z = w1*p.x + w2*p.y + b;
                  const s = sigmoid(z);
                  const err = s - p.c;
                  gw1 += err * p.x;
                  gw2 += err * p.y;
                  gb += err;
                }
                gw1 += lam * w1;
                gw2 += lam * w2;
                w1 -= lr * gw1;
                w2 -= lr * gw2;
                b -= lr * gb;
              }
              return {w1, w2, b};
            }

            VizEngine.createSlider(controls, '\u03bb', 0, 10, lambda, 0.1, v => { lambda = v; draw(); });

            function draw() {
              viz.clear();

              const {w1, w2, b: bias} = fitModel(lambda);
              const norm = Math.sqrt(w1*w1 + w2*w2);

              // Shade probability regions
              const regionStep = 5;
              for (let px = 0; px < viz.width - 100; px += regionStep) {
                for (let py = 0; py < viz.height; py += regionStep) {
                  const [mx, my] = viz.toMath(px, py);
                  const prob = sigmoid(w1*mx + w2*my + bias);
                  if (prob > 0.5) {
                    ctx.fillStyle = 'rgba(88,166,255,' + ((prob-0.5)*0.2).toFixed(3) + ')';
                  } else {
                    ctx.fillStyle = 'rgba(248,81,73,' + ((0.5-prob)*0.2).toFixed(3) + ')';
                  }
                  ctx.fillRect(px, py, regionStep, regionStep);
                }
              }

              viz.drawGrid(); viz.drawAxes();

              // Decision boundary
              if (norm > 0.01) {
                const perpX = -w2/norm, perpY = w1/norm;
                const cx = -bias*w1/(w1*w1+w2*w2), cy = -bias*w2/(w1*w1+w2*w2);
                viz.drawLine(cx-perpX*12, cy-perpY*12, cx+perpX*12, cy+perpY*12, viz.colors.green, 2.5);
              }

              // Data
              let acc = 0;
              for (const p of data) {
                const s = sigmoid(w1*p.x + w2*p.y + bias);
                if ((s >= 0.5 ? 1 : 0) === p.c) acc++;
                const col = p.c === 1 ? viz.colors.blue : viz.colors.red;
                const [sx, sy] = viz.toScreen(p.x, p.y);
                ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI*2);
                ctx.fillStyle = col; ctx.fill();
              }

              // Weight magnitude panel on right
              const panelX = viz.width - 90;
              ctx.fillStyle = 'rgba(12,12,32,0.9)';
              ctx.fillRect(panelX, 10, 80, 130);
              ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 1;
              ctx.strokeRect(panelX, 10, 80, 130);

              ctx.fillStyle = viz.colors.text; ctx.font = 'bold 11px -apple-system,sans-serif';
              ctx.textAlign = 'center'; ctx.fillText('Weights', panelX+40, 26);

              const maxBar = 55;
              const maxW = Math.max(Math.abs(w1), Math.abs(w2), 1);

              ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
              ctx.textAlign = 'left';
              ctx.fillText('w\u2081', panelX+5, 50);
              const bar1 = Math.abs(w1)/maxW * maxBar;
              ctx.fillStyle = viz.colors.blue; ctx.fillRect(panelX+22, 42, bar1, 12);
              ctx.fillStyle = viz.colors.text; ctx.fillText(w1.toFixed(2), panelX+25+bar1, 52);

              ctx.fillStyle = viz.colors.text;
              ctx.fillText('w\u2082', panelX+5, 75);
              const bar2 = Math.abs(w2)/maxW * maxBar;
              ctx.fillStyle = viz.colors.teal; ctx.fillRect(panelX+22, 67, bar2, 12);
              ctx.fillStyle = viz.colors.text; ctx.fillText(w2.toFixed(2), panelX+25+bar2, 77);

              ctx.fillStyle = viz.colors.text;
              ctx.fillText('b', panelX+5, 100);
              ctx.fillText(bias.toFixed(2), panelX+22, 100);

              ctx.fillStyle = viz.colors.text;
              ctx.fillText('||w||', panelX+5, 125);
              ctx.fillText(norm.toFixed(2), panelX+35, 125);

              viz.screenText('\u03bb = ' + lambda.toFixed(1) + '  Acc: ' + acc + '/' + data.length, (viz.width-100)/2, viz.height-14, viz.colors.text, 12);
            }
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Why does the unregularized logistic regression solution diverge (\\(\\|\\mathbf{w}\\| \\to \\infty\\)) when the data is linearly separable?',
          hint: 'Consider what happens to the NLL as the decision boundary becomes a hard step function.',
          solution: 'If the data is linearly separable, there exists \\(\\mathbf{w}^*\\) such that \\(y_i(\\mathbf{w}^{*\\top}\\mathbf{x}_i + b) &gt; 0\\) for all \\(i\\). Scaling \\(\\mathbf{w} = c\\mathbf{w}^*\\) with \\(c \\to \\infty\\) makes \\(\\sigma(z_i) \\to y_i\\) for every point (the sigmoid approaches a step function). The NLL approaches 0 from above but never reaches 0, so \\(c\\) must go to infinity. L2 regularization prevents this because the penalty \\(\\frac{\\lambda}{2}c^2\\|\\mathbf{w}^*\\|^2\\) grows, creating a finite optimum.'
        },
        {
          question: 'Show that the gradient of the regularized objective equals the unregularized gradient plus \\(\\lambda \\mathbf{w}\\). Explain why this is called "weight decay" in gradient descent.',
          hint: 'Differentiate \\(\\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2\\) with respect to \\(\\mathbf{w}\\).',
          solution: '\\(\\nabla_\\mathbf{w} \\frac{\\lambda}{2}\\|\\mathbf{w}\\|^2 = \\lambda \\mathbf{w}\\). So the total gradient is \\(\\nabla_{\\text{NLL}} + \\lambda \\mathbf{w}\\). The update becomes \\(\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta(\\nabla_{\\text{NLL}} + \\lambda \\mathbf{w}) = (1-\\eta\\lambda)\\mathbf{w} - \\eta \\nabla_{\\text{NLL}}\\). The factor \\((1-\\eta\\lambda) &lt; 1\\) multiplies the current weight at every step, causing the weight to "decay" toward zero. This is why L2 regularization in gradient descent is called weight decay.'
        },
        {
          question: 'For \\(K = 3\\) classes, write the softmax probability \\(p(y=2 \\mid \\mathbf{x})\\) and compute the gradient of \\(-\\ln p(y=2 \\mid \\mathbf{x})\\) with respect to \\(\\mathbf{w}_1\\).',
          hint: 'The softmax is \\(p(k) = e^{s_k}/\\sum_j e^{s_j}\\). Use the quotient rule or note that \\(\\partial \\ln \\sum_j e^{s_j} / \\partial s_1 = p(1)\\).',
          solution: '\\(p(y=2 \\mid \\mathbf{x}) = \\frac{e^{s_2}}{e^{s_1}+e^{s_2}+e^{s_3}}\\). The loss is \\(-\\ln p(2) = -s_2 + \\ln(e^{s_1}+e^{s_2}+e^{s_3})\\). Differentiating with respect to \\(\\mathbf{w}_1\\): \\(\\frac{\\partial}{\\partial \\mathbf{w}_1}(-s_2 + \\ln \\sum_j e^{s_j}) = 0 + \\frac{e^{s_1}}{\\sum_j e^{s_j}}\\mathbf{x} = p(1 \\mid \\mathbf{x})\\mathbf{x}\\). The gradient pushes \\(\\mathbf{w}_1\\) away from \\(\\mathbf{x}\\) in proportion to the (incorrect) probability assigned to class 1.'
        }
      ]
    }
  ]
});
