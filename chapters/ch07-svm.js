window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
  id: 'ch07',
  number: 7,
  title: 'Support Vector Machines',
  subtitle: 'Maximum Margins, Kernels, and the Geometry of Classification',
  sections: [

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 1: Maximum Margin Classifier
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch07-sec01',
      title: 'Maximum Margin Classifier',
      content: `
<h2>Maximum Margin Classifier</h2>

<p>In Chapter 4 we saw that a linear classifier separates data with a hyperplane \\(\\mathbf{w}^\\top\\mathbf{x} + b = 0\\). When the data is linearly separable, there are infinitely many hyperplanes that achieve zero training error. The <strong>maximum margin classifier</strong> selects the one that maximizes the <em>margin</em>, the distance between the decision boundary and the nearest data point from either class.</p>

<div class="env-block intuition"><div class="env-title">Why Maximize the Margin?</div><div class="env-body"><p>A hyperplane with a large margin is intuitively more robust: small perturbations of the data points are less likely to cause misclassifications. Statistical learning theory formalizes this intuition by showing that the generalization error bound depends inversely on the margin, providing a theoretical justification for margin maximization.</p></div></div>

<div class="env-block definition"><div class="env-title">Definition 7.1 (Functional and Geometric Margin)</div><div class="env-body">
<p>Given a hyperplane defined by \\((\\mathbf{w}, b)\\) and a labeled point \\((\\mathbf{x}_i, y_i)\\) with \\(y_i \\in \\{-1, +1\\}\\):</p>
<ul>
  <li>The <strong>functional margin</strong> is \\(\\hat{\\gamma}_i = y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b)\\). It is positive iff the point is correctly classified.</li>
  <li>The <strong>geometric margin</strong> is \\(\\gamma_i = \\frac{y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b)}{\\|\\mathbf{w}\\|}\\), the signed perpendicular distance from \\(\\mathbf{x}_i\\) to the hyperplane.</li>
</ul>
</div></div>

<p>The <strong>margin</strong> of the classifier on the training set is the minimum geometric margin:</p>
\\[\\gamma = \\min_{i=1,\\dots,n} \\frac{y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b)}{\\|\\mathbf{w}\\|}.\\]

<div class="env-block definition"><div class="env-title">Definition 7.2 (Hard-Margin SVM)</div><div class="env-body">
<p>The <strong>hard-margin support vector machine</strong> solves</p>
\\[\\max_{\\mathbf{w}, b} \\; \\frac{2}{\\|\\mathbf{w}\\|} \\quad \\text{subject to} \\quad y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) \\geq 1, \\quad i = 1, \\dots, n.\\]
<p>Equivalently (by minimizing \\(\\|\\mathbf{w}\\|^2\\)):</p>
\\[\\min_{\\mathbf{w}, b} \\; \\frac{1}{2}\\|\\mathbf{w}\\|^2 \\quad \\text{subject to} \\quad y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) \\geq 1, \\quad \\forall i.\\]
</div></div>

<div class="env-block remark"><div class="env-title">Why the Constraint is 1</div><div class="env-body"><p>The scaling \\(y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) \\geq 1\\) (rather than \\(\\geq \\gamma\\) for some variable \\(\\gamma\\)) is a normalization convention. Since \\((\\mathbf{w}, b)\\) and \\((c\\mathbf{w}, cb)\\) define the same hyperplane for any \\(c > 0\\), we can always rescale so that the nearest point has functional margin exactly 1. Under this convention, the geometric margin is \\(1/\\|\\mathbf{w}\\|\\), and the total margin (distance between the two margin hyperplanes) is \\(2/\\|\\mathbf{w}\\|\\).</p></div></div>

<h3>Support Vectors</h3>

<p>The <strong>support vectors</strong> are the training points that lie exactly on the margin boundaries, i.e., those with \\(y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) = 1\\). These are the hardest points to classify and they alone determine the optimal hyperplane. Removing any non-support-vector point from the training set would not change the solution.</p>

<div class="viz-placeholder" data-viz="viz-hard-margin"></div>
`,
      visualizations: [
        {
          id: 'viz-hard-margin',
          title: 'Maximum Margin Classifier',
          description: 'The green line is the maximum margin hyperplane. Support vectors (circled) lie on the dashed margin boundaries. The shaded band is the margin. Drag points to see how the margin and support vectors change.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 55, originX: 280, originY: 220 });

            // Linearly separable data
            const data = [
              { x: -2.5, y: 1.5, label: -1 },
              { x: -1.8, y: -0.5, label: -1 },
              { x: -2.0, y: 0.5, label: -1 },
              { x: -1.2, y: 1.8, label: -1 },
              { x: -3.0, y: -0.2, label: -1 },
              { x: -1.5, y: -1.2, label: -1 },
              { x: 1.5, y: -0.5, label: 1 },
              { x: 2.0, y: 1.0, label: 1 },
              { x: 1.8, y: -1.5, label: 1 },
              { x: 2.5, y: 0.5, label: 1 },
              { x: 1.2, y: 0.8, label: 1 },
              { x: 3.0, y: -0.8, label: 1 }
            ];

            function solveSVM(data) {
              // Simple iterative approach for 2D linearly separable data
              // Find the approximate max-margin hyperplane via projected subgradient
              let w1 = 1, w2 = 0, b = 0;
              const lr = 0.01;
              for (let iter = 0; iter < 2000; iter++) {
                let grad_w1 = w1, grad_w2 = w2, grad_b = 0;
                for (const p of data) {
                  const margin = p.label * (w1 * p.x + w2 * p.y + b);
                  if (margin < 1) {
                    grad_w1 -= 100 * p.label * p.x;
                    grad_w2 -= 100 * p.label * p.y;
                    grad_b -= 100 * p.label;
                  }
                }
                w1 -= lr * grad_w1;
                w2 -= lr * grad_w2;
                b -= lr * grad_b;
              }
              // Normalize so min functional margin = 1
              let minMargin = Infinity;
              for (const p of data) {
                const m = p.label * (w1 * p.x + w2 * p.y + b);
                if (m < minMargin) minMargin = m;
              }
              if (minMargin > 0.01) {
                w1 /= minMargin; w2 /= minMargin; b /= minMargin;
              }
              return { w1, w2, b };
            }

            function draw() {
              viz.clear();
              viz.drawGrid(1);
              viz.drawAxes();
              const ctx = viz.ctx;

              const { w1, w2, b } = solveSVM(data);
              const norm = Math.sqrt(w1 * w1 + w2 * w2);
              const margin = norm > 0.01 ? 1 / norm : 0;

              // Shade margin band
              if (norm > 0.01) {
                const nx = w1 / norm, ny = w2 / norm;

                // Decision boundary: w1*x + w2*y + b = 0
                // Margin boundaries: w1*x + w2*y + b = +/- 1
                for (let offset = -1; offset <= 1; offset++) {
                  let lx1, ly1, lx2, ly2;
                  if (Math.abs(w2) > Math.abs(w1)) {
                    lx1 = -6; ly1 = -(w1 * lx1 + b - offset) / w2;
                    lx2 = 6; ly2 = -(w1 * lx2 + b - offset) / w2;
                  } else {
                    ly1 = -6; lx1 = -(w2 * ly1 + b - offset) / w1;
                    ly2 = 6; lx2 = -(w2 * ly2 + b - offset) / w1;
                  }
                  if (offset === 0) {
                    viz.drawLine(lx1, ly1, lx2, ly2, viz.colors.green, 2.5);
                  } else {
                    viz.drawLine(lx1, ly1, lx2, ly2, viz.colors.green + '66', 1.5, true);
                  }
                }

                // Shade margin region
                const [sx0, sy0] = viz.toScreen(0, 0);
                if (Math.abs(w2) > Math.abs(w1)) {
                  ctx.fillStyle = viz.colors.green + '10';
                  ctx.beginPath();
                  const x1 = -6, x2 = 6;
                  const y1a = -(w1 * x1 + b - 1) / w2, y2a = -(w1 * x2 + b - 1) / w2;
                  const y1b = -(w1 * x1 + b + 1) / w2, y2b = -(w1 * x2 + b + 1) / w2;
                  const [sx1a, sy1a] = viz.toScreen(x1, y1a);
                  const [sx2a, sy2a] = viz.toScreen(x2, y2a);
                  const [sx1b, sy1b] = viz.toScreen(x1, y1b);
                  const [sx2b, sy2b] = viz.toScreen(x2, y2b);
                  ctx.moveTo(sx1a, sy1a); ctx.lineTo(sx2a, sy2a);
                  ctx.lineTo(sx2b, sy2b); ctx.lineTo(sx1b, sy1b);
                  ctx.closePath(); ctx.fill();
                }
              }

              // Draw data points
              for (const p of data) {
                const color = p.label === 1 ? viz.colors.blue : viz.colors.red;
                const funcMargin = p.label * (w1 * p.x + w2 * p.y + b);
                const isSV = Math.abs(funcMargin - 1) < 0.15;

                viz.drawPoint(p.x, p.y, color, null, 5);

                // Highlight support vectors
                if (isSV) {
                  const [sx, sy] = viz.toScreen(p.x, p.y);
                  ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 2;
                  ctx.beginPath(); ctx.arc(sx, sy, 10, 0, Math.PI * 2); ctx.stroke();
                }
              }

              // Info
              viz.screenText('Margin = 2/||w|| = ' + (2 * margin).toFixed(3), viz.width / 2, viz.height - 28, viz.colors.green, 12);
              const svCount = data.filter(p => Math.abs(p.label * (w1 * p.x + w2 * p.y + b) - 1) < 0.15).length;
              viz.screenText('Support vectors: ' + svCount + ' (circled in yellow)', viz.width / 2, viz.height - 12, viz.colors.yellow, 11);
            }

            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Show that the perpendicular distance from a point \\(\\mathbf{x}_0\\) to the hyperplane \\(\\mathbf{w}^\\top\\mathbf{x} + b = 0\\) is \\(|\\mathbf{w}^\\top\\mathbf{x}_0 + b|/\\|\\mathbf{w}\\|\\).',
          hint: 'Project \\(\\mathbf{x}_0\\) onto the normal direction \\(\\mathbf{w}/\\|\\mathbf{w}\\|\\) relative to any point on the hyperplane.',
          solution: 'Let \\(\\mathbf{x}_h\\) be any point on the hyperplane, so \\(\\mathbf{w}^\\top\\mathbf{x}_h + b = 0\\). The distance from \\(\\mathbf{x}_0\\) to the hyperplane is the absolute projection of \\(\\mathbf{x}_0 - \\mathbf{x}_h\\) onto the unit normal: \\(d = \\left|\\frac{\\mathbf{w}}{\\|\\mathbf{w}\\|}\\right|^\\top(\\mathbf{x}_0 - \\mathbf{x}_h)| = \\frac{|\\mathbf{w}^\\top\\mathbf{x}_0 - \\mathbf{w}^\\top\\mathbf{x}_h|}{\\|\\mathbf{w}\\|} = \\frac{|\\mathbf{w}^\\top\\mathbf{x}_0 + b|}{\\|\\mathbf{w}\\|}\\).'
        },
        {
          question: 'Why does the hard-margin SVM require that the data be linearly separable? What happens if it is not?',
          hint: 'Look at the constraints \\(y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) \\geq 1\\).',
          solution: 'If the data is not linearly separable, there is no \\((\\mathbf{w}, b)\\) such that all constraints \\(y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) \\geq 1\\) are simultaneously satisfied. The feasible set is empty, so the optimization problem has no solution. This motivates the soft-margin SVM (next section), which introduces slack variables to allow some constraint violations.'
        },
        {
          question: 'Prove that only support vectors (points on the margin boundary) determine the optimal hyperplane. Specifically, show that removing a non-support-vector does not change the solution.',
          hint: 'Consider the KKT complementarity conditions: \\(\\alpha_i[y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) - 1] = 0\\).',
          solution: 'At optimality, each constraint has a Lagrange multiplier \\(\\alpha_i \\geq 0\\). By complementary slackness, \\(\\alpha_i > 0\\) only if \\(y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) = 1\\) (the point is on the margin). For non-support vectors, \\(y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) > 1\\), so \\(\\alpha_i = 0\\). Since \\(\\mathbf{w} = \\sum_i \\alpha_i y_i \\mathbf{x}_i\\), points with \\(\\alpha_i = 0\\) do not contribute to \\(\\mathbf{w}\\). Removing them leaves the same active constraints and the same solution.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 2: Soft-Margin SVM
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch07-sec02',
      title: 'Soft-Margin SVM',
      content: `
<h2>Soft-Margin SVM</h2>

<p>Real data is rarely linearly separable. Even when it is, a small margin with perfect separation may generalize worse than a larger margin that tolerates a few misclassifications. The <strong>soft-margin SVM</strong> introduces <em>slack variables</em> to allow controlled violations of the margin constraints.</p>

<div class="env-block definition"><div class="env-title">Definition 7.3 (Soft-Margin SVM)</div><div class="env-body">
<p>The <strong>soft-margin SVM</strong> with parameter \\(C > 0\\) solves</p>
\\[\\min_{\\mathbf{w}, b, \\boldsymbol{\\xi}} \\; \\frac{1}{2}\\|\\mathbf{w}\\|^2 + C\\sum_{i=1}^{n} \\xi_i\\]
\\[\\text{subject to} \\quad y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) \\geq 1 - \\xi_i, \\quad \\xi_i \\geq 0, \\quad \\forall i.\\]
</div></div>

<h3>Interpreting Slack Variables</h3>

<p>For each training point \\(i\\):</p>
<ul>
  <li>\\(\\xi_i = 0\\): the point is correctly classified and lies on or outside the margin.</li>
  <li>\\(0 &lt; \\xi_i &lt; 1\\): the point is correctly classified but lies inside the margin.</li>
  <li>\\(\\xi_i = 1\\): the point lies exactly on the decision boundary.</li>
  <li>\\(\\xi_i > 1\\): the point is misclassified.</li>
</ul>

<p>The total slack \\(\\sum_i \\xi_i\\) is an upper bound on the number of misclassifications, so \\(C\\sum_i \\xi_i\\) penalizes misclassifications and margin violations.</p>

<h3>The Role of C</h3>

<div class="env-block intuition"><div class="env-title">C Controls the Bias-Variance Trade-off</div><div class="env-body">
<p>The parameter \\(C\\) balances two competing objectives:</p>
<ul>
  <li><strong>Large C</strong>: Heavy penalty on slack. The SVM tries hard to classify every training point correctly, producing a narrow margin. This has <em>low bias</em> but <em>high variance</em> (overfitting risk).</li>
  <li><strong>Small C</strong>: Light penalty on slack. The SVM allows more violations for a wider margin. This has <em>higher bias</em> but <em>lower variance</em> (better generalization).</li>
</ul>
<p>As \\(C \\to \\infty\\), the soft-margin SVM approaches the hard-margin SVM (if data is separable).</p>
</div></div>

<div class="env-block remark"><div class="env-title">Hinge Loss Interpretation</div><div class="env-body"><p>The soft-margin SVM can be rewritten as an unconstrained optimization over the <strong>hinge loss</strong>: \\(\\min_{\\mathbf{w}, b} \\frac{1}{2}\\|\\mathbf{w}\\|^2 + C\\sum_i \\max(0, 1 - y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b))\\). This is analogous to regularized logistic regression, with the hinge loss \\(\\ell(z) = \\max(0, 1-z)\\) replacing the logistic loss \\(\\ell(z) = \\log(1 + e^{-z})\\). The hinge loss is exactly zero when \\(z \\geq 1\\) (correct and outside margin), which produces the sparsity of support vectors.</p></div></div>

<div class="viz-placeholder" data-viz="viz-soft-margin"></div>
`,
      visualizations: [
        {
          id: 'viz-soft-margin',
          title: 'Soft-Margin SVM: Effect of C',
          description: 'Adjust C to see the trade-off between margin width and training errors. Large C produces a narrow margin with few violations; small C produces a wide margin that tolerates more errors.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 55, originX: 280, originY: 220 });

            let C = 1.0;
            VizEngine.createSlider(controls, 'C', -2, 3, 0, 0.1, v => { C = Math.pow(10, v); draw(); });

            // Data with some overlap
            const data = [
              { x: -2.5, y: 1.2, label: -1 }, { x: -1.5, y: -0.8, label: -1 },
              { x: -2.0, y: 0.3, label: -1 }, { x: -1.0, y: 1.5, label: -1 },
              { x: -3.0, y: -0.5, label: -1 }, { x: -1.8, y: -1.5, label: -1 },
              { x: -0.5, y: 0.2, label: -1 }, // close to boundary
              { x: 0.3, y: -0.5, label: -1 }, // in overlap zone
              { x: 1.5, y: -0.3, label: 1 }, { x: 2.0, y: 1.2, label: 1 },
              { x: 1.8, y: -1.2, label: 1 }, { x: 2.5, y: 0.8, label: 1 },
              { x: 1.0, y: 0.5, label: 1 }, { x: 3.0, y: -0.5, label: 1 },
              { x: 0.5, y: 1.0, label: 1 },  // close to boundary
              { x: -0.2, y: -0.8, label: 1 }  // in overlap zone
            ];

            function solveSoftSVM(data, C) {
              let w1 = 1, w2 = 0, b = 0;
              const lr = 0.005;
              for (let iter = 0; iter < 3000; iter++) {
                let g1 = w1, g2 = w2, gb = 0;
                for (const p of data) {
                  const m = p.label * (w1 * p.x + w2 * p.y + b);
                  if (m < 1) {
                    g1 -= C * p.label * p.x;
                    g2 -= C * p.label * p.y;
                    gb -= C * p.label;
                  }
                }
                w1 -= lr * g1; w2 -= lr * g2; b -= lr * gb;
              }
              return { w1, w2, b };
            }

            function draw() {
              viz.clear();
              viz.drawGrid(1);
              viz.drawAxes();
              const ctx = viz.ctx;

              const { w1, w2, b } = solveSoftSVM(data, C);
              const norm = Math.sqrt(w1 * w1 + w2 * w2);

              if (norm > 0.01) {
                // Draw margin boundaries
                for (let offset = -1; offset <= 1; offset++) {
                  let lx1, ly1, lx2, ly2;
                  if (Math.abs(w2) > Math.abs(w1)) {
                    lx1 = -6; ly1 = -(w1 * lx1 + b - offset) / w2;
                    lx2 = 6; ly2 = -(w1 * lx2 + b - offset) / w2;
                  } else {
                    ly1 = -6; lx1 = -(w2 * ly1 + b - offset) / w1;
                    ly2 = 6; lx2 = -(w2 * ly2 + b - offset) / w1;
                  }
                  if (offset === 0) {
                    viz.drawLine(lx1, ly1, lx2, ly2, viz.colors.green, 2.5);
                  } else {
                    viz.drawLine(lx1, ly1, lx2, ly2, viz.colors.green + '55', 1.5, true);
                  }
                }

                // Shade margin
                if (Math.abs(w2) > Math.abs(w1)) {
                  ctx.fillStyle = viz.colors.green + '08';
                  ctx.beginPath();
                  const xs = [-6, 6];
                  const [sx1a, sy1a] = viz.toScreen(xs[0], -(w1*xs[0]+b-1)/w2);
                  const [sx2a, sy2a] = viz.toScreen(xs[1], -(w1*xs[1]+b-1)/w2);
                  const [sx1b, sy1b] = viz.toScreen(xs[0], -(w1*xs[0]+b+1)/w2);
                  const [sx2b, sy2b] = viz.toScreen(xs[1], -(w1*xs[1]+b+1)/w2);
                  ctx.moveTo(sx1a, sy1a); ctx.lineTo(sx2a, sy2a);
                  ctx.lineTo(sx2b, sy2b); ctx.lineTo(sx1b, sy1b);
                  ctx.closePath(); ctx.fill();
                }
              }

              // Draw data points
              let nViolations = 0, nSV = 0;
              for (const p of data) {
                const color = p.label === 1 ? viz.colors.blue : viz.colors.red;
                const funcMargin = norm > 0.01 ? p.label * (w1 * p.x + w2 * p.y + b) : 1;
                const slack = Math.max(0, 1 - funcMargin);

                viz.drawPoint(p.x, p.y, color, null, 5);

                if (slack > 0.01) {
                  nSV++;
                  const [sx, sy] = viz.toScreen(p.x, p.y);
                  if (slack > 1) {
                    // Misclassified: X mark
                    nViolations++;
                    ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(sx - 8, sy - 8); ctx.lineTo(sx + 8, sy + 8);
                    ctx.moveTo(sx + 8, sy - 8); ctx.lineTo(sx - 8, sy + 8);
                    ctx.stroke();
                  } else {
                    // Inside margin: circle
                    ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 1.5;
                    ctx.beginPath(); ctx.arc(sx, sy, 10, 0, Math.PI * 2); ctx.stroke();
                  }
                } else if (Math.abs(funcMargin - 1) < 0.15) {
                  // On margin boundary
                  nSV++;
                  const [sx, sy] = viz.toScreen(p.x, p.y);
                  ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 2;
                  ctx.beginPath(); ctx.arc(sx, sy, 10, 0, Math.PI * 2); ctx.stroke();
                }
              }

              const marginWidth = norm > 0.01 ? (2 / norm).toFixed(2) : '---';
              viz.screenText('C = ' + C.toFixed(2) + '    Margin = ' + marginWidth, viz.width / 2, viz.height - 28, viz.colors.green, 12);
              viz.screenText('Margin violations: ' + nSV + '    Misclassified: ' + nViolations, viz.width / 2, viz.height - 12, viz.colors.yellow, 11);
            }

            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Show that the hinge loss \\(\\ell(z) = \\max(0, 1 - z)\\) is convex. Is it differentiable everywhere?',
          hint: 'The maximum of two convex functions is convex. Check differentiability at \\(z = 1\\).',
          solution: 'The function \\(\\ell(z) = \\max(0, 1-z)\\) is the pointwise maximum of two linear (hence convex) functions: \\(f_1(z) = 0\\) and \\(f_2(z) = 1 - z\\). The pointwise maximum of convex functions is convex. However, \\(\\ell\\) is not differentiable at \\(z = 1\\): the left derivative is \\(-1\\) and the right derivative is \\(0\\). It does have a well-defined subgradient at every point, which suffices for optimization.'
        },
        {
          question: 'If \\(C = 0\\) in the soft-margin SVM, what happens? What if \\(C \\to \\infty\\)?',
          hint: 'When \\(C = 0\\), the slack penalty vanishes. What does the remaining objective minimize?',
          solution: 'When \\(C = 0\\), the objective is just \\(\\frac{1}{2}\\|\\mathbf{w}\\|^2\\), which is minimized at \\(\\mathbf{w} = \\mathbf{0}\\). This gives a degenerate classifier with infinite margin but no discriminative power (all points are classified the same). When \\(C \\to \\infty\\), any slack \\(\\xi_i > 0\\) incurs infinite cost, forcing \\(\\xi_i = 0\\) for all \\(i\\). This recovers the hard-margin SVM. If the data is not linearly separable, the problem becomes infeasible as \\(C \\to \\infty\\).'
        },
        {
          question: 'Interpret the soft-margin SVM as a regularized empirical risk minimization problem. What is the loss function and what is the regularizer? How does \\(C\\) relate to the regularization strength?',
          hint: 'Rewrite the objective as \\(\\lambda\\|\\mathbf{w}\\|^2 + \\text{training loss}\\) with \\(\\lambda = 1/(2C)\\).',
          solution: 'The soft-margin SVM is equivalent to \\(\\min_{\\mathbf{w},b} \\frac{1}{2}\\|\\mathbf{w}\\|^2 + C\\sum_i \\max(0, 1 - y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b))\\). Dividing by \\(C\\): \\(\\min \\frac{1}{2C}\\|\\mathbf{w}\\|^2 + \\frac{1}{n}\\sum_i \\max(0, 1 - y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b))\\). This is \\(L_2\\)-regularized ERM with hinge loss, where \\(\\lambda = 1/(2C)\\). Large \\(C\\) means weak regularization (small \\(\\lambda\\)); small \\(C\\) means strong regularization (large \\(\\lambda\\)).'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 3: Dual Formulation
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch07-sec03',
      title: 'Dual Formulation',
      content: `
<h2>The Dual Problem</h2>

<p>The SVM primal is a convex quadratic program (QP) with linear constraints. Its <strong>Lagrangian dual</strong> reveals deep structure: the solution depends on the data only through inner products \\(\\mathbf{x}_i^\\top\\mathbf{x}_j\\), which will later enable the kernel trick.</p>

<h3>Deriving the Dual</h3>

<p>The Lagrangian of the hard-margin SVM is</p>
\\[\\mathcal{L}(\\mathbf{w}, b, \\boldsymbol{\\alpha}) = \\frac{1}{2}\\|\\mathbf{w}\\|^2 - \\sum_{i=1}^n \\alpha_i[y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) - 1],\\]
<p>where \\(\\alpha_i \\geq 0\\) are the Lagrange multipliers.</p>

<div class="env-block theorem"><div class="env-title">Theorem 7.1 (SVM Dual)</div><div class="env-body">
<p>Setting \\(\\nabla_{\\mathbf{w}}\\mathcal{L} = 0\\) gives \\(\\mathbf{w} = \\sum_i \\alpha_i y_i \\mathbf{x}_i\\). Setting \\(\\partial\\mathcal{L}/\\partial b = 0\\) gives \\(\\sum_i \\alpha_i y_i = 0\\). Substituting back yields the <strong>dual problem</strong>:</p>
\\[\\max_{\\boldsymbol{\\alpha}} \\; \\sum_{i=1}^n \\alpha_i - \\frac{1}{2}\\sum_{i,j} \\alpha_i \\alpha_j y_i y_j \\, \\mathbf{x}_i^\\top\\mathbf{x}_j\\]
\\[\\text{subject to} \\quad \\alpha_i \\geq 0, \\quad \\sum_i \\alpha_i y_i = 0.\\]
<p>For the soft-margin SVM, the only change is \\(0 \\leq \\alpha_i \\leq C\\).</p>
</div></div>

<h3>KKT Conditions and Support Vectors</h3>

<p>The Karush-Kuhn-Tucker (KKT) conditions provide necessary and sufficient conditions for optimality. The key condition is <strong>complementary slackness</strong>:</p>
\\[\\alpha_i[y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) - 1] = 0, \\quad \\forall i.\\]

<p>This means for each point, either \\(\\alpha_i = 0\\) (the point is irrelevant) or the constraint is <em>active</em> (\\(y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) = 1\\), the point is on the margin). Points with \\(\\alpha_i > 0\\) are the <strong>support vectors</strong>.</p>

<div class="env-block definition"><div class="env-title">Definition 7.4 (Support Vectors via Dual Variables)</div><div class="env-body">
<p>A training point \\((\\mathbf{x}_i, y_i)\\) is a <strong>support vector</strong> if and only if \\(\\alpha_i > 0\\). The optimal weight vector is a linear combination of support vectors only:</p>
\\[\\mathbf{w}^* = \\sum_{i \\in \\text{SV}} \\alpha_i y_i \\mathbf{x}_i.\\]
</div></div>

<div class="env-block remark"><div class="env-title">Sparsity</div><div class="env-body"><p>Typically only a small fraction of training points are support vectors. This sparsity is a hallmark of SVMs and makes prediction efficient: classifying a new point requires computing inner products only with the support vectors, not all \\(n\\) training points.</p></div></div>

<div class="viz-placeholder" data-viz="viz-dual-sv"></div>

<h3>Recovering \\(b\\)</h3>

<p>The bias \\(b\\) is found from any support vector \\(\\mathbf{x}_s\\) with \\(0 &lt; \\alpha_s &lt; C\\) (a free support vector):</p>
\\[b = y_s - \\mathbf{w}^\\top\\mathbf{x}_s = y_s - \\sum_{i \\in \\text{SV}} \\alpha_i y_i \\mathbf{x}_i^\\top\\mathbf{x}_s.\\]
<p>In practice, one averages over all free support vectors for numerical stability.</p>
`,
      visualizations: [
        {
          id: 'viz-dual-sv',
          title: 'Support Vectors and Dual Variables',
          description: 'Points with alpha > 0 are support vectors (highlighted). Drag the red point to see which points become/stop being support vectors as the data changes.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 55, originX: 280, originY: 220 });

            const data = [
              { x: -2.5, y: 1.0, label: -1 },
              { x: -1.5, y: -1.0, label: -1 },
              { x: -2.0, y: 0.0, label: -1 },
              { x: -1.0, y: 1.5, label: -1 },
              { x: -3.0, y: -0.5, label: -1 },
              { x: 1.5, y: -0.5, label: 1 },
              { x: 2.0, y: 1.0, label: 1 },
              { x: 1.8, y: -1.5, label: 1 },
              { x: 2.5, y: 0.5, label: 1 },
              { x: 1.0, y: 0.8, label: 1 }
            ];

            const dragPt = viz.addDraggable('dp', -0.5, 0.3, viz.colors.red, 8, () => draw());

            function solveSVM(pts) {
              let w1 = 1, w2 = 0, b = 0;
              const C = 50, lr = 0.005;
              for (let iter = 0; iter < 3000; iter++) {
                let g1 = w1, g2 = w2, gb = 0;
                for (const p of pts) {
                  const m = p.label * (w1 * p.x + w2 * p.y + b);
                  if (m < 1) { g1 -= C * p.label * p.x; g2 -= C * p.label * p.y; gb -= C * p.label; }
                }
                w1 -= lr * g1; w2 -= lr * g2; b -= lr * gb;
              }
              return { w1, w2, b };
            }

            function draw() {
              viz.clear();
              viz.drawGrid(1);
              viz.drawAxes();
              const ctx = viz.ctx;

              const allPts = [...data, { x: dragPt.x, y: dragPt.y, label: -1 }];
              const { w1, w2, b } = solveSVM(allPts);
              const norm = Math.sqrt(w1 * w1 + w2 * w2);

              if (norm > 0.01) {
                // Draw boundary and margins
                for (let offset = -1; offset <= 1; offset++) {
                  let lx1, ly1, lx2, ly2;
                  if (Math.abs(w2) > Math.abs(w1)) {
                    lx1 = -6; ly1 = -(w1 * lx1 + b - offset) / w2;
                    lx2 = 6; ly2 = -(w1 * lx2 + b - offset) / w2;
                  } else {
                    ly1 = -6; lx1 = -(w2 * ly1 + b - offset) / w1;
                    ly2 = 6; lx2 = -(w2 * ly2 + b - offset) / w1;
                  }
                  viz.drawLine(lx1, ly1, lx2, ly2, offset === 0 ? viz.colors.green : viz.colors.green + '55', offset === 0 ? 2.5 : 1.5, offset !== 0);
                }
              }

              // Draw points with alpha info
              const svInfo = [];
              for (const p of allPts) {
                const color = p.label === 1 ? viz.colors.blue : viz.colors.red;
                const funcMargin = norm > 0.01 ? p.label * (w1 * p.x + w2 * p.y + b) : 2;
                const isSV = funcMargin < 1.15;
                const alpha = isSV ? Math.max(0, 1 - funcMargin + 0.5) : 0;

                if (p !== allPts[allPts.length - 1]) { // skip draggable
                  viz.drawPoint(p.x, p.y, color, null, 5);
                }

                if (isSV) {
                  const [sx, sy] = viz.toScreen(p.x, p.y);
                  ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 2;
                  ctx.beginPath(); ctx.arc(sx, sy, 12, 0, Math.PI * 2); ctx.stroke();
                  // Show alpha label
                  ctx.fillStyle = viz.colors.yellow; ctx.font = '10px -apple-system,sans-serif';
                  ctx.textAlign = 'center';
                  ctx.fillText('\u03B1>0', sx, sy - 16);
                  svInfo.push(p);
                }
              }

              // Draw draggable
              viz.drawDraggables();

              // Weight vector representation
              viz.screenText('w = \u03A3 \u03B1\u1D62 y\u1D62 x\u1D62  (sum over ' + svInfo.length + ' support vectors)', viz.width / 2, viz.height - 28, viz.colors.white, 11);
              viz.screenText('Drag the red point to change which points are support vectors', viz.width / 2, viz.height - 12, viz.colors.text, 10);
            }

            draw();
            viz.animate(() => draw());
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Derive \\(\\mathbf{w} = \\sum_i \\alpha_i y_i \\mathbf{x}_i\\) from the Lagrangian of the hard-margin SVM.',
          hint: 'Take the gradient of \\(\\mathcal{L}(\\mathbf{w}, b, \\boldsymbol{\\alpha})\\) with respect to \\(\\mathbf{w}\\) and set it to zero.',
          solution: 'The Lagrangian is \\(\\mathcal{L} = \\frac{1}{2}\\|\\mathbf{w}\\|^2 - \\sum_i \\alpha_i[y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) - 1]\\). Taking the gradient: \\(\\nabla_{\\mathbf{w}}\\mathcal{L} = \\mathbf{w} - \\sum_i \\alpha_i y_i \\mathbf{x}_i = 0\\), which gives \\(\\mathbf{w} = \\sum_i \\alpha_i y_i \\mathbf{x}_i\\). This shows that the optimal weight vector is a linear combination of the training points, weighted by \\(\\alpha_i y_i\\). Since \\(\\alpha_i = 0\\) for non-support vectors, only support vectors contribute.'
        },
        {
          question: 'Show that the dual objective can be written as \\(\\sum_i \\alpha_i - \\frac{1}{2}\\boldsymbol{\\alpha}^\\top \\mathbf{Q} \\boldsymbol{\\alpha}\\) where \\(Q_{ij} = y_i y_j \\mathbf{x}_i^\\top \\mathbf{x}_j\\). Why is this a concave maximization problem?',
          hint: 'Show that \\(\\mathbf{Q}\\) is positive semidefinite.',
          solution: 'Substituting \\(\\mathbf{w} = \\sum_i \\alpha_i y_i \\mathbf{x}_i\\) into the Lagrangian and simplifying: \\(\\mathcal{L} = \\sum_i \\alpha_i - \\frac{1}{2}\\sum_{i,j} \\alpha_i \\alpha_j y_i y_j \\mathbf{x}_i^\\top \\mathbf{x}_j = \\mathbf{1}^\\top\\boldsymbol{\\alpha} - \\frac{1}{2}\\boldsymbol{\\alpha}^\\top\\mathbf{Q}\\boldsymbol{\\alpha}\\). The matrix \\(\\mathbf{Q}\\) with \\(Q_{ij} = y_i y_j \\mathbf{x}_i^\\top\\mathbf{x}_j\\) is positive semidefinite because for any \\(\\mathbf{v}\\): \\(\\mathbf{v}^\\top\\mathbf{Q}\\mathbf{v} = \\sum_{i,j} v_i v_j y_i y_j \\mathbf{x}_i^\\top\\mathbf{x}_j = \\|\\sum_i v_i y_i \\mathbf{x}_i\\|^2 \\geq 0\\). Therefore \\(-\\frac{1}{2}\\boldsymbol{\\alpha}^\\top\\mathbf{Q}\\boldsymbol{\\alpha}\\) is concave, making the dual a concave maximization (equivalently, convex minimization).'
        },
        {
          question: 'In the soft-margin dual, the constraint \\(0 \\leq \\alpha_i \\leq C\\) replaces \\(\\alpha_i \\geq 0\\). What happens to a support vector whose \\(\\alpha_i = C\\)? Is it on the margin boundary, inside the margin, or misclassified?',
          hint: 'Use the KKT conditions for the soft-margin SVM, which include both \\(\\alpha_i\\) and a dual variable for \\(\\xi_i \\geq 0\\).',
          solution: 'In the soft-margin SVM, the KKT conditions give \\(\\alpha_i[y_i(\\mathbf{w}^\\top\\mathbf{x}_i + b) - 1 + \\xi_i] = 0\\) and \\((C - \\alpha_i)\\xi_i = 0\\). When \\(\\alpha_i = C\\), the second condition allows \\(\\xi_i > 0\\), meaning the point violates the margin. If \\(\\xi_i &lt; 1\\), it is correctly classified but inside the margin. If \\(\\xi_i \\geq 1\\), it is misclassified. When \\(0 &lt; \\alpha_i &lt; C\\), we must have \\(\\xi_i = 0\\), so the point is exactly on the margin boundary. These "free" support vectors are the most useful for recovering \\(b\\).'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 4: Kernel SVM
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch07-sec04',
      title: 'Kernel SVM',
      content: `
<h2>The Kernel Trick</h2>

<p>The dual formulation reveals that the SVM depends on the data only through inner products \\(\\mathbf{x}_i^\\top\\mathbf{x}_j\\). The <strong>kernel trick</strong> replaces these inner products with a kernel function \\(k(\\mathbf{x}_i, \\mathbf{x}_j)\\) that implicitly computes inner products in a high-dimensional (possibly infinite-dimensional) feature space, without ever explicitly constructing that space.</p>

<div class="env-block definition"><div class="env-title">Definition 7.5 (Kernel Function)</div><div class="env-body">
<p>A function \\(k : \\mathbb{R}^d \\times \\mathbb{R}^d \\to \\mathbb{R}\\) is a <strong>valid kernel</strong> (positive definite kernel) if there exists a feature map \\(\\phi : \\mathbb{R}^d \\to \\mathcal{H}\\) into some Hilbert space \\(\\mathcal{H}\\) such that</p>
\\[k(\\mathbf{x}, \\mathbf{x}') = \\langle \\phi(\\mathbf{x}), \\phi(\\mathbf{x}') \\rangle_{\\mathcal{H}}.\\]
<p>Equivalently (by Mercer's theorem), \\(k\\) is a valid kernel iff the Gram matrix \\(K_{ij} = k(\\mathbf{x}_i, \\mathbf{x}_j)\\) is positive semidefinite for any finite set of points.</p>
</div></div>

<h3>Common Kernels</h3>

<table>
<thead><tr><th>Kernel</th><th>Formula</th><th>Feature space</th></tr></thead>
<tbody>
<tr><td>Linear</td><td>\\(k(\\mathbf{x}, \\mathbf{x}') = \\mathbf{x}^\\top\\mathbf{x}'\\)</td><td>\\(\\mathbb{R}^d\\) (original)</td></tr>
<tr><td>Polynomial</td><td>\\(k(\\mathbf{x}, \\mathbf{x}') = (\\mathbf{x}^\\top\\mathbf{x}' + c)^p\\)</td><td>Monomials up to degree \\(p\\)</td></tr>
<tr><td>RBF (Gaussian)</td><td>\\(k(\\mathbf{x}, \\mathbf{x}') = \\exp(-\\gamma\\|\\mathbf{x} - \\mathbf{x}'\\|^2)\\)</td><td>Infinite-dimensional</td></tr>
</tbody>
</table>

<div class="env-block theorem"><div class="env-title">Theorem 7.2 (Kernelized SVM)</div><div class="env-body">
<p>The kernel SVM dual replaces \\(\\mathbf{x}_i^\\top\\mathbf{x}_j\\) with \\(k(\\mathbf{x}_i, \\mathbf{x}_j)\\):</p>
\\[\\max_{\\boldsymbol{\\alpha}} \\; \\sum_i \\alpha_i - \\frac{1}{2}\\sum_{i,j} \\alpha_i \\alpha_j y_i y_j \\, k(\\mathbf{x}_i, \\mathbf{x}_j), \\quad 0 \\leq \\alpha_i \\leq C, \\quad \\sum_i \\alpha_i y_i = 0.\\]
<p>The decision function for a new point \\(\\mathbf{x}\\) is</p>
\\[f(\\mathbf{x}) = \\text{sign}\\!\\left(\\sum_{i \\in \\text{SV}} \\alpha_i y_i \\, k(\\mathbf{x}_i, \\mathbf{x}) + b\\right).\\]
</div></div>

<h3>The RBF Kernel</h3>

<p>The <strong>Radial Basis Function</strong> (RBF) kernel \\(k(\\mathbf{x}, \\mathbf{x}') = \\exp(-\\gamma\\|\\mathbf{x} - \\mathbf{x}'\\|^2)\\) is the most widely used kernel. The parameter \\(\\gamma > 0\\) controls the "reach" of each training point:</p>
<ul>
  <li><strong>Large \\(\\gamma\\)</strong>: each point influences only its immediate neighborhood, producing a complex, wiggly boundary (low bias, high variance).</li>
  <li><strong>Small \\(\\gamma\\)</strong>: each point has broad influence, producing a smooth boundary (high bias, low variance).</li>
</ul>

<div class="env-block intuition"><div class="env-title">Why the RBF Kernel Is So Powerful</div><div class="env-body"><p>The RBF kernel implicitly maps data into an infinite-dimensional space. This means the kernel SVM can represent arbitrarily complex decision boundaries, given enough support vectors and an appropriate \\(\\gamma\\). It can separate any finite dataset perfectly (though this may overfit). In practice, the regularization parameter \\(C\\) and the kernel parameter \\(\\gamma\\) together control model complexity.</p></div></div>

<div class="viz-placeholder" data-viz="viz-kernel-svm"></div>
`,
      visualizations: [
        {
          id: 'viz-kernel-svm',
          title: 'RBF Kernel SVM on Non-Linear Data',
          description: 'Data arranged in concentric circles cannot be separated by a line, but the RBF kernel SVM finds a non-linear boundary. Adjust gamma to see its effect on boundary complexity.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 55, originX: 280, originY: 220 });
            let gamma = 1.0;

            VizEngine.createSlider(controls, '\u03B3', 0.1, 5.0, gamma, 0.1, v => { gamma = v; draw(); });

            // Generate concentric circle data
            const data = [];
            const rng = (() => { let s = 77; return () => { s = (s * 16807) % 2147483647; return s / 2147483647; }; })();
            for (let i = 0; i < 20; i++) {
              const angle = rng() * 2 * Math.PI;
              const r = 0.5 + rng() * 0.5;
              data.push({ x: r * Math.cos(angle), y: r * Math.sin(angle), label: -1 });
            }
            for (let i = 0; i < 25; i++) {
              const angle = rng() * 2 * Math.PI;
              const r = 1.8 + rng() * 0.8;
              data.push({ x: r * Math.cos(angle), y: r * Math.sin(angle), label: 1 });
            }

            function rbf(x1, y1, x2, y2) {
              return Math.exp(-gamma * ((x1-x2)**2 + (y1-y2)**2));
            }

            function solveKernelSVM(data, gamma, C) {
              const n = data.length, alpha = new Array(n).fill(0); let b = 0;
              const K = []; for(let i=0;i<n;i++){K[i]=[]; for(let j=0;j<n;j++) K[i][j]=rbf(data[i].x,data[i].y,data[j].x,data[j].y);}

              // Simplified SMO
              for (let pass = 0; pass < 30; pass++) {
                for (let i = 0; i < n; i++) {
                  let ei = -data[i].label + b;
                  for (let k = 0; k < n; k++) ei += alpha[k] * data[k].label * K[i][k];

                  if ((data[i].label * ei < -0.01 && alpha[i] < C) ||
                      (data[i].label * ei > 0.01 && alpha[i] > 0)) {
                    let j = Math.floor(rng() * n);
                    if (j === i) j = (j + 1) % n;

                    let ej = -data[j].label + b;
                    for (let k = 0; k < n; k++) ej += alpha[k] * data[k].label * K[j][k];

                    const oldAi = alpha[i], oldAj = alpha[j];
                    let L, H;
                    if (data[i].label !== data[j].label) {
                      L = Math.max(0, alpha[j] - alpha[i]);
                      H = Math.min(C, C + alpha[j] - alpha[i]);
                    } else {
                      L = Math.max(0, alpha[i] + alpha[j] - C);
                      H = Math.min(C, alpha[i] + alpha[j]);
                    }
                    if (Math.abs(L - H) < 1e-5) continue;

                    const eta = 2 * K[i][j] - K[i][i] - K[j][j];
                    if (eta >= 0) continue;

                    alpha[j] -= data[j].label * (ei - ej) / eta;
                    alpha[j] = Math.min(H, Math.max(L, alpha[j]));

                    alpha[i] += data[i].label * data[j].label * (oldAj - alpha[j]);

                    const b1 = b - ei - data[i].label*(alpha[i]-oldAi)*K[i][i] - data[j].label*(alpha[j]-oldAj)*K[i][j];
                    const b2 = b - ej - data[i].label*(alpha[i]-oldAi)*K[i][j] - data[j].label*(alpha[j]-oldAj)*K[j][j];
                    if (alpha[i] > 0 && alpha[i] < C) b = b1;
                    else if (alpha[j] > 0 && alpha[j] < C) b = b2;
                    else b = (b1 + b2) / 2;
                  }
                }
              }
              return { alpha, b };
            }

            function predict(x, y, data, alpha, b) {
              let sum = b;
              for (let i = 0; i < data.length; i++) {
                if (alpha[i] > 1e-5) {
                  sum += alpha[i] * data[i].label * rbf(data[i].x, data[i].y, x, y);
                }
              }
              return sum;
            }

            function draw() {
              viz.clear();
              const ctx = viz.ctx;

              const { alpha, b } = solveKernelSVM(data, gamma, 5.0);

              // Draw decision regions
              const res = 80;
              const xMin = -4, xMax = 4, yMin = -3.5, yMax = 3.5;
              const dx = (xMax - xMin) / res, dy = (yMax - yMin) / res;

              for (let i = 0; i < res; i++) {
                for (let j = 0; j < res; j++) {
                  const x = xMin + (i + 0.5) * dx, y = yMin + (j + 0.5) * dy;
                  const val = predict(x, y, data, alpha, b);
                  const [sx, sy] = viz.toScreen(x, y);
                  ctx.fillStyle = val > 0 ? 'rgba(88,166,255,0.08)' : 'rgba(248,81,73,0.08)';
                  ctx.fillRect(sx - dx*viz.scale/2, sy - dy*viz.scale/2, dx*viz.scale+1, dy*viz.scale+1);
                }
              }

              // Draw boundary contour
              for (let i = 0; i < res - 1; i++) {
                for (let j = 0; j < res - 1; j++) {
                  const v00 = predict(xMin + i*dx, yMin + j*dy, data, alpha, b);
                  const v10 = predict(xMin + (i+1)*dx, yMin + j*dy, data, alpha, b);
                  const v01 = predict(xMin + i*dx, yMin + (j+1)*dy, data, alpha, b);
                  const v11 = predict(xMin + (i+1)*dx, yMin + (j+1)*dy, data, alpha, b);
                  const vals = [v00, v10, v01, v11];
                  if (vals.some(v => v > 0) && vals.some(v => v < 0)) {
                    const cx = xMin + (i + 0.5) * dx, cy = yMin + (j + 0.5) * dy;
                    const [sx, sy] = viz.toScreen(cx, cy);
                    ctx.fillStyle = viz.colors.green;
                    ctx.fillRect(sx - 1.5, sy - 1.5, 3, 3);
                  }
                }
              }

              viz.drawGrid(1);
              viz.drawAxes();

              // Draw data points
              let svCount = 0;
              for (let i = 0; i < data.length; i++) {
                const p = data[i];
                const color = p.label === 1 ? viz.colors.blue : viz.colors.red;
                viz.drawPoint(p.x, p.y, color, null, 4);
                if (alpha[i] > 0.01) {
                  svCount++;
                  const [sx, sy] = viz.toScreen(p.x, p.y);
                  ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 1.5;
                  ctx.beginPath(); ctx.arc(sx, sy, 9, 0, Math.PI * 2); ctx.stroke();
                }
              }

              viz.screenText('RBF Kernel SVM  \u03B3=' + gamma.toFixed(1) + '  SVs=' + svCount + '/' + data.length, viz.width/2, viz.height - 12, viz.colors.green, 12);
            }

            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Show that the polynomial kernel \\(k(\\mathbf{x}, \\mathbf{x}\') = (\\mathbf{x}^\\top\\mathbf{x}\' + 1)^2\\) for \\(\\mathbf{x} \\in \\mathbb{R}^2\\) corresponds to a specific feature map. What is \\(\\phi(\\mathbf{x})\\)?',
          hint: 'Expand \\((x_1 x_1\' + x_2 x_2\' + 1)^2\\) and identify the components of \\(\\phi\\).',
          solution: 'Expanding: \\((x_1 x_1\' + x_2 x_2\' + 1)^2 = x_1^2 x_1\'^2 + x_2^2 x_2\'^2 + 1 + 2x_1 x_2 x_1\' x_2\' + 2x_1 x_1\' + 2x_2 x_2\'\\). This equals \\(\\phi(\\mathbf{x})^\\top \\phi(\\mathbf{x}\')\\) with \\(\\phi(\\mathbf{x}) = (x_1^2, x_2^2, \\sqrt{2}\\,x_1 x_2, \\sqrt{2}\\,x_1, \\sqrt{2}\\,x_2, 1)^\\top\\). The kernel implicitly works in this 6-dimensional space without computing \\(\\phi\\) explicitly.'
        },
        {
          question: 'The RBF kernel \\(k(\\mathbf{x}, \\mathbf{x}\') = \\exp(-\\gamma\\|\\mathbf{x} - \\mathbf{x}\'\\|^2)\\) maps to an infinite-dimensional space. Show that \\(k(\\mathbf{x}, \\mathbf{x}) = 1\\) for all \\(\\mathbf{x}\\), and interpret this geometrically.',
          hint: 'Substitute \\(\\mathbf{x}\' = \\mathbf{x}\\) into the kernel formula.',
          solution: '\\(k(\\mathbf{x}, \\mathbf{x}) = \\exp(-\\gamma\\|\\mathbf{x} - \\mathbf{x}\\|^2) = \\exp(0) = 1\\). Since \\(k(\\mathbf{x}, \\mathbf{x}) = \\langle\\phi(\\mathbf{x}), \\phi(\\mathbf{x})\\rangle = \\|\\phi(\\mathbf{x})\\|^2\\), every point is mapped to a vector of unit norm in the feature space. Geometrically, all data points lie on the unit sphere in the (infinite-dimensional) feature space. This normalization is one reason the RBF kernel works well: the feature representations have comparable scales.'
        },
        {
          question: 'What happens to the RBF kernel SVM decision boundary as \\(\\gamma \\to \\infty\\)? As \\(\\gamma \\to 0\\)?',
          hint: 'Consider how the kernel values \\(k(\\mathbf{x}_i, \\mathbf{x})\\) behave for points near and far from \\(\\mathbf{x}_i\\).',
          solution: 'As \\(\\gamma \\to \\infty\\): \\(k(\\mathbf{x}_i, \\mathbf{x}) \\to 0\\) unless \\(\\mathbf{x} = \\mathbf{x}_i\\). Each support vector creates a tiny island of influence. The decision boundary becomes extremely complex, wrapping tightly around each training point. This is extreme overfitting. As \\(\\gamma \\to 0\\): \\(k(\\mathbf{x}_i, \\mathbf{x}) \\to 1\\) for all pairs. All kernel values become identical, so the SVM cannot distinguish points. The decision function becomes approximately constant, yielding extreme underfitting (a nearly linear, or even trivial, boundary).'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 5: SVM in Practice
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch07-sec05',
      title: 'SVM in Practice',
      content: `
<h2>Practical Considerations</h2>

<h3>Multi-Class Classification</h3>

<p>SVMs are inherently binary classifiers. For \\(K > 2\\) classes, two strategies are common:</p>

<div class="env-block definition"><div class="env-title">Definition 7.6 (Multi-Class SVM Strategies)</div><div class="env-body">
<ul>
  <li><strong>One-vs-All (OvA)</strong>: Train \\(K\\) binary SVMs, each separating class \\(k\\) from all others. Classify by the SVM with the largest decision value: \\(\\hat{y} = \\arg\\max_k f_k(\\mathbf{x})\\).</li>
  <li><strong>One-vs-One (OvO)</strong>: Train \\(\\binom{K}{2}\\) binary SVMs, one for each pair of classes. Classify by majority vote.</li>
</ul>
</div></div>

<p>OvA requires \\(K\\) classifiers but each trains on all \\(n\\) points. OvO requires \\(O(K^2)\\) classifiers but each trains on a subset of data. For large \\(K\\), OvO is often faster because each subproblem is smaller.</p>

<h3>Feature Scaling</h3>

<div class="env-block warning"><div class="env-title">Always Scale Features Before SVM</div><div class="env-body"><p>SVMs are <em>not</em> scale-invariant. The margin \\(2/\\|\\mathbf{w}\\|\\) and the RBF kernel \\(\\exp(-\\gamma\\|\\mathbf{x} - \\mathbf{x}'\\|^2)\\) both depend on the magnitude of features. If one feature ranges from 0 to 1000 and another from 0 to 1, the first will dominate the distance computations. Standard practice: standardize each feature to zero mean and unit variance, or scale to \\([0, 1]\\).</p></div></div>

<h3>SVM vs Logistic Regression</h3>

<p>Both linear SVMs and logistic regression find linear decision boundaries. Their key differences are:</p>

<table>
<thead><tr><th>Aspect</th><th>SVM (hinge loss)</th><th>Logistic Regression (log loss)</th></tr></thead>
<tbody>
<tr><td>Loss function</td><td>\\(\\max(0, 1 - y f(\\mathbf{x}))\\)</td><td>\\(\\log(1 + e^{-y f(\\mathbf{x})})\\)</td></tr>
<tr><td>Sparsity</td><td>Support vectors only</td><td>All points contribute</td></tr>
<tr><td>Probabilities</td><td>Not directly (needs Platt scaling)</td><td>Yes, calibrated by design</td></tr>
<tr><td>Kernels</td><td>Natural via dual</td><td>Possible but less common</td></tr>
<tr><td>Outlier sensitivity</td><td>Robust (hinge is flat for \\(z \\geq 1\\))</td><td>Slightly less (log loss always positive)</td></tr>
</tbody>
</table>

<div class="env-block remark"><div class="env-title">When to Use SVM</div><div class="env-body"><p>SVMs excel when: (1) the number of features is large relative to samples (text classification, genomics); (2) a non-linear kernel is needed; (3) you want a sparse solution. Logistic regression is preferred when: (1) calibrated probabilities are needed; (2) the dataset is very large (SVMs scale poorly beyond ~100k points); (3) interpretability of coefficients matters.</p></div></div>

<div class="viz-placeholder" data-viz="viz-svm-vs-lr"></div>
`,
      visualizations: [
        {
          id: 'viz-svm-vs-lr',
          title: 'SVM vs Logistic Regression: Decision Boundaries',
          description: 'Side-by-side comparison of linear SVM and logistic regression on the same 2D data. Both find linear boundaries, but SVM maximizes the margin while logistic regression maximizes the conditional likelihood.',
          setup(container, controls) {
            const viz = new VizEngine(container, { width: 560, height: 340, scale: 1, originX: 0, originY: 0 });
            const rng = (() => { let s = 55; return () => { s = (s * 16807) % 2147483647; return s / 2147483647; }; })();
            const data = [];
            for (let i = 0; i < 15; i++) { let u=0,v=0; for(let j=0;j<6;j++){u+=rng()-0.5;v+=rng()-0.5;} data.push({x:-1.5+u*0.8,y:0.5+v*0.8,label:-1}); }
            for (let i = 0; i < 15; i++) { let u=0,v=0; for(let j=0;j<6;j++){u+=rng()-0.5;v+=rng()-0.5;} data.push({x:1.5+u*0.8,y:-0.5+v*0.8,label:1}); }

            function solve(data, mode) {
              let w1=mode==='svm'?1:0, w2=0, b=0; const C=10, lr=mode==='svm'?0.005:0.05, lam=0.1;
              for (let iter=0; iter<3000; iter++) {
                let g1=mode==='svm'?w1:lam*w1, g2=mode==='svm'?w2:lam*w2, gb=0;
                for (const p of data) {
                  if (mode==='svm') { if (p.label*(w1*p.x+w2*p.y+b)<1) { g1-=C*p.label*p.x; g2-=C*p.label*p.y; gb-=C*p.label; } }
                  else { const sig=1/(1+Math.exp(p.label*(w1*p.x+w2*p.y+b))); g1-=sig*p.label*p.x/data.length; g2-=sig*p.label*p.y/data.length; gb-=sig*p.label/data.length; }
                }
                w1-=lr*g1; w2-=lr*g2; b-=lr*gb;
              }
              return {w1,w2,b};
            }
            const svmSol = solve(data,'svm'), lrSol = solve(data,'lr');

            function drawPanel(xOff, w, sol, title, showMargin) {
              const ctx=viz.ctx, pL=xOff+40, pR=xOff+w-10, pT=40, pB=viz.height-30, pw=pR-pL, ph=pB-pT;
              const xMn=-4,xMx=4,yMn=-3,yMx=3;
              ctx.strokeStyle=viz.colors.axis; ctx.lineWidth=1; ctx.strokeRect(pL,pT,pw,ph);
              ctx.fillStyle=viz.colors.white; ctx.font='bold 12px -apple-system,sans-serif'; ctx.textAlign='center';
              ctx.fillText(title,(pL+pR)/2,25);
              const toP=(x,y)=>[pL+(x-xMn)/(xMx-xMn)*pw, pB-(y-yMn)/(yMx-yMn)*ph];
              const {w1,w2,b}=sol, norm=Math.sqrt(w1*w1+w2*w2);
              // Shade regions
              const res=50, dxg=(xMx-xMn)/res, dyg=(yMx-yMn)/res;
              for(let i=0;i<res;i++) for(let j=0;j<res;j++) { const x=xMn+(i+.5)*dxg,y=yMn+(j+.5)*dyg; const [sx,sy]=toP(x,y); ctx.fillStyle=w1*x+w2*y+b>0?'rgba(88,166,255,0.08)':'rgba(248,81,73,0.08)'; ctx.fillRect(sx,sy-dyg/(yMx-yMn)*ph,dxg/(xMx-xMn)*pw+1,dyg/(yMx-yMn)*ph+1); }
              if(norm>0.01){
                const bLine=(off)=>{ let x1,y1,x2,y2; if(Math.abs(w2)>Math.abs(w1)){x1=xMn;y1=-(w1*x1+b-off)/w2;x2=xMx;y2=-(w1*x2+b-off)/w2;}else{y1=yMn;x1=-(w2*y1+b-off)/w1;y2=yMx;x2=-(w2*y2+b-off)/w1;} return[toP(x1,y1),toP(x2,y2)]; };
                const [[sx1,sy1],[sx2,sy2]]=bLine(0); ctx.strokeStyle=viz.colors.green; ctx.lineWidth=2.5; ctx.beginPath(); ctx.moveTo(sx1,sy1); ctx.lineTo(sx2,sy2); ctx.stroke();
                if(showMargin) for(const off of [-1,1]){ const [[a,b2],[c,d]]=bLine(off); ctx.strokeStyle=viz.colors.green+'55'; ctx.lineWidth=1; ctx.setLineDash([4,3]); ctx.beginPath(); ctx.moveTo(a,b2); ctx.lineTo(c,d); ctx.stroke(); ctx.setLineDash([]); }
              }
              for(const p of data){ const color=p.label===1?viz.colors.blue:viz.colors.red; const [sx,sy]=toP(p.x,p.y); ctx.fillStyle=color; ctx.beginPath(); ctx.arc(sx,sy,4,0,Math.PI*2); ctx.fill(); if(showMargin&&Math.abs(p.label*(w1*p.x+w2*p.y+b)-1)<0.2){ ctx.strokeStyle=viz.colors.yellow; ctx.lineWidth=1.5; ctx.beginPath(); ctx.arc(sx,sy,8,0,Math.PI*2); ctx.stroke(); } }
            }
            function draw() {
              viz.ctx.fillStyle=viz.colors.bg; viz.ctx.fillRect(0,0,viz.width,viz.height);
              drawPanel(0,viz.width/2,svmSol,'SVM (hinge loss)',true);
              drawPanel(viz.width/2,viz.width/2,lrSol,'Logistic Regression (log loss)',false);
              viz.ctx.fillStyle=viz.colors.text; viz.ctx.font='10px -apple-system,sans-serif'; viz.ctx.textAlign='center';
              viz.ctx.fillText('SVM margin boundaries shown as dashed lines; support vectors circled',viz.width/2,viz.height-8);
            }
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Compare the computational complexity of training an SVM vs logistic regression on \\(n\\) data points with \\(d\\) features. Why do SVMs scale poorly to very large datasets?',
          hint: 'SVM training involves solving a QP with \\(n\\) variables (dual variables). Logistic regression uses gradient descent on \\(d\\) parameters.',
          solution: 'Logistic regression via gradient descent has per-iteration cost \\(O(nd)\\) and typically converges in \\(O(1/\\epsilon)\\) iterations, giving total cost \\(O(nd/\\epsilon)\\). SVM training via standard QP solvers is \\(O(n^2 d)\\) to \\(O(n^3)\\) due to the kernel matrix and dual optimization. Even the efficient SMO algorithm is \\(O(n^2 d)\\) in the worst case. For \\(n > 10^5\\), the quadratic dependence on \\(n\\) makes SVMs impractical without approximations (e.g., random Fourier features or Nystrom approximation).'
        },
        {
          question: 'Explain Platt scaling for SVMs. Why is it needed, and how does it work?',
          hint: 'SVMs output \\(f(\\mathbf{x}) = \\mathbf{w}^\\top\\mathbf{x} + b\\), which is not a probability. How can we convert this to \\(P(y=1 \\mid \\mathbf{x})\\)?',
          solution: 'SVMs produce a decision value \\(f(\\mathbf{x})\\) but not a calibrated probability. Platt scaling fits a logistic regression on top of the SVM output: \\(P(y=1 \\mid \\mathbf{x}) = \\sigma(Af(\\mathbf{x}) + B)\\), where \\(A\\) and \\(B\\) are learned by maximum likelihood on a held-out set (or via cross-validation to avoid overfitting). This post-hoc calibration converts the SVM into a probabilistic classifier. It is needed because the hinge loss does not correspond to a probabilistic model, unlike the log loss of logistic regression.'
        },
        {
          question: 'You have a text classification problem with 50,000 features (bag-of-words) and 5,000 training documents. Would you recommend SVM with an RBF kernel, SVM with a linear kernel, or Naive Bayes? Justify your answer.',
          hint: 'Consider the ratio of features to samples and the computational cost of kernels.',
          solution: 'With \\(d = 50{,}000 \\gg n = 5{,}000\\), a linear classifier is likely sufficient (high-dimensional data is often linearly separable). A linear SVM is an excellent choice: it handles high dimensions well and produces a sparse solution. An RBF kernel SVM would require computing a \\(5000 \\times 5000\\) kernel matrix, and in such high dimensions, the RBF kernel tends to perform similarly to the linear kernel (the "curse of dimensionality" makes all pairwise distances similar). Naive Bayes is also reasonable as a fast baseline, especially since text features are approximately conditionally independent. The ranking would be: linear SVM (best), Naive Bayes (fast baseline), RBF SVM (unnecessary complexity).'
        }
      ]
    }
  ]
});
