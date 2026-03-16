window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
  id: 'ch02',
  number: 2,
  title: 'Linear Regression',
  subtitle: 'Least Squares, Regularization, and the Probabilistic Foundations',
  sections: [

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 1: Simple Linear Regression
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch02-sec01',
      title: 'Simple Linear Regression',
      content: `
<div class="env-block intuition"><div class="env-title">The Simplest Prediction Machine</div><div class="env-body"><p>Given paired observations \\((x_1, y_1), \\dots, (x_n, y_n)\\), we seek the straight line \\(\\hat{y} = wx + b\\) that best summarizes the relationship between \\(x\\) and \\(y\\). "Best" means minimizing the total squared vertical distance from each data point to the line. This is the <strong>method of least squares</strong>, introduced by Legendre (1805) and Gauss (1809).</p></div></div>

<h2>The Model</h2>

<div class="env-block definition"><div class="env-title">Definition 2.1 (Simple Linear Regression)</div><div class="env-body">
<p>The simple linear regression model assumes</p>
\\[y_i = w x_i + b + \\varepsilon_i, \\qquad i = 1, \\dots, n,\\]
<p>where \\(w\\) is the <strong>slope</strong> (weight), \\(b\\) is the <strong>intercept</strong> (bias), and \\(\\varepsilon_i\\) are independent noise terms with \\(\\mathbb{E}[\\varepsilon_i] = 0\\).</p>
</div></div>

<h2>Least Squares Objective</h2>

<p>We choose \\(w\\) and \\(b\\) to minimize the <strong>residual sum of squares</strong> (RSS):</p>
\\[\\mathcal{L}(w, b) = \\sum_{i=1}^{n} (y_i - w x_i - b)^2.\\]

<p>This is a convex quadratic in \\((w, b)\\), so the minimum is found by setting the partial derivatives to zero.</p>

<h2>Closed-Form Solution</h2>

<p>Setting \\(\\frac{\\partial \\mathcal{L}}{\\partial b} = 0\\) gives \\(b = \\bar{y} - w\\bar{x}\\), where \\(\\bar{x} = \\frac{1}{n}\\sum x_i\\) and \\(\\bar{y} = \\frac{1}{n}\\sum y_i\\). Substituting into \\(\\frac{\\partial \\mathcal{L}}{\\partial w} = 0\\):</p>

<div class="env-block theorem"><div class="env-title">Theorem 2.1 (OLS Estimators)</div><div class="env-body">
\\[\\hat{w} = \\frac{\\sum_{i=1}^n (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^n (x_i - \\bar{x})^2} = \\frac{\\mathrm{Cov}(x, y)}{\\mathrm{Var}(x)}, \\qquad \\hat{b} = \\bar{y} - \\hat{w}\\bar{x}.\\]
</div></div>

<div class="env-block remark"><div class="env-title">Remark</div><div class="env-body"><p>The regression line always passes through the centroid \\((\\bar{x}, \\bar{y})\\). The slope is the ratio of the sample covariance to the sample variance of \\(x\\).</p></div></div>

<h2>Residuals</h2>

<p>The residual for observation \\(i\\) is \\(e_i = y_i - \\hat{y}_i = y_i - \\hat{w} x_i - \\hat{b}\\). The residuals sum to zero (\\(\\sum e_i = 0\\)) and are uncorrelated with the fitted values (\\(\\sum \\hat{y}_i e_i = 0\\)). The coefficient of determination \\(R^2 = 1 - \\frac{\\text{RSS}}{\\text{TSS}}\\), where \\(\\text{TSS} = \\sum(y_i - \\bar{y})^2\\), measures the fraction of variance explained.</p>

<div class="viz-placeholder" data-viz="viz-simple-lr"></div>
`,
      visualizations: [
        {
          id: 'viz-simple-lr',
          title: 'Interactive Simple Linear Regression',
          description: 'Drag the data points to see the least-squares regression line and residuals update in real time. The blue line is the fit; dashed orange lines are residuals.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 50, originX: 60, originY: 340 });
            const pts = [
              {x:1,y:2.2},{x:1.8,y:2.8},{x:2.5,y:3.5},{x:3.2,y:3.1},
              {x:4,y:4.5},{x:4.8,y:4.2},{x:5.5,y:5.8},{x:6.2,y:5.5},
              {x:7,y:6.1},{x:7.8,y:7.0}
            ];
            const drags = pts.map((p,i) => viz.addDraggable('p'+i, p.x, p.y, viz.colors.teal, 7));
            let showResiduals = true;
            VizEngine.createButton(controls, 'Toggle Residuals', () => { showResiduals = !showResiduals; draw(); });

            function fitLine() {
              const n = drags.length;
              let sx=0,sy=0; for (const d of drags){sx+=d.x;sy+=d.y;}
              const mx=sx/n, my=sy/n;
              let num=0,den=0;
              for (const d of drags){num+=(d.x-mx)*(d.y-my);den+=(d.x-mx)**2;}
              const w = den > 1e-12 ? num/den : 0;
              const b = my - w*mx;
              let rss=0,tss=0;
              for (const d of drags){rss+=(d.y-w*d.x-b)**2;tss+=(d.y-my)**2;}
              const r2 = tss>1e-12 ? 1-rss/tss : 0;
              return {w,b,r2};
            }
            function draw() {
              viz.clear(); viz.drawGrid(1); viz.drawAxes();
              const {w,b,r2} = fitLine();
              // Regression line across visible range
              viz.drawLine(0, b, 10, w*10+b, viz.colors.blue, 2.5);
              // Residuals
              if (showResiduals) {
                for (const d of drags) {
                  const yhat = w*d.x+b;
                  viz.drawSegment(d.x, d.y, d.x, yhat, viz.colors.orange, 1.5, true);
                }
              }
              viz.drawDraggables();
              viz.screenText('y\u0302 = '+w.toFixed(2)+'x + '+b.toFixed(2), viz.width/2, 18, viz.colors.blue, 13);
              viz.screenText('R\u00B2 = '+r2.toFixed(3), viz.width/2, 36, viz.colors.text, 12);
            }
            for (const d of drags) d.onDrag = () => draw();
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Derive the OLS estimator \\(\\hat{w}\\) by differentiating \\(\\mathcal{L}(w,b)\\) with respect to \\(w\\), substituting \\(b = \\bar{y} - w\\bar{x}\\), and solving.',
          hint: 'After substituting, the loss becomes \\(\\sum (y_i - \\bar{y} - w(x_i - \\bar{x}))^2\\). Differentiate with respect to \\(w\\).',
          solution: 'Substituting \\(b = \\bar{y} - w\\bar{x}\\) gives \\(\\mathcal{L} = \\sum(y_i - \\bar{y} - w(x_i - \\bar{x}))^2\\). Setting \\(\\partial \\mathcal{L}/\\partial w = -2\\sum(x_i - \\bar{x})(y_i - \\bar{y} - w(x_i - \\bar{x})) = 0\\), we get \\(w\\sum(x_i - \\bar{x})^2 = \\sum(x_i - \\bar{x})(y_i - \\bar{y})\\), hence \\(\\hat{w} = \\frac{\\sum(x_i-\\bar{x})(y_i-\\bar{y})}{\\sum(x_i-\\bar{x})^2}\\).'
        },
        {
          question: 'Show that \\(\\sum_{i=1}^n e_i = 0\\) for OLS residuals.',
          hint: 'Use the first-order condition \\(\\partial \\mathcal{L}/\\partial b = 0\\).',
          solution: 'The first-order condition \\(\\partial \\mathcal{L}/\\partial b = -2\\sum(y_i - \\hat{w}x_i - \\hat{b}) = 0\\) directly implies \\(\\sum e_i = \\sum(y_i - \\hat{w}x_i - \\hat{b}) = 0\\).'
        },
        {
          question: 'If \\(y_i = 3x_i + 1 + \\varepsilon_i\\) with \\(\\varepsilon_i \\sim N(0,1)\\), what are the true values of \\(w\\) and \\(b\\)? As \\(n \\to \\infty\\), what does \\(\\hat{w}\\) converge to and why?',
          hint: 'Think about consistency of OLS.',
          solution: 'The true parameters are \\(w=3\\) and \\(b=1\\). By the law of large numbers, \\(\\hat{w} = \\mathrm{Cov}(x,y)/\\mathrm{Var}(x) \\to w = 3\\) as \\(n \\to \\infty\\), since the sample moments converge to their population counterparts. OLS is consistent under correct specification.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 2: Multiple Linear Regression
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch02-sec02',
      title: 'Multiple Linear Regression',
      content: `
<h2>Matrix Formulation</h2>

<p>With \\(d\\) features, the model becomes \\(y_i = \\mathbf{x}_i^\\top \\boldsymbol{\\beta} + \\varepsilon_i\\), where \\(\\boldsymbol{\\beta} = (\\beta_0, \\beta_1, \\dots, \\beta_d)^\\top\\) and \\(\\mathbf{x}_i = (1, x_{i1}, \\dots, x_{id})^\\top\\) includes a leading 1 for the intercept. Stacking all \\(n\\) observations:</p>

<div class="env-block definition"><div class="env-title">Definition 2.2 (Matrix Form)</div><div class="env-body">
\\[\\mathbf{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\varepsilon},\\]
<p>where \\(\\mathbf{y} \\in \\mathbb{R}^n\\), \\(\\mathbf{X} \\in \\mathbb{R}^{n \\times (d+1)}\\) is the <strong>design matrix</strong> with row \\(i\\) equal to \\(\\mathbf{x}_i^\\top\\), and \\(\\boldsymbol{\\varepsilon} \\in \\mathbb{R}^n\\).</p>
</div></div>

<h2>The Normal Equations</h2>

<p>The RSS in matrix form is \\(\\mathcal{L}(\\boldsymbol{\\beta}) = (\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})^\\top(\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})\\). Expanding and differentiating:</p>
\\[\\nabla_{\\boldsymbol{\\beta}} \\mathcal{L} = -2\\mathbf{X}^\\top(\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}) = \\mathbf{0}.\\]

<div class="env-block theorem"><div class="env-title">Theorem 2.2 (Normal Equations)</div><div class="env-body">
<p>If \\(\\mathbf{X}^\\top\\mathbf{X}\\) is invertible, the unique minimizer is</p>
\\[\\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top\\mathbf{y}.\\]
</div></div>

<h3>Geometric Interpretation</h3>

<p>The fitted values \\(\\hat{\\mathbf{y}} = \\mathbf{X}\\hat{\\boldsymbol{\\beta}} = \\mathbf{H}\\mathbf{y}\\), where \\(\\mathbf{H} = \\mathbf{X}(\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top\\) is the <strong>hat matrix</strong>. This is the orthogonal projection of \\(\\mathbf{y}\\) onto the column space of \\(\\mathbf{X}\\). The residual vector \\(\\mathbf{e} = \\mathbf{y} - \\hat{\\mathbf{y}}\\) is perpendicular to every column of \\(\\mathbf{X}\\), which is precisely the normal equation \\(\\mathbf{X}^\\top \\mathbf{e} = \\mathbf{0}\\).</p>

<div class="env-block remark"><div class="env-title">Computational Note</div><div class="env-body"><p>In practice, the normal equations are solved via QR decomposition or SVD rather than matrix inversion. If \\(\\mathbf{X} = \\mathbf{Q}\\mathbf{R}\\) (thin QR), then \\(\\hat{\\boldsymbol{\\beta}} = \\mathbf{R}^{-1}\\mathbf{Q}^\\top\\mathbf{y}\\), which is numerically stable and costs \\(O(nd^2)\\).</p></div></div>

<div class="viz-placeholder" data-viz="viz-multiple-lr"></div>
`,
      visualizations: [
        {
          id: 'viz-multiple-lr',
          title: 'Multiple Regression: Contour View of a 2-Feature Fit',
          description: 'A contour plot of the loss surface \\(\\mathcal{L}(\\beta_1, \\beta_2)\\) for a two-feature regression. The minimum (star) is the OLS solution. Adjust the correlation between features to see how collinearity elongates the contours.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 60, originX: 280, originY: 220 });
            const ctx = viz.ctx;
            let corr = 0.0;
            VizEngine.createSlider(controls, 'Feature correlation', -0.95, 0.95, 0, 0.05, v => { corr = v; draw(); });

            function draw() {
              viz.clear();
              // Generate XtX-like matrix given correlation
              // Two features, each standardized: XtX ≈ n*[[1,r],[r,1]]
              const n = 20;
              const a11 = n, a12 = n*corr, a22 = n;
              // True beta
              const bt1 = 2, bt2 = 1;
              // Loss(b1,b2) = (b-bt)^T A (b-bt) + const, A = XtX
              // Draw contours
              const levels = [0.5,1.5,3,5,8,12,18,25];
              for (const lev of levels) {
                const det = a11*a22 - a12*a12;
                if (det < 1e-6) continue;
                // Eigenvalues of A
                const tr = a11+a22, disc = Math.sqrt(Math.max(0,(a11-a22)**2+4*a12*a12));
                const l1 = (tr+disc)/2, l2 = (tr-disc)/2;
                if (l1 < 1e-6 || l2 < 1e-6) continue;
                // Eigenvector for l1
                let v1x, v1y;
                if (Math.abs(a12) > 1e-10) { v1x = l1-a22; v1y = a12; }
                else { v1x = 1; v1y = 0; }
                const vn = Math.sqrt(v1x*v1x+v1y*v1y);
                v1x /= vn; v1y /= vn;
                const angle = Math.atan2(-v1y, v1x); // screen y is inverted
                const rx = Math.sqrt(lev/l1);
                const ry = Math.sqrt(lev/l2);
                const [sx, sy] = viz.toScreen(bt1, bt2);
                const alpha = Math.max(0.08, 0.3 - lev*0.008);
                ctx.strokeStyle = `rgba(88,166,255,${alpha})`;
                ctx.lineWidth = 1.2;
                ctx.save(); ctx.translate(sx, sy); ctx.rotate(-Math.atan2(v1y, v1x));
                ctx.beginPath(); ctx.ellipse(0, 0, rx*viz.scale, ry*viz.scale, 0, 0, Math.PI*2); ctx.stroke();
                ctx.restore();
              }
              // Axes and labels
              viz.drawGrid(1); viz.drawAxes();
              // Mark optimum
              const [osx, osy] = viz.toScreen(bt1, bt2);
              ctx.fillStyle = viz.colors.yellow;
              ctx.font = '18px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
              ctx.fillText('\u2605', osx, osy);
              viz.screenText('\u03B2\u0302 = ('+bt1+', '+bt2+')', osx+10, osy-16, viz.colors.yellow, 12, 'left');
              viz.screenText('\u03B2\u2081', viz.width-15, viz.originY-10, viz.colors.text, 13, 'right');
              viz.screenText('\u03B2\u2082', viz.originX+10, 12, viz.colors.text, 13, 'left');
              viz.screenText('Corr = '+corr.toFixed(2)+(Math.abs(corr)>0.8?' (high collinearity!)':''), viz.width/2, viz.height-14, Math.abs(corr)>0.8?viz.colors.red:viz.colors.text, 12);
            }
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Show that the hat matrix \\(\\mathbf{H} = \\mathbf{X}(\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top\\) is idempotent (\\(\\mathbf{H}^2 = \\mathbf{H}\\)) and symmetric.',
          hint: 'Multiply \\(\\mathbf{H}\\) by itself and simplify. For symmetry, take the transpose.',
          solution: '\\(\\mathbf{H}^2 = \\mathbf{X}(\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top \\mathbf{X}(\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top = \\mathbf{X}(\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top = \\mathbf{H}\\). For symmetry: \\(\\mathbf{H}^\\top = (\\mathbf{X}(\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top)^\\top = \\mathbf{X}((\\mathbf{X}^\\top\\mathbf{X})^{-1})^\\top \\mathbf{X}^\\top = \\mathbf{X}(\\mathbf{X}^\\top\\mathbf{X})^{-1}\\mathbf{X}^\\top = \\mathbf{H}\\), since \\((\\mathbf{X}^\\top\\mathbf{X})^{-1}\\) is symmetric.'
        },
        {
          question: 'Why does high collinearity among features make estimation unstable? Relate this to the eigenvalues of \\(\\mathbf{X}^\\top\\mathbf{X}\\).',
          hint: 'What happens to \\((\\mathbf{X}^\\top\\mathbf{X})^{-1}\\) when one eigenvalue is near zero?',
          solution: 'If features are nearly collinear, \\(\\mathbf{X}^\\top\\mathbf{X}\\) has a near-zero eigenvalue \\(\\lambda_{\\min} \\approx 0\\). The inverse \\((\\mathbf{X}^\\top\\mathbf{X})^{-1}\\) has eigenvalue \\(1/\\lambda_{\\min}\\), which is enormous. This amplifies small perturbations in \\(\\mathbf{y}\\) into large swings in \\(\\hat{\\boldsymbol{\\beta}}\\). The condition number \\(\\kappa = \\lambda_{\\max}/\\lambda_{\\min}\\) quantifies this instability; collinearity drives \\(\\kappa \\to \\infty\\).'
        },
        {
          question: 'Suppose we add a constant feature \\(x_{\\text{new}} = 5\\) to every observation. Does the OLS fit change? Justify.',
          hint: 'Does the new column change the column space of \\(\\mathbf{X}\\)?',
          solution: 'No. If the model already includes an intercept (a column of ones), then the constant column \\(5 \\cdot \\mathbf{1}\\) is a scalar multiple of the intercept column, so \\(\\mathbf{X}^\\top\\mathbf{X}\\) becomes singular (or the column is dropped). The column space of \\(\\mathbf{X}\\) does not change, so \\(\\hat{\\mathbf{y}} = \\mathbf{H}\\mathbf{y}\\) is unchanged. Only the allocation of coefficients between the intercept and the new feature is indeterminate.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 3: Regularization
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch02-sec03',
      title: 'Regularization',
      content: `
<div class="env-block intuition"><div class="env-title">Taming the Coefficients</div><div class="env-body"><p>Unregularized OLS can produce wildly large coefficients when features are correlated or \\(d\\) is large relative to \\(n\\). Regularization adds a penalty that shrinks coefficients toward zero, trading a small increase in bias for a large reduction in variance.</p></div></div>

<h2>Ridge Regression (L2 Penalty)</h2>

<div class="env-block definition"><div class="env-title">Definition 2.3 (Ridge Regression)</div><div class="env-body">
\\[\\hat{\\boldsymbol{\\beta}}_{\\text{ridge}} = \\arg\\min_{\\boldsymbol{\\beta}} \\left\\{ \\|\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}\\|^2 + \\lambda \\|\\boldsymbol{\\beta}\\|_2^2 \\right\\},\\]
<p>where \\(\\lambda \\geq 0\\) is the <strong>regularization parameter</strong>. The closed-form solution is</p>
\\[\\hat{\\boldsymbol{\\beta}}_{\\text{ridge}} = (\\mathbf{X}^\\top\\mathbf{X} + \\lambda \\mathbf{I})^{-1}\\mathbf{X}^\\top\\mathbf{y}.\\]
</div></div>

<p>Adding \\(\\lambda \\mathbf{I}\\) shifts all eigenvalues of \\(\\mathbf{X}^\\top\\mathbf{X}\\) upward, so the matrix is always invertible. As \\(\\lambda \\to \\infty\\), all coefficients shrink to zero. As \\(\\lambda \\to 0\\), we recover OLS.</p>

<h2>Lasso Regression (L1 Penalty)</h2>

<div class="env-block definition"><div class="env-title">Definition 2.4 (Lasso)</div><div class="env-body">
\\[\\hat{\\boldsymbol{\\beta}}_{\\text{lasso}} = \\arg\\min_{\\boldsymbol{\\beta}} \\left\\{ \\|\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}\\|^2 + \\lambda \\|\\boldsymbol{\\beta}\\|_1 \\right\\},\\]
<p>where \\(\\|\\boldsymbol{\\beta}\\|_1 = \\sum_j |\\beta_j|\\). Unlike ridge, the Lasso has no closed-form solution, but it produces <strong>sparse</strong> solutions: some coefficients are driven exactly to zero.</p>
</div></div>

<div class="env-block remark"><div class="env-title">Why Sparsity?</div><div class="env-body"><p>The L1 constraint set \\(\\{\\boldsymbol{\\beta}: \\|\\boldsymbol{\\beta}\\|_1 \\leq t\\}\\) has corners aligned with the coordinate axes. The elliptical contours of the loss are likely to first touch a corner, setting one or more coefficients to exactly zero. The L2 ball has no corners, so ridge shrinks everything but sets nothing exactly to zero.</p></div></div>

<h2>Elastic Net</h2>

<p>The <strong>elastic net</strong> (Zou and Hastie, 2005) combines both penalties:</p>
\\[\\hat{\\boldsymbol{\\beta}}_{\\text{EN}} = \\arg\\min_{\\boldsymbol{\\beta}} \\left\\{ \\|\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}\\|^2 + \\lambda_1 \\|\\boldsymbol{\\beta}\\|_1 + \\lambda_2 \\|\\boldsymbol{\\beta}\\|_2^2 \\right\\}.\\]
<p>This inherits sparsity from L1 and stability under collinearity from L2.</p>

<div class="viz-placeholder" data-viz="viz-regularization"></div>
`,
      visualizations: [
        {
          id: 'viz-regularization',
          title: 'Ridge vs. Lasso Coefficient Paths',
          description: 'As the regularization strength \\(\\lambda\\) increases (slider), watch how Ridge (blue) shrinks all coefficients smoothly toward zero, while Lasso (orange) drives some exactly to zero, performing feature selection.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 1, originX: 80, originY: 200 });
            const ctx = viz.ctx;
            const W = viz.width, H = viz.height;
            // Simulate 5 coefficients: true betas
            const trueBetas = [3.0, -2.0, 1.5, -0.5, 0.2];
            const colors = [viz.colors.blue, viz.colors.red, viz.colors.green, viz.colors.purple, viz.colors.orange];
            let logLam = -1; // log10(lambda)
            VizEngine.createSlider(controls, 'log\u2081\u2080(\u03BB)', -2, 3, -1, 0.1, v => { logLam = v; draw(); });

            function ridgeCoef(lam) {
              // Simplified: orthogonal features, so ridge shrinks by factor 1/(1+lam)
              return trueBetas.map(b => b / (1 + lam));
            }
            function lassoCoef(lam) {
              // Orthogonal case: soft thresholding
              return trueBetas.map(b => Math.sign(b) * Math.max(0, Math.abs(b) - lam/2));
            }
            function draw() {
              ctx.fillStyle = viz.colors.bg; ctx.fillRect(0, 0, W, H);
              const lam = Math.pow(10, logLam);
              // Draw path: x-axis = log(lambda), y-axis = coefficient value
              const padL=80, padR=30, padT=30, padB=40;
              const pw=W-padL-padR, ph=H-padT-padB;
              // Axes
              ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
              ctx.beginPath(); ctx.moveTo(padL,padT); ctx.lineTo(padL,padT+ph); ctx.lineTo(padL+pw,padT+ph); ctx.stroke();
              // Y axis: -3.5 to 3.5
              const yMin=-3.5, yMax=3.5;
              const toSX = ll => padL + (ll-(-2))/(3-(-2))*pw;
              const toSY = v => padT + (yMax-v)/(yMax-yMin)*ph;
              // Zero line
              ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.8;
              ctx.beginPath(); ctx.moveTo(padL, toSY(0)); ctx.lineTo(padL+pw, toSY(0)); ctx.stroke();
              // Y ticks
              ctx.fillStyle = viz.colors.text; ctx.font = '11px sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
              for (let v=-3;v<=3;v++) { ctx.fillText(v, padL-6, toSY(v)); }
              // X ticks
              ctx.textAlign = 'center'; ctx.textBaseline = 'top';
              for (let l=-2;l<=3;l++) { ctx.fillText('10^'+l, toSX(l), padT+ph+6); }
              ctx.fillText('log(\u03BB)', padL+pw/2, padT+ph+24);
              // Draw coefficient paths
              const nSteps = 200;
              for (let ci=0; ci<trueBetas.length; ci++) {
                // Ridge path
                ctx.strokeStyle = colors[ci]; ctx.lineWidth = 2; ctx.globalAlpha = 0.8;
                ctx.beginPath();
                for (let s=0; s<=nSteps; s++) {
                  const ll = -2+s*(5/nSteps);
                  const l = Math.pow(10, ll);
                  const v = trueBetas[ci]/(1+l);
                  const sx = toSX(ll), sy = toSY(v);
                  s===0 ? ctx.moveTo(sx,sy) : ctx.lineTo(sx,sy);
                }
                ctx.stroke();
                // Lasso path (dashed)
                ctx.setLineDash([5,4]);
                ctx.beginPath();
                for (let s=0; s<=nSteps; s++) {
                  const ll = -2+s*(5/nSteps);
                  const l = Math.pow(10, ll);
                  const v = Math.sign(trueBetas[ci])*Math.max(0,Math.abs(trueBetas[ci])-l/2);
                  const sx = toSX(ll), sy = toSY(v);
                  s===0 ? ctx.moveTo(sx,sy) : ctx.lineTo(sx,sy);
                }
                ctx.stroke();
                ctx.setLineDash([]);
              }
              ctx.globalAlpha = 1;
              // Current lambda marker
              const curX = toSX(logLam);
              ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
              ctx.beginPath(); ctx.moveTo(curX, padT); ctx.lineTo(curX, padT+ph); ctx.stroke();
              ctx.setLineDash([]);
              // Current values
              const rCoefs = ridgeCoef(lam), lCoefs = lassoCoef(lam);
              for (let ci=0; ci<trueBetas.length; ci++) {
                ctx.fillStyle = colors[ci];
                ctx.beginPath(); ctx.arc(curX, toSY(rCoefs[ci]), 4, 0, Math.PI*2); ctx.fill();
                ctx.beginPath(); ctx.arc(curX, toSY(lCoefs[ci]), 3, 0, Math.PI*2);
                ctx.strokeStyle = colors[ci]; ctx.lineWidth = 2; ctx.stroke();
              }
              // Legend
              ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
              ctx.fillStyle = viz.colors.text; ctx.fillText('Solid = Ridge, Dashed = Lasso', padL+8, padT+12);
              for (let ci=0; ci<trueBetas.length; ci++) {
                ctx.fillStyle = colors[ci];
                ctx.fillText('\u03B2'+(ci+1)+': R='+rCoefs[ci].toFixed(2)+' L='+lCoefs[ci].toFixed(2), padL+pw-140, padT+12+ci*14);
              }
            }
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Show that the ridge solution \\(\\hat{\\boldsymbol{\\beta}}_{\\text{ridge}} = (\\mathbf{X}^\\top\\mathbf{X} + \\lambda\\mathbf{I})^{-1}\\mathbf{X}^\\top\\mathbf{y}\\) always exists, regardless of the rank of \\(\\mathbf{X}\\).',
          hint: 'What are the eigenvalues of \\(\\mathbf{X}^\\top\\mathbf{X} + \\lambda\\mathbf{I}\\) when \\(\\lambda > 0\\)?',
          solution: '\\(\\mathbf{X}^\\top\\mathbf{X}\\) is positive semidefinite, so its eigenvalues \\(\\sigma_j^2 \\geq 0\\). Adding \\(\\lambda\\mathbf{I}\\) shifts each eigenvalue to \\(\\sigma_j^2 + \\lambda > 0\\) for any \\(\\lambda > 0\\). Since all eigenvalues are strictly positive, the matrix is positive definite, hence invertible.'
        },
        {
          question: 'In the orthogonal design case (\\(\\mathbf{X}^\\top\\mathbf{X} = \\mathbf{I}\\)), derive the Lasso solution explicitly. What operation does it perform on each coefficient?',
          hint: 'The Lasso objective decomposes into independent subproblems, one per coefficient.',
          solution: 'When \\(\\mathbf{X}^\\top\\mathbf{X} = \\mathbf{I}\\), the OLS solution is \\(\\hat{\\boldsymbol{\\beta}}_{\\text{OLS}} = \\mathbf{X}^\\top\\mathbf{y}\\). The Lasso objective separates: \\(\\min_{\\beta_j}(\\beta_j - \\hat{\\beta}_j^{\\text{OLS}})^2 + \\lambda|\\beta_j|\\). The solution is the <strong>soft-thresholding operator</strong>: \\(\\hat{\\beta}_j = \\mathrm{sign}(\\hat{\\beta}_j^{\\text{OLS}})\\max(0, |\\hat{\\beta}_j^{\\text{OLS}}| - \\lambda/2)\\). It translates each coefficient toward zero by \\(\\lambda/2\\), setting it exactly to zero if \\(|\\hat{\\beta}_j^{\\text{OLS}}| \\leq \\lambda/2\\).'
        },
        {
          question: 'Why might you prefer the elastic net over pure Lasso when features are grouped (highly correlated subsets)?',
          hint: 'What does Lasso do when two features are identical?',
          solution: 'If two features are identical, Lasso arbitrarily selects one and sets the other to zero, since the L1 penalty has a non-unique solution along the constraint boundary. The elastic net, through its L2 component, encourages correlated features to share similar coefficient values (the "grouping effect"). This is more stable and interpretable when features naturally come in correlated groups.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 4: Polynomial Regression
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch02-sec04',
      title: 'Polynomial Regression',
      content: `
<h2>Feature Engineering via Basis Expansion</h2>

<p>Linear regression is "linear in the parameters," not in the features. By constructing nonlinear features from the raw input, we can fit curves. The simplest example is <strong>polynomial regression</strong>:</p>

<div class="env-block definition"><div class="env-title">Definition 2.5 (Polynomial Regression)</div><div class="env-body">
<p>Given a scalar input \\(x\\), polynomial regression of degree \\(p\\) models</p>
\\[y = \\beta_0 + \\beta_1 x + \\beta_2 x^2 + \\cdots + \\beta_p x^p + \\varepsilon.\\]
<p>Defining the feature map \\(\\phi(x) = (1, x, x^2, \\dots, x^p)^\\top\\), this is a standard linear regression \\(y = \\phi(x)^\\top\\boldsymbol{\\beta} + \\varepsilon\\).</p>
</div></div>

<h2>Overfitting and Model Complexity</h2>

<p>With \\(p = n - 1\\), the polynomial interpolates every data point exactly (training error = 0), but it oscillates wildly between data points. This is <strong>overfitting</strong>: the model has memorized noise rather than learned the signal. The key diagnostic is the gap between training error and test error.</p>

<div class="env-block theorem"><div class="env-title">Proposition 2.1 (Interpolation)</div><div class="env-body">
<p>Given \\(n\\) data points with distinct \\(x_i\\) values, a polynomial of degree \\(n-1\\) passes through all of them. The design matrix \\(\\mathbf{X}\\) (a Vandermonde matrix) is invertible when all \\(x_i\\) are distinct.</p>
</div></div>

<div class="env-block remark"><div class="env-title">The Bias-Variance Tradeoff in Action</div><div class="env-body"><p>Low-degree polynomials have high bias (they cannot capture the true pattern) but low variance (they are stable under resampling). High-degree polynomials have low bias but high variance. The optimal degree balances both, minimizing expected test error. We will formalize this in Chapter 3.</p></div></div>

<div class="viz-placeholder" data-viz="viz-polynomial"></div>
`,
      visualizations: [
        {
          id: 'viz-polynomial',
          title: 'Polynomial Degree vs. Overfitting',
          description: 'Adjust the polynomial degree to see how the fit changes. The blue curve fits the training points (teal). The red dashed curve shows the true function. The bar chart displays training error (blue) and test error (orange).',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 50, originX: 60, originY: 320 });
            const ctx = viz.ctx;
            // True function: sin(x) + 0.3x
            const trueF = x => Math.sin(x*1.2) + 0.3*x + 1;
            // Generate training data with noise
            const rng = (() => { let s=42; return ()=>{s=(s*1103515245+12345)&0x7fffffff;return s/0x7fffffff;}; })();
            const nTrain = 10, nTest = 50;
            const trainX = [], trainY = [];
            for (let i=0; i<nTrain; i++) { const x=0.5+i*0.7; trainX.push(x); trainY.push(trueF(x)+0.4*(rng()-0.5)*2); }
            const testX=[], testY=[];
            for (let i=0; i<nTest; i++) { const x=0.3+i*0.15; testX.push(x); testY.push(trueF(x)+0.4*(rng()-0.5)*2); }

            let degree = 1;
            VizEngine.createSlider(controls, 'Degree', 1, 9, 1, 1, v => { degree = Math.round(v); draw(); });

            function polyFit(xs, ys, deg) {
              const n = xs.length, m = deg+1;
              // Build Vandermonde
              const X = [];
              for (let i=0; i<n; i++) { const row=[]; for (let j=0; j<m; j++) row.push(Math.pow(xs[i],j)); X.push(row); }
              // Normal equations: XtX * beta = Xt * y
              const XtX = Array.from({length:m},()=>Array(m).fill(0));
              const Xty = Array(m).fill(0);
              for (let i=0; i<n; i++) for (let j=0; j<m; j++) { Xty[j] += X[i][j]*ys[i]; for (let k=0; k<m; k++) XtX[j][k] += X[i][j]*X[i][k]; }
              // Add small ridge for stability
              for (let j=0; j<m; j++) XtX[j][j] += 1e-8;
              // Solve via Gaussian elimination
              const A = XtX.map((r,i)=>[...r, Xty[i]]);
              for (let col=0; col<m; col++) {
                let mx=col; for (let r=col+1;r<m;r++) if (Math.abs(A[r][col])>Math.abs(A[mx][col])) mx=r;
                [A[col],A[mx]] = [A[mx],A[col]];
                if (Math.abs(A[col][col])<1e-14) continue;
                for (let r=col+1;r<m;r++) { const f=A[r][col]/A[col][col]; for (let c=col;c<=m;c++) A[r][c]-=f*A[col][c]; }
              }
              const beta = Array(m).fill(0);
              for (let i=m-1;i>=0;i--) { let s=A[i][m]; for (let j=i+1;j<m;j++) s-=A[i][j]*beta[j]; beta[i]=s/A[i][i]; }
              return beta;
            }
            function polyEval(beta, x) { let v=0; for (let j=0;j<beta.length;j++) v+=beta[j]*Math.pow(x,j); return v; }
            function mse(beta, xs, ys) { let s=0; for (let i=0;i<xs.length;i++) s+=(ys[i]-polyEval(beta,xs[i]))**2; return s/xs.length; }

            function draw() {
              viz.clear(); viz.drawGrid(1); viz.drawAxes();
              const beta = polyFit(trainX, trainY, degree);
              const trainErr = mse(beta, trainX, trainY);
              const testErr = mse(beta, testX, testY);
              // Draw true function
              ctx.strokeStyle = viz.colors.red; ctx.lineWidth = 1.5; ctx.setLineDash([6,4]);
              ctx.beginPath();
              for (let px=0; px<=viz.width; px++) {
                const [x] = viz.toMath(px, 0);
                const y = trueF(x);
                const [,sy] = viz.toScreen(0, y);
                px===0 ? ctx.moveTo(px, sy) : ctx.lineTo(px, sy);
              }
              ctx.stroke(); ctx.setLineDash([]);
              // Draw polynomial fit
              ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 2.5;
              ctx.beginPath();
              for (let px=0; px<=viz.width; px++) {
                const [x] = viz.toMath(px, 0);
                const y = polyEval(beta, x);
                const [,sy] = viz.toScreen(0, Math.max(-5, Math.min(15, y)));
                px===0 ? ctx.moveTo(px, sy) : ctx.lineTo(px, sy);
              }
              ctx.stroke();
              // Draw training data
              for (let i=0; i<nTrain; i++) viz.drawPoint(trainX[i], trainY[i], viz.colors.teal, null, 5);
              // Error bars (top-right corner)
              const bx = viz.width-160, by = 20, bw = 55, bh = 80;
              ctx.fillStyle = '#1a1a40cc'; ctx.fillRect(bx-10, by-5, 160, bh+25);
              const maxE = Math.max(trainErr, testErr, 0.5);
              const th = Math.min(bh, trainErr/maxE*bh);
              const teh = Math.min(bh, testErr/maxE*bh);
              ctx.fillStyle = viz.colors.blue; ctx.fillRect(bx, by+bh-th, bw, th);
              ctx.fillStyle = viz.colors.orange; ctx.fillRect(bx+bw+10, by+bh-teh, bw, teh);
              ctx.fillStyle = viz.colors.text; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
              ctx.fillText('Train', bx+bw/2, by+bh+12);
              ctx.fillText('Test', bx+bw+10+bw/2, by+bh+12);
              ctx.fillText(trainErr.toFixed(3), bx+bw/2, by+bh-th-4);
              ctx.fillText(testErr.toFixed(3), bx+bw+10+bw/2, by+bh-teh-4);
              viz.screenText('Degree '+degree, viz.width/2, 16, viz.colors.white, 13);
            }
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'For \\(n\\) data points, what is the training RSS of a degree \\(n-1\\) polynomial? Why is this not useful for prediction?',
          hint: 'A degree \\(n-1\\) polynomial has \\(n\\) free parameters.',
          solution: 'A degree \\(n-1\\) polynomial has \\(n\\) coefficients and passes through all \\(n\\) points exactly, so the training RSS is 0. However, this is pure interpolation: the polynomial captures all the noise in the training data and will oscillate wildly on new inputs, producing terrible test error. Zero training error does not imply good generalization.'
        },
        {
          question: 'Show that polynomial regression of degree \\(p\\) is a special case of multiple linear regression with \\(d = p\\) features.',
          hint: 'Define the feature map and write out the design matrix.',
          solution: 'Define \\(z_{ij} = x_i^j\\) for \\(j = 0, 1, \\dots, p\\). The design matrix \\(\\mathbf{X}\\) has entries \\(X_{ij} = x_i^j\\), which is the Vandermonde matrix. Then \\(y_i = \\sum_{j=0}^{p} \\beta_j x_i^j = \\mathbf{x}_i^\\top \\boldsymbol{\\beta}\\), which is exactly the multiple linear regression model with \\(d = p\\) features \\((1, x_i, x_i^2, \\dots, x_i^p)\\). All results (normal equations, hat matrix, etc.) apply directly.'
        },
        {
          question: 'If you fit polynomials of degree 1 through 15 to a dataset of 20 points, sketch the expected behavior of training MSE and test MSE as functions of degree. Where is the optimal degree?',
          hint: 'Training error always decreases (or stays the same) with more parameters. Test error has a U-shape.',
          solution: 'Training MSE is monotonically non-increasing: adding a term can only reduce training error (or leave it unchanged) since the lower-degree model is nested. It reaches 0 at degree 19. Test MSE initially decreases as the model captures the true relationship, reaches a minimum at some intermediate degree (the "sweet spot"), then increases as overfitting dominates. The optimal degree is at the minimum of the test MSE curve. This U-shape is the bias-variance tradeoff in action.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 5: Probabilistic View
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch02-sec05',
      title: 'Probabilistic View',
      content: `
<div class="env-block intuition"><div class="env-title">Why Probability?</div><div class="env-body"><p>So far, least squares looks like pure geometry. But it has a deep probabilistic interpretation: OLS is the maximum likelihood estimator under Gaussian noise, and regularized regression corresponds to maximum a posteriori estimation with specific priors. This connection unlocks uncertainty quantification and Bayesian extensions.</p></div></div>

<h2>MLE = Least Squares</h2>

<p>Assume the data-generating process is</p>
\\[y_i = \\mathbf{x}_i^\\top \\boldsymbol{\\beta} + \\varepsilon_i, \\qquad \\varepsilon_i \\stackrel{\\text{iid}}{\\sim} N(0, \\sigma^2).\\]

<p>The likelihood of the data given \\(\\boldsymbol{\\beta}\\) is</p>
\\[L(\\boldsymbol{\\beta}) = \\prod_{i=1}^n \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\!\\left(-\\frac{(y_i - \\mathbf{x}_i^\\top\\boldsymbol{\\beta})^2}{2\\sigma^2}\\right).\\]

<div class="env-block theorem"><div class="env-title">Theorem 2.3 (MLE = OLS)</div><div class="env-body">
<p>The log-likelihood is</p>
\\[\\ell(\\boldsymbol{\\beta}) = -\\frac{n}{2}\\log(2\\pi\\sigma^2) - \\frac{1}{2\\sigma^2}\\sum_{i=1}^n(y_i - \\mathbf{x}_i^\\top\\boldsymbol{\\beta})^2.\\]
<p>Maximizing \\(\\ell\\) over \\(\\boldsymbol{\\beta}\\) is equivalent to minimizing \\(\\sum(y_i - \\mathbf{x}_i^\\top\\boldsymbol{\\beta})^2\\), so the MLE is the OLS estimator.</p>
</div></div>

<h2>MAP = Regularized Regression</h2>

<p>Now place a prior on \\(\\boldsymbol{\\beta}\\). The posterior is \\(p(\\boldsymbol{\\beta}|\\mathbf{y}) \\propto L(\\boldsymbol{\\beta})\\, p(\\boldsymbol{\\beta})\\), and the <strong>maximum a posteriori</strong> (MAP) estimate maximizes \\(\\log L + \\log p\\).</p>

<div class="env-block theorem"><div class="env-title">Theorem 2.4 (MAP Correspondence)</div><div class="env-body">
<ul>
<li><strong>Gaussian prior</strong>: \\(\\beta_j \\stackrel{\\text{iid}}{\\sim} N(0, \\tau^2)\\). Then \\(-\\log p(\\boldsymbol{\\beta}) = \\frac{1}{2\\tau^2}\\|\\boldsymbol{\\beta}\\|_2^2 + \\text{const}\\). The MAP estimate is the <strong>ridge</strong> solution with \\(\\lambda = \\sigma^2/\\tau^2\\).</li>
<li><strong>Laplace prior</strong>: \\(\\beta_j \\stackrel{\\text{iid}}{\\sim} \\text{Laplace}(0, b)\\). Then \\(-\\log p(\\boldsymbol{\\beta}) = \\frac{1}{b}\\|\\boldsymbol{\\beta}\\|_1 + \\text{const}\\). The MAP estimate is the <strong>Lasso</strong> solution with \\(\\lambda = 2\\sigma^2/b\\).</li>
</ul>
</div></div>

<div class="env-block remark"><div class="env-title">Beyond Point Estimates</div><div class="env-body"><p>MAP gives a single "best" \\(\\boldsymbol{\\beta}\\), but the full Bayesian approach computes the entire posterior \\(p(\\boldsymbol{\\beta}|\\mathbf{y})\\). For Gaussian likelihood and Gaussian prior, the posterior is also Gaussian, with a closed-form mean and covariance. This is the subject of Chapter 14 (Bayesian Linear Regression).</p></div></div>

<div class="viz-placeholder" data-viz="viz-probabilistic"></div>
`,
      visualizations: [
        {
          id: 'viz-probabilistic',
          title: 'Likelihood Contours and MAP Prior',
          description: 'Blue ellipses show the Gaussian likelihood contours (data evidence). The red circle/diamond shows the prior constraint (Gaussian for Ridge, Laplace for Lasso). The MAP estimate (star) is where the prior and likelihood first meet. Toggle between Ridge and Lasso priors.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 55, originX: 280, originY: 230 });
            const ctx = viz.ctx;
            let mode = 0; // 0 = ridge, 1 = lasso
            let priorStrength = 1.5;
            VizEngine.createButton(controls, 'Ridge (Gaussian)', () => { mode = 0; draw(); });
            VizEngine.createButton(controls, 'Lasso (Laplace)', () => { mode = 1; draw(); });
            VizEngine.createSlider(controls, 'Prior strength', 0.3, 4, 1.5, 0.1, v => { priorStrength = v; draw(); });

            // MLE at (2.5, 1.5), likelihood contours as ellipses
            const mle = [2.5, 1.5];
            // XtX eigenvalues (determines contour shape)
            const l1 = 1.0, l2 = 0.4;
            const angle = 0.3; // radians

            function draw() {
              viz.clear(); viz.drawGrid(1); viz.drawAxes();
              // Draw likelihood contours
              for (let lev of [0.5, 1.5, 3, 5, 8, 12]) {
                const rx = Math.sqrt(lev/l1), ry = Math.sqrt(lev/l2);
                const [sx, sy] = viz.toScreen(mle[0], mle[1]);
                ctx.strokeStyle = `rgba(88,166,255,${Math.max(0.1,0.4-lev*0.02)})`;
                ctx.lineWidth = 1.5;
                ctx.save(); ctx.translate(sx, sy); ctx.rotate(-angle);
                ctx.beginPath(); ctx.ellipse(0, 0, rx*viz.scale, ry*viz.scale, 0, 0, Math.PI*2); ctx.stroke();
                ctx.restore();
              }
              // Draw prior
              const [ox, oy] = viz.toScreen(0, 0);
              if (mode === 0) {
                // Gaussian prior: circle
                for (let k=1; k<=3; k++) {
                  const r = priorStrength * k * 0.5;
                  ctx.strokeStyle = `rgba(248,81,73,${0.5-k*0.12})`;
                  ctx.lineWidth = 1.5;
                  ctx.beginPath(); ctx.arc(ox, oy, r*viz.scale, 0, Math.PI*2); ctx.stroke();
                }
                // Fill smallest
                ctx.fillStyle = 'rgba(248,81,73,0.06)';
                ctx.beginPath(); ctx.arc(ox, oy, priorStrength*0.5*viz.scale, 0, Math.PI*2); ctx.fill();
              } else {
                // Laplace prior: diamond |b1|+|b2| <= t
                for (let k=1; k<=3; k++) {
                  const t = priorStrength * k * 0.5;
                  ctx.strokeStyle = `rgba(248,81,73,${0.5-k*0.12})`;
                  ctx.lineWidth = 1.5;
                  const pts = [[t,0],[0,t],[-t,0],[0,-t]];
                  ctx.beginPath();
                  pts.forEach(([px,py],i) => {
                    const [sx2,sy2] = viz.toScreen(px,py);
                    i===0 ? ctx.moveTo(sx2,sy2) : ctx.lineTo(sx2,sy2);
                  });
                  ctx.closePath(); ctx.stroke();
                }
                // Fill smallest
                const t0 = priorStrength*0.5;
                ctx.fillStyle = 'rgba(248,81,73,0.06)';
                ctx.beginPath();
                [[t0,0],[0,t0],[-t0,0],[0,-t0]].forEach(([px,py],i) => {
                  const [sx2,sy2] = viz.toScreen(px,py);
                  i===0 ? ctx.moveTo(sx2,sy2) : ctx.lineTo(sx2,sy2);
                });
                ctx.closePath(); ctx.fill();
              }
              // Compute approximate MAP location
              // Ridge: beta_map = (XtX + lambda*I)^-1 XtX beta_mle (simplified for display)
              const lam = 1.0 / (priorStrength * priorStrength);
              let mapX, mapY;
              if (mode === 0) {
                // Ridge shrinks toward origin
                const ca = Math.cos(angle), sa = Math.sin(angle);
                // Transform MLE to eigen-coordinates, shrink, transform back
                const u = ca*mle[0]+sa*mle[1], v = -sa*mle[0]+ca*mle[1];
                const u2 = u*l1/(l1+lam), v2 = v*l2/(l2+lam);
                mapX = ca*u2-sa*v2; mapY = sa*u2+ca*v2;
              } else {
                // Lasso: approximate soft threshold in eigen-coords
                const ca = Math.cos(angle), sa = Math.sin(angle);
                const u = ca*mle[0]+sa*mle[1], v = -sa*mle[0]+ca*mle[1];
                const st = t => Math.sign(t)*Math.max(0, Math.abs(t)-lam*0.5);
                const u2 = st(u), v2 = st(v);
                mapX = ca*u2-sa*v2; mapY = sa*u2+ca*v2;
              }
              // Mark MLE
              const [mleSx, mleSy] = viz.toScreen(mle[0], mle[1]);
              ctx.fillStyle = viz.colors.blue; ctx.font = '16px sans-serif'; ctx.textAlign = 'center';
              ctx.fillText('\u25CF', mleSx, mleSy);
              viz.screenText('MLE', mleSx+14, mleSy-10, viz.colors.blue, 11, 'left');
              // Mark MAP
              const [mapSx, mapSy] = viz.toScreen(mapX, mapY);
              ctx.fillStyle = viz.colors.yellow; ctx.font = '18px sans-serif';
              ctx.fillText('\u2605', mapSx, mapSy);
              viz.screenText('MAP', mapSx+14, mapSy-10, viz.colors.yellow, 11, 'left');
              // Labels
              viz.screenText('\u03B2\u2081', viz.width-15, viz.originY-10, viz.colors.text, 13, 'right');
              viz.screenText('\u03B2\u2082', viz.originX+10, 12, viz.colors.text, 13, 'left');
              viz.screenText(mode===0?'Prior: Gaussian (Ridge)':'Prior: Laplace (Lasso)', viz.width/2, viz.height-14, viz.colors.red, 12);
            }
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Derive the MLE for \\(\\sigma^2\\) under the Gaussian noise model. Is it unbiased?',
          hint: 'Differentiate the log-likelihood with respect to \\(\\sigma^2\\) and set to zero.',
          solution: 'Taking \\(\\partial \\ell / \\partial \\sigma^2 = -\\frac{n}{2\\sigma^2} + \\frac{1}{2\\sigma^4}\\sum(y_i - \\mathbf{x}_i^\\top\\hat{\\boldsymbol{\\beta}})^2 = 0\\), we get \\(\\hat{\\sigma}^2_{\\text{MLE}} = \\frac{1}{n}\\sum e_i^2 = \\text{RSS}/n\\). This is biased: \\(\\mathbb{E}[\\hat{\\sigma}^2_{\\text{MLE}}] = \\frac{n-d-1}{n}\\sigma^2\\). The unbiased estimator divides by \\(n - d - 1\\) instead of \\(n\\), accounting for the \\(d+1\\) parameters estimated.'
        },
        {
          question: 'Show that the Laplace distribution \\(p(\\beta) = \\frac{1}{2b}\\exp(-|\\beta|/b)\\) leads to an L1 penalty in the MAP objective.',
          hint: 'Take the negative log of the prior.',
          solution: 'For independent Laplace priors, \\(p(\\boldsymbol{\\beta}) = \\prod_j \\frac{1}{2b}\\exp(-|\\beta_j|/b)\\). Taking the negative log: \\(-\\log p(\\boldsymbol{\\beta}) = d\\log(2b) + \\frac{1}{b}\\sum_j|\\beta_j| = \\frac{1}{b}\\|\\boldsymbol{\\beta}\\|_1 + \\text{const}\\). The MAP objective \\(\\min_\\beta [-\\ell(\\boldsymbol{\\beta}) - \\log p(\\boldsymbol{\\beta})]\\) becomes \\(\\frac{1}{2\\sigma^2}\\|\\mathbf{y}-\\mathbf{X}\\boldsymbol{\\beta}\\|^2 + \\frac{1}{b}\\|\\boldsymbol{\\beta}\\|_1 + \\text{const}\\), which is the Lasso with \\(\\lambda = 2\\sigma^2/b\\).'
        },
        {
          question: 'If the prior variance \\(\\tau^2 \\to \\infty\\) in the Gaussian prior, what happens to the MAP estimate? What does this mean about "uninformative" priors?',
          hint: 'What is \\(\\lambda = \\sigma^2/\\tau^2\\) in this limit?',
          solution: 'As \\(\\tau^2 \\to \\infty\\), the regularization parameter \\(\\lambda = \\sigma^2/\\tau^2 \\to 0\\). The ridge penalty vanishes, and the MAP estimate converges to the MLE (OLS). A very diffuse prior effectively says "I have no prior information about \\(\\boldsymbol{\\beta}\\)," so the data likelihood dominates entirely. This connects the frequentist and Bayesian viewpoints: the MLE is the MAP estimate under a flat (improper) prior.'
        }
      ]
    }
  ]
});
