window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
  id: 'ch04',
  number: 4,
  title: 'Linear Classification',
  subtitle: 'Decision Boundaries, Perceptrons, Fisher Discriminants, and Multi-class Methods',
  sections: [

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 1: Decision Boundaries
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch04-sec01',
      title: 'Decision Boundaries',
      content: `
<h2>Decision Boundaries</h2>

<p>Classification assigns each input \\(\\mathbf{x} \\in \\mathbb{R}^d\\) to one of \\(K\\) discrete classes. A <strong>linear classifier</strong> partitions the input space using hyperplanes, producing <strong>decision boundaries</strong> that are affine subspaces of dimension \\(d-1\\).</p>

<div class="env-block definition"><div class="env-title">Definition 4.1 (Linear Decision Boundary)</div><div class="env-body">
<p>A <strong>linear decision boundary</strong> in \\(\\mathbb{R}^d\\) is the set</p>
\\[\\mathcal{H} = \\{\\mathbf{x} \\in \\mathbb{R}^d : \\mathbf{w}^\\top \\mathbf{x} + b = 0\\},\\]
<p>where \\(\\mathbf{w} \\in \\mathbb{R}^d\\) is the <strong>weight vector</strong> (normal to the boundary) and \\(b \\in \\mathbb{R}\\) is the <strong>bias</strong>. A point \\(\\mathbf{x}\\) is classified as</p>
\\[y(\\mathbf{x}) = \\begin{cases} +1 & \\text{if } \\mathbf{w}^\\top \\mathbf{x} + b \\geq 0, \\\\ -1 & \\text{if } \\mathbf{w}^\\top \\mathbf{x} + b &lt; 0. \\end{cases}\\]
</div></div>

<h3>Geometry of the Weight Vector</h3>

<p>The weight vector \\(\\mathbf{w}\\) encodes two pieces of information. Its <em>direction</em> determines the orientation of the decision boundary (\\(\\mathbf{w}\\) is always perpendicular to \\(\\mathcal{H}\\)), and its <em>magnitude</em> together with \\(b\\) determines the signed distance from any point to the boundary:</p>
\\[\\text{dist}(\\mathbf{x}, \\mathcal{H}) = \\frac{\\mathbf{w}^\\top \\mathbf{x} + b}{\\|\\mathbf{w}\\|}.\\]
<p>Points with positive signed distance fall on the \\(+1\\) side; points with negative signed distance fall on the \\(-1\\) side. The perpendicular distance from the origin to the boundary is \\(|b|/\\|\\mathbf{w}\\|\\).</p>

<div class="env-block intuition"><div class="env-title">Why Linear Boundaries?</div><div class="env-body"><p>Linear classifiers are the simplest non-trivial classifiers. Despite their simplicity, they form the backbone of many powerful methods. Logistic regression, SVMs, and even neural networks in their final layer all rely on linear decision boundaries. Understanding the geometry of a single hyperplane is essential before studying more complex models.</p></div></div>

<div class="viz-placeholder" data-viz="viz-decision-boundary"></div>

<h3>Classification Regions</h3>

<p>A linear decision boundary divides \\(\\mathbb{R}^d\\) into exactly two <strong>half-spaces</strong>:</p>
\\[\\mathcal{R}_+ = \\{\\mathbf{x} : \\mathbf{w}^\\top \\mathbf{x} + b \\geq 0\\}, \\quad \\mathcal{R}_- = \\{\\mathbf{x} : \\mathbf{w}^\\top \\mathbf{x} + b &lt; 0\\}.\\]
<p>Each region is <em>convex</em>: any line segment connecting two points in the same region lies entirely within that region. This convexity has important consequences for what patterns a linear classifier can and cannot represent.</p>

<div class="env-block remark"><div class="env-title">Remark</div><div class="env-body"><p>The convexity of decision regions immediately implies that a linear classifier cannot learn the XOR function, since XOR requires non-convex regions. This limitation motivates the study of nonlinear classifiers and multi-layer models.</p></div></div>

<div class="env-block example"><div class="env-title">Example 4.1</div><div class="env-body">
<p>In \\(\\mathbb{R}^2\\) with \\(\\mathbf{w} = (2, -1)^\\top\\) and \\(b = 1\\), the decision boundary is the line \\(2x_1 - x_2 + 1 = 0\\), equivalently \\(x_2 = 2x_1 + 1\\). The weight vector \\((2, -1)\\) points into the \\(+1\\) half-space. For instance, the origin \\((0,0)\\) satisfies \\(0 + 0 + 1 = 1 &gt; 0\\), so it lies in \\(\\mathcal{R}_+\\).</p>
</div></div>
`,
      visualizations: [
        {
          id: 'viz-decision-boundary',
          title: 'Draggable Decision Boundary',
          description: 'Drag the red handle on the decision boundary line to reposition it. The weight vector (white arrow) is always perpendicular to the boundary. Blue region = class +1, red region = class -1. Points are colored by their true label; a yellow cross marks misclassifications.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 50 });
            const ctx = viz.ctx;

            const data = [
              {x:1.2,y:2.0,c:1},{x:2.5,y:1.5,c:1},{x:1.8,y:3.0,c:1},{x:3.0,y:2.2,c:1},
              {x:2.0,y:0.5,c:1},{x:3.5,y:1.0,c:1},{x:0.8,y:1.5,c:1},
              {x:-1.5,y:-0.5,c:-1},{x:-2.0,y:1.0,c:-1},{x:-0.5,y:-1.5,c:-1},
              {x:-1.0,y:-2.0,c:-1},{x:-2.5,y:-1.0,c:-1},{x:-1.8,y:0.3,c:-1},{x:-0.2,y:-2.5,c:-1}
            ];

            let angle = Math.PI / 4;
            let offset = 0;
            const handle = viz.addDraggable('bnd', 0, 0, viz.colors.red, 10, (mx, my) => {
              angle = Math.atan2(my, mx) + Math.PI / 2;
              offset = Math.cos(angle) * mx + Math.sin(angle) * my;
              draw();
            });

            function draw() {
              viz.clear(); viz.drawGrid(); viz.drawAxes();
              const w1 = Math.cos(angle), w2 = Math.sin(angle);
              const b = -offset;

              // Shade half-planes
              const norm = Math.sqrt(w1*w1 + w2*w2);
              const perpX = -w2, perpY = w1;
              const cx = -b * w1, cy = -b * w2;
              const [sx1, sy1] = viz.toScreen(cx - perpX*10, cy - perpY*10);
              const [sx2, sy2] = viz.toScreen(cx + perpX*10, cy + perpY*10);
              const push = 800;
              const snx = w1 * push, sny = -w2 * push;
              ctx.fillStyle = 'rgba(88,166,255,0.07)';
              ctx.beginPath(); ctx.moveTo(sx1,sy1); ctx.lineTo(sx2,sy2);
              ctx.lineTo(sx2+snx,sy2+sny); ctx.lineTo(sx1+snx,sy1+sny); ctx.closePath(); ctx.fill();
              ctx.fillStyle = 'rgba(248,81,73,0.07)';
              ctx.beginPath(); ctx.moveTo(sx1,sy1); ctx.lineTo(sx2,sy2);
              ctx.lineTo(sx2-snx,sy2-sny); ctx.lineTo(sx1-snx,sy1-sny); ctx.closePath(); ctx.fill();

              viz.drawLine(cx - perpX*10, cy - perpY*10, cx + perpX*10, cy + perpY*10, viz.colors.green, 2.5);
              const arrLen = 1.2;
              viz.drawVector(cx, cy, cx + w1*arrLen, cy + w2*arrLen, viz.colors.white, 'w', 2);

              // Data points
              let correct = 0;
              for (const p of data) {
                const pred = (w1*p.x + w2*p.y + b >= 0) ? 1 : -1;
                if (pred === p.c) correct++;
                const col = p.c === 1 ? viz.colors.blue : viz.colors.red;
                const [sx, sy] = viz.toScreen(p.x, p.y);
                ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI*2); ctx.fillStyle = col; ctx.fill();
                if (pred !== p.c) {
                  ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 2;
                  ctx.beginPath(); ctx.moveTo(sx-8,sy-8); ctx.lineTo(sx+8,sy+8);
                  ctx.moveTo(sx+8,sy-8); ctx.lineTo(sx-8,sy+8); ctx.stroke();
                }
              }

              handle.x = cx; handle.y = cy;
              viz.drawDraggables();
              const pct = Math.round(correct / data.length * 100);
              viz.screenText('Accuracy: ' + correct + '/' + data.length + ' (' + pct + '%)', viz.width/2, viz.height-14, correct===data.length ? viz.colors.green : viz.colors.yellow, 12);
            }
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Given \\(\\mathbf{w} = (3, 4)^\\top\\) and \\(b = -10\\), compute the signed distance from the point \\(\\mathbf{x} = (1, 2)^\\top\\) to the decision boundary.',
          hint: 'Use the formula \\(\\text{dist} = (\\mathbf{w}^\\top \\mathbf{x} + b)/\\|\\mathbf{w}\\|\\).',
          solution: 'We have \\(\\mathbf{w}^\\top \\mathbf{x} + b = 3(1) + 4(2) - 10 = 1\\), and \\(\\|\\mathbf{w}\\| = \\sqrt{9+16} = 5\\). So the signed distance is \\(1/5 = 0.2\\). Since this is positive, the point lies on the \\(+1\\) side.'
        },
        {
          question: 'Prove that the decision regions produced by a linear classifier are convex sets.',
          hint: 'Take two points \\(\\mathbf{x}_1, \\mathbf{x}_2\\) both satisfying \\(\\mathbf{w}^\\top \\mathbf{x} + b \\geq 0\\) and show any convex combination does too.',
          solution: 'Let \\(\\mathbf{x}_1, \\mathbf{x}_2 \\in \\mathcal{R}_+\\), so \\(\\mathbf{w}^\\top \\mathbf{x}_1 + b \\geq 0\\) and \\(\\mathbf{w}^\\top \\mathbf{x}_2 + b \\geq 0\\). For any \\(\\lambda \\in [0,1]\\), let \\(\\mathbf{x}_\\lambda = \\lambda \\mathbf{x}_1 + (1-\\lambda)\\mathbf{x}_2\\). Then \\(\\mathbf{w}^\\top \\mathbf{x}_\\lambda + b = \\lambda(\\mathbf{w}^\\top \\mathbf{x}_1 + b) + (1-\\lambda)(\\mathbf{w}^\\top \\mathbf{x}_2 + b) \\geq 0\\). Thus \\(\\mathbf{x}_\\lambda \\in \\mathcal{R}_+\\), proving convexity.'
        },
        {
          question: 'Show that scaling \\((\\mathbf{w}, b) \\mapsto (\\alpha \\mathbf{w}, \\alpha b)\\) for \\(\\alpha &gt; 0\\) does not change the decision boundary or the classification of any point.',
          hint: 'Substitute \\(\\alpha \\mathbf{w}\\) and \\(\\alpha b\\) into the classification rule and factor out \\(\\alpha\\).',
          solution: 'The boundary is \\(\\{\\mathbf{x} : \\alpha \\mathbf{w}^\\top \\mathbf{x} + \\alpha b = 0\\} = \\{\\mathbf{x} : \\alpha(\\mathbf{w}^\\top \\mathbf{x} + b) = 0\\}\\). Since \\(\\alpha &gt; 0\\), this equals \\(\\{\\mathbf{x} : \\mathbf{w}^\\top \\mathbf{x} + b = 0\\}\\). For classification, \\(\\text{sign}(\\alpha(\\mathbf{w}^\\top \\mathbf{x} + b)) = \\text{sign}(\\mathbf{w}^\\top \\mathbf{x} + b)\\) since \\(\\alpha &gt; 0\\). The boundary and all classifications are unchanged.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 2: Perceptron Algorithm
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch04-sec02',
      title: 'Perceptron Algorithm',
      content: `
<h2>The Perceptron Algorithm</h2>

<p>The perceptron algorithm, introduced by Frank Rosenblatt in 1958, is one of the oldest machine learning algorithms. It learns a linear decision boundary by iteratively correcting misclassified points.</p>

<div class="env-block definition"><div class="env-title">Algorithm 4.1 (Perceptron Learning Rule)</div><div class="env-body">
<p>Given training data \\(\\{(\\mathbf{x}_i, y_i)\\}_{i=1}^n\\) with \\(y_i \\in \\{+1, -1\\}\\), the perceptron algorithm proceeds as follows:</p>
<ol>
  <li>Initialize \\(\\mathbf{w} \\leftarrow \\mathbf{0}\\), \\(b \\leftarrow 0\\).</li>
  <li>Repeat until convergence:
    <ul>
      <li>For each \\((\\mathbf{x}_i, y_i)\\): if \\(y_i(\\mathbf{w}^\\top \\mathbf{x}_i + b) \\leq 0\\) (misclassified), update:
        \\[\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta \\, y_i \\, \\mathbf{x}_i, \\quad b \\leftarrow b + \\eta \\, y_i\\]
      </li>
    </ul>
  </li>
</ol>
<p>where \\(\\eta &gt; 0\\) is the <strong>learning rate</strong> (often set to 1).</p>
</div></div>

<h3>Why the Update Works</h3>

<p>Consider a misclassified point \\((\\mathbf{x}_i, y_i)\\) where \\(y_i = +1\\) but \\(\\mathbf{w}^\\top \\mathbf{x}_i + b &lt; 0\\). After the update \\(\\mathbf{w}' = \\mathbf{w} + \\mathbf{x}_i\\), the new score is</p>
\\[\\mathbf{w}'^\\top \\mathbf{x}_i + b' = (\\mathbf{w} + \\mathbf{x}_i)^\\top \\mathbf{x}_i + (b + 1) = \\mathbf{w}^\\top \\mathbf{x}_i + b + \\|\\mathbf{x}_i\\|^2 + 1.\\]
<p>Since \\(\\|\\mathbf{x}_i\\|^2 + 1 &gt; 0\\), the score moves in the correct direction. Of course, fixing one point may break others, but the convergence theorem guarantees eventual termination for linearly separable data.</p>

<div class="env-block theorem"><div class="env-title">Theorem 4.1 (Perceptron Convergence)</div><div class="env-body">
<p>If the training data is linearly separable with margin \\(\\gamma = \\min_i y_i(\\mathbf{w}^{*\\top} \\mathbf{x}_i + b^*) / \\|\\mathbf{w}^*\\|\\) and \\(R = \\max_i \\|\\mathbf{x}_i\\|\\), then the perceptron algorithm makes at most \\((R/\\gamma)^2\\) mistakes before converging.</p>
</div></div>

<div class="env-block proof"><div class="env-title">Proof sketch</div><div class="env-body">
<p>Let \\(\\mathbf{w}^*\\) be a separating weight vector with \\(\\|\\mathbf{w}^*\\| = 1\\). After \\(t\\) updates on misclassified points \\(\\mathbf{x}_{i_1}, \\dots, \\mathbf{x}_{i_t}\\):</p>
<p><strong>Lower bound:</strong> \\(\\mathbf{w}^{*\\top} \\mathbf{w}_t \\geq t\\gamma\\) (each update increases the dot product by at least \\(\\gamma\\)).</p>
<p><strong>Upper bound:</strong> \\(\\|\\mathbf{w}_t\\|^2 \\leq tR^2\\) (each update adds at most \\(R^2\\)).</p>
<p>By Cauchy-Schwarz: \\(t\\gamma \\leq \\mathbf{w}^{*\\top}\\mathbf{w}_t \\leq \\|\\mathbf{w}_t\\| \\leq \\sqrt{t}R\\), giving \\(t \\leq (R/\\gamma)^2\\).</p>
<div class="qed">&#8718;</div>
</div></div>

<div class="env-block remark"><div class="env-title">Remark</div><div class="env-body"><p>If the data is <em>not</em> linearly separable, the perceptron algorithm never converges. It cycles through different weight configurations indefinitely. This motivates algorithms like logistic regression and SVMs that handle non-separable data gracefully.</p></div></div>

<div class="viz-placeholder" data-viz="viz-perceptron-learn"></div>
`,
      visualizations: [
        {
          id: 'viz-perceptron-learn',
          title: 'Perceptron Learning Animation',
          description: 'Press Step to advance the perceptron by one update. Press Run to animate. Press Reset to start over. Watch the decision boundary adjust each time a misclassified point is encountered.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 50 });
            const ctx = viz.ctx;

            const data = [
              {x:1.5,y:2.5,c:1},{x:2.0,y:1.0,c:1},{x:2.5,y:2.0,c:1},{x:1.0,y:1.5,c:1},
              {x:3.0,y:1.5,c:1},{x:1.8,y:3.2,c:1},
              {x:-1.5,y:-1.0,c:-1},{x:-2.0,y:0.5,c:-1},{x:-0.5,y:-2.0,c:-1},
              {x:-1.0,y:-2.5,c:-1},{x:-2.5,y:-0.5,c:-1},{x:-1.2,y:-1.8,c:-1}
            ];

            let w1 = 0, w2 = 0, bias = 0, step = 0, mistakes = 0;
            let running = false, animId = null;
            let lastMis = -1;

            function findMisclassified() {
              for (let i = 0; i < data.length; i++) {
                const idx = (step + i) % data.length;
                const p = data[idx];
                if (p.c * (w1*p.x + w2*p.y + bias) <= 0) return idx;
              }
              return -1;
            }

            function doStep() {
              const idx = findMisclassified();
              if (idx === -1) { lastMis = -1; running = false; draw(); return false; }
              const p = data[idx];
              w1 += p.c * p.x; w2 += p.c * p.y; bias += p.c;
              step++; mistakes++; lastMis = idx;
              draw();
              return true;
            }

            VizEngine.createButton(controls, 'Step', () => { running = false; doStep(); });
            VizEngine.createButton(controls, 'Run', () => {
              if (running) return;
              running = true;
              (function tick() {
                if (!running) return;
                if (!doStep()) return;
                animId = setTimeout(tick, 400);
              })();
            });
            VizEngine.createButton(controls, 'Reset', () => {
              running = false; if (animId) clearTimeout(animId);
              w1 = 0; w2 = 0; bias = 0; step = 0; mistakes = 0; lastMis = -1;
              draw();
            });

            function draw() {
              viz.clear(); viz.drawGrid(); viz.drawAxes();
              const norm = Math.sqrt(w1*w1 + w2*w2);

              if (norm > 0.01) {
                const perpX = -w2/norm, perpY = w1/norm;
                const cx = -bias*w1/(w1*w1+w2*w2), cy = -bias*w2/(w1*w1+w2*w2);
                const [sx1,sy1] = viz.toScreen(cx-perpX*12, cy-perpY*12);
                const [sx2,sy2] = viz.toScreen(cx+perpX*12, cy+perpY*12);
                const push = 800;
                const snx = (w1/norm)*push, sny = -(w2/norm)*push;
                ctx.fillStyle = 'rgba(88,166,255,0.06)';
                ctx.beginPath(); ctx.moveTo(sx1,sy1); ctx.lineTo(sx2,sy2);
                ctx.lineTo(sx2+snx,sy2+sny); ctx.lineTo(sx1+snx,sy1+sny); ctx.closePath(); ctx.fill();
                ctx.fillStyle = 'rgba(248,81,73,0.06)';
                ctx.beginPath(); ctx.moveTo(sx1,sy1); ctx.lineTo(sx2,sy2);
                ctx.lineTo(sx2-snx,sy2-sny); ctx.lineTo(sx1-snx,sy1-sny); ctx.closePath(); ctx.fill();
                viz.drawLine(cx-perpX*12, cy-perpY*12, cx+perpX*12, cy+perpY*12, viz.colors.green, 2);
              }

              for (let i = 0; i < data.length; i++) {
                const p = data[i];
                const col = p.c === 1 ? viz.colors.blue : viz.colors.red;
                const [sx, sy] = viz.toScreen(p.x, p.y);
                const r = (i === lastMis) ? 9 : 6;
                ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI*2); ctx.fillStyle = col; ctx.fill();
                if (i === lastMis) {
                  ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 2.5;
                  ctx.beginPath(); ctx.arc(sx, sy, 12, 0, Math.PI*2); ctx.stroke();
                }
                const pred = (w1*p.x + w2*p.y + bias > 0) ? 1 : -1;
                if (pred !== p.c) {
                  ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 1.5;
                  ctx.beginPath(); ctx.moveTo(sx-7,sy-7); ctx.lineTo(sx+7,sy+7);
                  ctx.moveTo(sx+7,sy-7); ctx.lineTo(sx-7,sy+7); ctx.stroke();
                }
              }

              const numCorrect = data.filter(p => (w1*p.x+w2*p.y+bias>0?1:-1)===p.c).length;
              const converged = numCorrect === data.length;
              viz.screenText('Step: ' + step + '  Mistakes: ' + mistakes + '  Correct: ' + numCorrect + '/' + data.length, viz.width/2, viz.height-14, converged ? viz.colors.green : viz.colors.text, 12);
              if (converged) viz.screenText('Converged!', viz.width/2, 18, viz.colors.green, 16);
            }
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'Starting from \\(\\mathbf{w} = (0,0)^\\top\\) and \\(b = 0\\) with \\(\\eta = 1\\), apply one step of the perceptron update on the misclassified point \\(\\mathbf{x} = (2, 1)^\\top\\), \\(y = +1\\). What are the new weights and bias?',
          hint: 'Check that the point is misclassified (\\(y(\\mathbf{w}^\\top \\mathbf{x} + b) \\leq 0\\)), then apply \\(\\mathbf{w} \\leftarrow \\mathbf{w} + y\\mathbf{x}\\) and \\(b \\leftarrow b + y\\).',
          solution: 'With \\(\\mathbf{w} = (0,0)\\) and \\(b=0\\): \\(\\mathbf{w}^\\top \\mathbf{x} + b = 0\\). Since \\(y \\cdot 0 = 0 \\leq 0\\), the point is misclassified. Update: \\(\\mathbf{w} = (0,0) + 1 \\cdot (2,1) = (2,1)\\), \\(b = 0 + 1 = 1\\).'
        },
        {
          question: 'Explain why the perceptron convergence bound \\(t \\leq (R/\\gamma)^2\\) does not depend on the number of training points \\(n\\) or the dimension \\(d\\).',
          hint: 'Look at what \\(R\\) and \\(\\gamma\\) measure. They encode the geometry of the data and the margin, not the combinatorial structure.',
          solution: 'The bound depends on \\(R = \\max_i \\|\\mathbf{x}_i\\|\\) (data radius) and \\(\\gamma\\) (margin), which are geometric quantities. A well-separated dataset in 1000 dimensions can converge faster than a barely separable dataset in 2 dimensions. The bound captures the intrinsic difficulty of the problem (how much "room" the separating hyperplane has relative to the data spread), independent of \\(n\\) or \\(d\\).'
        },
        {
          question: 'Show that the perceptron learning rule can be seen as stochastic gradient descent on the <em>perceptron loss</em> \\(L(\\mathbf{w}, b) = \\sum_{i \\in \\mathcal{M}} -y_i(\\mathbf{w}^\\top \\mathbf{x}_i + b)\\), where \\(\\mathcal{M}\\) is the set of misclassified points.',
          hint: 'Differentiate \\(L\\) with respect to \\(\\mathbf{w}\\) and \\(b\\).',
          solution: 'The perceptron loss is \\(L = \\sum_{i \\in \\mathcal{M}} -y_i(\\mathbf{w}^\\top \\mathbf{x}_i + b)\\). Taking gradients: \\(\\nabla_{\\mathbf{w}} L = -\\sum_{i \\in \\mathcal{M}} y_i \\mathbf{x}_i\\) and \\(\\partial L / \\partial b = -\\sum_{i \\in \\mathcal{M}} y_i\\). Stochastic gradient descent on a single misclassified sample \\(i\\) gives the update \\(\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta y_i \\mathbf{x}_i\\), \\(b \\leftarrow b + \\eta y_i\\), which is exactly the perceptron rule.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 3: Fisher's Linear Discriminant
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch04-sec03',
      title: "Fisher's Linear Discriminant",
      content: `
<h2>Fisher's Linear Discriminant</h2>

<p>Rather than learning a decision boundary directly, Fisher's approach asks: what is the <strong>best direction</strong> onto which to project the data so that the two classes are maximally separated? This transforms a \\(d\\)-dimensional classification problem into a one-dimensional one.</p>

<div class="env-block definition"><div class="env-title">Definition 4.2 (Fisher's Criterion)</div><div class="env-body">
<p>Given two classes with means \\(\\boldsymbol{\\mu}_1, \\boldsymbol{\\mu}_2 \\in \\mathbb{R}^d\\) and within-class scatter matrices</p>
\\[\\mathbf{S}_k = \\sum_{i \\in C_k} (\\mathbf{x}_i - \\boldsymbol{\\mu}_k)(\\mathbf{x}_i - \\boldsymbol{\\mu}_k)^\\top, \\quad \\mathbf{S}_W = \\mathbf{S}_1 + \\mathbf{S}_2,\\]
<p><strong>Fisher's criterion</strong> seeks the projection direction \\(\\mathbf{w}\\) that maximizes</p>
\\[J(\\mathbf{w}) = \\frac{(\\mathbf{w}^\\top \\boldsymbol{\\mu}_1 - \\mathbf{w}^\\top \\boldsymbol{\\mu}_2)^2}{\\mathbf{w}^\\top \\mathbf{S}_W \\mathbf{w}} = \\frac{\\mathbf{w}^\\top \\mathbf{S}_B \\mathbf{w}}{\\mathbf{w}^\\top \\mathbf{S}_W \\mathbf{w}},\\]
<p>where \\(\\mathbf{S}_B = (\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2)(\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2)^\\top\\) is the <strong>between-class scatter matrix</strong>.</p>
</div></div>

<h3>Optimal Projection Direction</h3>

<p>The ratio \\(J(\\mathbf{w})\\) is a <strong>generalized Rayleigh quotient</strong>. Setting its gradient to zero leads to the generalized eigenvalue problem \\(\\mathbf{S}_B \\mathbf{w} = \\lambda \\mathbf{S}_W \\mathbf{w}\\). Since \\(\\mathbf{S}_B \\mathbf{w}\\) is always proportional to \\(\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2\\) (because \\(\\mathbf{S}_B\\) has rank 1), the optimal direction has a closed-form solution.</p>

<div class="env-block theorem"><div class="env-title">Theorem 4.2 (Fisher's Optimal Direction)</div><div class="env-body">
<p>The direction \\(\\mathbf{w}^*\\) that maximizes \\(J(\\mathbf{w})\\) is</p>
\\[\\mathbf{w}^* \\propto \\mathbf{S}_W^{-1}(\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2).\\]
</div></div>

<div class="env-block proof"><div class="env-title">Proof</div><div class="env-body">
<p>Differentiating \\(J(\\mathbf{w})\\) and setting to zero: \\((\\mathbf{w}^\\top \\mathbf{S}_W \\mathbf{w}) \\mathbf{S}_B \\mathbf{w} = (\\mathbf{w}^\\top \\mathbf{S}_B \\mathbf{w}) \\mathbf{S}_W \\mathbf{w}\\). Since \\(\\mathbf{S}_B \\mathbf{w} = (\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2)(\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2)^\\top \\mathbf{w}\\), this is proportional to \\(\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2\\). Substituting and noting that the scalar prefactors do not affect the direction: \\(\\mathbf{S}_W \\mathbf{w} \\propto \\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2\\), giving \\(\\mathbf{w} \\propto \\mathbf{S}_W^{-1}(\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2)\\).</p>
<div class="qed">&#8718;</div>
</div></div>

<div class="env-block intuition"><div class="env-title">Intuition</div><div class="env-body"><p>Projecting onto \\(\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2\\) (the line connecting class means) maximizes between-class separation but ignores within-class spread. If one class is elongated along this direction, the projections overlap heavily. Fisher's discriminant "whitens" the data by \\(\\mathbf{S}_W^{-1}\\), adjusting the projection to account for class covariance structure.</p></div></div>

<div class="viz-placeholder" data-viz="viz-fisher-discriminant"></div>

<div class="env-block remark"><div class="env-title">Connection to LDA</div><div class="env-body"><p>Fisher's Linear Discriminant is equivalent to <strong>Linear Discriminant Analysis</strong> (LDA) under the assumption that both classes have equal covariance matrices. When the class-conditional distributions are Gaussian with \\(\\Sigma_1 = \\Sigma_2 = \\Sigma\\), the Bayes-optimal boundary is linear and Fisher's direction coincides with it.</p></div></div>
`,
      visualizations: [
        {
          id: 'viz-fisher-discriminant',
          title: "Fisher's Linear Discriminant Projection",
          description: 'Two classes of data (blue and red) are projected onto the Fisher direction (green line). The bottom strip shows the projected distributions. Drag the class means to see how the optimal projection direction changes.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 40 });
            const ctx = viz.ctx;

            function randn() { let u=0,v=0; while(!u) u=Math.random(); while(!v) v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }

            let mu1x = 2, mu1y = 2, mu2x = -2, mu2y = -1;
            const n = 40;
            let class1 = [], class2 = [];

            function genData() {
              class1 = []; class2 = [];
              for (let i = 0; i < n; i++) {
                class1.push({x: mu1x + randn()*0.8 + randn()*0.3, y: mu1y + randn()*0.3 + randn()*0.8});
                class2.push({x: mu2x + randn()*0.8 - randn()*0.2, y: mu2y - randn()*0.2 + randn()*0.8});
              }
            }
            genData();

            const d1 = viz.addDraggable('m1', mu1x, mu1y, viz.colors.blue, 10, (x,y) => { mu1x=x; mu1y=y; genData(); draw(); });
            const d2 = viz.addDraggable('m2', mu2x, mu2y, viz.colors.red, 10, (x,y) => { mu2x=x; mu2y=y; genData(); draw(); });

            VizEngine.createButton(controls, 'Regenerate', () => { genData(); draw(); });

            function draw() {
              viz.clear(); viz.drawGrid(); viz.drawAxes();

              // Compute means
              let m1 = [0,0], m2 = [0,0];
              for (const p of class1) { m1[0]+=p.x; m1[1]+=p.y; }
              for (const p of class2) { m2[0]+=p.x; m2[1]+=p.y; }
              m1[0]/=n; m1[1]/=n; m2[0]/=n; m2[1]/=n;

              // Within-class scatter
              let sw = [[0,0],[0,0]];
              for (const p of class1) {
                const dx=p.x-m1[0], dy=p.y-m1[1];
                sw[0][0]+=dx*dx; sw[0][1]+=dx*dy; sw[1][0]+=dy*dx; sw[1][1]+=dy*dy;
              }
              for (const p of class2) {
                const dx=p.x-m2[0], dy=p.y-m2[1];
                sw[0][0]+=dx*dx; sw[0][1]+=dx*dy; sw[1][0]+=dy*dx; sw[1][1]+=dy*dy;
              }

              // Fisher direction: S_W^{-1} (m1 - m2)
              const det = sw[0][0]*sw[1][1] - sw[0][1]*sw[1][0];
              const diff = [m1[0]-m2[0], m1[1]-m2[1]];
              let wf;
              if (Math.abs(det) < 1e-10) {
                wf = VizEngine.normalize(diff);
              } else {
                const inv = [[sw[1][1]/det, -sw[0][1]/det], [-sw[1][0]/det, sw[0][0]/det]];
                wf = VizEngine.normalize([inv[0][0]*diff[0]+inv[0][1]*diff[1], inv[1][0]*diff[0]+inv[1][1]*diff[1]]);
              }

              // Draw Fisher direction line
              viz.drawLine(-wf[0]*10, -wf[1]*10, wf[0]*10, wf[1]*10, viz.colors.green, 2);
              viz.drawVector(0, 0, wf[0]*2, wf[1]*2, viz.colors.green, 'w*', 2);

              // Draw data and projections
              const projStrip = viz.height - 45;
              const projScale = 30;
              const projCenter = viz.width / 2;

              // Projection strip background
              ctx.fillStyle = 'rgba(26,26,64,0.8)';
              ctx.fillRect(0, projStrip - 15, viz.width, 30);

              let proj1 = [], proj2 = [];
              for (const p of class1) {
                viz.drawPoint(p.x, p.y, viz.colors.blue + '88', null, 4);
                const t = wf[0]*p.x + wf[1]*p.y;
                proj1.push(t);
                const px = projCenter + t * projScale;
                ctx.fillStyle = viz.colors.blue + '99';
                ctx.fillRect(px-1, projStrip-8, 3, 16);
              }
              for (const p of class2) {
                viz.drawPoint(p.x, p.y, viz.colors.red + '88', null, 4);
                const t = wf[0]*p.x + wf[1]*p.y;
                proj2.push(t);
                const px = projCenter + t * projScale;
                ctx.fillStyle = viz.colors.red + '99';
                ctx.fillRect(px-1, projStrip-8, 3, 16);
              }

              // Projected means
              const pm1 = proj1.reduce((a,b)=>a+b,0)/n;
              const pm2 = proj2.reduce((a,b)=>a+b,0)/n;
              ctx.fillStyle = viz.colors.blue;
              ctx.beginPath(); ctx.arc(projCenter+pm1*projScale, projStrip, 5, 0, Math.PI*2); ctx.fill();
              ctx.fillStyle = viz.colors.red;
              ctx.beginPath(); ctx.arc(projCenter+pm2*projScale, projStrip, 5, 0, Math.PI*2); ctx.fill();

              // Class means in 2D
              viz.drawPoint(m1[0], m1[1], viz.colors.blue, null, 7);
              viz.drawPoint(m2[0], m2[1], viz.colors.red, null, 7);

              d1.x = mu1x; d1.y = mu1y;
              d2.x = mu2x; d2.y = mu2y;
              viz.drawDraggables();

              viz.screenText('Projected distributions', viz.width/2, projStrip+20, viz.colors.text, 11);

              // Fisher ratio
              const var1 = proj1.reduce((s,v)=>s+(v-pm1)**2,0)/n;
              const var2 = proj2.reduce((s,v)=>s+(v-pm2)**2,0)/n;
              const J = (pm1-pm2)**2 / (var1+var2+1e-10);
              viz.screenText('J(w) = ' + J.toFixed(2), viz.width/2, 18, viz.colors.teal, 14);
            }
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'For two classes in \\(\\mathbb{R}^1\\) with means \\(\\mu_1 = 3\\), \\(\\mu_2 = -1\\) and within-class variances \\(\\sigma_1^2 = 2\\), \\(\\sigma_2^2 = 1\\), compute the Fisher criterion value. (In 1D, the projection direction is trivially \\(w=1\\).)',
          hint: 'In 1D: \\(J = (\\mu_1 - \\mu_2)^2 / (\\sigma_1^2 + \\sigma_2^2)\\).',
          solution: '\\(J = (3 - (-1))^2 / (2 + 1) = 16/3 \\approx 5.33\\). A larger \\(J\\) means better class separation relative to within-class spread.'
        },
        {
          question: 'If \\(\\mathbf{S}_W = \\sigma^2 \\mathbf{I}\\) (isotropic within-class scatter), show that Fisher\'s direction simplifies to \\(\\mathbf{w}^* \\propto \\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2\\).',
          hint: 'Compute \\(\\mathbf{S}_W^{-1}\\) when \\(\\mathbf{S}_W = \\sigma^2 \\mathbf{I}\\).',
          solution: 'When \\(\\mathbf{S}_W = \\sigma^2 \\mathbf{I}\\), we have \\(\\mathbf{S}_W^{-1} = \\sigma^{-2} \\mathbf{I}\\). Thus \\(\\mathbf{w}^* \\propto \\mathbf{S}_W^{-1}(\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2) = \\sigma^{-2}(\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2) \\propto \\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2\\). When within-class scatter is isotropic, projecting onto the line connecting the means is already optimal.'
        },
        {
          question: 'Explain why the between-class scatter matrix \\(\\mathbf{S}_B\\) has rank 1 in the two-class case and what this implies for multi-class generalization.',
          hint: '\\(\\mathbf{S}_B = \\mathbf{d}\\mathbf{d}^\\top\\) where \\(\\mathbf{d} = \\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2\\). What is the rank of an outer product?',
          solution: '\\(\\mathbf{S}_B = (\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2)(\\boldsymbol{\\mu}_1 - \\boldsymbol{\\mu}_2)^\\top\\) is an outer product of a vector with itself, so it has rank 1. This means there is only one discriminant direction. For \\(K\\) classes, the generalized between-class scatter has rank at most \\(K-1\\), so we can extract up to \\(K-1\\) discriminant directions, reducing the problem to at most \\(K-1\\) dimensions.'
        }
      ]
    },

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 4: Multi-class Classification
    // ─────────────────────────────────────────────────────────────────────────
    {
      id: 'ch04-sec04',
      title: 'Multi-class Classification',
      content: `
<h2>Multi-class Classification</h2>

<p>When there are \\(K &gt; 2\\) classes, we need to extend binary classification methods. Three principal strategies exist: <strong>one-vs-all</strong>, <strong>one-vs-one</strong>, and the <strong>softmax</strong> (multinomial) approach.</p>

<h3>One-vs-All (OvA)</h3>

<div class="env-block definition"><div class="env-title">Definition 4.3 (One-vs-All)</div><div class="env-body">
<p>Train \\(K\\) binary classifiers \\(f_k(\\mathbf{x}) = \\mathbf{w}_k^\\top \\mathbf{x} + b_k\\), each distinguishing class \\(k\\) from all other classes. Classify a new point by</p>
\\[y(\\mathbf{x}) = \\arg\\max_{k=1,\\dots,K} f_k(\\mathbf{x}).\\]
</div></div>

<p>OvA is simple and widely used, but can produce <strong>ambiguous regions</strong> where multiple classifiers claim the point (or none does). Despite this, it works well in practice, especially with calibrated classifiers.</p>

<h3>One-vs-One (OvO)</h3>

<div class="env-block definition"><div class="env-title">Definition 4.4 (One-vs-One)</div><div class="env-body">
<p>Train \\(\\binom{K}{2}\\) binary classifiers, one for each pair of classes. Classify by majority vote: assign \\(\\mathbf{x}\\) to the class that wins the most pairwise comparisons.</p>
</div></div>

<p>OvO trains more classifiers (\\(K(K-1)/2\\)) but each on a smaller subset of the data. This can be advantageous when the base classifier is expensive to train but benefits from smaller datasets.</p>

<h3>Softmax (Direct Multi-class)</h3>

<div class="env-block definition"><div class="env-title">Definition 4.5 (Softmax Classifier)</div><div class="env-body">
<p>Assign a linear score \\(s_k(\\mathbf{x}) = \\mathbf{w}_k^\\top \\mathbf{x} + b_k\\) to each class \\(k\\). The predicted probability is</p>
\\[p(y = k \\mid \\mathbf{x}) = \\frac{\\exp(s_k(\\mathbf{x}))}{\\sum_{j=1}^{K} \\exp(s_j(\\mathbf{x}))}.\\]
<p>Classification selects \\(y(\\mathbf{x}) = \\arg\\max_k s_k(\\mathbf{x})\\).</p>
</div></div>

<div class="env-block remark"><div class="env-title">Remark</div><div class="env-body"><p>The softmax approach naturally produces a single consistent set of decision regions with no ambiguity. The boundary between classes \\(j\\) and \\(k\\) is \\(\\{\\mathbf{x} : s_j(\\mathbf{x}) = s_k(\\mathbf{x})\\}\\), which is a hyperplane defined by \\((\\mathbf{w}_j - \\mathbf{w}_k)^\\top \\mathbf{x} + (b_j - b_k) = 0\\).</p></div></div>

<div class="env-block theorem"><div class="env-title">Theorem 4.3 (Convexity of Multi-class Regions)</div><div class="env-body">
<p>The decision region for class \\(k\\) in a softmax classifier, \\(\\mathcal{R}_k = \\{\\mathbf{x} : s_k(\\mathbf{x}) \\geq s_j(\\mathbf{x}) \\text{ for all } j\\}\\), is a <strong>convex polytope</strong> (intersection of half-spaces).</p>
</div></div>

<div class="env-block proof"><div class="env-title">Proof</div><div class="env-body">
<p>\\(\\mathcal{R}_k = \\bigcap_{j \\neq k} \\{\\mathbf{x} : s_k(\\mathbf{x}) \\geq s_j(\\mathbf{x})\\} = \\bigcap_{j \\neq k} \\{\\mathbf{x} : (\\mathbf{w}_k - \\mathbf{w}_j)^\\top \\mathbf{x} + (b_k - b_j) \\geq 0\\}\\). Each set in the intersection is a half-space (convex), and the intersection of convex sets is convex.</p>
<div class="qed">&#8718;</div>
</div></div>

<div class="viz-placeholder" data-viz="viz-multiclass-regions"></div>

<div class="env-block example"><div class="env-title">Example 4.2 (Comparing Strategies)</div><div class="env-body">
<p>With \\(K = 10\\) classes (e.g., digit recognition), OvA trains 10 classifiers, OvO trains \\(\\binom{10}{2} = 45\\) classifiers, while softmax trains a single model with a 10-way output. The softmax approach is generally preferred for linear models because it jointly optimizes all class boundaries and provides calibrated probabilities.</p>
</div></div>
`,
      visualizations: [
        {
          id: 'viz-multiclass-regions',
          title: 'Multi-class Decision Regions (3 Classes)',
          description: 'Three linear classifiers partition 2D space into colored regions. Drag the weight handles (colored dots) to change each class boundary. The region for each class is a convex polygon formed by intersecting half-spaces.',
          setup(container, controls) {
            const viz = new VizEngine(container, { scale: 40 });
            const ctx = viz.ctx;

            // 3 class weight vectors and biases (softmax style)
            let params = [
              { w1: 1.5, w2: 0.8, b: 0.5 },
              { w1: -1.0, w2: 1.5, b: 0.3 },
              { w1: -0.3, w2: -1.8, b: -0.2 }
            ];
            const classColors = [viz.colors.blue, viz.colors.teal, viz.colors.orange];
            const classNames = ['A', 'B', 'C'];
            const classAlpha = ['rgba(88,166,255,', 'rgba(63,185,160,', 'rgba(240,136,62,'];

            // Generate some data points per class
            function randn() { let u=0,v=0; while(!u) u=Math.random(); while(!v) v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
            const pts = [];
            const centers = [[2.5,1.5],[-1.5,2.5],[-0.5,-2.5]];
            for (let k = 0; k < 3; k++) {
              for (let i = 0; i < 12; i++) {
                pts.push({x: centers[k][0]+randn()*0.8, y: centers[k][1]+randn()*0.8, c: k});
              }
            }

            VizEngine.createSlider(controls, 'Class A: w\u2081', -3, 3, params[0].w1, 0.1, v => { params[0].w1=v; draw(); });
            VizEngine.createSlider(controls, 'Class A: w\u2082', -3, 3, params[0].w2, 0.1, v => { params[0].w2=v; draw(); });
            VizEngine.createSlider(controls, 'Class B: w\u2081', -3, 3, params[1].w1, 0.1, v => { params[1].w1=v; draw(); });
            VizEngine.createSlider(controls, 'Class B: w\u2082', -3, 3, params[1].w2, 0.1, v => { params[1].w2=v; draw(); });

            function draw() {
              viz.clear();

              // Pixel-level classification for regions
              const step = 4;
              for (let px = 0; px < viz.width; px += step) {
                for (let py = 0; py < viz.height; py += step) {
                  const [mx, my] = viz.toMath(px, py);
                  let bestK = 0, bestS = -Infinity;
                  for (let k = 0; k < 3; k++) {
                    const s = params[k].w1*mx + params[k].w2*my + params[k].b;
                    if (s > bestS) { bestS = s; bestK = k; }
                  }
                  ctx.fillStyle = classAlpha[bestK] + '0.12)';
                  ctx.fillRect(px, py, step, step);
                }
              }

              viz.drawGrid(); viz.drawAxes();

              // Draw pairwise boundaries
              for (let i = 0; i < 3; i++) {
                for (let j = i+1; j < 3; j++) {
                  const dw1 = params[i].w1 - params[j].w1;
                  const dw2 = params[i].w2 - params[j].w2;
                  const db = params[i].b - params[j].b;
                  const norm = Math.sqrt(dw1*dw1 + dw2*dw2);
                  if (norm < 0.01) continue;
                  const perpX = -dw2/norm, perpY = dw1/norm;
                  const cx = -db*dw1/(dw1*dw1+dw2*dw2), cy = -db*dw2/(dw1*dw1+dw2*dw2);
                  viz.drawLine(cx-perpX*10, cy-perpY*10, cx+perpX*10, cy+perpY*10, '#ffffff44', 1.5);
                }
              }

              // Data points
              let correct = 0;
              for (const p of pts) {
                let bestK = 0, bestS = -Infinity;
                for (let k = 0; k < 3; k++) {
                  const s = params[k].w1*p.x + params[k].w2*p.y + params[k].b;
                  if (s > bestS) { bestS = s; bestK = k; }
                }
                if (bestK === p.c) correct++;
                const [sx, sy] = viz.toScreen(p.x, p.y);
                ctx.beginPath(); ctx.arc(sx, sy, 5, 0, Math.PI*2);
                ctx.fillStyle = classColors[p.c]; ctx.fill();
                if (bestK !== p.c) {
                  ctx.strokeStyle = viz.colors.yellow; ctx.lineWidth = 1.5;
                  ctx.beginPath(); ctx.moveTo(sx-6,sy-6); ctx.lineTo(sx+6,sy+6);
                  ctx.moveTo(sx+6,sy-6); ctx.lineTo(sx-6,sy+6); ctx.stroke();
                }
              }

              // Legend
              for (let k = 0; k < 3; k++) {
                ctx.fillStyle = classColors[k];
                ctx.beginPath(); ctx.arc(viz.width-70, 20+k*20, 5, 0, Math.PI*2); ctx.fill();
                ctx.fillStyle = viz.colors.text; ctx.font = '12px -apple-system,sans-serif';
                ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                ctx.fillText('Class ' + classNames[k], viz.width-60, 20+k*20);
              }

              viz.screenText('Accuracy: ' + correct + '/' + pts.length, viz.width/2, viz.height-14, correct===pts.length ? viz.colors.green : viz.colors.text, 12);
            }
            draw();
            return viz;
          }
        }
      ],
      exercises: [
        {
          question: 'In a one-vs-all scheme with \\(K = 4\\) classes, how many binary classifiers are trained? In one-vs-one?',
          hint: 'OvA trains one per class; OvO trains one per pair.',
          solution: 'OvA: \\(K = 4\\) classifiers. OvO: \\(\\binom{4}{2} = 6\\) classifiers. In general, OvA trains \\(K\\) and OvO trains \\(K(K-1)/2\\).'
        },
        {
          question: 'Show that the boundary between classes \\(j\\) and \\(k\\) in a softmax classifier is a hyperplane, and write its equation.',
          hint: 'Set \\(p(y=j \\mid \\mathbf{x}) = p(y=k \\mid \\mathbf{x})\\) and simplify.',
          solution: 'Setting \\(p(j \\mid \\mathbf{x}) = p(k \\mid \\mathbf{x})\\) gives \\(\\exp(s_j) / Z = \\exp(s_k) / Z\\), so \\(s_j(\\mathbf{x}) = s_k(\\mathbf{x})\\). This means \\(\\mathbf{w}_j^\\top \\mathbf{x} + b_j = \\mathbf{w}_k^\\top \\mathbf{x} + b_k\\), or equivalently \\((\\mathbf{w}_j - \\mathbf{w}_k)^\\top \\mathbf{x} + (b_j - b_k) = 0\\), which is a hyperplane with normal \\(\\mathbf{w}_j - \\mathbf{w}_k\\).'
        },
        {
          question: 'Explain why the softmax function is invariant to adding a constant to all scores, i.e., \\(\\text{softmax}(s_1 + c, \\dots, s_K + c) = \\text{softmax}(s_1, \\dots, s_K)\\). What does this imply for the number of free parameters?',
          hint: 'Factor out \\(e^c\\) from numerator and denominator.',
          solution: 'Adding \\(c\\) to all scores: \\(\\frac{e^{s_k+c}}{\\sum_j e^{s_j+c}} = \\frac{e^c e^{s_k}}{e^c \\sum_j e^{s_j}} = \\frac{e^{s_k}}{\\sum_j e^{s_j}}\\). The \\(e^c\\) cancels. This means we can set one class (say class \\(K\\)) as the reference with \\(\\mathbf{w}_K = \\mathbf{0}\\) and \\(b_K = 0\\), reducing the parameters from \\(K(d+1)\\) to \\((K-1)(d+1)\\).'
        }
      ]
    }
  ]
});
