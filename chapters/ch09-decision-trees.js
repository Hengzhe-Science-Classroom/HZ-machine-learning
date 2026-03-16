// === Chapter 9: Decision Trees ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch09',
    number: 9,
    title: 'Decision Trees',
    subtitle: 'Recursive partitioning, splitting criteria, pruning, and CART: interpretable nonlinear models',
    sections: [
        // ========== SECTION 1: Tree Construction ==========
        {
            id: 'ch09-sec01',
            title: 'Tree Construction',
            content: `
<h2>9.1 Tree Construction</h2>

<div class="env-block intuition">
<strong>From Kernels to Trees.</strong>
In Chapter 8, kernel methods achieved non-linear decision boundaries by implicit mappings into high-dimensional spaces. Decision trees take a fundamentally different approach: they recursively partition the input space into axis-aligned rectangular regions, assigning a label (or value) to each region. The result is a model that is inherently interpretable, requires no feature scaling, and handles mixed data types naturally.
</div>

<h3>Recursive Binary Splitting</h3>

<p>A decision tree partitions \\(\\mathcal{X} \\subseteq \\mathbb{R}^d\\) by a sequence of binary tests. At each internal node, we choose a feature \\(j \\in \\{1, \\ldots, d\\}\\) and a threshold \\(t \\in \\mathbb{R}\\), and split the data into two child nodes:</p>

\\[
\\text{Left: } \\{\\mathbf{x} : x_j \\leq t\\}, \\qquad \\text{Right: } \\{\\mathbf{x} : x_j &gt; t\\}.
\\]

<p>Each leaf node stores a prediction: for classification, the majority class among training points in that leaf; for regression, the mean response.</p>

<h3>Choosing the Best Split</h3>

<p>At each node with dataset \\(S\\), we choose the split \\((j, t)\\) that maximally reduces <em>impurity</em>. Define the impurity of a node as \\(H(S)\\). The <strong>information gain</strong> of a split is</p>

\\[
\\Delta(S, j, t) = H(S) - \\frac{|S_L|}{|S|} H(S_L) - \\frac{|S_R|}{|S|} H(S_R),
\\]

<p>where \\(S_L = \\{\\mathbf{x} \\in S : x_j \\leq t\\}\\) and \\(S_R = S \\setminus S_L\\). We select</p>

\\[
(j^*, t^*) = \\arg\\max_{j, t} \\Delta(S, j, t).
\\]

<h3>Impurity Measures for Classification</h3>

<p>Let \\(p_k\\) denote the fraction of class \\(k\\) in node \\(S\\). The two standard impurity measures are:</p>

<div class="env-block definition">
<strong>Definition 9.1.1 (Entropy).</strong>
\\[
H_{\\text{entropy}}(S) = -\\sum_{k=1}^{K} p_k \\log_2 p_k,
\\]
with the convention \\(0 \\log 0 = 0\\).
</div>

<div class="env-block definition">
<strong>Definition 9.1.2 (Gini Impurity).</strong>
\\[
H_{\\text{Gini}}(S) = \\sum_{k=1}^{K} p_k (1 - p_k) = 1 - \\sum_{k=1}^{K} p_k^2.
\\]
</div>

<p>Both measures equal zero when a node is pure (all one class) and are maximized when all classes are equally represented. The key difference: entropy uses logarithms and penalizes mixed nodes more sharply, while Gini is computationally simpler (no logarithms).</p>

<h3>The Greedy Algorithm</h3>

<p>Tree construction proceeds greedily, top-down:</p>
<ol>
<li>Start with the entire dataset at the root.</li>
<li>For each node, try every feature \\(j\\) and every threshold \\(t\\) (typically midpoints between consecutive sorted values).</li>
<li>Pick the \\((j, t)\\) that maximizes \\(\\Delta\\).</li>
<li>Split the data and recurse on each child.</li>
<li>Stop when a stopping criterion is met (e.g., max depth, min samples per leaf, or zero impurity).</li>
</ol>

<div class="env-block remark">
<strong>Complexity.</strong> At each node with \\(m\\) samples, evaluating all splits costs \\(O(d \\cdot m \\log m)\\) (sorting each feature). The total cost over all nodes is \\(O(d \\cdot n \\log^2 n)\\) in the best case, \\(O(d \\cdot n^2 \\log n)\\) in the worst case.
</div>

<div class="viz-placeholder" data-viz="viz-tree-construction"></div>
`,
            visualizations: [
                {
                    id: 'viz-tree-construction',
                    title: 'Decision Tree Growing Step by Step',
                    description: 'Watch a decision tree grow on 2D data. Use the depth slider to control how many levels of splits are made. The left panel shows the data with decision boundaries; the right panel shows the tree structure.',
                    setup(container, controls) {
                        const wrapper = document.createElement('div');
                        wrapper.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;justify-content:center;';
                        container.appendChild(wrapper);

                        const vizL = new VizEngine(wrapper, { width: 280, height: 280, scale: 45, originX: 140, originY: 140 });
                        const vizR = new VizEngine(wrapper, { width: 280, height: 280, scale: 45, originX: 140, originY: 140 });

                        let maxDepth = 0;
                        VizEngine.createSlider(controls, 'Depth', 0, 4, 0, 1, v => { maxDepth = Math.round(v); });

                        // Generate clustered 2D data
                        const rng = (s) => { let v = s; return () => { v = (v * 16807) % 2147483647; return v / 2147483647; }; };
                        const rand = rng(77);
                        const pts = [];
                        // Class 0: top-left and bottom-right
                        for (let i = 0; i < 15; i++) pts.push({ x: -1.5 + rand()*1.2, y: 0.8 + rand()*1.5, c: 0 });
                        for (let i = 0; i < 10; i++) pts.push({ x: 0.8 + rand()*1.5, y: -1.5 + rand()*1.2, c: 0 });
                        // Class 1: center and bottom-left
                        for (let i = 0; i < 15; i++) pts.push({ x: -0.5 + rand()*1.8, y: -0.5 + rand()*1.5, c: 1 });
                        for (let i = 0; i < 10; i++) pts.push({ x: 0.3 + rand()*1.5, y: 0.5 + rand()*1.5, c: 1 });

                        // Build a simple decision tree
                        function buildTree(data, depth, maxD, xMin, xMax, yMin, yMax) {
                            if (depth >= maxD || data.length < 3) {
                                const c0 = data.filter(p => p.c === 0).length;
                                const c1 = data.length - c0;
                                return { leaf: true, cls: c1 > c0 ? 1 : 0, n: data.length, xMin, xMax, yMin, yMax };
                            }
                            // Try all splits
                            let bestGain = -1, bestFeat = 'x', bestThresh = 0;
                            const gini = (arr) => { if (arr.length === 0) return 0; const p = arr.filter(p => p.c === 1).length / arr.length; return 2*p*(1-p); };
                            const parentG = gini(data);

                            for (const feat of ['x', 'y']) {
                                const vals = [...new Set(data.map(p => p[feat]))].sort((a,b) => a-b);
                                for (let i = 0; i < vals.length - 1; i++) {
                                    const t = (vals[i] + vals[i+1]) / 2;
                                    const left = data.filter(p => p[feat] <= t);
                                    const right = data.filter(p => p[feat] > t);
                                    if (left.length === 0 || right.length === 0) continue;
                                    const gain = parentG - (left.length/data.length)*gini(left) - (right.length/data.length)*gini(right);
                                    if (gain > bestGain) { bestGain = gain; bestFeat = feat; bestThresh = t; }
                                }
                            }
                            if (bestGain <= 0) {
                                const c0 = data.filter(p => p.c === 0).length;
                                return { leaf: true, cls: data.length - c0 > c0 ? 1 : 0, n: data.length, xMin, xMax, yMin, yMax };
                            }
                            const leftData = data.filter(p => p[bestFeat] <= bestThresh);
                            const rightData = data.filter(p => p[bestFeat] > bestThresh);
                            const lxMin = xMin, lxMax = bestFeat === 'x' ? bestThresh : xMax;
                            const rxMin = bestFeat === 'x' ? bestThresh : xMin, rxMax = xMax;
                            const lyMin = yMin, lyMax = bestFeat === 'y' ? bestThresh : yMax;
                            const ryMin = bestFeat === 'y' ? bestThresh : yMin, ryMax = yMax;
                            return {
                                leaf: false, feat: bestFeat, thresh: bestThresh,
                                left: buildTree(leftData, depth+1, maxD, lxMin, lxMax, lyMin, lyMax),
                                right: buildTree(rightData, depth+1, maxD, rxMin, rxMax, ryMin, ryMax),
                                xMin, xMax, yMin, yMax
                            };
                        }

                        function getLeaves(node) {
                            if (node.leaf) return [node];
                            return [...getLeaves(node.left), ...getLeaves(node.right)];
                        }

                        function drawSplits(viz, node, depth) {
                            if (node.leaf) return;
                            const ctx = viz.ctx;
                            const col = depth === 0 ? viz.colors.teal : depth === 1 ? viz.colors.purple : depth === 2 ? viz.colors.yellow : viz.colors.pink;
                            ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash([6, 3]);
                            if (node.feat === 'x') {
                                const [sx, sy1] = viz.toScreen(node.thresh, node.yMin);
                                const [, sy2] = viz.toScreen(node.thresh, node.yMax);
                                ctx.beginPath(); ctx.moveTo(sx, sy1); ctx.lineTo(sx, sy2); ctx.stroke();
                            } else {
                                const [sx1, sy] = viz.toScreen(node.xMin, node.thresh);
                                const [sx2] = viz.toScreen(node.xMax, node.thresh);
                                ctx.beginPath(); ctx.moveTo(sx1, sy); ctx.lineTo(sx2, sy); ctx.stroke();
                            }
                            ctx.setLineDash([]);
                            drawSplits(viz, node.left, depth+1);
                            drawSplits(viz, node.right, depth+1);
                        }

                        function drawTreeDiagram(viz, node, x, y, spread, depth) {
                            const ctx = viz.ctx;
                            const rad = 14;
                            if (node.leaf) {
                                const col = node.cls === 0 ? viz.colors.blue : viz.colors.orange;
                                ctx.fillStyle = col + '55'; ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI*2); ctx.fill();
                                ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI*2); ctx.stroke();
                                ctx.fillStyle = viz.colors.white; ctx.font = '10px -apple-system,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                                ctx.fillText('C' + node.cls, x, y);
                                return;
                            }
                            const col = viz.colors.teal;
                            ctx.fillStyle = '#1a1a40'; ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI*2); ctx.fill();
                            ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI*2); ctx.stroke();
                            ctx.fillStyle = viz.colors.white; ctx.font = '9px -apple-system,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                            ctx.fillText(node.feat + '\u2264' + node.thresh.toFixed(1), x, y);

                            const childY = y + 55;
                            const lx = x - spread, rx = x + spread;
                            // Draw edges
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1.5;
                            ctx.beginPath(); ctx.moveTo(x, y + rad); ctx.lineTo(lx, childY - rad); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(x, y + rad); ctx.lineTo(rx, childY - rad); ctx.stroke();
                            // Labels
                            ctx.fillStyle = viz.colors.text; ctx.font = '8px -apple-system,sans-serif';
                            ctx.textAlign = 'right'; ctx.fillText('Y', (x+lx)/2 - 4, (y+childY)/2 - 2);
                            ctx.textAlign = 'left'; ctx.fillText('N', (x+rx)/2 + 4, (y+childY)/2 - 2);

                            drawTreeDiagram(viz, node.left, lx, childY, spread * 0.5, depth+1);
                            drawTreeDiagram(viz, node.right, rx, childY, spread * 0.5, depth+1);
                        }

                        function draw() {
                            const tree = buildTree(pts, 0, maxDepth, -3, 3, -3, 3);

                            // Left: data with boundaries
                            vizL.clear(); vizL.drawGrid(1); vizL.drawAxes();
                            vizL.screenText('Decision Boundaries', 140, 16, vizL.colors.white, 12);

                            // Shade leaves
                            const leaves = getLeaves(tree);
                            for (const lf of leaves) {
                                const col = lf.cls === 0 ? vizL.colors.blue : vizL.colors.orange;
                                const [sx1, sy1] = vizL.toScreen(lf.xMin, lf.yMax);
                                const [sx2, sy2] = vizL.toScreen(lf.xMax, lf.yMin);
                                vizL.ctx.fillStyle = col + '15';
                                vizL.ctx.fillRect(sx1, sy1, sx2-sx1, sy2-sy1);
                            }
                            drawSplits(vizL, tree, 0);
                            for (const p of pts) {
                                const col = p.c === 0 ? vizL.colors.blue : vizL.colors.orange;
                                vizL.drawPoint(p.x, p.y, col, null, 4);
                            }

                            // Right: tree diagram
                            vizR.clear();
                            vizR.ctx.fillStyle = vizR.colors.bg; vizR.ctx.fillRect(0, 0, vizR.width, vizR.height);
                            vizR.screenText('Tree Structure', 140, 16, vizR.colors.white, 12);
                            if (maxDepth === 0) {
                                const c0 = pts.filter(p => p.c === 0).length;
                                const cls = pts.length - c0 > c0 ? 1 : 0;
                                const col = cls === 0 ? vizR.colors.blue : vizR.colors.orange;
                                vizR.ctx.fillStyle = col + '55'; vizR.ctx.beginPath(); vizR.ctx.arc(140, 100, 20, 0, Math.PI*2); vizR.ctx.fill();
                                vizR.ctx.strokeStyle = col; vizR.ctx.lineWidth = 2; vizR.ctx.beginPath(); vizR.ctx.arc(140, 100, 20, 0, Math.PI*2); vizR.ctx.stroke();
                                vizR.ctx.fillStyle = vizR.colors.white; vizR.ctx.font = '11px -apple-system,sans-serif'; vizR.ctx.textAlign = 'center'; vizR.ctx.textBaseline = 'middle';
                                vizR.ctx.fillText('C' + cls, 140, 100);
                                vizR.screenText('(No splits yet)', 140, 150, vizR.colors.text, 11);
                            } else {
                                drawTreeDiagram(vizR, tree, 140, 45, maxDepth >= 3 ? 65 : maxDepth >= 2 ? 60 : 55, 0);
                            }
                        }
                        vizL.animate(draw);
                        return { stopAnimation: () => vizL.stopAnimation() };
                    }
                }
            ],
            exercises: [
                {
                    question: 'Compute the information gain (using Gini impurity) for splitting a node with 10 class-0 and 10 class-1 samples by a threshold that sends 8 class-0 and 2 class-1 to the left, and 2 class-0 and 8 class-1 to the right.',
                    hint: 'Gini of the parent is \\(2 \\cdot 0.5 \\cdot 0.5 = 0.5\\). Compute Gini of each child and take the weighted average.',
                    solution: 'Parent Gini: \\(1 - (10/20)^2 - (10/20)^2 = 0.5\\). Left child (8, 2): \\(1 - (8/10)^2 - (2/10)^2 = 1 - 0.64 - 0.04 = 0.32\\). Right child (2, 8): same by symmetry, \\(0.32\\). Weighted child impurity: \\(\\frac{10}{20}(0.32) + \\frac{10}{20}(0.32) = 0.32\\). Information gain: \\(0.5 - 0.32 = 0.18\\).'
                },
                {
                    question: 'Why is decision tree construction a greedy algorithm? Why not find the globally optimal tree?',
                    hint: 'Think about the number of possible trees and the computational complexity of exhaustive search.',
                    solution: 'Finding the globally optimal decision tree (minimizing training error for a given size) is NP-hard: the number of possible binary trees with \\(d\\) features and \\(n\\) thresholds per feature grows super-exponentially. The greedy approach (picking the locally best split at each node) runs in polynomial time and works well in practice, even though it may not find the global optimum. This is the same reason we use greedy algorithms for other combinatorial optimization problems when exact solutions are intractable.'
                },
                {
                    question: 'Explain why decision trees naturally handle categorical features without one-hot encoding, unlike SVMs or logistic regression.',
                    hint: 'A split on a categorical feature with values \\(\\{A, B, C\\}\\) can test membership in any subset.',
                    solution: 'For a categorical feature with \\(m\\) levels, a tree can split on any subset \\(S \\subseteq \\{1, \\ldots, m\\}\\): left if the feature value is in \\(S\\), right otherwise. There are \\(2^{m-1} - 1\\) possible subsets to try. This is natural because each split is a binary question. In contrast, SVMs and logistic regression require numeric inputs, so categorical features must be one-hot encoded, which inflates dimensionality and can create correlated features. Trees directly test "is this category in this subset?" without any numeric encoding.'
                }
            ]
        },

        // ========== SECTION 2: Splitting Criteria ==========
        {
            id: 'ch09-sec02',
            title: 'Splitting Criteria',
            content: `
<h2>9.2 Splitting Criteria</h2>

<div class="env-block intuition">
<strong>Comparing Impurity Measures.</strong>
Both entropy and Gini impurity measure how "mixed" a node is. Despite their different mathematical forms, they behave very similarly in practice. This section examines their properties, compares them analytically, and discusses when the choice matters.
</div>

<h3>Entropy (Information Gain / ID3 / C4.5)</h3>

<p>The entropy of a node measures the expected number of bits needed to encode the class label:</p>

\\[
H_{\\text{entropy}}(S) = -\\sum_{k=1}^{K} p_k \\log_2 p_k.
\\]

<p>For a binary classification problem (\\(K=2\\)), with \\(p = p_1\\):</p>

\\[
H(p) = -p \\log_2 p - (1-p) \\log_2(1-p).
\\]

<p>This function is concave, symmetric about \\(p = 0.5\\), with maximum \\(H(0.5) = 1\\) bit.</p>

<h3>Gini Impurity (CART)</h3>

<p>The Gini impurity has a probabilistic interpretation: it is the probability that two randomly chosen samples from the node belong to different classes.</p>

\\[
G(p) = 2p(1-p) \\quad (\\text{binary case}).
\\]

<p>Maximum at \\(p = 0.5\\) with \\(G(0.5) = 0.5\\). The factor of 2 is sometimes omitted.</p>

<h3>Comparison</h3>

<p>For binary classification, both functions are concave and symmetric. They agree on which split is best in the vast majority of cases. The main differences are:</p>
<ul>
<li><strong>Sensitivity to class proportions:</strong> Entropy gives a slightly stronger penalty to impure nodes near \\(p = 0\\) or \\(p = 1\\), because \\(-p \\log p\\) decreases more slowly than \\(p(1-p)\\) near the boundary.</li>
<li><strong>Computational cost:</strong> Gini requires only multiplications, while entropy requires logarithms. In practice, this difference is negligible.</li>
<li><strong>Multi-class behavior:</strong> With \\(K &gt; 2\\) classes, Gini tends to produce slightly more balanced splits (it penalizes majority-class dominance less).</li>
</ul>

<div class="env-block theorem">
<strong>Proposition 9.2.1.</strong> For binary classification, the Gini impurity \\(G(p) = 2p(1-p)\\) is a scaled version of the first-order Taylor approximation of the entropy around \\(p = 0.5\\):
\\[
H(p) \\approx 1 - \\frac{2}{\\ln 2}(p - 0.5)^2 = \\frac{2}{\\ln 2}\\bigl[p(1-p) + \\tfrac{1}{4}\\bigr] - \\frac{1}{\\ln 2} + 1.
\\]
After rescaling, the Gini approximation matches entropy closely, explaining why the two criteria rarely disagree.
</div>

<h3>Misclassification Error</h3>

<p>A third impurity measure is the misclassification error: \\(E(p) = 1 - \\max_k p_k\\). While natural, it is <em>not</em> differentiable at \\(p = 0.5\\) and produces a piecewise linear function that does not reward "progress" toward purity as well as entropy or Gini. It is primarily used for pruning, not for growing.</p>

<div class="viz-placeholder" data-viz="viz-entropy-gini-curves"></div>
`,
            visualizations: [
                {
                    id: 'viz-entropy-gini-curves',
                    title: 'Entropy vs. Gini vs. Misclassification Error',
                    description: 'Compare the three impurity measures for binary classification as the class probability \\(p\\) varies from 0 to 1. Use the slider to highlight a particular \\(p\\) value and see the impurity values.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 200, originX: 60, originY: null });
                        viz.originY = viz.height - 40;
                        viz.scale = Math.min(viz.width - 120, 500);
                        const scaleY = viz.height - 80;

                        let pVal = 0.5;
                        VizEngine.createSlider(controls, 'p', 0, 1, 0.5, 0.01, v => { pVal = v; });

                        const entropy = p => {
                            if (p <= 0.001 || p >= 0.999) return 0;
                            return -p * Math.log2(p) - (1-p) * Math.log2(1-p);
                        };
                        const gini = p => 2 * p * (1 - p);
                        const misclass = p => 1 - Math.max(p, 1 - p);

                        function draw() {
                            viz.clear();
                            const ctx = viz.ctx;
                            const oX = 60, oY = viz.height - 40;
                            const w = viz.width - 120, h = viz.height - 80;

                            // Grid
                            ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                            for (let i = 0; i <= 10; i++) {
                                const x = oX + (i/10) * w;
                                ctx.beginPath(); ctx.moveTo(x, oY); ctx.lineTo(x, oY - h); ctx.stroke();
                            }
                            for (let i = 0; i <= 5; i++) {
                                const y = oY - (i/5) * h;
                                ctx.beginPath(); ctx.moveTo(oX, y); ctx.lineTo(oX + w, y); ctx.stroke();
                            }

                            // Axes
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1.5;
                            ctx.beginPath(); ctx.moveTo(oX, oY); ctx.lineTo(oX + w, oY); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(oX, oY); ctx.lineTo(oX, oY - h); ctx.stroke();

                            // Axis labels
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                            for (let i = 0; i <= 10; i += 2) ctx.fillText((i/10).toFixed(1), oX + (i/10)*w, oY + 4);
                            ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
                            for (let i = 0; i <= 5; i++) ctx.fillText((i/5).toFixed(1), oX - 6, oY - (i/5)*h);
                            ctx.textAlign = 'center'; ctx.fillText('p', oX + w/2, oY + 20);

                            const curves = [
                                { fn: entropy, color: viz.colors.blue, name: 'Entropy' },
                                { fn: gini, color: viz.colors.orange, name: 'Gini' },
                                { fn: misclass, color: viz.colors.green, name: 'Misclass' }
                            ];

                            // Draw curves
                            const steps = 200;
                            for (const curve of curves) {
                                ctx.beginPath(); ctx.strokeStyle = curve.color; ctx.lineWidth = 2.5;
                                for (let i = 0; i <= steps; i++) {
                                    const p = i / steps;
                                    const val = curve.fn(p);
                                    const sx = oX + p * w;
                                    const sy = oY - val * h;
                                    i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                                }
                                ctx.stroke();
                            }

                            // Vertical line at pVal
                            const px = oX + pVal * w;
                            ctx.setLineDash([4, 4]); ctx.strokeStyle = viz.colors.white + '80'; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(px, oY); ctx.lineTo(px, oY - h); ctx.stroke();
                            ctx.setLineDash([]);

                            // Dots and values
                            const vals = curves.map(c => c.fn(pVal));
                            for (let ci = 0; ci < curves.length; ci++) {
                                const sy = oY - vals[ci] * h;
                                ctx.fillStyle = curves[ci].color; ctx.beginPath(); ctx.arc(px, sy, 5, 0, Math.PI*2); ctx.fill();
                            }

                            // Legend
                            const lx = oX + w - 120, ly = oY - h + 12;
                            for (let ci = 0; ci < curves.length; ci++) {
                                ctx.fillStyle = curves[ci].color; ctx.font = '12px -apple-system,sans-serif';
                                ctx.textAlign = 'left';
                                ctx.fillRect(lx, ly + ci * 18, 12, 12);
                                ctx.fillText(curves[ci].name + ': ' + vals[ci].toFixed(3), lx + 16, ly + ci * 18 + 10);
                            }

                            viz.screenText('p = ' + pVal.toFixed(2), oX + w/2, oY - h - 8, viz.colors.white, 13);
                        }
                        viz.animate(draw);
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Prove that the entropy \\(H(p) = -p\\log_2 p - (1-p)\\log_2(1-p)\\) is maximized at \\(p = 0.5\\).',
                    hint: 'Take the derivative with respect to \\(p\\), set it to zero, and verify the second derivative is negative.',
                    solution: '\\(H\'(p) = -\\log_2 p - \\frac{1}{\\ln 2} + \\log_2(1-p) + \\frac{1}{\\ln 2} = \\log_2\\frac{1-p}{p}\\). Setting \\(H\'(p) = 0\\) gives \\(\\frac{1-p}{p} = 1\\), so \\(p = 0.5\\). The second derivative \\(H\'\'(p) = -\\frac{1}{p(1-p)\\ln 2} &lt; 0\\) for all \\(p \\in (0,1)\\), confirming this is a maximum. \\(H(0.5) = -2 \\cdot 0.5 \\cdot \\log_2(0.5) = 1\\) bit.'
                },
                {
                    question: 'Show that the Gini impurity \\(G = 1 - \\sum_k p_k^2\\) equals the probability that two independently drawn samples from the node have different class labels.',
                    hint: 'The probability of drawing two samples of the same class is \\(\\sum_k p_k^2\\) (draw class \\(k\\) twice).',
                    solution: 'Draw two independent samples. The probability both are class \\(k\\) is \\(p_k \\cdot p_k = p_k^2\\). The probability they are the <em>same</em> class (any class) is \\(\\sum_k p_k^2\\). So the probability they are <em>different</em> is \\(1 - \\sum_k p_k^2 = G\\). This gives Gini a clean probabilistic interpretation: it measures the "confusion" in a node as the chance of misclassifying a random sample if you labeled it by randomly drawing from the empirical distribution.'
                },
                {
                    question: 'Why is misclassification error not suitable as a splitting criterion for growing trees, even though it is a natural measure of impurity?',
                    hint: 'Consider a node with 50-50 class split. Compare a split that yields (45, 5) and (5, 45) children with one that yields (50, 0) and (0, 50). What does misclassification error say about each?',
                    solution: 'Misclassification error is piecewise linear: \\(E(p) = \\min(p, 1-p)\\). Consider a parent with \\(p=0.5\\) (error = 0.5). A split into (90% class 0, 10% class 1) and (10% class 0, 90% class 1), each with half the data, gives weighted child error \\(0.5 \\cdot 0.1 + 0.5 \\cdot 0.1 = 0.1\\). A different split into (60% class 0) and (60% class 1) gives \\(0.5 \\cdot 0.4 + 0.5 \\cdot 0.4 = 0.4\\). So far both criteria agree, but misclassification error does not distinguish between splits that improve purity within the majority class without changing the predicted label. In contrast, entropy and Gini are strictly concave, rewarding <em>any</em> reduction in class mixing, making them better at guiding the greedy search.'
                }
            ]
        },

        // ========== SECTION 3: Pruning ==========
        {
            id: 'ch09-sec03',
            title: 'Pruning',
            content: `
<h2>9.3 Pruning</h2>

<div class="env-block intuition">
<strong>The Overfitting Problem.</strong>
A fully grown tree can achieve zero training error by creating one leaf per training point. This is extreme overfitting. Pruning removes branches that improve training accuracy but hurt generalization, trading a small increase in training error for a large decrease in test error. There are two strategies: stop early (pre-pruning) or grow fully and cut back (post-pruning).
</div>

<h3>Pre-Pruning (Early Stopping)</h3>

<p>Pre-pruning halts tree growth before the tree becomes too complex. Common stopping rules:</p>
<ul>
<li><strong>Maximum depth:</strong> stop splitting when a node reaches depth \\(D_{\\max}\\).</li>
<li><strong>Minimum samples per leaf:</strong> require at least \\(n_{\\min}\\) samples in each leaf.</li>
<li><strong>Minimum information gain:</strong> only split if \\(\\Delta &gt; \\Delta_{\\min}\\).</li>
</ul>

<p>Pre-pruning is simple and fast, but it has a weakness: the <em>horizon effect</em>. A split with low immediate gain may enable a highly informative split at the next level. By stopping too early, we miss this structure.</p>

<h3>Post-Pruning (Cost-Complexity Pruning / Minimal Cost-Complexity)</h3>

<p>Post-pruning grows the full tree \\(T_{\\max}\\), then removes subtrees that do not justify their complexity. The standard method is <strong>cost-complexity pruning</strong> (also called weakest-link pruning), used by CART.</p>

<div class="env-block definition">
<strong>Definition 9.3.1 (Cost-Complexity Criterion).</strong> For a subtree \\(T\\) of \\(T_{\\max}\\) with \\(|T|\\) leaf nodes, define
\\[
C_\\alpha(T) = \\sum_{m=1}^{|T|} \\sum_{\\mathbf{x}_i \\in R_m} L(y_i, \\hat{y}_m) + \\alpha |T|,
\\]
where \\(R_m\\) is the region corresponding to leaf \\(m\\), \\(\\hat{y}_m\\) is the leaf prediction, and \\(\\alpha \\geq 0\\) is the complexity parameter.
</div>

<p>For each \\(\\alpha\\), there is a unique smallest subtree \\(T_\\alpha\\) that minimizes \\(C_\\alpha\\). As \\(\\alpha\\) increases, the optimal tree becomes smaller:</p>
<ul>
<li>\\(\\alpha = 0\\): the full tree \\(T_{\\max}\\).</li>
<li>\\(\\alpha \\to \\infty\\): the root node only (no splits).</li>
</ul>

<p>The algorithm produces a nested sequence of subtrees \\(T_{\\max} \\supset T_1 \\supset T_2 \\supset \\cdots \\supset \\{\\text{root}\\}\\). We select \\(\\alpha\\) by cross-validation, choosing the value that minimizes test error.</p>

<div class="env-block theorem">
<strong>Proposition 9.3.2 (Weakest-Link Cutting).</strong> At each step, collapse the internal node whose removal causes the smallest per-leaf increase in training error. Formally, collapse the node \\(t\\) minimizing
\\[
\\frac{R(t) - R(T_t)}{|T_t| - 1},
\\]
where \\(R(t)\\) is the error when \\(t\\) is a leaf and \\(R(T_t)\\) is the error of the subtree rooted at \\(t\\).
</div>

<div class="env-block remark">
<strong>Practical Advice.</strong> In scikit-learn, the parameter <code>ccp_alpha</code> controls the cost-complexity parameter. A common workflow: (1) grow a full tree, (2) compute the "effective alpha" path using <code>cost_complexity_pruning_path</code>, (3) cross-validate over \\(\\alpha\\) values on the path.
</div>

<div class="viz-placeholder" data-viz="viz-pruning-depth"></div>
`,
            visualizations: [
                {
                    id: 'viz-pruning-depth',
                    title: 'Effect of Tree Depth on Decision Boundary',
                    description: 'Adjust the maximum depth to see how pruning affects the decision boundary. Too shallow (underfitting) misses patterns; too deep (overfitting) creates jagged regions around individual points.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 55, originX: null, originY: null });
                        viz.originX = viz.width / 2;
                        viz.originY = viz.height / 2;

                        let depth = 2;
                        VizEngine.createSlider(controls, 'Max Depth', 1, 8, 2, 1, v => { depth = Math.round(v); });

                        // Generate a dataset with two clusters per class
                        const rng = (s) => { let v = s; return () => { v = (v * 16807) % 2147483647; return v / 2147483647; }; };
                        const rand = rng(55);
                        const pts = [];
                        const addCluster = (cx, cy, n, cls) => {
                            for (let i = 0; i < n; i++) {
                                const a = rand() * 2 * Math.PI, r = (rand() + rand()) * 0.5;
                                pts.push({ x: cx + r * Math.cos(a), y: cy + r * Math.sin(a), c: cls });
                            }
                        };
                        addCluster(-1.2, 1.0, 15, 0);
                        addCluster(1.0, -0.8, 15, 0);
                        addCluster(0.8, 1.2, 12, 1);
                        addCluster(-0.8, -1.0, 12, 1);
                        // A few noise points
                        for (let i = 0; i < 6; i++) pts.push({ x: (rand()-0.5)*4, y: (rand()-0.5)*4, c: rand() > 0.5 ? 1 : 0 });

                        function buildTree(data, d, maxD, xMin, xMax, yMin, yMax) {
                            if (d >= maxD || data.length < 2) {
                                const c0 = data.filter(p => p.c === 0).length;
                                return { leaf: true, cls: data.length - c0 > c0 ? 1 : 0, xMin, xMax, yMin, yMax };
                            }
                            let bestGain = -1, bestFeat = 'x', bestThresh = 0;
                            const gini = (arr) => { if (arr.length === 0) return 0; const p = arr.filter(q => q.c === 1).length / arr.length; return 2*p*(1-p); };
                            const parentG = gini(data);
                            for (const feat of ['x', 'y']) {
                                const vals = [...new Set(data.map(p => p[feat]))].sort((a,b) => a-b);
                                for (let i = 0; i < vals.length - 1; i++) {
                                    const t = (vals[i] + vals[i+1]) / 2;
                                    const left = data.filter(p => p[feat] <= t);
                                    const right = data.filter(p => p[feat] > t);
                                    if (left.length === 0 || right.length === 0) continue;
                                    const gain = parentG - (left.length/data.length)*gini(left) - (right.length/data.length)*gini(right);
                                    if (gain > bestGain) { bestGain = gain; bestFeat = feat; bestThresh = t; }
                                }
                            }
                            if (bestGain <= 1e-8) {
                                const c0 = data.filter(p => p.c === 0).length;
                                return { leaf: true, cls: data.length - c0 > c0 ? 1 : 0, xMin, xMax, yMin, yMax };
                            }
                            const leftData = data.filter(p => p[bestFeat] <= bestThresh);
                            const rightData = data.filter(p => p[bestFeat] > bestThresh);
                            return {
                                leaf: false, feat: bestFeat, thresh: bestThresh,
                                left: buildTree(leftData, d+1, maxD,
                                    xMin, bestFeat === 'x' ? bestThresh : xMax,
                                    yMin, bestFeat === 'y' ? bestThresh : yMax),
                                right: buildTree(rightData, d+1, maxD,
                                    bestFeat === 'x' ? bestThresh : xMin, xMax,
                                    bestFeat === 'y' ? bestThresh : yMin, yMax)
                            };
                        }

                        function predict(tree, x, y) {
                            if (tree.leaf) return tree.cls;
                            const val = tree.feat === 'x' ? x : y;
                            return val <= tree.thresh ? predict(tree.left, x, y) : predict(tree.right, x, y);
                        }

                        function countLeaves(t) { return t.leaf ? 1 : countLeaves(t.left) + countLeaves(t.right); }

                        function draw() {
                            viz.clear();
                            const ctx = viz.ctx;
                            const tree = buildTree(pts, 0, depth, -3, 3, -3, 3);

                            // Shade regions
                            const res = 4;
                            for (let sx = 0; sx < viz.width; sx += res) {
                                for (let sy = 0; sy < viz.height; sy += res) {
                                    const [mx, my] = viz.toMath(sx, sy);
                                    const cls = predict(tree, mx, my);
                                    ctx.fillStyle = cls === 0 ? viz.colors.blue + '20' : viz.colors.orange + '20';
                                    ctx.fillRect(sx, sy, res, res);
                                }
                            }

                            // Draw boundary
                            for (let sx = 0; sx < viz.width; sx += 2) {
                                for (let sy = 0; sy < viz.height; sy += 2) {
                                    const [mx, my] = viz.toMath(sx, sy);
                                    const c0 = predict(tree, mx, my);
                                    const [mx2] = viz.toMath(sx + 2, sy);
                                    const [, my2] = viz.toMath(sx, sy + 2);
                                    if (predict(tree, mx2, my) !== c0 || predict(tree, mx, my2) !== c0) {
                                        ctx.fillStyle = viz.colors.white + '60';
                                        ctx.fillRect(sx, sy, 2, 2);
                                    }
                                }
                            }

                            viz.drawGrid(1); viz.drawAxes();
                            for (const p of pts) {
                                const col = p.c === 0 ? viz.colors.blue : viz.colors.orange;
                                viz.drawPoint(p.x, p.y, col, null, 4);
                            }

                            const nLeaves = countLeaves(tree);
                            const errors = pts.filter(p => predict(tree, p.x, p.y) !== p.c).length;
                            viz.screenText('Depth ' + depth + '  |  ' + nLeaves + ' leaves  |  ' + errors + '/' + pts.length + ' errors', viz.width/2, 18, viz.colors.white, 12);
                        }
                        viz.animate(draw);
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Explain the "horizon effect" in pre-pruning. Give a concrete example where stopping early misses an important pattern.',
                    hint: 'Consider data where the first split has near-zero gain, but the second split in one branch achieves perfect separation.',
                    solution: 'Consider an XOR dataset: \\(\\{(0,0,\\text{+}), (1,1,\\text{+}), (0,1,\\text{-}), (1,0,\\text{-})\\}\\). Splitting on \\(x_1 = 0.5\\) gives two children, each with one + and one - sample, so the Gini gain is zero. A pre-pruning rule requiring \\(\\Delta &gt; 0\\) would stop here and predict the majority class. But if we split further on \\(x_2\\) in each child, we achieve perfect separation. The first split had zero gain because it is only useful in combination with the second. Post-pruning avoids this by growing the full tree first and only removing branches that do not help on validation data.'
                },
                {
                    question: 'In cost-complexity pruning, show that increasing \\(\\alpha\\) can only decrease or maintain (never increase) the number of leaves in the optimal tree \\(T_\\alpha\\).',
                    hint: 'Suppose \\(\\alpha_1 &lt; \\alpha_2\\). Compare \\(C_{\\alpha_2}(T_{\\alpha_1})\\) with \\(C_{\\alpha_2}(T_{\\alpha_2})\\).',
                    solution: 'Let \\(T_1 = T_{\\alpha_1}\\) and \\(T_2 = T_{\\alpha_2}\\) with \\(\\alpha_1 &lt; \\alpha_2\\). Since \\(T_1\\) minimizes \\(C_{\\alpha_1}\\), we have \\(R(T_1) + \\alpha_1 |T_1| \\leq R(T_2) + \\alpha_1 |T_2|\\). Since \\(T_2\\) minimizes \\(C_{\\alpha_2}\\), we have \\(R(T_2) + \\alpha_2 |T_2| \\leq R(T_1) + \\alpha_2 |T_1|\\). Adding these: \\((\\alpha_2 - \\alpha_1)(|T_2| - |T_1|) \\leq (\\alpha_2 - \\alpha_1)(|T_2| - |T_1|)\\) is trivially true, but from the second inequality alone: \\(R(T_2) - R(T_1) \\leq \\alpha_2(|T_1| - |T_2|)\\). Since more leaves means less training error (\\(R(T_1) \\leq R(T_2)\\) if \\(|T_1| \\geq |T_2|\\)), the penalty term \\(\\alpha_2|T|\\) grows faster with larger \\(\\alpha\\), favoring smaller trees. Formally, the nested sequence of optimal subtrees is monotonically shrinking with \\(\\alpha\\).'
                },
                {
                    question: 'A tree with 50 leaves achieves 2% training error. After pruning to 10 leaves, training error rises to 8% but test error drops from 25% to 12%. Explain this in terms of bias-variance.',
                    hint: 'The full tree has low bias but high variance; the pruned tree trades some bias for much lower variance.',
                    solution: 'The 50-leaf tree nearly memorizes the training data (2% error), so it has very low bias. But it overfits, leading to high variance and 25% test error (a 23% generalization gap). Pruning to 10 leaves introduces some bias (training error increases to 8%) because the model can no longer represent fine-grained patterns. However, the simpler model generalizes much better (variance drops), reducing test error to 12%. The net effect is a large reduction in test error (25% to 12%) at the cost of a small increase in training error (2% to 8%). This is the classic bias-variance tradeoff: the optimal model complexity balances the two sources of error.'
                }
            ]
        },

        // ========== SECTION 4: CART for Regression ==========
        {
            id: 'ch09-sec04',
            title: 'CART for Regression',
            content: `
<h2>9.4 CART for Regression</h2>

<div class="env-block intuition">
<strong>From Classification to Regression.</strong>
Everything we have developed for classification trees carries over to regression with one key change: the impurity measure becomes the variance (or squared error), and each leaf predicts a constant value (the mean of the training responses in that region). The result is a piecewise-constant function that approximates the underlying regression surface.
</div>

<h3>Regression Tree Prediction</h3>

<p>A regression tree partitions \\(\\mathcal{X}\\) into \\(M\\) disjoint regions \\(R_1, R_2, \\ldots, R_M\\) and predicts a constant \\(c_m\\) in each region:</p>

\\[
\\hat{f}(\\mathbf{x}) = \\sum_{m=1}^{M} c_m \\cdot \\mathbf{1}(\\mathbf{x} \\in R_m).
\\]

<p>The optimal constant for each region (minimizing squared error) is the mean of the training responses:</p>

\\[
c_m = \\frac{1}{|\\{i : \\mathbf{x}_i \\in R_m\\}|} \\sum_{\\mathbf{x}_i \\in R_m} y_i.
\\]

<h3>Splitting Criterion: Variance Reduction</h3>

<p>The impurity of a regression node \\(S\\) is the within-node sum of squared residuals:</p>

\\[
H(S) = \\sum_{i \\in S} (y_i - \\bar{y}_S)^2,
\\]

<p>where \\(\\bar{y}_S\\) is the mean response in node \\(S\\). Equivalently, \\(H(S) = |S| \\cdot \\text{Var}(S)\\). The best split maximizes the variance reduction:</p>

\\[
\\Delta(S, j, t) = H(S) - H(S_L) - H(S_R).
\\]

<div class="env-block theorem">
<strong>Proposition 9.4.1.</strong> For the 1D regression problem \\(y = f(x) + \\varepsilon\\), a regression tree with \\(M\\) leaves produces a piecewise-constant approximation. As \\(M \\to \\infty\\) and \\(n \\to \\infty\\) with appropriate regularization, the tree approximation converges to the true regression function in \\(L^2\\).
</div>

<h3>Bias-Variance for Regression Trees</h3>

<p>The piecewise-constant nature of regression trees has important implications:</p>
<ul>
<li><strong>Bias:</strong> A tree approximates a smooth function with step functions. The bias decreases as depth increases (more, smaller regions), but the approximation is always discontinuous at split boundaries.</li>
<li><strong>Variance:</strong> Each leaf prediction is the mean of a (possibly small) subset of training points. With deeper trees, each leaf has fewer points, increasing variance.</li>
<li><strong>Discontinuity:</strong> Unlike kernel regression or splines, regression trees produce discontinuous predictions at region boundaries. This is a fundamental limitation that ensemble methods (Chapter 10) address by averaging many trees.</li>
</ul>

<div class="env-block remark">
<strong>Why Not Fit Lines in Each Leaf?</strong> <em>Model trees</em> (e.g., M5) fit linear regressions in each leaf, producing piecewise-linear (continuous) predictions. This reduces bias but adds complexity. Standard CART uses constants for simplicity and because ensembles of piecewise-constant trees work remarkably well.
</div>

<div class="viz-placeholder" data-viz="viz-regression-tree"></div>
`,
            visualizations: [
                {
                    id: 'viz-regression-tree',
                    title: 'Regression Tree: Staircase Fit',
                    description: 'A 1D regression dataset with a non-linear underlying function (blue curve). The regression tree (orange staircases) approximates it with piecewise constants. Adjust the depth to see the bias-variance tradeoff: shallow trees underfit, deep trees overfit.',
                    setup(container, controls) {
                        const viz = new VizEngine(container, { scale: 50, originX: null, originY: null });
                        viz.originX = 60;
                        viz.originY = viz.height * 0.65;

                        let treeDepth = 2;
                        VizEngine.createSlider(controls, 'Depth', 1, 7, 2, 1, v => { treeDepth = Math.round(v); });

                        // Generate 1D data from a non-linear function with noise
                        const rng = (s) => { let v = s; return () => { v = (v * 16807) % 2147483647; return v / 2147483647; }; };
                        const rand = rng(31);
                        const trueF = x => Math.sin(1.5 * x) + 0.3 * x;
                        const dataX = [], dataY = [];
                        for (let i = 0; i < 50; i++) {
                            const x = rand() * 8 - 1;
                            const y = trueF(x) + (rand() - 0.5) * 0.6;
                            dataX.push(x);
                            dataY.push(y);
                        }

                        function buildRegTree(indices, d, maxD, xMin, xMax) {
                            const mean = indices.reduce((s, i) => s + dataY[i], 0) / indices.length;
                            if (d >= maxD || indices.length < 3) {
                                return { leaf: true, val: mean, xMin, xMax };
                            }
                            // Find best split
                            let bestReduction = -1, bestThresh = 0;
                            const sortedX = indices.map(i => dataX[i]).sort((a,b) => a-b);
                            const parentSS = indices.reduce((s, i) => s + (dataY[i] - mean) ** 2, 0);

                            for (let k = 0; k < sortedX.length - 1; k++) {
                                const t = (sortedX[k] + sortedX[k+1]) / 2;
                                const leftIdx = indices.filter(i => dataX[i] <= t);
                                const rightIdx = indices.filter(i => dataX[i] > t);
                                if (leftIdx.length < 1 || rightIdx.length < 1) continue;
                                const lMean = leftIdx.reduce((s, i) => s + dataY[i], 0) / leftIdx.length;
                                const rMean = rightIdx.reduce((s, i) => s + dataY[i], 0) / rightIdx.length;
                                const lSS = leftIdx.reduce((s, i) => s + (dataY[i] - lMean) ** 2, 0);
                                const rSS = rightIdx.reduce((s, i) => s + (dataY[i] - rMean) ** 2, 0);
                                const reduction = parentSS - lSS - rSS;
                                if (reduction > bestReduction) { bestReduction = reduction; bestThresh = t; }
                            }
                            if (bestReduction <= 1e-8) return { leaf: true, val: mean, xMin, xMax };

                            const leftIdx = indices.filter(i => dataX[i] <= bestThresh);
                            const rightIdx = indices.filter(i => dataX[i] > bestThresh);
                            return {
                                leaf: false, thresh: bestThresh,
                                left: buildRegTree(leftIdx, d+1, maxD, xMin, bestThresh),
                                right: buildRegTree(rightIdx, d+1, maxD, bestThresh, xMax),
                                xMin, xMax
                            };
                        }

                        function predictTree(tree, x) {
                            if (tree.leaf) return tree.val;
                            return x <= tree.thresh ? predictTree(tree.left, x) : predictTree(tree.right, x);
                        }

                        function getSegments(tree) {
                            if (tree.leaf) return [{ xMin: tree.xMin, xMax: tree.xMax, val: tree.val }];
                            return [...getSegments(tree.left), ...getSegments(tree.right)];
                        }

                        function draw() {
                            viz.clear();
                            const ctx = viz.ctx;

                            // Grid and axes
                            viz.drawGrid(1); viz.drawAxes();

                            // True function
                            ctx.beginPath(); ctx.strokeStyle = viz.colors.blue; ctx.lineWidth = 2;
                            const xRange = [-1, 7], steps = 200;
                            for (let i = 0; i <= steps; i++) {
                                const x = xRange[0] + (xRange[1] - xRange[0]) * i / steps;
                                const y = trueF(x);
                                const [sx, sy] = viz.toScreen(x, y);
                                i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                            }
                            ctx.stroke();

                            // Build tree and draw staircase
                            const allIdx = dataX.map((_, i) => i);
                            const tree = buildRegTree(allIdx, 0, treeDepth, -1.5, 7.5);
                            const segs = getSegments(tree);

                            ctx.beginPath(); ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 2.5;
                            for (const seg of segs) {
                                const [sx1, sy1] = viz.toScreen(seg.xMin, seg.val);
                                const [sx2, sy2] = viz.toScreen(seg.xMax, seg.val);
                                ctx.moveTo(sx1, sy1); ctx.lineTo(sx2, sy2);
                            }
                            ctx.stroke();

                            // Vertical lines at split boundaries
                            ctx.setLineDash([3, 3]); ctx.strokeStyle = viz.colors.orange + '60'; ctx.lineWidth = 1;
                            for (let i = 0; i < segs.length - 1; i++) {
                                const [sx, sy1] = viz.toScreen(segs[i].xMax, segs[i].val);
                                const [, sy2] = viz.toScreen(segs[i+1].xMin, segs[i+1].val);
                                ctx.beginPath(); ctx.moveTo(sx, sy1); ctx.lineTo(sx, sy2); ctx.stroke();
                            }
                            ctx.setLineDash([]);

                            // Data points
                            for (let i = 0; i < dataX.length; i++) {
                                viz.drawPoint(dataX[i], dataY[i], viz.colors.white + 'aa', null, 3);
                            }

                            // Compute MSE
                            const mse = dataX.reduce((s, x, i) => s + (predictTree(tree, x) - dataY[i]) ** 2, 0) / dataX.length;
                            viz.screenText('Depth ' + treeDepth + '  |  ' + segs.length + ' leaves  |  MSE: ' + mse.toFixed(3), viz.width/2, 18, viz.colors.white, 12);

                            // Legend
                            ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'left';
                            ctx.fillStyle = viz.colors.blue; ctx.fillText('\u2014 True f(x)', 80, viz.height - 12);
                            ctx.fillStyle = viz.colors.orange; ctx.fillText('\u2014 Tree fit', 200, viz.height - 12);
                        }
                        viz.animate(draw);
                        return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Prove that the optimal constant prediction in a regression tree leaf (minimizing MSE) is the mean of the training responses in that leaf.',
                    hint: 'Minimize \\(\\sum_{i \\in R_m} (y_i - c)^2\\) with respect to \\(c\\) by taking the derivative and setting it to zero.',
                    solution: 'Let \\(L(c) = \\sum_{i \\in R_m} (y_i - c)^2\\). Taking the derivative: \\(\\frac{dL}{dc} = -2 \\sum_{i \\in R_m}(y_i - c) = 0\\), which gives \\(c = \\frac{1}{|R_m|} \\sum_{i \\in R_m} y_i = \\bar{y}_m\\). The second derivative \\(\\frac{d^2L}{dc^2} = 2|R_m| &gt; 0\\) confirms this is a minimum. Thus the mean response is the unique minimizer of MSE within each leaf.'
                },
                {
                    question: 'A regression tree with 1 leaf has MSE = 10 on 100 training points. After one split, the left child has 60 points with MSE = 4 and the right child has 40 points with MSE = 5. Compute the variance reduction.',
                    hint: 'The total sum of squares reduces from \\(n \\cdot \\text{MSE}_{\\text{parent}}\\) to \\(n_L \\cdot \\text{MSE}_L + n_R \\cdot \\text{MSE}_R\\).',
                    solution: 'Parent total SS: \\(100 \\times 10 = 1000\\). Left child SS: \\(60 \\times 4 = 240\\). Right child SS: \\(40 \\times 5 = 200\\). Total child SS: \\(240 + 200 = 440\\). Variance reduction: \\(1000 - 440 = 560\\). Equivalently, the MSE of the piecewise-constant fit drops from 10 to \\(440/100 = 4.4\\). The split explains 56% of the parent node variance, which is a very good split.'
                },
                {
                    question: 'Why are regression tree predictions always discontinuous at split boundaries? Suggest one method to obtain continuous predictions from tree-like models.',
                    hint: 'Each leaf predicts a different constant, and at a split boundary the prediction jumps from one constant to another.',
                    solution: 'A regression tree assigns constant predictions \\(c_L\\) and \\(c_R\\) to the two sides of a split at threshold \\(t\\). At \\(x = t\\), the prediction jumps from \\(c_L\\) to \\(c_R\\) (a discontinuity). This is inherent to piecewise-constant models. To obtain continuous predictions: (1) <em>Model trees</em> (M5) fit linear regressions in each leaf and can enforce continuity at boundaries; (2) <em>Random forests</em> average many trees, producing much smoother predictions (each individual tree is discontinuous, but the average of many randomized discontinuities approaches a smooth function); (3) <em>Soft decision trees</em> use sigmoid gating functions instead of hard thresholds, making the routing probabilities continuous.'
                }
            ]
        }
    ]
});
