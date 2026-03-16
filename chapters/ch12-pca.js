window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch12',
    number: 12,
    title: 'Principal Component Analysis',
    subtitle: 'Finding the directions of maximum variance to compress, visualize, and denoise high-dimensional data',
    sections: [
        // ===================== Section 1: PCA via Maximum Variance =====================
        {
            id: 'ch12-sec01',
            title: 'PCA via Maximum Variance',
            content: `<h2>PCA via Maximum Variance</h2>

                <div class="env-block intuition">
                    <div class="env-title">The Core Idea</div>
                    <div class="env-body"><p>Given a dataset in \\(\\mathbb{R}^D\\), we want to find a lower-dimensional subspace that captures as much variability as possible. The direction along which the data varies most is the <strong>first principal component</strong>. Subsequent components capture maximum remaining variance subject to orthogonality constraints.</p></div>
                </div>

                <h3>Setup and Notation</h3>

                <p>Let \\(\\mathbf{x}_1, \\dots, \\mathbf{x}_N \\in \\mathbb{R}^D\\) be our data, assumed centered so that \\(\\bar{\\mathbf{x}} = \\frac{1}{N}\\sum_{i=1}^N \\mathbf{x}_i = \\mathbf{0}\\). The <strong>sample covariance matrix</strong> is</p>
                \\[\\mathbf{S} = \\frac{1}{N}\\sum_{i=1}^{N} \\mathbf{x}_i \\mathbf{x}_i^\\top = \\frac{1}{N}\\mathbf{X}^\\top \\mathbf{X}\\]
                <p>where \\(\\mathbf{X} \\in \\mathbb{R}^{N \\times D}\\) is the data matrix with rows \\(\\mathbf{x}_i^\\top\\).</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Projection onto a Direction)</div>
                    <div class="env-body"><p>Given a unit vector \\(\\mathbf{u} \\in \\mathbb{R}^D\\) with \\(\\|\\mathbf{u}\\| = 1\\), the <strong>projection</strong> of \\(\\mathbf{x}_i\\) onto \\(\\mathbf{u}\\) is the scalar \\(z_i = \\mathbf{u}^\\top \\mathbf{x}_i\\). The variance of the projected data is
                    \\[\\text{Var}(z) = \\frac{1}{N}\\sum_{i=1}^{N} (\\mathbf{u}^\\top \\mathbf{x}_i)^2 = \\mathbf{u}^\\top \\mathbf{S} \\, \\mathbf{u}\\]</p></div>
                </div>

                <h3>The Maximum Variance Objective</h3>

                <p>The first principal component direction \\(\\mathbf{u}_1\\) maximizes projected variance:</p>
                \\[\\mathbf{u}_1 = \\arg\\max_{\\|\\mathbf{u}\\|=1} \\; \\mathbf{u}^\\top \\mathbf{S} \\, \\mathbf{u}\\]

                <div class="env-block theorem">
                    <div class="env-title">Theorem (PC1 is the Top Eigenvector)</div>
                    <div class="env-body"><p>The solution to \\(\\max_{\\|\\mathbf{u}\\|=1} \\mathbf{u}^\\top \\mathbf{S} \\, \\mathbf{u}\\) is the eigenvector of \\(\\mathbf{S}\\) corresponding to the largest eigenvalue \\(\\lambda_1\\). The maximum variance equals \\(\\lambda_1\\).</p>
                    <p><em>Proof.</em> Form the Lagrangian \\(\\mathcal{L} = \\mathbf{u}^\\top \\mathbf{S} \\, \\mathbf{u} - \\lambda(\\mathbf{u}^\\top \\mathbf{u} - 1)\\). Setting \\(\\nabla_{\\mathbf{u}} \\mathcal{L} = 2\\mathbf{S}\\mathbf{u} - 2\\lambda \\mathbf{u} = \\mathbf{0}\\) gives \\(\\mathbf{S}\\mathbf{u} = \\lambda \\mathbf{u}\\). Substituting back: \\(\\mathbf{u}^\\top \\mathbf{S} \\, \\mathbf{u} = \\lambda\\). To maximize, we pick the largest eigenvalue. \\(\\square\\)</p></div>
                </div>

                <div class="env-block theorem">
                    <div class="env-title">Theorem (Subsequent Principal Components)</div>
                    <div class="env-body"><p>The \\(k\\)-th principal component direction \\(\\mathbf{u}_k\\) solves
                    \\[\\mathbf{u}_k = \\arg\\max_{\\substack{\\|\\mathbf{u}\\|=1 \\\\ \\mathbf{u} \\perp \\mathbf{u}_1,\\dots,\\mathbf{u}_{k-1}}} \\mathbf{u}^\\top \\mathbf{S} \\, \\mathbf{u}\\]
                    and is the eigenvector of \\(\\mathbf{S}\\) with the \\(k\\)-th largest eigenvalue \\(\\lambda_k\\).</p></div>
                </div>

                <p>Geometrically, PCA rotates the coordinate system to align with the directions of greatest spread. The first axis points along the "longest" direction of the data cloud, the second axis is the longest direction perpendicular to the first, and so on.</p>

                <div class="viz-placeholder" data-viz="viz-pca-max-var"></div>

                <div class="env-block remark">
                    <div class="env-title">Remark (Centering is Essential)</div>
                    <div class="env-body"><p>PCA requires centered data. If \\(\\bar{\\mathbf{x}} \\neq \\mathbf{0}\\), the "maximum variance" direction would be dominated by the mean offset rather than the data's intrinsic spread. Always subtract the sample mean before computing the covariance.</p></div>
                </div>`,

            visualizations: [
                {
                    id: 'viz-pca-max-var',
                    title: '2D Data with Principal Component Directions',
                    description: 'Drag the data cloud angle to see how PC1 and PC2 align with the directions of maximum and minimum variance. Dashed lines show projections onto PC1.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {scale: 40});
                        var ctx = viz.ctx;
                        var angle = 0.6;
                        var spread1 = 3.0, spread2 = 0.8;

                        VizEngine.createSlider(controls, 'Rotation', -1.57, 1.57, 0.6, 0.05, function(v) { angle = v; });
                        VizEngine.createSlider(controls, 'Spread ratio', 0.1, 1.0, 0.27, 0.05, function(v) { spread2 = v * spread1; });

                        // Generate fixed random data (seeded)
                        var rawPts = [];
                        var seed = 42;
                        function srand() { seed = (seed * 16807 + 0) % 2147483647; return (seed - 1) / 2147483646; }
                        for (var i = 0; i < 60; i++) {
                            var u1 = srand(), u2 = srand();
                            var z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                            var z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
                            rawPts.push([z1, z2]);
                        }

                        function draw() {
                            viz.clear(); viz.drawGrid(); viz.drawAxes();
                            var ca = Math.cos(angle), sa = Math.sin(angle);
                            // Transform points
                            var pts = rawPts.map(function(p) {
                                var x = p[0] * spread1, y = p[1] * spread2;
                                return [ca * x - sa * y, sa * x + ca * y];
                            });
                            // Draw projections onto PC1
                            var pc1 = [ca, sa];
                            for (var i = 0; i < pts.length; i++) {
                                var dot = pts[i][0] * pc1[0] + pts[i][1] * pc1[1];
                                var px = dot * pc1[0], py = dot * pc1[1];
                                viz.drawSegment(pts[i][0], pts[i][1], px, py, viz.colors.blue + '44', 0.8, true);
                                viz.drawPoint(px, py, viz.colors.blue + '88', null, 2.5);
                            }
                            // Draw data points
                            for (var i = 0; i < pts.length; i++) {
                                viz.drawPoint(pts[i][0], pts[i][1], viz.colors.white + 'cc', null, 3);
                            }
                            // Draw PC arrows
                            var arrowLen = spread1 * 1.2;
                            viz.drawVector(0, 0, pc1[0] * arrowLen, pc1[1] * arrowLen, viz.colors.blue, 'PC1', 2.5);
                            var pc2 = [-sa, ca];
                            viz.drawVector(0, 0, pc2[0] * spread2 * 1.2, pc2[1] * spread2 * 1.2, viz.colors.orange, 'PC2', 2.5);
                            // Variance labels
                            ctx.font = '12px -apple-system,sans-serif';
                            ctx.fillStyle = viz.colors.blue; ctx.textAlign = 'left';
                            ctx.fillText('\u03BB\u2081 = ' + (spread1 * spread1).toFixed(1), 10, 20);
                            ctx.fillStyle = viz.colors.orange;
                            ctx.fillText('\u03BB\u2082 = ' + (spread2 * spread2).toFixed(1), 10, 38);
                        }
                        draw();
                        var slider = controls.querySelectorAll('.viz-slider');
                        slider.forEach(function(s) { s.addEventListener('input', draw); });
                        return { stopAnimation: function() {} };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Show that for centered data, maximizing \\(\\mathbf{u}^\\top \\mathbf{S} \\, \\mathbf{u}\\) subject to \\(\\|\\mathbf{u}\\| = 1\\) yields the eigenvector equation \\(\\mathbf{S}\\mathbf{u} = \\lambda \\mathbf{u}\\). Why does the constraint matter?',
                    hint: 'Use the method of Lagrange multipliers. Without the constraint \\(\\|\\mathbf{u}\\|=1\\), we could make the objective arbitrarily large by scaling \\(\\mathbf{u}\\).',
                    solution: 'The Lagrangian is \\(\\mathcal{L} = \\mathbf{u}^\\top \\mathbf{S}\\mathbf{u} - \\lambda(\\mathbf{u}^\\top \\mathbf{u} - 1)\\). Setting \\(\\nabla_{\\mathbf{u}}\\mathcal{L} = 2\\mathbf{S}\\mathbf{u} - 2\\lambda\\mathbf{u} = \\mathbf{0}\\) gives \\(\\mathbf{S}\\mathbf{u} = \\lambda\\mathbf{u}\\), the eigenvalue equation. Without the unit-norm constraint, \\(\\mathbf{u}^\\top \\mathbf{S}\\mathbf{u}\\) is unbounded (scale \\(\\mathbf{u}\\) by any constant \\(c\\) and the objective scales by \\(c^2\\)).'
                },
                {
                    question: 'Consider 2D data where \\(\\mathbf{S} = \\begin{pmatrix} 5 & 3 \\\\ 3 & 2 \\end{pmatrix}\\). Find the eigenvalues and eigenvectors. Which direction is PC1?',
                    hint: 'Solve \\(\\det(\\mathbf{S} - \\lambda \\mathbf{I}) = 0\\) to get the characteristic polynomial \\(\\lambda^2 - 7\\lambda + 1 = 0\\).',
                    solution: 'The characteristic equation is \\(\\lambda^2 - 7\\lambda + 1 = 0\\), giving \\(\\lambda = \\frac{7 \\pm \\sqrt{45}}{2}\\), so \\(\\lambda_1 \\approx 6.85\\) and \\(\\lambda_2 \\approx 0.15\\). For \\(\\lambda_1\\): \\((5 - 6.85)u_1 + 3u_2 = 0\\) gives \\(u_2/u_1 \\approx 0.617\\), so PC1 \\(\\approx (0.851, 0.526)^\\top\\) after normalization. PC1 points roughly at \\(31.7^\\circ\\) from the horizontal, capturing 97.9% of total variance.'
                },
                {
                    question: 'If \\(\\mathbf{S}\\) has eigenvalues \\(\\lambda_1 = 10, \\lambda_2 = 3, \\lambda_3 = 0.5\\), what fraction of variance does projecting onto PC1 alone capture? Onto the first two PCs?',
                    hint: 'Variance captured by the first \\(k\\) PCs is \\(\\sum_{j=1}^k \\lambda_j\\) out of \\(\\sum_{j=1}^D \\lambda_j\\).',
                    solution: 'Total variance is \\(10 + 3 + 0.5 = 13.5\\). PC1 alone captures \\(10/13.5 \\approx 74.1\\%\\). The first two PCs capture \\((10+3)/13.5 \\approx 96.3\\%\\). So two components suffice for a nearly lossless projection.'
                }
            ]
        },

        // ===================== Section 2: PCA via Minimum Reconstruction Error =====================
        {
            id: 'ch12-sec02',
            title: 'PCA via Minimum Reconstruction Error',
            content: `<h2>PCA via Minimum Reconstruction Error</h2>

                <div class="env-block intuition">
                    <div class="env-title">An Equivalent Perspective</div>
                    <div class="env-body"><p>Instead of asking "which direction has maximum variance?", we can ask "which \\(d\\)-dimensional subspace best reconstructs the data?" These turn out to be the same problem, but the reconstruction viewpoint gives additional insight into how PCA approximates data.</p></div>
                </div>

                <h3>Reconstruction in a Subspace</h3>

                <p>Let \\(\\mathbf{U}_d = [\\mathbf{u}_1, \\dots, \\mathbf{u}_d] \\in \\mathbb{R}^{D \\times d}\\) be a matrix whose columns form an orthonormal basis for a \\(d\\)-dimensional subspace. The <strong>projection</strong> of \\(\\mathbf{x}\\) onto this subspace is</p>
                \\[\\hat{\\mathbf{x}} = \\mathbf{U}_d \\mathbf{U}_d^\\top \\mathbf{x}\\]
                <p>and the <strong>reconstruction error</strong> for a single point is \\(\\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|^2\\).</p>

                <div class="env-block theorem">
                    <div class="env-title">Theorem (Minimum Reconstruction Error = Maximum Variance)</div>
                    <div class="env-body"><p>The subspace \\(\\mathbf{U}_d\\) that minimizes the total reconstruction error
                    \\[\\mathcal{E}(\\mathbf{U}_d) = \\frac{1}{N}\\sum_{i=1}^N \\|\\mathbf{x}_i - \\mathbf{U}_d \\mathbf{U}_d^\\top \\mathbf{x}_i\\|^2\\]
                    is spanned by the eigenvectors of \\(\\mathbf{S}\\) corresponding to the \\(d\\) largest eigenvalues. The minimum error equals \\(\\sum_{j=d+1}^{D} \\lambda_j\\).</p>
                    <p><em>Proof sketch.</em> Using \\(\\|\\mathbf{x}\\|^2 = \\|\\hat{\\mathbf{x}}\\|^2 + \\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|^2\\) (Pythagorean theorem for orthogonal projections), the average reconstruction error is
                    \\[\\mathcal{E} = \\frac{1}{N}\\sum_i \\|\\mathbf{x}_i\\|^2 - \\frac{1}{N}\\sum_i \\|\\mathbf{U}_d^\\top \\mathbf{x}_i\\|^2 = \\text{tr}(\\mathbf{S}) - \\text{tr}(\\mathbf{U}_d^\\top \\mathbf{S} \\, \\mathbf{U}_d)\\]
                    Since \\(\\text{tr}(\\mathbf{S})\\) is constant, minimizing \\(\\mathcal{E}\\) is equivalent to maximizing \\(\\text{tr}(\\mathbf{U}_d^\\top \\mathbf{S} \\, \\mathbf{U}_d) = \\sum_{j=1}^d \\lambda_j\\). \\(\\square\\)</p></div>
                </div>

                <p>This equivalence is powerful: every unit of variance <em>captured</em> by the projection equals one unit of reconstruction error <em>removed</em>.</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Reconstruction Error)</div>
                    <div class="env-body"><p>For a PCA projection retaining \\(d\\) components, the <strong>reconstruction error</strong> (also called <strong>residual variance</strong>) is
                    \\[\\mathcal{E}_d = \\sum_{j=d+1}^{D} \\lambda_j = \\text{tr}(\\mathbf{S}) - \\sum_{j=1}^{d} \\lambda_j\\]
                    This equals the sum of eigenvalues corresponding to the discarded dimensions.</p></div>
                </div>

                <div class="viz-placeholder" data-viz="viz-recon-error"></div>

                <div class="env-block remark">
                    <div class="env-title">Remark (Connection to SVD)</div>
                    <div class="env-body"><p>PCA is intimately connected to the Singular Value Decomposition. If \\(\\mathbf{X} = \\mathbf{U}\\boldsymbol{\\Sigma}\\mathbf{V}^\\top\\) is the SVD of the centered data matrix, then the columns of \\(\\mathbf{V}\\) are the eigenvectors of \\(\\mathbf{S} = \\frac{1}{N}\\mathbf{X}^\\top\\mathbf{X}\\), and \\(\\lambda_j = \\sigma_j^2 / N\\). The Eckart-Young theorem guarantees that the best rank-\\(d\\) approximation \\(\\mathbf{X}_d = \\mathbf{U}_d \\boldsymbol{\\Sigma}_d \\mathbf{V}_d^\\top\\) minimizes \\(\\|\\mathbf{X} - \\mathbf{X}_d\\|_F^2\\).</p></div>
                </div>`,

            visualizations: [
                {
                    id: 'viz-recon-error',
                    title: 'Reconstruction Error vs. Number of Components',
                    description: 'Adjust the number of retained components to see how reconstruction error decreases. The eigenvalues shown are from a sample covariance matrix.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var numComp = 3;
                        var eigenvals = [5.2, 3.1, 1.8, 0.9, 0.5, 0.3, 0.15, 0.05];
                        var D = eigenvals.length;
                        var totalVar = eigenvals.reduce(function(a, b) { return a + b; }, 0);

                        VizEngine.createSlider(controls, 'Components d', 0, D, 3, 1, function(v) { numComp = Math.round(v); draw(); });

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);
                            var margin = {top: 40, right: 30, bottom: 50, left: 60};
                            var pw = W - margin.left - margin.right;
                            var ph = H - margin.top - margin.bottom;
                            // Axes
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(margin.left, margin.top); ctx.lineTo(margin.left, margin.top + ph); ctx.lineTo(margin.left + pw, margin.top + ph); ctx.stroke();
                            // Labels
                            ctx.fillStyle = viz.colors.text; ctx.font = '12px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.fillText('Number of retained components (d)', margin.left + pw / 2, H - 8);
                            ctx.save(); ctx.translate(14, margin.top + ph / 2); ctx.rotate(-Math.PI / 2);
                            ctx.fillText('Reconstruction Error', 0, 0); ctx.restore();
                            // Compute errors for each d
                            var errors = [];
                            for (var d = 0; d <= D; d++) {
                                var err = 0;
                                for (var j = d; j < D; j++) err += eigenvals[j];
                                errors.push(err);
                            }
                            var maxErr = errors[0];
                            // Bar width
                            var barW = pw / (D + 1) * 0.6;
                            // Tick marks and bars
                            for (var d = 0; d <= D; d++) {
                                var cx = margin.left + (d + 0.5) / (D + 1) * pw;
                                var barH = (errors[d] / maxErr) * ph * 0.9;
                                var color = d <= numComp ? viz.colors.teal : viz.colors.blue;
                                if (d === numComp) color = viz.colors.orange;
                                ctx.fillStyle = color + '99';
                                ctx.fillRect(cx - barW / 2, margin.top + ph - barH, barW, barH);
                                ctx.strokeStyle = color; ctx.lineWidth = 1;
                                ctx.strokeRect(cx - barW / 2, margin.top + ph - barH, barW, barH);
                                // x-axis label
                                ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                                ctx.textAlign = 'center'; ctx.fillText(d.toString(), cx, margin.top + ph + 16);
                                // Value on bar
                                ctx.fillStyle = viz.colors.white; ctx.font = '10px -apple-system,sans-serif';
                                if (barH > 15) ctx.fillText(errors[d].toFixed(2), cx, margin.top + ph - barH + 12);
                            }
                            // Highlight selected
                            var selX = margin.left + (numComp + 0.5) / (D + 1) * pw;
                            ctx.fillStyle = viz.colors.orange; ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center';
                            var pctExplained = ((totalVar - errors[numComp]) / totalVar * 100).toFixed(1);
                            ctx.fillText('d=' + numComp + ': error=' + errors[numComp].toFixed(2) + ', explained=' + pctExplained + '%', W / 2, 22);
                        }
                        draw();
                        return { stopAnimation: function() {} };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Prove the Pythagorean identity used in the reconstruction error proof: for any orthogonal projection \\(\\hat{\\mathbf{x}} = \\mathbf{U}_d \\mathbf{U}_d^\\top \\mathbf{x}\\), we have \\(\\|\\mathbf{x}\\|^2 = \\|\\hat{\\mathbf{x}}\\|^2 + \\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|^2\\).',
                    hint: 'Show that \\(\\hat{\\mathbf{x}}\\) and \\(\\mathbf{x} - \\hat{\\mathbf{x}}\\) are orthogonal, i.e., \\(\\hat{\\mathbf{x}}^\\top(\\mathbf{x} - \\hat{\\mathbf{x}}) = 0\\).',
                    solution: 'Let \\(\\mathbf{P} = \\mathbf{U}_d \\mathbf{U}_d^\\top\\). Note \\(\\mathbf{P}^2 = \\mathbf{P}\\) and \\(\\mathbf{P}^\\top = \\mathbf{P}\\). Then \\(\\hat{\\mathbf{x}}^\\top(\\mathbf{x} - \\hat{\\mathbf{x}}) = \\mathbf{x}^\\top \\mathbf{P}(\\mathbf{I} - \\mathbf{P})\\mathbf{x} = \\mathbf{x}^\\top(\\mathbf{P} - \\mathbf{P}^2)\\mathbf{x} = 0\\). So \\(\\|\\mathbf{x}\\|^2 = \\|\\hat{\\mathbf{x}} + (\\mathbf{x} - \\hat{\\mathbf{x}})\\|^2 = \\|\\hat{\\mathbf{x}}\\|^2 + 2\\hat{\\mathbf{x}}^\\top(\\mathbf{x}-\\hat{\\mathbf{x}}) + \\|\\mathbf{x}-\\hat{\\mathbf{x}}\\|^2 = \\|\\hat{\\mathbf{x}}\\|^2 + \\|\\mathbf{x}-\\hat{\\mathbf{x}}\\|^2\\).'
                },
                {
                    question: 'A dataset has eigenvalues \\(\\lambda_1 = 20, \\lambda_2 = 5, \\lambda_3 = 2, \\lambda_4 = 1\\). What is the minimum number of components \\(d\\) needed to explain at least 90% of the total variance?',
                    hint: 'Compute cumulative explained variance: \\(\\sum_{j=1}^d \\lambda_j / \\sum_{j=1}^D \\lambda_j\\) for increasing \\(d\\).',
                    solution: 'Total variance = 28. For \\(d=1\\): 20/28 = 71.4%. For \\(d=2\\): 25/28 = 89.3%. For \\(d=3\\): 27/28 = 96.4%. So \\(d = 3\\) is the minimum to exceed 90%.'
                },
                {
                    question: 'Show that minimizing reconstruction error and maximizing projected variance give the same solution by deriving \\(\\mathcal{E}_d = \\text{tr}(\\mathbf{S}) - \\text{tr}(\\mathbf{U}_d^\\top \\mathbf{S} \\mathbf{U}_d)\\).',
                    hint: 'Start from \\(\\mathcal{E}_d = \\frac{1}{N}\\sum_i \\|\\mathbf{x}_i - \\mathbf{U}_d\\mathbf{U}_d^\\top\\mathbf{x}_i\\|^2\\) and expand the squared norm.',
                    solution: '\\(\\mathcal{E}_d = \\frac{1}{N}\\sum_i \\|(\\mathbf{I} - \\mathbf{U}_d\\mathbf{U}_d^\\top)\\mathbf{x}_i\\|^2 = \\frac{1}{N}\\sum_i \\mathbf{x}_i^\\top(\\mathbf{I} - \\mathbf{U}_d\\mathbf{U}_d^\\top)^2\\mathbf{x}_i\\). Since \\(\\mathbf{I} - \\mathbf{U}_d\\mathbf{U}_d^\\top\\) is an orthogonal projector, it is idempotent, so \\(\\mathcal{E}_d = \\text{tr}((\\mathbf{I}-\\mathbf{U}_d\\mathbf{U}_d^\\top)\\mathbf{S}) = \\text{tr}(\\mathbf{S}) - \\text{tr}(\\mathbf{U}_d^\\top\\mathbf{S}\\mathbf{U}_d)\\). Since \\(\\text{tr}(\\mathbf{S})\\) is fixed, minimizing \\(\\mathcal{E}_d\\) is equivalent to maximizing \\(\\text{tr}(\\mathbf{U}_d^\\top\\mathbf{S}\\mathbf{U}_d) = \\sum_{j=1}^d \\lambda_j\\).'
                }
            ]
        },

        // ===================== Section 3: Eigenvalue Decomposition & Scree Plot =====================
        {
            id: 'ch12-sec03',
            title: 'Eigenvalue Decomposition & Scree Plot',
            content: `<h2>Eigenvalue Decomposition &amp; Scree Plot</h2>

                <h3>Spectral Decomposition of the Covariance</h3>

                <p>Since \\(\\mathbf{S}\\) is a real symmetric positive semi-definite matrix, it admits an eigendecomposition</p>
                \\[\\mathbf{S} = \\mathbf{U} \\boldsymbol{\\Lambda} \\mathbf{U}^\\top = \\sum_{j=1}^{D} \\lambda_j \\mathbf{u}_j \\mathbf{u}_j^\\top\\]
                <p>where \\(\\mathbf{U} = [\\mathbf{u}_1, \\dots, \\mathbf{u}_D]\\) is orthogonal and \\(\\boldsymbol{\\Lambda} = \\text{diag}(\\lambda_1, \\dots, \\lambda_D)\\) with \\(\\lambda_1 \\geq \\cdots \\geq \\lambda_D \\geq 0\\).</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Explained Variance Ratio)</div>
                    <div class="env-body"><p>The <strong>explained variance ratio</strong> of the \\(j\\)-th component is
                    \\[r_j = \\frac{\\lambda_j}{\\sum_{k=1}^D \\lambda_k}\\]
                    This is the fraction of total variance captured by the \\(j\\)-th principal component. The <strong>cumulative explained variance</strong> after \\(d\\) components is \\(R_d = \\sum_{j=1}^{d} r_j\\).</p></div>
                </div>

                <h3>The Scree Plot</h3>

                <p>A <strong>scree plot</strong> (Cattell, 1966) displays the eigenvalues \\(\\lambda_1, \\lambda_2, \\dots\\) in decreasing order. It is the primary visual tool for deciding how many components to retain. The name comes from the geological term for the rubble at the base of a cliff: we look for the point where the steep descent transitions to a flat "scree" of small eigenvalues.</p>

                <div class="env-block definition">
                    <div class="env-title">Definition (Elbow Method)</div>
                    <div name="env-body"><div class="env-body"><p>The <strong>elbow method</strong> selects the number of components \\(d\\) at the "elbow" of the scree plot, i.e., the point where the rate of decrease in eigenvalues abruptly slows. Formally, one looks for the \\(d\\) that maximizes the curvature of the scree curve, though in practice this is done visually.</p></div></div>
                </div>

                <div class="env-block remark">
                    <div class="env-title">Remark (Alternative Selection Criteria)</div>
                    <div class="env-body"><p>Besides the elbow method, common criteria include: (1) retaining enough components to explain a target fraction (e.g., 95%) of total variance; (2) Kaiser's rule, which retains components with \\(\\lambda_j \\geq \\bar{\\lambda} = \\text{tr}(\\mathbf{S})/D\\) (i.e., above-average variance); (3) cross-validation, holding out data and choosing \\(d\\) that minimizes held-out reconstruction error.</p></div>
                </div>

                <h3>Practical Computation</h3>

                <p>For moderate \\(D\\), we compute the full eigendecomposition of \\(\\mathbf{S}\\). For high-dimensional data (\\(D \\gg N\\)), it is more efficient to compute the SVD of \\(\\mathbf{X}\\) directly, or use randomized SVD algorithms (Halko, Martinsson, Tropp 2011) that compute only the top \\(d\\) singular values/vectors in \\(O(NDd)\\) time.</p>

                <div class="viz-placeholder" data-viz="viz-scree-plot"></div>`,

            visualizations: [
                {
                    id: 'viz-scree-plot',
                    title: 'Interactive Scree Plot with Cumulative Variance',
                    description: 'The bar chart shows individual eigenvalues (explained variance per component). The orange curve shows cumulative explained variance. The dashed line marks the selected cutoff.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var cutoff = 4;
                        // Eigenvalue spectrum (exponential-ish decay with elbow)
                        var eigenvals = [8.5, 5.2, 3.8, 1.2, 0.7, 0.4, 0.25, 0.15, 0.08, 0.02];
                        var D = eigenvals.length;
                        var totalVar = eigenvals.reduce(function(a, b) { return a + b; }, 0);

                        VizEngine.createSlider(controls, 'Cutoff d', 1, D, 4, 1, function(v) { cutoff = Math.round(v); draw(); });

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);
                            var m = {top: 45, right: 60, bottom: 50, left: 60};
                            var pw = W - m.left - m.right;
                            var ph = H - m.top - m.bottom;
                            // Axes
                            ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                            ctx.beginPath(); ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + ph); ctx.lineTo(m.left + pw, m.top + ph); ctx.stroke();
                            // Right axis for cumulative
                            ctx.beginPath(); ctx.moveTo(m.left + pw, m.top); ctx.lineTo(m.left + pw, m.top + ph); ctx.stroke();
                            // Labels
                            ctx.fillStyle = viz.colors.text; ctx.font = '12px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            ctx.fillText('Component index j', m.left + pw / 2, H - 8);
                            ctx.save(); ctx.translate(14, m.top + ph / 2); ctx.rotate(-Math.PI / 2);
                            ctx.fillStyle = viz.colors.blue; ctx.fillText('Eigenvalue \u03BBj', 0, 0); ctx.restore();
                            ctx.save(); ctx.translate(W - 8, m.top + ph / 2); ctx.rotate(Math.PI / 2);
                            ctx.fillStyle = viz.colors.orange; ctx.fillText('Cumulative %', 0, 0); ctx.restore();
                            var maxEig = eigenvals[0];
                            var barW = pw / D * 0.55;
                            // Bars
                            for (var j = 0; j < D; j++) {
                                var cx = m.left + (j + 0.5) / D * pw;
                                var barH = (eigenvals[j] / maxEig) * ph * 0.85;
                                var color = j < cutoff ? viz.colors.blue : viz.colors.text + '66';
                                ctx.fillStyle = color; ctx.fillRect(cx - barW / 2, m.top + ph - barH, barW, barH);
                                ctx.fillStyle = viz.colors.text; ctx.font = '10px -apple-system,sans-serif';
                                ctx.textAlign = 'center'; ctx.fillText((j + 1).toString(), cx, m.top + ph + 14);
                            }
                            // Cumulative variance curve
                            var cumSum = 0;
                            ctx.strokeStyle = viz.colors.orange; ctx.lineWidth = 2.5;
                            ctx.beginPath();
                            for (var j = 0; j < D; j++) {
                                cumSum += eigenvals[j];
                                var pct = cumSum / totalVar;
                                var cx = m.left + (j + 0.5) / D * pw;
                                var cy = m.top + ph - pct * ph * 0.85;
                                if (j === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
                            }
                            ctx.stroke();
                            // Cumulative dots
                            cumSum = 0;
                            for (var j = 0; j < D; j++) {
                                cumSum += eigenvals[j];
                                var pct = cumSum / totalVar;
                                var cx = m.left + (j + 0.5) / D * pw;
                                var cy = m.top + ph - pct * ph * 0.85;
                                ctx.fillStyle = viz.colors.orange;
                                ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
                                // Percentage labels
                                ctx.fillStyle = viz.colors.orange; ctx.font = '9px -apple-system,sans-serif';
                                ctx.textAlign = 'left'; ctx.fillText((pct * 100).toFixed(0) + '%', cx + 6, cy - 2);
                            }
                            // Cutoff line
                            var cutX = m.left + (cutoff - 0.0) / D * pw;
                            ctx.strokeStyle = viz.colors.red + 'aa'; ctx.lineWidth = 1.5;
                            ctx.setLineDash([6, 4]);
                            ctx.beginPath(); ctx.moveTo(cutX, m.top); ctx.lineTo(cutX, m.top + ph); ctx.stroke();
                            ctx.setLineDash([]);
                            // Title info
                            cumSum = 0; for (var j = 0; j < cutoff; j++) cumSum += eigenvals[j];
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            ctx.fillText('Retaining d=' + cutoff + ' components: ' + (cumSum / totalVar * 100).toFixed(1) + '% variance explained', W / 2, 22);
                            // Right axis ticks
                            ctx.fillStyle = viz.colors.orange; ctx.font = '10px -apple-system,sans-serif'; ctx.textAlign = 'left';
                            for (var p = 0; p <= 100; p += 25) {
                                var yy = m.top + ph - (p / 100) * ph * 0.85;
                                ctx.fillText(p + '%', m.left + pw + 4, yy + 3);
                            }
                        }
                        draw();
                        return { stopAnimation: function() {} };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Given the eigenvalue spectrum \\(\\lambda_1 = 10, \\lambda_2 = 4, \\lambda_3 = 2, \\lambda_4 = 1, \\lambda_5 = 0.5\\), compute the explained variance ratio for each component and the cumulative explained variance for \\(d = 1, 2, 3\\).',
                    hint: 'Total variance is the sum of all eigenvalues.',
                    solution: 'Total = 17.5. Ratios: \\(r_1 = 10/17.5 = 57.1\\%\\), \\(r_2 = 22.9\\%\\), \\(r_3 = 11.4\\%\\), \\(r_4 = 5.7\\%\\), \\(r_5 = 2.9\\%\\). Cumulative: \\(R_1 = 57.1\\%\\), \\(R_2 = 80.0\\%\\), \\(R_3 = 91.4\\%\\).'
                },
                {
                    question: 'The Kaiser criterion retains components with \\(\\lambda_j \\geq \\bar{\\lambda}\\) where \\(\\bar{\\lambda} = \\text{tr}(\\mathbf{S})/D\\). For the eigenvalues \\(\\{8.5, 5.2, 3.8, 1.2, 0.7, 0.4, 0.25, 0.15, 0.08, 0.02\\}\\), how many components does Kaiser retain?',
                    hint: 'Compute the mean eigenvalue \\(\\bar{\\lambda}\\) and count how many eigenvalues exceed it.',
                    solution: '\\(\\bar{\\lambda} = 20.35/10 = 2.035\\). Components with \\(\\lambda_j \\geq 2.035\\): \\(\\lambda_1 = 8.5, \\lambda_2 = 5.2, \\lambda_3 = 3.8\\). Kaiser retains \\(d = 3\\) components, explaining \\((8.5+5.2+3.8)/20.35 = 86.0\\%\\) of variance.'
                },
                {
                    question: 'Explain why randomized SVD is preferred over full eigendecomposition when \\(D = 10{,}000\\) but we only need \\(d = 50\\) components. What is the computational advantage?',
                    hint: 'Compare the complexity of full eigendecomposition \\(O(D^3)\\) vs. randomized SVD \\(O(NDd)\\).',
                    solution: 'Full eigendecomposition of \\(\\mathbf{S} \\in \\mathbb{R}^{10000 \\times 10000}\\) costs \\(O(D^3) = O(10^{12})\\). Randomized SVD computes only the top-50 singular values/vectors in \\(O(NDd) = O(N \\cdot 10000 \\cdot 50) = O(5 \\times 10^5 N)\\), which is vastly cheaper. It works by projecting onto a random low-dimensional subspace, then computing the SVD in that subspace. The Halko-Martinsson-Tropp algorithm guarantees near-optimal approximation with high probability.'
                }
            ]
        },

        // ===================== Section 4: PCA Applications =====================
        {
            id: 'ch12-sec04',
            title: 'PCA Applications',
            content: `<h2>PCA Applications</h2>

                <h3>Whitening (Sphering)</h3>

                <div class="env-block definition">
                    <div class="env-title">Definition (PCA Whitening)</div>
                    <div class="env-body"><p><strong>PCA whitening</strong> transforms the data so that each principal component has unit variance and the components are uncorrelated. For centered data \\(\\mathbf{x}\\), the whitened representation is
                    \\[\\mathbf{z} = \\boldsymbol{\\Lambda}^{-1/2} \\mathbf{U}^\\top \\mathbf{x}\\]
                    where \\(\\mathbf{U}\\) and \\(\\boldsymbol{\\Lambda}\\) come from the eigendecomposition \\(\\mathbf{S} = \\mathbf{U}\\boldsymbol{\\Lambda}\\mathbf{U}^\\top\\). The covariance of \\(\\mathbf{z}\\) is the identity: \\(\\text{Cov}(\\mathbf{z}) = \\mathbf{I}\\).</p></div>
                </div>

                <p>Whitening is useful as a preprocessing step for many algorithms (e.g., ICA, neural networks) because it removes correlations and equalizes scales. Geometrically, it transforms an elliptical data cloud into a spherical one.</p>

                <div class="env-block remark">
                    <div class="env-title">Remark (ZCA Whitening)</div>
                    <div class="env-body"><p><strong>ZCA whitening</strong> (zero-phase component analysis) applies \\(\\mathbf{z} = \\mathbf{U}\\boldsymbol{\\Lambda}^{-1/2}\\mathbf{U}^\\top\\mathbf{x}\\), which also produces identity covariance but stays as close as possible to the original space. ZCA is often preferred for image preprocessing because it preserves spatial structure.</p></div>
                </div>

                <h3>Dimensionality Reduction for Visualization</h3>

                <p>Projecting onto the first two or three principal components provides a low-dimensional view of high-dimensional data. While PCA only finds linear projections, it is fast, deterministic, and provides a natural ordering of components by importance. It is commonly the first step in exploratory data analysis.</p>

                <h3>Denoising via PCA</h3>

                <div class="env-block theorem">
                    <div class="env-title">Proposition (PCA Denoising)</div>
                    <div class="env-body"><p>If the data model is \\(\\mathbf{x} = \\mathbf{s} + \\boldsymbol{\\epsilon}\\), where \\(\\mathbf{s}\\) lies in a \\(d\\)-dimensional subspace and \\(\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(0, \\sigma^2 \\mathbf{I})\\) is isotropic noise, then projecting onto the top-\\(d\\) principal components recovers an estimate of \\(\\mathbf{s}\\) that removes the noise in the discarded dimensions. The reconstruction \\(\\hat{\\mathbf{s}} = \\mathbf{U}_d \\mathbf{U}_d^\\top \\mathbf{x}\\) has lower MSE than the raw observation \\(\\mathbf{x}\\) whenever \\(\\sigma^2 (D - d) \\gt 0\\).</p></div>
                </div>

                <p>The intuition is simple: noise contributes equally to all directions, while signal concentrates in the top components. By discarding the bottom \\(D - d\\) components, we remove more noise than signal.</p>

                <h3>Feature Extraction and Compression</h3>

                <p>In practice, PCA is used as a preprocessing step to reduce dimensionality before applying other algorithms (SVM, logistic regression, k-NN). It reduces computational cost, mitigates the curse of dimensionality, and can even improve generalization by removing noisy features. The classic application is Eigenfaces (Turk and Pentland, 1991), where PCA on face images yields a compact basis of "eigenface" components.</p>

                <div class="viz-placeholder" data-viz="viz-whitening"></div>`,

            visualizations: [
                {
                    id: 'viz-whitening',
                    title: 'PCA Whitening: Before and After',
                    description: 'Left: original correlated data (elliptical cloud). Right: after PCA whitening (spherical cloud). Toggle to see the transformation.',
                    setup: function(body, controls) {
                        var viz = new VizEngine(body, {scale: 35});
                        var ctx = viz.ctx;
                        var showWhitened = false;
                        var animProgress = 0;
                        var animating = false;

                        VizEngine.createButton(controls, 'Toggle Whitening', function() {
                            showWhitened = !showWhitened;
                            animating = true;
                            animProgress = 0;
                        });

                        // Generate correlated 2D data
                        var rawPts = [];
                        var seed = 123;
                        function srand() { seed = (seed * 16807 + 0) % 2147483647; return (seed - 1) / 2147483646; }
                        for (var i = 0; i < 80; i++) {
                            var u1 = srand(), u2 = srand();
                            var z1 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                            var z2 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);
                            rawPts.push([z1, z2]);
                        }
                        // Covariance: correlated ellipse
                        var angle = 0.5, s1 = 2.5, s2 = 0.7;
                        var ca = Math.cos(angle), sa = Math.sin(angle);
                        var origPts = rawPts.map(function(p) {
                            var x = p[0] * s1, y = p[1] * s2;
                            return [ca * x - sa * y, sa * x + ca * y];
                        });
                        // Whitened = just the raw standard normal
                        var whitePts = rawPts.map(function(p) { return [p[0], p[1]]; });

                        function draw(t) {
                            viz.clear(); viz.drawGrid(); viz.drawAxes();
                            if (animating) {
                                animProgress += 0.03;
                                if (animProgress >= 1) { animProgress = 1; animating = false; }
                            }
                            var blend = showWhitened ? animProgress : (1 - animProgress);
                            for (var i = 0; i < origPts.length; i++) {
                                var ox = origPts[i][0], oy = origPts[i][1];
                                var wx = whitePts[i][0], wy = whitePts[i][1];
                                var px = ox + (wx - ox) * blend;
                                var py = oy + (wy - oy) * blend;
                                viz.drawPoint(px, py, viz.colors.blue + 'cc', null, 3.5);
                            }
                            // Draw ellipse / circle outline
                            if (blend < 0.5) {
                                viz.drawEllipse(0, 0, s1 * 1.5, s2 * 1.5, angle, null, viz.colors.orange + '66');
                            } else {
                                viz.drawCircle(0, 0, 1.5, null, viz.colors.teal + '66', 1.5);
                            }
                            // Labels
                            ctx.font = 'bold 14px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            if (blend < 0.5) {
                                ctx.fillStyle = viz.colors.orange;
                                ctx.fillText('Original: correlated, elliptical', viz.width / 2, 20);
                            } else {
                                ctx.fillStyle = viz.colors.teal;
                                ctx.fillText('Whitened: uncorrelated, spherical (Cov = I)', viz.width / 2, 20);
                            }
                        }
                        viz.animate(draw);
                        return viz;
                    }
                }
            ],

            exercises: [
                {
                    question: 'Verify that after PCA whitening \\(\\mathbf{z} = \\boldsymbol{\\Lambda}^{-1/2}\\mathbf{U}^\\top\\mathbf{x}\\), we have \\(\\text{Cov}(\\mathbf{z}) = \\mathbf{I}\\).',
                    hint: 'Compute \\(\\text{Cov}(\\mathbf{z}) = \\mathbb{E}[\\mathbf{z}\\mathbf{z}^\\top]\\) using \\(\\mathbf{S} = \\mathbf{U}\\boldsymbol{\\Lambda}\\mathbf{U}^\\top\\).',
                    solution: '\\(\\text{Cov}(\\mathbf{z}) = \\boldsymbol{\\Lambda}^{-1/2}\\mathbf{U}^\\top \\, \\text{Cov}(\\mathbf{x}) \\, \\mathbf{U}\\boldsymbol{\\Lambda}^{-1/2} = \\boldsymbol{\\Lambda}^{-1/2}\\mathbf{U}^\\top \\mathbf{U}\\boldsymbol{\\Lambda}\\mathbf{U}^\\top\\mathbf{U}\\boldsymbol{\\Lambda}^{-1/2} = \\boldsymbol{\\Lambda}^{-1/2}\\boldsymbol{\\Lambda}\\boldsymbol{\\Lambda}^{-1/2} = \\mathbf{I}\\).'
                },
                {
                    question: 'A dataset of 100-pixel images (D=100) is corrupted by Gaussian noise with \\(\\sigma^2 = 0.5\\). The top 10 eigenvalues of the noisy data are all above 3.0, while eigenvalues 11 through 100 are all near 0.5. How many PCA components should you retain for denoising, and why?',
                    hint: 'Noise contributes \\(\\sigma^2 = 0.5\\) equally to every eigenvalue. Signal eigenvalues exceed this floor.',
                    solution: 'Retain \\(d = 10\\) components. The noise contributes \\(\\sigma^2 = 0.5\\) to each eigenvalue. Components 11-100 have eigenvalues near 0.5, meaning they carry essentially only noise. Components 1-10 have eigenvalues well above 0.5, indicating signal. Projecting onto the top 10 removes noise from 90 dimensions while preserving the signal subspace. The noise reduction is approximately \\(90 \\times 0.5 = 45\\) units of variance removed.'
                },
                {
                    question: 'Explain why ZCA whitening \\(\\mathbf{z} = \\mathbf{U}\\boldsymbol{\\Lambda}^{-1/2}\\mathbf{U}^\\top\\mathbf{x}\\) produces identity covariance but stays closer to the original data than PCA whitening. In what sense is it "closest"?',
                    hint: 'ZCA whitening is the whitening transform that minimizes \\(\\mathbb{E}[\\|\\mathbf{z} - \\mathbf{x}\\|^2]\\). Think about the extra rotation \\(\\mathbf{U}\\) vs. not rotating back.',
                    solution: 'Both PCA whitening \\(\\boldsymbol{\\Lambda}^{-1/2}\\mathbf{U}^\\top\\) and ZCA whitening \\(\\mathbf{U}\\boldsymbol{\\Lambda}^{-1/2}\\mathbf{U}^\\top\\) produce identity covariance. PCA whitening rotates the data into the eigenvector coordinate system; ZCA rotates it back. Among all whitening transforms \\(\\mathbf{W}\\) satisfying \\(\\mathbf{W}\\mathbf{S}\\mathbf{W}^\\top = \\mathbf{I}\\), the ZCA solution \\(\\mathbf{W}_{\\text{ZCA}} = \\mathbf{S}^{-1/2}\\) minimizes \\(\\|\\mathbf{W} - \\mathbf{I}\\|_F\\), making it the closest whitening transform to the identity. This preserves spatial locality in images.'
                }
            ]
        }
    ]
});
