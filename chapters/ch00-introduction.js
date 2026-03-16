// === Chapter 0: Introduction & Learning Paradigms ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch00',
    number: 0,
    title: 'Introduction & Learning Paradigms',
    subtitle: 'What machines learn, how they learn it, and the three paradigms that organize the field',
    sections: [
        // ========== SECTION 1: What is Machine Learning? ==========
        {
            id: 'sec01-what-is-ml',
            title: 'What is Machine Learning?',
            content: `
<h2>0.1 What is Machine Learning?</h2>
<div class="env-block intuition">
<strong>The Core Idea.</strong> Traditional programming requires a human to write explicit rules mapping inputs to outputs. Machine learning inverts this: given examples, the algorithm discovers the rules automatically. This shift from <em>programming rules</em> to <em>learning rules from data</em> is what makes ML powerful for problems where hand-coded rules are infeasible.
</div>
<h3>Mitchell's Definition</h3>
<div class="env-block definition">
<strong>Definition 0.1.1 (Machine Learning, Mitchell 1997).</strong> A computer program is said to <em>learn</em> from experience \\(E\\) with respect to some class of tasks \\(T\\) and performance measure \\(P\\), if its performance at tasks in \\(T\\), as measured by \\(P\\), improves with experience \\(E\\).
</div>
<p>Every ML system has three components:</p>
<ul>
<li><strong>Task \\(T\\)</strong>: What the system should do (classify images, predict prices, generate text).</li>
<li><strong>Performance \\(P\\)</strong>: How we measure success (accuracy, mean squared error, BLEU score).</li>
<li><strong>Experience \\(E\\)</strong>: The data the system learns from (labeled examples, game trajectories, raw text).</li>
</ul>
<div class="env-block example">
<strong>Example 0.1.2 (Spam Filtering).</strong> \\(T\\) = classify emails as spam or not-spam; \\(P\\) = fraction correctly classified; \\(E\\) = a corpus of emails each labeled "spam" or "ham." The algorithm discovers patterns (keywords, sender addresses) that distinguish spam from legitimate mail.
</div>
<h3>Types of Experience</h3>
<p>The nature of \\(E\\) determines the <em>learning paradigm</em>:</p>
<ul>
<li><strong>Supervised learning</strong>: \\(E\\) consists of input-output pairs \\(\\{(\\mathbf{x}_i, y_i)\\}_{i=1}^n\\). The algorithm learns \\(f: \\mathcal{X} \\to \\mathcal{Y}\\).</li>
<li><strong>Unsupervised learning</strong>: \\(E\\) consists of inputs only \\(\\{\\mathbf{x}_i\\}_{i=1}^n\\). The algorithm discovers structure.</li>
<li><strong>Reinforcement learning</strong>: \\(E\\) consists of state-action-reward sequences. The agent learns a policy maximizing cumulative reward.</li>
</ul>
<div class="viz-placeholder" data-viz="viz-supervised-demo"></div>
<h3>Why "Learning" Works</h3>
<p>ML rests on a regularity assumption: patterns in training data generalize to unseen data. Formally, we assume train and test examples are drawn from the same distribution \\(P(\\mathbf{x}, y)\\). The goal is to minimize the <em>expected risk</em>:</p>
\\[
R(\\hat{f}) = \\mathbb{E}_{(\\mathbf{x}, y) \\sim P}\\bigl[\\ell(\\hat{f}(\\mathbf{x}), y)\\bigr].
\\]
<p>Since \\(P\\) is unknown, we approximate with the <em>empirical risk</em> over \\(n\\) samples:</p>
\\[
\\hat{R}(\\hat{f}) = \\frac{1}{n} \\sum_{i=1}^n \\ell(\\hat{f}(\\mathbf{x}_i), y_i).
\\]
<div class="env-block remark">
<strong>The Fundamental Tension.</strong> Minimizing empirical risk too aggressively leads to <em>overfitting</em>: the model memorizes training data but fails on new data. Managing this tension is the central challenge of ML.
</div>
`,
            visualizations: [{
                id: 'viz-supervised-demo',
                title: 'Interactive Supervised Learning',
                description: 'Click to add training points (blue = class 0, orange = class 1). Toggle class with the button. A logistic boundary updates in real time. Drag points to see the boundary shift.',
                setup(body, controls) {
                    const viz = new VizEngine(body, { scale: 40, originX: 280, originY: 240 });
                    let points = [
                        {x:-2,y:1,c:0},{x:-1.5,y:2,c:0},{x:-1,y:0.5,c:0},{x:-2.5,y:-0.5,c:0},{x:-0.5,y:1.5,c:0},
                        {x:1.5,y:-1,c:1},{x:2,y:0,c:1},{x:1,y:-1.5,c:1},{x:2.5,y:0.5,c:1},{x:1.8,y:-0.5,c:1}
                    ];
                    let curClass = 0, dragIdx = -1;
                    const classBtn = VizEngine.createButton(controls, 'Current: Class 0 (blue)', () => {
                        curClass = 1 - curClass;
                        classBtn.textContent = curClass === 0 ? 'Current: Class 0 (blue)' : 'Current: Class 1 (orange)';
                    });
                    VizEngine.createButton(controls, 'Clear All', () => { points = []; });

                    function fitBoundary() {
                        if (points.length < 2) return null;
                        let w0=0, w1=0, b=0; const lr = 0.3;
                        for (let iter = 0; iter < 200; iter++) {
                            let dw0=0, dw1=0, db=0;
                            for (const p of points) { const s = 1/(1+Math.exp(-(w0*p.x+w1*p.y+b))); const e = s-p.c; dw0+=e*p.x; dw1+=e*p.y; db+=e; }
                            w0 -= lr*dw0/points.length; w1 -= lr*dw1/points.length; b -= lr*db/points.length;
                        }
                        return {w0,w1,b};
                    }
                    viz.canvas.addEventListener('mousedown', (e) => {
                        const r = viz.canvas.getBoundingClientRect();
                        const [wx,wy] = viz.toMath(e.clientX-r.left, e.clientY-r.top);
                        dragIdx = -1;
                        for (let i = 0; i < points.length; i++) { if (Math.hypot(points[i].x-wx, points[i].y-wy) < 0.4) { dragIdx = i; return; } }
                        points.push({x:wx, y:wy, c:curClass});
                    });
                    viz.canvas.addEventListener('mousemove', (e) => {
                        if (dragIdx < 0) return;
                        const r = viz.canvas.getBoundingClientRect();
                        const [wx,wy] = viz.toMath(e.clientX-r.left, e.clientY-r.top);
                        points[dragIdx].x = wx; points[dragIdx].y = wy;
                    });
                    viz.canvas.addEventListener('mouseup', () => { dragIdx = -1; });
                    viz.canvas.addEventListener('mouseleave', () => { dragIdx = -1; });

                    function draw() {
                        viz.clear(); viz.drawGrid(); viz.drawAxes();
                        const m = fitBoundary();
                        if (m) {
                            const ctx = viz.ctx, st = 4;
                            for (let px = 0; px < viz.width; px += st) for (let py = 0; py < viz.height; py += st) {
                                const [wx,wy] = viz.toMath(px,py);
                                ctx.fillStyle = 1/(1+Math.exp(-(m.w0*wx+m.w1*wy+m.b))) > 0.5 ? 'rgba(240,136,62,0.07)' : 'rgba(88,166,255,0.07)';
                                ctx.fillRect(px,py,st,st);
                            }
                            if (Math.abs(m.w1) > 1e-6) { const xL=-8,xR=8; viz.drawSegment(xL,-(m.w0*xL+m.b)/m.w1,xR,-(m.w0*xR+m.b)/m.w1,viz.colors.white+'aa',2); }
                            else if (Math.abs(m.w0) > 1e-6) { const xB=-m.b/m.w0; viz.drawSegment(xB,-8,xB,8,viz.colors.white+'aa',2); }
                        }
                        for (const p of points) viz.drawPoint(p.x, p.y, p.c===0 ? viz.colors.blue : viz.colors.orange, null, 6);
                        viz.screenText('Click to add points. Drag to move.', viz.width/2, viz.height-14, viz.colors.text, 11);
                    }
                    viz.animate(draw);
                    return viz;
                }
            }],
            exercises: [
                { question: 'For each scenario, identify \\(T\\), \\(P\\), and \\(E\\) in Mitchell\'s framework: (a) a self-driving car learning to stay in lane; (b) a movie recommendation system.',
                  hint: 'Think about what data the system consumes, what output it produces, and how you measure quality.',
                  solution: '(a) \\(T\\) = steer to stay centered in lane; \\(P\\) = deviation from center (lower is better); \\(E\\) = sequences of camera images paired with steering angles. (b) \\(T\\) = rank movies for a user; \\(P\\) = click-through rate or precision@k; \\(E\\) = historical user-movie interaction data (ratings, clicks, watch times).' },
                { question: 'Explain why minimizing empirical risk \\(\\hat{R}(\\hat{f})\\) does not guarantee good performance on unseen data. Give a concrete example where \\(\\hat{R}=0\\) yet \\(R\\) is terrible.',
                  hint: 'Consider a lookup table that memorizes every training example.',
                  solution: 'A lookup table storing every \\((\\mathbf{x}_i, y_i)\\) and outputting randomly for unseen \\(\\mathbf{x}\\) achieves \\(\\hat{R}=0\\) but essentially random \\(R\\). Empirical risk is computed on a finite sample and may not reflect the true distribution, especially when the model is flexible enough to fit noise. Regularization, early stopping, or architectural constraints are needed for generalization.' },
                { question: 'Algorithm A memorizes all training spam emails and flags exact matches. Algorithm B learns weighted keyword features. Which generalizes better and why?',
                  hint: 'What happens when a spammer changes a single word?',
                  solution: 'Algorithm B generalizes better. Exact-match memorization fails on any email not seen verbatim, since spammers constantly vary messages. Weighted keywords abstract to general patterns (e.g., "lottery" is spam-indicative regardless of context). This illustrates the importance of learning smooth, generalizable features over memorizing instances.' }
            ]
        },
        // ========== SECTION 2: Supervised Learning ==========
        {
            id: 'sec02-supervised-learning',
            title: 'Supervised Learning',
            content: `
<h2>0.2 Supervised Learning</h2>
<p>The setup: a training set \\(\\mathcal{D} = \\{(\\mathbf{x}_i, y_i)\\}_{i=1}^n\\), where \\(\\mathbf{x}_i \\in \\mathcal{X}\\) is a feature vector and \\(y_i \\in \\mathcal{Y}\\) is its label. The goal is to learn \\(f: \\mathcal{X} \\to \\mathcal{Y}\\) that predicts well on unseen data.</p>
<h3>Classification vs. Regression</h3>
<div class="env-block definition">
<strong>Definition 0.2.1 (Classification).</strong> When \\(\\mathcal{Y}\\) is a finite set of discrete labels (e.g., \\(\\{0,1\\}\\) or \\(\\{1,\\ldots,K\\}\\)), the task is <em>classification</em>. Common losses: 0-1 loss \\(\\ell(\\hat{y},y)=\\mathbf{1}[\\hat{y}\\neq y]\\), cross-entropy.
</div>
<div class="env-block definition">
<strong>Definition 0.2.2 (Regression).</strong> When \\(\\mathcal{Y} = \\mathbb{R}\\), the task is <em>regression</em>. The standard loss is squared error \\(\\ell(\\hat{y},y)=(\\hat{y}-y)^2\\).
</div>
<h3>Training, Validation, and Test</h3>
<ul>
<li><strong>Training set</strong>: fit model parameters.</li>
<li><strong>Validation set</strong>: tune hyperparameters and select among models.</li>
<li><strong>Test set</strong>: used <em>once</em> at the end for an unbiased generalization estimate.</li>
</ul>
<div class="env-block remark">
<strong>The Cardinal Rule.</strong> Never use the test set for model selection or hyperparameter tuning. Doing so leaks test information and produces optimistic generalization estimates.
</div>
<h3>Model Complexity and Overfitting</h3>
<p>Too-simple models <em>underfit</em> (high train and test error). Too-complex models <em>overfit</em> (low train error, high test error from memorizing noise). The characteristic U-shaped test error curve as a function of complexity is the <em>bias-variance tradeoff</em> (Chapter 3).</p>
<div class="viz-placeholder" data-viz="viz-bias-variance"></div>
<div class="env-block example">
<strong>Example 0.2.3 (Polynomial Regression).</strong> Given noisy data from \\(y = \\sin(1.5x) + \\epsilon\\), a degree-1 polynomial underfits, degree 2-3 fits well, and degree 12+ overfits wildly. Test error is minimized near the true complexity.
</div>
`,
            visualizations: [{
                id: 'viz-bias-variance',
                title: 'Train/Test Error vs. Model Complexity',
                description: 'Adjust polynomial degree to fit noisy data. Watch training error decrease monotonically while test error follows a U-shape.',
                setup(body, controls) {
                    const viz = new VizEngine(body, { scale: 1, originX: 60, originY: 300 });
                    const ctx = viz.ctx, W = viz.width, H = viz.height;
                    let degree = 2;
                    // Seeded RNG
                    const rng = s => { let v=s; return()=>{v=(v*16807)%2147483647; return v/2147483647;}; };
                    const rand = rng(42);
                    const nTr=15, nTe=50, trX=[], trY=[], teX=[], teY=[];
                    for (let i=0;i<nTr;i++){const x=rand()*4-0.5; trX.push(x); trY.push(Math.sin(1.5*x)+(rand()-0.5)*0.6);}
                    for (let i=0;i<nTe;i++){const x=rand()*4-0.5; teX.push(x); teY.push(Math.sin(1.5*x)+(rand()-0.5)*0.6);}
                    VizEngine.createSlider(controls, 'Polynomial degree', 0, 12, 2, 1, v => { degree = Math.round(v); });

                    function polyFit(xs, ys, d) {
                        const n=xs.length, dd=Math.min(d,n-1), m=dd+1;
                        const XtX=Array.from({length:m},()=>new Array(m).fill(0)), XtY=new Array(m).fill(0);
                        for (let i=0;i<m;i++){for(let j=0;j<m;j++){for(let k=0;k<n;k++)XtX[i][j]+=Math.pow(xs[k],i)*Math.pow(xs[k],j);}XtX[i][i]+=1e-8;for(let k=0;k<n;k++)XtY[i]+=Math.pow(xs[k],i)*ys[k];}
                        const A=XtX.map((r,i)=>[...r,XtY[i]]);
                        for(let c=0;c<m;c++){let mr=c;for(let r=c+1;r<m;r++)if(Math.abs(A[r][c])>Math.abs(A[mr][c]))mr=r;[A[c],A[mr]]=[A[mr],A[c]];if(Math.abs(A[c][c])<1e-12)continue;for(let r=c+1;r<m;r++){const f=A[r][c]/A[c][c];for(let j=c;j<=m;j++)A[r][j]-=f*A[c][j];}}
                        const co=new Array(m).fill(0);for(let i=m-1;i>=0;i--){co[i]=A[i][m];for(let j=i+1;j<m;j++)co[i]-=A[i][j]*co[j];co[i]/=A[i][i];}return co;
                    }
                    function polyEval(c,x){let y=0;for(let i=0;i<c.length;i++)y+=c[i]*Math.pow(x,i);return y;}
                    function mse(xs,ys,c){let s=0;for(let i=0;i<xs.length;i++){const e=ys[i]-polyEval(c,xs[i]);s+=e*e;}return s/xs.length;}
                    const maxD=12, trErr=[], teErr=[];
                    for(let d=0;d<=maxD;d++){const c=polyFit(trX,trY,d);trErr.push(mse(trX,trY,c));teErr.push(mse(teX,teY,c));}
                    // Panel coords
                    const dL=60,dR=W/2-20,dT=30,dB=H-50, eL=W/2+30,eR=W-20,eT=30,eB=H-50;
                    const mapDX=x=>dL+(x+0.5)/4.5*(dR-dL), mapDY=y=>dB-(y+2)/4*(dB-dT);
                    const mapEX=d=>eL+d/maxD*(eR-eL);
                    const maxE=Math.max(0.5,...trErr,...teErr)*1.1;
                    const mapEY=e=>eB-e/maxE*(eB-eT);

                    function draw() {
                        ctx.fillStyle=viz.colors.bg; ctx.fillRect(0,0,W,H);
                        const coeffs=polyFit(trX,trY,degree);
                        // Left: data + fit
                        ctx.strokeStyle=viz.colors.grid;ctx.lineWidth=0.5;ctx.strokeRect(dL,dT,dR-dL,dB-dT);
                        ctx.strokeStyle=viz.colors.teal;ctx.lineWidth=2;ctx.beginPath();let first=true;
                        for(let px=dL;px<=dR;px+=2){const x=-0.5+(px-dL)/(dR-dL)*4.5;let y=Math.max(-2,Math.min(2,polyEval(coeffs,x)));const py=mapDY(y);first?(ctx.moveTo(px,py),first=false):ctx.lineTo(px,py);}ctx.stroke();
                        ctx.strokeStyle=viz.colors.white+'44';ctx.lineWidth=1;ctx.setLineDash([4,4]);ctx.beginPath();first=true;
                        for(let px=dL;px<=dR;px+=2){const x=-0.5+(px-dL)/(dR-dL)*4.5;const py=mapDY(Math.sin(1.5*x));first?(ctx.moveTo(px,py),first=false):ctx.lineTo(px,py);}ctx.stroke();ctx.setLineDash([]);
                        for(let i=0;i<nTr;i++){ctx.fillStyle=viz.colors.blue;ctx.beginPath();ctx.arc(mapDX(trX[i]),mapDY(trY[i]),4,0,Math.PI*2);ctx.fill();}
                        ctx.fillStyle=viz.colors.white;ctx.font='13px -apple-system,sans-serif';ctx.textAlign='center';
                        ctx.fillText('Degree '+degree+' Fit',(dL+dR)/2,dT-10);
                        ctx.fillStyle=viz.colors.text;ctx.font='10px -apple-system,sans-serif';
                        ctx.fillText('blue = data, teal = fit, dashed = true fn',(dL+dR)/2,dB+30);
                        // Right: error curves
                        ctx.strokeStyle=viz.colors.grid;ctx.lineWidth=0.5;ctx.strokeRect(eL,eT,eR-eL,eB-eT);
                        ctx.strokeStyle=viz.colors.blue;ctx.lineWidth=2;ctx.beginPath();for(let d=0;d<=maxD;d++){const px=mapEX(d),py=mapEY(trErr[d]);d===0?ctx.moveTo(px,py):ctx.lineTo(px,py);}ctx.stroke();
                        ctx.strokeStyle=viz.colors.orange;ctx.lineWidth=2;ctx.beginPath();for(let d=0;d<=maxD;d++){const px=mapEX(d),py=mapEY(teErr[d]);d===0?ctx.moveTo(px,py):ctx.lineTo(px,py);}ctx.stroke();
                        ctx.fillStyle=viz.colors.white;ctx.beginPath();ctx.arc(mapEX(degree),mapEY(trErr[degree]),5,0,Math.PI*2);ctx.fill();
                        ctx.fillStyle=viz.colors.orange;ctx.beginPath();ctx.arc(mapEX(degree),mapEY(teErr[degree]),5,0,Math.PI*2);ctx.fill();
                        ctx.fillStyle=viz.colors.white;ctx.font='13px -apple-system,sans-serif';ctx.textAlign='center';
                        ctx.fillText('Error vs. Complexity',(eL+eR)/2,eT-10);
                        ctx.font='11px -apple-system,sans-serif';ctx.fillStyle=viz.colors.blue;ctx.textAlign='left';ctx.fillText('Train',eR-75,eT+18);
                        ctx.fillStyle=viz.colors.orange;ctx.fillText('Test',eR-75,eT+34);
                        ctx.fillStyle=viz.colors.text;ctx.font='10px -apple-system,sans-serif';ctx.textAlign='center';
                        for(let d=0;d<=maxD;d+=2)ctx.fillText(d,mapEX(d),eB+14);
                        ctx.fillText('Complexity (degree)',(eL+eR)/2,eB+30);
                    }
                    viz.animate(draw); return viz;
                }
            }],
            exercises: [
                { question: 'A hospital predicts disease status (positive/negative) from blood tests. Is this classification or regression? Why might accuracy alone be misleading?',
                  hint: 'Consider what happens when the disease has 1% prevalence.',
                  solution: 'Binary <em>classification</em>. A model always predicting "negative" achieves 99% accuracy at 1% prevalence, yet catches zero cases. Metrics like precision, recall, F1, or AUC-ROC are more informative for imbalanced data.' },
                { question: 'Explain the difference between a validation set and a test set. Why do we need both?',
                  hint: 'If you use the test set to choose among models, is the test estimate still unbiased?',
                  solution: 'The validation set tunes hyperparameters and selects models. Since we repeatedly evaluate on it, we mildly overfit to it. The test set is held out entirely until final evaluation, providing an unbiased generalization estimate. Without a separate test set, we would have no reliable performance estimate on truly unseen data.' },
                { question: 'Describe qualitatively what happens to (a) training error, (b) test error, and (c) the fitted curve as polynomial degree goes from 1 to 20 on 15 training points.',
                  hint: 'A degree-\\(n\\) polynomial has \\(n+1\\) parameters. What happens when parameters \\(\\geq\\) data points?',
                  solution: '(a) Training error decreases monotonically, reaching zero when degree \\(\\geq 14\\) (exact interpolation of 15 points). (b) Test error initially decreases, reaches a minimum at moderate degree, then increases sharply. (c) The curve goes from a rigid line (underfitting) to a good fit at moderate degree, then to wild oscillations between data points (Runge phenomenon) at high degree.' }
            ]
        },
        // ========== SECTION 3: Unsupervised Learning ==========
        {
            id: 'sec03-unsupervised-learning',
            title: 'Unsupervised Learning',
            content: `
<h2>0.3 Unsupervised Learning</h2>
<p>In unsupervised learning, we observe only inputs \\(\\{\\mathbf{x}_i\\}_{i=1}^n\\) without labels. The goal is to discover hidden structure in the data.</p>
<h3>Clustering</h3>
<div class="env-block definition">
<strong>Definition 0.3.1 (K-Means Clustering).</strong> Given \\(\\{\\mathbf{x}_i\\}_{i=1}^n\\), K-means partitions data into \\(K\\) groups minimizing the within-cluster sum of squares:
\\[
J = \\sum_{k=1}^K \\sum_{\\mathbf{x}_i \\in C_k} \\|\\mathbf{x}_i - \\boldsymbol{\\mu}_k\\|^2,
\\]
where \\(\\boldsymbol{\\mu}_k\\) is the centroid of cluster \\(k\\). The algorithm alternates: (1) assign each point to its nearest centroid, (2) recompute centroids as cluster means.
</div>
<div class="env-block remark">
<strong>Convergence.</strong> Each step decreases \\(J\\), so K-means converges, but only to a <em>local minimum</em>. The result depends on initialization; in practice, run multiple restarts and keep the best.
</div>
<div class="viz-placeholder" data-viz="viz-kmeans"></div>
<h3>Dimensionality Reduction</h3>
<div class="env-block definition">
<strong>Definition 0.3.2 (PCA, Informal).</strong> Principal Component Analysis projects data onto the \\(d\\)-dimensional subspace capturing maximum variance. If \\(\\mathbf{X}\\) is the centered data matrix, PCA finds \\(\\mathbf{Z} = \\mathbf{X}\\mathbf{W}\\) where \\(\\mathbf{W}\\)'s columns are the top \\(d\\) eigenvectors of \\(\\mathbf{X}^\\top \\mathbf{X}/n\\). Details in Chapter 12.
</div>
<h3>Density Estimation</h3>
<p><em>Density estimation</em> learns \\(p(\\mathbf{x})\\) from data. Applications include anomaly detection (flag low-density points) and generative modeling (sample new data). Gaussian mixture models (Chapter 13) and kernel density estimation are classical approaches.</p>
`,
            visualizations: [{
                id: 'viz-kmeans',
                title: 'K-Means Clustering Animation',
                description: 'Press "Step" to advance one iteration (assign + update), or "Run" to animate. "Reset" generates fresh data. Centroids are marked with crosses.',
                setup(body, controls) {
                    const viz = new VizEngine(body, { scale: 40, originX: 280, originY: 220 });
                    const ctx = viz.ctx;
                    const cc = [viz.colors.blue, viz.colors.orange, viz.colors.teal, viz.colors.purple, viz.colors.pink];
                    let K=3, pts=[], cents=[], asgn=[], running=false, animId=null, converged=false;
                    const rng=s=>{let v=s;return()=>{v=(v*16807+7)%2147483647;return v/2147483647;};}; let rand=rng(123);

                    function genData() {
                        rand=rng(Date.now()%100000); pts=[]; const cens=[];
                        for(let k=0;k<K;k++) cens.push({x:(rand()-0.5)*8,y:(rand()-0.5)*6});
                        for(let k=0;k<K;k++){const n=15+Math.floor(rand()*10);for(let i=0;i<n;i++) pts.push({x:cens[k].x+(rand()-0.5)*3,y:cens[k].y+(rand()-0.5)*3});}
                        const sh=[...pts].sort(()=>rand()-0.5); cents=[];
                        for(let k=0;k<K;k++) cents.push({x:sh[k].x,y:sh[k].y});
                        asgn=new Array(pts.length).fill(0); converged=false;
                    }
                    function step() {
                        let changed=false;
                        for(let i=0;i<pts.length;i++){let b=0,bd=Infinity;for(let k=0;k<K;k++){const d=(pts[i].x-cents[k].x)**2+(pts[i].y-cents[k].y)**2;if(d<bd){bd=d;b=k;}}if(asgn[i]!==b)changed=true;asgn[i]=b;}
                        for(let k=0;k<K;k++){let sx=0,sy=0,c=0;for(let i=0;i<pts.length;i++)if(asgn[i]===k){sx+=pts[i].x;sy+=pts[i].y;c++;}if(c>0)cents[k]={x:sx/c,y:sy/c};}
                        if(!changed) converged=true;
                    }
                    genData();
                    VizEngine.createSlider(controls,'K',2,5,3,1,v=>{K=Math.round(v);genData();draw();});
                    VizEngine.createButton(controls,'Step',()=>{if(!converged){step();draw();}});
                    const runBtn=VizEngine.createButton(controls,'Run',()=>{
                        if(running){running=false;runBtn.textContent='Run';if(animId)clearInterval(animId);}
                        else{running=true;runBtn.textContent='Pause';animId=setInterval(()=>{if(converged){running=false;runBtn.textContent='Run';clearInterval(animId);return;}step();draw();},500);}
                    });
                    VizEngine.createButton(controls,'Reset',()=>{running=false;runBtn.textContent='Run';if(animId)clearInterval(animId);genData();draw();});

                    function draw() {
                        viz.clear(); viz.drawGrid(); viz.drawAxes();
                        const st=6;
                        for(let px=0;px<viz.width;px+=st)for(let py=0;py<viz.height;py+=st){
                            const [wx,wy]=viz.toMath(px,py);let b=0,bd=Infinity;
                            for(let k=0;k<K;k++){const d=(wx-cents[k].x)**2+(wy-cents[k].y)**2;if(d<bd){bd=d;b=k;}}
                            ctx.fillStyle=cc[b%cc.length]+'10';ctx.fillRect(px,py,st,st);
                        }
                        for(let i=0;i<pts.length;i++) viz.drawPoint(pts[i].x,pts[i].y,cc[asgn[i]%cc.length],null,4);
                        for(let k=0;k<K;k++){
                            const [sx,sy]=viz.toScreen(cents[k].x,cents[k].y);
                            ctx.strokeStyle=viz.colors.white;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx-7,sy-7);ctx.lineTo(sx+7,sy+7);ctx.moveTo(sx+7,sy-7);ctx.lineTo(sx-7,sy+7);ctx.stroke();
                            ctx.fillStyle=cc[k%cc.length];ctx.beginPath();ctx.arc(sx,sy,5,0,Math.PI*2);ctx.fill();
                        }
                        if(converged) viz.screenText('Converged!',viz.width/2,viz.height-14,viz.colors.green,12);
                    }
                    draw();
                    return {stopAnimation(){if(animId)clearInterval(animId);viz.stopAnimation();}};
                }
            }],
            exercises: [
                { question: 'K-means requires choosing \\(K\\) in advance. Describe two methods for selecting \\(K\\) and their tradeoffs.',
                  hint: 'Consider the "elbow method" and the silhouette score.',
                  solution: '<strong>Elbow method</strong>: plot \\(J\\) vs. \\(K\\). The rate of decrease often slows at a "knee" suggesting the natural \\(K\\). Simple but the elbow is often ambiguous. <strong>Silhouette score</strong>: for each point, measure intra-cluster similarity vs. nearest-other-cluster similarity (range \\([-1,1]\\)). Maximize average silhouette over \\(K\\). More principled but expensive for large datasets.' },
                { question: 'Why can K-means converge to a suboptimal solution? Illustrate with points \\(\\{0,1,10,11\\}\\) and \\(K=2\\).',
                  hint: 'Try initializing centroids at \\(5\\) and \\(11\\).',
                  solution: 'Optimal: \\(\\{0,1\\}\\) and \\(\\{10,11\\}\\) with \\(J=1\\). But centroids initialized at \\(5\\) and \\(11\\) assign \\(\\{0,1,10\\}\\) to centroid 1 and \\(\\{11\\}\\) to centroid 2, a much worse local minimum. K-means is greedy coordinate descent and cannot escape local minima. Multiple random restarts mitigate this.' },
                { question: 'Name three applications where unsupervised learning is more natural than supervised learning, explaining why labels are impractical.',
                  hint: 'Think about settings with vast data but expensive or undefined labels.',
                  solution: '(1) <strong>Customer segmentation</strong>: millions of transactions but no predefined "types"; clustering reveals segments for marketing. (2) <strong>Network anomaly detection</strong>: normal traffic is abundant, attacks are rare and diverse; density estimation flags unusual patterns without labeled attack types. (3) <strong>Gene expression analysis</strong>: thousands of genes measured but functional categories are unknown; dimensionality reduction and clustering reveal co-regulated gene modules.' }
            ]
        },
        // ========== SECTION 4: Reinforcement Learning & Beyond ==========
        {
            id: 'sec04-rl-and-beyond',
            title: 'Reinforcement Learning & Beyond',
            content: `
<h2>0.4 Reinforcement Learning & Beyond</h2>
<h3>The RL Framework</h3>
<p>An <em>agent</em> interacts with an <em>environment</em>, takes <em>actions</em>, receives <em>rewards</em>, and learns a <em>policy</em> maximizing cumulative reward.</p>
<div class="env-block definition">
<strong>Definition 0.4.1 (Markov Decision Process).</strong> An MDP is \\((\\mathcal{S}, \\mathcal{A}, P, R, \\gamma)\\): states \\(\\mathcal{S}\\), actions \\(\\mathcal{A}\\), transitions \\(P(s'|s,a)\\), rewards \\(R(s,a)\\), discount \\(\\gamma \\in [0,1)\\). The agent seeks \\(\\pi^*\\) maximizing
\\[
V^\\pi(s) = \\mathbb{E}\\!\\left[\\sum_{t=0}^{\\infty} \\gamma^t R(s_t, a_t) \\;\\middle|\\; s_0=s,\\,\\pi\\right].
\\]
</div>
<div class="env-block remark">
<strong>Exploration vs. Exploitation.</strong> The agent must balance exploiting known-good actions with exploring potentially better ones. Too much exploitation leads to suboptimal local behavior; too much exploration wastes time.
</div>
<div class="viz-placeholder" data-viz="viz-gridworld"></div>
<h3>Semi-Supervised Learning</h3>
<p>A small labeled set \\(\\{(\\mathbf{x}_i,y_i)\\}_{i=1}^l\\) plus a large unlabeled set \\(\\{\\mathbf{x}_j\\}_{j=1}^u\\) with \\(u \\gg l\\). The unlabeled data reveals input structure \\(p(\\mathbf{x})\\) that constrains the decision boundary.</p>
<h3>Self-Supervised Learning</h3>
<p>Create supervisory signals from data itself: mask tokens and predict them (BERT), predict the next token (GPT), or reconstruct masked image regions (MAE). No human labels required.</p>
<div class="env-block remark">
<strong>Foundation Models.</strong> Self-supervised pretraining on massive unlabeled corpora, followed by fine-tuning on small labeled sets, has become the dominant paradigm. Models like GPT-4, BERT, and CLIP are pretrained this way, then adapted to diverse downstream tasks.
</div>
<h3>Comparing the Paradigms</h3>
<table style="width:100%;border-collapse:collapse;margin:16px 0;">
<thead><tr style="border-bottom:2px solid #30363d;">
<th style="text-align:left;padding:6px;">Paradigm</th><th style="text-align:left;padding:6px;">Data</th><th style="text-align:left;padding:6px;">Signal</th><th style="text-align:left;padding:6px;">Example</th></tr></thead>
<tbody>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">Supervised</td><td style="padding:6px;">\\((\\mathbf{x},y)\\) pairs</td><td style="padding:6px;">Human labels</td><td style="padding:6px;">Image classification</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">Unsupervised</td><td style="padding:6px;">\\(\\mathbf{x}\\) only</td><td style="padding:6px;">None (structure)</td><td style="padding:6px;">Clustering, PCA</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">Reinforcement</td><td style="padding:6px;">Trajectories</td><td style="padding:6px;">Rewards</td><td style="padding:6px;">Game playing</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">Semi-supervised</td><td style="padding:6px;">Few labels + many unlabeled</td><td style="padding:6px;">Labels + structure</td><td style="padding:6px;">Medical imaging</td></tr>
<tr><td style="padding:6px;">Self-supervised</td><td style="padding:6px;">\\(\\mathbf{x}\\) only</td><td style="padding:6px;">Pseudo-labels from \\(\\mathbf{x}\\)</td><td style="padding:6px;">GPT, BERT</td></tr>
</tbody></table>
`,
            visualizations: [{
                id: 'viz-gridworld',
                title: 'Gridworld: Value Iteration',
                description: 'A 7x7 grid. Agent (S) seeks goal (G) while avoiding walls. Press "Step" for one value iteration sweep, or "Run" to animate. Heatmap shows \\(V(s)\\); arrows show the greedy policy. "Show Policy" animates the optimal path.',
                setup(body, controls) {
                    const viz = new VizEngine(body, { scale: 1, originX: 0, originY: 0 });
                    const ctx = viz.ctx, W = viz.width, H = viz.height;
                    const R=7, C=7, cW=Math.min(60,(W-80)/C), cH=cW;
                    const oX=(W-C*cW)/2, oY=(H-R*cH)/2;
                    const grid=Array.from({length:R},()=>new Array(C).fill(0));
                    [[1,2],[2,2],[3,2],[3,4],[4,4],[5,4],[1,5],[2,5]].forEach(([r,c])=>{if(r<R&&c<C)grid[r][c]=1;});
                    const gR=1,gC=5; grid[gR][gC]=2;
                    const sR=5,sC=1, gam=0.9;
                    let V=Array.from({length:R},()=>new Array(C).fill(0)), iters=0, running=false, showPol=false;
                    let path=[], pStep=0, runId=null, pathId=null;
                    const acts=[[-1,0],[1,0],[0,-1],[0,1]], arrows=['\u2191','\u2193','\u2190','\u2192'];

                    function vi(){
                        const nV=Array.from({length:R},()=>new Array(C).fill(0));
                        for(let r=0;r<R;r++)for(let c=0;c<C;c++){
                            if(grid[r][c]===1){nV[r][c]=0;continue;}if(grid[r][c]===2){nV[r][c]=1;continue;}
                            let best=-Infinity;for(const[dr,dc]of acts){const nr=r+dr,nc=c+dc;
                            if(nr>=0&&nr<R&&nc>=0&&nc<C&&grid[nr][nc]!==1){best=Math.max(best,(grid[nr][nc]===2?1:-0.04)+gam*V[nr][nc]);}
                            else best=Math.max(best,-0.04+gam*V[r][c]);}nV[r][c]=best;}
                        V=nV;iters++;
                    }
                    function bestAct(r,c){if(grid[r][c]!==0)return-1;let b=-1,bv=-Infinity;for(let a=0;a<4;a++){const[dr,dc]=acts[a],nr=r+dr,nc=c+dc;let v;if(nr>=0&&nr<R&&nc>=0&&nc<C&&grid[nr][nc]!==1)v=(grid[nr][nc]===2?1:-0.04)+gam*V[nr][nc];else v=-0.04+gam*V[r][c];if(v>bv){bv=v;b=a;}}return b;}
                    function computePath(){path=[{r:sR,c:sC}];let r=sR,c=sC;const vis=new Set();for(let i=0;i<50;i++){const k=r*C+c;if(vis.has(k)||grid[r][c]===2)break;vis.add(k);const a=bestAct(r,c);if(a<0)break;const[dr,dc]=acts[a],nr=r+dr,nc=c+dc;if(nr>=0&&nr<R&&nc>=0&&nc<C&&grid[nr][nc]!==1){r=nr;c=nc;}path.push({r,c});}}

                    VizEngine.createButton(controls,'Step',()=>{vi();showPol=false;draw();});
                    const runBtn=VizEngine.createButton(controls,'Run',()=>{
                        if(running){running=false;runBtn.textContent='Run';if(runId)clearInterval(runId);}
                        else{running=true;runBtn.textContent='Pause';runId=setInterval(()=>{if(iters>=50){running=false;runBtn.textContent='Run';clearInterval(runId);return;}vi();draw();},200);}
                    });
                    VizEngine.createButton(controls,'Show Policy',()=>{showPol=true;computePath();pStep=0;if(pathId)clearInterval(pathId);pathId=setInterval(()=>{if(pStep>=path.length-1){clearInterval(pathId);pathId=null;return;}pStep++;draw();},300);draw();});
                    VizEngine.createButton(controls,'Reset',()=>{V=Array.from({length:R},()=>new Array(C).fill(0));iters=0;running=false;showPol=false;runBtn.textContent='Run';if(runId)clearInterval(runId);if(pathId)clearInterval(pathId);draw();});

                    function draw(){
                        ctx.fillStyle=viz.colors.bg;ctx.fillRect(0,0,W,H);
                        let vMin=Infinity,vMax=-Infinity;for(let r=0;r<R;r++)for(let c=0;c<C;c++)if(grid[r][c]===0){vMin=Math.min(vMin,V[r][c]);vMax=Math.max(vMax,V[r][c]);}if(vMax<=vMin){vMin=0;vMax=1;}
                        for(let r=0;r<R;r++)for(let c=0;c<C;c++){
                            const x=oX+c*cW,y=oY+r*cH;
                            if(grid[r][c]===1){ctx.fillStyle='#2a2a4a';}else if(grid[r][c]===2){ctx.fillStyle=viz.colors.green+'55';}
                            else{const t=(V[r][c]-vMin)/(vMax-vMin);ctx.fillStyle=`rgb(${20+t*40|0},${20+t*100|0},${40+t*80|0})`;}
                            ctx.fillRect(x,y,cW,cH);ctx.strokeStyle=viz.colors.grid;ctx.lineWidth=1;ctx.strokeRect(x,y,cW,cH);
                            if(grid[r][c]===0&&iters>0){ctx.fillStyle=viz.colors.white+'99';ctx.font='10px -apple-system,sans-serif';ctx.textAlign='center';ctx.textBaseline='bottom';ctx.fillText(V[r][c].toFixed(2),x+cW/2,y+cH-3);const a=bestAct(r,c);if(a>=0){ctx.fillStyle=viz.colors.white+'66';ctx.font='16px -apple-system,sans-serif';ctx.textBaseline='top';ctx.fillText(arrows[a],x+cW/2,y+3);}}
                            if(grid[r][c]===2){ctx.fillStyle=viz.colors.green;ctx.font='bold 18px -apple-system,sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('G',x+cW/2,y+cH/2);}
                        }
                        if(showPol&&path.length>0){for(let i=0;i<=Math.min(pStep,path.length-1);i++){const p=path[i];ctx.fillStyle=viz.colors.blue+'33';ctx.fillRect(oX+p.c*cW+2,oY+p.r*cH+2,cW-4,cH-4);}const ap=path[Math.min(pStep,path.length-1)];const ax=oX+ap.c*cW+cW/2,ay=oY+ap.r*cH+cH/2;ctx.fillStyle=viz.colors.blue;ctx.beginPath();ctx.arc(ax,ay,cW*0.3,0,Math.PI*2);ctx.fill();ctx.fillStyle=viz.colors.white;ctx.font='bold 13px -apple-system,sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('A',ax,ay);}
                        else{const sx=oX+sC*cW+cW/2,sy=oY+sR*cH+cH/2;ctx.fillStyle=viz.colors.blue+'88';ctx.beginPath();ctx.arc(sx,sy,cW*0.25,0,Math.PI*2);ctx.fill();ctx.fillStyle=viz.colors.white;ctx.font='11px -apple-system,sans-serif';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('S',sx,sy);}
                        ctx.fillStyle=viz.colors.white;ctx.font='12px -apple-system,sans-serif';ctx.textAlign='center';ctx.fillText('Iterations: '+iters,W/2,oY-10);
                    }
                    draw();
                    return {stopAnimation(){if(runId)clearInterval(runId);if(pathId)clearInterval(pathId);viz.stopAnimation();}};
                }
            }],
            exercises: [
                { question: 'In the gridworld MDP, explain the role of \\(\\gamma\\). What happens as \\(\\gamma \\to 0\\) vs. \\(\\gamma \\to 1\\)?',
                  hint: 'Consider how much the agent values distant future rewards.',
                  solution: '\\(\\gamma\\) controls time preference. As \\(\\gamma \\to 0\\), the agent is myopic, caring only about the immediate reward; it may avoid step penalties but never reach the goal. As \\(\\gamma \\to 1\\), the agent is far-sighted, valuing future rewards nearly as much as immediate ones, and will plan long optimal paths to reach the goal even through many steps.' },
                { question: 'Classify each as supervised, unsupervised, RL, semi-supervised, or self-supervised: (a) Training a chatbot with human feedback on response quality. (b) Grouping news articles by topic without predefined categories. (c) Pretraining a language model to predict the next word. (d) Using 50 labeled and 10,000 unlabeled medical images.',
                  hint: 'For (a), human feedback is a reward signal after the model generates a response.',
                  solution: '(a) <strong>RL</strong> (specifically RLHF): the chatbot generates responses (actions), receives feedback (reward). (b) <strong>Unsupervised</strong> (clustering): no labels. (c) <strong>Self-supervised</strong>: "labels" are derived from text itself. (d) <strong>Semi-supervised</strong>: small labeled set + large unlabeled set.' },
                { question: 'Why has self-supervised pretraining become dominant? What advantages does it have over supervised pretraining on ImageNet-scale labeled data?',
                  hint: 'Consider label cost, data scale, and representation generality.',
                  solution: 'Three advantages. (1) <strong>Scale</strong>: unlabeled data (web text, images) is virtually unlimited; labeled datasets require expensive annotation. (2) <strong>Generality</strong>: self-supervised objectives (next-token prediction, masked reconstruction) learn general-purpose representations, not features tied to a specific label taxonomy. (3) <strong>Transfer</strong>: general representations transfer well to diverse downstream tasks via fine-tuning, consistently outperforming training from scratch.' }
            ]
        }
    ]
});
