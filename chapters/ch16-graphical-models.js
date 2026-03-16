// === Chapter 16: Graphical Models ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch16',
    number: 16,
    title: 'Graphical Models',
    subtitle: 'Encoding conditional independence structure through directed and undirected graphs',
    sections: [
        // ========== SECTION 1: Bayesian Networks ==========
        {
            id: 'ch16-sec01',
            title: 'Bayesian Networks',
            content: `
<h2>16.1 Bayesian Networks</h2>

<div class="env-block intuition"><div class="env-title">Intuition -- From Joint Distributions to Graphs</div><div class="env-body">
A joint distribution over \\(d\\) binary variables requires \\(2^d - 1\\) parameters. For \\(d = 30\\), that is over a billion numbers. Bayesian networks exploit <em>conditional independence</em> to factor the joint into a product of small local terms, each involving only a variable and its parents. A graph makes these factorization assumptions transparent.
</div></div>

<h3>Definition and Factorization</h3>

<div class="env-block definition"><div class="env-title">Definition 16.1.1 -- Bayesian Network Factorization</div><div class="env-body">
A distribution \\(p(x_1, \\ldots, x_d)\\) factorizes according to a DAG \\(G\\) if
\\[
p(x_1, \\ldots, x_d) = \\prod_{i=1}^{d} p(x_i \\mid \\mathrm{pa}(x_i)),
\\]
where \\(\\mathrm{pa}(x_i)\\) denotes the parents of node \\(i\\) in \\(G\\).
</div></div>

<p>Each factor \\(p(x_i \\mid \\mathrm{pa}(x_i))\\) is a <em>local</em> conditional distribution. When the maximum in-degree is small, total storage is \\(O(d \\cdot 2^k)\\) rather than \\(O(2^d)\\).</p>

<h3>d-Separation</h3>

<div class="env-block definition"><div class="env-title">Definition 16.1.2 -- d-Separation</div><div class="env-body">
A path between nodes \\(A\\) and \\(B\\) in a DAG is <em>blocked</em> by a set \\(C\\) if there exists a node \\(v\\) on the path such that either:
<ul>
<li>\\(v \\in C\\) and \\(v\\) is a <strong>chain</strong> (\\(\\to v \\to\\)) or <strong>fork</strong> (\\(\\leftarrow v \\to\\)); or</li>
<li>\\(v\\) is a <strong>collider</strong> (\\(\\to v \\leftarrow\\)) and neither \\(v\\) nor any descendant of \\(v\\) is in \\(C\\).</li>
</ul>
\\(A\\) and \\(B\\) are <em>d-separated</em> given \\(C\\) if every path is blocked by \\(C\\).
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 16.1.3 -- Soundness and Completeness</div><div class="env-body">
If \\(p\\) factorizes according to DAG \\(G\\), then every d-separation implies conditional independence. Conversely, if \\(p\\) is <em>faithful</em> to \\(G\\), then every CI in \\(p\\) corresponds to a d-separation.
</div></div>

<h3>Three Canonical Structures</h3>

<table style="width:100%;border-collapse:collapse;margin:1em 0;">
<tr style="border-bottom:1px solid #30363d;"><th style="padding:6px;">Structure</th><th>Unconditional</th><th>Conditioned on middle</th></tr>
<tr style="border-bottom:1px solid #30363d;"><td style="padding:6px;"><strong>Chain</strong>: \\(A \\to B \\to C\\)</td><td>\\(A \\not\\!\\perp\\!\\!\\!\\perp C\\)</td><td>\\(A \\perp\\!\\!\\!\\perp C \\mid B\\)</td></tr>
<tr style="border-bottom:1px solid #30363d;"><td style="padding:6px;"><strong>Fork</strong>: \\(A \\leftarrow B \\to C\\)</td><td>\\(A \\not\\!\\perp\\!\\!\\!\\perp C\\)</td><td>\\(A \\perp\\!\\!\\!\\perp C \\mid B\\)</td></tr>
<tr><td style="padding:6px;"><strong>Collider</strong>: \\(A \\to B \\leftarrow C\\)</td><td>\\(A \\perp\\!\\!\\!\\perp C\\)</td><td>\\(A \\not\\!\\perp\\!\\!\\!\\perp C \\mid B\\)</td></tr>
</table>

<p>The collider is the surprising case: conditioning on the common effect <em>creates</em> dependence ("explaining away").</p>

<div class="env-block example"><div class="env-title">Example 16.1.4 -- Alarm Network</div><div class="env-body">
Burglary \\(B\\) and Earthquake \\(E\\) both cause Alarm \\(A\\), which causes JohnCalls \\(J\\) and MaryCalls \\(M\\). The joint factorizes as \\(p(B)p(E)p(A \\mid B,E)p(J \\mid A)p(M \\mid A)\\). Without observing \\(A\\), \\(B \\perp\\!\\!\\!\\perp E\\). Once \\(A\\) is observed (a collider), \\(B\\) and \\(E\\) become dependent.
</div></div>

<div class="viz-placeholder" data-viz="viz-bayesian-network"></div>
`,
            visualizations: [
                {
                    id: 'viz-bayesian-network',
                    title: 'Interactive Bayesian Network',
                    description: 'Click nodes to toggle observed state (cycle: unobserved, true, false). Watch beliefs update via exact enumeration.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 1, originX: 0, originY: 0 });
                        const ctx = viz.ctx, W = viz.width, H = viz.height;
                        const nodes = [
                            { id:'B', label:'Burglary', x:W*0.25, y:50 },
                            { id:'E', label:'Earthquake', x:W*0.75, y:50 },
                            { id:'A', label:'Alarm', x:W*0.5, y:150 },
                            { id:'J', label:'JohnCalls', x:W*0.25, y:260 },
                            { id:'M', label:'MaryCalls', x:W*0.75, y:260 }
                        ];
                        const edges = [[0,2],[1,2],[2,3],[2,4]];
                        const cptA = {'TT':0.95,'TF':0.94,'FT':0.29,'FF':0.001};
                        const cptJ = {'T':0.90,'F':0.05}, cptM = {'T':0.70,'F':0.01};
                        let observed = [null,null,null,null,null];
                        function infer() {
                            const vals = [false,true]; let marg=[0,0,0,0,0], Z=0;
                            for(let b=0;b<2;b++) for(let e=0;e<2;e++) for(let a=0;a<2;a++)
                            for(let j=0;j<2;j++) for(let m=0;m<2;m++) {
                                const st=[vals[b],vals[e],vals[a],vals[j],vals[m]];
                                let ok=true;
                                for(let i=0;i<5;i++) if(observed[i]!==null&&observed[i]!==st[i]){ok=false;break;}
                                if(!ok) continue;
                                const bk=st[0]?'T':'F', ek=st[1]?'T':'F', ak=st[2]?'T':'F';
                                let p=(st[0]?0.001:0.999)*(st[1]?0.002:0.998);
                                p*=(st[2]?cptA[bk+ek]:1-cptA[bk+ek]);
                                p*=(st[3]?cptJ[ak]:1-cptJ[ak]);
                                p*=(st[4]?cptM[ak]:1-cptM[ak]);
                                Z+=p; for(let i=0;i<5;i++) if(st[i]) marg[i]+=p;
                            }
                            return marg.map(m=>Z>0?m/Z:0);
                        }
                        const nCol=[viz.colors.orange,viz.colors.teal,viz.colors.red,viz.colors.blue,viz.colors.purple];
                        function draw() {
                            viz.clear(); const bel=infer();
                            ctx.strokeStyle='#4a4a7a'; ctx.lineWidth=2;
                            for(const[from,to] of edges){
                                const n1=nodes[from],n2=nodes[to];
                                const dx=n2.x-n1.x,dy=(n2.y-22)-(n1.y+22),ang=Math.atan2(dy,dx);
                                ctx.beginPath();ctx.moveTo(n1.x,n1.y+22);ctx.lineTo(n2.x,n2.y-22);ctx.stroke();
                                ctx.fillStyle='#4a4a7a';ctx.beginPath();ctx.moveTo(n2.x,n2.y-22);
                                ctx.lineTo(n2.x-10*Math.cos(ang-0.5),n2.y-22-10*Math.sin(ang-0.5));
                                ctx.lineTo(n2.x-10*Math.cos(ang+0.5),n2.y-22-10*Math.sin(ang+0.5));
                                ctx.closePath();ctx.fill();
                            }
                            for(let i=0;i<5;i++){
                                const n=nodes[i],isObs=observed[i]!==null,col=nCol[i];
                                ctx.beginPath();ctx.arc(n.x,n.y,22,0,Math.PI*2);
                                ctx.fillStyle=isObs?col:'#1a1a40';ctx.fill();
                                ctx.strokeStyle=isObs?'#fff':col;ctx.lineWidth=2;ctx.stroke();
                                ctx.fillStyle=isObs?'#fff':col;ctx.font='bold 11px -apple-system,sans-serif';
                                ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(n.id,n.x,n.y);
                                ctx.fillStyle=viz.colors.text;ctx.font='10px -apple-system,sans-serif';
                                ctx.fillText(n.label,n.x,n.y+33);
                                const bw=50,bh=6,bx=n.x-bw/2,by=n.y+40;
                                ctx.fillStyle='#1a1a40';ctx.fillRect(bx,by,bw,bh);
                                ctx.fillStyle=col;ctx.fillRect(bx,by,bw*bel[i],bh);
                                ctx.strokeStyle='#30363d';ctx.lineWidth=0.5;ctx.strokeRect(bx,by,bw,bh);
                                ctx.fillStyle=viz.colors.white;ctx.font='9px -apple-system,sans-serif';
                                ctx.fillText((bel[i]*100).toFixed(1)+'%',n.x,by+bh+10);
                                if(isObs){ctx.fillStyle=viz.colors.yellow;ctx.fillText(observed[i]?'TRUE':'FALSE',n.x,n.y+60);}
                            }
                            ctx.fillStyle=viz.colors.text;ctx.font='11px -apple-system,sans-serif';ctx.textAlign='center';
                            ctx.fillText('Click node: unobserved \u2192 true \u2192 false \u2192 unobserved',W/2,H-12);
                        }
                        viz.canvas.addEventListener('click',e=>{
                            const r=viz.canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
                            for(let i=0;i<5;i++){const dx=mx-nodes[i].x,dy=my-nodes[i].y;
                                if(dx*dx+dy*dy<625){observed[i]=observed[i]===null?true:observed[i]===true?false:null;draw();break;}}
                        });
                        VizEngine.createButton(controls,'Reset All',()=>{observed=[null,null,null,null,null];draw();});
                        VizEngine.createButton(controls,'Observe Alarm=T',()=>{observed[2]=true;draw();});
                        draw(); return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'In the DAG \\(A \\to C \\leftarrow B\\), are \\(A\\) and \\(B\\) unconditionally independent? What happens when we condition on \\(C\\)?',
                    hint: 'Identify the collider and apply d-separation.',
                    solution: '\\(C\\) is a collider, so the path \\(A \\to C \\leftarrow B\\) is blocked when \\(C\\) is unobserved: \\(A \\perp\\!\\!\\!\\perp B\\). Conditioning on \\(C\\) unblocks the path, so \\(A \\not\\!\\perp\\!\\!\\!\\perp B \\mid C\\). This is the "explaining away" effect.'
                },
                {
                    question: 'Show that the chain rule \\(p(x_1,\\ldots,x_d) = \\prod_{i=1}^d p(x_i \\mid x_1,\\ldots,x_{i-1})\\) corresponds to the fully-connected DAG. What does a sparse DAG buy us?',
                    hint: 'In the full DAG, \\(\\mathrm{pa}(x_i) = \\{x_1,\\ldots,x_{i-1}\\}\\).',
                    solution: 'The chain rule uses all predecessors as parents, giving a complete DAG. A sparse DAG asserts \\(p(x_i \\mid x_1,\\ldots,x_{i-1}) = p(x_i \\mid \\mathrm{pa}(x_i))\\), reducing parameters from \\(O(2^d)\\) to \\(O(d \\cdot 2^k)\\) where \\(k\\) is the max in-degree.'
                },
                {
                    question: 'Given \\(X_1 \\to X_3 \\leftarrow X_2\\), \\(X_3 \\to X_4\\), determine whether \\(X_1 \\perp\\!\\!\\!\\perp X_2 \\mid X_4\\).',
                    hint: '\\(X_4\\) is a descendant of the collider \\(X_3\\).',
                    solution: 'The path \\(X_1 \\to X_3 \\leftarrow X_2\\) has a collider at \\(X_3\\). Although \\(X_3 \\notin \\{X_4\\}\\), its descendant \\(X_4\\) is in the conditioning set. Conditioning on a descendant of a collider unblocks the path, so \\(X_1 \\not\\!\\perp\\!\\!\\!\\perp X_2 \\mid X_4\\).'
                }
            ]
        },

        // ========== SECTION 2: Markov Random Fields ==========
        {
            id: 'ch16-sec02',
            title: 'Markov Random Fields',
            content: `
<h2>16.2 Markov Random Fields</h2>

<div class="env-block intuition"><div class="env-title">Intuition -- When Influence Has No Direction</div><div class="env-body">
Not every dependency has a causal direction. Neighboring pixels tend to be similar, but neither "causes" the other. Markov random fields model symmetric relationships using undirected graphs, where edges encode direct influence.
</div></div>

<div class="env-block definition"><div class="env-title">Definition 16.2.1 -- Markov Random Field</div><div class="env-body">
An MRF is an undirected graph \\(G = (V, E)\\) with potential functions \\(\\psi_C(x_C)\\) on cliques \\(C\\). The joint distribution is
\\[
p(\\mathbf{x}) = \\frac{1}{Z} \\prod_{C \\in \\mathcal{C}} \\psi_C(x_C), \\qquad Z = \\sum_{\\mathbf{x}} \\prod_{C \\in \\mathcal{C}} \\psi_C(x_C).
\\]
</div></div>

<p>Potentials \\(\\psi_C\\) need not be proper distributions. They assign non-negative "affinity scores" to configurations. Computing \\(Z\\) is \\(\\#P\\)-hard in general.</p>

<h3>Gibbs Distribution and Energy</h3>

<p>Parameterizing in log-space gives the energy form:</p>
\\[
p(\\mathbf{x}) = \\frac{1}{Z} \\exp(-E(\\mathbf{x})), \\quad E(\\mathbf{x}) = \\sum_{C} E_C(x_C).
\\]

<div class="env-block example"><div class="env-title">Example 16.2.2 -- Ising Model</div><div class="env-body">
Binary variables \\(x_i \\in \\{-1, +1\\}\\) on a grid with energy
\\[
E(\\mathbf{x}) = -J \\sum_{(i,j) \\in E} x_i x_j - h \\sum_i x_i.
\\]
When \\(J \\gt 0\\), neighbors prefer to agree (ferromagnetic). The partition function sums over \\(2^n\\) configurations.
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 16.2.3 -- Local Markov Property</div><div class="env-body">
In an MRF, each variable is conditionally independent of all others given its neighbors:
\\[
X_i \\perp\\!\\!\\!\\perp X_{V \\setminus \\{i\\} \\setminus \\mathrm{ne}(i)} \\mid X_{\\mathrm{ne}(i)}.
\\]
</div></div>

<h3>Image Denoising as an MRF</h3>

<p>Given a noisy image \\(\\mathbf{y}\\), model the clean image \\(\\mathbf{x}\\) as an MRF with unary potentials \\(\\psi_i(x_i, y_i)\\) and pairwise potentials \\(\\psi_{ij}(x_i, x_j)\\). The MAP estimate \\(\\hat{\\mathbf{x}} = \\arg\\max p(\\mathbf{x} \\mid \\mathbf{y})\\) is the denoised image.</p>

<div class="viz-placeholder" data-viz="viz-mrf-grid"></div>
`,
            visualizations: [
                {
                    id: 'viz-mrf-grid',
                    title: 'MRF Image Denoising',
                    description: 'Left: noisy binary image. Right: MRF-denoised via iterated conditional modes (ICM). Adjust coupling strength \\(J\\) and noise level.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 1, originX: 0, originY: 0 });
                        const ctx = viz.ctx, W = viz.width, H = viz.height, N = 32;
                        const cw = Math.floor(W*0.42/N);
                        const trueImg = [];
                        for (let r=0;r<N;r++){trueImg[r]=[];for(let c=0;c<N;c++){
                            const dx=c-N/2,dy=r-N/2;trueImg[r][c]=(dx*dx+dy*dy<(N*0.35)**2)?1:-1;}}
                        let noise=0.3, J=1.5, noisy=[], rest=[];
                        function addNoise(){noisy=trueImg.map(r=>r.map(v=>Math.random()<noise?-v:v));rest=noisy.map(r=>[...r]);}
                        function icm(n){for(let it=0;it<n;it++) for(let r=0;r<N;r++) for(let c=0;c<N;c++){
                            let s=0;if(r>0)s+=rest[r-1][c];if(r<N-1)s+=rest[r+1][c];
                            if(c>0)s+=rest[r][c-1];if(c<N-1)s+=rest[r][c+1];
                            rest[r][c]=(-J*s-noisy[r][c]<J*s+noisy[r][c])?1:-1;}}
                        function drawGrid(data,ox,oy,title){
                            ctx.fillStyle=viz.colors.white;ctx.font='bold 12px -apple-system,sans-serif';
                            ctx.textAlign='center';ctx.fillText(title,ox+N*cw/2,oy-8);
                            for(let r=0;r<N;r++) for(let c=0;c<N;c++){
                                ctx.fillStyle=data[r][c]>0?'#58a6ff':'#1a1a40';ctx.fillRect(ox+c*cw,oy+r*cw,cw-1,cw-1);}}
                        function draw(){viz.clear();drawGrid(noisy,15,30,'Noisy');drawGrid(rest,W-N*cw-15,30,'Restored');
                            let ok=0;for(let r=0;r<N;r++)for(let c=0;c<N;c++)if(rest[r][c]===trueImg[r][c])ok++;
                            ctx.fillStyle=viz.colors.teal;ctx.font='12px -apple-system,sans-serif';ctx.textAlign='center';
                            ctx.fillText('Accuracy: '+(100*ok/(N*N)).toFixed(1)+'%',W/2,H-10);}
                        addNoise();icm(5);
                        VizEngine.createSlider(controls,'Noise',0.05,0.5,noise,0.05,v=>{noise=v;addNoise();icm(5);draw();});
                        VizEngine.createSlider(controls,'J',0.1,4.0,J,0.1,v=>{J=v;rest=noisy.map(r=>[...r]);icm(5);draw();});
                        VizEngine.createButton(controls,'Run 10 ICM',()=>{icm(10);draw();});
                        VizEngine.createButton(controls,'New Noise',()=>{addNoise();icm(5);draw();});
                        draw(); return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'State the Hammersley-Clifford theorem and explain why strict positivity is required.',
                    hint: 'Consider what happens if some configurations have zero probability.',
                    solution: 'Hammersley-Clifford: if \\(p(\\mathbf{x}) > 0\\) for all \\(\\mathbf{x}\\) and \\(p\\) satisfies the local Markov property w.r.t. \\(G\\), then \\(p\\) factorizes as \\(\\frac{1}{Z}\\prod_C \\psi_C(x_C)\\). Strict positivity is needed because the proof uses \\(\\log p\\) decomposed via Mobius inversion; if \\(p(\\mathbf{x})=0\\) for some \\(\\mathbf{x}\\), the logarithm is undefined. Without positivity, there exist distributions satisfying the Markov property but not admitting a clique factorization.'
                },
                {
                    question: 'For the Ising model on a \\(2 \\times 2\\) grid with \\(J=1, h=0\\), compute \\(Z\\).',
                    hint: 'There are \\(2^4 = 16\\) configurations. Group by number of agreeing edges.',
                    solution: 'Edges: \\((1,2),(1,3),(2,4),(3,4)\\). All-same (2 configs): \\(E=-4\\), weight \\(e^4\\). Checkerboard (2 configs): \\(E=4\\), weight \\(e^{-4}\\). Remaining 12 configs have \\(E=0\\), weight 1. So \\(Z = 2e^4 + 12 + 2e^{-4} \\approx 121.97\\).'
                },
                {
                    question: 'Why is computing \\(Z\\) intractable, and what are the practical consequences for learning?',
                    hint: 'Connect to the number of configurations and the gradient of the log-likelihood.',
                    solution: 'Computing \\(Z = \\sum_{\\mathbf{x}} \\prod_C \\psi_C\\) sums over exponentially many configurations (\\(2^n\\) for binary), making it \\(\\#P\\)-hard. For learning, the log-likelihood gradient requires \\(\\nabla \\log Z = \\mathbb{E}_p[\\nabla E]\\), an expectation under the model, which also requires evaluating \\(Z\\) or sampling from \\(p\\). This forces the use of approximate methods: MCMC for gradient estimation (contrastive divergence), variational inference, or pseudolikelihood.'
                }
            ]
        },

        // ========== SECTION 3: Inference ==========
        {
            id: 'ch16-sec03',
            title: 'Inference',
            content: `
<h2>16.3 Inference in Graphical Models</h2>

<div class="env-block intuition"><div class="env-title">Intuition -- Message Passing</div><div class="env-body">
Inference means answering queries like "What is \\(p(X_5 \\mid X_1 = 1)\\)?" Naive enumeration is exponential. The graph structure enables <em>message-passing</em> algorithms that push local computations along edges, exploiting the factorization for tractable (or approximate) inference.
</div></div>

<h3>Variable Elimination</h3>

<div class="env-block definition"><div class="env-title">Definition 16.3.1 -- Variable Elimination</div><div class="env-body">
To compute \\(p(X_q)\\), iteratively eliminate variables \\(X_i \\neq X_q\\):
<ol>
<li>Collect all factors involving \\(X_i\\).</li>
<li>Compute \\(\\tau_i(\\cdot) = \\sum_{x_i} \\prod_k f_k(\\cdot, x_i)\\).</li>
<li>Replace collected factors with \\(\\tau_i\\).</li>
</ol>
The <em>elimination ordering</em> determines complexity.
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 16.3.2 -- Complexity</div><div class="env-body">
Variable elimination runs in \\(O(n \\cdot d^{w+1})\\), where \\(w\\) is the <em>treewidth</em> of the induced graph. Finding the optimal ordering is NP-hard.
</div></div>

<h3>Belief Propagation</h3>

<p>On trees, the <em>sum-product algorithm</em> computes all marginals simultaneously via messages along edges.</p>

<div class="env-block definition"><div class="env-title">Definition 16.3.3 -- Sum-Product Messages</div><div class="env-body">
On a factor graph, variable-to-factor: \\(\\mu_{x \\to f}(x) = \\prod_{g \\in \\mathrm{ne}(x) \\setminus f} \\mu_{g \\to x}(x)\\).
Factor-to-variable: \\(\\mu_{f \\to x}(x) = \\sum_{\\mathbf{x}_f \\setminus x} f(\\mathbf{x}_f) \\prod_{y \\in \\mathrm{ne}(f) \\setminus x} \\mu_{y \\to f}(y)\\).
Belief: \\(b(x) \\propto \\prod_{f \\in \\mathrm{ne}(x)} \\mu_{f \\to x}(x)\\).
</div></div>

<div class="env-block remark"><div class="env-title">Remark -- Loopy BP</div><div class="env-body">
On graphs with cycles, iterative belief propagation ("loopy BP") is not exact but often gives good approximations. Messages may not converge, and resulting beliefs are not guaranteed marginals. Nevertheless, loopy BP is widely used (error-correcting codes, computer vision).
</div></div>

<div class="viz-placeholder" data-viz="viz-message-passing"></div>
`,
            visualizations: [
                {
                    id: 'viz-message-passing',
                    title: 'Message Passing on a Factor Graph',
                    description: 'Watch belief propagation on a small factor graph. Circles are variables, squares are factors. Step through message rounds and see beliefs converge.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 1, originX: 0, originY: 0 });
                        const ctx = viz.ctx, W = viz.width, H = viz.height;
                        const vn = [
                            {id:'X1',x:60,y:H*0.5},{id:'X2',x:W*0.35,y:H*0.35},
                            {id:'X3',x:W*0.6,y:H*0.35},{id:'X4',x:W-60,y:H*0.35},
                            {id:'X5',x:W*0.35,y:H*0.75}
                        ];
                        const fn = [
                            {id:'f1',x:W*0.18,y:H*0.42,vs:[0,1],pot:[[0.3,0.7],[0.8,0.2]]},
                            {id:'f2',x:W*0.48,y:H*0.35,vs:[1,2],pot:[[0.9,0.1],[0.2,0.8]]},
                            {id:'f3',x:W*0.78,y:H*0.35,vs:[2,3],pot:[[0.6,0.4],[0.1,0.9]]},
                            {id:'f4',x:W*0.35,y:H*0.56,vs:[1,4],pot:[[0.5,0.5],[0.3,0.7]]}
                        ];
                        let msgs={}, bel=[[.5,.5],[.5,.5],[.5,.5],[.5,.5],[.5,.5]], step=0, log=[];
                        const norm=a=>{const s=a[0]+a[1];return s>0?[a[0]/s,a[1]/s]:[.5,.5];};
                        function init(){msgs={};log=[];step=0;bel=[[.5,.5],[.5,.5],[.5,.5],[.5,.5],[.5,.5]];
                            for(const f of fn) for(const vi of f.vs){msgs['v'+vi+'f'+fn.indexOf(f)]=[1,1];msgs['f'+fn.indexOf(f)+'v'+vi]=[1,1];}}
                        function doStep(){if(step>=8) return;
                            if(step%2===0){for(let vi=0;vi<5;vi++){const af=fn.map((f,fi)=>f.vs.includes(vi)?fi:-1).filter(x=>x>=0);
                                for(const fi of af){let m=[1,1];for(const ofi of af)if(ofi!==fi){m[0]*=msgs['f'+ofi+'v'+vi][0];m[1]*=msgs['f'+ofi+'v'+vi][1];}
                                msgs['v'+vi+'f'+fi]=norm(m);}}log.push('Step '+(step+1)+': var\u2192fac');
                            } else {for(let fi=0;fi<fn.length;fi++){const f=fn[fi];for(const vi of f.vs){
                                const ov=f.vs.filter(v=>v!==vi)[0],mIn=msgs['v'+ov+'f'+fi],vidx=f.vs.indexOf(vi);
                                let m=[0,0];for(let a=0;a<2;a++)for(let b=0;b<2;b++){const args=vidx===0?[a,b]:[b,a];m[a]+=f.pot[args[0]][args[1]]*mIn[b];}
                                msgs['f'+fi+'v'+vi]=norm(m);}}
                                for(let vi=0;vi<5;vi++){let b=[1,1];const af=fn.map((f,fi)=>f.vs.includes(vi)?fi:-1).filter(x=>x>=0);
                                for(const fi of af){b[0]*=msgs['f'+fi+'v'+vi][0];b[1]*=msgs['f'+fi+'v'+vi][1];}bel[vi]=norm(b);}
                                log.push('Step '+(step+1)+': fac\u2192var');}
                            step++;}
                        const vc=[viz.colors.blue,viz.colors.teal,viz.colors.green,viz.colors.purple,viz.colors.pink];
                        function draw(){viz.clear();
                            ctx.strokeStyle='#3a3a6a';ctx.lineWidth=2;
                            for(let fi=0;fi<fn.length;fi++){const f=fn[fi];for(const vi of f.vs){
                                ctx.beginPath();ctx.moveTo(vn[vi].x,vn[vi].y);ctx.lineTo(f.x,f.y);ctx.stroke();}}
                            for(const f of fn){ctx.fillStyle='#2a2a50';ctx.fillRect(f.x-14,f.y-14,28,28);
                                ctx.strokeStyle=viz.colors.orange;ctx.lineWidth=1.5;ctx.strokeRect(f.x-14,f.y-14,28,28);
                                ctx.fillStyle=viz.colors.orange;ctx.font='bold 11px -apple-system,sans-serif';
                                ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(f.id,f.x,f.y);}
                            for(let vi=0;vi<5;vi++){const v=vn[vi],b=bel[vi];
                                ctx.beginPath();ctx.arc(v.x,v.y,18,0,Math.PI*2);ctx.fillStyle='#1a1a40';ctx.fill();
                                ctx.strokeStyle=vc[vi];ctx.lineWidth=2;ctx.stroke();
                                ctx.fillStyle=vc[vi];ctx.font='bold 12px -apple-system,sans-serif';
                                ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(v.id,v.x,v.y);
                                const bw=36,bh=5;ctx.fillStyle='#333';ctx.fillRect(v.x-bw/2,v.y+24,bw,bh);
                                ctx.fillStyle=vc[vi];ctx.fillRect(v.x-bw/2,v.y+24,bw*b[1],bh);
                                ctx.fillStyle=viz.colors.white;ctx.font='9px -apple-system,sans-serif';
                                ctx.fillText('p(1)='+b[1].toFixed(2),v.x,v.y+40);}
                            ctx.fillStyle=viz.colors.text;ctx.font='11px -apple-system,sans-serif';ctx.textAlign='left';
                            const ls=Math.max(0,log.length-3);for(let i=ls;i<log.length;i++) ctx.fillText(log[i],10,H-40+(i-ls)*14);
                            ctx.fillStyle=viz.colors.yellow;ctx.textAlign='center';ctx.fillText('Round '+step+'/8',W/2,H-8);}
                        init();
                        VizEngine.createButton(controls,'Step',()=>{doStep();draw();});
                        VizEngine.createButton(controls,'Run All',()=>{while(step<8)doStep();draw();});
                        VizEngine.createButton(controls,'Reset',()=>{init();draw();});
                        draw(); return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Perform variable elimination on \\(p(x_1,x_2,x_3)=f_1(x_1)f_2(x_1,x_2)f_3(x_2,x_3)\\) to compute \\(p(x_3)\\).',
                    hint: 'Eliminate \\(x_1\\) first, then \\(x_2\\).',
                    solution: 'Step 1: Eliminate \\(x_1\\). Factors: \\(f_1(x_1), f_2(x_1,x_2)\\). New factor: \\(\\tau_1(x_2) = \\sum_{x_1} f_1(x_1)f_2(x_1,x_2)\\). Step 2: Eliminate \\(x_2\\). Factors: \\(\\tau_1(x_2), f_3(x_2,x_3)\\). New factor: \\(\\tau_2(x_3) = \\sum_{x_2} \\tau_1(x_2)f_3(x_2,x_3)\\). Then \\(p(x_3) \\propto \\tau_2(x_3)\\), normalized.'
                },
                {
                    question: 'Prove that belief propagation gives exact marginals on trees.',
                    hint: 'On a tree, removing an edge splits the graph into independent subtrees.',
                    solution: 'On a tree, each message from subtree \\(T_A\\) to \\(T_B\\) along edge \\((a,b)\\) summarizes all information in \\(T_A\\). Since subtrees share no variables (no cycles), these summaries are independent, and each message is computed exactly once from independent evidence. The product of incoming messages at each node gives the exact marginal. On loopy graphs, the same evidence can reach a node through multiple paths, causing double-counting.'
                },
                {
                    question: 'What is the treewidth of a \\(\\sqrt{n} \\times \\sqrt{n}\\) grid? What does this imply for exact inference?',
                    hint: 'Consider eliminating row by row.',
                    solution: 'The treewidth is \\(\\sqrt{n}\\). Eliminating row by row creates factors spanning an entire row (\\(\\sqrt{n}\\) variables). Variable elimination takes \\(O(n \\cdot d^{\\sqrt{n}+1})\\). For a \\(100 \\times 100\\) binary image, this is \\(O(10^4 \\cdot 2^{101})\\), completely intractable. This motivates approximate methods.'
                }
            ]
        },

        // ========== SECTION 4: Hidden Markov Models ==========
        {
            id: 'ch16-sec04',
            title: 'Hidden Markov Models',
            content: `
<h2>16.4 Hidden Markov Models</h2>

<div class="env-block intuition"><div class="env-title">Intuition -- Inferring Hidden States</div><div class="env-body">
A friend texts you the weather each day, but sometimes lies. You know the weather is persistent (sunny follows sunny) and you know how often your friend is correct. An HMM captures exactly this: hidden states (true weather) evolve as a Markov chain, with noisy observations (the texts) emitted at each step.
</div></div>

<div class="env-block definition"><div class="env-title">Definition 16.4.1 -- Hidden Markov Model</div><div class="env-body">
An HMM has hidden states \\(z_t \\in \\{1,\\ldots,K\\}\\), transition matrix \\(A_{jk} = p(z_t=k \\mid z_{t-1}=j)\\), emission distribution \\(p(x_t \\mid z_t)\\), and initial distribution \\(\\pi_k = p(z_1=k)\\). The joint is
\\[
p(\\mathbf{z}, \\mathbf{x}) = \\pi_{z_1} \\prod_{t=2}^T A_{z_{t-1},z_t} \\prod_{t=1}^T p(x_t \\mid z_t).
\\]
</div></div>

<h3>Forward-Backward Algorithm</h3>

<div class="env-block definition"><div class="env-title">Definition 16.4.2 -- Forward and Backward Variables</div><div class="env-body">
\\(\\alpha_t(k) = p(x_1,\\ldots,x_t, z_t=k)\\) satisfies
\\[
\\alpha_t(k) = p(x_t \\mid z_t=k) \\sum_j A_{jk}\\,\\alpha_{t-1}(j).
\\]
\\(\\beta_t(k) = p(x_{t+1},\\ldots,x_T \\mid z_t=k)\\) satisfies
\\[
\\beta_t(j) = \\sum_k A_{jk}\\,p(x_{t+1} \\mid z_{t+1}=k)\\,\\beta_{t+1}(k).
\\]
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 16.4.3 -- Smoothed Posterior</div><div class="env-body">
\\[
p(z_t=k \\mid \\mathbf{x}) = \\frac{\\alpha_t(k)\\,\\beta_t(k)}{\\sum_j \\alpha_t(j)\\,\\beta_t(j)}, \\qquad p(\\mathbf{x}) = \\sum_k \\alpha_T(k).
\\]
</div></div>

<h3>Viterbi Algorithm</h3>

<div class="env-block definition"><div class="env-title">Definition 16.4.4 -- Viterbi Recursion</div><div class="env-body">
\\(\\delta_t(k) = \\max_{z_1,\\ldots,z_{t-1}} p(z_1,\\ldots,z_t=k, x_1,\\ldots,x_t)\\). Recursion:
\\[
\\delta_t(k) = p(x_t \\mid z_t=k) \\max_j A_{jk}\\,\\delta_{t-1}(j), \\quad \\psi_t(k) = \\arg\\max_j A_{jk}\\,\\delta_{t-1}(j).
\\]
Backtrack: \\(\\hat{z}_T = \\arg\\max_k \\delta_T(k)\\), then \\(\\hat{z}_t = \\psi_{t+1}(\\hat{z}_{t+1})\\).
</div></div>

<h3>Baum-Welch (EM for HMMs)</h3>

<p>When parameters are unknown, learn via EM. The E-step uses forward-backward to compute expected sufficient statistics; the M-step updates \\(A\\), \\(\\pi\\), and emission parameters in closed form.</p>

<div class="viz-placeholder" data-viz="viz-hmm-trellis"></div>
`,
            visualizations: [
                {
                    id: 'viz-hmm-trellis',
                    title: 'HMM Trellis with Viterbi Path',
                    description: 'Three hidden states (Sunny, Cloudy, Rainy) with observations (Walk, Shop, Clean). The Viterbi path is highlighted in yellow; node size is proportional to the posterior.',
                    setup(body, controls) {
                        const viz = new VizEngine(body, { scale: 1, originX: 0, originY: 0 });
                        const ctx = viz.ctx, W = viz.width, H = viz.height, K = 3;
                        const sNames = ['Sunny','Cloudy','Rainy'], sCols = ['#f0883e','#8b949e','#58a6ff'];
                        const oNames = ['Walk','Shop','Clean'];
                        let A = [[.6,.3,.1],[.2,.5,.3],[.1,.3,.6]];
                        const B = [[.7,.2,.1],[.3,.4,.3],[.1,.2,.7]], pi = [.5,.3,.2];
                        let obs = [0,1,2,2,1,0,0,2]; const T = obs.length;
                        function viterbi(){
                            const d=[],p=[];d[0]=pi.map((v,k)=>Math.log(v+1e-30)+Math.log(B[k][obs[0]]+1e-30));p[0]=[0,0,0];
                            for(let t=1;t<T;t++){d[t]=[];p[t]=[];for(let k=0;k<K;k++){let bv=-Infinity,bj=0;
                                for(let j=0;j<K;j++){const v=d[t-1][j]+Math.log(A[j][k]+1e-30);if(v>bv){bv=v;bj=j;}}
                                d[t][k]=bv+Math.log(B[k][obs[t]]+1e-30);p[t][k]=bj;}}
                            const path=new Array(T);let bk=0;for(let k=1;k<K;k++)if(d[T-1][k]>d[T-1][bk])bk=k;
                            path[T-1]=bk;for(let t=T-2;t>=0;t--)path[t]=p[t+1][path[t+1]];return path;}
                        function fwdBwd(){
                            const al=[];al[0]=pi.map((v,k)=>v*B[k][obs[0]]);
                            for(let t=1;t<T;t++){al[t]=[];for(let k=0;k<K;k++){let s=0;for(let j=0;j<K;j++)s+=A[j][k]*al[t-1][j];
                                al[t][k]=B[k][obs[t]]*s;}const sc=al[t].reduce((a,b)=>a+b,0);if(sc>0)al[t]=al[t].map(v=>v/sc);}
                            const be=[];be[T-1]=[1,1,1];
                            for(let t=T-2;t>=0;t--){be[t]=[];for(let j=0;j<K;j++){let s=0;for(let k=0;k<K;k++)s+=A[j][k]*B[k][obs[t+1]]*be[t+1][k];
                                be[t][j]=s;}const sc=be[t].reduce((a,b)=>a+b,0);if(sc>0)be[t]=be[t].map(v=>v/sc);}
                            const gam=[];for(let t=0;t<T;t++){const raw=al[t].map((a,k)=>a*be[t][k]);
                                const s=raw.reduce((a,b)=>a+b,0);gam[t]=raw.map(v=>s>0?v/s:1/K);}return gam;}
                        function draw(){
                            viz.clear();const path=viterbi(),gam=fwdBwd();
                            const mg=55,topY=50,botY=H-70,sw=(W-2*mg)/(T-1),sg=(botY-topY)/(K-1);
                            ctx.fillStyle=viz.colors.white;ctx.font='bold 12px -apple-system,sans-serif';
                            ctx.textAlign='center';ctx.fillText('HMM Trellis',W/2,16);
                            ctx.textAlign='right';ctx.font='11px -apple-system,sans-serif';
                            for(let k=0;k<K;k++){ctx.fillStyle=sCols[k];ctx.fillText(sNames[k],mg-10,topY+k*sg+4);}
                            ctx.globalAlpha=0.15;for(let t=0;t<T-1;t++)for(let j=0;j<K;j++)for(let k=0;k<K;k++){
                                if(A[j][k]<0.05)continue;ctx.strokeStyle=sCols[j];ctx.lineWidth=A[j][k]*3;
                                ctx.beginPath();ctx.moveTo(mg+t*sw,topY+j*sg);ctx.lineTo(mg+(t+1)*sw,topY+k*sg);ctx.stroke();}
                            ctx.globalAlpha=1;ctx.strokeStyle=viz.colors.yellow;ctx.lineWidth=3;
                            for(let t=0;t<T-1;t++){ctx.beginPath();ctx.moveTo(mg+t*sw,topY+path[t]*sg);
                                ctx.lineTo(mg+(t+1)*sw,topY+path[t+1]*sg);ctx.stroke();}
                            for(let t=0;t<T;t++)for(let k=0;k<K;k++){const x=mg+t*sw,y=topY+k*sg,r=6+gam[t][k]*10;
                                ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=path[t]===k?sCols[k]:sCols[k]+'44';ctx.fill();
                                if(path[t]===k){ctx.strokeStyle=viz.colors.yellow;ctx.lineWidth=2;ctx.stroke();}}
                            ctx.font='10px -apple-system,sans-serif';ctx.textAlign='center';
                            for(let t=0;t<T;t++){ctx.fillStyle=viz.colors.white;ctx.fillText('t='+(t+1),mg+t*sw,botY+20);
                                ctx.fillStyle=viz.colors.teal;ctx.fillText(oNames[obs[t]],mg+t*sw,botY+34);}
                            ctx.fillStyle=viz.colors.yellow;ctx.font='10px -apple-system,sans-serif';ctx.textAlign='left';
                            ctx.fillText('\u2014 Viterbi path',10,H-10);ctx.fillStyle=viz.colors.text;
                            ctx.fillText('Node size \u221D posterior',W*0.35,H-10);}
                        let selfP=0.5;
                        VizEngine.createSlider(controls,'Self-trans',0.1,0.9,selfP,0.05,v=>{selfP=v;
                            for(let j=0;j<K;j++){A[j][j]=selfP;const r=(1-selfP)/2;for(let k=0;k<K;k++)if(k!==j)A[j][k]=r;}draw();});
                        VizEngine.createButton(controls,'Random Obs',()=>{obs=Array.from({length:T},()=>Math.floor(Math.random()*3));draw();});
                        draw(); return viz;
                    }
                }
            ],
            exercises: [
                {
                    question: 'Show that the forward algorithm computes \\(p(\\mathbf{x})\\) in \\(O(TK^2)\\) versus \\(O(K^T)\\) for naive enumeration.',
                    hint: 'Count operations in the forward recursion.',
                    solution: 'Each \\(\\alpha_t(k)\\) sums over \\(K\\) previous states, costing \\(O(K)\\). For \\(K\\) states at \\(T\\) steps: \\(O(TK^2)\\). Naive enumeration sums over \\(K^T\\) sequences, each costing \\(O(T)\\), giving \\(O(T \\cdot K^T)\\). For \\(K=3, T=100\\): \\(\\sim 900\\) operations vs \\(3^{100} \\approx 5 \\times 10^{47}\\).'
                },
                {
                    question: 'Explain the difference between the Viterbi path and the pointwise MAP sequence. Give an example where they differ.',
                    hint: 'Pointwise MAP maximizes \\(p(z_t \\mid \\mathbf{x})\\) independently at each \\(t\\).',
                    solution: 'Pointwise MAP selects \\(\\hat{z}_t = \\arg\\max_k p(z_t=k \\mid \\mathbf{x})\\) independently, while Viterbi maximizes the joint \\(p(\\mathbf{z} \\mid \\mathbf{x})\\). They differ when pointwise MAP produces impossible transitions. If \\(A_{12}=0\\), pointwise MAP might pick state 1 at \\(t\\) and state 2 at \\(t+1\\), but Viterbi never would since \\(p(z_t=1,z_{t+1}=2)=0\\).'
                },
                {
                    question: 'Derive the Baum-Welch update \\(\\hat{A}_{jk} = \\frac{\\sum_{t} \\xi_t(j,k)}{\\sum_{t} \\gamma_t(j)}\\).',
                    hint: 'Maximize the expected complete-data log-likelihood subject to \\(\\sum_k A_{jk}=1\\).',
                    solution: 'The expected log-likelihood contains \\(Q_A = \\sum_t \\sum_{j,k} \\xi_t(j,k) \\log A_{jk}\\). With Lagrange multiplier for \\(\\sum_k A_{jk}=1\\): \\(\\partial Q_A/\\partial A_{jk} = \\sum_t \\xi_t(j,k)/A_{jk} - \\lambda_j = 0\\), so \\(A_{jk} \\propto \\sum_t \\xi_t(j,k)\\). The constraint gives \\(\\lambda_j = \\sum_t \\gamma_t(j)\\), yielding the update. This is the expected number of \\(j \\to k\\) transitions divided by the expected number of visits to state \\(j\\).'
                }
            ]
        }
    ]
});
