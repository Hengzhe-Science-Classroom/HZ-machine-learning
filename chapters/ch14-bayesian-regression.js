// === Chapter 14: Bayesian Linear Regression ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch14',
    number: 14,
    title: 'Bayesian Linear Regression',
    subtitle: 'Treating parameters as random variables to quantify uncertainty in predictions',
    sections: [
        // ===================== Section 1: Bayesian Approach =====================
        {
            id: 'ch14-sec01',
            title: 'Bayesian Approach',
            content: `<h2>14.1 The Bayesian Approach to Regression</h2>
<div class="env-block intuition"><div class="env-title">Intuition</div><div class="env-body">
<p>In ordinary least squares we find a single "best" weight vector \\(\\mathbf{w}^*\\). But with limited data, many weight vectors fit the observations nearly as well. The Bayesian approach maintains a full probability distribution over \\(\\mathbf{w}\\), representing our uncertainty. Predictions integrate over all plausible weights, yielding calibrated uncertainty bands.</p>
</div></div>

<h3>Probabilistic Model</h3>
<p>Given input \\(\\mathbf{x}\\), the observation model is \\(y = \\mathbf{w}^\\top \\boldsymbol{\\phi}(\\mathbf{x}) + \\epsilon\\), \\(\\epsilon \\sim \\mathcal{N}(0, \\sigma^2)\\), where \\(\\boldsymbol{\\phi}(\\mathbf{x}) \\in \\mathbb{R}^M\\) is a basis-function vector. With \\(N\\) observations, define the design matrix \\(\\boldsymbol{\\Phi} \\in \\mathbb{R}^{N \\times M}\\) and \\(\\mathbf{y} = (y_1,\\ldots,y_N)^\\top\\).</p>

<div class="env-block definition"><div class="env-title">Definition 14.1.1 -- Bayesian Linear Regression</div><div class="env-body">
<ul>
<li><strong>Prior:</strong> \\(p(\\mathbf{w}) = \\mathcal{N}(\\mathbf{m}_0, \\mathbf{S}_0)\\)</li>
<li><strong>Likelihood:</strong> \\(p(\\mathbf{y} \\mid \\mathbf{w}) = \\mathcal{N}(\\mathbf{y};\\, \\boldsymbol{\\Phi}\\mathbf{w},\\, \\sigma^2 \\mathbf{I})\\)</li>
<li><strong>Posterior:</strong> \\(p(\\mathbf{w} \\mid \\mathbf{y}) \\propto p(\\mathbf{y} \\mid \\mathbf{w})\\, p(\\mathbf{w})\\)</li>
</ul></div></div>

<div class="env-block theorem"><div class="env-title">Theorem 14.1.2 -- Posterior Distribution</div><div class="env-body">
<p>The posterior is Gaussian: \\(p(\\mathbf{w} \\mid \\mathbf{y}) = \\mathcal{N}(\\mathbf{m}_N, \\mathbf{S}_N)\\), where</p>
\\[\\mathbf{S}_N^{-1} = \\mathbf{S}_0^{-1} + \\sigma^{-2} \\boldsymbol{\\Phi}^\\top \\boldsymbol{\\Phi}, \\qquad \\mathbf{m}_N = \\mathbf{S}_N(\\mathbf{S}_0^{-1}\\mathbf{m}_0 + \\sigma^{-2}\\boldsymbol{\\Phi}^\\top \\mathbf{y}).\\]
</div></div>

<div class="env-block proof"><div class="env-title">Proof</div><div class="env-body">
<p>The log unnormalized posterior is \\(-\\frac{1}{2\\sigma^2}\\|\\mathbf{y}-\\boldsymbol{\\Phi}\\mathbf{w}\\|^2 - \\frac{1}{2}(\\mathbf{w}-\\mathbf{m}_0)^\\top\\mathbf{S}_0^{-1}(\\mathbf{w}-\\mathbf{m}_0) + C\\). Collecting quadratic terms in \\(\\mathbf{w}\\) gives precision \\(\\mathbf{S}_N^{-1} = \\mathbf{S}_0^{-1} + \\sigma^{-2}\\boldsymbol{\\Phi}^\\top\\boldsymbol{\\Phi}\\) and linear term \\(\\mathbf{S}_N^{-1}\\mathbf{m}_N = \\mathbf{S}_0^{-1}\\mathbf{m}_0 + \\sigma^{-2}\\boldsymbol{\\Phi}^\\top\\mathbf{y}\\).</p>
<div class="qed">&#8718;</div></div></div>

<div class="env-block remark"><div class="env-title">Remark -- Connection to Ridge Regression</div><div class="env-body">
<p>With \\(\\mathbf{m}_0=\\mathbf{0}\\) and \\(\\mathbf{S}_0=\\alpha^{-1}\\mathbf{I}\\), the posterior mean equals the ridge solution with \\(\\lambda=\\alpha\\sigma^2\\). Regularization arises from the prior.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-bayesian-prior-posterior"></div>`,
            visualizations: [{
                id: 'viz-bayesian-prior-posterior',
                title: 'Prior and Posterior Weight Distributions',
                description: 'Click the canvas to generate data from the true model (green dot). The dashed ellipses show the prior; solid ellipses show the posterior contracting toward the true weights.',
                setup(body, controls) {
                    const viz = new VizEngine(body, { width: 680, height: 360, scale: 50, originX: 340, originY: 180 });
                    const ctx = viz.ctx;
                    const trueW = [0.5, 1.2], sigma2 = 0.3;
                    let alpha = 1.0, pts = [];
                    function rng() { let u=0,v=0; while(!u) u=Math.random(); while(!v) v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
                    function eigenD(S) {
                        const a=S[0],b=S[1],d=S[3], tr=a+d, det=a*d-b*b;
                        const disc=Math.max(tr*tr-4*det,0), s=Math.sqrt(disc);
                        const l1=Math.max((tr+s)/2,1e-10), l2=Math.max((tr-s)/2,1e-10);
                        let v1,v2;
                        if(Math.abs(b)>1e-10){v1=VizEngine.normalize([l1-d,b]);v2=VizEngine.normalize([l2-d,b]);}
                        else{v1=[1,0];v2=[0,1];}
                        return {l1,l2,v1,v2};
                    }
                    function drawEll(mx,my,Sf,color,dash) {
                        const e=eigenD(Sf);
                        for(let k=1;k<=3;k++){
                            ctx.strokeStyle=color; ctx.lineWidth=k===1?2:1;
                            if(dash)ctx.setLineDash([5,4]);
                            ctx.beginPath();
                            const r=Math.sqrt(2*k);
                            for(let i=0;i<=64;i++){
                                const t=i/64*2*Math.PI;
                                const wx=mx+e.v1[0]*r*Math.sqrt(e.l1)*Math.cos(t)+e.v2[0]*r*Math.sqrt(e.l2)*Math.sin(t);
                                const wy=my+e.v1[1]*r*Math.sqrt(e.l1)*Math.cos(t)+e.v2[1]*r*Math.sqrt(e.l2)*Math.sin(t);
                                const [sx,sy]=viz.toScreen(wx,wy);
                                i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);
                            }
                            ctx.stroke(); ctx.setLineDash([]);
                        }
                    }
                    function posterior() {
                        let a00=alpha,a01=0,a10=0,a11=alpha, r0=0,r1=0;
                        for(const p of pts){
                            const p0=1,p1=p.x;
                            a00+=p0*p0/sigma2; a01+=p0*p1/sigma2; a10+=p1*p0/sigma2; a11+=p1*p1/sigma2;
                            r0+=p0*p.y/sigma2; r1+=p1*p.y/sigma2;
                        }
                        const det=a00*a11-a01*a10;
                        const s00=a11/det,s01=-a01/det,s10=-a10/det,s11=a00/det;
                        return {mN:[s00*r0+s01*r1,s10*r0+s11*r1],SN:[s00,s01,s10,s11]};
                    }
                    function draw() {
                        viz.clear(); viz.drawGrid(1); viz.drawAxes();
                        viz.screenText('w\u2080',viz.width-20,viz.originY-10,viz.colors.text,11);
                        viz.screenText('w\u2081',viz.originX+10,14,viz.colors.text,11);
                        drawEll(0,0,[1/alpha,0,0,1/alpha],viz.colors.purple+'88',true);
                        viz.screenText('Prior',60,20,viz.colors.purple,12,'left');
                        if(pts.length>0){
                            const{mN,SN}=posterior();
                            drawEll(mN[0],mN[1],SN,viz.colors.blue,false);
                            viz.drawPoint(mN[0],mN[1],viz.colors.blue,'MAP',4);
                            viz.screenText('Posterior',60,38,viz.colors.blue,12,'left');
                        }
                        viz.drawPoint(trueW[0],trueW[1],viz.colors.green,'True w',5);
                        viz.screenText('N = '+pts.length,viz.width-60,20,viz.colors.white,13);
                    }
                    viz.canvas.addEventListener('click',function(e){
                        const rect=viz.canvas.getBoundingClientRect();
                        const x=(e.clientX-rect.left-viz.originX)/viz.scale;
                        pts.push({x, y:trueW[0]+trueW[1]*x+rng()*Math.sqrt(sigma2)});
                        draw();
                    });
                    VizEngine.createSlider(controls,'\u03b1',0.1,5,1,0.1,v=>{alpha=v;draw();});
                    VizEngine.createButton(controls,'Reset',()=>{pts=[];draw();});
                    VizEngine.createButton(controls,'Add 5',()=>{for(let i=0;i<5;i++){const x=(Math.random()-0.5)*6;pts.push({x,y:trueW[0]+trueW[1]*x+rng()*Math.sqrt(sigma2)});}draw();});
                    draw(); return viz;
                }
            }],
            exercises: [
                { question:'Show that the posterior precision is the sum of prior precision and data precision: \\(\\mathbf{S}_N^{-1}=\\mathbf{S}_0^{-1}+\\beta\\boldsymbol{\\Phi}^\\top\\boldsymbol{\\Phi}\\) where \\(\\beta=1/\\sigma^2\\).',
                  hint:'Expand the log-posterior and collect terms quadratic in \\(\\mathbf{w}\\).',
                  solution:'The log-posterior is \\(-\\frac{\\beta}{2}\\|\\mathbf{y}-\\boldsymbol{\\Phi}\\mathbf{w}\\|^2 -\\frac{1}{2}(\\mathbf{w}-\\mathbf{m}_0)^\\top\\mathbf{S}_0^{-1}(\\mathbf{w}-\\mathbf{m}_0)+C\\). The coefficient of \\(-\\frac{1}{2}\\mathbf{w}^\\top(\\cdot)\\mathbf{w}\\) is \\(\\beta\\boldsymbol{\\Phi}^\\top\\boldsymbol{\\Phi}+\\mathbf{S}_0^{-1}=\\mathbf{S}_N^{-1}\\).' },
                { question:'With zero-mean isotropic prior \\(\\mathcal{N}(\\mathbf{0},\\alpha^{-1}\\mathbf{I})\\), show that \\(\\mathbf{m}_N=(\\boldsymbol{\\Phi}^\\top\\boldsymbol{\\Phi}+\\alpha\\sigma^2\\mathbf{I})^{-1}\\boldsymbol{\\Phi}^\\top\\mathbf{y}\\).',
                  hint:'Set \\(\\mathbf{m}_0=\\mathbf{0}\\), \\(\\mathbf{S}_0^{-1}=\\alpha\\mathbf{I}\\), and factor out \\(\\sigma^{-2}\\).',
                  solution:'\\(\\mathbf{m}_N=(\\alpha\\mathbf{I}+\\sigma^{-2}\\boldsymbol{\\Phi}^\\top\\boldsymbol{\\Phi})^{-1}\\sigma^{-2}\\boldsymbol{\\Phi}^\\top\\mathbf{y}\\). Multiplying by \\(\\sigma^2/\\sigma^2\\): \\(\\mathbf{m}_N=(\\alpha\\sigma^2\\mathbf{I}+\\boldsymbol{\\Phi}^\\top\\boldsymbol{\\Phi})^{-1}\\boldsymbol{\\Phi}^\\top\\mathbf{y}\\), the ridge regression solution with \\(\\lambda=\\alpha\\sigma^2\\).' },
                { question:'In the limits \\(\\alpha\\to 0\\) and \\(\\alpha\\to\\infty\\), what does the posterior mean converge to?',
                  hint:'Consider the regularization interpretation.',
                  solution:'As \\(\\alpha\\to 0\\): \\(\\mathbf{m}_N\\to(\\boldsymbol{\\Phi}^\\top\\boldsymbol{\\Phi})^{-1}\\boldsymbol{\\Phi}^\\top\\mathbf{y}\\), the OLS solution. As \\(\\alpha\\to\\infty\\): \\(\\mathbf{m}_N\\to\\mathbf{0}\\), the prior mean. The data is ignored.' }
            ]
        },

        // ===================== Section 2: Sequential Bayesian Update =====================
        {
            id: 'ch14-sec02',
            title: 'Sequential Bayesian Update',
            content: `<h2>14.2 Sequential Bayesian Update</h2>
<div class="env-block intuition"><div class="env-title">Intuition</div><div class="env-body">
<p>Bayesian inference naturally supports online learning. After observing one data point, the posterior becomes the prior for the next. The result is identical whether we process all data at once or one point at a time.</p>
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 14.2.1 -- Sequential Update</div><div class="env-body">
<p>Given posterior \\(\\mathcal{N}(\\mathbf{m}_n,\\mathbf{S}_n)\\) and new observation \\((\\mathbf{x}_{n+1},y_{n+1})\\), the updated posterior is \\(\\mathcal{N}(\\mathbf{m}_{n+1},\\mathbf{S}_{n+1})\\):</p>
\\[\\mathbf{S}_{n+1}^{-1}=\\mathbf{S}_n^{-1}+\\sigma^{-2}\\boldsymbol{\\phi}_{n+1}\\boldsymbol{\\phi}_{n+1}^\\top,\\qquad \\mathbf{m}_{n+1}=\\mathbf{S}_{n+1}(\\mathbf{S}_n^{-1}\\mathbf{m}_n+\\sigma^{-2}y_{n+1}\\boldsymbol{\\phi}_{n+1}).\\]
</div></div>

<div class="env-block proof"><div class="env-title">Proof</div><div class="env-body">
<p>Apply Theorem 14.1.2 with "prior" \\(\\mathcal{N}(\\mathbf{m}_n,\\mathbf{S}_n)\\) and "design matrix" being the single row \\(\\boldsymbol{\\phi}_{n+1}^\\top\\). Then \\(\\boldsymbol{\\Phi}^\\top\\boldsymbol{\\Phi}=\\boldsymbol{\\phi}_{n+1}\\boldsymbol{\\phi}_{n+1}^\\top\\) and \\(\\boldsymbol{\\Phi}^\\top\\mathbf{y}=y_{n+1}\\boldsymbol{\\phi}_{n+1}\\).</p>
<div class="qed">&#8718;</div></div></div>

<h3>Geometric Interpretation</h3>
<p>Each observation constrains \\(\\mathbf{w}\\) near the hyperplane \\(\\mathbf{w}^\\top\\boldsymbol{\\phi}_{n+1}\\approx y_{n+1}\\). The posterior contracts most in the direction of \\(\\boldsymbol{\\phi}_{n+1}\\). Diverse input locations contract the posterior in complementary directions, efficiently pinning down all weights.</p>

<div class="env-block definition"><div class="env-title">Definition 14.2.2 -- Likelihood Line</div><div class="env-body">
<p>For \\(\\boldsymbol{\\phi}(x)=(1,x)^\\top\\), each observation \\((x_n,y_n)\\) defines a <strong>likelihood line</strong> \\(w_0+w_1 x_n=y_n\\) in weight space. The Gaussian likelihood concentrates probability near this line.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-sequential-update"></div>`,
            visualizations: [{
                id: 'viz-sequential-update',
                title: 'Sequential Bayesian Update',
                description: 'Click the right panel to add data. Left: posterior over \\((w_0,w_1)\\) contracts with each point. Right: predictive mean and uncertainty.',
                setup(body, controls) {
                    const viz = new VizEngine(body, {width:680,height:360,scale:1,originX:0,originY:0});
                    const ctx = viz.ctx;
                    const trueW=[0.3,0.8], sigma2=0.2, alpha=0.5;
                    let pts=[];
                    const wOx=165,wOy=180,wSc=55, dOx=510,dOy=180,dSc=48;
                    function rng(){let u=0,v=0;while(!u)u=Math.random();while(!v)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
                    function eigenD(S){const a=S[0],b=S[1],d=S[3],tr=a+d,dt=a*d-b*b,disc=Math.max(tr*tr-4*dt,0),s=Math.sqrt(disc);const l1=Math.max((tr+s)/2,1e-10),l2=Math.max((tr-s)/2,1e-10);let v1,v2;if(Math.abs(b)>1e-10){v1=VizEngine.normalize([l1-d,b]);v2=VizEngine.normalize([l2-d,b]);}else{v1=[1,0];v2=[0,1];}return{l1,l2,v1,v2};}
                    function posterior(){
                        let a00=alpha,a01=0,a11=alpha,r0=0,r1=0;
                        for(const p of pts){a00+=1/sigma2;a01+=p.x/sigma2;a11+=p.x*p.x/sigma2;r0+=p.y/sigma2;r1+=p.x*p.y/sigma2;}
                        const det=a00*a11-a01*a01,s00=a11/det,s01=-a01/det,s11=a00/det;
                        return{mN:[s00*r0+s01*r1,s01*r0+s11*r1],SN:[s00,s01,s01,s11]};
                    }
                    function drawEll(mx,my,Sf,color,ox,oy,sc){
                        const e=eigenD(Sf);
                        for(let k=1;k<=3;k++){ctx.strokeStyle=color;ctx.lineWidth=k===1?2:1;ctx.beginPath();
                        const r=Math.sqrt(2*k);for(let i=0;i<=60;i++){const t=i/60*2*Math.PI;
                        const wx=mx+e.v1[0]*r*Math.sqrt(e.l1)*Math.cos(t)+e.v2[0]*r*Math.sqrt(e.l2)*Math.sin(t);
                        const wy=my+e.v1[1]*r*Math.sqrt(e.l1)*Math.cos(t)+e.v2[1]*r*Math.sqrt(e.l2)*Math.sin(t);
                        const sx=ox+wx*sc,sy=oy-wy*sc;i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}ctx.stroke();}
                    }
                    function draw(){
                        ctx.fillStyle=viz.colors.bg;ctx.fillRect(0,0,680,360);
                        ctx.strokeStyle=viz.colors.grid;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(340,0);ctx.lineTo(340,360);ctx.stroke();
                        // Left: weight space
                        ctx.strokeStyle=viz.colors.axis;ctx.lineWidth=1;
                        ctx.beginPath();ctx.moveTo(0,wOy);ctx.lineTo(330,wOy);ctx.stroke();
                        ctx.beginPath();ctx.moveTo(wOx,0);ctx.lineTo(wOx,360);ctx.stroke();
                        ctx.fillStyle=viz.colors.white;ctx.font='12px -apple-system,sans-serif';ctx.textAlign='left';ctx.fillText('Weight Space',10,18);
                        ctx.setLineDash([4,3]);drawEll(0,0,[1/alpha,0,0,1/alpha],viz.colors.purple+'66',wOx,wOy,wSc);ctx.setLineDash([]);
                        if(pts.length>0){const{mN,SN}=posterior();drawEll(mN[0],mN[1],SN,viz.colors.blue,wOx,wOy,wSc);
                        ctx.fillStyle=viz.colors.blue;ctx.beginPath();ctx.arc(wOx+mN[0]*wSc,wOy-mN[1]*wSc,4,0,2*Math.PI);ctx.fill();}
                        ctx.fillStyle=viz.colors.green;ctx.beginPath();ctx.arc(wOx+trueW[0]*wSc,wOy-trueW[1]*wSc,5,0,2*Math.PI);ctx.fill();
                        // Right: data space
                        ctx.strokeStyle=viz.colors.axis;ctx.lineWidth=1;
                        ctx.beginPath();ctx.moveTo(350,dOy);ctx.lineTo(680,dOy);ctx.stroke();
                        ctx.beginPath();ctx.moveTo(dOx,0);ctx.lineTo(dOx,360);ctx.stroke();
                        ctx.fillStyle=viz.colors.white;ctx.font='12px -apple-system,sans-serif';ctx.textAlign='left';ctx.fillText('Data (click to add)',350,18);
                        // True line
                        ctx.strokeStyle=viz.colors.green+'55';ctx.lineWidth=1;ctx.setLineDash([3,3]);
                        ctx.beginPath();ctx.moveTo(dOx-3*dSc,dOy-(trueW[0]-3*trueW[1])*dSc);ctx.lineTo(dOx+3*dSc,dOy-(trueW[0]+3*trueW[1])*dSc);ctx.stroke();ctx.setLineDash([]);
                        if(pts.length>0){
                            const{mN,SN}=posterior();const nX=60;
                            // Uncertainty band
                            ctx.fillStyle=viz.colors.blue+'22';ctx.beginPath();
                            for(let i=0;i<=nX;i++){const x=-3+6*i/nX,phi=[1,x],mu=mN[0]+mN[1]*x,v=sigma2+phi[0]*(SN[0]*phi[0]+SN[1]*phi[1])+phi[1]*(SN[2]*phi[0]+SN[3]*phi[1]);
                            const sx=dOx+x*dSc,sy=dOy-(mu+2*Math.sqrt(v))*dSc;i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}
                            for(let i=nX;i>=0;i--){const x=-3+6*i/nX,phi=[1,x],mu=mN[0]+mN[1]*x,v=sigma2+phi[0]*(SN[0]*phi[0]+SN[1]*phi[1])+phi[1]*(SN[2]*phi[0]+SN[3]*phi[1]);
                            ctx.lineTo(dOx+x*dSc,dOy-(mu-2*Math.sqrt(v))*dSc);}ctx.fill();
                            ctx.strokeStyle=viz.colors.blue;ctx.lineWidth=2;ctx.beginPath();
                            for(let i=0;i<=nX;i++){const x=-3+6*i/nX,sx=dOx+x*dSc,sy=dOy-(mN[0]+mN[1]*x)*dSc;i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}ctx.stroke();
                        }
                        for(const p of pts){ctx.fillStyle=viz.colors.orange;ctx.beginPath();ctx.arc(dOx+p.x*dSc,dOy-p.y*dSc,4,0,2*Math.PI);ctx.fill();}
                        ctx.fillStyle=viz.colors.text;ctx.font='11px -apple-system,sans-serif';ctx.textAlign='right';ctx.fillText('N='+pts.length,670,355);
                    }
                    viz.canvas.addEventListener('click',function(e){const rect=viz.canvas.getBoundingClientRect();const cx=e.clientX-rect.left;
                        if(cx>340){const x=(cx-dOx)/dSc;pts.push({x,y:trueW[0]+trueW[1]*x+rng()*Math.sqrt(sigma2)});draw();}});
                    VizEngine.createButton(controls,'Reset',()=>{pts=[];draw();});
                    VizEngine.createButton(controls,'Add 10',()=>{for(let i=0;i<10;i++){const x=(Math.random()-0.5)*5;pts.push({x,y:trueW[0]+trueW[1]*x+rng()*Math.sqrt(sigma2)});}draw();});
                    draw(); return viz;
                }
            }],
            exercises: [
                { question:'Show that processing data sequentially yields the same posterior as batch processing.',
                  hint:'Write \\(\\mathbf{S}_n^{-1}=\\mathbf{S}_0^{-1}+\\sigma^{-2}\\sum_{i=1}^n\\boldsymbol{\\phi}_i\\boldsymbol{\\phi}_i^\\top\\) and compare.',
                  solution:'After \\(n\\) sequential updates: \\(\\mathbf{S}_n^{-1}=\\mathbf{S}_0^{-1}+\\sigma^{-2}\\sum_{i=1}^n\\boldsymbol{\\phi}_i\\boldsymbol{\\phi}_i^\\top=\\mathbf{S}_0^{-1}+\\sigma^{-2}\\boldsymbol{\\Phi}_n^\\top\\boldsymbol{\\Phi}_n\\). Similarly \\(\\mathbf{S}_n^{-1}\\mathbf{m}_n=\\mathbf{S}_0^{-1}\\mathbf{m}_0+\\sigma^{-2}\\boldsymbol{\\Phi}_n^\\top\\mathbf{y}_n\\). Both match the batch formula.' },
                { question:'For \\(\\boldsymbol{\\phi}(x)=(1,x)^\\top\\), in which direction does the posterior shrink most when observing a point at \\(x=3\\)?',
                  hint:'The rank-1 update adds information only along \\(\\boldsymbol{\\phi}\\).',
                  solution:'The update adds \\(\\sigma^{-2}(1,3)^\\top(1,3)\\) to the precision. The eigenvector is \\((1,3)/\\sqrt{10}\\) with eigenvalue \\(10\\sigma^{-2}\\). The posterior contracts most along \\((1,3)\\).' },
                { question:'Why does posterior uncertainty decrease fastest with diverse input locations?',
                  hint:'Consider which directions in weight space each point constrains.',
                  solution:'Each point at \\(x_n\\) constrains weights along \\(\\boldsymbol{\\phi}(x_n)\\). If all data is at \\(x=0\\), only \\(w_0\\) is constrained. With diverse \\(x\\) values, constraints span different directions, shrinking the posterior in all directions of weight space.' }
            ]
        },

        // ===================== Section 3: Predictive Distribution =====================
        {
            id: 'ch14-sec03',
            title: 'Predictive Distribution',
            content: `<h2>14.3 Predictive Distribution</h2>
<div class="env-block intuition"><div class="env-title">Intuition</div><div class="env-body">
<p>Rather than predicting with a single weight vector, the Bayesian approach averages over all plausible weights. The resulting predictive distribution is wider where we are uncertain (far from data) and narrower where we are confident.</p>
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 14.3.1 -- Predictive Distribution</div><div class="env-body">
<p>For a new input \\(\\mathbf{x}_*\\):</p>
\\[p(y_*\\mid\\mathbf{x}_*,\\mathbf{y})=\\mathcal{N}(y_*;\\,\\mu_*,\\,\\sigma_*^2),\\]
\\[\\mu_*=\\mathbf{m}_N^\\top\\boldsymbol{\\phi}(\\mathbf{x}_*),\\qquad \\sigma_*^2=\\underbrace{\\sigma^2}_{\\text{aleatoric}}+\\underbrace{\\boldsymbol{\\phi}_*^\\top\\mathbf{S}_N\\boldsymbol{\\phi}_*}_{\\text{epistemic}}.\\]
</div></div>

<div class="env-block proof"><div class="env-title">Proof</div><div class="env-body">
<p>Since \\(y_*=\\mathbf{w}^\\top\\boldsymbol{\\phi}_*+\\epsilon\\) with \\(\\mathbf{w}\\sim\\mathcal{N}(\\mathbf{m}_N,\\mathbf{S}_N)\\) and \\(\\epsilon\\sim\\mathcal{N}(0,\\sigma^2)\\) independent, the mean is \\(\\mathbf{m}_N^\\top\\boldsymbol{\\phi}_*\\). By the law of total variance, \\(\\operatorname{Var}(y_*)=\\sigma^2+\\boldsymbol{\\phi}_*^\\top\\mathbf{S}_N\\boldsymbol{\\phi}_*\\).</p>
<div class="qed">&#8718;</div></div></div>

<h3>Two Sources of Uncertainty</h3>
<ul>
<li><strong>Aleatoric</strong> (\\(\\sigma^2\\)): irreducible observation noise; persists even with infinite data.</li>
<li><strong>Epistemic</strong> (\\(\\boldsymbol{\\phi}_*^\\top\\mathbf{S}_N\\boldsymbol{\\phi}_*\\)): uncertainty from limited data; vanishes as \\(N\\to\\infty\\).</li>
</ul>

<div class="env-block remark"><div class="env-title">Remark</div><div class="env-body">
<p>With polynomial features, \\(\\boldsymbol{\\phi}_*^\\top\\mathbf{S}_N\\boldsymbol{\\phi}_*\\) grows for \\(x_*\\) far from training data, producing widening uncertainty bands. Bayesian models "know what they don't know."</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-predictive-dist"></div>`,
            visualizations: [{
                id: 'viz-predictive-dist',
                title: 'Predictive Distribution with Uncertainty Bands',
                description: 'Click to place data. The blue line is the predictive mean; shaded bands show \\(\\pm 1\\sigma\\) and \\(\\pm 2\\sigma\\). Bands widen away from data. Use the degree slider to change the basis.',
                setup(body, controls) {
                    const viz = new VizEngine(body, {width:680,height:360,scale:55,originX:340,originY:250});
                    const ctx = viz.ctx;
                    let pts=[], degree=1;
                    const sigma2=0.1, alpha=0.01;
                    function phi(x,d){const p=[];for(let i=0;i<=d;i++)p.push(Math.pow(x,i));return p;}
                    function matInv(A){
                        const n=A.length;const aug=A.map((r,i)=>[...r,...Array(n).fill(0).map((_,j)=>i===j?1:0)]);
                        for(let i=0;i<n;i++){let mx=i;for(let r=i+1;r<n;r++)if(Math.abs(aug[r][i])>Math.abs(aug[mx][i]))mx=r;
                        [aug[i],aug[mx]]=[aug[mx],aug[i]];const pv=aug[i][i];if(Math.abs(pv)<1e-14)return null;
                        for(let j=0;j<2*n;j++)aug[i][j]/=pv;for(let r=0;r<n;r++){if(r===i)continue;const f=aug[r][i];for(let j=0;j<2*n;j++)aug[r][j]-=f*aug[i][j];}}
                        return aug.map(r=>r.slice(n));
                    }
                    function pred(xs){
                        const M=degree+1;let SNinv=[];
                        for(let i=0;i<M;i++){SNinv[i]=[];for(let j=0;j<M;j++)SNinv[i][j]=i===j?alpha:0;}
                        let PhiTy=new Array(M).fill(0);
                        for(const p of pts){const ph=phi(p.x,degree);for(let i=0;i<M;i++){for(let j=0;j<M;j++)SNinv[i][j]+=ph[i]*ph[j]/sigma2;PhiTy[i]+=ph[i]*p.y/sigma2;}}
                        const SN=matInv(SNinv);if(!SN)return xs.map(()=>({mu:0,v:sigma2}));
                        let mN=new Array(M).fill(0);for(let i=0;i<M;i++)for(let j=0;j<M;j++)mN[i]+=SN[i][j]*PhiTy[j];
                        return xs.map(x=>{const ph=phi(x,degree);let mu=0,v=sigma2;for(let i=0;i<M;i++){mu+=mN[i]*ph[i];for(let j=0;j<M;j++)v+=ph[i]*SN[i][j]*ph[j];}return{mu,v};});
                    }
                    function draw(){
                        viz.clear();viz.drawGrid(1);viz.drawAxes();
                        if(pts.length>0){
                            const nX=120,xs=[];for(let i=0;i<=nX;i++)xs.push(-6+12*i/nX);
                            const preds=pred(xs);
                            ctx.fillStyle=viz.colors.blue+'18';ctx.beginPath();
                            for(let i=0;i<=nX;i++){const[sx,sy]=viz.toScreen(xs[i],preds[i].mu+2*Math.sqrt(preds[i].v));i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}
                            for(let i=nX;i>=0;i--){const[sx,sy]=viz.toScreen(xs[i],preds[i].mu-2*Math.sqrt(preds[i].v));ctx.lineTo(sx,sy);}ctx.fill();
                            ctx.fillStyle=viz.colors.blue+'28';ctx.beginPath();
                            for(let i=0;i<=nX;i++){const[sx,sy]=viz.toScreen(xs[i],preds[i].mu+Math.sqrt(preds[i].v));i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}
                            for(let i=nX;i>=0;i--){const[sx,sy]=viz.toScreen(xs[i],preds[i].mu-Math.sqrt(preds[i].v));ctx.lineTo(sx,sy);}ctx.fill();
                            ctx.strokeStyle=viz.colors.blue;ctx.lineWidth=2.5;ctx.beginPath();
                            for(let i=0;i<=nX;i++){const[sx,sy]=viz.toScreen(xs[i],preds[i].mu);i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}ctx.stroke();
                        }
                        for(const p of pts)viz.drawPoint(p.x,p.y,viz.colors.orange,'',5);
                        viz.screenText('Degree '+degree,viz.width-90,20,viz.colors.white,13);
                        viz.screenText('N='+pts.length,viz.width-90,38,viz.colors.text,12);
                    }
                    viz.canvas.addEventListener('click',function(e){const rect=viz.canvas.getBoundingClientRect();const[mx,my]=viz.toMath(e.clientX-rect.left,e.clientY-rect.top);pts.push({x:mx,y:my});draw();});
                    VizEngine.createSlider(controls,'Degree',1,6,1,1,v=>{degree=Math.round(v);draw();});
                    VizEngine.createButton(controls,'Reset',()=>{pts=[];draw();});
                    VizEngine.createButton(controls,'Sample',()=>{
                        const rng=()=>{let u=0,v=0;while(!u)u=Math.random();while(!v)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
                        pts=[];for(let i=0;i<8;i++){const x=-2+4*Math.random();pts.push({x,y:0.3*Math.sin(x*1.5)+0.5+rng()*Math.sqrt(sigma2)});}draw();});
                    draw(); return viz;
                }
            }],
            exercises: [
                { question:'Derive the predictive variance using the law of total variance.',
                  hint:'\\(\\operatorname{Var}(y_*)=\\mathbb{E}_\\mathbf{w}[\\operatorname{Var}(y_*\\mid\\mathbf{w})]+\\operatorname{Var}_\\mathbf{w}(\\mathbb{E}[y_*\\mid\\mathbf{w}])\\).',
                  solution:'\\(\\mathbb{E}_\\mathbf{w}[\\operatorname{Var}(y_*\\mid\\mathbf{w})]=\\sigma^2\\) (constant). \\(\\operatorname{Var}_\\mathbf{w}(\\mathbf{w}^\\top\\boldsymbol{\\phi}_*)=\\boldsymbol{\\phi}_*^\\top\\mathbf{S}_N\\boldsymbol{\\phi}_*\\). Sum: \\(\\sigma_*^2=\\sigma^2+\\boldsymbol{\\phi}_*^\\top\\mathbf{S}_N\\boldsymbol{\\phi}_*\\).' },
                { question:'Show that epistemic uncertainty vanishes as \\(N\\to\\infty\\).',
                  hint:'Consider eigenvalues of \\(\\mathbf{S}_N\\).',
                  solution:'\\(\\mathbf{S}_N^{-1}=\\mathbf{S}_0^{-1}+\\sigma^{-2}\\boldsymbol{\\Phi}^\\top\\boldsymbol{\\Phi}\\). As \\(N\\to\\infty\\) with sufficient coverage, eigenvalues of \\(\\boldsymbol{\\Phi}^\\top\\boldsymbol{\\Phi}\\to\\infty\\), so \\(\\mathbf{S}_N\\to\\mathbf{0}\\) and \\(\\boldsymbol{\\phi}_*^\\top\\mathbf{S}_N\\boldsymbol{\\phi}_*\\to 0\\). The variance converges to \\(\\sigma^2\\).' },
                { question:'The predictive mean equals the MAP prediction. Why prefer the Bayesian approach?',
                  hint:'Think beyond point estimates.',
                  solution:'The Bayesian approach provides calibrated uncertainty via the predictive variance. This is essential for: (1) knowing when to trust the model, (2) active learning (query where uncertainty is highest), (3) decision-making under risk, and (4) model comparison via marginal likelihood.' }
            ]
        },

        // ===================== Section 4: Model Comparison =====================
        {
            id: 'ch14-sec04',
            title: 'Model Comparison',
            content: `<h2>14.4 Model Comparison</h2>
<div class="env-block intuition"><div class="env-title">Bayesian Occam's Razor</div><div class="env-body">
<p>A complex model spreads its prior probability over a vast space of possible datasets, assigning little probability to any one. A simpler model concentrates its probability on fewer datasets. The <em>marginal likelihood</em> automatically penalizes complexity, selecting the simplest model that explains the data.</p>
</div></div>

<div class="env-block definition"><div class="env-title">Definition 14.4.1 -- Marginal Likelihood</div><div class="env-body">
\\[p(\\mathbf{y}\\mid\\mathcal{M})=\\int p(\\mathbf{y}\\mid\\mathbf{w},\\mathcal{M})\\,p(\\mathbf{w}\\mid\\mathcal{M})\\,d\\mathbf{w}.\\]
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 14.4.2 -- Closed-Form Marginal Likelihood</div><div class="env-body">
<p>With prior \\(\\mathcal{N}(\\mathbf{0},\\alpha^{-1}\\mathbf{I})\\) and likelihood \\(\\mathcal{N}(\\mathbf{y};\\boldsymbol{\\Phi}\\mathbf{w},\\sigma^2\\mathbf{I})\\):</p>
\\[p(\\mathbf{y}\\mid\\mathcal{M})=\\mathcal{N}(\\mathbf{y};\\,\\mathbf{0},\\,\\mathbf{C}),\\quad \\mathbf{C}=\\sigma^2\\mathbf{I}+\\alpha^{-1}\\boldsymbol{\\Phi}\\boldsymbol{\\Phi}^\\top.\\]
\\[\\ln p(\\mathbf{y}\\mid\\mathcal{M})=\\underbrace{-\\tfrac{1}{2}\\mathbf{y}^\\top\\mathbf{C}^{-1}\\mathbf{y}}_{\\text{data fit}}\\underbrace{-\\tfrac{1}{2}\\ln|\\mathbf{C}|}_{\\text{complexity}}\\underbrace{-\\tfrac{N}{2}\\ln(2\\pi)}_{\\text{const}}.\\]
</div></div>

<div class="env-block proof"><div class="env-title">Proof</div><div class="env-body">
<p>Since \\(\\mathbf{y}=\\boldsymbol{\\Phi}\\mathbf{w}+\\boldsymbol{\\epsilon}\\) with independent \\(\\mathbf{w}\\sim\\mathcal{N}(\\mathbf{0},\\alpha^{-1}\\mathbf{I})\\) and \\(\\boldsymbol{\\epsilon}\\sim\\mathcal{N}(\\mathbf{0},\\sigma^2\\mathbf{I})\\), we have \\(\\mathbf{y}\\sim\\mathcal{N}(\\mathbf{0},\\alpha^{-1}\\boldsymbol{\\Phi}\\boldsymbol{\\Phi}^\\top+\\sigma^2\\mathbf{I})\\).</p>
<div class="qed">&#8718;</div></div></div>

<div class="env-block definition"><div class="env-title">Definition 14.4.3 -- Bayes Factor</div><div class="env-body">
<p>\\(\\mathrm{BF}_{12}=p(\\mathbf{y}\\mid\\mathcal{M}_1)/p(\\mathbf{y}\\mid\\mathcal{M}_2)\\). Values above 100 are "decisive" evidence for \\(\\mathcal{M}_1\\) on Jeffreys' scale.</p>
</div></div>

<div class="viz-placeholder" data-viz="viz-model-comparison"></div>`,
            visualizations: [{
                id: 'viz-model-comparison',
                title: 'Marginal Likelihood vs. Model Complexity',
                description: 'Bars show log marginal likelihood for polynomial degrees 1--6. Data is generated from a quadratic. The marginal likelihood peaks at the true complexity, not the highest degree. Click "New Data" to regenerate.',
                setup(body, controls) {
                    const viz = new VizEngine(body, {width:680,height:360,scale:1,originX:0,originY:0});
                    const ctx = viz.ctx;
                    const sigma2=0.15, alpha=0.1;
                    let pts=[];
                    function rng(){let u=0,v=0;while(!u)u=Math.random();while(!v)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
                    function genData(n){pts=[];for(let i=0;i<n;i++){const x=-2+4*Math.random();pts.push({x,y:0.5-0.3*x+0.2*x*x+rng()*Math.sqrt(sigma2)});}}
                    function logML(deg){
                        const M=deg+1,N=pts.length;if(N===0)return-Infinity;
                        const Phi=[],y=[];for(const p of pts){const r=[];for(let d=0;d<=deg;d++)r.push(Math.pow(p.x,d));Phi.push(r);y.push(p.y);}
                        const C=[];for(let i=0;i<N;i++){C[i]=[];for(let j=0;j<N;j++){let s=i===j?sigma2:0;for(let k=0;k<M;k++)s+=Phi[i][k]*Phi[j][k]/alpha;C[i][j]=s;}}
                        const L=[];for(let i=0;i<N;i++)L[i]=new Array(N).fill(0);
                        for(let i=0;i<N;i++)for(let j=0;j<=i;j++){let s=C[i][j];for(let k=0;k<j;k++)s-=L[i][k]*L[j][k];L[i][j]=i===j?Math.sqrt(Math.max(s,1e-10)):s/L[j][j];}
                        let ld=0;for(let i=0;i<N;i++)ld+=2*Math.log(L[i][i]);
                        const z=new Array(N).fill(0);for(let i=0;i<N;i++){let s=y[i];for(let j=0;j<i;j++)s-=L[i][j]*z[j];z[i]=s/L[i][i];}
                        let q=0;for(let i=0;i<N;i++)q+=z[i]*z[i];
                        return-0.5*N*Math.log(2*Math.PI)-0.5*ld-0.5*q;
                    }
                    function draw(){
                        ctx.fillStyle=viz.colors.bg;ctx.fillRect(0,0,680,360);
                        const maxDeg=6,lmls=[];for(let d=1;d<=maxDeg;d++)lmls.push(logML(d));
                        const best=Math.max(...lmls),worst=Math.min(...lmls),range=Math.max(best-worst,1);
                        // Left: bar chart
                        const bL=40,bR=330,bT=50,bB=320;
                        ctx.fillStyle=viz.colors.white;ctx.font='13px -apple-system,sans-serif';ctx.textAlign='center';ctx.fillText('Log Marginal Likelihood',(bL+bR)/2,25);
                        ctx.strokeStyle=viz.colors.axis;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(bL,bB);ctx.lineTo(bR,bB);ctx.stroke();
                        ctx.beginPath();ctx.moveTo(bL,bT);ctx.lineTo(bL,bB);ctx.stroke();
                        const bW=(bR-bL-20)/maxDeg;
                        for(let d=0;d<maxDeg;d++){
                            const h=((lmls[d]-worst+0.1*range)/(range*1.3))*(bB-bT-20);
                            const bx=bL+10+d*bW,by=bB-h,isBest=lmls[d]===best;
                            ctx.fillStyle=isBest?viz.colors.green:viz.colors.blue;ctx.fillRect(bx+4,by,bW-8,h);
                            ctx.fillStyle=viz.colors.text;ctx.font='11px -apple-system,sans-serif';ctx.textAlign='center';
                            ctx.fillText('d='+(d+1),bx+bW/2,bB+16);
                            ctx.fillStyle=isBest?viz.colors.green:viz.colors.text;ctx.fillText(lmls[d].toFixed(1),bx+bW/2,by-8);
                        }
                        // Right: data
                        const ox=510,oy=200,sc=48;
                        ctx.strokeStyle=viz.colors.grid;ctx.lineWidth=0.5;ctx.beginPath();ctx.moveTo(350,0);ctx.lineTo(350,360);ctx.stroke();
                        ctx.strokeStyle=viz.colors.axis;ctx.lineWidth=1;
                        ctx.beginPath();ctx.moveTo(360,oy);ctx.lineTo(670,oy);ctx.stroke();
                        ctx.beginPath();ctx.moveTo(ox,30);ctx.lineTo(ox,350);ctx.stroke();
                        ctx.fillStyle=viz.colors.white;ctx.font='13px -apple-system,sans-serif';ctx.textAlign='center';ctx.fillText('Data & Best Fit',520,25);
                        const bestD=lmls.indexOf(best)+1;
                        if(pts.length>0){
                            const M=bestD+1;let Si=[];for(let i=0;i<M;i++){Si[i]=[];for(let j=0;j<M;j++)Si[i][j]=i===j?alpha:0;}
                            let Py=new Array(M).fill(0);
                            for(const p of pts){const ph=[];for(let d=0;d<=bestD;d++)ph.push(Math.pow(p.x,d));for(let i=0;i<M;i++){for(let j=0;j<M;j++)Si[i][j]+=ph[i]*ph[j]/sigma2;Py[i]+=ph[i]*p.y/sigma2;}}
                            const aug=Si.map((r,i)=>[...r,Py[i]]);
                            for(let i=0;i<M;i++){let mx=i;for(let r=i+1;r<M;r++)if(Math.abs(aug[r][i])>Math.abs(aug[mx][i]))mx=r;[aug[i],aug[mx]]=[aug[mx],aug[i]];for(let r=i+1;r<M;r++){const f=aug[r][i]/aug[i][i];for(let c=i;c<=M;c++)aug[r][c]-=f*aug[i][c];}}
                            const mN=new Array(M).fill(0);for(let i=M-1;i>=0;i--){let s=aug[i][M];for(let j=i+1;j<M;j++)s-=aug[i][j]*mN[j];mN[i]=s/aug[i][i];}
                            ctx.strokeStyle=viz.colors.green;ctx.lineWidth=2;ctx.beginPath();
                            for(let i=0;i<=100;i++){const x=-3+6*i/100;let y=0;for(let d=0;d<M;d++)y+=mN[d]*Math.pow(x,d);const sx=ox+x*sc,sy=oy-y*sc;i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}ctx.stroke();
                            ctx.fillStyle=viz.colors.green;ctx.font='11px -apple-system,sans-serif';ctx.textAlign='left';ctx.fillText('Best: degree '+bestD,365,350);
                        }
                        for(const p of pts){ctx.fillStyle=viz.colors.orange;ctx.beginPath();ctx.arc(ox+p.x*sc,oy-p.y*sc,4,0,2*Math.PI);ctx.fill();}
                    }
                    genData(12);
                    VizEngine.createButton(controls,'New Data',()=>{genData(12);draw();});
                    VizEngine.createButton(controls,'More (25)',()=>{genData(25);draw();});
                    draw(); return viz;
                }
            }],
            exercises: [
                { question:'Why does a degree-10 polynomial typically have lower marginal likelihood than degree-2 when the truth is quadratic?',
                  hint:'Consider the complexity penalty \\(-\\frac{1}{2}\\ln|\\mathbf{C}|\\).',
                  solution:'The degree-10 model has 11 parameters, spreading prior probability over a much larger volume. The determinant \\(|\\mathbf{C}|\\) is correspondingly larger, so the penalty \\(-\\frac{1}{2}\\ln|\\mathbf{C}|\\) is more negative. The slight improvement in data fit cannot compensate. The degree-2 model concentrates probability on quadratic functions, assigning higher probability to quadratic data.' },
                { question:'Show that the marginal likelihood satisfies \\(p(\\mathbf{y})=p(\\mathbf{y}\\mid\\hat{\\mathbf{w}})\\cdot p(\\hat{\\mathbf{w}})/p(\\hat{\\mathbf{w}}\\mid\\mathbf{y})\\) for any \\(\\hat{\\mathbf{w}}\\).',
                  hint:'Start from Bayes\' theorem and rearrange.',
                  solution:'Bayes\' rule: \\(p(\\mathbf{y})=p(\\mathbf{y}\\mid\\mathbf{w})p(\\mathbf{w})/p(\\mathbf{w}\\mid\\mathbf{y})\\) for any \\(\\mathbf{w}\\). Evaluating at \\(\\hat{\\mathbf{w}}\\) gives the result. The denominator \\(p(\\hat{\\mathbf{w}}\\mid\\mathbf{y})\\) acts as the "Occam factor," penalizing models where the posterior is diffuse.' },
                { question:'Give two scenarios where marginal likelihood has advantages over cross-validation.',
                  hint:'Consider sample size and computational cost.',
                  solution:'(1) When data is scarce, CV must hold out data, further reducing training set size. Marginal likelihood uses all data. (2) When optimizing continuous hyperparameters (\\(\\alpha,\\sigma^2\\)), the marginal likelihood gradient is available in closed form, enabling efficient optimization. CV requires expensive retraining for each hyperparameter setting.' }
            ]
        }
    ]
});
