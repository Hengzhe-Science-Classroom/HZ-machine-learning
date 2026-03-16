window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
  id: 'ch10', number: 10,
  title: 'Ensemble Methods',
  subtitle: 'Combining Weak Learners into Strong Predictors',
  sections: [
    // ── SECTION 1: Bagging ──────────────────────────────────────────────────
    {
      id: 'ch10-sec01', title: 'Bagging',
      content: `
<div class="env-block intuition"><div class="env-title">The Wisdom of Crowds</div><div class="env-body"><p>Ask one person to estimate the number of jelly beans in a jar, and they will likely be far off. Average the guesses of a hundred people, and the result is often remarkably close. Ensemble methods exploit the same principle: combining many imperfect models yields a predictor substantially better than any single model alone.</p></div></div>
<h2>Bootstrap Aggregating (Bagging)</h2>
<p>Leo Breiman introduced <strong>bagging</strong> (Bootstrap AGGregatING) in 1996 to reduce the variance of a high-variance estimator. The idea: train multiple copies of a model on different <em>bootstrap samples</em>, then average their predictions.</p>
<div class="env-block definition"><div class="env-title">Definition 10.1 (Bootstrap Sample)</div><div class="env-body">
<p>Given a dataset \\(\\mathcal{D} = \\{(\\mathbf{x}_i, y_i)\\}_{i=1}^n\\), a <strong>bootstrap sample</strong> \\(\\mathcal{D}^*\\) is a dataset of \\(n\\) points drawn uniformly at random <em>with replacement</em> from \\(\\mathcal{D}\\). Each bootstrap sample includes roughly \\(63.2\\%\\) of the original points, while about \\(36.8\\%\\) are left out.</p>
</div></div>
<p>The fraction \\(1 - (1 - 1/n)^n \\to 1 - 1/e \\approx 0.632\\) follows from the probability that a given point is selected at least once in \\(n\\) draws.</p>
<div class="env-block definition"><div class="env-title">Algorithm 10.1 (Bagging)</div><div class="env-body">
<p><strong>Input:</strong> Training set \\(\\mathcal{D}\\), base learner \\(\\mathcal{A}\\), number of models \\(B\\).</p>
<ol>
  <li>For \\(b = 1, \\dots, B\\): draw bootstrap sample \\(\\mathcal{D}_b^*\\) and train \\(h_b = \\mathcal{A}(\\mathcal{D}_b^*)\\).</li>
  <li><strong>Regression:</strong> \\(\\hat{f}(\\mathbf{x}) = \\frac{1}{B}\\sum_{b=1}^B h_b(\\mathbf{x})\\).</li>
  <li><strong>Classification:</strong> \\(\\hat{f}(\\mathbf{x}) = \\text{majority vote of } h_1(\\mathbf{x}), \\dots, h_B(\\mathbf{x})\\).</li>
</ol>
</div></div>
<h3>Why Bagging Reduces Variance</h3>
<p>Suppose each base model \\(h_b\\) has variance \\(\\sigma^2\\) and pairwise correlation \\(\\rho\\). The variance of the bagged prediction is:</p>
\\[\\text{Var}\\!\\left(\\frac{1}{B}\\sum_{b=1}^B h_b\\right) = \\rho \\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2.\\]
<p>As \\(B \\to \\infty\\), the second term vanishes, leaving \\(\\rho\\sigma^2\\). Bagging primarily reduces <strong>variance</strong> without affecting <strong>bias</strong>, making it most effective for high-variance learners like deep decision trees.</p>
<div class="viz-placeholder" data-viz="viz-bagging-variance"></div>`,
      visualizations: [{
        id: 'viz-bagging-variance',
        title: 'Bagging: Variance Reduction Through Averaging',
        description: 'Thin colored lines are models on bootstrap samples. The thick white line is their average. Increase B to see variance reduction.',
        setup(container, controls) {
          const viz = new VizEngine(container, {scale:40, originX:60, originY:280});
          let B = 5, seed = 42;
          const rng0 = (function(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}})(seed);
          function mkRng(s){return function(){s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
          const trueF = x => 2*Math.sin(x*0.8)+0.5*x;
          const baseData = [];
          for(let i=0;i<30;i++){const x=rng0()*10;baseData.push({x,y:trueF(x)+(rng0()-0.5)*2.5});}
          VizEngine.createSlider(controls,'Models (B)',1,20,B,1,v=>{B=Math.round(v);draw();});
          VizEngine.createButton(controls,'New Seed',()=>{seed=Math.floor(Math.random()*10000);draw();});
          function fitPoly(data,deg){const n=data.length,cols=deg+1;const XtX=Array.from({length:cols},()=>new Array(cols).fill(0)),Xty=new Array(cols).fill(0);for(let i=0;i<n;i++){const row=[];for(let d=0;d<=deg;d++)row.push(Math.pow(data[i].x,d));for(let j=0;j<cols;j++){Xty[j]+=row[j]*data[i].y;for(let k=0;k<cols;k++)XtX[j][k]+=row[j]*row[k];}}const A=XtX.map((r,i)=>[...r,Xty[i]]);for(let i=0;i<cols;i++){let mx=i;for(let k=i+1;k<cols;k++)if(Math.abs(A[k][i])>Math.abs(A[mx][i]))mx=k;[A[i],A[mx]]=[A[mx],A[i]];if(Math.abs(A[i][i])<1e-12)continue;for(let k=i+1;k<cols;k++){const f=A[k][i]/A[i][i];for(let j=i;j<=cols;j++)A[k][j]-=f*A[i][j];}}const c=new Array(cols).fill(0);for(let i=cols-1;i>=0;i--){c[i]=A[i][cols];for(let j=i+1;j<cols;j++)c[i]-=A[i][j]*c[j];c[i]/=A[i][i];}return x=>c.reduce((s,v,d)=>s+v*Math.pow(x,d),0);}
          function draw(){
            viz.clear(); const ctx=viz.ctx, W=viz.width, H=viz.height;
            ctx.strokeStyle=viz.colors.grid;ctx.lineWidth=0.5;
            for(let gx=0;gx<=10;gx++){const sx=60+gx*(W-80)/10;ctx.beginPath();ctx.moveTo(sx,20);ctx.lineTo(sx,H-30);ctx.stroke();}
            ctx.strokeStyle=viz.colors.axis;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(60,H-30);ctx.lineTo(W-20,H-30);ctx.stroke();ctx.beginPath();ctx.moveTo(60,20);ctx.lineTo(60,H-30);ctx.stroke();
            const toSx=x=>60+x*(W-80)/10, toSy=y=>H-30-(y+4)*(H-50)/14;
            ctx.strokeStyle=viz.colors.green;ctx.lineWidth=2;ctx.setLineDash([6,4]);ctx.beginPath();
            for(let p=0;p<=200;p++){const x=p*10/200,sx=toSx(x),sy=toSy(trueF(x));p===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}ctx.stroke();ctx.setLineDash([]);
            const rng=mkRng(seed+100), modelPreds=[];
            const cols=['#58a6ff','#3fb9a0','#f0883e','#bc8cff','#f778ba','#f85149','#d29922','#66ccff','#99ff66','#ff9966','#cc66ff','#66ffcc','#ffcc66','#ff66cc','#6699ff','#ccff66','#ff6699','#66ffff','#ffff66','#ff66ff'];
            for(let b=0;b<B;b++){const sample=[];for(let i=0;i<baseData.length;i++)sample.push(baseData[Math.floor(rng()*baseData.length)]);const model=fitPoly(sample,6);const preds=[];ctx.strokeStyle=cols[b%cols.length]+'55';ctx.lineWidth=1;ctx.beginPath();for(let p=0;p<=200;p++){const x=p*10/200,y=model(x);preds.push(y);const sx=toSx(x),sy=toSy(y);p===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}ctx.stroke();modelPreds.push(preds);}
            ctx.strokeStyle=viz.colors.white;ctx.lineWidth=3;ctx.beginPath();
            for(let p=0;p<=200;p++){const avg=modelPreds.reduce((s,pr)=>s+pr[p],0)/B;const sx=toSx(p*10/200),sy=toSy(avg);p===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}ctx.stroke();
            for(const p of baseData){ctx.fillStyle=viz.colors.text+'88';ctx.beginPath();ctx.arc(toSx(p.x),toSy(p.y),3,0,Math.PI*2);ctx.fill();}
            ctx.font='12px -apple-system,sans-serif';ctx.fillStyle=viz.colors.green;ctx.fillText('--- True function',W-180,35);ctx.fillStyle=viz.colors.white;ctx.fillText('--- Bagged (avg of '+B+')',W-180,55);ctx.fillStyle=viz.colors.text;ctx.fillText('--- Individual models',W-180,75);
            viz.screenText('B = '+B+' bootstrap models',W/2,H-8,viz.colors.text,12);
          }
          draw(); return viz;
        }
      }],
      exercises: [
        {question:'A single decision tree has prediction variance \\(\\sigma^2 = 100\\). If we bag \\(B = 25\\) trees with pairwise correlation \\(\\rho = 0.4\\), what is the variance of the bagged predictor?',hint:'Use \\(\\text{Var} = \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2\\).',solution:'\\(\\text{Var} = 0.4 \\times 100 + \\frac{0.6}{25} \\times 100 = 40 + 2.4 = 42.4\\). Variance drops from 100 to 42.4 (57.6% reduction). Even as \\(B \\to \\infty\\), variance cannot drop below \\(\\rho\\sigma^2 = 40\\).'},
        {question:'Show that the probability of a specific training example being excluded from a bootstrap sample of size \\(n\\) converges to \\(1/e\\) as \\(n \\to \\infty\\).',hint:'The probability of not being selected in a single draw is \\(1 - 1/n\\). Use the limit definition of \\(e\\).',solution:'The probability of not being selected in any of \\(n\\) draws is \\((1 - 1/n)^n\\). Taking the limit: \\(\\lim_{n \\to \\infty}(1 - 1/n)^n = e^{-1} \\approx 0.368\\). So approximately 36.8% of points are left out, and 63.2% are included.'},
        {question:'Why is bagging ineffective for linear regression?',hint:'Consider what happens when you average multiple linear fits on bootstrapped data.',solution:'Linear regression is a stable (low-variance) learner. Bootstrap samples produce nearly identical fitted lines because OLS is a linear function of the data. Averaging near-identical predictions yields almost the same result, so variance reduction is negligible. If the true relationship is nonlinear, every bootstrapped model has the same bias, and averaging cannot reduce bias.'}
      ]
    },
    // ── SECTION 2: Random Forests ───────────────────────────────────────────
    {
      id: 'ch10-sec02', title: 'Random Forests',
      content: `
<h2>Random Forests</h2>
<p>Bagging reduces variance, but the trees remain correlated because they all split on the same strong predictors first. <strong>Random forests</strong> (Breiman, 2001) add a second source of randomness: at each split, only a random subset of features is considered.</p>
<div class="env-block definition"><div class="env-title">Algorithm 10.2 (Random Forest)</div><div class="env-body">
<p><strong>Input:</strong> Training set \\(\\mathcal{D}\\), number of trees \\(B\\), feature subset size \\(m\\).</p>
<ol>
  <li>For \\(b = 1, \\dots, B\\): draw bootstrap sample \\(\\mathcal{D}_b^*\\).</li>
  <li>Grow tree \\(T_b\\), at each node selecting \\(m\\) random features and splitting on the best among those \\(m\\).</li>
  <li>Aggregate by averaging (regression) or majority vote (classification).</li>
</ol>
<p>Defaults: \\(m = \\lfloor \\sqrt{p} \\rfloor\\) for classification, \\(m = \\lfloor p/3 \\rfloor\\) for regression.</p>
</div></div>
<p>Feature subsampling <em>decorrelates</em> the trees, reducing \\(\\rho\\) in the variance formula. Even if one feature dominates, some trees lack access to it, forcing them to find alternative patterns.</p>
<h3>Out-of-Bag (OOB) Error</h3>
<p>Each bootstrap sample leaves out ~36.8% of points. For each \\(\\mathbf{x}_i\\), we aggregate predictions only from trees that did <em>not</em> include \\(\\mathbf{x}_i\\) in their bootstrap sample. This <strong>OOB error</strong> approximates test error without needing a separate validation set.</p>
<h3>Feature Importance</h3>
<p>Random forests measure feature importance via <strong>permutation importance</strong>: randomly permute feature \\(j\\) in the OOB data and measure the increase in error. Large increases indicate important features.</p>
<div class="env-block remark"><div class="env-title">Strengths</div><div class="env-body"><p>Random forests handle high-dimensional data, mixed types, missing values, and nonlinear relationships. They resist overfitting (more trees never hurts), are parallelizable, and require minimal tuning.</p></div></div>
<div class="viz-placeholder" data-viz="viz-rf-boundary"></div>`,
      visualizations: [{
        id: 'viz-rf-boundary',
        title: 'Random Forest vs. Single Tree: Decision Boundaries',
        description: 'Left: a single tree produces a jagged, overfit boundary. Right: a random forest produces a smooth boundary.',
        setup(container, controls) {
          const viz = new VizEngine(container, {scale:40, originX:280, originY:200});
          let numTrees = 10;
          function mkRng(s){return function(){s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
          const rng0=mkRng(77); const data=[];
          for(let i=0;i<40;i++){if(i<20)data.push({x:0.3+rng0()*2.5,y:0.3+rng0()*2.5,c:0});else data.push({x:3.5+rng0()*2.5,y:0.3+rng0()*2.5,c:1});}
          for(let i=0;i<8;i++)data.push({x:2.5+(rng0()-0.5)*1.5,y:rng0()*3,c:rng0()>0.5?1:0});
          VizEngine.createSlider(controls,'Trees',1,30,numTrees,1,v=>{numTrees=Math.round(v);draw();});
          function buildTree(subset,depth,maxD,rng){const c1=subset.filter(p=>p.c===1).length,c0=subset.length-c1;if(depth>=maxD||subset.length<=3||c0===0||c1===0)return{leaf:true,prob:c1/subset.length};let bestF=0,bestT=3,bestG=Infinity;const feats=rng()>0.5?[0]:[1];if(rng()>0.3)feats.push(feats[0]===0?1:0);for(const f of feats){const vals=subset.map(p=>f===0?p.x:p.y).sort((a,b)=>a-b);for(let i=0;i<vals.length-1;i++){const th=(vals[i]+vals[i+1])/2,L=subset.filter(p=>(f===0?p.x:p.y)<=th),R=subset.filter(p=>(f===0?p.x:p.y)>th);if(!L.length||!R.length)continue;const gL=1-Math.pow(L.filter(p=>p.c===0).length/L.length,2)-Math.pow(L.filter(p=>p.c===1).length/L.length,2);const gR=1-Math.pow(R.filter(p=>p.c===0).length/R.length,2)-Math.pow(R.filter(p=>p.c===1).length/R.length,2);const g=(L.length*gL+R.length*gR)/subset.length;if(g<bestG){bestG=g;bestF=f;bestT=th;}}}const L=subset.filter(p=>(bestF===0?p.x:p.y)<=bestT),R=subset.filter(p=>(bestF===0?p.x:p.y)>bestT);if(!L.length||!R.length)return{leaf:true,prob:c1/subset.length};return{leaf:false,feat:bestF,thresh:bestT,left:buildTree(L,depth+1,maxD,rng),right:buildTree(R,depth+1,maxD,rng)};}
          function predict(t,x,y){if(t.leaf)return t.prob;return(t.feat===0?x:y)<=t.thresh?predict(t.left,x,y):predict(t.right,x,y);}
          function draw(){
            viz.clear();const ctx=viz.ctx,W=viz.width,H=viz.height,halfW=Math.floor(W/2)-10,mg=10;
            const singleTree=buildTree(data,0,6,mkRng(42));
            const trees=[];for(let t=0;t<numTrees;t++){const rng=mkRng(t*137+7);const s=[];for(let i=0;i<data.length;i++)s.push(data[Math.floor(rng()*data.length)]);trees.push(buildTree(s,0,5,rng));}
            const res=3;
            for(let panel=0;panel<2;panel++){const ox=panel===0?mg:halfW+2*mg;
              for(let px=0;px<halfW;px+=res)for(let py=20;py<H-30;py+=res){const x=px/halfW*6.5,y=(1-(py-20)/(H-50))*3.5;let prob=panel===0?predict(singleTree,x,y):trees.reduce((s,tr)=>s+predict(tr,x,y),0)/numTrees;const r=Math.round(248*(1-prob)+88*prob),g=Math.round(81*(1-prob)+166*prob),b=Math.round(73*(1-prob)+255*prob);ctx.fillStyle=`rgba(${r},${g},${b},0.15)`;ctx.fillRect(ox+px,py,res,res);}
              for(const p of data){const sx=ox+(p.x/6.5)*halfW,sy=20+(1-p.y/3.5)*(H-50);ctx.fillStyle=p.c===1?viz.colors.blue:viz.colors.red;ctx.beginPath();ctx.arc(sx,sy,4,0,Math.PI*2);ctx.fill();}
              ctx.strokeStyle=viz.colors.axis;ctx.lineWidth=1;ctx.strokeRect(ox,20,halfW,H-50);
            }
            viz.screenText('Single Tree (depth 6)',mg+halfW/2,12,viz.colors.white,13);
            viz.screenText('Random Forest ('+numTrees+' trees)',halfW+2*mg+halfW/2,12,viz.colors.white,13);
          }
          draw(); return viz;
        }
      }],
      exercises: [
        {question:'With \\(p = 100\\) features, how many does a random forest consider at each split (classification default)?',hint:'Default is \\(m = \\lfloor \\sqrt{p} \\rfloor\\).',solution:'\\(m = \\lfloor \\sqrt{100} \\rfloor = 10\\). Each split considers only 10 of 100 features, decorrelating the trees.'},
        {question:'Explain why OOB error is preferable to training error and more convenient than \\(k\\)-fold cross-validation.',hint:'Consider which data each tree is evaluated on, and the extra computation of CV.',solution:'Training error is optimistic because each tree is evaluated on data it trained on. OOB error evaluates each point using only trees whose bootstrap sample excluded it, providing an honest estimate. Unlike \\(k\\)-fold CV, OOB requires no additional model training; the models already exist, and we just track which trees excluded which points. It is essentially free.'},
        {question:'If one feature dominates and 99 are weak, explain how random forests still leverage the weak features.',hint:'Consider when the strong feature is excluded from the candidate set.',solution:'In bagged trees, every tree sees all 100 features and always splits on the dominant one, making trees highly correlated. In a random forest with \\(m=10\\), the strong feature is included with probability \\(10/100 = 0.1\\), so 90% of the time a given split must use weaker features. This discovers secondary patterns and reduces inter-tree correlation \\(\\rho\\), yielding better variance reduction.'}
      ]
    },
    // ── SECTION 3: Boosting ─────────────────────────────────────────────────
    {
      id: 'ch10-sec03', title: 'Boosting',
      content: `
<h2>Boosting: From Weak to Strong</h2>
<p>While bagging combines models trained in parallel, <strong>boosting</strong> trains models <em>sequentially</em>, with each new model focusing on its predecessors' mistakes.</p>
<div class="env-block definition"><div class="env-title">Definition 10.2 (Weak Learner)</div><div class="env-body">
<p>A <strong>weak learner</strong> is a classifier guaranteed to have error rate strictly less than \\(1/2\\) on any distribution. It is merely better than a coin flip.</p>
</div></div>
<h3>AdaBoost</h3>
<p>Freund and Schapire (1997) introduced AdaBoost, which maintains weights over training examples. Misclassified examples receive higher weights, forcing the next learner to focus on them.</p>
<div class="env-block definition"><div class="env-title">Algorithm 10.3 (AdaBoost)</div><div class="env-body">
<p><strong>Input:</strong> \\(\\{(\\mathbf{x}_i, y_i)\\}_{i=1}^n\\) with \\(y_i \\in \\{-1, +1\\}\\), rounds \\(T\\). Initialize \\(w_i^{(1)} = 1/n\\).</p>
<p>For \\(t = 1, \\dots, T\\):</p>
<ol>
  <li>Train weak learner \\(h_t\\) on weighted data.</li>
  <li>Weighted error: \\(\\epsilon_t = \\sum_{i: h_t(\\mathbf{x}_i) \\neq y_i} w_i^{(t)}\\).</li>
  <li>Learner weight: \\(\\alpha_t = \\frac{1}{2}\\ln\\frac{1 - \\epsilon_t}{\\epsilon_t}\\).</li>
  <li>Update: \\(w_i^{(t+1)} \\propto w_i^{(t)} \\exp(-\\alpha_t y_i h_t(\\mathbf{x}_i))\\).</li>
</ol>
<p><strong>Output:</strong> \\(H(\\mathbf{x}) = \\text{sign}\\!\\left(\\sum_{t=1}^T \\alpha_t h_t(\\mathbf{x})\\right)\\).</p>
</div></div>
<p>Correct predictions multiply the weight by \\(e^{-\\alpha_t} &lt; 1\\) (decrease); misclassifications multiply by \\(e^{\\alpha_t} > 1\\) (increase).</p>
<div class="env-block theorem"><div class="env-title">Theorem 10.1 (AdaBoost Training Error Bound)</div><div class="env-body">
<p>If each learner has edge \\(\\gamma_t = 1/2 - \\epsilon_t > 0\\), the training error satisfies \\(\\text{err} \\leq \\prod_{t=1}^T \\sqrt{1 - 4\\gamma_t^2}\\). If \\(\\gamma_t \\geq \\gamma\\) for all \\(t\\), then \\(\\text{err} \\leq e^{-2T\\gamma^2}\\).</p>
</div></div>
<div class="viz-placeholder" data-viz="viz-adaboost-weights"></div>`,
      visualizations: [{
        id: 'viz-adaboost-weights',
        title: 'AdaBoost: Watch Weights Grow on Misclassified Points',
        description: 'Click "Step" to add one weak learner. Point sizes reflect weights; misclassified points grow larger.',
        setup(container, controls) {
          const viz = new VizEngine(container, {scale:40, originX:280, originY:200});
          function mkRng(s){return function(){s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
          const rng=mkRng(55);const data=[];
          for(let i=0;i<30;i++){const x=rng()*6,y=rng()*4;data.push({x,y,label:(y>2+0.8*Math.sin(x*1.5)+(rng()-0.5)*0.8)?1:-1});}
          const n=data.length;let weights=data.map(()=>1/n),stumps=[],alphas=[],round=0;
          function findBest(){let bF=0,bT=0,bP=1,bE=Infinity;for(let f=0;f<2;f++){const vals=data.map(p=>f===0?p.x:p.y).sort((a,b)=>a-b);for(let i=0;i<vals.length-1;i++){const th=(vals[i]+vals[i+1])/2;for(const pol of[1,-1]){let err=0;for(let j=0;j<n;j++){const v=f===0?data[j].x:data[j].y;if(pol*(v<=th?-1:1)!==data[j].label)err+=weights[j];}if(err<bE){bE=err;bF=f;bT=th;bP=pol;}}}}return{feat:bF,thresh:bT,pol:bP,err:bE};}
          function sPred(s,x,y){return s.pol*((s.feat===0?x:y)<=s.thresh?-1:1);}
          function ensPred(x,y){let sc=0;for(let t=0;t<stumps.length;t++)sc+=alphas[t]*sPred(stumps[t],x,y);return sc;}
          function doStep(){if(round>=15)return;const s=findBest();const eps=Math.max(s.err,1e-10);const alpha=0.5*Math.log((1-eps)/eps);stumps.push(s);alphas.push(alpha);const nW=weights.map((w,i)=>w*Math.exp(-alpha*data[i].label*sPred(s,data[i].x,data[i].y)));const ws=nW.reduce((a,b)=>a+b,0);weights=nW.map(w=>w/ws);round++;}
          const btnRow=document.createElement('div');btnRow.style.cssText='display:flex;gap:6px;';
          VizEngine.createButton(btnRow,'Step',()=>{doStep();draw();});
          VizEngine.createButton(btnRow,'Run 5',()=>{for(let i=0;i<5;i++)doStep();draw();});
          VizEngine.createButton(btnRow,'Reset',()=>{weights=data.map(()=>1/n);stumps=[];alphas=[];round=0;draw();});
          controls.appendChild(btnRow);
          function draw(){
            viz.clear();const ctx=viz.ctx,W=viz.width,H=viz.height;
            const toSx=x=>40+x*(W-60)/6, toSy=y=>20+(1-y/4)*(H-50);
            if(stumps.length>0){const res=4;for(let px=40;px<W-20;px+=res)for(let py=20;py<H-30;py+=res){const x=(px-40)/(W-60)*6,y=(1-(py-20)/(H-50))*4;const sc=ensPred(x,y);const p=1/(1+Math.exp(-2*sc));ctx.fillStyle=p>0.5?`rgba(88,166,255,${(p-0.5)*0.3})`:`rgba(248,81,73,${(0.5-p)*0.3})`;ctx.fillRect(px,py,res,res);}}
            for(let t=0;t<stumps.length;t++){const s=stumps[t];ctx.strokeStyle=viz.colors.text+'44';ctx.lineWidth=1;ctx.setLineDash([3,3]);if(s.feat===0){const sx=toSx(s.thresh);ctx.beginPath();ctx.moveTo(sx,20);ctx.lineTo(sx,H-30);ctx.stroke();}else{const sy=toSy(s.thresh);ctx.beginPath();ctx.moveTo(40,sy);ctx.lineTo(W-20,sy);ctx.stroke();}ctx.setLineDash([]);}
            ctx.strokeStyle=viz.colors.axis;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(40,H-30);ctx.lineTo(W-20,H-30);ctx.stroke();ctx.beginPath();ctx.moveTo(40,20);ctx.lineTo(40,H-30);ctx.stroke();
            const maxW=Math.max(...weights);
            for(let i=0;i<n;i++){const p=data[i],sx=toSx(p.x),sy=toSy(p.y),r=4+(weights[i]/maxW)*14;ctx.fillStyle=p.label===1?viz.colors.blue:viz.colors.red;ctx.globalAlpha=0.7;ctx.beginPath();ctx.arc(sx,sy,r,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;ctx.strokeStyle=p.label===1?viz.colors.blue:viz.colors.red;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(sx,sy,r,0,Math.PI*2);ctx.stroke();}
            const err=stumps.length>0?data.filter(p=>Math.sign(ensPred(p.x,p.y))!==p.label).length/n:0.5;
            viz.screenText('Round: '+round+'  |  Error: '+(err*100).toFixed(1)+'%',W/2,H-10,viz.colors.text,12);
            viz.screenText('AdaBoost (point size = weight)',W/2,10,viz.colors.white,13);
          }
          draw(); return viz;
        }
      }],
      exercises: [
        {question:'A weak learner achieves \\(\\epsilon_t = 0.3\\). Compute \\(\\alpha_t\\) and the weight increase factor for misclassified points.',hint:'\\(\\alpha_t = \\frac{1}{2}\\ln\\frac{1-\\epsilon_t}{\\epsilon_t}\\). Increase factor is \\(e^{\\alpha_t}\\).',solution:'\\(\\alpha_t = \\frac{1}{2}\\ln\\frac{0.7}{0.3} \\approx 0.424\\). Misclassified points are multiplied by \\(e^{0.424} \\approx 1.528\\), a 52.8% weight increase before renormalization.'},
        {question:'What happens if a weak learner has error \\(\\epsilon_t = 0.5\\)? What if \\(\\epsilon_t > 0.5\\)?',hint:'Compute \\(\\alpha_t\\) for these cases.',solution:'If \\(\\epsilon_t = 0.5\\): \\(\\alpha_t = \\frac{1}{2}\\ln(1) = 0\\), so it gets zero weight. If \\(\\epsilon_t > 0.5\\): \\(\\alpha_t < 0\\), which flips the learner\'s predictions (a classifier worse than random becomes useful by negation). Typically the algorithm stops if \\(\\epsilon_t \\geq 0.5\\).'},
        {question:'After \\(T = 10\\) rounds, each with edge \\(\\gamma = 0.1\\) (\\(\\epsilon_t = 0.4\\)), bound the training error.',hint:'Use \\(\\text{err} \\leq e^{-2T\\gamma^2}\\).',solution:'\\(\\text{err} \\leq e^{-2 \\cdot 10 \\cdot 0.01} = e^{-0.2} \\approx 0.819\\). With \\(T = 100\\): \\(e^{-2} \\approx 0.135\\). With \\(T = 500\\): \\(e^{-10} \\approx 4.5 \\times 10^{-5}\\), showing exponential decay.'}
      ]
    },
    // ── SECTION 4: Gradient Boosting ────────────────────────────────────────
    {
      id: 'ch10-sec04', title: 'Gradient Boosting',
      content: `
<h2>Gradient Boosting</h2>
<p><strong>Gradient boosting</strong> (Friedman, 2001) generalizes boosting by viewing it as <em>gradient descent in function space</em>. Each new model is trained to predict the <strong>negative gradient</strong> (pseudo-residuals) of the loss.</p>
<div class="env-block definition"><div class="env-title">Algorithm 10.4 (Gradient Boosting)</div><div class="env-body">
<p><strong>Input:</strong> Loss \\(L(y, F(\\mathbf{x}))\\), stages \\(M\\), learning rate \\(\\nu\\). Initialize \\(F_0(\\mathbf{x}) = \\arg\\min_c \\sum_i L(y_i, c)\\).</p>
<p>For \\(m = 1, \\dots, M\\):</p>
<ol>
  <li>Pseudo-residuals: \\(r_{im} = -\\partial L(y_i, F(\\mathbf{x}_i))/\\partial F(\\mathbf{x}_i)\\big|_{F=F_{m-1}}\\).</li>
  <li>Fit base learner \\(h_m\\) to \\(\\{(\\mathbf{x}_i, r_{im})\\}\\).</li>
  <li>Update: \\(F_m = F_{m-1} + \\nu \\cdot h_m\\).</li>
</ol>
</div></div>
<p>For squared loss, pseudo-residuals are ordinary residuals \\(r_{im} = y_i - F_{m-1}(\\mathbf{x}_i)\\). Each tree literally fits the residuals of the current ensemble.</p>
<h3>Shrinkage</h3>
<p>The learning rate \\(\\nu \\in (0, 1]\\) controls regularization. Smaller \\(\\nu\\) requires more stages but generally yields better generalization by preventing any single tree from dominating.</p>
<div class="env-block remark"><div class="env-title">XGBoost, LightGBM, CatBoost</div><div class="env-body"><p>Modern implementations add regularization to the tree objective: \\(\\mathcal{L} = \\sum_i L(y_i, \\hat{y}_i) + \\sum_m \\Omega(h_m)\\), where \\(\\Omega(h) = \\gamma T + \\frac{1}{2}\\lambda\\|w\\|^2\\). XGBoost uses second-order approximation and histogram-based splitting. LightGBM introduced leaf-wise growth. CatBoost handles categorical features natively.</p></div></div>
<div class="viz-placeholder" data-viz="viz-gb-residuals"></div>`,
      visualizations: [{
        id: 'viz-gb-residuals',
        title: 'Gradient Boosting: Residuals Shrink With Each Stage',
        description: 'Top: ensemble prediction vs data. Bottom: residuals. Click "Add Stage" to fit a new tree to residuals.',
        setup(container, controls) {
          const viz = new VizEngine(container, {scale:40, originX:60, originY:150});
          let stages=0, lr=0.3;
          function mkRng(s){return function(){s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
          const rng=mkRng(99); const trueF=x=>2*Math.sin(x)+0.3*x;
          const data=[];for(let i=0;i<25;i++){const x=rng()*8;data.push({x,y:trueF(x)+(rng()-0.5)*1.5});}data.sort((a,b)=>a.x-b.x);
          function fitTree(pts,depth){if(depth<=0||pts.length<=3){const m=pts.reduce((s,p)=>s+p.y,0)/pts.length;return()=>m;}let bX=pts[0].x,bE=Infinity;for(let i=1;i<pts.length;i++){const th=(pts[i-1].x+pts[i].x)/2,L=pts.filter(p=>p.x<=th),R=pts.filter(p=>p.x>th);if(!L.length||!R.length)continue;const mL=L.reduce((s,p)=>s+p.y,0)/L.length,mR=R.reduce((s,p)=>s+p.y,0)/R.length;const e=L.reduce((s,p)=>s+(p.y-mL)**2,0)+R.reduce((s,p)=>s+(p.y-mR)**2,0);if(e<bE){bE=e;bX=th;}}const L=pts.filter(p=>p.x<=bX),R=pts.filter(p=>p.x>bX);if(!L.length||!R.length){const m=pts.reduce((s,p)=>s+p.y,0)/pts.length;return()=>m;}const lf=fitTree(L,depth-1),rf=fitTree(R,depth-1);return x=>x<=bX?lf(x):rf(x);}
          let treeFns=[];const initP=data.reduce((s,p)=>s+p.y,0)/data.length;
          function curPred(x){let p=initP;for(const f of treeFns)p+=lr*f(x);return p;}
          function addStage(){if(stages>=20)return;const res=data.map(p=>({x:p.x,y:p.y-curPred(p.x)}));treeFns.push(fitTree(res,3));stages++;}
          VizEngine.createSlider(controls,'Learning Rate',0.05,1.0,lr,0.05,v=>{lr=v;treeFns=[];stages=0;draw();});
          const btnRow=document.createElement('div');btnRow.style.cssText='display:flex;gap:6px;';
          VizEngine.createButton(btnRow,'Add Stage',()=>{addStage();draw();});
          VizEngine.createButton(btnRow,'Add 5',()=>{for(let i=0;i<5;i++)addStage();draw();});
          VizEngine.createButton(btnRow,'Reset',()=>{treeFns=[];stages=0;draw();});
          controls.appendChild(btnRow);
          function draw(){
            viz.clear();const ctx=viz.ctx,W=viz.width,H=viz.height,topH=Math.floor(H*0.55),botH=H-topH;
            const toSx=x=>60+x*(W-80)/8, toSyT=y=>15+(1-(y+3)/8)*(topH-30), toSyB=y=>topH+10+(1-(y+3)/6)*(botH-30);
            ctx.strokeStyle=viz.colors.axis;ctx.lineWidth=1;ctx.strokeRect(55,10,W-70,topH-20);
            ctx.strokeStyle=viz.colors.green;ctx.lineWidth=1.5;ctx.setLineDash([5,3]);ctx.beginPath();
            for(let p=0;p<=200;p++){const x=p*8/200,sx=toSx(x),sy=toSyT(trueF(x));p===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}ctx.stroke();ctx.setLineDash([]);
            ctx.strokeStyle=viz.colors.orange;ctx.lineWidth=2.5;ctx.beginPath();
            for(let p=0;p<=200;p++){const x=p*8/200,sx=toSx(x),sy=toSyT(curPred(x));p===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);}ctx.stroke();
            for(const p of data){ctx.fillStyle=viz.colors.blue;ctx.beginPath();ctx.arc(toSx(p.x),toSyT(p.y),4,0,Math.PI*2);ctx.fill();}
            ctx.strokeStyle=viz.colors.axis;ctx.lineWidth=1;ctx.strokeRect(55,topH+5,W-70,botH-20);
            const zY=toSyB(0);ctx.strokeStyle=viz.colors.text+'44';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(55,zY);ctx.lineTo(W-15,zY);ctx.stroke();
            let mse=0;for(const p of data){const r=p.y-curPred(p.x);mse+=r*r;const sx=toSx(p.x),sy=toSyB(r);ctx.strokeStyle=viz.colors.red+'88';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(sx,zY);ctx.lineTo(sx,sy);ctx.stroke();ctx.fillStyle=viz.colors.red;ctx.beginPath();ctx.arc(sx,sy,3.5,0,Math.PI*2);ctx.fill();}mse/=data.length;
            viz.screenText('Prediction (orange) vs Data',W/2,8,viz.colors.white,12,'center','top');
            viz.screenText('Residuals',W/2,topH+3,viz.colors.white,12,'center','top');
            viz.screenText('Stages: '+stages+'  |  LR: '+lr.toFixed(2)+'  |  MSE: '+mse.toFixed(3),W/2,H-5,viz.colors.text,11);
          }
          draw(); return viz;
        }
      }],
      exercises: [
        {question:'For squared loss \\(L(y,F)=\\frac{1}{2}(y-F)^2\\), show the pseudo-residuals are ordinary residuals.',hint:'Differentiate \\(L\\) with respect to \\(F\\) and negate.',solution:'\\(-\\frac{\\partial L}{\\partial F} = -(-(y-F)) = y - F\\). The pseudo-residual at stage \\(m\\) is \\(r_{im} = y_i - F_{m-1}(\\mathbf{x}_i)\\), the ordinary residual.'},
        {question:'Why does a smaller learning rate \\(\\nu\\) typically improve generalization?',hint:'Think about regularization and correcting early mistakes.',solution:'With small \\(\\nu\\), each tree contributes a small fraction of its prediction. This acts as shrinkage regularization, preventing overfitting by ensuring no single tree dominates. Later trees can make finer corrections. Empirically, \\(\\nu \\in [0.01, 0.1]\\) with many trees outperforms \\(\\nu = 1\\) with few trees, at increased computational cost.'},
        {question:'Derive the pseudo-residuals for binary classification with log-loss \\(L(y,F) = \\log(1+e^{-yF})\\) where \\(y \\in \\{-1,+1\\}\\).',hint:'Compute \\(-\\partial L/\\partial F\\).',solution:'\\(-\\frac{\\partial L}{\\partial F} = \\frac{ye^{-yF}}{1+e^{-yF}} = \\frac{y}{1+e^{yF}}\\). Letting \\(p = 1/(1+e^{-F})\\), for \\(y=+1\\) the pseudo-residual is \\(1-p\\), and for \\(y=-1\\) it is \\(-p\\). Each tree learns the gap between the true label and predicted probability.'}
      ]
    },
    // ── SECTION 5: Stacking & Comparison ────────────────────────────────────
    {
      id: 'ch10-sec05', title: 'Stacking & Comparison',
      content: `
<h2>Stacking (Stacked Generalization)</h2>
<p><strong>Stacking</strong> (Wolpert, 1992) trains heterogeneous base models and uses a <strong>meta-learner</strong> to combine their predictions optimally.</p>
<div class="env-block definition"><div class="env-title">Algorithm 10.5 (Stacking)</div><div class="env-body">
<ol>
  <li>Train base learners \\(h_1, \\dots, h_K\\) on \\(\\mathcal{D}\\).</li>
  <li>Generate meta-features via cross-validation: out-of-fold predictions \\(\\hat{y}_i^{(k)}\\) for all \\(i, k\\).</li>
  <li>Train meta-learner on \\(\\{((\\hat{y}_i^{(1)}, \\dots, \\hat{y}_i^{(K)}), y_i)\\}\\).</li>
  <li>Predict: \\(\\hat{y} = \\text{meta}(h_1(\\mathbf{x}), \\dots, h_K(\\mathbf{x}))\\).</li>
</ol>
</div></div>
<div class="env-block remark"><div class="env-title">Why Cross-Validation?</div><div class="env-body"><p>Using in-sample predictions would let an overfitting base model appear perfect and dominate the meta-learner. Out-of-fold predictions provide honest accuracy estimates.</p></div></div>
<h2>When to Use What?</h2>
<table style="width:100%;border-collapse:collapse;font-size:0.9rem;margin:1em 0;">
<thead><tr style="border-bottom:2px solid #30363d;"><th style="text-align:left;padding:6px;">Method</th><th style="padding:6px;">Reduces</th><th style="padding:6px;">Training</th><th style="padding:6px;">Best For</th><th style="padding:6px;">Overfit Risk</th></tr></thead>
<tbody>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">Bagging</td><td>Variance</td><td>Parallel</td><td>High-variance models</td><td>Low</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">Random Forest</td><td>Variance</td><td>Parallel</td><td>General purpose</td><td>Low</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">AdaBoost</td><td>Bias</td><td>Sequential</td><td>Weak learners</td><td>Moderate</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">Gradient Boosting</td><td>Bias + Var</td><td>Sequential</td><td>Tabular data</td><td>Higher</td></tr>
<tr style="border-bottom:1px solid #21262d;"><td style="padding:6px;">Stacking</td><td>Both</td><td>Parallel + meta</td><td>Diverse models</td><td>Moderate</td></tr>
</tbody></table>
<div class="env-block intuition"><div class="env-title">Practical Guidance</div><div class="env-body"><p><strong>Start with a random forest</strong> for a reliable baseline. <strong>Move to gradient boosting</strong> (XGBoost, LightGBM) for best tabular performance with tuning. <strong>Use stacking</strong> to squeeze out the last fraction of a percent. For unstructured data (images, text), deep learning outperforms tree-based ensembles.</p></div></div>
<div class="viz-placeholder" data-viz="viz-ensemble-comparison"></div>`,
      visualizations: [{
        id: 'viz-ensemble-comparison',
        title: 'Ensemble Method Comparison',
        description: 'Compare test accuracy of different ensemble methods. Adjust noise to see how robustness varies.',
        setup(container, controls) {
          const viz = new VizEngine(container, {scale:40, originX:280, originY:200});
          let noise=0.3;
          function mkRng(s){return function(){s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
          VizEngine.createSlider(controls,'Noise',0.0,1.0,noise,0.05,v=>{noise=v;draw();});
          function draw(){
            viz.clear();const ctx=viz.ctx,W=viz.width,H=viz.height;const rng=mkRng(42);
            const train=[],test=[];
            for(let i=0;i<60;i++){const x=rng()*6,y=rng()*4,b=2+Math.sin(x)+(rng()-0.5)*noise*4;train.push({x,y,c:y>b?1:0});}
            const rT=mkRng(999);for(let i=0;i<100;i++){const x=rT()*6,y=rT()*4;test.push({x,y,c:y>2+Math.sin(x)?1:0});}
            function bestStump(d,w){let bF=0,bT=0,bP=1,bE=Infinity;for(let f=0;f<2;f++){const vs=[...new Set(d.map(p=>f===0?p.x:p.y))].sort((a,b)=>a-b);for(let i=0;i<vs.length-1;i++){const th=(vs[i]+vs[i+1])/2;for(const pol of[1,-1]){let e=0;for(let j=0;j<d.length;j++){const v=f===0?d[j].x:d[j].y;if((pol*(v>th?1:0))!==d[j].c)e+=(w?w[j]:1/d.length);}if(e<bE){bE=e;bF=f;bT=th;bP=pol;}}}}return{f:bF,t:bT,p:bP,err:bE};}
            function sp(s,x,y){return s.p*((s.f===0?x:y)>s.t?1:0)>0?1:0;}
            const single=bestStump(train,null);const sAcc=test.filter(p=>sp(single,p.x,p.y)===p.c).length/test.length;
            const bags=[];for(let b=0;b<10;b++){const r=mkRng(b*73+1),s=[];for(let i=0;i<train.length;i++)s.push(train[Math.floor(r()*train.length)]);bags.push(bestStump(s,null));}
            const bAcc=test.filter(p=>{const v=bags.map(s=>sp(s,p.x,p.y));return(v.reduce((a,b)=>a+b,0)>5?1:0)===p.c;}).length/test.length;
            let aw=train.map(()=>1/train.length);const aS=[],aA=[];
            for(let t=0;t<10;t++){const s=bestStump(train,aw);const eps=Math.max(s.err,1e-10);if(eps>=0.5)break;const al=0.5*Math.log((1-eps)/eps);aS.push(s);aA.push(al);const nW=aw.map((w,i)=>w*Math.exp(sp(s,train[i].x,train[i].y)!==train[i].c?al:-al));const ws=nW.reduce((a,b)=>a+b,0);aw=nW.map(w=>w/ws);}
            const adaAcc=test.filter(p=>{const sc=aS.reduce((s,st,t)=>s+aA[t]*(sp(st,p.x,p.y)===1?1:-1),0);return(sc>0?1:0)===p.c;}).length/test.length;
            let gbP=train.map(()=>0.5);const gbM=[];
            for(let m=0;m<10;m++){const res=train.map((p,i)=>({...p,c:p.c-gbP[i]>0?1:0}));const md=bestStump(res,null);gbM.push(md);for(let i=0;i<train.length;i++)gbP[i]+=0.3*(sp(md,train[i].x,train[i].y)-0.5);}
            const gbAcc=test.filter(p=>{let pr=0.5;for(const m of gbM)pr+=0.3*(sp(m,p.x,p.y)-0.5);return(pr>0.5?1:0)===p.c;}).length/test.length;
            const methods=[{name:'Single Model',acc:sAcc,color:viz.colors.text},{name:'Bagging (10)',acc:bAcc,color:viz.colors.teal},{name:'AdaBoost (10)',acc:adaAcc,color:viz.colors.orange},{name:'Grad Boost (10)',acc:gbAcc,color:viz.colors.purple}];
            const bW=80,gap=(W-80-methods.length*bW)/(methods.length+1),cBot=H-50,cTop=50,cH=cBot-cTop;
            for(let pct=0;pct<=100;pct+=20){const y=cBot-(pct/100)*cH;ctx.strokeStyle=viz.colors.grid;ctx.lineWidth=0.5;ctx.beginPath();ctx.moveTo(50,y);ctx.lineTo(W-20,y);ctx.stroke();ctx.fillStyle=viz.colors.text;ctx.font='11px -apple-system,sans-serif';ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(pct+'%',45,y);}
            methods.forEach((m,i)=>{const x=60+gap+i*(bW+gap),bH=m.acc*cH,y=cBot-bH;ctx.fillStyle=m.color+'cc';ctx.beginPath();const r=4;ctx.moveTo(x,cBot);ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.lineTo(x+bW-r,y);ctx.quadraticCurveTo(x+bW,y,x+bW,y+r);ctx.lineTo(x+bW,cBot);ctx.closePath();ctx.fill();ctx.fillStyle=viz.colors.white;ctx.font='bold 13px -apple-system,sans-serif';ctx.textAlign='center';ctx.textBaseline='bottom';ctx.fillText((m.acc*100).toFixed(1)+'%',x+bW/2,y-4);ctx.fillStyle=m.color;ctx.font='11px -apple-system,sans-serif';ctx.textBaseline='top';ctx.fillText(m.name,x+bW/2,cBot+6);});
            viz.screenText('Test Accuracy (noise = '+noise.toFixed(2)+')',W/2,15,viz.colors.white,14);
          }
          draw(); return viz;
        }
      }],
      exercises: [
        {question:'In stacking with linear regression as meta-learner and 3 base models, explain why the meta-learner learns optimal combination weights.',hint:'Write out the meta-learner regression equation.',solution:'The meta-learner fits \\(\\hat{y} = \\beta_0 + \\beta_1 h_1(\\mathbf{x}) + \\beta_2 h_2(\\mathbf{x}) + \\beta_3 h_3(\\mathbf{x})\\). Coefficients \\(\\beta_k\\) become optimal weights minimizing squared error on out-of-fold predictions. Accurate models get large coefficients; redundant or poor models get small or negative ones. The intercept corrects systematic bias.'},
        {question:'A gradient boosting model with \\(M=100, \\nu=0.1\\) achieves 95% accuracy. With \\(M=1000, \\nu=0.01\\) it achieves 96%. Both have total correction \\(M\\nu=10\\). Why does the slower learner do better?',hint:'Think about granularity of corrections.',solution:'The slower learner makes 10x more, smaller corrections. Each tree corrects a smaller residual fraction, giving subsequent trees the chance to refine without overfitting. Smaller steps navigate function space more precisely, analogous to gradient descent with smaller step size finding smoother optima. The increased granularity translates to better generalization.'},
        {question:'Why can random forests not overfit as \\(B\\) grows, while gradient boosting can overfit as \\(M\\) grows?',hint:'Consider how each additional model relates to the ensemble.',solution:'In random forests, each tree is independently trained on a bootstrap sample; adding trees improves averaging and reduces variance. The prediction converges and stabilizes. In gradient boosting, each new tree corrects errors of the current ensemble. With enough stages, it memorizes training data by fitting residuals including noise. Without regularization (early stopping, shrinkage, depth limits), capacity grows unboundedly. Random forests need only "more trees"; gradient boosting requires careful tuning.'}
      ]
    }
  ]
});
