window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
  id: 'ch11', number: 11,
  title: 'K-Means & Clustering',
  subtitle: 'Discovering Structure in Unlabeled Data',
  sections: [
    // ── SECTION 1: K-Means Algorithm ────────────────────────────────────────
    {
      id: 'ch11-sec01', title: 'K-Means Algorithm',
      content: `
<div class="env-block intuition"><div class="env-title">Unsupervised Learning</div><div class="env-body"><p>In all the methods we have studied so far, each training example came with a label. But in many real-world settings, we have data without labels and want to discover natural groupings. Clustering is the quintessential unsupervised learning task: given \\(n\\) points, partition them into \\(K\\) groups so that points within each group are similar and points across groups are different.</p></div></div>

<h2>The K-Means Problem</h2>

<p>K-means is the most widely used clustering algorithm, valued for its simplicity and speed. It seeks to partition \\(n\\) data points \\(\\{\\mathbf{x}_1, \\dots, \\mathbf{x}_n\\}\\) into \\(K\\) clusters \\(C_1, \\dots, C_K\\) by minimizing the <strong>within-cluster sum of squares</strong> (WCSS):</p>

\\[\\min_{C_1,\\dots,C_K} \\sum_{k=1}^K \\sum_{\\mathbf{x}_i \\in C_k} \\|\\mathbf{x}_i - \\boldsymbol{\\mu}_k\\|^2,\\]

<p>where \\(\\boldsymbol{\\mu}_k = \\frac{1}{|C_k|}\\sum_{\\mathbf{x}_i \\in C_k} \\mathbf{x}_i\\) is the centroid of cluster \\(k\\).</p>

<div class="env-block remark"><div class="env-title">NP-Hardness</div><div class="env-body"><p>Finding the globally optimal K-means solution is NP-hard even for \\(K = 2\\). Lloyd's algorithm finds a local optimum, which is often good enough in practice.</p></div></div>

<div class="env-block definition"><div class="env-title">Algorithm 11.1 (Lloyd's Algorithm for K-Means)</div><div class="env-body">
<p><strong>Input:</strong> Data \\(\\{\\mathbf{x}_i\\}_{i=1}^n\\), number of clusters \\(K\\).</p>
<p><strong>Initialize:</strong> Choose \\(K\\) initial centroids \\(\\boldsymbol{\\mu}_1, \\dots, \\boldsymbol{\\mu}_K\\).</p>
<p><strong>Repeat</strong> until convergence:</p>
<ol>
  <li><strong>Assign:</strong> For each \\(\\mathbf{x}_i\\), set \\(c_i = \\arg\\min_k \\|\\mathbf{x}_i - \\boldsymbol{\\mu}_k\\|^2\\).</li>
  <li><strong>Update:</strong> For each \\(k\\), set \\(\\boldsymbol{\\mu}_k = \\frac{1}{|C_k|}\\sum_{i: c_i = k} \\mathbf{x}_i\\).</li>
</ol>
</div></div>

<div class="env-block theorem"><div class="env-title">Proposition 11.1 (Convergence)</div><div class="env-body">
<p>Lloyd's algorithm monotonically decreases the WCSS objective at each step and converges in a finite number of iterations. However, it may converge to a local minimum that is far from the global optimum.</p>
</div></div>

<p>The assignment step minimizes the objective over cluster assignments (holding centroids fixed), and the update step minimizes over centroids (holding assignments fixed). Since each step can only decrease or maintain the objective, and there are finitely many partitions, the algorithm must converge.</p>

<div class="env-block remark"><div class="env-title">Sensitivity to Initialization</div><div class="env-body"><p>K-means is notoriously sensitive to the choice of initial centroids. Bad initialization can lead to poor local optima. A common practice is to run the algorithm multiple times with different random initializations and keep the result with the lowest WCSS.</p></div></div>

<div class="viz-placeholder" data-viz="viz-kmeans-steps"></div>
`,
      visualizations: [{
        id: 'viz-kmeans-steps',
        title: 'K-Means Step-by-Step',
        description: 'Click "Step" to alternate between assignment (coloring points) and centroid update (moving centroids). Watch the algorithm converge.',
        setup(container, controls) {
          const viz = new VizEngine(container, {scale:40, originX:280, originY:200});
          function mkRng(s){return function(){s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
          let K=3, seed=42;
          const clusterColors = ['#58a6ff','#f0883e','#3fb950','#bc8cff','#f778ba','#d29922','#f85149'];

          function generateData(s) {
            const rng=mkRng(s); const pts=[];
            const centers=[[2,5],[5,2],[8,6],[3,8],[7,3]];
            for(let k=0;k<Math.min(K+1,5);k++) for(let i=0;i<15;i++) pts.push({x:centers[k][0]+(rng()-0.5)*3,y:centers[k][1]+(rng()-0.5)*3,c:-1});
            return pts;
          }

          let data=generateData(seed);
          let centroids=[], assignments=[], stepNum=0, phase='init', converged=false;

          function initCentroids(){
            const rng=mkRng(seed*7+3);centroids=[];
            for(let k=0;k<K;k++){const idx=Math.floor(rng()*data.length);centroids.push({x:data[idx].x,y:data[idx].y});}
            assignments=data.map(()=>-1);stepNum=0;phase='assign';converged=false;
          }

          function doAssign(){
            let changed=false;
            for(let i=0;i<data.length;i++){
              let best=0,bestD=Infinity;
              for(let k=0;k<K;k++){const dx=data[i].x-centroids[k].x,dy=data[i].y-centroids[k].y,d=dx*dx+dy*dy;if(d<bestD){bestD=d;best=k;}}
              if(assignments[i]!==best){assignments[i]=best;changed=true;}
            }
            if(!changed)converged=true;
            phase='update';
          }

          function doUpdate(){
            for(let k=0;k<K;k++){
              const pts=data.filter((_,i)=>assignments[i]===k);
              if(pts.length>0){centroids[k]={x:pts.reduce((s,p)=>s+p.x,0)/pts.length,y:pts.reduce((s,p)=>s+p.y,0)/pts.length};}
            }
            phase='assign';stepNum++;
          }

          VizEngine.createSlider(controls,'K',2,6,K,1,v=>{K=Math.round(v);data=generateData(seed);initCentroids();draw();});
          const btnRow=document.createElement('div');btnRow.style.cssText='display:flex;gap:6px;';
          VizEngine.createButton(btnRow,'Step',()=>{if(converged)return;if(phase==='assign')doAssign();else doUpdate();draw();});
          VizEngine.createButton(btnRow,'Run to End',()=>{for(let i=0;i<100&&!converged;i++){if(phase==='assign')doAssign();else doUpdate();}draw();});
          VizEngine.createButton(btnRow,'Reset',()=>{seed=Math.floor(Math.random()*10000);data=generateData(seed);initCentroids();draw();});
          controls.appendChild(btnRow);
          initCentroids();

          function draw(){
            viz.clear();const ctx=viz.ctx,W=viz.width,H=viz.height;
            const toSx=x=>30+x*(W-60)/10, toSy=y=>H-30-(y)*(H-60)/10;

            // Grid
            ctx.strokeStyle=viz.colors.grid;ctx.lineWidth=0.5;
            for(let g=0;g<=10;g+=2){const sx=toSx(g),sy=toSy(g);ctx.beginPath();ctx.moveTo(sx,20);ctx.lineTo(sx,H-30);ctx.stroke();ctx.beginPath();ctx.moveTo(30,sy);ctx.lineTo(W-30,sy);ctx.stroke();}

            // Draw Voronoi-like regions (simple per-pixel)
            if(assignments.some(a=>a>=0)){
              const res=5;
              for(let px=30;px<W-30;px+=res)for(let py=20;py<H-30;py+=res){
                const x=(px-30)/(W-60)*10,y=(H-30-py)/(H-60)*10;
                let best=0,bestD=Infinity;
                for(let k=0;k<K;k++){const dx=x-centroids[k].x,dy=y-centroids[k].y,d=dx*dx+dy*dy;if(d<bestD){bestD=d;best=k;}}
                ctx.fillStyle=clusterColors[best%7]+'11';ctx.fillRect(px,py,res,res);
              }
            }

            // Data points
            for(let i=0;i<data.length;i++){
              const p=data[i],sx=toSx(p.x),sy=toSy(p.y);
              const col=assignments[i]>=0?clusterColors[assignments[i]%7]:viz.colors.text;
              ctx.fillStyle=col;ctx.beginPath();ctx.arc(sx,sy,5,0,Math.PI*2);ctx.fill();
            }

            // Centroids
            for(let k=0;k<K;k++){
              const c=centroids[k],sx=toSx(c.x),sy=toSy(c.y);
              ctx.strokeStyle=clusterColors[k%7];ctx.lineWidth=2.5;
              ctx.beginPath();ctx.moveTo(sx-8,sy-8);ctx.lineTo(sx+8,sy+8);ctx.stroke();
              ctx.beginPath();ctx.moveTo(sx+8,sy-8);ctx.lineTo(sx-8,sy+8);ctx.stroke();
              ctx.fillStyle=clusterColors[k%7];ctx.beginPath();ctx.arc(sx,sy,3,0,Math.PI*2);ctx.fill();
            }

            // WCSS
            let wcss=0;
            for(let i=0;i<data.length;i++){if(assignments[i]>=0){const c=centroids[assignments[i]];wcss+=(data[i].x-c.x)**2+(data[i].y-c.y)**2;}}

            const statusText=converged?'Converged!':'Next: '+(phase==='assign'?'Assign points':'Update centroids');
            viz.screenText('K-Means (K='+K+')  |  Iteration: '+stepNum+'  |  '+statusText,W/2,12,converged?viz.colors.green:viz.colors.white,12);
            viz.screenText('WCSS: '+wcss.toFixed(1),W/2,H-10,viz.colors.text,12);
          }
          draw(); return viz;
        }
      }],
      exercises: [
        {question:'Show that the centroid \\(\\boldsymbol{\\mu}_k = \\frac{1}{|C_k|}\\sum_{\\mathbf{x}_i \\in C_k}\\mathbf{x}_i\\) minimizes \\(\\sum_{\\mathbf{x}_i \\in C_k}\\|\\mathbf{x}_i - \\mathbf{c}\\|^2\\) over all \\(\\mathbf{c}\\).',hint:'Take the gradient with respect to \\(\\mathbf{c}\\) and set it to zero.',solution:'Let \\(f(\\mathbf{c}) = \\sum_{i \\in C_k}\\|\\mathbf{x}_i - \\mathbf{c}\\|^2\\). Then \\(\\nabla_\\mathbf{c} f = -2\\sum_{i \\in C_k}(\\mathbf{x}_i - \\mathbf{c}) = 0\\), giving \\(|C_k|\\mathbf{c} = \\sum_{i \\in C_k}\\mathbf{x}_i\\), so \\(\\mathbf{c} = \\boldsymbol{\\mu}_k\\). The Hessian is \\(2|C_k|\\mathbf{I} \\succ 0\\), confirming a minimum.'},
        {question:'Why can K-means converge to different solutions depending on initialization? Give a simple example with 4 points and \\(K=2\\).',hint:'Place 4 points at the corners of a square and consider two different initial centroid placements.',solution:'Consider points at \\((0,0), (0,1), (1,0), (1,1)\\) with \\(K=2\\). If centroids start at \\((0,0.5)\\) and \\((1,0.5)\\), we get a vertical split: \\(\\{(0,0),(0,1)\\}\\) and \\(\\{(1,0),(1,1)\\}\\) with WCSS = 2. If centroids start at \\((0.5,0)\\) and \\((0.5,1)\\), we get a horizontal split with the same WCSS = 2. Both are valid local optima. With asymmetric data, different initializations can yield partitions with genuinely different WCSS values.'},
        {question:'What happens if K-means is run with \\(K\\) larger than the true number of clusters? What about \\(K=1\\) and \\(K=n\\)?',hint:'Consider the WCSS in the extreme cases.',solution:'With \\(K=1\\), all points are in one cluster and \\(\\text{WCSS} = \\sum_i\\|\\mathbf{x}_i - \\bar{\\mathbf{x}}\\|^2\\), the total variance. With \\(K=n\\), each point is its own cluster and WCSS = 0. As \\(K\\) increases, WCSS monotonically decreases. If \\(K\\) exceeds the true number of clusters, K-means splits natural clusters unnecessarily, reducing WCSS but not reflecting true structure. The "elbow method" looks for a \\(K\\) where the WCSS decrease slows, indicating diminishing returns.'}
      ]
    },
    // ── SECTION 2: K-Means Variants ─────────────────────────────────────────
    {
      id: 'ch11-sec02', title: 'K-Means Variants',
      content: `
<h2>K-Means++ Initialization</h2>

<p>Random initialization can lead to poor clusterings. Arthur and Vassilvitskii (2007) proposed <strong>K-means++</strong>, which spreads initial centroids apart by choosing each new centroid with probability proportional to its squared distance from the nearest existing centroid.</p>

<div class="env-block definition"><div class="env-title">Algorithm 11.2 (K-Means++ Initialization)</div><div class="env-body">
<ol>
  <li>Choose \\(\\boldsymbol{\\mu}_1\\) uniformly at random from the data.</li>
  <li>For \\(k = 2, \\dots, K\\): choose \\(\\boldsymbol{\\mu}_k = \\mathbf{x}_j\\) with probability \\(\\frac{D(\\mathbf{x}_j)^2}{\\sum_i D(\\mathbf{x}_i)^2}\\), where \\(D(\\mathbf{x}_j)\\) is the distance from \\(\\mathbf{x}_j\\) to its nearest existing centroid.</li>
  <li>Proceed with standard K-means.</li>
</ol>
</div></div>

<div class="env-block theorem"><div class="env-title">Theorem 11.1 (K-Means++ Guarantee)</div><div class="env-body">
<p>K-means++ initialization guarantees that the expected WCSS is at most \\(O(\\log K)\\) times the optimal WCSS. In practice, it consistently finds solutions close to optimal.</p>
</div></div>

<h3>Other Variants</h3>

<p><strong>Mini-batch K-means</strong> updates centroids using small random subsets of data at each iteration rather than the full dataset, trading a small accuracy loss for much faster computation on large datasets.</p>

<p><strong>K-medoids</strong> (PAM algorithm) uses actual data points as cluster representatives instead of computed means. This makes it robust to outliers (the median is more robust than the mean) and applicable to any distance metric, not just Euclidean.</p>

<div class="env-block remark"><div class="env-title">Choosing K</div><div class="env-body"><p>The <strong>elbow method</strong> plots WCSS vs. \\(K\\) and looks for a bend. The <strong>silhouette score</strong> measures how similar each point is to its own cluster vs. neighboring clusters (values near +1 are good). The <strong>gap statistic</strong> compares WCSS to that of a uniform reference distribution.</p></div></div>

<div class="viz-placeholder" data-viz="viz-kmeans-pp"></div>
`,
      visualizations: [{
        id: 'viz-kmeans-pp',
        title: 'K-Means++ vs. Random Initialization',
        description: 'Left: random init (centroids may clump). Right: K-means++ (centroids spread out). Click "New Data" to regenerate.',
        setup(container, controls) {
          const viz = new VizEngine(container, {scale:40, originX:280, originY:200});
          let seed=42, K=3;
          function mkRng(s){return function(){s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
          const colors=['#58a6ff','#f0883e','#3fb950','#bc8cff','#f778ba','#d29922'];

          function genData(s){const rng=mkRng(s),pts=[];const cens=[[2,2],[5,7],[8,3],[3,8],[7,7]];for(let k=0;k<Math.min(K+1,5);k++)for(let i=0;i<12;i++)pts.push({x:cens[k][0]+(rng()-0.5)*2.5,y:cens[k][1]+(rng()-0.5)*2.5});return pts;}

          function randomInit(data,K,rng){const c=[];for(let k=0;k<K;k++)c.push({...data[Math.floor(rng()*data.length)]});return c;}

          function kppInit(data,K,rng){
            const c=[{...data[Math.floor(rng()*data.length)]}];
            for(let k=1;k<K;k++){
              const dists=data.map(p=>{let mn=Infinity;for(const ci of c){const dx=p.x-ci.x,dy=p.y-ci.y;mn=Math.min(mn,dx*dx+dy*dy);}return mn;});
              const total=dists.reduce((a,b)=>a+b,0);
              let r=rng()*total,acc=0;
              for(let i=0;i<data.length;i++){acc+=dists[i];if(acc>=r){c.push({...data[i]});break;}}
            }
            return c;
          }

          function runKmeans(data,centroids,K){
            for(let iter=0;iter<30;iter++){
              const assign=data.map(p=>{let b=0,bd=Infinity;for(let k=0;k<K;k++){const dx=p.x-centroids[k].x,dy=p.y-centroids[k].y,d=dx*dx+dy*dy;if(d<bd){bd=d;b=k;}}return b;});
              let changed=false;
              for(let k=0;k<K;k++){const pts=data.filter((_,i)=>assign[i]===k);if(pts.length>0){const nx=pts.reduce((s,p)=>s+p.x,0)/pts.length,ny=pts.reduce((s,p)=>s+p.y,0)/pts.length;if(Math.abs(centroids[k].x-nx)>1e-6||Math.abs(centroids[k].y-ny)>1e-6)changed=true;centroids[k]={x:nx,y:ny};}}
              if(!changed)break;
            }
            const assign=data.map(p=>{let b=0,bd=Infinity;for(let k=0;k<K;k++){const dx=p.x-centroids[k].x,dy=p.y-centroids[k].y,d=dx*dx+dy*dy;if(d<bd){bd=d;b=k;}}return b;});
            let wcss=0;for(let i=0;i<data.length;i++){const c=centroids[assign[i]];wcss+=(data[i].x-c.x)**2+(data[i].y-c.y)**2;}
            return{assign,wcss};
          }

          VizEngine.createSlider(controls,'K',2,5,K,1,v=>{K=Math.round(v);draw();});
          VizEngine.createButton(controls,'New Data',()=>{seed=Math.floor(Math.random()*10000);draw();});

          function draw(){
            viz.clear();const ctx=viz.ctx,W=viz.width,H=viz.height,halfW=Math.floor(W/2)-10,mg=10;
            const data=genData(seed);
            const rngR=mkRng(seed*3+1),rngP=mkRng(seed*5+2);
            const randC=randomInit(data,K,rngR),ppC=kppInit(data,K,rngP);
            const resR=runKmeans(data,[...randC.map(c=>({...c}))],K);
            const resP=runKmeans(data,[...ppC.map(c=>({...c}))],K);

            for(let panel=0;panel<2;panel++){
              const ox=panel===0?mg:halfW+2*mg;
              const res=panel===0?resR:resP;
              const toSx=x=>ox+x*(halfW)/10, toSy=y=>H-30-(y)*(H-60)/10;

              // Colored points
              for(let i=0;i<data.length;i++){
                const p=data[i],sx=toSx(p.x),sy=toSy(p.y);
                ctx.fillStyle=colors[res.assign[i]%6];ctx.beginPath();ctx.arc(sx,sy,4,0,Math.PI*2);ctx.fill();
              }

              // Final centroids
              const finalC=panel===0?randC:ppC;
              // Re-run to get final centroids (use result assignments to compute)
              for(let k=0;k<K;k++){
                const pts=data.filter((_,i)=>res.assign[i]===k);
                if(pts.length>0){
                  const cx=pts.reduce((s,p)=>s+p.x,0)/pts.length,cy=pts.reduce((s,p)=>s+p.y,0)/pts.length;
                  const sx=toSx(cx),sy=toSy(cy);
                  ctx.strokeStyle='#fff';ctx.lineWidth=2.5;
                  ctx.beginPath();ctx.moveTo(sx-7,sy-7);ctx.lineTo(sx+7,sy+7);ctx.stroke();
                  ctx.beginPath();ctx.moveTo(sx+7,sy-7);ctx.lineTo(sx-7,sy+7);ctx.stroke();
                }
              }

              // Initial centroids (hollow circles)
              const initC=panel===0?randC:ppC;
              for(let k=0;k<K;k++){
                const sx=toSx(initC[k].x),sy=toSy(initC[k].y);
                ctx.strokeStyle=colors[k%6];ctx.lineWidth=2;ctx.setLineDash([3,3]);
                ctx.beginPath();ctx.arc(sx,sy,8,0,Math.PI*2);ctx.stroke();
                ctx.setLineDash([]);
              }

              ctx.strokeStyle=viz.colors.axis;ctx.lineWidth=1;ctx.strokeRect(ox,20,halfW,H-50);
              viz.screenText('WCSS: '+res.wcss.toFixed(1),ox+halfW/2,H-8,viz.colors.text,11);
            }
            viz.screenText('Random Init',mg+halfW/2,12,viz.colors.white,13);
            viz.screenText('K-Means++ Init',halfW+2*mg+halfW/2,12,viz.colors.white,13);
            viz.screenText('Dashed circles = initial centroids, X = final centroids',W/2,32,viz.colors.text,10);
          }
          draw(); return viz;
        }
      }],
      exercises: [
        {question:'In K-means++ initialization, after the first centroid is chosen, what is the probability of selecting a point that is very far from it as the second centroid? Why is this desirable?',hint:'The probability is proportional to \\(D(\\mathbf{x})^2\\). Consider what happens in the extreme.',solution:'A point with distance \\(D\\) from the first centroid is selected with probability proportional to \\(D^2\\). Very far points have much higher probability of being selected. This is desirable because it spreads centroids apart, ensuring they start in different regions of the data. If all centroids initialized in the same cluster, the algorithm would likely converge to a poor local optimum.'},
        {question:'K-medoids uses actual data points as representatives. Why is this more robust to outliers than K-means?',hint:'Consider the influence of an extreme outlier on the mean vs. choosing the most central data point.',solution:'K-means computes centroids as the mean of cluster members. An extreme outlier pulls the mean substantially toward it. K-medoids selects the data point that minimizes the sum of distances to all other cluster members; an outlier, being distant from most points, is never selected as the medoid. The medoid is essentially a multivariate median, which is robust to outliers.'},
        {question:'Show that the silhouette score \\(s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}\\) lies in \\([-1, 1]\\), where \\(a(i)\\) is the average distance to same-cluster points and \\(b(i)\\) is the average distance to the nearest other cluster.',hint:'Consider the cases \\(a(i) < b(i)\\), \\(a(i) > b(i)\\), and \\(a(i) = b(i)\\).',solution:'If \\(a(i) < b(i)\\): \\(s(i) = (b(i)-a(i))/b(i) = 1 - a(i)/b(i) \\in (0, 1]\\). If \\(a(i) > b(i)\\): \\(s(i) = (b(i)-a(i))/a(i) = b(i)/a(i) - 1 \\in [-1, 0)\\). If \\(a(i) = b(i)\\): \\(s(i) = 0\\). Since \\(a(i), b(i) \\geq 0\\), the score is always in \\([-1, 1]\\). Values near +1 mean the point is well-clustered; near -1 means it is likely in the wrong cluster.'}
      ]
    },
    // ── SECTION 3: Hierarchical Clustering ──────────────────────────────────
    {
      id: 'ch11-sec03', title: 'Hierarchical Clustering',
      content: `
<h2>Hierarchical Clustering</h2>

<p>Unlike K-means, which requires specifying \\(K\\) in advance, <strong>hierarchical clustering</strong> builds a nested sequence of clusterings for all values of \\(K\\) from \\(n\\) down to 1. The result is a tree structure called a <strong>dendrogram</strong>.</p>

<div class="env-block definition"><div class="env-title">Algorithm 11.3 (Agglomerative Clustering)</div><div class="env-body">
<p><strong>Input:</strong> Data \\(\\{\\mathbf{x}_i\\}_{i=1}^n\\), linkage criterion.</p>
<ol>
  <li>Start with \\(n\\) clusters, each containing one point.</li>
  <li><strong>Repeat</strong> until one cluster remains:</li>
  <li>&nbsp;&nbsp;&nbsp;&nbsp;Find the two closest clusters according to the linkage criterion.</li>
  <li>&nbsp;&nbsp;&nbsp;&nbsp;Merge them into a single cluster.</li>
  <li>&nbsp;&nbsp;&nbsp;&nbsp;Record the merge distance (height in dendrogram).</li>
</ol>
</div></div>

<h3>Linkage Criteria</h3>

<p>The choice of linkage defines "distance between clusters":</p>
<ul>
  <li><strong>Single linkage:</strong> \\(d(A,B) = \\min_{\\mathbf{a} \\in A, \\mathbf{b} \\in B}\\|\\mathbf{a} - \\mathbf{b}\\|\\). Tends to produce long, "chaining" clusters.</li>
  <li><strong>Complete linkage:</strong> \\(d(A,B) = \\max_{\\mathbf{a} \\in A, \\mathbf{b} \\in B}\\|\\mathbf{a} - \\mathbf{b}\\|\\). Produces compact, spherical clusters.</li>
  <li><strong>Average linkage:</strong> \\(d(A,B) = \\frac{1}{|A||B|}\\sum_{\\mathbf{a} \\in A}\\sum_{\\mathbf{b} \\in B}\\|\\mathbf{a} - \\mathbf{b}\\|\\). A compromise between single and complete.</li>
  <li><strong>Ward's method:</strong> Merges the pair that causes the smallest increase in total WCSS. Often produces the most balanced clusters.</li>
</ul>

<div class="env-block remark"><div class="env-title">Reading a Dendrogram</div><div class="env-body"><p>The y-axis of a dendrogram shows the merge distance. To obtain \\(K\\) clusters, draw a horizontal line that cuts exactly \\(K\\) branches. Points in the same branch below the cut belong to the same cluster. The height of each merge indicates how dissimilar the merged clusters were; large jumps suggest natural cluster boundaries.</p></div></div>

<div class="env-block remark"><div class="env-title">Complexity</div><div class="env-body"><p>Agglomerative clustering has time complexity \\(O(n^3)\\) in general and \\(O(n^2 \\log n)\\) with efficient data structures. This makes it impractical for very large datasets, where K-means or DBSCAN are preferred.</p></div></div>

<div class="viz-placeholder" data-viz="viz-dendrogram"></div>
`,
      visualizations: [{
        id: 'viz-dendrogram',
        title: 'Hierarchical Clustering Dendrogram',
        description: 'Adjust the cut height to see how many clusters result. The dendrogram shows merge distances.',
        setup(container, controls) {
          const viz = new VizEngine(container, {scale:40, originX:280, originY:200});
          let cutHeight=5.0;
          function mkRng(s){return function(){s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
          const rng=mkRng(42);
          const colors=['#58a6ff','#f0883e','#3fb950','#bc8cff','#f778ba','#d29922','#f85149','#3fb9a0'];

          // Generate clustered data
          const data=[];
          const trueCenters=[[2,2],[6,7],[9,3]];
          for(const c of trueCenters)for(let i=0;i<6;i++)data.push({x:c[0]+(rng()-0.5)*2,y:c[1]+(rng()-0.5)*2});

          // Perform agglomerative clustering (average linkage)
          const n=data.length;
          const dist=(a,b)=>Math.sqrt((a.x-b.x)**2+(a.y-b.y)**2);

          // Build merge history
          let clusters=data.map((_,i)=>[i]);
          const merges=[]; // {i, j, height, newCluster}

          function clusterDist(A,B){
            let sum=0,count=0;
            for(const a of A)for(const b of B){sum+=dist(data[a],data[b]);count++;}
            return sum/count;
          }

          const clusterList=[...clusters];
          while(clusterList.length>1){
            let bestI=0,bestJ=1,bestD=Infinity;
            for(let i=0;i<clusterList.length;i++)for(let j=i+1;j<clusterList.length;j++){
              const d=clusterDist(clusterList[i],clusterList[j]);
              if(d<bestD){bestD=d;bestI=i;bestJ=j;}
            }
            const merged=[...clusterList[bestI],...clusterList[bestJ]];
            merges.push({a:clusterList[bestI].slice(),b:clusterList[bestJ].slice(),height:bestD,merged:merged.slice()});
            clusterList.splice(bestJ,1);clusterList.splice(bestI,1);
            clusterList.push(merged);
          }

          const maxH=merges.length>0?merges[merges.length-1].height:1;
          VizEngine.createSlider(controls,'Cut Height',0.5,maxH*1.1,cutHeight,0.1,v=>{cutHeight=v;draw();});

          function getClusterAssignments(){
            // Replay merges up to cutHeight
            const assign=data.map((_,i)=>i);
            for(const m of merges){
              if(m.height>cutHeight)break;
              const target=assign[m.a[0]];
              const replace=assign[m.b[0]];
              for(let i=0;i<n;i++)if(assign[i]===replace)assign[i]=target;
            }
            // Renumber
            const unique=[...new Set(assign)];
            return assign.map(a=>unique.indexOf(a));
          }

          function draw(){
            viz.clear();const ctx=viz.ctx,W=viz.width,H=viz.height;
            const dendH=Math.floor(H*0.55)-10, scatterH=H-dendH-20;
            const assign=getClusterAssignments();
            const nClusters=new Set(assign).size;

            // ── Dendrogram (top) ──
            // Compute leaf order via merge tree
            function getLeafOrder(indices){
              if(indices.length<=1)return indices;
              for(const m of merges){
                if(m.merged.length===indices.length && m.merged.every(x=>indices.includes(x))){
                  return[...getLeafOrder(m.a),...getLeafOrder(m.b)];
                }
              }
              return indices;
            }
            const leafOrder=getLeafOrder(data.map((_,i)=>i));
            const leafX={};leafOrder.forEach((idx,pos)=>{leafX[idx]=30+(pos+0.5)*(W-60)/n;});

            // Draw merges
            function getMergeX(indices){return indices.reduce((s,i)=>s+leafX[i],0)/indices.length;}
            const hScale=(dendH-30)/maxH;

            for(const m of merges){
              const x1=getMergeX(m.a),x2=getMergeX(m.b);
              const yM=dendH+10-m.height*hScale;
              // Find heights of children
              const hA=merges.find(mm=>mm.merged.length===m.a.length&&mm.merged.every(x=>m.a.includes(x)));
              const hB=merges.find(mm=>mm.merged.length===m.b.length&&mm.merged.every(x=>m.b.includes(x)));
              const yA=hA?dendH+10-hA.height*hScale:dendH+10;
              const yB=hB?dendH+10-hB.height*hScale:dendH+10;

              ctx.strokeStyle=m.height<=cutHeight?viz.colors.white:viz.colors.text+'66';
              ctx.lineWidth=1.5;
              ctx.beginPath();ctx.moveTo(x1,yA);ctx.lineTo(x1,yM);ctx.lineTo(x2,yM);ctx.lineTo(x2,yB);ctx.stroke();
            }

            // Cut line
            const cutY=dendH+10-cutHeight*hScale;
            ctx.strokeStyle=viz.colors.red;ctx.lineWidth=1.5;ctx.setLineDash([6,4]);
            ctx.beginPath();ctx.moveTo(20,cutY);ctx.lineTo(W-20,cutY);ctx.stroke();ctx.setLineDash([]);
            ctx.fillStyle=viz.colors.red;ctx.font='11px -apple-system,sans-serif';ctx.textAlign='left';
            ctx.fillText('cut = '+cutHeight.toFixed(1)+' ('+nClusters+' clusters)',W-150,cutY-5);

            // Leaf points at bottom of dendrogram
            for(let i=0;i<n;i++){
              ctx.fillStyle=colors[assign[i]%8];
              ctx.beginPath();ctx.arc(leafX[i],dendH+10,4,0,Math.PI*2);ctx.fill();
            }

            // ── Scatter plot (bottom) ──
            const scatterTop=dendH+25;
            ctx.strokeStyle=viz.colors.axis;ctx.lineWidth=1;ctx.strokeRect(30,scatterTop,W-60,scatterH-10);
            const toSx=x=>30+x*(W-60)/11, toSy=y=>scatterTop+scatterH-10-(y)*(scatterH-20)/9;
            for(let i=0;i<n;i++){
              const sx=toSx(data[i].x),sy=toSy(data[i].y);
              ctx.fillStyle=colors[assign[i]%8];ctx.beginPath();ctx.arc(sx,sy,5,0,Math.PI*2);ctx.fill();
            }

            viz.screenText('Dendrogram (average linkage)',W/2,8,viz.colors.white,13);
          }
          draw(); return viz;
        }
      }],
      exercises: [
        {question:'Explain why single linkage tends to produce "chain-like" clusters while complete linkage produces compact ones.',hint:'Consider what determines the merge: the closest pair vs. the farthest pair.',solution:'Single linkage merges the two clusters with the smallest minimum inter-cluster distance. Two elongated clusters connected by a single pair of close points will merge, even if most points are far apart, producing long chains. Complete linkage merges clusters with the smallest maximum distance, so it will not merge two clusters until every point in one is close to every point in the other. This forces compact, roughly spherical clusters.'},
        {question:'A dendrogram shows merges at heights 1.2, 1.5, 2.0, 5.1, 5.3. How many clusters should you choose, and why?',hint:'Look for the largest gap between consecutive merge heights.',solution:'The merge heights are 1.2, 1.5, 2.0, 5.1, 5.3. The largest gap is between 2.0 and 5.1 (a gap of 3.1). Cutting anywhere in this gap (e.g., at height 3.5) gives 3 clusters. The large gap indicates that merging from 3 to 2 clusters requires combining groups that are much farther apart, suggesting 3 is the natural number of clusters.'},
        {question:'What is the time complexity of agglomerative clustering with \\(n\\) points? Why is it impractical for \\(n = 10^6\\)?',hint:'At each step, we must find the closest pair among all remaining clusters.',solution:'With \\(n\\) points, there are \\(n-1\\) merge steps. Naively, finding the closest pair requires \\(O(n^2)\\) comparisons, giving \\(O(n^3)\\) total. With a priority queue, this improves to \\(O(n^2 \\log n)\\). For \\(n = 10^6\\), even \\(O(n^2)\\) means \\(10^{12}\\) distance computations, which is infeasible. K-means (\\(O(nKT)\\)) or DBSCAN (\\(O(n \\log n)\\) with spatial indexing) are far more practical at this scale.'}
      ]
    },
    // ── SECTION 4: Density-Based Clustering ─────────────────────────────────
    {
      id: 'ch11-sec04', title: 'Density-Based Clustering',
      content: `
<h2>DBSCAN: Density-Based Spatial Clustering</h2>

<p>K-means and hierarchical clustering struggle with clusters of arbitrary shape. <strong>DBSCAN</strong> (Ester et al., 1996) takes a fundamentally different approach: clusters are dense regions of points separated by sparse regions.</p>

<div class="env-block definition"><div class="env-title">Definition 11.1 (Core, Border, and Noise Points)</div><div class="env-body">
<p>Given parameters \\(\\varepsilon > 0\\) (neighborhood radius) and \\(\\text{minPts}\\) (minimum neighbors):</p>
<ul>
  <li>A point \\(\\mathbf{x}\\) is a <strong>core point</strong> if \\(|N_\\varepsilon(\\mathbf{x})| \\geq \\text{minPts}\\), where \\(N_\\varepsilon(\\mathbf{x}) = \\{\\mathbf{x}_j : \\|\\mathbf{x}_j - \\mathbf{x}\\| \\leq \\varepsilon\\}\\).</li>
  <li>A point is a <strong>border point</strong> if it is not a core point but falls within \\(\\varepsilon\\) of a core point.</li>
  <li>A point is a <strong>noise point</strong> if it is neither core nor border.</li>
</ul>
</div></div>

<div class="env-block definition"><div class="env-title">Algorithm 11.4 (DBSCAN)</div><div class="env-body">
<ol>
  <li>Label each point as core, border, or noise.</li>
  <li>Connect core points that are within \\(\\varepsilon\\) of each other.</li>
  <li>Each connected component of core points forms a cluster.</li>
  <li>Assign border points to the cluster of their nearest core point.</li>
  <li>Noise points remain unassigned.</li>
</ol>
</div></div>

<div class="env-block remark"><div class="env-title">Advantages of DBSCAN</div><div class="env-body">
<ul>
  <li>Does not require specifying \\(K\\).</li>
  <li>Discovers clusters of arbitrary shape (rings, crescents, etc.).</li>
  <li>Naturally identifies outliers as noise points.</li>
  <li>Time complexity: \\(O(n^2)\\) in general, \\(O(n \\log n)\\) with spatial indexing.</li>
</ul>
</div></div>

<div class="env-block remark"><div class="env-title">Limitations</div><div class="env-body"><p>DBSCAN struggles when clusters have varying densities: a single \\(\\varepsilon\\) cannot capture both tight and loose clusters. HDBSCAN (hierarchical DBSCAN) addresses this by varying \\(\\varepsilon\\) and extracting the most stable clusters.</p></div></div>

<div class="viz-placeholder" data-viz="viz-dbscan"></div>
`,
      visualizations: [{
        id: 'viz-dbscan',
        title: 'DBSCAN: Core, Border, and Noise Points',
        description: 'Adjust epsilon to see how the clustering changes. Large filled circles = core, small filled = border, hollow = noise.',
        setup(container, controls) {
          const viz = new VizEngine(container, {scale:40, originX:280, originY:200});
          let eps=1.2, minPts=4;
          function mkRng(s){return function(){s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
          const rng=mkRng(77);
          const colors=['#58a6ff','#f0883e','#3fb950','#bc8cff','#f778ba','#d29922'];

          // Generate data with different shapes
          const data=[];
          // Cluster 1: arc
          for(let i=0;i<25;i++){const angle=Math.PI*0.3+i*Math.PI*0.06;const r=3+(rng()-0.5)*0.6;data.push({x:5+r*Math.cos(angle),y:4+r*Math.sin(angle)});}
          // Cluster 2: blob
          for(let i=0;i<15;i++)data.push({x:2+(rng()-0.5)*2,y:2+(rng()-0.5)*2});
          // Noise
          for(let i=0;i<6;i++)data.push({x:rng()*10,y:rng()*8});

          VizEngine.createSlider(controls,'Epsilon (\\u03b5)',0.3,3.0,eps,0.1,v=>{eps=v;draw();});
          VizEngine.createSlider(controls,'minPts',2,8,minPts,1,v=>{minPts=Math.round(v);draw();});

          function runDBSCAN(){
            const n=data.length;
            const neighbors=data.map((_,i)=>{const nb=[];for(let j=0;j<n;j++){if(i===j)continue;const dx=data[i].x-data[j].x,dy=data[i].y-data[j].y;if(Math.sqrt(dx*dx+dy*dy)<=eps)nb.push(j);}return nb;});
            const types=data.map((_,i)=>neighbors[i].length>=minPts?'core':'noise');
            // Mark border points
            for(let i=0;i<n;i++){if(types[i]==='noise'){for(const j of neighbors[i]){if(types[j]==='core'){types[i]='border';break;}}}}
            // BFS to find connected components of core points
            const clusterIds=new Array(n).fill(-1);
            let clusterId=0;
            for(let i=0;i<n;i++){
              if(types[i]!=='core'||clusterIds[i]>=0)continue;
              const queue=[i];clusterIds[i]=clusterId;
              while(queue.length>0){const cur=queue.shift();for(const j of neighbors[cur]){if(clusterIds[j]>=0)continue;if(types[j]==='core'){clusterIds[j]=clusterId;queue.push(j);}}}
              clusterId++;
            }
            // Assign border points
            for(let i=0;i<n;i++){if(types[i]==='border'){for(const j of neighbors[i]){if(clusterIds[j]>=0){clusterIds[i]=clusterIds[j];break;}}}}
            return{types,clusterIds,neighbors,nClusters:clusterId};
          }

          function draw(){
            viz.clear();const ctx=viz.ctx,W=viz.width,H=viz.height;
            const toSx=x=>30+x*(W-60)/10, toSy=y=>H-30-(y)*(H-60)/8;
            const res=runDBSCAN();

            // Grid
            ctx.strokeStyle=viz.colors.grid;ctx.lineWidth=0.5;
            for(let g=0;g<=10;g+=2){const sx=toSx(g);ctx.beginPath();ctx.moveTo(sx,20);ctx.lineTo(sx,H-30);ctx.stroke();}
            for(let g=0;g<=8;g+=2){const sy=toSy(g);ctx.beginPath();ctx.moveTo(30,sy);ctx.lineTo(W-30,sy);ctx.stroke();}

            // Draw epsilon circles for core points (faintly)
            for(let i=0;i<data.length;i++){
              if(res.types[i]==='core'){
                const sx=toSx(data[i].x),sy=toSy(data[i].y);
                const rPx=eps*(W-60)/10;
                ctx.strokeStyle=(res.clusterIds[i]>=0?colors[res.clusterIds[i]%6]:viz.colors.text)+'15';
                ctx.lineWidth=1;ctx.beginPath();ctx.arc(sx,sy,rPx,0,Math.PI*2);ctx.stroke();
              }
            }

            // Draw connections between nearby core points in same cluster
            for(let i=0;i<data.length;i++){
              if(res.types[i]!=='core')continue;
              for(const j of res.neighbors[i]){
                if(j<=i||res.types[j]!=='core'||res.clusterIds[i]!==res.clusterIds[j])continue;
                ctx.strokeStyle=colors[res.clusterIds[i]%6]+'33';ctx.lineWidth=1;
                ctx.beginPath();ctx.moveTo(toSx(data[i].x),toSy(data[i].y));ctx.lineTo(toSx(data[j].x),toSy(data[j].y));ctx.stroke();
              }
            }

            // Draw points
            for(let i=0;i<data.length;i++){
              const sx=toSx(data[i].x),sy=toSy(data[i].y);
              const col=res.clusterIds[i]>=0?colors[res.clusterIds[i]%6]:viz.colors.text;
              if(res.types[i]==='core'){
                ctx.fillStyle=col;ctx.beginPath();ctx.arc(sx,sy,7,0,Math.PI*2);ctx.fill();
              } else if(res.types[i]==='border'){
                ctx.fillStyle=col;ctx.beginPath();ctx.arc(sx,sy,4,0,Math.PI*2);ctx.fill();
              } else {
                ctx.strokeStyle=viz.colors.text;ctx.lineWidth=1.5;
                ctx.beginPath();ctx.arc(sx,sy,4,0,Math.PI*2);ctx.stroke();
              }
            }

            // Legend
            const lx=W-160,ly=30;
            ctx.fillStyle=viz.colors.blue;ctx.beginPath();ctx.arc(lx,ly,6,0,Math.PI*2);ctx.fill();
            ctx.fillStyle=viz.colors.white;ctx.font='11px -apple-system,sans-serif';ctx.textAlign='left';ctx.fillText('Core',lx+12,ly+4);
            ctx.fillStyle=viz.colors.blue;ctx.beginPath();ctx.arc(lx,ly+18,3.5,0,Math.PI*2);ctx.fill();
            ctx.fillStyle=viz.colors.white;ctx.fillText('Border',lx+12,ly+22);
            ctx.strokeStyle=viz.colors.text;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(lx,ly+36,3.5,0,Math.PI*2);ctx.stroke();
            ctx.fillStyle=viz.colors.white;ctx.fillText('Noise',lx+12,ly+40);

            const nNoise=res.types.filter(t=>t==='noise').length;
            viz.screenText('DBSCAN: \\u03b5='+eps.toFixed(1)+', minPts='+minPts+'  |  '+res.nClusters+' clusters, '+nNoise+' noise',W/2,12,viz.colors.white,12);
          }
          draw(); return viz;
        }
      }],
      exercises: [
        {question:'With \\(\\varepsilon = 1\\) and \\(\\text{minPts} = 3\\), consider 5 points at coordinates \\((0,0), (0.5,0), (1,0), (5,5), (5.5,5)\\). Classify each as core, border, or noise.',hint:'For each point, count how many other points fall within distance \\(\\varepsilon = 1\\).',solution:'\\((0,0)\\): neighbors within \\(\\varepsilon=1\\) are \\((0.5,0)\\) and \\((1,0)\\), so 3 total (including itself). Core. \\((0.5,0)\\): neighbors are \\((0,0)\\) and \\((1,0)\\), 3 total. Core. \\((1,0)\\): neighbors are \\((0,0)\\) at distance 1 and \\((0.5,0)\\), 3 total. Core. \\((5,5)\\): only neighbor is \\((5.5,5)\\) at distance 0.5. Only 2, less than minPts=3. Not core. But \\((5.5,5)\\) has only neighbor \\((5,5)\\), also only 2. Neither is core. Since neither is near a core point, both are noise.'},
        {question:'Why can DBSCAN discover clusters of arbitrary shape while K-means cannot?',hint:'Consider the assumptions each algorithm makes about cluster geometry.',solution:'K-means assigns each point to the nearest centroid, implicitly assuming clusters are convex, roughly spherical Voronoi cells. A crescent-shaped cluster would be split. DBSCAN defines clusters as connected dense regions: two points are in the same cluster if they are connected by a chain of core points, each within \\(\\varepsilon\\) of the next. This allows clusters of any shape, as long as they form a connected dense region.'},
        {question:'How should \\(\\varepsilon\\) and \\(\\text{minPts}\\) be chosen in practice?',hint:'Consider the k-distance plot and the effect of each parameter.',solution:'A common heuristic: set \\(\\text{minPts} = 2d\\) where \\(d\\) is the dimensionality (or at least 3). To choose \\(\\varepsilon\\), compute the distance to the \\(\\text{minPts}\\)-th nearest neighbor for all points, sort these distances, and plot them (the "k-distance plot"). The value of \\(\\varepsilon\\) at the "elbow" of this curve separates dense regions from sparse ones. Too small \\(\\varepsilon\\) classifies most points as noise; too large merges distinct clusters.'}
      ]
    }
  ]
});
