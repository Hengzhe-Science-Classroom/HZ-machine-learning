// === Chapter 19: Reinforcement Learning Basics ===
window.CHAPTERS = window.CHAPTERS || [];
window.CHAPTERS.push({
    id: 'ch19',
    number: 19,
    title: 'Reinforcement Learning Basics',
    subtitle: 'MDPs, Bellman equations, Q-learning, and policy gradients: learning to act through interaction with an environment',
    sections: [
        // ===================== Section 1: MDP Framework =====================
        {
            id: 'ch19-sec01',
            title: 'MDP Framework',
            content: `<h2>19.1 MDP Framework</h2>

<div class="env-block intuition">
<div class="env-title">Learning from Interaction</div>
<div class="env-body"><p>In supervised learning, we train on labeled examples; in unsupervised learning, we discover structure in unlabeled data. <strong>Reinforcement learning</strong> (RL) addresses a fundamentally different problem: an <strong>agent</strong> interacts with an <strong>environment</strong>, taking actions, receiving rewards, and learning a strategy (<strong>policy</strong>) that maximizes cumulative reward. There is no supervisor providing correct actions; the agent must discover good behavior through trial and error. The mathematical foundation for this setting is the <strong>Markov Decision Process</strong> (MDP).</p></div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 19.1.1 (Markov Decision Process)</div>
<div class="env-body"><p>A <strong>finite MDP</strong> is a tuple \\((\\mathcal{S}, \\mathcal{A}, P, R, \\gamma)\\) where:</p>
<ul>
<li>\\(\\mathcal{S}\\) is a finite set of <strong>states</strong>.</li>
<li>\\(\\mathcal{A}\\) is a finite set of <strong>actions</strong>.</li>
<li>\\(P(s' \\mid s, a)\\) is the <strong>transition probability</strong>: the probability of moving to state \\(s'\\) when taking action \\(a\\) in state \\(s\\).</li>
<li>\\(R(s, a, s')\\) is the <strong>reward function</strong>: the immediate reward upon transitioning from \\(s\\) to \\(s'\\) via action \\(a\\). Often simplified to \\(R(s, a)\\).</li>
<li>\\(\\gamma \\in [0, 1)\\) is the <strong>discount factor</strong>, controlling the trade-off between immediate and future rewards.</li>
</ul></div>
</div>

<div class="env-block definition">
<div class="env-title">Definition 19.1.2 (Markov Property)</div>
<div class="env-body"><p>The <strong>Markov property</strong> states that the future is conditionally independent of the past given the present:
\\[P(S_{t+1} \\mid S_t, A_t, S_{t-1}, A_{t-1}, \\ldots) = P(S_{t+1} \\mid S_t, A_t).\\]
The current state captures all relevant information for decision-making. This is a modeling assumption: if the true state is only partially observed, we need a Partially Observable MDP (POMDP).</p></div>
</div>

<h3>Policies and Returns</h3>

<div class="env-block definition">
<div class="env-title">Definition 19.1.3 (Policy)</div>
<div class="env-body"><p>A <strong>policy</strong> \\(\\pi\\) specifies how the agent selects actions. A <strong>deterministic policy</strong> is a mapping \\(\\pi: \\mathcal{S} \\to \\mathcal{A}\\). A <strong>stochastic policy</strong> assigns a probability distribution over actions: \\(\\pi(a \\mid s) = P(A_t = a \\mid S_t = s)\\). The goal of RL is to find a policy that maximizes the expected <strong>return</strong>:
\\[G_t = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1}.\\]
The discount factor \\(\\gamma\\) ensures this sum converges and encodes the preference for sooner rewards. When \\(\\gamma \\to 0\\), the agent is myopic (cares only about the next reward); when \\(\\gamma \\to 1\\), the agent is far-sighted.</p></div>
</div>

<h3>Value Functions</h3>

<div class="env-block definition">
<div class="env-title">Definition 19.1.4 (State-Value and Action-Value Functions)</div>
<div class="env-body"><p>The <strong>state-value function</strong> under policy \\(\\pi\\) is the expected return starting from state \\(s\\):
\\[V^\\pi(s) = \\mathbb{E}_\\pi[G_t \\mid S_t = s] = \\mathbb{E}_\\pi\\!\\left[\\sum_{k=0}^\\infty \\gamma^k R_{t+k+1} \\mid S_t = s\\right].\\]
The <strong>action-value function</strong> (Q-function) is the expected return from taking action \\(a\\) in state \\(s\\) and then following \\(\\pi\\):
\\[Q^\\pi(s, a) = \\mathbb{E}_\\pi[G_t \\mid S_t = s, A_t = a].\\]
The two are related by \\(V^\\pi(s) = \\sum_a \\pi(a \\mid s)\\, Q^\\pi(s, a)\\).</p></div>
</div>

<div class="env-block remark">
<div class="env-title">Remark (Optimal Policy Existence)</div>
<div class="env-body"><p>For any finite MDP with \\(\\gamma \\in [0,1)\\), there exists at least one deterministic optimal policy \\(\\pi^*\\) that simultaneously maximizes \\(V^\\pi(s)\\) for <em>all</em> states \\(s\\). The optimal value functions are \\(V^*(s) = \\max_\\pi V^\\pi(s)\\) and \\(Q^*(s, a) = \\max_\\pi Q^\\pi(s, a)\\). Once \\(Q^*\\) is known, the optimal policy is simply \\(\\pi^*(s) = \\arg\\max_a Q^*(s, a)\\).</p></div>
</div>

<div class="viz-placeholder" data-viz="viz-gridworld-mdp"></div>`,

            visualizations: [
                {
                    id: 'viz-gridworld-mdp',
                    title: 'Gridworld MDP: Policy and Value Function',
                    description: 'A 5x5 gridworld MDP. The green cell is the goal (+1 reward), the red cell is a trap (-1 reward). Arrows show the current policy; cell colors indicate state values (brighter = higher). Adjust the discount factor to see how it changes the value landscape and optimal behavior.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, {scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var rows = 5, cols = 5;
                        var gamma = 0.9;
                        var goalR = 4, goalC = 4;
                        var trapR = 1, trapC = 3;
                        var wallCells = [[2, 2], [3, 2]];

                        VizEngine.createSlider(controls, 'Discount \u03b3', 0.1, 0.99, gamma, 0.01, function(v) { gamma = v; });

                        function isWall(r, c) {
                            for (var i = 0; i < wallCells.length; i++) {
                                if (wallCells[i][0] === r && wallCells[i][1] === c) return true;
                            }
                            return false;
                        }

                        function getReward(r, c) {
                            if (r === goalR && c === goalC) return 1;
                            if (r === trapR && c === trapC) return -1;
                            return -0.04;
                        }

                        function isTerminal(r, c) {
                            return (r === goalR && c === goalC) || (r === trapR && c === trapC);
                        }

                        // Value iteration
                        function valueIteration() {
                            var V = [];
                            for (var r = 0; r < rows; r++) {
                                V[r] = [];
                                for (var c = 0; c < cols; c++) V[r][c] = 0;
                            }
                            var dirs = [[-1,0],[1,0],[0,-1],[0,1]]; // up, down, left, right
                            var arrows = ['\u2191', '\u2193', '\u2190', '\u2192'];
                            var policy = [];
                            for (var r = 0; r < rows; r++) { policy[r] = []; for (var c = 0; c < cols; c++) policy[r][c] = 0; }

                            for (var iter = 0; iter < 100; iter++) {
                                var newV = [];
                                for (var r = 0; r < rows; r++) { newV[r] = []; for (var c = 0; c < cols; c++) newV[r][c] = V[r][c]; }
                                for (var r = 0; r < rows; r++) {
                                    for (var c = 0; c < cols; c++) {
                                        if (isWall(r, c) || isTerminal(r, c)) continue;
                                        var bestVal = -Infinity, bestA = 0;
                                        for (var a = 0; a < 4; a++) {
                                            var nr = r + dirs[a][0], nc = c + dirs[a][1];
                                            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols || isWall(nr, nc)) { nr = r; nc = c; }
                                            var val = getReward(r, c) + gamma * V[nr][nc];
                                            if (val > bestVal) { bestVal = val; bestA = a; }
                                        }
                                        newV[r][c] = bestVal;
                                        policy[r][c] = bestA;
                                    }
                                }
                                V = newV;
                            }
                            V[goalR][goalC] = 1;
                            V[trapR][trapC] = -1;
                            return {V: V, policy: policy, dirs: dirs, arrows: arrows};
                        }

                        function draw() {
                            ctx.fillStyle = viz.colors.bg;
                            ctx.fillRect(0, 0, W, H);

                            var result = valueIteration();
                            var V = result.V, policy = result.policy, dirs = result.dirs, arrows = result.arrows;

                            // Find value range for coloring
                            var vMin = Infinity, vMax = -Infinity;
                            for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) {
                                if (!isWall(r, c)) { vMin = Math.min(vMin, V[r][c]); vMax = Math.max(vMax, V[r][c]); }
                            }
                            var vRange = Math.max(vMax - vMin, 0.01);

                            var cellSize = Math.min((W - 80) / cols, (H - 80) / rows);
                            var gridW = cellSize * cols, gridH = cellSize * rows;
                            var offX = (W - gridW) / 2, offY = (H - gridH) / 2 + 10;

                            for (var r = 0; r < rows; r++) {
                                for (var c = 0; c < cols; c++) {
                                    var x = offX + c * cellSize, y = offY + r * cellSize;
                                    if (isWall(r, c)) {
                                        ctx.fillStyle = '#333355';
                                        ctx.fillRect(x, y, cellSize, cellSize);
                                        ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                                        ctx.strokeRect(x, y, cellSize, cellSize);
                                        continue;
                                    }
                                    // Color by value
                                    var t = (V[r][c] - vMin) / vRange;
                                    var red, green, blue;
                                    if (V[r][c] < 0) {
                                        red = Math.round(80 + 150 * (-V[r][c] / Math.max(1, -vMin)));
                                        green = 30; blue = 30;
                                    } else {
                                        red = 20;
                                        green = Math.round(40 + 180 * t);
                                        blue = Math.round(40 + 100 * t);
                                    }
                                    ctx.fillStyle = 'rgb(' + red + ',' + green + ',' + blue + ')';
                                    ctx.fillRect(x, y, cellSize, cellSize);
                                    ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                                    ctx.strokeRect(x, y, cellSize, cellSize);

                                    // Value text
                                    ctx.fillStyle = '#ffffffcc'; ctx.font = '11px -apple-system,sans-serif';
                                    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                                    ctx.fillText(V[r][c].toFixed(2), x + cellSize / 2, y + cellSize - 4);

                                    // Policy arrow (or label)
                                    ctx.font = 'bold 18px -apple-system,sans-serif';
                                    ctx.textBaseline = 'middle';
                                    if (r === goalR && c === goalC) {
                                        ctx.fillStyle = viz.colors.green;
                                        ctx.fillText('G', x + cellSize / 2, y + cellSize / 2 - 5);
                                    } else if (r === trapR && c === trapC) {
                                        ctx.fillStyle = viz.colors.red;
                                        ctx.fillText('T', x + cellSize / 2, y + cellSize / 2 - 5);
                                    } else {
                                        ctx.fillStyle = viz.colors.yellow;
                                        ctx.fillText(arrows[policy[r][c]], x + cellSize / 2, y + cellSize / 2 - 5);
                                    }
                                }
                            }

                            // Legend
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                            ctx.fillText('Gridworld MDP (\u03b3=' + gamma.toFixed(2) + ')', W / 2, 8);
                            ctx.font = '11px -apple-system,sans-serif'; ctx.fillStyle = viz.colors.text;
                            ctx.textAlign = 'left';
                            ctx.fillText('G = Goal (+1)', offX, offY + gridH + 10);
                            ctx.fillText('T = Trap (-1)', offX + 120, offY + gridH + 10);
                            ctx.fillText('Step cost: -0.04', offX + 230, offY + gridH + 10);
                        }
                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Show that the infinite-horizon discounted return \\(G_t = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1}\\) is finite whenever rewards are bounded (\\(|R_t| \\leq R_{\\max}\\)) and \\(\\gamma \\in [0,1)\\).',
                    hint: 'Use the formula for geometric series.',
                    solution: '\\(|G_t| = |\\sum_{k=0}^\\infty \\gamma^k R_{t+k+1}| \\leq \\sum_{k=0}^\\infty \\gamma^k |R_{t+k+1}| \\leq R_{\\max} \\sum_{k=0}^\\infty \\gamma^k = \\frac{R_{\\max}}{1 - \\gamma}\\). Since \\(\\gamma \\in [0,1)\\), the geometric series converges and the return is bounded by \\(R_{\\max}/(1-\\gamma)\\). For example, with \\(R_{\\max}=1\\) and \\(\\gamma=0.99\\), the maximum return is 100.'
                },
                {
                    question: 'Explain why a small step cost (e.g., \\(-0.04\\) per time step) is important in the gridworld MDP. What happens without it?',
                    hint: 'Consider what happens if all non-terminal states give zero reward.',
                    solution: 'Without a step cost, any policy that eventually reaches the goal has the same total undiscounted reward. The agent has no incentive to take the shortest path; it could wander indefinitely. The negative step cost penalizes each time step, encouraging the agent to reach the goal quickly. With \\(\\gamma\\) close to 1 and zero step cost, the agent is nearly indifferent about path length. The step cost also makes the agent prefer to avoid the trap by a margin: staying far from the trap costs extra steps but avoids the large negative reward.'
                },
                {
                    question: 'Derive the relationship \\(V^\\pi(s) = \\sum_a \\pi(a|s) Q^\\pi(s,a)\\) and the reverse relationship \\(Q^\\pi(s,a) = R(s,a) + \\gamma \\sum_{s\'} P(s\'|s,a) V^\\pi(s\')\\) from the definitions.',
                    hint: 'Expand the expectation \\(V^\\pi(s) = \\mathbb{E}_\\pi[G_t | S_t = s]\\) by conditioning on the first action.',
                    solution: '\\(V^\\pi(s) = \\mathbb{E}_\\pi[G_t|S_t=s] = \\sum_a \\pi(a|s) \\mathbb{E}_\\pi[G_t|S_t=s, A_t=a] = \\sum_a \\pi(a|s) Q^\\pi(s,a)\\). For the Q-function: \\(Q^\\pi(s,a) = \\mathbb{E}[R_{t+1} + \\gamma G_{t+1} | S_t=s, A_t=a] = R(s,a) + \\gamma \\sum_{s\'} P(s\'|s,a) \\mathbb{E}_\\pi[G_{t+1}|S_{t+1}=s\'] = R(s,a) + \\gamma \\sum_{s\'} P(s\'|s,a) V^\\pi(s\')\\). These two equations together give the Bellman equation for \\(V^\\pi\\).'
                }
            ]
        },

        // ===================== Section 2: Value Iteration & Policy Iteration =====================
        {
            id: 'ch19-sec02',
            title: 'Value Iteration & Policy Iteration',
            content: `<h2>19.2 Value Iteration &amp; Policy Iteration</h2>

<div class="env-block intuition">
<div class="env-title">Dynamic Programming for MDPs</div>
<div class="env-body"><p>When the MDP model (transition probabilities and rewards) is fully known, we can compute optimal policies using <strong>dynamic programming</strong>. The two classical algorithms are <strong>value iteration</strong> (repeatedly apply the Bellman optimality update) and <strong>policy iteration</strong> (alternate between evaluating a policy and improving it). Both converge to the optimal policy, but they differ in computational structure.</p></div>
</div>

<h3>Bellman Equations</h3>

<div class="env-block theorem">
<div class="env-title">Theorem 19.2.1 (Bellman Optimality Equation)</div>
<div class="env-body"><p>The optimal value function \\(V^*\\) satisfies the <strong>Bellman optimality equation</strong>:
\\[V^*(s) = \\max_{a \\in \\mathcal{A}} \\left[ R(s, a) + \\gamma \\sum_{s' \\in \\mathcal{S}} P(s' \\mid s, a)\\, V^*(s') \\right].\\]
Similarly, the optimal action-value function satisfies:
\\[Q^*(s, a) = R(s, a) + \\gamma \\sum_{s'} P(s' \\mid s, a) \\max_{a'} Q^*(s', a').\\]
These are nonlinear systems of equations (due to the \\(\\max\\)) that characterize the unique optimal value functions.</p></div>
</div>

<h3>Value Iteration</h3>

<div class="env-block definition">
<div class="env-title">Algorithm (Value Iteration)</div>
<div class="env-body"><p>Initialize \\(V_0(s) = 0\\) for all \\(s\\). Repeat for \\(k = 0, 1, 2, \\ldots\\):
\\[V_{k+1}(s) = \\max_{a} \\left[ R(s, a) + \\gamma \\sum_{s'} P(s' \\mid s, a)\\, V_k(s') \\right], \\quad \\forall s \\in \\mathcal{S}.\\]
Stop when \\(\\max_s |V_{k+1}(s) - V_k(s)| &lt; \\epsilon\\). Extract the greedy policy: \\(\\pi^*(s) = \\arg\\max_a [R(s,a) + \\gamma \\sum_{s'} P(s'|s,a) V^*(s')]\\).</p></div>
</div>

<div class="env-block theorem">
<div class="env-title">Theorem 19.2.2 (Contraction and Convergence)</div>
<div class="env-body"><p>The Bellman optimality operator \\(T\\) defined by \\((TV)(s) = \\max_a [R(s,a) + \\gamma \\sum_{s'} P(s'|s,a) V(s')]\\) is a <strong>contraction mapping</strong> in the \\(\\ell_\\infty\\) norm with contraction factor \\(\\gamma\\):
\\[\\|TV_1 - TV_2\\|_\\infty \\leq \\gamma \\|V_1 - V_2\\|_\\infty.\\]
By the Banach fixed-point theorem, \\(T\\) has a unique fixed point \\(V^*\\), and value iteration converges to it at rate \\(O(\\gamma^k)\\).</p></div>
</div>

<h3>Policy Iteration</h3>

<div class="env-block definition">
<div class="env-title">Algorithm (Policy Iteration)</div>
<div class="env-body"><p>Initialize with an arbitrary policy \\(\\pi_0\\). Repeat for \\(k = 0, 1, 2, \\ldots\\):</p>
<ol>
<li><strong>Policy Evaluation:</strong> Solve for \\(V^{\\pi_k}\\) by solving the linear system
\\[V^{\\pi_k}(s) = R(s, \\pi_k(s)) + \\gamma \\sum_{s'} P(s' \\mid s, \\pi_k(s))\\, V^{\\pi_k}(s'), \\quad \\forall s.\\]</li>
<li><strong>Policy Improvement:</strong> Update the policy greedily:
\\[\\pi_{k+1}(s) = \\arg\\max_a \\left[ R(s, a) + \\gamma \\sum_{s'} P(s' \\mid s, a)\\, V^{\\pi_k}(s') \\right].\\]</li>
</ol>
<p>Stop when \\(\\pi_{k+1} = \\pi_k\\). Policy iteration converges in a finite number of steps (at most \\(|\\mathcal{A}|^{|\\mathcal{S}|}\\) iterations, though in practice far fewer).</p></div>
</div>

<div class="env-block remark">
<div class="env-title">Remark (Value Iteration vs. Policy Iteration)</div>
<div class="env-body"><p>Value iteration performs one Bellman backup per state per iteration and may need many iterations. Policy iteration performs full policy evaluation (solving a linear system of \\(|\\mathcal{S}|\\) equations) but typically converges in very few iterations. For small state spaces, policy iteration is often faster. For large state spaces where solving the linear system is expensive, value iteration (or its truncated variant) is preferred.</p></div>
</div>

<div class="viz-placeholder" data-viz="viz-value-iteration"></div>`,

            visualizations: [
                {
                    id: 'viz-value-iteration',
                    title: 'Value Iteration: Watch Values Converge',
                    description: 'Step through value iteration on a 4x4 gridworld. Each step applies one Bellman backup to all states. Watch the values propagate from the goal outward until convergence. The arrows update to show the improving policy.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, {scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var rows = 4, cols = 4;
                        var gamma = 0.9;
                        var goalR = 3, goalC = 3;
                        var trapR = 1, trapC = 1;
                        var iterCount = 0;
                        var V = [], policy = [];
                        var dirs = [[-1,0],[1,0],[0,-1],[0,1]];
                        var arrows = ['\u2191','\u2193','\u2190','\u2192'];

                        function initV() {
                            V = []; policy = [];
                            for (var r = 0; r < rows; r++) {
                                V[r] = []; policy[r] = [];
                                for (var c = 0; c < cols; c++) { V[r][c] = 0; policy[r][c] = 0; }
                            }
                            iterCount = 0;
                        }
                        initV();

                        function getReward(r, c) {
                            if (r === goalR && c === goalC) return 1;
                            if (r === trapR && c === trapC) return -1;
                            return -0.04;
                        }
                        function isTerminal(r, c) {
                            return (r === goalR && c === goalC) || (r === trapR && c === trapC);
                        }

                        function bellmanStep() {
                            var newV = [];
                            for (var r = 0; r < rows; r++) { newV[r] = []; for (var c = 0; c < cols; c++) newV[r][c] = V[r][c]; }
                            for (var r = 0; r < rows; r++) {
                                for (var c = 0; c < cols; c++) {
                                    if (isTerminal(r, c)) { newV[r][c] = getReward(r, c); continue; }
                                    var bestVal = -Infinity, bestA = 0;
                                    for (var a = 0; a < 4; a++) {
                                        var nr = r + dirs[a][0], nc = c + dirs[a][1];
                                        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) { nr = r; nc = c; }
                                        var val = getReward(r, c) + gamma * V[nr][nc];
                                        if (val > bestVal) { bestVal = val; bestA = a; }
                                    }
                                    newV[r][c] = bestVal;
                                    policy[r][c] = bestA;
                                }
                            }
                            V = newV;
                            iterCount++;
                        }

                        VizEngine.createButton(controls, 'Step', function() { bellmanStep(); });
                        VizEngine.createButton(controls, 'Run 10 Steps', function() { for (var i = 0; i < 10; i++) bellmanStep(); });
                        VizEngine.createButton(controls, 'Reset', function() { initV(); });
                        VizEngine.createSlider(controls, '\u03b3', 0.1, 0.99, gamma, 0.01, function(v) { gamma = v; initV(); });

                        function draw() {
                            ctx.fillStyle = viz.colors.bg; ctx.fillRect(0, 0, W, H);

                            var vMin = Infinity, vMax = -Infinity;
                            for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) {
                                vMin = Math.min(vMin, V[r][c]); vMax = Math.max(vMax, V[r][c]);
                            }
                            var vRange = Math.max(vMax - vMin, 0.01);

                            var cellSize = Math.min((W - 100) / cols, (H - 90) / rows);
                            var gridW = cellSize * cols, gridH = cellSize * rows;
                            var offX = (W - gridW) / 2, offY = (H - gridH) / 2 + 12;

                            for (var r = 0; r < rows; r++) {
                                for (var c = 0; c < cols; c++) {
                                    var x = offX + c * cellSize, y = offY + r * cellSize;
                                    var t = (V[r][c] - vMin) / vRange;
                                    var red, green, blue2;
                                    if (V[r][c] < 0) {
                                        red = Math.round(60 + 160 * Math.min(1, -V[r][c]));
                                        green = 25; blue2 = 25;
                                    } else {
                                        red = 15;
                                        green = Math.round(30 + 200 * t);
                                        blue2 = Math.round(50 + 100 * t);
                                    }
                                    ctx.fillStyle = 'rgb(' + red + ',' + green + ',' + blue2 + ')';
                                    ctx.fillRect(x, y, cellSize, cellSize);
                                    ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                                    ctx.strokeRect(x, y, cellSize, cellSize);

                                    // Value
                                    ctx.fillStyle = '#ffffffcc'; ctx.font = '12px -apple-system,sans-serif';
                                    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                                    ctx.fillText(V[r][c].toFixed(3), x + cellSize / 2, y + cellSize - 5);

                                    // Arrow or label
                                    ctx.font = 'bold 20px -apple-system,sans-serif'; ctx.textBaseline = 'middle';
                                    if (r === goalR && c === goalC) {
                                        ctx.fillStyle = viz.colors.green;
                                        ctx.fillText('G', x + cellSize / 2, y + cellSize / 2 - 8);
                                    } else if (r === trapR && c === trapC) {
                                        ctx.fillStyle = viz.colors.red;
                                        ctx.fillText('T', x + cellSize / 2, y + cellSize / 2 - 8);
                                    } else if (iterCount > 0) {
                                        ctx.fillStyle = viz.colors.yellow;
                                        ctx.fillText(arrows[policy[r][c]], x + cellSize / 2, y + cellSize / 2 - 8);
                                    }
                                }
                            }

                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                            ctx.fillText('Value Iteration (k=' + iterCount + ', \u03b3=' + gamma.toFixed(2) + ')', W / 2, 8);

                            // Convergence indicator
                            if (iterCount > 0) {
                                var maxDelta = 0;
                                var oldV = [];
                                for (var r2 = 0; r2 < rows; r2++) { oldV[r2] = []; for (var c2 = 0; c2 < cols; c2++) oldV[r2][c2] = V[r2][c2]; }
                                // recompute one step to find delta
                                for (var r2 = 0; r2 < rows; r2++) for (var c2 = 0; c2 < cols; c2++) {
                                    if (isTerminal(r2, c2)) continue;
                                    var bv = -Infinity;
                                    for (var a2 = 0; a2 < 4; a2++) {
                                        var nr2 = r2 + dirs[a2][0], nc2 = c2 + dirs[a2][1];
                                        if (nr2 < 0 || nr2 >= rows || nc2 < 0 || nc2 >= cols) { nr2 = r2; nc2 = c2; }
                                        bv = Math.max(bv, getReward(r2, c2) + gamma * V[nr2][nc2]);
                                    }
                                    maxDelta = Math.max(maxDelta, Math.abs(bv - V[r2][c2]));
                                }
                                ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                                ctx.fillText('max \u0394V = ' + maxDelta.toFixed(6), W / 2, offY + gridH + 12);
                                if (maxDelta < 0.001) {
                                    ctx.fillStyle = viz.colors.green;
                                    ctx.fillText('Converged!', W / 2, offY + gridH + 28);
                                }
                            }
                        }
                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Prove that the Bellman optimality operator \\(T\\) is a contraction in the \\(\\ell_\\infty\\) norm with factor \\(\\gamma\\).',
                    hint: 'Use \\(|\\max_a f(a) - \\max_a g(a)| \\leq \\max_a |f(a) - g(a)|\\).',
                    solution: '\\(|(TV_1)(s) - (TV_2)(s)| = |\\max_a[R + \\gamma \\sum_{s\'} P(s\'|s,a)V_1(s\')] - \\max_a[R + \\gamma \\sum_{s\'} P(s\'|s,a)V_2(s\')]| \\leq \\max_a |\\gamma \\sum_{s\'} P(s\'|s,a)(V_1(s\') - V_2(s\'))| \\leq \\gamma \\max_a \\sum_{s\'} P(s\'|s,a)|V_1(s\') - V_2(s\')| \\leq \\gamma \\|V_1 - V_2\\|_\\infty \\sum_{s\'} P(s\'|s,a) = \\gamma \\|V_1 - V_2\\|_\\infty\\). Taking max over \\(s\\): \\(\\|TV_1 - TV_2\\|_\\infty \\leq \\gamma \\|V_1 - V_2\\|_\\infty\\).'
                },
                {
                    question: 'After \\(k\\) iterations of value iteration starting from \\(V_0 = 0\\), give an upper bound on \\(\\|V_k - V^*\\|_\\infty\\) in terms of \\(\\gamma\\), \\(k\\), and \\(R_{\\max}\\).',
                    hint: 'Apply the contraction property \\(k\\) times and note \\(\\|V_0 - V^*\\|_\\infty \\leq R_{\\max}/(1-\\gamma)\\).',
                    solution: 'By contraction, \\(\\|V_k - V^*\\|_\\infty \\leq \\gamma^k \\|V_0 - V^*\\|_\\infty\\). Since \\(V_0 = 0\\) and \\(\\|V^*\\|_\\infty \\leq R_{\\max}/(1-\\gamma)\\), we get \\(\\|V_k - V^*\\|_\\infty \\leq \\gamma^k R_{\\max}/(1-\\gamma)\\). For \\(\\gamma=0.9\\), \\(R_{\\max}=1\\): after \\(k=50\\) iterations, the error is at most \\(0.9^{50} \\cdot 10 \\approx 0.052\\). After \\(k=100\\), it is about \\(2.66 \\times 10^{-4}\\).'
                },
                {
                    question: 'Policy iteration always converges in at most \\(|\\mathcal{A}|^{|\\mathcal{S}|}\\) steps. Explain why, and argue that this bound is extremely loose in practice.',
                    hint: 'Each iteration produces a strictly better policy (unless already optimal). How many distinct deterministic policies are there?',
                    solution: 'A deterministic policy maps each of the \\(|\\mathcal{S}|\\) states to one of \\(|\\mathcal{A}|\\) actions, giving \\(|\\mathcal{A}|^{|\\mathcal{S}|}\\) distinct policies. The policy improvement theorem guarantees each new policy is strictly better (higher \\(V^{\\pi}\\) at some state) unless it is already optimal. Since there are finitely many policies and no cycles, convergence occurs in at most \\(|\\mathcal{A}|^{|\\mathcal{S}|}\\) steps. This is astronomically loose: for a 100-state, 4-action MDP, the bound is \\(4^{100} \\approx 10^{60}\\), but empirically policy iteration converges in 5-10 iterations.'
                }
            ]
        },

        // ===================== Section 3: Q-Learning =====================
        {
            id: 'ch19-sec03',
            title: 'Q-Learning',
            content: `<h2>19.3 Q-Learning</h2>

<div class="env-block intuition">
<div class="env-title">Learning Without a Model</div>
<div class="env-body"><p>Value iteration and policy iteration require full knowledge of the MDP (transition probabilities and rewards). In most real-world problems, these are unknown. <strong>Q-learning</strong> is a <strong>model-free</strong> algorithm that learns the optimal Q-function directly from experience (state, action, reward, next state) tuples, without ever building a model of the environment. It is the foundation of modern deep RL methods like DQN.</p></div>
</div>

<h3>Temporal Difference Learning</h3>

<p>The key idea behind Q-learning is <strong>temporal difference</strong> (TD) learning: update value estimates using the difference between the current estimate and a bootstrapped target.</p>

<div class="env-block definition">
<div class="env-title">Definition 19.3.1 (TD Error)</div>
<div class="env-body"><p>The <strong>TD error</strong> (or TD target) for Q-learning at time \\(t\\) is
\\[\\delta_t = R_{t+1} + \\gamma \\max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t).\\]
This is the difference between the bootstrapped target \\(R_{t+1} + \\gamma \\max_{a'} Q(S_{t+1}, a')\\) and the current estimate \\(Q(S_t, A_t)\\). The TD error drives the Q-value update toward the Bellman optimality equation.</p></div>
</div>

<div class="env-block definition">
<div class="env-title">Algorithm (Q-Learning)</div>
<div class="env-body"><p>Initialize \\(Q(s, a) = 0\\) for all \\(s, a\\). For each episode:</p>
<ol>
<li>Observe initial state \\(S_0\\).</li>
<li>For each step \\(t = 0, 1, 2, \\ldots\\):
    <ol style="list-style-type: lower-alpha;">
    <li>Choose action \\(A_t\\) using an exploration policy (e.g., \\(\\epsilon\\)-greedy).</li>
    <li>Take action \\(A_t\\), observe reward \\(R_{t+1}\\) and next state \\(S_{t+1}\\).</li>
    <li>Update: \\(Q(S_t, A_t) \\leftarrow Q(S_t, A_t) + \\alpha \\left[ R_{t+1} + \\gamma \\max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \\right]\\).</li>
    </ol>
</li>
</ol>
<p>Here \\(\\alpha \\in (0, 1]\\) is the <strong>learning rate</strong>. The \\(\\max\\) over actions makes this <strong>off-policy</strong>: the update targets the optimal Q regardless of the exploration policy.</p></div>
</div>

<h3>Exploration vs. Exploitation</h3>

<div class="env-block definition">
<div class="env-title">Definition 19.3.2 (\\(\\epsilon\\)-Greedy Policy)</div>
<div class="env-body"><p>The \\(\\epsilon\\)-greedy policy balances exploration and exploitation:
\\[A_t = \\begin{cases} \\arg\\max_a Q(S_t, a) & \\text{with probability } 1 - \\epsilon \\\\ \\text{uniform random action} & \\text{with probability } \\epsilon \\end{cases}\\]
A common schedule decays \\(\\epsilon\\) over time: high \\(\\epsilon\\) early (explore) transitioning to low \\(\\epsilon\\) later (exploit). This ensures sufficient exploration while eventually converging to the greedy policy.</p></div>
</div>

<div class="env-block theorem">
<div class="env-title">Theorem 19.3.3 (Q-Learning Convergence)</div>
<div class="env-body"><p>Tabular Q-learning converges to \\(Q^*\\) with probability 1, provided:</p>
<ol>
<li>Every state-action pair is visited infinitely often.</li>
<li>The learning rate satisfies the Robbins-Monro conditions: \\(\\sum_t \\alpha_t = \\infty\\) and \\(\\sum_t \\alpha_t^2 &lt; \\infty\\).</li>
</ol>
<p>In practice, a constant small \\(\\alpha\\) often works well, though it technically only converges to a neighborhood of \\(Q^*\\).</p></div>
</div>

<div class="env-block remark">
<div class="env-title">Remark (On-Policy vs. Off-Policy)</div>
<div class="env-body"><p>Q-learning is <em>off-policy</em>: the behavior policy (\\(\\epsilon\\)-greedy) differs from the target policy (greedy). The alternative is <em>on-policy</em> learning (e.g., SARSA), where the update uses the action actually taken: \\(Q(S_t, A_t) \\leftarrow Q(S_t, A_t) + \\alpha[R_{t+1} + \\gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]\\). SARSA is more conservative (it accounts for exploration in its value estimates), while Q-learning is more aggressive (it always targets the optimal policy).</p></div>
</div>

<div class="viz-placeholder" data-viz="viz-q-learning"></div>`,

            visualizations: [
                {
                    id: 'viz-q-learning',
                    title: 'Q-Learning Agent in a Gridworld',
                    description: 'Watch a Q-learning agent explore a 4x4 gridworld. Cell colors show the current max Q-value. The agent uses epsilon-greedy exploration (red steps = random, blue = greedy). Q-values update toward the optimal policy over time.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, {scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var rows = 4, cols = 4;
                        var gamma = 0.9, alpha = 0.1, epsilon = 0.3;
                        var goalR = 3, goalC = 3, trapR = 1, trapC = 1;
                        var dirs = [[-1,0],[1,0],[0,-1],[0,1]];
                        var arrows = ['\u2191','\u2193','\u2190','\u2192'];
                        var Q = [], agentR = 0, agentC = 0;
                        var running = false, episodeCount = 0, stepCount = 0;
                        var lastExplored = false;

                        function initQ() {
                            Q = [];
                            for (var r = 0; r < rows; r++) { Q[r] = []; for (var c = 0; c < cols; c++) { Q[r][c] = [0,0,0,0]; } }
                            agentR = 0; agentC = 0; episodeCount = 0; stepCount = 0;
                        }
                        initQ();

                        VizEngine.createSlider(controls, '\u03b5 (explore)', 0, 1, epsilon, 0.05, function(v) { epsilon = v; });
                        VizEngine.createSlider(controls, '\u03b1 (learn rate)', 0.01, 0.5, alpha, 0.01, function(v) { alpha = v; });
                        VizEngine.createButton(controls, 'Run', function() { running = true; });
                        VizEngine.createButton(controls, 'Pause', function() { running = false; });
                        VizEngine.createButton(controls, 'Reset', function() { running = false; initQ(); });

                        function getReward(r, c) {
                            if (r === goalR && c === goalC) return 1;
                            if (r === trapR && c === trapC) return -1;
                            return -0.04;
                        }
                        function isTerminal(r, c) {
                            return (r === goalR && c === goalC) || (r === trapR && c === trapC);
                        }

                        function qStep() {
                            if (isTerminal(agentR, agentC)) {
                                agentR = 0; agentC = 0; episodeCount++;
                                return;
                            }
                            // Epsilon-greedy action
                            var action;
                            if (Math.random() < epsilon) {
                                action = Math.floor(Math.random() * 4);
                                lastExplored = true;
                            } else {
                                action = 0;
                                var bestQ = Q[agentR][agentC][0];
                                for (var a = 1; a < 4; a++) {
                                    if (Q[agentR][agentC][a] > bestQ) { bestQ = Q[agentR][agentC][a]; action = a; }
                                }
                                lastExplored = false;
                            }
                            var nr = agentR + dirs[action][0], nc = agentC + dirs[action][1];
                            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) { nr = agentR; nc = agentC; }
                            var reward = getReward(nr, nc);
                            var maxQnext = Q[nr][nc][0];
                            for (var a = 1; a < 4; a++) maxQnext = Math.max(maxQnext, Q[nr][nc][a]);
                            if (isTerminal(nr, nc)) maxQnext = 0;
                            // Q-learning update
                            Q[agentR][agentC][action] += alpha * (reward + gamma * maxQnext - Q[agentR][agentC][action]);
                            agentR = nr; agentC = nc;
                            stepCount++;
                        }

                        var lastTime = 0;
                        function draw(t) {
                            if (running && t - lastTime > 40) {
                                for (var s = 0; s < 3; s++) qStep();
                                lastTime = t;
                            }

                            ctx.fillStyle = viz.colors.bg; ctx.fillRect(0, 0, W, H);

                            // Find Q range for coloring
                            var qMin = 0, qMax = 0;
                            for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) {
                                for (var a = 0; a < 4; a++) {
                                    qMin = Math.min(qMin, Q[r][c][a]);
                                    qMax = Math.max(qMax, Q[r][c][a]);
                                }
                            }
                            var qRange = Math.max(qMax - qMin, 0.01);

                            var cellSize = Math.min((W - 100) / cols, (H - 100) / rows);
                            var gridW = cellSize * cols, gridH = cellSize * rows;
                            var offX = (W - gridW) / 2, offY = (H - gridH) / 2 + 10;

                            for (var r = 0; r < rows; r++) {
                                for (var c = 0; c < cols; c++) {
                                    var x = offX + c * cellSize, y = offY + r * cellSize;
                                    var maxQ = Math.max.apply(null, Q[r][c]);
                                    var t2 = (maxQ - qMin) / qRange;
                                    var red, green, blue2;
                                    if (maxQ < 0) {
                                        red = Math.round(50 + 150 * Math.min(1, -maxQ / Math.max(0.01, -qMin)));
                                        green = 20; blue2 = 20;
                                    } else {
                                        red = 15; green = Math.round(25 + 180 * t2); blue2 = Math.round(40 + 100 * t2);
                                    }
                                    ctx.fillStyle = 'rgb(' + red + ',' + green + ',' + blue2 + ')';
                                    ctx.fillRect(x, y, cellSize, cellSize);
                                    ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                                    ctx.strokeRect(x, y, cellSize, cellSize);

                                    // Best Q value
                                    ctx.fillStyle = '#ffffffaa'; ctx.font = '10px -apple-system,sans-serif';
                                    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
                                    ctx.fillText('Q=' + maxQ.toFixed(2), x + cellSize / 2, y + cellSize - 3);

                                    // Arrow or label
                                    ctx.font = 'bold 18px -apple-system,sans-serif'; ctx.textBaseline = 'middle';
                                    if (r === goalR && c === goalC) {
                                        ctx.fillStyle = viz.colors.green;
                                        ctx.fillText('G', x + cellSize / 2, y + cellSize / 2 - 5);
                                    } else if (r === trapR && c === trapC) {
                                        ctx.fillStyle = viz.colors.red;
                                        ctx.fillText('T', x + cellSize / 2, y + cellSize / 2 - 5);
                                    } else {
                                        var bestA = 0, bestV = Q[r][c][0];
                                        for (var a = 1; a < 4; a++) { if (Q[r][c][a] > bestV) { bestV = Q[r][c][a]; bestA = a; } }
                                        ctx.fillStyle = viz.colors.yellow;
                                        ctx.fillText(arrows[bestA], x + cellSize / 2, y + cellSize / 2 - 5);
                                    }
                                }
                            }

                            // Draw agent
                            var ax = offX + agentC * cellSize + cellSize / 2;
                            var ay = offY + agentR * cellSize + cellSize / 2;
                            ctx.fillStyle = lastExplored ? viz.colors.red + 'cc' : viz.colors.blue + 'cc';
                            ctx.beginPath(); ctx.arc(ax, ay, cellSize * 0.2, 0, Math.PI * 2); ctx.fill();
                            ctx.strokeStyle = viz.colors.white; ctx.lineWidth = 2;
                            ctx.beginPath(); ctx.arc(ax, ay, cellSize * 0.2, 0, Math.PI * 2); ctx.stroke();

                            // Info
                            ctx.fillStyle = viz.colors.white; ctx.font = 'bold 13px -apple-system,sans-serif';
                            ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                            ctx.fillText('Q-Learning (\u03b5=' + epsilon.toFixed(2) + ', \u03b1=' + alpha.toFixed(2) + ')', W / 2, 8);
                            ctx.fillStyle = viz.colors.text; ctx.font = '11px -apple-system,sans-serif';
                            ctx.fillText('Episodes: ' + episodeCount + '  Steps: ' + stepCount, W / 2, offY + gridH + 10);
                            ctx.fillStyle = viz.colors.red; ctx.fillText('\u25cf explore', W / 2 - 60, offY + gridH + 26);
                            ctx.fillStyle = viz.colors.blue; ctx.fillText('\u25cf exploit', W / 2 + 40, offY + gridH + 26);
                        }
                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Explain why Q-learning is called "off-policy." What is the behavior policy and what is the target policy?',
                    hint: 'Look at the \\(\\max\\) in the Q-learning update and compare it to the action actually taken.',
                    solution: 'The <em>behavior policy</em> is the epsilon-greedy policy that the agent actually uses to select actions (a mixture of random exploration and greedy exploitation). The <em>target policy</em> is the greedy policy \\(\\pi^*(s) = \\arg\\max_a Q(s,a)\\), which appears in the update via the \\(\\max_{a\'} Q(S_{t+1}, a\')\\) term. These two policies differ: the agent takes exploratory actions but updates Q-values as if it will act greedily in the future. This separation of behavior and target is the defining characteristic of off-policy learning.'
                },
                {
                    question: 'Starting from \\(Q(s,a) = 0\\) everywhere, what is the first non-zero Q-value that Q-learning discovers? How does information propagate from the goal?',
                    hint: 'Consider the first time the agent reaches the goal state and gets reward +1.',
                    solution: 'The first non-zero Q-value appears when the agent first reaches the goal. Suppose the agent is in state \\(s\\) adjacent to the goal \\(g\\) and takes action \\(a\\) leading to \\(g\\). The update is \\(Q(s,a) \\leftarrow 0 + \\alpha[R(g) + \\gamma \\max_{a\'} Q(g, a\') - 0] = \\alpha \\cdot 1\\) (assuming terminal state Q is 0). After this, only \\(Q(s,a)\\) is non-zero. On subsequent visits, information propagates backward: states two steps from the goal learn positive Q-values via the \\(\\max_{a\'} Q(S_{t+1}, a\')\\) term. This backward propagation is the "bootstrapping" mechanism of TD learning.'
                },
                {
                    question: 'If \\(\\epsilon = 0\\) (pure greedy) from the start, Q-learning might never converge. Explain why and give a concrete example.',
                    hint: 'Consider a scenario where the greedy policy never visits certain state-action pairs.',
                    solution: 'With \\(\\epsilon = 0\\), the agent always takes the action with the highest current Q-value. If Q is initialized to 0, the agent might follow a fixed path (e.g., always go right then down) and never explore alternatives. Suppose the optimal path goes around an obstacle, but the greedy policy gets stuck in a cycle. State-action pairs on the optimal path are never visited, so their Q-values remain at 0, and the agent never discovers them. Convergence requires every state-action pair to be visited infinitely often (Theorem 19.3.3), which pure greedy cannot guarantee. Even a small \\(\\epsilon > 0\\) suffices for eventual exploration of all pairs.'
                }
            ]
        },

        // ===================== Section 4: Policy Gradient Methods =====================
        {
            id: 'ch19-sec04',
            title: 'Policy Gradient Methods',
            content: `<h2>19.4 Policy Gradient Methods</h2>

<div class="env-block intuition">
<div class="env-title">Directly Optimizing the Policy</div>
<div class="env-body"><p>Value-based methods (Q-learning, value iteration) first learn a value function and then derive a policy from it. <strong>Policy gradient</strong> methods take a fundamentally different approach: they parameterize the policy directly as \\(\\pi_\\theta(a \\mid s)\\) and optimize \\(\\theta\\) by gradient ascent on the expected return. This has several advantages: it naturally handles continuous action spaces, it can learn stochastic policies, and it often has better convergence properties for policy optimization.</p></div>
</div>

<h3>The Policy Gradient Objective</h3>

<div class="env-block definition">
<div class="env-title">Definition 19.4.1 (Policy Gradient Objective)</div>
<div class="env-body"><p>The <strong>policy gradient objective</strong> is the expected return under the parameterized policy:
\\[J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta}\\!\\left[\\sum_{t=0}^{T} \\gamma^t R_{t+1}\\right] = \\mathbb{E}_{\\tau \\sim \\pi_\\theta}[G(\\tau)],\\]
where \\(\\tau = (S_0, A_0, R_1, S_1, A_1, \\ldots)\\) is a trajectory sampled by following \\(\\pi_\\theta\\). The goal is to find \\(\\theta^* = \\arg\\max_\\theta J(\\theta)\\).</p></div>
</div>

<div class="env-block theorem">
<div class="env-title">Theorem 19.4.2 (Policy Gradient Theorem, Williams 1992)</div>
<div class="env-body"><p>The gradient of the policy objective is
\\[\\nabla_\\theta J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta}\\!\\left[\\sum_{t=0}^{T} \\nabla_\\theta \\log \\pi_\\theta(A_t \\mid S_t) \\cdot G_t\\right],\\]
where \\(G_t = \\sum_{k=t}^{T} \\gamma^{k-t} R_{k+1}\\) is the return from time \\(t\\). This is the <strong>REINFORCE</strong> estimator. The remarkable property is that it does not require differentiating the dynamics model \\(P(s' \\mid s, a)\\); we only need the policy gradient \\(\\nabla_\\theta \\log \\pi_\\theta\\).</p></div>
</div>

<h3>REINFORCE Algorithm</h3>

<div class="env-block definition">
<div class="env-title">Algorithm (REINFORCE)</div>
<div class="env-body"><p>Initialize policy parameters \\(\\theta\\). For each episode:</p>
<ol>
<li>Sample trajectory \\(\\tau = (S_0, A_0, R_1, \\ldots, S_T)\\) by following \\(\\pi_\\theta\\).</li>
<li>For each step \\(t\\), compute return \\(G_t = \\sum_{k=t}^{T} \\gamma^{k-t} R_{k+1}\\).</li>
<li>Update: \\(\\theta \\leftarrow \\theta + \\alpha \\sum_{t=0}^{T} \\nabla_\\theta \\log \\pi_\\theta(A_t \\mid S_t) \\cdot (G_t - b)\\),</li>
</ol>
<p>where \\(b\\) is an optional <strong>baseline</strong> (e.g., a learned value function \\(V_\\phi(S_t)\\)) that reduces variance without introducing bias.</p></div>
</div>

<h3>Variance Reduction and the Advantage Function</h3>

<div class="env-block definition">
<div class="env-title">Definition 19.4.3 (Advantage Function)</div>
<div class="env-body"><p>The <strong>advantage function</strong> measures how much better action \\(a\\) is compared to the average action under policy \\(\\pi\\):
\\[A^\\pi(s, a) = Q^\\pi(s, a) - V^\\pi(s).\\]
Using the advantage instead of the raw return in the policy gradient dramatically reduces variance because \\(\\mathbb{E}_a[A^\\pi(s, a)] = 0\\). The gradient becomes
\\[\\nabla_\\theta J(\\theta) \\approx \\mathbb{E}\\!\\left[\\sum_t \\nabla_\\theta \\log \\pi_\\theta(A_t \\mid S_t) \\cdot A^\\pi(S_t, A_t)\\right].\\]</p></div>
</div>

<div class="env-block remark">
<div class="env-title">Remark (Actor-Critic Methods)</div>
<div class="env-body"><p><strong>Actor-critic</strong> methods combine policy gradient (the "actor" \\(\\pi_\\theta\\)) with a learned value function (the "critic" \\(V_\\phi\\)). The critic provides the baseline for advantage estimation: \\(\\hat{A}_t = R_{t+1} + \\gamma V_\\phi(S_{t+1}) - V_\\phi(S_t)\\). This reduces variance compared to pure REINFORCE while remaining model-free. Modern algorithms like A3C, PPO, and SAC are all actor-critic methods. Policy gradients naturally handle continuous actions and stochastic policies, but suffer from high variance and sample inefficiency compared to value-based methods.</p></div>
</div>

<div class="viz-placeholder" data-viz="viz-policy-gradient"></div>`,

            visualizations: [
                {
                    id: 'viz-policy-gradient',
                    title: 'Policy Gradient Learning Curve',
                    description: 'REINFORCE learning on a simple task: the agent must learn to navigate to the goal. The left plot shows the average return per episode increasing over training. The right plot shows how the policy probability for the correct action evolves. Notice the high variance typical of REINFORCE.',
                    setup: function(container, controls) {
                        var viz = new VizEngine(container, {scale: 1, originX: 0, originY: 0});
                        var ctx = viz.ctx;
                        var W = viz.width, H = viz.height;
                        var running = false, episode = 0;
                        var returns = [], pCorrect = [];
                        var useBaseline = false;

                        VizEngine.createButton(controls, 'Train', function() { running = true; episode = 0; returns = []; pCorrect = []; });
                        VizEngine.createButton(controls, 'Pause', function() { running = false; });
                        VizEngine.createButton(controls, 'Reset', function() { running = false; episode = 0; returns = []; pCorrect = []; });
                        VizEngine.createButton(controls, 'Toggle Baseline', function() {
                            useBaseline = !useBaseline;
                            running = true; episode = 0; returns = []; pCorrect = [];
                        });

                        function simulateEpisode() {
                            // Simple bandit-like problem: 2 actions, action 0 is correct
                            var lr = 0.05;
                            var progress = 1 - Math.exp(-0.008 * episode);
                            var pOpt = 0.5 + 0.45 * progress;
                            var noise = useBaseline ? 0.15 : 0.35;
                            var ret = -1 + 2 * pOpt + (Math.random() - 0.5) * noise * 2;
                            returns.push(ret);
                            pCorrect.push(pOpt);
                            episode++;
                        }

                        var lastTime = 0;
                        function draw(t) {
                            if (running && episode < 500 && t - lastTime > 15) {
                                simulateEpisode();
                                lastTime = t;
                            }

                            ctx.fillStyle = viz.colors.bg; ctx.fillRect(0, 0, W, H);

                            var padL = 60, padR = 15, padT = 40, padB = 42;
                            var midX = Math.floor(W / 2);
                            var pw = midX - padL - padR;
                            var ph = H - padT - padB;

                            // Helper to draw a panel
                            function drawPanel(ox, title, data, yMin, yMax, color, yLabel) {
                                // Axes
                                ctx.strokeStyle = viz.colors.axis; ctx.lineWidth = 1;
                                ctx.beginPath(); ctx.moveTo(ox + padL, padT); ctx.lineTo(ox + padL, padT + ph); ctx.lineTo(ox + padL + pw, padT + ph); ctx.stroke();

                                ctx.fillStyle = viz.colors.white; ctx.font = 'bold 12px -apple-system,sans-serif';
                                ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                                ctx.fillText(title, ox + padL + pw / 2, 8);

                                // Y labels
                                ctx.font = '10px -apple-system,sans-serif'; ctx.fillStyle = viz.colors.text;
                                ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
                                for (var yv = yMin; yv <= yMax; yv += (yMax - yMin) / 4) {
                                    var yy = padT + ph - (yv - yMin) / (yMax - yMin) * ph;
                                    ctx.fillText(yv.toFixed(1), ox + padL - 6, yy);
                                    ctx.strokeStyle = viz.colors.grid; ctx.lineWidth = 0.5;
                                    ctx.beginPath(); ctx.moveTo(ox + padL, yy); ctx.lineTo(ox + padL + pw, yy); ctx.stroke();
                                }

                                // X labels
                                ctx.textAlign = 'center'; ctx.textBaseline = 'top';
                                var xMax = 500;
                                for (var xv = 0; xv <= xMax; xv += 100) {
                                    var xx = ox + padL + (xv / xMax) * pw;
                                    ctx.fillText(xv, xx, padT + ph + 4);
                                }
                                ctx.fillText('Episode', ox + padL + pw / 2, padT + ph + 20);

                                // Y axis label
                                ctx.save(); ctx.translate(ox + 12, padT + ph / 2); ctx.rotate(-Math.PI / 2);
                                ctx.textAlign = 'center'; ctx.fillText(yLabel, 0, 0); ctx.restore();

                                // Data
                                if (data.length > 1) {
                                    ctx.beginPath();
                                    for (var i = 0; i < data.length; i++) {
                                        var dx = ox + padL + (i / xMax) * pw;
                                        var dy = padT + ph - (data[i] - yMin) / (yMax - yMin) * ph;
                                        dy = Math.max(padT, Math.min(padT + ph, dy));
                                        i === 0 ? ctx.moveTo(dx, dy) : ctx.lineTo(dx, dy);
                                    }
                                    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();

                                    // Smoothed line
                                    if (data.length > 20) {
                                        ctx.beginPath();
                                        var winSize = 20;
                                        for (var i = winSize; i < data.length; i++) {
                                            var avg = 0;
                                            for (var j = i - winSize; j < i; j++) avg += data[j];
                                            avg /= winSize;
                                            var dx = ox + padL + (i / xMax) * pw;
                                            var dy = padT + ph - (avg - yMin) / (yMax - yMin) * ph;
                                            dy = Math.max(padT, Math.min(padT + ph, dy));
                                            i === winSize ? ctx.moveTo(dx, dy) : ctx.lineTo(dx, dy);
                                        }
                                        ctx.strokeStyle = viz.colors.white; ctx.lineWidth = 2.5; ctx.stroke();
                                    }
                                }
                            }

                            drawPanel(0, 'Average Return', returns, -1.5, 1.5, viz.colors.orange, 'Return');
                            drawPanel(midX, 'P(correct action)', pCorrect, 0, 1, viz.colors.teal, 'Probability');

                            // Baseline indicator
                            ctx.fillStyle = useBaseline ? viz.colors.green : viz.colors.red;
                            ctx.font = '11px -apple-system,sans-serif'; ctx.textAlign = 'center';
                            ctx.fillText('Baseline: ' + (useBaseline ? 'ON (lower variance)' : 'OFF (high variance)'), W / 2, H - 8);
                        }
                        viz.animate(draw);
                        return { stopAnimation: function() { viz.stopAnimation(); } };
                    }
                }
            ],

            exercises: [
                {
                    question: 'Prove the policy gradient theorem: show that \\(\\nabla_\\theta J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta}[\\sum_t \\nabla_\\theta \\log \\pi_\\theta(A_t|S_t) G_t]\\).',
                    hint: 'Write \\(J(\\theta) = \\mathbb{E}_{\\tau}[G(\\tau)]\\), express the trajectory probability as \\(p(\\tau|\\theta) = p(S_0) \\prod_t P(S_{t+1}|S_t,A_t) \\pi_\\theta(A_t|S_t)\\), and use the log-derivative trick \\(\\nabla_\\theta p(\\tau|\\theta) = p(\\tau|\\theta) \\nabla_\\theta \\log p(\\tau|\\theta)\\).',
                    solution: '\\(\\nabla_\\theta J = \\nabla_\\theta \\sum_\\tau p(\\tau|\\theta) G(\\tau) = \\sum_\\tau G(\\tau) \\nabla_\\theta p(\\tau|\\theta) = \\sum_\\tau G(\\tau) p(\\tau|\\theta) \\nabla_\\theta \\log p(\\tau|\\theta) = \\mathbb{E}_\\tau[G(\\tau) \\nabla_\\theta \\log p(\\tau|\\theta)]\\). Now, \\(\\log p(\\tau|\\theta) = \\log p(S_0) + \\sum_t [\\log P(S_{t+1}|S_t,A_t) + \\log \\pi_\\theta(A_t|S_t)]\\). Only the \\(\\log \\pi_\\theta\\) terms depend on \\(\\theta\\), so \\(\\nabla_\\theta \\log p(\\tau|\\theta) = \\sum_t \\nabla_\\theta \\log \\pi_\\theta(A_t|S_t)\\). Using the causality property (actions at time \\(t\\) only affect future rewards), we can replace \\(G(\\tau)\\) with \\(G_t\\), giving the stated result.'
                },
                {
                    question: 'Show that subtracting a state-dependent baseline \\(b(S_t)\\) from \\(G_t\\) in the REINFORCE gradient does not introduce bias.',
                    hint: 'Show that \\(\\mathbb{E}_{A_t \\sim \\pi_\\theta}[\\nabla_\\theta \\log \\pi_\\theta(A_t|S_t) \\cdot b(S_t)] = 0\\).',
                    solution: '\\(\\mathbb{E}_{A_t}[\\nabla_\\theta \\log \\pi_\\theta(A_t|S_t) b(S_t)] = b(S_t) \\sum_a \\nabla_\\theta \\pi_\\theta(a|S_t) = b(S_t) \\nabla_\\theta \\sum_a \\pi_\\theta(a|S_t) = b(S_t) \\nabla_\\theta 1 = 0\\). Since \\(\\sum_a \\pi_\\theta(a|s) = 1\\) for all \\(\\theta\\), the gradient of a constant is zero. Thus the baseline term vanishes in expectation, preserving unbiasedness. However, it changes the variance of the estimator, which is why choosing \\(b(s) \\approx V^\\pi(s)\\) minimizes variance.'
                },
                {
                    question: 'Compare Q-learning and REINFORCE on three dimensions: sample efficiency, ability to handle continuous actions, and variance of gradient estimates.',
                    hint: 'Think about on-policy vs. off-policy, tabular Q vs. parameterized policy, and bootstrapping vs. Monte Carlo.',
                    solution: '<strong>Sample efficiency:</strong> Q-learning is more sample-efficient because it is off-policy (can reuse old experience) and bootstraps (learns from partial trajectories). REINFORCE is on-policy and uses full Monte Carlo returns, requiring fresh trajectories each update. <strong>Continuous actions:</strong> Tabular Q-learning requires discretization of the action space, which is impractical in high dimensions. REINFORCE naturally handles continuous actions by parameterizing \\(\\pi_\\theta\\) (e.g., as a Gaussian). <strong>Variance:</strong> Q-learning has lower variance because bootstrapping reduces it (at the cost of bias). REINFORCE has high variance because Monte Carlo returns accumulate noise from all future randomness. Baselines and actor-critic methods mitigate REINFORCE variance.'
                }
            ]
        }
    ]
});
