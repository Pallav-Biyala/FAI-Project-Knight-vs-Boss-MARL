"""Generate publication-grade CS 329 Academic Research Report PDF using headless Chrome.

Two-column layout ('two sides every page'), full black, matching proposal format.
Produces docs/CS329_Knight_vs_Boss_Research_Report.pdf and CS329_Knight_vs_Boss_Research_Report.pdf.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
FIG_DIR = DOCS_DIR / "figures"
OUT_PDF = DOCS_DIR / "CS329_Knight_vs_Boss_Research_Report.pdf"
OUT_HTML = DOCS_DIR / "report.html"
CHROME_EXE = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def fig_uri(filename: str) -> str:
    p = FIG_DIR / filename
    if not p.is_file():
        raise FileNotFoundError(f"Figure not found: {p}")
    return p.as_uri()


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CAN A WEAKER AGENT LEARN TO WIN? ASYMMETRIC CO-ADAPTATION IN MULTI-AGENT COMBAT</title>
<style>
  @page {
    size: letter;
    margin: 0.52in 0.60in 0.52in 0.60in;
    @bottom-center {
      content: counter(page);
      font-family: "Times New Roman", Times, serif;
      font-size: 8.5pt;
      color: #000000;
    }
  }
  * { box-sizing: border-box; }
  body {
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 8.8pt;
    line-height: 1.26;
    color: #000000;
    margin: 0;
    padding: 0;
    text-align: justify;
    hyphens: auto;
  }

  /* Running header like proposal */
  .running-header {
    width: 100%;
    display: flex;
    justify-content: space-between;
    font-size: 8.0pt;
    font-style: italic;
    color: #000000;
    border-bottom: 0.4pt solid #000000;
    padding-bottom: 2pt;
    margin-bottom: 6pt;
  }

  /* Proposal-exact Title Block with Top & Bottom Thick Rules */
  .title-block {
    text-align: center;
    margin-bottom: 8pt;
  }
  .thick-rule {
    border-top: 1.8pt solid #000000;
    margin: 0;
  }
  h1.paper-title {
    font-size: 13.0pt;
    font-weight: bold;
    letter-spacing: 0.3pt;
    line-height: 1.18;
    margin: 4pt 0 2pt 0;
    color: #000000;
    text-transform: uppercase;
  }
  .paper-subtitle {
    font-size: 10.5pt;
    font-weight: bold;
    letter-spacing: 0.2pt;
    margin: 2pt 0;
    color: #000000;
    text-transform: uppercase;
  }
  .paper-course {
    font-size: 8.8pt;
    font-weight: bold;
    margin: 2pt 0 4pt 0;
    color: #000000;
    text-transform: uppercase;
    letter-spacing: 0.2pt;
  }

  /* Author row: 4 equal columns matching proposal */
  .authors-table {
    width: 100%;
    margin: 8pt 0 4pt 0;
    border-collapse: collapse;
  }
  .authors-table td {
    width: 25%;
    text-align: center;
    font-size: 8.6pt;
    padding: 1pt 2pt;
    vertical-align: top;
    color: #000000;
  }
  .author-name {
    font-weight: bold;
    display: block;
    margin-bottom: 1pt;
  }
  .author-id {
    font-family: "Times New Roman", Times, serif;
    font-size: 8.2pt;
    color: #000000;
  }
  .institution-line {
    text-align: center;
    font-size: 7.8pt;
    font-style: italic;
    color: #000000;
    margin-top: 3pt;
    margin-bottom: 8pt;
  }

  /* Abstract box */
  .abstract-container {
    margin: 6pt 16pt 10pt 16pt;
    font-size: 8.4pt;
    line-height: 1.23;
    text-align: justify;
  }
  .abstract-heading {
    text-align: center;
    font-weight: bold;
    font-size: 9.0pt;
    margin-bottom: 3pt;
    text-transform: uppercase;
    letter-spacing: 0.5pt;
  }
  .keywords-line {
    margin-top: 3pt;
    font-size: 8.0pt;
  }

  /* Two-column layout ('two sides every page') */
  .columns-flow {
    column-count: 2;
    column-gap: 16pt;
    text-align: justify;
  }

  /* Full-span wide elements (spanning across both sides/columns) */
  .span-both {
    column-span: all;
    margin: 6pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  h2 {
    font-size: 9.8pt;
    font-weight: bold;
    margin: 7pt 0 2pt 0;
    color: #000000;
    text-align: left;
    page-break-after: avoid;
    break-after: avoid;
  }
  h3 {
    font-size: 8.8pt;
    font-weight: bold;
    margin: 5pt 0 1.5pt 0;
    color: #000000;
    text-align: left;
    page-break-after: avoid;
    break-after: avoid;
  }
  p { margin: 0 0 3.2pt 0; }
  ul, ol { margin: 1pt 0 3.2pt 0; padding-left: 11pt; }
  li { margin-bottom: 1pt; }
  code {
    font-family: Consolas, "Courier New", monospace;
    font-size: 7.5pt;
    color: #000000;
    background-color: #f5f5f5;
    padding: 0.2pt 1.2pt;
  }

  .math-block {
    text-align: center;
    font-size: 8.4pt;
    margin: 2.5pt 0 3.5pt 0;
    padding: 1.5pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
  }

  /* Academic Booktabs Table Styling: full black */
  table.booktabs {
    width: 100%;
    border-collapse: collapse;
    font-size: 7.1pt;
    line-height: 1.12;
    margin: 3pt 0 3pt 0;
    page-break-inside: avoid;
    break-inside: avoid;
    color: #000000;
  }
  table.booktabs th, table.booktabs td {
    padding: 1.6pt 2.5pt;
    border: none;
    color: #000000;
  }
  table.booktabs thead tr:first-child th {
    border-top: 1.2pt solid #000000;
  }
  table.booktabs thead tr:last-child th {
    border-bottom: 0.6pt solid #000000;
    font-weight: bold;
  }
  table.booktabs tbody tr:last-child td {
    border-bottom: 1.2pt solid #000000;
  }
  table.booktabs tr.divider td {
    border-top: 0.4pt solid #000000;
  }
  table.booktabs td.center, table.booktabs th.center { text-align: center; }
  table.booktabs td.right, table.booktabs th.right { text-align: right; }

  .table-title {
    font-size: 7.8pt;
    font-weight: bold;
    margin-top: 4pt;
    margin-bottom: 1pt;
    color: #000000;
    page-break-after: avoid;
    break-after: avoid;
  }
  .table-note {
    font-size: 6.8pt;
    font-style: italic;
    color: #000000;
    margin-top: 1.5pt;
    margin-bottom: 4pt;
    line-height: 1.14;
  }

  .fig-container {
    margin: 5pt 0;
    text-align: center;
    page-break-inside: avoid;
    break-inside: avoid;
  }
  .fig-container img {
    max-width: 100%;
    height: auto;
    border: 0.4pt solid #000000;
  }
  .fig-caption {
    font-size: 7.2pt;
    color: #000000;
    text-align: justify;
    margin-top: 2pt;
    line-height: 1.14;
  }

  .ref-list {
    font-size: 7.7pt;
    line-height: 1.16;
    margin: 2pt 0;
    padding-left: 12pt;
    color: #000000;
  }
  .ref-list li { margin-bottom: 2pt; text-align: left; }
</style>
</head>
<body>

<!-- Title Block: Spans Across Page 1 Matching Proposal Format -->
<div class="title-block">
  <div class="thick-rule"></div>
  <h1 class="paper-title">Can a Weaker Agent Learn to Win?</h1>
  <div class="paper-subtitle">Asymmetric Co-Adaptation in Multi-Agent Combat</div>
  <div class="paper-course">Foundations of Artificial Intelligence &bull; Final Research Report</div>
  <div class="thick-rule"></div>

  <table class="authors-table">
    <tr>
      <td>
        <span class="author-name">Chirag Agarwalla</span>
        <span class="author-id">24110093</span>
      </td>
      <td>
        <span class="author-name">Parth Kale</span>
        <span class="author-id">24110242</span>
      </td>
      <td>
        <span class="author-name">Niraj Kumar</span>
        <span class="author-id">24110222</span>
      </td>
      <td>
        <span class="author-name">Pallav Biyala</span>
        <span class="author-id">24110234</span>
      </td>
    </tr>
  </table>
  <div class="institution-line">Department of Computer Science &amp; Engineering, Indian Institute of Technology Gandhinagar (IITGN)</div>
</div>

<div class="abstract-container">
  <div class="abstract-heading">Abstract</div>
  <p>Can an intrinsically disadvantaged reinforcement learning agent learn to overcome a physically superior adversary through tactical maneuver, and how do multi-agent learning dynamics shape combat capability frontiers? We investigate this question in an 8-action asymmetric combat environment. A learning agent (the Knight) possesses tactical evasive and restorative abilities (<code>DODGE</code> granting <em>u<sub>K</sub></em> = 2 invincibility frames and <code>HEAL</code> restoring +30 HP across <em>q<sub>K</sub></em> = 3 charges), contending against a physically superior adversary (the Boss) armed with telegraphed Area-of-Effect specials (<code>SPECIAL_1</code> 3-tile cardinal beam and <code>SPECIAL_2</code> 5&times;5 radial shockwave). To isolate distinct modes of physical advantage, Boss physical capability is decoupled into independent durability (&alpha;<sub>H</sub>) and lethality (&alpha;<sub>D</sub>) scaling factors evaluated over a 5&times;5 grid (&alpha;<sub>H</sub>, &alpha;<sub>D</sub> &isin; {0.50, 1.00, 1.25, 1.50, 2.00}<sup>2</sup>, with non-uniform step sizes &Delta;&alpha; &isin; {0.25, 0.50}) across three learning regimes: <code>fixed_opponent</code> (1,500 episodes against a scripted heuristic Boss), <code>simultaneous_learning</code> (750 episodes of concurrent PPO updates), and <code>alternating_learning</code> (750 episodes alternating in 125-episode phases). Evaluations employ 3 independent training seeds (0, 1, 2) with 500 evaluation games per seed (1,500 games per cell), complemented by audits against 54 fresh Best-Response PPO Bosses (<code>BR-Boss</code>).</p>
  <p><strong>Central Empirical Findings:</strong></p>
  <ul>
    <li><strong>1. Durability vs. Lethality Asymmetry:</strong> Decoupling shows that below parity, Boss lethality (&alpha;<sub>D</sub>) constrains the Knight more than durability (&alpha;<sub>H</sub>). At equal hit-count ratio (&alpha;<sub>H</sub> &middot; &alpha;<sub>D</sub> = 1.0), the Knight wins 93.0% &plusmn; 2.6% at (2.00, 0.50) but 40.4% &plusmn; 1.0% at (0.50, 2.00) (52.6-point gap). At the parity frontier the two are indistinguishable (crossings 1.17 vs. 1.16). We hypothesize that the fixed +30 HP of <code>HEAL</code> loses value against amplified damage.</li>
    <li><strong>2. Stationary Pursuer Capability Frontier:</strong> On the mean curve across seeds against the heuristic pursuer, standard PPO learns effective tactical maneuver at parity ((1.00, 1.00)), achieving a 74.3% &plusmn; 2.1% win rate. Beyond parity, the durability slice (&alpha;<sub>D</sub> = 1.00) fails at &alpha;<sub>H</sub> = 1.25 (38.3% &plusmn; 17.8%; interpolated 50% crossing 1.17, per-seed spread [1.12, 1.15, 1.29] with Seed 2 maintaining &alpha;*<sub>grid</sub> = 1.25 via an interpolated crossing of 1.29 (58.2% at 1.25)); the lethality slice (&alpha;<sub>H</sub> = 1.00) fails at &alpha;<sub>D</sub> = 1.25 (35.8% &plusmn; 9.9%; crossing 1.16, spread [1.12, 1.18, 1.19]); and the diagonal slice fails at 1.25 (1.7% &plusmn; 0.8%; crossing 1.08, spread [1.08, 1.08, 1.09]).</li>
    <li><strong>3. 2D Contour Geometry:</strong> Exactly 9 cells achieve &ge; 50% win rate, extending along &alpha;<sub>D</sub> = 0.50 up to &alpha;<sub>H</sub> = 2.00 (93.0% &plusmn; 2.6%) and along &alpha;<sub>H</sub> = 0.50 up to &alpha;<sub>D</sub> = 1.50 (77.5% &plusmn; 5.4%). In parameter value space, the 50% contour crosses &alpha;<sub>D</sub> &approx; 0.76 at &alpha;<sub>H</sub> = 1.50 and &alpha;<sub>D</sub> &approx; 0.73 at &alpha;<sub>H</sub> = 2.00 within the bracket &alpha;<sub>D</sub> &in; [0.50, 1.00) (approximate crossings interpolated across a 0.50 step from two grid points: 99.3% &rarr; 2.9% and 93.0% &rarr; 0.0%). The contour never extends beyond parity when both attributes simultaneously satisfy &alpha;<sub>H</sub>, &alpha;<sub>D</sub> &ge; 1.00.</li>
    <li><strong>4. Rule-Based Baseline &amp; Solvability Limits:</strong> In baseline calibration at parity, the scripted Knight defeats the scripted Boss in 100.0% of games (0.0% losses). Beyond parity, the scripted Knight maintains &ge; 50% win rate up to &alpha;<sub>H</sub> = 2.00 (62.8%) on durability and up to &alpha;<sub>D</sub> = 1.50 (84.2%) on lethality, but collapses on the diagonal by &alpha; = 1.50 (9.4%). The failure of PPO beyond parity is consistent with an exploration difficulty against telegraphed AoE specials.</li>
    <li><strong>5. Co-Adaptation Dynamics in Self-Play:</strong> Against the matched partner, the Knight's win-rate threshold falls below 0.50 in all slices (&alpha;*<sub>grid</sub> &lt; 0.50). Matches are dominated by mutual defensive stalling (79.8% &plusmn; 13.0% simultaneous and 84.9% &plusmn; 16.8% alternating timeouts at sub-parity). In alternating learning the Boss trains in the final block against a frozen Knight; at parity this recency is consistent with 98.4% &plusmn; 2.0% Boss wins and only 1.6% timeouts, though schedule order was not tested. When evaluated out-of-distribution against the scripted Boss at sub-parity, these same self-play Knights win 99.8% and 97.5% (though a random Knight already achieves 83.6%, so the baseline bar is low).</li>
    <li><strong>6. Specialization vs. Adversary Exploitation:</strong> Stationary training collapses from 74.3% &rarr; 2.7% &plusmn; 1.7% against <code>BR-Boss</code> at parity, reflecting specialization to predictable heuristic behaviors alongside the aggressive competence of learned adversaries. In self-play, Knights fail to learn winning combat play, achieving near-zero win rates at parity against their partner (0.1% and 0.0%) and the scripted Boss (1.1% and 0.0%), exhibiting no evidence of robustness at parity.</li>
  </ul>
  <div class="keywords-line"><strong>Keywords:</strong> Multi-Agent Reinforcement Learning &middot; Stochastic Games &middot; PPO &middot; Asymmetric Competition &middot; Capability Decoupling &middot; Robustness Audits</div>
</div>

<!-- Two-Column Body Content ('two sides every page') -->
<div class="columns-flow">

  <h2>1 Introduction &amp; Research Problem</h2>
  <p>Competitive multi-agent combat presents a fundamental artificial intelligence dilemma: when two competing agents possess unequal physical attributes&mdash;such as health or attack damage&mdash;can superior tactical intelligence compensate for physical inferiority? In this project, we formalize this problem on a discrete grid where an underdog agent (the Knight) contends against a physically superior adversary (the Boss) within a Markov game framework [1].</p>
  <p>Our investigation centers on three primary questions:</p>
  <ol>
    <li>Under what degree of physical asymmetry can a tactically versatile underdog agent learn to achieve a winning record against a physically superior opponent?</li>
    <li>Which dimension of physical disadvantage&mdash;durability (&alpha;<sub>H</sub>) or lethality (&alpha;<sub>D</sub>)&mdash;imposes the primary constraint on underdog learning?</li>
    <li>How do multi-agent co-adaptation dynamics (simultaneous and alternating learning) differ from stationary training against a fixed rule-based opponent?</li>
  </ol>

  <h3>1.1 Capability Decoupling &amp; Frontier Metrics</h3>
  <p>To evaluate combat capability rigorously, this study establishes two core methodological foundations:</p>
  <ul>
    <li><strong>Decoupling Durability from Lethality:</strong> A single compound scalar conflates two distinct modes of physical advantage. We decouple Boss capability into independent durability (&alpha;<sub>H</sub>) and lethality (&alpha;<sub>D</sub>) scaling factors:
      <div class="math-block">
        <em>H<sub>B</sub></em>(&alpha;<sub>H</sub>) = round(150 &middot; &alpha;<sub>H</sub>), &emsp; <em>D<sub>B</sub></em>(&alpha;<sub>D</sub>) = round(20 &middot; &alpha;<sub>D</sub>)
      </div>
      This decoupling enables independent evaluation of a durability-only slice (&alpha;<sub>D</sub> = 1.00), a lethality-only slice (&alpha;<sub>H</sub> = 1.00), a diagonal slice (&alpha;<sub>H</sub> = &alpha;<sub>D</sub>), and a complete 5&times;5 capability heatmap.
    </li>
    <li><strong>Reporting Critical Thresholds (&alpha;*) with Grid Resolution &amp; Uncertainty:</strong> The critical capability threshold &alpha;* is defined as the maximum parameter value for which the underdog agent maintains &ge; 50% win rate. Because grid evaluation is discrete, we report the local grid resolution &Delta;&alpha; &isin; {0.25, 0.50}, empirical brackets [&alpha;*<sub>grid</sub>, &alpha;<sub>fail</sub>), linearly interpolated 50% crossings (&alpha;*<sub>50%,win</sub>), and empirical per-seed spreads [min, med, max] across seeds [0, 1, 2] (bootstrap intervals are omitted because <em>n</em> = 3). Censored values (&lt; 0.50 or &ge; 2.00) are explicitly marked.
    </li>
  </ul>

  <h2>2 Problem Formulation &amp; Environment Specifications</h2>

  <h3>2.1 Environment Mechanics</h3>
  <p>Combat occurs on a bounded discrete 9&times;9 grid with rigid boundary walls (where out-of-bounds movement attempts are clamped to the current cell, non-toroidal)<sup>1</sup>, following the PettingZoo Parallel API standard [5].</p>
  <ul>
    <li><strong>State Space &amp; Observations:</strong> Agents observe normalized feature vectors encoding own and opponent positions, health, health fractions, attack cooldowns, combat statistics, windups, remaining steps, and mechanism state. Max episode length is <em>T</em> = 300 steps.</li>
    <li><strong>Knight Action Space (&Ascr;<sub>K</sub> = 8):</strong> &Ascr;<sub>K</sub> = {LEFT, RIGHT, UP, DOWN, ATTACK, WAIT, DODGE, HEAL}. Cardinal single-tile steps. Attack deals 20 damage (<em>c<sub>K</sub></em> = 1). <code>DODGE</code> grants <em>u<sub>K</sub></em> = 2 invincibility frames with cooldown <em>c</em><sub>dodge</sub> = 4 steps (zero damage from all attacks). <code>HEAL</code> restores +30 HP (capped at 150) using one of <em>q<sub>K</sub></em> = 3 charges per episode.</li>
    <li><strong>Boss Action Space (&Ascr;<sub>B</sub> = 8):</strong> &Ascr;<sub>B</sub> = {LEFT, RIGHT, UP, DOWN, ATTACK, WAIT, SPECIAL_1, SPECIAL_2}. Basic attack deals <em>D<sub>B</sub></em>(&alpha;<sub>D</sub>) damage (<em>c<sub>B</sub></em> = 2). <code>SPECIAL_1</code> (Cardinal Beam) deals round(50 &middot; &alpha;<sub>D</sub>) damage along a 3-tile cardinal line after a 1-step telegraph windup (<em>z<sub>t</sub></em>). <code>SPECIAL_2</code> (Radial Shockwave) deals round(40 &middot; &alpha;<sub>D</sub>) damage across a 5&times;5 square area centered on the Boss (Chebyshev radius 2) after a 1-step telegraph windup (<em>z<sub>t</sub></em>). Both specials share cooldown <em>c<sub>B</sub></em> = 2.</li>
    <li><strong>Health Scaling:</strong> <em>H<sub>K</sub></em> = 150, <em>H<sub>B</sub></em>(&alpha;<sub>H</sub>) = round(150 &middot; &alpha;<sub>H</sub>).</li>
    <li><strong>Termination &amp; Outcomes:</strong> Knight win: <em>h<sub>B</sub></em> &le; 0 &and; <em>h<sub>K</sub></em> &gt; 0 (<em>r</em><sub>term</sub> = +1). Boss win: <em>h<sub>K</sub></em> &le; 0 &and; <em>h<sub>B</sub></em> &gt; 0 (<em>r</em><sub>term</sub> = -1). Draw: simultaneous knockout (<em>h<sub>K</sub>, h<sub>B</sub></em> &le; 0) or 300-step timeout stalling (<em>r</em><sub>term</sub> = 0).</li>
  </ul>

  <h3>2.2 Reward Formulation &amp; Game Theory</h3>
  <p>The standard shaped reward implements the project's formal reward function, inspired by reward shaping principles [4] (noting that the non-potential max(0, &Delta;h) formulation does not satisfy formal policy invariance):</p>
  <div class="math-block">
    boss_drop = max(0, <em>h<sub>B</sub><sup>(t)</sup></em> - <em>h<sub>B</sub><sup>(t+1)</sup></em>) / <em>H<sub>B</sub></em>(&alpha;<sub>H</sub>),<br>
    knight_drop = max(0, <em>h<sub>K</sub><sup>(t)</sup></em> - <em>h<sub>K</sub><sup>(t+1)</sup></em>) / <em>H<sub>K</sub></em>,<br>
    <em>R<sub>K</sub></em> = <em>r</em><sub>term</sub> + <em>w</em> &middot; (boss_drop - knight_drop), &emsp; <em>R<sub>B</sub></em> = -<em>R<sub>K</sub></em>,
  </div>
  <p>where <em>w</em> = 0.5 (<code>shaping_weight</code>). This formulation is strictly zero-sum (<em>R<sub>K</sub></em> + <em>R<sub>B</sub></em> = 0).</p>

  <h3>2.3 Training Regimes &amp; Architecture</h3>
  <p>Agents are trained using PPO [2] with GAE [3]. Hyperparameters: [128, 128] MLP architecture, learning rate 3&times;10<sup>-4</sup>, discount factor &gamma; = 0.995, GAE parameter &lambda; = 0.95, clip range &epsilon; = 0.2, entropy coefficient &beta;<sub>ent</sub> = 0.02, value coefficient 0.5, batch size 64, and rollout length 1,024 steps.</p>
  <p><strong>Multi-Agent Implementation Architecture:</strong> Because Stable-Baselines3 [6] natively supports only single-agent Gym interfaces, multi-agent co-learning was implemented via a custom PyTorch multi-agent trainer (<code>MultiAgentPPOTrainer</code>) operating over the PettingZoo Parallel environment. In <code>simultaneous_learning</code>, both agents update separate actor-critic networks concurrently after each 1,024-step rollout; in <code>alternating_learning</code>, policy updates alternate in 125-episode blocks.</p>
  <p>Training spans three regimes: (1) <code>fixed_opponent</code> (1,500 episodes against <code>ProposalHeuristicBoss</code>); (2) <code>simultaneous_learning</code> (750 episodes concurrent); and (3) <code>alternating_learning</code> (750 episodes across six 125-episode blocks).</p>
  <p><strong>Alternating Training Order &amp; Best-Response Asymmetry:</strong> In <code>alternating_learning</code>, training begins with the Knight in Block 1 (episodes 0--125) and alternates through six phases: Knight (0--125), Boss (125--250), Knight (250--375), Boss (375--500), Knight (500--625), and Boss (625--750). The Boss trains in the 6th and final block against a frozen late-stage Knight, so the evaluated Boss has a recency advantage. At parity this is consistent with the observed 98.4% &plusmn; 2.0% Boss wins and 1.6% timeouts, which matches a fresh <code>BR-Boss</code> against the same Knights (99.1%, Table 4). At sub-parity it is not sufficient: the partner Boss wins only 5.3% whereas <code>BR-Boss</code> wins 30.7%, so the final-block Boss is not an approximate best response there. We did not test schedule order; a Boss-first schedule, or evaluating the Knight after block 5 against the block-4 Boss, would isolate recency from co-adaptation difficulty.</p>
  <p><strong>Methodological Note on Cross-Regime Confounders:</strong> Comparing &alpha;* across regimes involves two confounders: (1) <em>Opponent class:</em> <code>fixed_opponent</code> faces a stationary bot, whereas self-play regimes face an adaptive co-evolving neural network; and (2) <em>Effective training horizon:</em> Knights receive 1,500 episodes in fixed, 750 in simultaneous, and 375 active training episodes in alternating. Lower self-play thresholds therefore reflect both co-adaptation difficulty and reduced training volume, and our design does not separate them. A control of simultaneous learning for 1,500 episodes would isolate the horizon effect; we leave this to future work.</p>

  <h3>2.4 Deviations from Proposal</h3>
  <ul>
    <li><strong>Evaluation Seeds:</strong> Evaluated across 3 independent seeds ([0, 1, 2]) rather than 5 due to compute constraints.</li>
    <li><strong>Confidence Intervals:</strong> Reported as 3-seed sample standard deviation plus empirical per-seed spreads [min, med, max] (bootstrap intervals omitted because <em>n</em> = 3).</li>
    <li><strong>Parameter Grid:</strong> Discrete grid &alpha; &isin; {0.50, 1.00, 1.25, 1.50, 2.00} rather than {1.0, 1.5, 2.0, 3.0}, enabling sub-parity calibration and finer resolution near parity (&Delta;&alpha; &isin; {0.25, 0.50}).</li>
    <li><strong>Capability Decoupling:</strong> Decoupled Boss physical scaling into independent durability (&alpha;<sub>H</sub>) and lethality (&alpha;<sub>D</sub>) axes rather than a single compound scalar.</li>
    <li><strong>Sample Size per Cell:</strong> Evaluated across 500 games per seed (1,500 games per cell) rather than 100, reducing within-policy evaluation noise (while policy-level variance across the <em>n</em> = 3 seeds is captured by the reported standard deviations and per-seed spreads).</li>
    <li><strong>Three-Opponent Coverage:</strong> Evaluated across 4 key anchor pairs ((0.50, 0.50), (1.00, 1.00), (1.50, 1.00), (1.00, 1.50)) rather than all 25 grid cells.</li>
  </ul>

  <!-- Wide Table 1 Spanning Both Columns -->
  <div class="span-both">
    <div class="table-title">Table 1: Performance Across Key Capability Pairs in Proposal Combat (Mean &plusmn; Std across 3 Seeds, 500 Games/Seed)</div>
    <table class="booktabs">
      <thead>
        <tr>
          <th style="width: 18%;">Capability Pair (&alpha;<sub>H</sub>, &alpha;<sub>D</sub>)</th>
          <th style="width: 28%;">Fixed Opponent (vs Heur Boss)<br>Win / Loss / Draw (TO) [Ep Len]</th>
          <th style="width: 27%;">Simultaneous (vs Matched PPO)<br>Win / Loss / Draw (TO) [Ep Len]</th>
          <th style="width: 27%;">Alternating (vs Matched PPO)<br>Win / Loss / Draw (TO) [Ep Len]</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>(0.50, 0.50)</strong> Sub-Parity</td>
          <td>100.0% &plusmn; 0.0% / 0.0% &plusmn; 0.0% / 0.0% (TO: 0.0%) [13.2 &plusmn; 0.1]</td>
          <td>12.3% &plusmn; 1.4% / 7.7% &plusmn; 11.5% / 80.0% (TO: 79.8%) [280.8 &plusmn; 18.5]</td>
          <td>9.9% &plusmn; 11.5% / 5.3% &plusmn; 5.3% / 84.9% (TO: 84.9%) [278.1 &plusmn; 32.0]</td>
        </tr>
        <tr>
          <td><strong>(1.00, 1.00)</strong> Parity</td>
          <td>74.3% &plusmn; 2.1% / 19.8% &plusmn; 3.3% / 5.9% (TO: 0.0%) [24.4 &plusmn; 0.6]</td>
          <td>0.1% &plusmn; 0.2% / 56.5% &plusmn; 30.7% / 43.4% (TO: 43.4%) [176.5 &plusmn; 90.0]</td>
          <td>0.0% &plusmn; 0.0% / 98.4% &plusmn; 2.0% / 1.6% (TO: 1.6%) [68.2 &plusmn; 19.3]</td>
        </tr>
        <tr>
          <td><strong>(1.25, 1.25)</strong> Asymmetric</td>
          <td>1.7% &plusmn; 0.8% / 97.6% &plusmn; 1.2% / 0.7% (TO: 0.0%) [20.4 &plusmn; 0.5]</td>
          <td>0.0% &plusmn; 0.0% / 96.9% &plusmn; 3.6% / 3.1% (TO: 3.1%) [60.7 &plusmn; 36.1]</td>
          <td>0.0% &plusmn; 0.0% / 96.7% &plusmn; 3.2% / 3.3% (TO: 3.3%) [76.8 &plusmn; 30.1]</td>
        </tr>
        <tr>
          <td><strong>(1.50, 1.00)</strong> 225 HP Boss</td>
          <td>2.9% &plusmn; 0.4% / 96.0% &plusmn; 1.2% / 1.1% (TO: 0.0%) [24.9 &plusmn; 1.0]</td>
          <td>0.0% &plusmn; 0.0% / 56.7% &plusmn; 28.4% / 43.3% (TO: 43.3%) [222.1 &plusmn; 45.8]</td>
          <td>0.0% &plusmn; 0.0% / 97.7% &plusmn; 2.4% / 2.3% (TO: 2.3%) [77.8 &plusmn; 33.8]</td>
        </tr>
        <tr>
          <td><strong>(1.00, 1.50)</strong> 1.5&times; Dmg Boss</td>
          <td>3.8% &plusmn; 2.5% / 94.1% &plusmn; 1.8% / 2.1% (TO: 0.0%) [17.6 &plusmn; 0.8]</td>
          <td>0.1% &plusmn; 0.1% / 92.5% &plusmn; 7.6% / 7.4% (TO: 7.4%) [68.1 &plusmn; 30.1]</td>
          <td>0.0% &plusmn; 0.0% / 98.6% &plusmn; 2.4% / 1.4% (TO: 1.4%) [53.4 &plusmn; 20.7]</td>
        </tr>
        <tr>
          <td><strong>(1.50, 1.50)</strong> Compound Hard</td>
          <td>0.0% &plusmn; 0.0% / 100.0% &plusmn; 0.0% / 0.0% (TO: 0.0%) [18.2 &plusmn; 0.5]</td>
          <td>0.0% &plusmn; 0.0% / 72.3% &plusmn; 32.9% / 27.7% (TO: 27.7%) [126.5 &plusmn; 84.1]</td>
          <td>0.0% &plusmn; 0.0% / 98.7% &plusmn; 2.2% / 1.3% (TO: 1.3%) [42.0 &plusmn; 4.3]</td>
        </tr>
        <tr>
          <td><strong>(2.00, 1.00)</strong> 300 HP Boss</td>
          <td>0.0% &plusmn; 0.0% / 100.0% &plusmn; 0.0% / 0.0% (TO: 0.0%) [25.8 &plusmn; 1.3]</td>
          <td>0.0% &plusmn; 0.0% / 95.6% &plusmn; 6.3% / 4.4% (TO: 4.4%) [85.3 &plusmn; 18.5]</td>
          <td>0.0% &plusmn; 0.0% / 98.5% &plusmn; 1.0% / 1.5% (TO: 1.5%) [89.8 &plusmn; 11.3]</td>
        </tr>
        <tr>
          <td><strong>(1.00, 2.00)</strong> 2.0&times; Dmg Boss</td>
          <td>0.4% &plusmn; 0.2% / 99.5% &plusmn; 0.3% / 0.1% (TO: 0.0%) [14.5 &plusmn; 0.4]</td>
          <td>0.0% &plusmn; 0.0% / 91.9% &plusmn; 12.8% / 8.1% (TO: 8.1%) [88.4 &plusmn; 68.9]</td>
          <td>0.0% &plusmn; 0.0% / 99.2% &plusmn; 1.0% / 0.8% (TO: 0.8%) [36.9 &plusmn; 5.7]</td>
        </tr>
        <tr>
          <td><strong>(2.00, 2.00)</strong> Extreme Hard</td>
          <td>0.0% &plusmn; 0.0% / 100.0% &plusmn; 0.0% / 0.0% (TO: 0.0%) [16.9 &plusmn; 0.9]</td>
          <td>0.0% &plusmn; 0.0% / 95.9% &plusmn; 5.0% / 4.1% (TO: 4.1%) [80.8 &plusmn; 8.0]</td>
          <td>0.0% &plusmn; 0.0% / 99.9% &plusmn; 0.1% / 0.1% (TO: 0.1%) [31.5 &plusmn; 9.2]</td>
        </tr>
      </tbody>
    </table>
    <div class="table-note"><em>Note: In fixed opponent evaluation, all draws are simultaneous knockouts (mutual annihilation at step &lt; 300); timeouts are strictly 0.0%. In self-play, draws are predominantly step-300 timeouts (stalling), reaching 79.8% in simultaneous and 84.9% in alternating at sub-parity.</em></div>
  </div>

  <h2>3 Empirical Results: Slices, Contours &amp; Critical Thresholds</h2>
  <p>Table 1 summarizes performance and mean episode lengths across 9 key capability pairs for all three learning regimes.</p>

  <h3>3.1 Decoupled Capability Slices &amp; 5&times;5 Heatmap</h3>
  <p>Evaluating decoupled capability slices isolates the Knight's tolerance to Boss durability versus lethality:</p>
  <ul>
    <li><strong>Durability Slice (&alpha;<sub>D</sub> = 1.00):</strong> Boss HP scales from 75 to 300 with damage fixed at 20. Win rate falls from 98.1% &plusmn; 1.8% at &alpha;<sub>H</sub> = 0.50 (14.0 &plusmn; 0.7 steps), to 74.3% &plusmn; 2.1% at &alpha;<sub>H</sub> = 1.00 (24.4 &plusmn; 0.6 steps), to 38.3% &plusmn; 17.8% at &alpha;<sub>H</sub> = 1.25 (27.8 &plusmn; 0.5 steps), 2.9% &plusmn; 0.4% at &alpha;<sub>H</sub> = 1.50 (24.9 &plusmn; 1.0 steps), and 0.0% &plusmn; 0.0% at &alpha;<sub>H</sub> = 2.00 (25.8 &plusmn; 1.3 steps). On the mean curve, the 50% crossing occurs at &alpha;*<sub>50%,win</sub> = 1.17; across individual seeds, Seed 2 maintains 58.2% win rate at &alpha;<sub>H</sub> = 1.25 (per-seed crossing spread [1.12, 1.15, 1.29]).</li>
    <li><strong>Lethality Slice (&alpha;<sub>H</sub> = 1.00):</strong> Boss damage scales from 10 to 40 with HP fixed at 150. Win rate falls from 100.0% &plusmn; 0.0% at &alpha;<sub>D</sub> = 0.50 (22.2 &plusmn; 0.9 steps), to 74.3% &plusmn; 2.1% at &alpha;<sub>D</sub> = 1.00 (24.4 &plusmn; 0.6 steps), to 35.8% &plusmn; 9.9% at &alpha;<sub>D</sub> = 1.25 (23.9 &plusmn; 0.8 steps), 3.8% &plusmn; 2.5% at &alpha;<sub>D</sub> = 1.50 (17.6 &plusmn; 0.8 steps), and 0.4% &plusmn; 0.2% at &alpha;<sub>D</sub> = 2.00 (14.5 &plusmn; 0.4 steps) due to rapid lethal strikes (crossing 1.16, spread [1.12, 1.18, 1.19]).</li>
    <li><strong>Diagonal Slice (&alpha;<sub>H</sub> = &alpha;<sub>D</sub>):</strong> Joint scaling drops from 100.0% &plusmn; 0.0% at &alpha; = 0.50 (13.2 &plusmn; 0.1 steps), to 74.3% &plusmn; 2.1% at &alpha; = 1.00 (24.4 &plusmn; 0.6 steps), to 1.7% &plusmn; 0.8% at &alpha; = 1.25 (20.4 &plusmn; 0.5 steps), and 0.0% &plusmn; 0.0% at &alpha; &ge; 1.50 (crossing 1.08, spread [1.08, 1.08, 1.09]).</li>
  </ul>

  <div class="table-title">5&times;5 Win Rate Heatmap (Fixed Opponent vs Scripted Boss)</div>
  <table class="booktabs">
    <thead>
      <tr>
        <th>Boss Dmg \ HP</th>
        <th class="center">0.50</th>
        <th class="center">1.00</th>
        <th class="center">1.25</th>
        <th class="center">1.50</th>
        <th class="center">2.00</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>0.50</strong></td>
        <td class="center">100.0%</td>
        <td class="center">100.0%</td>
        <td class="center">100.0%</td>
        <td class="center">99.3%</td>
        <td class="center">93.0%</td>
      </tr>
      <tr>
        <td><strong>1.00</strong></td>
        <td class="center">98.1%</td>
        <td class="center">74.3%</td>
        <td class="center">38.3%</td>
        <td class="center">2.9%</td>
        <td class="center">0.0%</td>
      </tr>
      <tr>
        <td><strong>1.25</strong></td>
        <td class="center">95.1%</td>
        <td class="center">35.8%</td>
        <td class="center">1.7%</td>
        <td class="center">0.2%</td>
        <td class="center">0.0%</td>
      </tr>
      <tr>
        <td><strong>1.50</strong></td>
        <td class="center">77.5%</td>
        <td class="center">3.8%</td>
        <td class="center">0.3%</td>
        <td class="center">0.0%</td>
        <td class="center">0.0%</td>
      </tr>
      <tr>
        <td><strong>2.00</strong></td>
        <td class="center">40.4%</td>
        <td class="center">0.4%</td>
        <td class="center">0.0%</td>
        <td class="center">0.0%</td>
        <td class="center">0.0%</td>
      </tr>
    </tbody>
  </table>

  <p><strong>Durability vs. Lethality Asymmetry (&alpha;<sub>H</sub> vs. &alpha;<sub>D</sub>):</strong> Because the ratio of hits required to win is &alpha;<sub>H</sub> &middot; &alpha;<sub>D</sub>, mirror cells (&alpha;<sub>H</sub>, &alpha;<sub>D</sub>) &harr; (&alpha;<sub>D</sub>, &alpha;<sub>H</sub>) share the same hit-count ratio, so differences between them isolate which attribute matters more. Below parity, lethality is clearly more punishing: (0.50, 1.25) wins 95.1% vs. 100.0% for (1.25, 0.50); (0.50, 1.50) wins 77.5% vs. 99.3% for (1.50, 0.50); and (0.50, 2.00) wins 40.4% &plusmn; 1.0% vs. 93.0% &plusmn; 2.6% for (2.00, 0.50), a 52.6-point gap. Above parity the asymmetry disappears: (1.00, 1.25) wins 35.8% &plusmn; 9.9% vs. 38.3% &plusmn; 17.8% for (1.25, 1.00) (overlapping seed spreads, crossings 1.16 vs. 1.17), and (1.00, 1.50) vs. (1.50, 1.00) gives 3.8% vs. 2.9%, within seed noise. We therefore conclude that lethality is the dominant constraint <em>in the sub-parity regime, where the Knight can still win</em>, and that the two attributes are indistinguishable at the parity frontier.</p>

  <p>A plausible mechanism is that <code>HEAL</code> restores a fixed +30 HP (+90 HP across 3 charges). Against <em>D<sub>B</sub></em> = 10 one heal absorbs 3 basic hits; against <em>D<sub>B</sub></em> = 40 it absorbs only 0.75 of a hit, and telegraphed specials deal up to 80&ndash;100 damage. This is an untested hypothesis; scaling heal amount with &alpha;<sub>D</sub> at (0.50, 2.00) and checking whether the gap closes would test it directly.</p>

  <p><strong>2D Passing Contour Analysis:</strong> Exactly <strong>9 cells</strong> achieve &ge; 50% win rate: five along &alpha;<sub>D</sub> = 0.50 ((0.50, 0.50) to (2.00, 0.50)), three along &alpha;<sub>H</sub> = 0.50 ((0.50, 1.00) to (0.50, 1.50)), and parity (1.00, 1.00). In parameter value space (Figure 2), the 50% contour crosses &alpha;<sub>D</sub> &approx; 0.76 at &alpha;<sub>H</sub> = 1.50 and &alpha;<sub>D</sub> &approx; 0.73 at &alpha;<sub>H</sub> = 2.00 within the bracket &alpha;<sub>D</sub> &in; [0.50, 1.00) (approximate crossings interpolated across a 0.50 step from two grid points: 99.3% &rarr; 2.9% and 93.0% &rarr; 0.0%), &alpha;<sub>D</sub> = 1.00 at &alpha;<sub>H</sub> = 1.17, and &alpha;<sub>D</sub> = 1.16 at &alpha;<sub>H</sub> = 1.00. The contour never extends beyond parity when both attributes simultaneously satisfy &alpha;<sub>H</sub> &ge; 1.00 and &alpha;<sub>D</sub> &ge; 1.00.</p>

  <div class="fig-container">
    <img src="{fig_proposal_heatmap}" alt="Figure 2: 5x5 Heatmap">
    <div class="fig-caption"><em>Figure 2: 5&times;5 Combat capability heatmap plotted in parameter value space for Fixed Opponent PPO Knight vs. Scripted Heuristic Boss with the 50% win rate contour (&alpha;*) delineated (dashed line; crossings at &alpha;<sub>H</sub> &in; {1.50, 2.00} fall in the [0.50, 1.00) bracket and are approximate due to grid interpolation).</em></div>
  </div>

  <!-- Wide Table 2 Spanning Both Columns -->
  <div class="span-both">
    <div class="table-title">Table 2: Critical Capability Thresholds &alpha;* &amp; Uncertainty Spreads</div>
    <table class="booktabs">
      <thead>
        <tr>
          <th>Regime</th>
          <th>Opponent</th>
          <th>Slice</th>
          <th class="center">&alpha;*<sub>grid</sub></th>
          <th class="center">Local Grid Bracket</th>
          <th class="center">&alpha;*<sub>50%,win</sub></th>
          <th>Per-Seed Spread [min, med, max] / Wilson CI</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code>fixed_opponent</code></td>
          <td>HeuristicBoss</td>
          <td>Diagonal (&alpha;<sub>H</sub> = &alpha;<sub>D</sub>)</td>
          <td class="center">1.00 (fail 1.25)</td>
          <td class="center">[1.00, 1.25) (&Delta;&alpha;=0.25)</td>
          <td class="center">1.08</td>
          <td>[1.08, 1.08, 1.09]</td>
        </tr>
        <tr>
          <td><code>fixed_opponent</code></td>
          <td>HeuristicBoss</td>
          <td>Durability (&alpha;<sub>D</sub> = 1.00)</td>
          <td class="center">1.00 (fail 1.25)</td>
          <td class="center">[1.00, 1.25) (&Delta;&alpha;=0.25)</td>
          <td class="center">1.17</td>
          <td>[1.12, 1.15, 1.29]</td>
        </tr>
        <tr>
          <td><code>fixed_opponent</code></td>
          <td>HeuristicBoss</td>
          <td>Lethality (&alpha;<sub>H</sub> = 1.00)</td>
          <td class="center">1.00 (fail 1.25)</td>
          <td class="center">[1.00, 1.25) (&Delta;&alpha;=0.25)</td>
          <td class="center">1.16</td>
          <td>[1.12, 1.18, 1.19]</td>
        </tr>
        <tr>
          <td><code>simultaneous</code></td>
          <td>Matched PPO</td>
          <td>Diagonal (&alpha;<sub>H</sub> = &alpha;<sub>D</sub>)</td>
          <td class="center">&lt; 0.50 (censored)</td>
          <td class="center">[0.00, 0.50)</td>
          <td class="center">&lt; 0.50</td>
          <td>[&lt; 0.50, &lt; 0.50, &lt; 0.50]</td>
        </tr>
        <tr>
          <td><code>simultaneous</code></td>
          <td>Matched PPO</td>
          <td>Durability (&alpha;<sub>D</sub> = 1.00)</td>
          <td class="center">&lt; 0.50 (censored)</td>
          <td class="center">[0.00, 0.50)</td>
          <td class="center">&lt; 0.50</td>
          <td>[&lt; 0.50, &lt; 0.50, &lt; 0.50]</td>
        </tr>
        <tr>
          <td><code>simultaneous</code></td>
          <td>Matched PPO</td>
          <td>Lethality (&alpha;<sub>H</sub> = 1.00)</td>
          <td class="center">&lt; 0.50 (censored)</td>
          <td class="center">[0.00, 0.50)</td>
          <td class="center">&lt; 0.50</td>
          <td>[&lt; 0.50, &lt; 0.50, &lt; 0.50]</td>
        </tr>
        <tr>
          <td><code>alternating</code></td>
          <td>Matched PPO</td>
          <td>Diagonal (&alpha;<sub>H</sub> = &alpha;<sub>D</sub>)</td>
          <td class="center">&lt; 0.50 (censored)</td>
          <td class="center">[0.00, 0.50)</td>
          <td class="center">&lt; 0.50</td>
          <td>[&lt; 0.50, &lt; 0.50, &lt; 0.50]</td>
        </tr>
        <tr>
          <td><code>alternating</code></td>
          <td>Matched PPO</td>
          <td>Durability (&alpha;<sub>D</sub> = 1.00)</td>
          <td class="center">&lt; 0.50 (censored)</td>
          <td class="center">[0.00, 0.50)</td>
          <td class="center">&lt; 0.50</td>
          <td>[&lt; 0.50, &lt; 0.50, &lt; 0.50]</td>
        </tr>
        <tr>
          <td><code>alternating</code></td>
          <td>Matched PPO</td>
          <td>Lethality (&alpha;<sub>H</sub> = 1.00)</td>
          <td class="center">&lt; 0.50 (censored)</td>
          <td class="center">[0.00, 0.50)</td>
          <td class="center">&lt; 0.50</td>
          <td>[&lt; 0.50, &lt; 0.50, &lt; 0.50]</td>
        </tr>
        <tr class="divider">
          <td>Calibrated HeurKnight</td>
          <td>HeuristicBoss</td>
          <td>Diagonal (&alpha;<sub>H</sub> = &alpha;<sub>D</sub>)</td>
          <td class="center">1.25 (fail 1.50)</td>
          <td class="center">[1.25, 1.50) (&Delta;&alpha;=0.25)</td>
          <td class="center">1.38</td>
          <td>At 1.25: 96.6% [94.6%, 97.9%]; At 1.50: 9.4% [7.1%, 12.3%]</td>
        </tr>
        <tr>
          <td>Calibrated HeurKnight</td>
          <td>HeuristicBoss</td>
          <td>Durability (&alpha;<sub>D</sub> = 1.00)</td>
          <td class="center">&ge; 2.00 (censored)</td>
          <td class="center">[2.00, &infin;)</td>
          <td class="center">&ge; 2.00</td>
          <td>At 2.00: 62.8% [58.5%, 66.9%]</td>
        </tr>
        <tr>
          <td>Calibrated HeurKnight</td>
          <td>HeuristicBoss</td>
          <td>Lethality (&alpha;<sub>H</sub> = 1.00)</td>
          <td class="center">1.50 (fail 2.00)</td>
          <td class="center">[1.50, 2.00) (&Delta;&alpha;=0.50)</td>
          <td class="center">1.74</td>
          <td>At 1.50: 84.2% [80.7%, 87.1%]; At 2.00: 13.0% [10.3%, 16.2%]</td>
        </tr>
      </tbody>
    </table>
    <div class="table-note"><em>Note: For PPO learning regimes, the last column reports empirical per-seed spread [min, med, max] across seeds [0, 1, 2]. For the rule-based Heuristic Knight baseline, evaluations are stochastic across 500 games; the last column reports empirical win rates and Wilson 95% confidence intervals at anchor crossing points.</em></div>
  </div>

  <h3>3.2 Critical Thresholds &amp; Uncertainty Spreads</h3>
  <p>Table 2 presents critical capability thresholds &alpha;* across regimes and slices. Discrete grid thresholds &alpha;*<sub>grid</sub> are derived from the mean curve across seeds, local grid brackets [&alpha;*<sub>grid</sub>, &alpha;<sub>fail</sub>) reflect actual grid steps, linearly interpolated 50% crossings &alpha;*<sub>50%,win</sub> are calculated from neighboring points, and per-seed spreads [min, med, max] capture empirical variation across seeds [0, 1, 2].</p>

  <h3>3.3 Calibrated Baselines &amp; Feasibility Lower Bound</h3>
  <p>To determine whether the failure of PPO beyond &alpha; = 1.00 reflects a fundamental physical barrier or an exploration limitation of standard RL, we evaluate calibrated baselines:</p>
  <ul>
    <li><strong>Scripted-vs-Scripted Calibration Check:</strong> Scripted Knight vs. scripted Boss at parity (1.00, 1.00) yields a 100.0% Knight win rate (0.0% losses across 500 games, Wilson 95% CI [99.2%, 100.0%], mean episode length 21.18 steps), establishing that the scripted Boss is the weaker agent under rule-based play at equal physical scaling.</li>
    <li><strong>Random Knight Baseline (<code>RandomAgent</code>):</strong> Achieves 0.2% win rate (Wilson 95% CI [0.0%, 1.1%]) at parity with average episode length 32.2 steps. At sub-parity (0.50, 0.50), random actions achieve 83.6% win rate (Wilson CI [80.1%, 86.6%], episode length 41.7 steps) against the heuristic Boss. In contrast, co-trained self-play Knights win only 12.3% against their matched partner at (0.50, 0.50). This is not primarily because the partner Boss out-fights them: the Knight loses only 7.7% &plusmn; 11.5% of games, while 79.8% end in step-300 timeouts. The low win rate reflects mutual avoidance under zero-sum health-delta shaping (Limitation 3), not a stronger pursuer. Consistent with this, a fresh <code>BR-Boss</code>, which is trained to force engagement, ends most stalling (about 50.4% draws) and raises the Knight's win rate to 23.4% (Table 4). Comparing a random Knight against the heuristic Boss with a self-play Knight against its partner therefore compares different opponent behaviors, not relative Knight skill.</li>
    <li><strong>Scripted Heuristic Baseline (<code>ProposalHeuristicKnight</code>):</strong> Beyond parity, the heuristic baseline maintains &ge; 50% win rate up to &alpha;<sub>H</sub> = 2.00 on durability (62.8%, Wilson CI [58.5%, 66.9%], episode length 36.0 steps) and up to &alpha;<sub>D</sub> = 1.50 on lethality (84.2%, Wilson CI [80.7%, 87.1%], episode length 21.24 steps), but drops to 9.4% on the diagonal by &alpha; = 1.50 (26.1 steps).</li>
    <li><strong>Feasibility Lower Bound &amp; Exploration Hypothesis:</strong> The heuristic baseline demonstrates that defeating an &alpha;<sub>H</sub> = 2.00 or &alpha;<sub>D</sub> = 1.50 Boss is physically achievable within the Proposal action space, establishing a feasibility lower bound. While the failure of standard PPO at &alpha; = 1.25 is consistent with an exploration difficulty in discovering the precise coordination of evasive invincibility frames against multi-tile telegraphed shockwaves, whether this limitation stems from exploration bottlenecks, reward shaping bias, or value approximation remains an untested hypothesis requiring targeted exploration ablations.</li>
  </ul>

  <!-- Wide Figure 1 Spanning Both Columns -->
  <div class="span-both">
    <div class="fig-container">
      <img src="{fig_proposal_curves}" style="max-height: 200pt;" alt="Figure 1: Frontiers and Training Curves">
      <div class="fig-caption"><em>Figure 1: (a) Knight win rate as a function of the diagonal physical asymmetry parameter &alpha; (&alpha;<sub>H</sub> = &alpha;<sub>D</sub>) across learning regimes and baselines (error bars indicate &plusmn; 1 standard deviation across 3 seeds; Wilson 95% CIs for baselines). (b) Mean training rollout reward trajectories across completed training episodes at parity (1.00, 1.00) for Knight and Boss (shaded areas indicate &plusmn; 1 standard deviation across seeds [0, 1, 2]).</em></div>
    </div>
  </div>

  <h3>3.4 Capability Frontiers and Training Dynamics Curves</h3>
  <p>Figure 1 illustrates combat frontiers and training progression:</p>
  <ul>
    <li><strong>Win-Rate vs. Capability Curves (Panel a):</strong> Against the fixed heuristic pursuer, PPO maintains victory through parity (100.0% at &alpha;=0.50, 74.3% at &alpha;=1.00), before collapsing abruptly to 1.7% at &alpha;=1.25. In simultaneous and alternating learning, Knight win rate remains below 15% even at sub-parity (12.3% and 9.9%), collapsing to 0.1% and 0.0% at parity. In contrast, the calibrated heuristic Knight maintains &ge; 50% win rate up to &alpha;=1.25 (96.6%), crossing 50% at &alpha; = 1.38.</li>
    <li><strong>Training Reward Dynamics (Panel b):</strong> In <code>fixed_opponent</code>, the Knight's rollout reward (sum of transition rewards received across each 1,024-step rollout buffer, &sum; <em>R<sub>K</sub><sup>(t)</sup></em>) begins at -50.07 (Seed 0; -50.07 to -41.77 across seeds, reflecting initial Boss dominance) and trends upward into positive territory (reaching +16.00 in Seed 0, +11.50 in Seed 2) by episode 1,500. In <code>simultaneous_learning</code>, co-training produces non-stationary reward dynamics: the Knight's rollout reward remains persistently suppressed in negative territory (ranging [-39.57, +0.07] across seeds; Seed 0: [-30.83, -0.97]), while the Boss establishes dominance with rollout rewards ranging [-0.07, +39.57] (Seed 0: [+0.97, +30.83]), exactly mirroring the Knight's returns under the zero-sum formulation (<em>R<sub>B</sub></em> = -<em>R<sub>K</sub></em> on each concurrent rollout). In <code>alternating_learning</code>, because the trainer updates and logs only the active agent during its 125-episode phase, Knight and Boss entries reflect distinct rollout buffers at different training intervals rather than concurrent steps. Across active training phases, the Knight's rollout reward ranges between [-27.70, 0.00] across seeds (Seed 0: [-27.27, 0.00]), while the Boss's reward ranges between [0.00, +32.57] (Seed 0: [0.00, +32.57]). These training reward profiles are consistent with final evaluation results, where the Boss wins 56.5% &plusmn; 30.7% (simultaneous) and 98.4% &plusmn; 2.0% (alternating) at parity; in simultaneous learning, 43.4% of final games end in step-300 timeouts, indicating that mutual defensive stalling persists alongside negative Knight training returns.</li>
  </ul>

  <h2>4 Opponent Evaluation &amp; Exploitability Analysis</h2>

  <h3>4.1 Cross-Opponent Evaluation &amp; Three-Opponent Matrix</h3>
  <p>To address the proposal's three-opponent evaluation deliverable, Table 3 compares trained Knight checkpoints evaluated against (a) the scripted Heuristic Boss, (b) the co-trained Matched Boss, and (c) a fresh Best-Response PPO Boss (<code>BR-Boss</code>) across 4 anchor capability pairs.</p>

  <!-- Wide Table 3 Spanning Both Columns -->
  <div class="span-both">
    <div class="table-title">Table 3: Three-Opponent Evaluation Matrix (Knight Win Rate % &plusmn; Std across 3 Seeds)</div>
    <table class="booktabs">
      <thead>
        <tr>
          <th>Capability Anchor (&alpha;<sub>H</sub>, &alpha;<sub>D</sub>)</th>
          <th>Training Regime</th>
          <th class="center">(a) vs Scripted Heuristic Boss</th>
          <th class="center">(b) vs Co-Trained Matched Boss</th>
          <th class="center">(c) vs Fresh BR-Boss</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td rowspan="3"><strong>(0.50, 0.50)</strong> Sub-Parity</td>
          <td><code>fixed_opponent</code></td>
          <td class="center">100.0% &plusmn; 0.0%</td>
          <td class="center">N/A (trained vs heuristic)</td>
          <td class="center">38.0% &plusmn; 8.9%</td>
        </tr>
        <tr>
          <td><code>simultaneous_learning</code></td>
          <td class="center">99.8% &plusmn; 0.2%</td>
          <td class="center">12.3% &plusmn; 1.4%</td>
          <td class="center">23.4% &plusmn; 15.0%</td>
        </tr>
        <tr>
          <td><code>alternating_learning</code></td>
          <td class="center">97.5% &plusmn; 3.8%</td>
          <td class="center">9.9% &plusmn; 11.5%</td>
          <td class="center">25.4% &plusmn; 7.7%</td>
        </tr>
        <tr class="divider">
          <td rowspan="3"><strong>(1.00, 1.00)</strong> Parity</td>
          <td><code>fixed_opponent</code></td>
          <td class="center">74.3% &plusmn; 2.1%</td>
          <td class="center">N/A (trained vs heuristic)</td>
          <td class="center">2.7% &plusmn; 1.7%</td>
        </tr>
        <tr>
          <td><code>simultaneous_learning</code></td>
          <td class="center">1.1% &plusmn; 2.0%</td>
          <td class="center">0.1% &plusmn; 0.2%</td>
          <td class="center">0.1% &plusmn; 0.1%</td>
        </tr>
        <tr>
          <td><code>alternating_learning</code></td>
          <td class="center">0.0% &plusmn; 0.0%</td>
          <td class="center">0.0% &plusmn; 0.0%</td>
          <td class="center">0.0% &plusmn; 0.0%</td>
        </tr>
        <tr class="divider">
          <td rowspan="3"><strong>(1.50, 1.00)</strong> 225 HP Boss</td>
          <td><code>fixed_opponent</code></td>
          <td class="center">2.9% &plusmn; 0.4%</td>
          <td class="center">N/A (trained vs heuristic)</td>
          <td class="center">0.1% &plusmn; 0.2%</td>
        </tr>
        <tr>
          <td><code>simultaneous_learning</code></td>
          <td class="center">0.0% &plusmn; 0.0%</td>
          <td class="center">0.0% &plusmn; 0.0%</td>
          <td class="center">0.0% &plusmn; 0.0%</td>
        </tr>
        <tr>
          <td><code>alternating_learning</code></td>
          <td class="center">0.0% &plusmn; 0.0%</td>
          <td class="center">0.0% &plusmn; 0.0%</td>
          <td class="center">0.0% &plusmn; 0.0%</td>
        </tr>
        <tr class="divider">
          <td rowspan="3"><strong>(1.00, 1.50)</strong> 1.5&times; Dmg Boss</td>
          <td><code>fixed_opponent</code></td>
          <td class="center">3.8% &plusmn; 2.5%</td>
          <td class="center">N/A (trained vs heuristic)</td>
          <td class="center">0.2% &plusmn; 0.3%</td>
        </tr>
        <tr>
          <td><code>simultaneous_learning</code></td>
          <td class="center">0.2% &plusmn; 0.3%</td>
          <td class="center">0.1% &plusmn; 0.1%</td>
          <td class="center">0.0% &plusmn; 0.0%</td>
        </tr>
        <tr>
          <td><code>alternating_learning</code></td>
          <td class="center">0.1% &plusmn; 0.2%</td>
          <td class="center">0.0% &plusmn; 0.0%</td>
          <td class="center">0.0% &plusmn; 0.0%</td>
        </tr>
      </tbody>
    </table>
  </div>

  <p>Evaluating self-play trained Knights against the scripted Heuristic Boss shows that their capability profile depends heavily on opponent type: at sub-parity (0.50, 0.50), self-play Knights defeat the scripted Boss in 99.8% &plusmn; 0.2% (simultaneous) and 97.5% &plusmn; 3.8% (alternating) of games (Table 3, column a). This shows that self-play Knights acquire basic evasive and offensive maneuvers against stationary bots, though a random Knight already wins 83.6% at this anchor, so the bar is low. However, at parity (1.00, 1.00), their inability to win extends beyond their training partners: Simultaneous Knights achieve 1.1% &plusmn; 2.0% win rate (98.6% &plusmn; 2.4% losses, 0.3% &plusmn; 0.5% draws); Alternating Knights achieve 0.0% &plusmn; 0.0% win rate (100.0% &plusmn; 0.0% losses). At parity, self-play Knights fail against the scripted Boss as they do against their partners.</p>

  <h3>4.2 Best-Response Boss (<code>BR-Boss</code>) Exploitability Audit</h3>
  <p>To test whether learned Knight strategies generalize or represent overfit solutions, we trained <strong>54 fresh Best-Response PPO Bosses (<code>BR-Boss</code>)</strong> for 500 episodes each against frozen Knight checkpoints across 6 anchor capability pairs (3 learning regimes &times; 3 seeds).</p>

  <!-- Wide Table 4 Spanning Both Columns -->
  <div class="span-both">
    <div class="table-title">Table 4: Exploitability Audit Against Fresh Best-Response Bosses (<code>BR-Boss</code>)</div>
    <table class="booktabs">
      <thead>
        <tr>
          <th>Capability Anchor (&alpha;<sub>H</sub>, &alpha;<sub>D</sub>)</th>
          <th class="center">Fixed Opponent Knight vs BR-Boss</th>
          <th class="center">Simultaneous Knight vs BR-Boss</th>
          <th class="center">Alternating Knight vs BR-Boss</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>(0.50, 0.50)</strong> Sub-Parity</td>
          <td class="center">38.0% &plusmn; 8.9% (Boss: 37.5%)</td>
          <td class="center">23.4% &plusmn; 15.0% (Boss: 26.2%)</td>
          <td class="center">25.4% &plusmn; 7.7% (Boss: 30.7%)</td>
        </tr>
        <tr>
          <td><strong>(1.00, 1.00)</strong> Parity</td>
          <td class="center">2.7% &plusmn; 1.7% (Boss: 86.4%)</td>
          <td class="center">0.1% &plusmn; 0.1% (Boss: 99.0%)</td>
          <td class="center">0.0% &plusmn; 0.0% (Boss: 99.1%)</td>
        </tr>
        <tr>
          <td><strong>(1.25, 1.25)</strong> Asymmetric</td>
          <td class="center">0.2% &plusmn; 0.2% (Boss: 95.8%)</td>
          <td class="center">0.0% &plusmn; 0.0% (Boss: 99.1%)</td>
          <td class="center">0.0% &plusmn; 0.0% (Boss: 99.7%)</td>
        </tr>
        <tr>
          <td><strong>(1.50, 1.00)</strong> 225 HP Boss</td>
          <td class="center">0.1% &plusmn; 0.2% (Boss: 95.5%)</td>
          <td class="center">0.0% &plusmn; 0.0% (Boss: 97.5%)</td>
          <td class="center">0.0% &plusmn; 0.0% (Boss: 98.2%)</td>
        </tr>
        <tr>
          <td><strong>(1.00, 1.50)</strong> 1.5&times; Dmg Boss</td>
          <td class="center">0.2% &plusmn; 0.3% (Boss: 97.6%)</td>
          <td class="center">0.0% &plusmn; 0.0% (Boss: 97.2%)</td>
          <td class="center">0.0% &plusmn; 0.0% (Boss: 99.2%)</td>
        </tr>
        <tr>
          <td><strong>(1.50, 1.50)</strong> Compound Hard</td>
          <td class="center">0.0% &plusmn; 0.0% (Boss: 98.3%)</td>
          <td class="center">0.0% &plusmn; 0.0% (Boss: 98.7%)</td>
          <td class="center">0.0% &plusmn; 0.0% (Boss: 99.5%)</td>
        </tr>
      </tbody>
    </table>
    <div class="table-note"><em>Note: Values report Knight Win Rate % &plusmn; Std (with Boss Win Rate). Draws make up the remainder.</em></div>
  </div>

  <ul>
    <li><strong>Specialization to Static Pursuer vs. Learned Adversaries:</strong> In <code>fixed_opponent</code>, Knight win rate collapses from <strong>74.3% &rarr; 2.7% &plusmn; 1.7%</strong> at parity under <code>BR-Boss</code>. While this collapse demonstrates that the stationary policy fails to generalize, the decline reflects two compounding factors: (1) the policy has specialized to the predictable behavioral heuristics of the static pursuer, and (2) a trained PPO Boss is fundamentally more capable and aggressive than the rule-based heuristic bot (which the scripted Knight defeats in 100% of games). We hypothesize that <code>BR-Boss</code> exploits timing gaps in the Knight's defensive policy, such as striking with <code>SPECIAL_2</code> shockwaves while <code>DODGE</code> is on cooldown.</li>
    <li><strong>Self-Play Knights Fail to Learn Winning Play:</strong> In <code>simultaneous_learning</code> and <code>alternating_learning</code>, Knights fail to learn winning play at parity, achieving near-zero win rates against both their training partners (0.1% and 0.0%) and <code>BR-Boss</code> (0.1% &plusmn; 0.1% and 0.0% &plusmn; 0.0%). At sub-parity, Knight win rates against <code>BR-Boss</code> rise to 23.4% and 25.4% (compared to 12.3% and 9.9% against their co-trained partners); however, because the Boss's win rate also rises and timeout stalling drops sharply, this reflects forced engagement rather than superior robustness. Across all evaluation opponents, self-play Knights fail to establish winning combat play at parity and exhibit no evidence of robustness.</li>
    <li><strong>Empirical Approximation of Best Response &amp; Forced Engagement:</strong> Because each <code>BR-Boss</code> is trained for 500 episodes against a frozen Knight, it represents an empirical approximation of a best response. At sub-parity, <code>BR-Boss</code> ends most of the stalling (about 50.4% draws against <code>BR-Boss</code> vs. 79.8% against the partner in simultaneous learning), so both sides win more often (Knight 23.4% vs. 12.3%, Boss 26.2% vs. 7.7%). This demonstrates that <code>BR-Boss</code> forces active combat engagement, rather than indicating greater or lesser Knight robustness. At parity and beyond, <code>BR-Boss</code> decisively defeats the Knight (86.4% to 99.1% Boss wins), establishing that these performance collapses represent empirical lower bounds on exploitability.</li>
  </ul>

  <h2>5 Limitations</h2>
  <ol>
    <li><strong>Single-Agent RL Architecture in Multi-Agent Combat:</strong> Training independent PPO agents via standard actor-critic updates treats the opposing player as part of a non-stationary environment. Non-stationarity invalidates standard MDP Markovian convergence guarantees, leading to cyclic policy drift and unstable learning trajectories.</li>
    <li><strong>Co-Learning Policy Cycling &amp; Training Asymmetry:</strong> Single-partner self-play (both simultaneous and alternating) may exhibit cyclic policy drift. In alternating learning, the policy active in the final block possesses an inherent recency advantage. Co-adaptation fails to discover winning combat tactics, and audit evaluations demonstrate no evidence of robustness against external opponents. Developing generalizable combat strategies requires population-based training or league-play mechanisms.</li>
    <li><strong>Stalling to Timeout in Self-Play:</strong> In multi-agent regimes, self-play frequently degenerates into passive stalling: timeout rates reach 79.8% at sub-parity and 43.4% at parity in simultaneous learning. We hypothesize that under zero-sum health-delta reward shaping, agents find that avoiding contact minimizes immediate negative reward from damage trades, leading to passive stalling. A high non-loss rate (1 - Boss Win Rate) in self-play reflects mutual avoidance, not an underdog capability frontier.</li>
    <li><strong>Simulator-Specific Combat Mechanics:</strong> All conclusions are grounded in the bounded discrete grid simulator, wall collision rules, Manhattan attack ranges, Chebyshev AoE shockwave radii, and specific telegraph timing windows (<em>z<sub>t</sub></em> = 1). Results cannot be assumed to generalize directly to continuous-control combat domains without dedicated replication.</li>
  </ol>

  <h2>6 Conclusion</h2>
  <p>Returning to our core research question&mdash;<em>can a weaker agent learn to win against a physically superior adversary?</em>&mdash;our empirical findings provide three primary insights:</p>
  <ol>
    <li><strong>Durability vs. Lethality Asymmetry:</strong> Below parity, Boss lethality constrains underdog performance more than durability: at equal hit-count ratio the Knight wins 93.0% at (2.00, 0.50) but only 40.4% at (0.50, 2.00). At and beyond the parity frontier the two attributes are indistinguishable within seed variation. The fixed-heal mechanism is a hypothesis, not a tested result.</li>
    <li><strong>Learned vs. Heuristic Capability Frontiers:</strong> A calibrated rule-based Knight wins up to &alpha;<sub>H</sub> = 2.00 (62.8%) on durability and up to &alpha;<sub>D</sub> = 1.50 (84.2%) on lethality, but collapses on the diagonal by &alpha; = 1.50 (9.4%). Under standard PPO against a stationary pursuer, the learned frontier on the mean curve across seeds is bounded at parity (&alpha;*<sub>grid</sub> = 1.00, failing at &alpha; = 1.25), though individual seed variation allows Seed 2 to reach &alpha;*<sub>grid</sub> = 1.25 in the durability slice (interpolated crossing 1.29, 58.2% win rate at &alpha;<sub>H</sub> = 1.25). The failure of PPO beyond parity is consistent with an exploration barrier against telegraphed AoE specials.</li>
    <li><strong>Co-Adaptation Failure &amp; Partner Specialization:</strong> Under multi-agent co-adaptation, the Knight's win-rate threshold falls below 0.50 in all slices against its matched partner (&alpha;*<sub>grid</sub> &lt; 0.50), reflecting widespread timeout stalling (43.4% to 84.9%) and alternating training recency, while achieving no evidence of robustness at parity. However, when evaluated out-of-distribution against the scripted Boss at sub-parity, these same self-play Knights achieve &ge; 97.5% win rates (a random Knight wins 83.6% at the same anchor), suggesting that co-adaptation acquires basic combat maneuvers but fails to establish dominance against actively countering adversaries.</li>
  </ol>

  <h2>Acknowledgements</h2>
  <p>We express our sincere gratitude to the course instructor and teaching assistants of CS 329 (Foundations of Artificial Intelligence, Semester-I, 2026&ndash;2027) at IIT Gandhinagar for their insightful feedback, methodological guidance, and support throughout this course project.</p>

  <h2>References</h2>
  <ol class="ref-list">
    <li>M. L. Littman, "Markov games as a framework for multi-agent reinforcement learning," in <em>Proceedings of the Eleventh International Conference on Machine Learning (ICML)</em>, pp. 157&ndash;163, 1994.</li>
    <li>J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," <em>arXiv preprint arXiv:1707.06347</em>, 2017.</li>
    <li>J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, "High-dimensional continuous control using generalized advantage estimation," in <em>Proceedings of the International Conference on Learning Representations (ICLR)</em>, 2016.</li>
    <li>A. Y. Ng, D. Harada, and S. Russell, "Policy invariance under reward transformations: Theory and application to reward shaping," in <em>Proceedings of the Sixteenth International Conference on Machine Learning (ICML)</em>, pp. 278&ndash;287, 1999.</li>
    <li>J. K. Terry et al., "PettingZoo: Gym for multi-agent reinforcement learning," in <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, 2021.</li>
    <li>A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable reinforcement learning implementations," <em>Journal of Machine Learning Research</em>, vol. 22, no. 268, pp. 1&ndash;8, 2021.</li>
  </ol>

</div>

</body>
</html>
"""


def main() -> None:
    html = HTML_TEMPLATE
    html = html.replace("{fig_proposal_curves}", fig_uri("fig_proposal_curves.png"))
    html = html.replace("{fig_proposal_heatmap}", fig_uri("fig_proposal_heatmap.png"))

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote report HTML to {OUT_HTML}")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(CHROME_EXE),
        "--headless=new",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUT_PDF}",
        OUT_HTML.as_uri(),
    ]
    print(f"Running Chrome to generate PDF...")
    subprocess.run(cmd, check=True)
    size = OUT_PDF.stat().st_size
    print(f"Generated PDF successfully ({size:,} bytes): {OUT_PDF}")

    # Copy to root and brain artifact directory
    root_pdf = PROJECT_ROOT / "CS329_Knight_vs_Boss_Research_Report.pdf"
    shutil.copy2(OUT_PDF, root_pdf)
    print(f"Copied to root: {root_pdf}")

    artifact_pdf = Path(
        r"C:\Users\agarw\.gemini\antigravity\brain\339fefc9-8769-4e0d-b2c4-bb5caccb9aba\CS329_Knight_vs_Boss_Research_Report.pdf"
    )
    shutil.copy2(OUT_PDF, artifact_pdf)
    print(f"Copied to artifact dir: {artifact_pdf}")


if __name__ == "__main__":
    main()
