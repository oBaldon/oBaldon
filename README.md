<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Prompt Runner — IA Edition</title>
  <meta name="description" content="Desvie de tokens e alucinações, colete dados limpos e treine seu modelo!">
  <style>
    :root {
      --bg: #070b18; --panel:#0e1430; --ink:#e6edf3; --muted:#8aa1b1;
      --acc:#71e3a2; --warn:#ffd166; --danger:#ff6b6b; --neon:#39d353;
    }
    *{box-sizing:border-box}
    html,body{height:100%;margin:0;background:radial-gradient(1000px 700px at 20% 10%,#0b1330,var(--bg));color:var(--ink);font:500 15px/1.5 Inter,system-ui,Segoe UI,Roboto,Ubuntu,Arial}
    .wrap{min-height:100%;display:grid;place-items:center;padding:20px}
    .frame{width:min(95vw,540px);aspect-ratio:9/16;position:relative;border-radius:24px;overflow:hidden;
      background:linear-gradient(180deg,#0b1126 0%, #0a0f22 60%, #070b18 100%);border:1px solid #1f2a44;
      box-shadow:0 12px 34px rgba(0,0,0,.45), inset 0 0 0 1px rgba(255,255,255,.03)}
    canvas{width:100%;height:100%;display:block}
    .hud{position:absolute;inset:0;pointer-events:none;padding:10px}
    .row{display:flex;justify-content:space-between;gap:8px}
    .badge{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.08);padding:6px 10px;border-radius:999px;font-size:13px}
    .center{position:absolute;inset:0;display:grid;place-items:center;text-align:center;padding:16px}
    .title{font-weight:900;letter-spacing:.3px;font-size:24px}
    .muted{color:var(--muted)}
    .btn{pointer-events:auto;display:inline-flex;align-items:center;justify-content:center;gap:.6rem;
      padding:.8rem 1.1rem;border-radius:14px;border:1px solid #243357;background:linear-gradient(#152247,#0e1833);
      box-shadow:0 6px 14px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.05);color:var(--ink);font-weight:800;cursor:pointer;
      transition:transform .08s ease,border-color .2s ease}
    .btn:hover{transform:translateY(-1px);border-color:#3a5fb3}
    .btn:active{transform:translateY(0)}
    .hint{position:absolute;left:50%;bottom:8px;translate:-50% 0;font-size:12px;color:var(--muted)}
    .terminal{position:absolute;left:8px;right:8px;bottom:40px;max-height:26%;overflow:auto;
      background:rgba(3,6,14,.7);border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:8px;display:none}
    .log{font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size:12px; white-space:pre-wrap}
    .controls{position:absolute;left:50%;bottom:12px;translate:-50% 0;display:grid;gap:8px;pointer-events:auto}
    .pad{display:grid;grid-template-columns:repeat(3,52px);grid-template-rows:repeat(2,52px);gap:8px;place-items:center}
    .key{width:52px;height:52px;border-radius:14px;border:1px solid #243357;background:linear-gradient(#152247,#0e1833);
      box-shadow:inset 0 0 0 1px rgba(255,255,255,.05);display:grid;place-items:center;font-weight:900}
    @media (hover:hover) and (pointer:fine){ .controls{display:none} }
    .pulse{animation:pulse 1.3s ease-in-out infinite}
    @keyframes pulse{0%,100%{opacity:.55}50%{opacity:1}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="frame">
      <canvas id="game" width="360" height="640" aria-label="Prompt Runner — IA Edition"></canvas>

      <div class="hud">
        <div class="row">
          <div class="badge" id="score">Score: 0</div>
          <div class="badge" id="best">Best: 0</div>
        </div>
      </div>

      <div class="center" id="overlay">
        <div>
          <div class="title">PROMPT RUNNER — IA</div>
          <p class="muted" style="margin:.25rem 0 .75rem">Você é um LLM correndo no fluxo de tokens.<br>Use ← → (ou A/D). Espaço pausa.</p>
          <button id="play" class="btn">▶ Treinar e Rodar</button>
        </div>
      </div>

      <div class="terminal" id="terminal"><div class="log" id="log"></div></div>

      <div class="controls" id="touchControls" aria-hidden="true">
        <div class="pad">
          <div></div>
          <div class="key" data-dir="left">◀</div>
          <div></div>
          <div></div>
          <div class="key" data-dir="right">▶</div>
          <div></div>
        </div>
      </div>

      <div class="hint pulse">Feito com ❤️ + JS puro · Tema: IA</div>
    </div>
  </div>

  <script>
  (() => {
    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    const scoreEl = document.getElementById('score');
    const bestEl  = document.getElementById('best');
    const overlay = document.getElementById('overlay');
    const playBtn = document.getElementById('play');
    const term    = document.getElementById('terminal');
    const logEl   = document.getElementById('log');

    // Utils
    const clamp = (v,a,b)=>Math.max(a,Math.min(b,v));
    const rand  = (a,b)=>Math.random()*(b-a)+a;
    const choice= arr=>arr[(Math.random()*arr.length)|0];

    // State
    let running=false, paused=false, tPrev=0, acc=0, score=0, best=+localStorage.getItem('pr_best')||0;
    const keys = new Set();
    const obstacles=[], pickups=[];
    let spawnObT=0, spawnPkT=0;

    const player = {x: W/2-12, y: H-80, w: 24, h: 24, vx:0, speed: 260};

    // Content
    const obstacleKinds = [
      {label:'TOKEN', color:'#6ea5ff', minW:18, maxW:42, speed: [120,220]},
      {label:'ALUCINAÇÃO', color:'#ff6b6b', minW:24, maxW:44, speed: [140,240]},
      {label:'CONTEXT\nOVERFLOW', color:'#f97316', minW:26, maxW:46, speed: [160,260]},
    ];
    const pickupKinds = [
      {label:'DADOS\nLIMPOS', color:'#71e3a2', bonus:15, speed:[100,160]},
      {label:'RLHF', color:'#ffd166', bonus:25, speed:[120,180]},
      {label:'GPU', color:'#a78bfa', bonus:20, speed:[120,180]},
    ];

    const crashMessages = [
`RuntimeError: ContextWindowOverflow
  at Decoder.forward (/model/llm.js:256:17)
  hint: trunque seu prompt ou aumente a janela`,
`ValidationError: BiasDetected
  at EthicsGuard.check (/policy/rlhf.js:88:5)
  hint: normalize dados e re-treine`,
`NaNError: Loss became NaN
  cause: learning_rate muito alta
  fix: reduza LR e reinicie o treino`,
`HallucinationException: unsupported fact
  at Generator.sample (/model/sampling.js:133:9)
  note: cite fontes ou ajuste temperatura`,
    ];

    // Input
    addEventListener('keydown', e=>{
      if(['ArrowLeft','ArrowRight','a','d','A','D',' '].includes(e.key)) e.preventDefault();
      if(e.key===' ' && running){ paused=!paused; overlay.style.display=paused?'grid':'none'; }
      if(['ArrowLeft','a','A'].includes(e.key)) keys.add('left');
      if(['ArrowRight','d','D'].includes(e.key)) keys.add('right');
    });
    addEventListener('keyup', e=>{
      if(['ArrowLeft','a','A'].includes(e.key)) keys.delete('left');
      if(['ArrowRight','d','D'].includes(e.key)) keys.delete('right');
    });
    // touch
    document.querySelectorAll('.key').forEach(btn=>{
      const dir=btn.dataset.dir;
      const on = ()=>keys.add(dir);
      const off= ()=>keys.delete(dir);
      btn.addEventListener('touchstart',e=>{e.preventDefault();on()},{passive:false});
      btn.addEventListener('touchend',e=>{e.preventDefault();off()},{passive:false});
      btn.addEventListener('touchcancel',off);
      btn.addEventListener('mousedown',on);
      btn.addEventListener('mouseup',off);
      btn.addEventListener('mouseleave',off);
    });

    // Core
    function reset(){
      obstacles.length=0; pickups.length=0;
      player.x=W/2-12; player.vx=0;
      score=0; acc=0; spawnObT=0; spawnPkT=0; paused=false;
      scoreEl.textContent='Score: 0';
      term.style.display='none';
      logEl.textContent='';
    }

    function rect(a,b){ return a.x < b.x+b.w && a.x+a.w > b.x && a.y < b.y+b.h && a.y+a.h > b.y; }

    function spawnObstacle(){
      const k = choice(obstacleKinds);
      const w = rand(k.minW, k.maxW);
      const x = rand(8, W-8-w);
      const vy = rand(k.speed[0], k.speed[1]) + Math.min(score*0.4, 180);
      obstacles.push({x, y:-w, w, h:w, vy, kind:k});
      if(obstacles.length>80) obstacles.shift();
    }
    function spawnPickup(){
      const k = choice(pickupKinds);
      const s = 18;
      const x = rand(8, W-8-s);
      const vy = rand(k.speed[0], k.speed[1]);
      pickups.push({x, y:-s, w:s, h:s, vy, kind:k});
      if(pickups.length>30) pickups.shift();
    }

    function drawBG(){
      // deep space grid
      ctx.fillStyle='#0a1024';
      ctx.fillRect(0,0,W,H);
      ctx.globalAlpha=0.6;
      for(let i=0;i<18;i++){
        const y=(i*36 + (performance.now()/20)%36);
        ctx.fillStyle='rgba(255,255,255,.05)';
        ctx.fillRect(0,y,W,1);
      }
      ctx.globalAlpha=1;
      // edge glow
      const g=ctx.createLinearGradient(0,0,W,0);
      g.addColorStop(0,'rgba(57,211,83,.18)');
      g.addColorStop(1,'rgba(167,139,250,.18)');
      ctx.fillStyle=g; ctx.fillRect(0,0,W,6);
    }

    function drawPlayer(){
      // chip-like
      ctx.fillStyle = '#39d353';
      ctx.fillRect(player.x, player.y, player.w, player.h);
      // contacts
      ctx.fillStyle='rgba(255,255,255,.15)';
      ctx.fillRect(player.x-3, player.y+6, 3, 4);
      ctx.fillRect(player.x+player.w, player.y+6, 3, 4);
      ctx.fillRect(player.x-3, player.y+player.h-10, 3, 4);
      ctx.fillRect(player.x+player.w, player.y+player.h-10, 3, 4);
    }

    function drawObstacle(o){
      ctx.fillStyle=o.kind.color;
      ctx.fillRect(o.x,o.y,o.w,o.h);
      // label
      ctx.fillStyle='rgba(0,0,0,.28)';
      ctx.font='700 9px ui-monospace,monospace';
      const lines = String(o.kind.label).split('\n');
      for(let i=0;i<lines.length;i++){
        ctx.fillText(lines[i], o.x+3, o.y+10+i*10);
      }
    }
    function drawPickup(p){
      ctx.fillStyle=p.kind.color;
      ctx.fillRect(p.x,p.y,p.w,p.h);
      ctx.fillStyle='rgba(0,0,0,.3)';
      ctx.font='700 9px ui-monospace,monospace';
      const lines = String(p.kind.label).split('\n');
      for(let i=0;i<lines.length;i++){
        ctx.fillText(lines[i], p.x+3, p.y+10+i*10);
      }
    }

    function update(dt){
      if(paused) return;

      // move player
      player.vx=0;
      if(keys.has('left'))  player.vx -= player.speed;
      if(keys.has('right')) player.vx += player.speed;
      player.x += player.vx * dt;
      player.x = clamp(player.x, 8, W - player.w - 8);

      // spawn pacing
      spawnObT += dt; spawnPkT += dt;
      const obEvery = Math.max(0.18, 0.9 - score*0.01);
      if(spawnObT > obEvery){ spawnObstacle(); spawnObT=0; }
      if(spawnPkT > 1.1){ spawnPickup(); spawnPkT=0; }

      // move
      for(const o of obstacles) o.y += o.vy * dt;
      for(const p of pickups)   p.y += p.vy * dt;

      // cull
      while(obstacles.length && obstacles[0].y>H+60) obstacles.shift();
      while(pickups.length && pickups[0].y>H+40) pickups.shift();

      // collisions
      for(const o of obstacles){
        if(rect(player,o)){ return gameOver(o.kind.label); }
      }
      for(let i=pickups.length-1;i>=0;i--){
        const p = pickups[i];
        if(rect(player,p)){
          score += p.kind.bonus;
          pickups.splice(i,1);
          scoreEl.textContent='Score: '+score;
        }
      }

      // passive score
      acc += dt;
      if(acc >= 0.1){ score += 1; acc=0; scoreEl.textContent='Score: '+score; }
    }

    function render(){
      drawBG();
      for(const p of pickups)   drawPickup(p);
      for(const o of obstacles) drawObstacle(o);
      drawPlayer();

      // glass top
      const g=ctx.createLinearGradient(0,0,0,120);
      g.addColorStop(0,'rgba(255,255,255,.06)');
      g.addColorStop(1,'rgba(255,255,255,0)');
      ctx.fillStyle=g; ctx.fillRect(0,0,W,120);
    }

    function loop(ts){
      if(!running) return;
      const dt = Math.min(1/30, (ts - tPrev)/1000 || 0);
      tPrev = ts;
      update(dt);
      render();
      requestAnimationFrame(loop);
    }

    function start(){
      reset();
      bestEl.textContent='Best: '+best;
      overlay.style.display='none';
      running=true;
      tPrev=performance.now();
      requestAnimationFrame(loop);
    }

    function logCrash(label){
      term.style.display='block';
      const msg = choice(crashMessages);
      const stamp = new Date().toLocaleTimeString();
      logEl.textContent =
`[${stamp}] FATAL: ${label}
${msg}

--- diag ---
temp=0.9  top_p=0.95  lr=3e-4  batch=64
checkpoint: v${1 + (best%7)}
`;
      term.scrollTop = term.scrollHeight;
    }

    function gameOver(label){
      running=false;
      best = Math.max(best, score);
      localStorage.setItem('pr_best', best);
      bestEl.textContent='Best: '+best;
      logCrash(label);
      overlay.innerHTML = `
        <div>
          <div class="title">💥 Falha no Inference</div>
          <p class="muted" style="margin:.25rem 0 1rem">Score: <strong>${score}</strong> · Recorde: <strong>${best}</strong></p>
          <button id="play" class="btn">↻ Re-treinar e Tentar de novo</button>
        </div>`;
      overlay.style.display='grid';
      overlay.querySelector('#play').addEventListener('click', start);
    }

    playBtn.addEventListener('click', start);
  })();
  </script>
</body>
</html>
