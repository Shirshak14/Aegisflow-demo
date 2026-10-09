// Motion layer for the dashboard: Lenis smooth scroll, GSAP entrances, and React Bits-style effects
// (CountUp, ShinyText, ClickSpark). Vendored libs, no build step. Purely cosmetic:
// nothing here touches data, and every effect is skipped when the user prefers reduced motion.
"use strict";
(function () {
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const FX = (window.FX = { reduced });

  // ---- Lenis smooth scroll (tables with .scroll keep native wheel scrolling via data-lenis-prevent)
  if (!reduced && window.Lenis) {
    const lenis = new Lenis({ lerp: 0.1, wheelMultiplier: 1 });
    gsap.ticker.add(time => lenis.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);
    FX.lenis = lenis;
  }

  // ---- GSAP entrance: cards fade/slide in, staggered. Call once the cards exist.
  FX.revealCards = function () {
    if (reduced) return;
    const cards = document.querySelectorAll("main > .card");
    gsap.fromTo(cards, { y: 18, opacity: 0 },
      { y: 0, opacity: 1, duration: 0.6, ease: "power3.out", stagger: 0.05, clearProps: "transform,opacity" });
  };

  // ---- ClickSpark: short radial sparks where you click
  if (!reduced) {
    const canvas = document.createElement("canvas");
    canvas.className = "fx-sparks";
    document.body.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    let sparks = [], running = false;
    const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
    fit(); addEventListener("resize", fit);
    const color = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#4aa3ff";
    const step = now => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      sparks = sparks.filter(s => now - s.t0 < 300);
      for (const s of sparks) {
        const p = (now - s.t0) / 300, e = 1 - (1 - p) * (1 - p);
        const d = 4 + 9 * e, len = 5 * (1 - e);
        ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.6 * (1 - p);
        ctx.beginPath();
        ctx.moveTo(s.x + d * Math.cos(s.a), s.y + d * Math.sin(s.a));
        ctx.lineTo(s.x + (d + len) * Math.cos(s.a), s.y + (d + len) * Math.sin(s.a));
        ctx.stroke();
      }
      if (sparks.length) requestAnimationFrame(step); else running = false;
    };
    document.addEventListener("click", e => {
      const t0 = performance.now();
      for (let i = 0; i < 6; i++) sparks.push({ x: e.clientX, y: e.clientY, a: (Math.PI * 2 * i) / 6, t0 });
      if (!running) { running = true; requestAnimationFrame(step); }
    });
  }

  // ---- React components (htm). Exposed on FX so app.js can use them.
  const { useEffect, useRef } = React;
  const html = htm.bind(React.createElement);

  // CountUp: tweens the displayed number to the new value; formats with toLocaleString.
  FX.CountUp = function CountUp({ value, decimals = 0 }) {
    const ref = useRef(null), shown = useRef(0);
    const fmt = v => Number(v).toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    useEffect(() => {
      const el = ref.current, to = +value || 0;
      if (reduced) { shown.current = to; el.textContent = fmt(to); return; }
      const o = { v: shown.current };
      const tw = gsap.to(o, { v: to, duration: 0.8, ease: "power2.out",
        onUpdate: () => { shown.current = o.v; el.textContent = fmt(o.v); } });
      return () => tw.kill();
    }, [value, decimals]);
    return html`<span ref=${ref}>${fmt(0)}</span>`;
  };

  // ShinyText: a light sweep across the text (CSS animation, see app.css .shiny)
  FX.ShinyText = function ShinyText({ children }) {
    return html`<span class=${reduced ? "" : "shiny"}>${children}</span>`;
  };
})();
