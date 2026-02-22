import { useEffect, useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { usePipelineStore } from '../../store/pipeline';

export default function PrismHero() {
  const currentStage = usePipelineStore((s) => s.currentStage);

  const [animKey, setAnimKey] = useState(0);
  const [beamActive, setBeamActive] = useState(false);
  const [splitActive, setSplitActive] = useState(false);

  const runAnimation = useCallback(() => {
    setBeamActive(false);
    setSplitActive(false);
    setAnimKey((k) => k + 1);

    const t1 = setTimeout(() => setBeamActive(true), 400);
    const t2 = setTimeout(() => setSplitActive(true), 1100);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, []);

  // Re-trigger on mount and on every stage change
  useEffect(() => {
    const cleanup = runAnimation();
    return cleanup;
  }, [currentStage, runAnimation]);

  const rainbowColors = [
    '#ef4444',
    '#f97316',
    '#eab308',
    '#22c55e',
    '#3b82f6',
    '#8b5cf6',
    '#d946ef',
  ];

  const prismW = 360;
  const prismH = 120;
  const depth = 80;

  const letters = ['P', 'R', 'I', 'S', 'M'];

  return (
    <div className="relative w-full mb-8 select-none overflow-hidden">
      {/* Background glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div
          className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[350px] rounded-full transition-opacity duration-[2000ms] ${
            splitActive ? 'opacity-30' : 'opacity-0'
          }`}
          style={{
            background:
              'radial-gradient(ellipse, rgba(99,102,241,0.25) 0%, rgba(139,92,246,0.1) 40%, transparent 70%)',
          }}
        />
        <div
          className={`absolute top-1/2 left-[58%] -translate-y-1/2 w-[500px] h-[250px] rounded-full transition-opacity duration-[2500ms] ${
            splitActive ? 'opacity-20' : 'opacity-0'
          }`}
          style={{
            background:
              'conic-gradient(from 180deg, #ef444440, #f9731640, #eab30840, #22c55e40, #3b82f640, #8b5cf640, #d946ef40, transparent)',
            filter: 'blur(50px)',
          }}
        />
      </div>

      <div key={animKey} className="relative flex flex-col items-center py-10">
        <div className="relative h-[220px] w-full max-w-4xl flex items-center justify-center scale-[0.55] sm:scale-75 md:scale-90 lg:scale-100 origin-center">

          {/* ── Incoming white beam ── */}
          <motion.div
            initial={{ scaleX: 0, opacity: 0 }}
            animate={beamActive ? { scaleX: 1, opacity: 1 } : { scaleX: 0, opacity: 0 }}
            transition={{ duration: 0.7, ease: 'easeOut' }}
            className="absolute right-1/2 mr-[180px] top-1/2 -translate-y-1/2 origin-right"
            style={{ width: '300px', height: '5px' }}
          >
            <div
              className="absolute inset-0 rounded-full"
              style={{
                background:
                  'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.03) 15%, rgba(255,255,255,0.4) 65%, #fff 100%)',
                boxShadow:
                  '0 0 25px rgba(255,255,255,0.25), 0 0 50px rgba(255,255,255,0.1)',
              }}
            />
            <div
              className="absolute top-[1px] left-0 right-0 h-[2px] rounded-full"
              style={{
                background:
                  'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.15) 30%, #fff 100%)',
              }}
            />
          </motion.div>

          {/* ── Beam hitting left face — glow on entry ── */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={beamActive ? { opacity: 1 } : { opacity: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="absolute z-20 pointer-events-none"
            style={{
              left: 'calc(50% - 195px)',
              top: '50%',
              transform: 'translate(-50%, -50%)',
              width: '30px',
              height: '60px',
              background: 'radial-gradient(ellipse, rgba(255,255,255,0.4) 0%, transparent 70%)',
              filter: 'blur(6px)',
            }}
          />

          {/* ══════ SOLID 3D WHITE TRANSLUCENT PRISM ══════ */}
          <motion.div
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 0.15 }}
            className="relative z-10"
            style={{ perspective: '900px' }}
          >
            <div
              style={{
                transformStyle: 'preserve-3d',
                transform: 'rotateX(-14deg) rotateY(-22deg)',
                width: `${prismW}px`,
                height: `${prismH}px`,
                position: 'relative',
              }}
            >
              {/* ── FRONT FACE ── white translucent glass with 3D text */}
              <div
                style={{
                  position: 'absolute',
                  width: `${prismW}px`,
                  height: `${prismH}px`,
                  background: 'linear-gradient(180deg, rgba(200,200,220,0.35) 0%, rgba(170,170,195,0.3) 40%, rgba(150,150,175,0.28) 100%)',
                  borderRadius: '6px',
                  transform: `translateZ(${depth / 2}px)`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px solid rgba(255,255,255,0.25)',
                  backdropFilter: 'blur(2px)',
                  overflow: 'hidden',
                }}
              >
                {/* Internal glass reflection sweep */}
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: 'linear-gradient(160deg, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0.05) 30%, transparent 55%)',
                    pointerEvents: 'none',
                  }}
                />

                {/* Animated rainbow caustics traveling through the prism */}
                <motion.div
                  initial={{ opacity: 0, x: -prismW }}
                  animate={
                    splitActive
                      ? { opacity: [0, 0.7, 0.7, 0], x: [-prismW, -100, 100, prismW] }
                      : { opacity: 0, x: -prismW }
                  }
                  transition={{ duration: 2, ease: 'easeInOut' }}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: `${prismW}px`,
                    height: `${prismH}px`,
                    background: `linear-gradient(90deg,
                      transparent 0%,
                      #ef444420 10%, #f9731620 20%, #eab30820 30%,
                      #22c55e28 40%, #3b82f628 50%, #8b5cf620 60%,
                      #d946ef20 70%, transparent 85%)`,
                    pointerEvents: 'none',
                    filter: 'blur(8px)',
                  }}
                />

                {/* Persistent subtle internal rainbow after animation */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={splitActive ? { opacity: 1 } : { opacity: 0 }}
                  transition={{ duration: 1.5, delay: 1.5 }}
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: `linear-gradient(125deg,
                      transparent 10%,
                      #ef444408 20%, #f9731608 28%, #eab30808 36%,
                      #22c55e0a 44%, #3b82f60a 52%, #8b5cf608 60%,
                      #d946ef08 68%, transparent 80%)`,
                    pointerEvents: 'none',
                  }}
                />

                {/* ── 3D LETTERS ── */}
                <div style={{ position: 'relative', zIndex: 2, display: 'flex', gap: '2px' }}>
                  {letters.map((letter, i) => (
                    <motion.span
                      key={i}
                      initial={{ opacity: 0, y: 15 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: 0.3 + i * 0.08 }}
                      style={{
                        fontSize: '68px',
                        fontFamily: "'Inter', system-ui, sans-serif",
                        fontWeight: 900,
                        lineHeight: 1,
                        letterSpacing: '-0.02em',
                        color: 'rgba(255, 255, 255, 0.65)',
                        textShadow: [
                          '0 1px 0 rgba(220,220,235,0.5)',
                          '0 2px 0 rgba(200,200,220,0.4)',
                          '0 3px 0 rgba(180,180,205,0.35)',
                          '0 4px 0 rgba(160,160,190,0.3)',
                          '0 5px 0 rgba(140,140,170,0.25)',
                          '0 6px 0 rgba(120,120,155,0.2)',
                          '0 8px 16px rgba(0,0,0,0.3)',
                          '0 0 25px rgba(255,255,255,0.1)',
                        ].join(', '),
                        position: 'relative',
                        display: 'inline-block',
                      }}
                    >
                      {letter}
                      {/* Glassy top-light on each letter */}
                      <span
                        aria-hidden="true"
                        style={{
                          position: 'absolute',
                          inset: 0,
                          backgroundImage: 'linear-gradient(175deg, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.12) 30%, transparent 55%)',
                          WebkitBackgroundClip: 'text',
                          backgroundClip: 'text',
                          color: 'transparent',
                          pointerEvents: 'none',
                        }}
                      >
                        {letter}
                      </span>
                      {/* Rainbow refraction tint per letter */}
                      <motion.span
                        aria-hidden="true"
                        initial={{ opacity: 0 }}
                        animate={splitActive ? { opacity: 1 } : { opacity: 0 }}
                        transition={{ duration: 1.2, delay: 1.2 + i * 0.1 }}
                        style={{
                          position: 'absolute',
                          inset: 0,
                          backgroundImage: `linear-gradient(${100 + i * 25}deg, transparent 10%, ${rainbowColors[i]}35 40%, ${rainbowColors[(i + 2) % 7]}28 65%, transparent 90%)`,
                          WebkitBackgroundClip: 'text',
                          backgroundClip: 'text',
                          color: 'transparent',
                          pointerEvents: 'none',
                          mixBlendMode: 'screen',
                        }}
                      >
                        {letter}
                      </motion.span>
                    </motion.span>
                  ))}
                </div>
              </div>

              {/* ── TOP FACE ── bright white, clearly visible */}
              <div
                style={{
                  position: 'absolute',
                  width: `${prismW}px`,
                  height: `${depth}px`,
                  background: 'linear-gradient(180deg, rgba(220,220,240,0.4) 0%, rgba(190,190,215,0.35) 100%)',
                  borderRadius: '6px 6px 0 0',
                  transformOrigin: 'bottom center',
                  transform: `translateY(-${depth}px) translateZ(0px) rotateX(90deg)`,
                  borderTop: '1px solid rgba(255,255,255,0.35)',
                  borderLeft: '1px solid rgba(255,255,255,0.15)',
                  borderRight: '1px solid rgba(255,255,255,0.15)',
                }}
              >
                {/* Specular highlight */}
                <div
                  style={{
                    position: 'absolute',
                    top: '25%',
                    left: '5%',
                    right: '5%',
                    height: '1px',
                    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
                  }}
                />
                {/* Second specular */}
                <div
                  style={{
                    position: 'absolute',
                    top: '55%',
                    left: '15%',
                    right: '15%',
                    height: '1px',
                    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent)',
                  }}
                />
                {/* Animated light traveling across top face */}
                <motion.div
                  initial={{ opacity: 0, left: '-30%' }}
                  animate={
                    beamActive
                      ? { opacity: [0, 0.6, 0], left: ['-30%', '50%', '130%'] }
                      : {}
                  }
                  transition={{ duration: 1.5, delay: 0.6, ease: 'easeInOut' }}
                  style={{
                    position: 'absolute',
                    top: 0,
                    width: '40%',
                    height: '100%',
                    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)',
                    pointerEvents: 'none',
                  }}
                />
              </div>

              {/* ── BOTTOM FACE ── */}
              <div
                style={{
                  position: 'absolute',
                  width: `${prismW}px`,
                  height: `${depth}px`,
                  background: 'linear-gradient(180deg, rgba(80,80,110,0.25) 0%, rgba(50,50,75,0.3) 100%)',
                  transformOrigin: 'top center',
                  transform: `translateY(${prismH}px) translateZ(0px) rotateX(-90deg)`,
                }}
              />

              {/* ── RIGHT FACE ── clearly visible, white translucent with all-color exit glow */}
              <div
                style={{
                  position: 'absolute',
                  width: `${depth}px`,
                  height: `${prismH}px`,
                  background: 'linear-gradient(180deg, rgba(180,180,210,0.3) 0%, rgba(160,160,190,0.28) 50%, rgba(140,140,170,0.25) 100%)',
                  transformOrigin: 'left center',
                  transform: `translateX(${prismW}px) rotateY(90deg)`,
                  borderRight: '1px solid rgba(255,255,255,0.2)',
                  borderTop: '1px solid rgba(255,255,255,0.12)',
                  borderBottom: '1px solid rgba(255,255,255,0.08)',
                  overflow: 'hidden',
                }}
              >
                {/* Glass sheen on right face */}
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: 'linear-gradient(160deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.05) 40%, transparent 70%)',
                    pointerEvents: 'none',
                  }}
                />
                {/* All-color rainbow light exiting through right face */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={splitActive ? { opacity: 1 } : { opacity: 0 }}
                  transition={{ duration: 1.2, delay: 0.3 }}
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: `linear-gradient(180deg,
                      #ef444430, #f9731630, #eab30830,
                      #22c55e35,
                      #3b82f630, #8b5cf630, #d946ef30)`,
                    pointerEvents: 'none',
                  }}
                />
                {/* Bright rainbow exit edge */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={splitActive ? { opacity: 1 } : { opacity: 0 }}
                  transition={{ duration: 0.8, delay: 0.5 }}
                  style={{
                    position: 'absolute',
                    top: '5%',
                    bottom: '5%',
                    right: 0,
                    width: '4px',
                    background: `linear-gradient(180deg,
                      #ef4444, #f97316, #eab308, #22c55e, #3b82f6, #8b5cf6, #d946ef)`,
                    filter: 'blur(2px)',
                    borderRadius: '2px',
                  }}
                />
              </div>

              {/* ── LEFT FACE ── */}
              <div
                style={{
                  position: 'absolute',
                  width: `${depth}px`,
                  height: `${prismH}px`,
                  background: 'linear-gradient(180deg, rgba(160,160,190,0.25) 0%, rgba(130,130,165,0.22) 100%)',
                  transformOrigin: 'right center',
                  transform: 'translateX(0px) rotateY(-90deg)',
                  borderLeft: '1px solid rgba(255,255,255,0.1)',
                  overflow: 'hidden',
                }}
              >
                {/* Beam entry glow on left face */}
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={beamActive ? { opacity: 1 } : { opacity: 0 }}
                  transition={{ duration: 0.6, delay: 0.5 }}
                  style={{
                    position: 'absolute',
                    top: '25%',
                    bottom: '25%',
                    right: '15%',
                    width: '50%',
                    background: 'radial-gradient(ellipse, rgba(255,255,255,0.35) 0%, transparent 70%)',
                    filter: 'blur(4px)',
                  }}
                />
              </div>

              {/* ── BACK FACE ── */}
              <div
                style={{
                  position: 'absolute',
                  width: `${prismW}px`,
                  height: `${prismH}px`,
                  background: 'rgba(100,100,130,0.15)',
                  transform: `translateZ(-${depth / 2}px)`,
                  borderRadius: '6px',
                }}
              />

              {/* ── EDGE HIGHLIGHTS ── */}
              {/* Top-front edge */}
              <div
                style={{
                  position: 'absolute',
                  width: `${prismW}px`,
                  height: '1px',
                  background: 'linear-gradient(90deg, rgba(255,255,255,0.1), rgba(255,255,255,0.35), rgba(255,255,255,0.1))',
                  transform: `translateZ(${depth / 2}px)`,
                  top: 0,
                  left: 0,
                }}
              />
              {/* Bottom-front edge */}
              <div
                style={{
                  position: 'absolute',
                  width: `${prismW}px`,
                  height: '1px',
                  background: 'linear-gradient(90deg, rgba(255,255,255,0.04), rgba(255,255,255,0.12), rgba(255,255,255,0.04))',
                  transform: `translateZ(${depth / 2}px)`,
                  bottom: 0,
                  left: 0,
                }}
              />
              {/* Right-front vertical edge */}
              <div
                style={{
                  position: 'absolute',
                  width: '1px',
                  height: `${prismH}px`,
                  background: 'linear-gradient(180deg, rgba(255,255,255,0.3), rgba(255,255,255,0.1), rgba(255,255,255,0.05))',
                  transform: `translateZ(${depth / 2}px)`,
                  top: 0,
                  right: 0,
                }}
              />
              {/* Top-right edge */}
              <div
                style={{
                  position: 'absolute',
                  width: '1px',
                  height: `${depth}px`,
                  background: 'linear-gradient(180deg, rgba(255,255,255,0.25), rgba(255,255,255,0.1))',
                  transformOrigin: 'top left',
                  transform: `translate(${prismW}px, 0px) translateZ(${depth / 2}px) rotateX(90deg)`,
                }}
              />
              {/* Top-left edge (back) */}
              <div
                style={{
                  position: 'absolute',
                  width: '1px',
                  height: `${depth}px`,
                  background: 'linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.04))',
                  transformOrigin: 'top left',
                  transform: `translate(0px, 0px) translateZ(${depth / 2}px) rotateX(90deg)`,
                }}
              />
            </div>
          </motion.div>

          {/* ── RAINBOW SPLIT BEAMS — all 7 colors ── */}
          <div className="absolute left-1/2 ml-[200px] top-1/2 -translate-y-1/2">
            {rainbowColors.map((color, i) => {
              const totalBeams = rainbowColors.length;
              const spreadAngle = 36;
              const angle =
                -spreadAngle / 2 + (spreadAngle / (totalBeams - 1)) * i;

              return (
                <motion.div
                  key={color}
                  initial={{ scaleX: 0, opacity: 0 }}
                  animate={
                    splitActive
                      ? { scaleX: 1, opacity: 1 }
                      : { scaleX: 0, opacity: 0 }
                  }
                  transition={{
                    duration: 0.9,
                    delay: 0.08 * i,
                    ease: 'easeOut',
                  }}
                  className="absolute origin-left"
                  style={{
                    width: '280px',
                    height: '2.5px',
                    top: '0px',
                    left: '0px',
                    transform: `rotate(${angle}deg)`,
                  }}
                >
                  <div
                    className="h-full rounded-full"
                    style={{
                      background: `linear-gradient(90deg, ${color} 0%, ${color}88 50%, transparent 100%)`,
                      boxShadow: `0 0 10px ${color}44, 0 0 30px ${color}18`,
                    }}
                  />
                  <div
                    className="absolute inset-0 rounded-full"
                    style={{
                      background: `linear-gradient(90deg, ${color}30 0%, transparent 55%)`,
                      filter: 'blur(5px)',
                      transform: 'scaleY(4)',
                    }}
                  />
                </motion.div>
              );
            })}
          </div>

          {/* Sparkle particles at exit point */}
          {splitActive &&
            Array.from({ length: 12 }).map((_, i) => (
              <motion.div
                key={`p-${i}`}
                className="absolute z-20"
                style={{ left: 'calc(50% + 190px)', top: '50%' }}
                initial={{ scale: 0, opacity: 1, x: 0, y: 0 }}
                animate={{
                  scale: [0, 1.3, 0],
                  opacity: [0, 1, 0],
                  x: [0, 25 + Math.random() * 55],
                  y: [0, (Math.random() - 0.5) * 90],
                }}
                transition={{
                  duration: 1.3 + Math.random() * 0.8,
                  delay: i * 0.07,
                  ease: 'easeOut',
                }}
              >
                <div
                  className="w-1.5 h-1.5 rounded-full"
                  style={{
                    backgroundColor: rainbowColors[i % rainbowColors.length],
                    boxShadow: `0 0 8px ${rainbowColors[i % rainbowColors.length]}, 0 0 18px ${rainbowColors[i % rainbowColors.length]}44`,
                  }}
                />
              </motion.div>
            ))}
        </div>

        {/* Subtitle */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.5, duration: 0.8 }}
          className="text-center mt-2"
        >
          <p
            className="text-sm font-medium tracking-[0.3em] uppercase"
            style={{
              background:
                'linear-gradient(90deg, #ef4444, #f97316, #eab308, #22c55e, #3b82f6, #8b5cf6, #d946ef)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}
          >
            Revealing What Clean Data Hides
          </p>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 2, duration: 0.8 }}
            className="text-xs text-text-muted mt-2 tracking-wide"
          >
            The first AI platform that catches bias your cleaning introduces —
            in real time.
          </motion.p>
        </motion.div>
      </div>
    </div>
  );
}
