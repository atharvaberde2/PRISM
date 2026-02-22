import { useEffect, useState } from 'react';

interface Props {
  score: number;
  size?: 'sm' | 'md' | 'lg';
  animate?: boolean;
}

function getScoreColor(score: number): string {
  if (score >= 70) return 'text-positive';
  if (score >= 40) return 'text-warning';
  return 'text-negative';
}

function getScoreBg(score: number): string {
  if (score >= 70) return 'bg-positive-bg border-positive/30';
  if (score >= 40) return 'bg-warning-bg border-warning/30';
  return 'bg-negative-bg border-negative/30';
}

function getLabel(score: number): string {
  if (score >= 80) return 'Excellent';
  if (score >= 70) return 'Good';
  if (score >= 50) return 'Fair';
  if (score >= 40) return 'Caution';
  return 'High Risk';
}

const sizeClasses = {
  sm: 'text-lg px-3 py-1',
  md: 'text-3xl px-5 py-2',
  lg: 'text-5xl px-8 py-4',
};

export default function BiasScoreBadge({ score, size = 'md', animate = true }: Props) {
  const [displayScore, setDisplayScore] = useState(animate ? 0 : score);

  useEffect(() => {
    if (!animate) {
      setDisplayScore(score);
      return;
    }

    const duration = 1200;
    const startTime = Date.now();
    const startVal = displayScore;

    const tick = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = Math.round((startVal + (score - startVal) * eased) * 10) / 10;
      setDisplayScore(current);
      if (progress < 1) requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
  }, [score]);

  return (
    <div
      className={`inline-flex flex-col items-center rounded-lg border ${getScoreBg(displayScore)} ${sizeClasses[size]}`}
    >
      <span className={`font-bold font-mono ${getScoreColor(displayScore)}`}>
        {Number(displayScore).toFixed(1)}
      </span>
      {size !== 'sm' && (
        <span className={`text-[10px] uppercase tracking-wider mt-0.5 ${getScoreColor(displayScore)} opacity-70`}>
          {getLabel(displayScore)}
        </span>
      )}
    </div>
  );
}
