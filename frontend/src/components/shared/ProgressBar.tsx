import { motion } from 'framer-motion';

interface Props {
  value: number;
  max?: number;
  color?: 'accent' | 'positive' | 'negative' | 'warning';
  height?: 'sm' | 'md';
  showLabel?: boolean;
}

const colorClasses = {
  accent: 'bg-accent',
  positive: 'bg-positive',
  negative: 'bg-negative',
  warning: 'bg-warning',
};

export default function ProgressBar({
  value,
  max = 100,
  color = 'accent',
  height = 'sm',
  showLabel = false,
}: Props) {
  const percent = Math.min((value / max) * 100, 100);

  return (
    <div className="flex items-center gap-2">
      <div
        className={`flex-1 rounded-full bg-bg-card overflow-hidden ${
          height === 'sm' ? 'h-1.5' : 'h-2.5'
        }`}
      >
        <motion.div
          className={`h-full rounded-full ${colorClasses[color]}`}
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
      </div>
      {showLabel && (
        <span className="text-xs font-mono text-text-muted w-10 text-right">
          {Math.round(percent)}%
        </span>
      )}
    </div>
  );
}
