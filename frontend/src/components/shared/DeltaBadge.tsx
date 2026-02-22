import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface Props {
  delta: number;
  size?: 'sm' | 'md';
  showLabel?: boolean;
}

export default function DeltaBadge({ delta, size = 'md', showLabel = true }: Props) {
  const isPositive = delta > 0;
  const isNegative = delta < 0;

  const color = isPositive
    ? 'text-positive bg-positive-bg border-positive/30'
    : isNegative
      ? 'text-negative bg-negative-bg border-negative/30'
      : 'text-text-muted bg-bg-card border-border';

  const Icon = isPositive ? TrendingUp : isNegative ? TrendingDown : Minus;
  const iconSize = size === 'sm' ? 12 : 14;

  return (
    <div
      className={`inline-flex items-center gap-1.5 rounded-md border font-mono font-semibold
        ${color}
        ${size === 'sm' ? 'text-xs px-2 py-0.5' : 'text-sm px-2.5 py-1'}
      `}
    >
      <Icon size={iconSize} />
      <span>{isPositive ? '+' : ''}{delta}</span>
      {showLabel && (
        <span className="text-[10px] font-normal opacity-70 ml-0.5">
          {isPositive ? 'less bias' : isNegative ? 'more bias' : 'neutral'}
        </span>
      )}
    </div>
  );
}
