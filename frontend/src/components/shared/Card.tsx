import type { ReactNode } from 'react';

interface Props {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  glow?: boolean;
  onClick?: () => void;
}

export default function Card({ children, className = '', hover = false, glow = false, onClick }: Props) {
  return (
    <div
      onClick={onClick}
      className={`
        bg-bg-card border border-border rounded-xl p-5
        ${hover ? 'hover:bg-bg-card-hover hover:border-border-light transition-all duration-200' : ''}
        ${glow ? 'shadow-[0_0_20px_rgba(99,102,241,0.08)]' : ''}
        ${className}
      `}
    >
      {children}
    </div>
  );
}
