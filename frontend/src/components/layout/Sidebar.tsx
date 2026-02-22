import { usePipelineStore } from '../../store/pipeline';
import type { Stage } from '../../lib/types';
import {
  Upload,
  Sparkles,
  ShieldCheck,
  Brain,
  FileText,
  Lock,
  Check,
} from 'lucide-react';

const stages: { num: Stage; label: string; icon: typeof Upload }[] = [
  { num: 1, label: 'Understand', icon: Upload },
  { num: 2, label: 'Clean', icon: Sparkles },
  { num: 3, label: 'Bias Gate', icon: ShieldCheck },
  { num: 4, label: 'Train', icon: Brain },
  { num: 5, label: 'Explain', icon: FileText },
];

export default function Sidebar() {
  const { currentStage, maxUnlockedStage, setStage } = usePipelineStore();

  return (
    <aside className="w-64 min-h-screen bg-bg-secondary border-r border-border flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-accent flex items-center justify-center">
            <span className="text-white font-bold text-sm">P</span>
          </div>
          <div>
            <h1 className="text-lg font-bold text-text-primary tracking-tight">
              PRISM
            </h1>
            <p className="text-[11px] text-text-muted leading-none">
              Bias-Aware AutoML
            </p>
          </div>
        </div>
      </div>

      {/* Stage Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {stages.map((stage, idx) => {
          const isActive = currentStage === stage.num;
          const isUnlocked = stage.num <= maxUnlockedStage;
          const isCompleted = stage.num < maxUnlockedStage;
          const Icon = stage.icon;

          return (
            <button
              key={stage.num}
              onClick={() => isUnlocked && setStage(stage.num)}
              disabled={!isUnlocked}
              className={`
                w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium
                transition-all duration-200 relative group
                ${isActive
                  ? 'bg-accent/15 text-accent-light border border-accent/30'
                  : isUnlocked
                    ? 'text-text-secondary hover:bg-bg-card hover:text-text-primary border border-transparent'
                    : 'text-text-muted cursor-not-allowed border border-transparent opacity-50'
                }
              `}
            >
              {/* Connector line */}
              {idx < stages.length - 1 && (
                <div
                  className={`absolute left-[25px] top-[38px] w-[2px] h-[12px] ${
                    isCompleted ? 'bg-positive' : 'bg-border'
                  }`}
                />
              )}

              {/* Icon */}
              <div
                className={`w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0 ${
                  isCompleted
                    ? 'bg-positive/20 text-positive'
                    : isActive
                      ? 'bg-accent/20 text-accent-light'
                      : 'bg-bg-card text-text-muted'
                }`}
              >
                {isCompleted ? (
                  <Check size={14} />
                ) : !isUnlocked ? (
                  <Lock size={14} />
                ) : (
                  <Icon size={14} />
                )}
              </div>

              {/* Label */}
              <div className="text-left">
                <div className="text-[11px] text-text-muted font-normal">
                  Stage {stage.num}
                </div>
                <div className="-mt-0.5">{stage.label}</div>
              </div>
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-border">
        <div className="text-[11px] text-text-muted text-center">
          Glass box with a paper trail
        </div>
      </div>
    </aside>
  );
}
