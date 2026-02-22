import Sidebar from './components/layout/Sidebar';
import TopBar from './components/layout/TopBar';
import PrismHero from './components/shared/PrismHero';
import Stage1Upload from './components/stages/Stage1Upload';
import Stage2Clean from './components/stages/Stage2Clean';
import Stage3Gate from './components/stages/Stage3Gate';
import Stage4Train from './components/stages/Stage4Train';
import Stage5Explain from './components/stages/Stage5Explain';
import { usePipelineStore } from './store/pipeline';

const stageComponents = {
  1: Stage1Upload,
  2: Stage2Clean,
  3: Stage3Gate,
  4: Stage4Train,
  5: Stage5Explain,
};

export default function App() {
  const currentStage = usePipelineStore((s) => s.currentStage);
  const StageComponent = stageComponents[currentStage as keyof typeof stageComponents];

  return (
    <div className="flex min-h-screen bg-bg-primary">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <TopBar />
        <main className="flex-1 p-8 overflow-y-auto">
          <PrismHero />
          <StageComponent />
        </main>
      </div>
    </div>
  );
}
