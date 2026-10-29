/** Shell: top bar with hash-routed steps, the print-job ticket (rail) and the active view. */

import { useEffect, useState } from "react";
import { AppProvider } from "./state";
import { Rail } from "./ui/Rail";
import { Bench } from "./views/Bench";
import { Campaign } from "./views/Campaign";
import { Experiments } from "./views/Experiments";
import { ModelView } from "./views/ModelView";
import { Optimise } from "./views/Optimise";
import { Study } from "./views/Study";

const REPO = "https://github.com/BAFOOD123/RobustPrint";
const VIEWS = [
  { id: "bench", label: "Bench", View: Bench },
  { id: "campaign", label: "Campaign", View: Campaign },
  { id: "model", label: "Model", View: ModelView },
  { id: "optimise", label: "Optimise", View: Optimise },
  { id: "experiments", label: "Next experiments", View: Experiments },
  { id: "study", label: "Study results", View: Study },
] as const;
type ViewId = (typeof VIEWS)[number]["id"];

function currentView(): ViewId {
  const h = window.location.hash.replace(/^#/, "");
  return (VIEWS.find((v) => v.id === h)?.id ?? "bench") as ViewId;
}

export function App() {
  const [view, setView] = useState<ViewId>(currentView);
  useEffect(() => {
    const onHash = () => {
      setView(currentView());
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const { View } = VIEWS.find((v) => v.id === view)!;
  return (
    <AppProvider>
      <div className="app">
        <header className="topbar">
          <div className="brand">
            <h1>RobustPrint Virtual Lab</h1>
            <span className="tag">recycled feedstock · FDM</span>
            <span className="synthetic-badge" title="All printing here happens in a simulator with assumed physics">synthetic simulator</span>
          </div>
          <nav className="tabs" aria-label="Steps">
            {VIEWS.map((v, i) => (
              <a key={v.id} href={`#${v.id}`} aria-current={v.id === view ? "page" : undefined}>
                <span className="step">{i + 1}</span>
                {v.label}
              </a>
            ))}
          </nav>
        </header>
        <div className="layout">
          <Rail />
          <main className="workspace">
            <View />
          </main>
        </div>
        <footer className="footer">
          <span>
            Simulated physics with assumed coefficients; not validated against real recycled material, and not for flight or
            safety-critical use.
          </span>
          <span>
            <a href={REPO} target="_blank" rel="noreferrer">Source and study</a> · MIT licence
          </span>
        </footer>
      </div>
    </AppProvider>
  );
}
