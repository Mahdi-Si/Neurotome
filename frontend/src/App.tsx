import { TopBar } from "./components/TopBar";
import { api } from "./lib/api";
import { useRoute } from "./lib/router";
import { useAsync } from "./lib/useAsync";
import { GraphViewPage } from "./views/GraphView";
import { HomeView } from "./views/HomeView";
import { TensorViewPage } from "./views/TensorView";
import { TraceViewPage } from "./views/TraceView";

export function App() {
  const route = useRoute();
  const run = useAsync(() => (route.runId ? api.run(route.runId) : null), [route.runId]);

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", minWidth: 1100 }}>
      {!route.runId && <HomeView />}
      {route.runId && !run.data && (
        <>
          <TopBar run={null} tab={route.tab} />
          <div className={run.error ? "empty err" : "empty lbl"}>{run.error ?? "Loading run…"}</div>
        </>
      )}
      {route.runId && run.data && run.data.id === route.runId && (
        route.tab === "tensors" ? <TensorViewPage key={run.data.id} run={run.data} route={route} />
          : route.tab === "trace" ? <TraceViewPage key={run.data.id} run={run.data} route={route} />
            : <GraphViewPage key={run.data.id} run={run.data} route={route} />
      )}
    </div>
  );
}
