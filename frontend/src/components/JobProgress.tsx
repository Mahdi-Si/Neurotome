// A job's progress: 4px track with an accent fill, a mono status line with Cancel; nothing once done.
import { isTerminal, progressText } from "../lib/jobs";
import type { Job } from "../lib/types";

export function JobProgress({ job, onCancel }: { job: Job<unknown> | null; onCancel: () => void }) {
  if (!job || job.status === "done") return null;
  if (job.status === "error") return <div className="err mono">{job.error || progressText(job)}</div>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
      <div className="job-track" role="progressbar" aria-label={`${job.kind} job`} aria-valuetext={progressText(job)} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(job.progress * 100)}>
        <div style={{ width: `${Math.min(1, Math.max(0, job.progress)) * 100}%` }} />
      </div>
      <div className="lbl dim" style={{ display: "flex", gap: 8 }}>
        <span style={{ flex: 1, minWidth: 0, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{progressText(job)}</span>
        {!isTerminal(job.status) && <button type="button" className="chart-export lbl" onClick={onCancel}>Cancel</button>}
      </div>
    </div>
  );
}
