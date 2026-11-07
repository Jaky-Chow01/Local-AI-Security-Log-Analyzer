document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadError = document.getElementById("upload-error");
    
    const uploadView = document.getElementById("upload-view");
    const progressView = document.getElementById("progress-view");
    const reportView = document.getElementById("report-view");
    
    const jobFilename = document.getElementById("job-filename");
    const jobStatusText = document.getElementById("job-status-text");
    const progressBar = document.getElementById("progress-bar");
    const completedChunksCount = document.getElementById("completed-chunks-count");
    const totalChunksCount = document.getElementById("total-chunks-count");
    const chunkGrid = document.getElementById("chunk-grid");
    const cancelBtn = document.getElementById("cancel-btn");
    const newAnalysisBtn = document.getElementById("new-analysis-btn");
    const downloadReportBtn = document.getElementById("download-report-btn");
    const rawLineViewer = document.getElementById("raw-line-viewer");
    const modelName = document.getElementById("model-name");
    const modelRuntime = document.getElementById("model-runtime");
    const modelTelemetry = document.getElementById("model-telemetry");
    const chunkDetail = document.getElementById("chunk-detail");
    const liveCoverage = document.getElementById("coverage-live");
    const queuePosition = document.getElementById("queue-position");

    let currentJobId = null;
    let pollInterval = null;
    let startTime = null;
    let latestReport = null;
    const inspectedLines = new Set();

    loadModelStatus();
    setInterval(loadModelStatus, 3000);

    async function loadModelStatus() {
        try {
            const response = await fetch("/api/model-status");
            if (!response.ok) throw new Error("Could not load model configuration.");
            const status = await response.json();
            modelName.textContent = `Model: ${status.model_filename}`;
            modelRuntime.textContent = status.mode;
            modelRuntime.title = status.runtime;
            const memory = status.memory_mb === null
                ? "RAM: unavailable"
                : `RAM: ${status.memory_mb.toFixed(0)} / ${status.memory_budget_mb} MB target${status.memory_kind === "peak-rss-estimate" ? " peak" : ""}`;
            const inference = status.last_inference_seconds === null
                ? "Last inference: —"
                : `Last inference: ${status.last_inference_seconds.toFixed(2)} s`;
            modelTelemetry.textContent = `${memory} · ${inference}`;
            modelTelemetry.classList.toggle(
                "over-budget",
                status.memory_mb !== null && status.memory_mb >= status.memory_budget_mb
            );
            modelRuntime.textContent = `${status.mode} · ${status.model_loaded ? "loaded" : "configured"}`;
        } catch (err) {
            modelName.textContent = "Model: configuration unavailable";
            console.error("Model status error", err);
        }
    }

    // Drag and Drop Handlers
    dropZone.addEventListener("click", () => fileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--primary)";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "var(--border-color)";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--border-color)";
        if (e.dataTransfer.files.length > 0) {
            handleUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleUpload(e.target.files[0]);
        }
    });

    cancelBtn.addEventListener("click", async () => {
        if (!currentJobId) return;
        try {
            await fetch(`/api/jobs/${currentJobId}/cancel`, { method: "POST" });
        } catch (err) {
            console.error("Cancel failed", err);
        }
    });

    newAnalysisBtn.addEventListener("click", () => {
        if (pollInterval) clearInterval(pollInterval);
        reportView.classList.add("hidden");
        progressView.classList.add("hidden");
        uploadView.classList.remove("hidden");
        fileInput.value = "";
        currentJobId = null;
        latestReport = null;
        inspectedLines.clear();
        rawLineViewer.replaceChildren();
    });

    downloadReportBtn.addEventListener("click", async () => {
        if (!currentJobId || !latestReport) return;
        try {
            const response = await fetch(`/api/jobs/${currentJobId}/report`);
            if (!response.ok) throw new Error("Failed to download the report.");
            const blob = new Blob([JSON.stringify(latestReport, null, 2)], {
                type: "application/json"
            });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `security-report-${currentJobId}.json`;
            link.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            jobStatusText.textContent = `Error: ${err.message}`;
        }
    });

    async function handleUpload(file) {
        uploadError.classList.add("hidden");
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail?.message || "Upload validation failed.");
            }

            currentJobId = data.job_id;
            jobFilename.textContent = data.filename;
            totalChunksCount.textContent = data.total_chunks;
            
            uploadView.classList.add("hidden");
            progressView.classList.remove("hidden");
            
            startTime = Date.now();
            startPolling(currentJobId);

        } catch (err) {
            uploadError.textContent = err.message;
            uploadError.classList.remove("hidden");
        }
    }

    function startPolling(jobId) {
        if (pollInterval) clearInterval(pollInterval);
        
        pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/jobs/${jobId}`);
                if (!res.ok) return;
                const job = await res.json();
                if (job.model_status === "unavailable") {
                    jobStatusText.textContent = "Local model unavailable; detector-only analysis.";
                } else if (job.model_status === "degraded") {
                    jobStatusText.textContent = "Some model chunks failed; report will show coverage gaps.";
                } else if (job.model_status === "filtered") {
                    jobStatusText.textContent = "Deterministic prefiltering limited model calls; review filtered ranges.";
                } else {
                    jobStatusText.textContent = `Processing (${job.model_status} model status)...`;
                }

                // Update progress metrics
                const completed = job.completed_chunks;
                const total = job.total_chunks;
                completedChunksCount.textContent = completed;
                totalChunksCount.textContent = total;
                queuePosition.textContent = job.queue_position === 0
                    ? "Queue: processing"
                    : job.queue_position
                        ? `Queue position: ${job.queue_position}`
                        : "Queue: not waiting";
                
                const pct = total > 0 ? (completed / total) * 100 : 0;
                progressBar.style.width = `${pct}%`;

                // Elapsed time
                const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
                document.getElementById("elapsed-time").textContent = `Elapsed: ${elapsedSec}s`;

                // Render chunk grid
                chunkGrid.innerHTML = "";
                const failedChunks = [];
                const filteredChunks = [];
                let activeChunk = null;
                job.chunks.forEach((chunk, index) => {
                    const box = document.createElement("div");
                    box.className = `chunk-box ${chunk.status}`;
                    box.textContent = index + 1;
                    const overlap = chunk.overlap_line_numbers || [];
                    const duration = Number.isFinite(chunk.inference_seconds)
                        ? ` · ${chunk.inference_seconds.toFixed(2)} s`
                        : "";
                    box.title = `Chunk ${index + 1}: lines ${chunk.start_line}-${chunk.end_line}, ${chunk.token_count} tokens, overlap ${overlap.length ? overlap.join(", ") : "none"}${duration}`;
                    chunkGrid.appendChild(box);
                    if (["running", "retrying"].includes(chunk.status)) activeChunk = chunk;
                    if (chunk.status === "failed" || chunk.status === "pending") {
                        failedChunks.push(`${chunk.start_line}-${chunk.end_line} (${chunk.status})`);
                    }
                    if (chunk.status === "skipped") {
                        filteredChunks.push(`${chunk.start_line}-${chunk.end_line}`);
                    }
                });
                chunkDetail.textContent = activeChunk
                    ? `Current chunk: lines ${activeChunk.start_line}-${activeChunk.end_line} · ${activeChunk.token_count} tokens · overlap: ${(activeChunk.overlap_line_numbers || []).join(", ") || "none"}`
                    : `${total} chunks · hover a chunk for token budget, line range, overlap, and inference time.`;
                liveCoverage.textContent = [
                    failedChunks.length ? `Failed/unprocessed: ${failedChunks.join(", ")}` : "No failed or unprocessed ranges.",
                    filteredChunks.length ? `Prefiltered from model: ${filteredChunks.join(", ")}` : ""
                ].filter(Boolean).join(" ");

                if (job.status === "complete" || job.status === "partial" || job.status === "failed") {
                    clearInterval(pollInterval);
                    fetchReport(jobId);
                }
            } catch (err) {
                console.error("Polling error", err);
            }
        }, 2000);
    }

    async function fetchReport(jobId) {
        try {
            const res = await fetch(`/api/jobs/${jobId}/report`);
            if (!res.ok) throw new Error("Failed to fetch report.");
            const report = await res.json();
            renderReport(report);
        } catch (err) {
            jobStatusText.textContent = `Error: ${err.message}`;
        }
    }

    function renderReport(report) {
        latestReport = report;
        progressView.classList.add("hidden");
        reportView.classList.remove("hidden");

        // Summary metrics
        const riskEl = document.getElementById("metric-overall-risk");
        riskEl.textContent = report.summary.overall_risk;
        riskEl.className = `metric-value badge-${report.summary.overall_risk}`;

        document.getElementById("count-critical").textContent = report.summary.counts_by_severity.critical || 0;
        document.getElementById("count-high").textContent = report.summary.counts_by_severity.high || 0;
        document.getElementById("count-medium").textContent = report.summary.counts_by_severity.medium || 0;
        document.getElementById("count-low").textContent = report.summary.counts_by_severity.low || 0;

        document.getElementById("exec-summary-text").textContent = report.summary.executive_summary;

        const coverage = report.coverage;
        const modelText = report.source.model_status === "unavailable"
            ? "Local model unavailable: only deterministic detectors ran."
            : report.source.model_status === "degraded"
                ? `Local model ${report.source.model || ""} had failed chunks.`
                : `Local model: ${report.source.model || "not reported"} (${report.source.model_status}).`;
        document.getElementById("model-availability").textContent = modelText;
        document.getElementById("coverage-failed-ranges").textContent =
            `Failed or unfinished ranges: ${coverage.failed_ranges.length
                ? coverage.failed_ranges.map(range =>
                    `${range.start_line}-${range.end_line} (${range.status})`).join(", ")
                : "none"}`;
        document.getElementById("coverage-filtered-ranges").textContent =
            `Ranges skipped by deterministic prefilter: ${coverage.filtered_ranges?.length
                ? coverage.filtered_ranges.map(range =>
                    `${range.start_line}-${range.end_line}`).join(", ")
                : "none"}`;
        document.getElementById("coverage-truncated-lines").textContent =
            `Truncated source lines: ${coverage.truncated_lines.length
                ? coverage.truncated_lines.join(", ")
                : "none"}`;
        document.getElementById("coverage-rejected-findings").textContent =
            `Rejected model findings: ${coverage.rejected_model_findings}`;

        renderIncidents(report.incidents || []);

        // Findings table (Using textContent strictly for XSS protection)
        const tbody = document.getElementById("findings-tbody");
        tbody.innerHTML = "";

        report.findings.forEach(f => {
            const tr = document.createElement("tr");
            
            const tdSev = document.createElement("td");
            const badge = document.createElement("span");
            badge.className = `badge badge-${f.severity}`;
            badge.textContent = f.severity;
            tdSev.appendChild(badge);
            tr.appendChild(tdSev);

            const tdType = document.createElement("td");
            tdType.textContent = f.attack_type;
            tr.appendChild(tdType);

            const tdId = document.createElement("td");
            tdId.textContent = f.identifier;
            tr.appendChild(tdId);

            const tdLines = document.createElement("td");
            (f.evidence_lines || []).forEach((lineNo, index) => {
                if (index) tdLines.appendChild(document.createTextNode(", "));
                tdLines.appendChild(createEvidenceButton(lineNo));
            });
            tr.appendChild(tdLines);

            const tdReason = document.createElement("td");
            tdReason.textContent = f.reasoning;
            tr.appendChild(tdReason);

            tbody.appendChild(tr);
        });
    }

    function renderIncidents(incidents) {
        const container = document.getElementById("incidents-container");
        const list = document.getElementById("incidents-list");
        list.replaceChildren();
        container.classList.toggle("hidden", incidents.length === 0);

        incidents.forEach(incident => {
            const card = document.createElement("article");
            card.className = "incident-card";
            const title = document.createElement("h4");
            title.textContent = incident.title;
            card.appendChild(title);

            const summary = document.createElement("p");
            summary.textContent = `${incident.severity} · ${incident.identifier}`;
            card.appendChild(summary);

            const timeline = document.createElement("ul");
            timeline.className = "incident-timeline";
            (incident.timeline || []).forEach(item => {
                const row = document.createElement("li");
                row.appendChild(createEvidenceButton(item.line));
                row.appendChild(document.createTextNode(` — ${item.event}`));
                timeline.appendChild(row);
            });
            card.appendChild(timeline);
            list.appendChild(card);
        });
    }

    function createEvidenceButton(lineNo) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "evidence-line-btn";
        button.textContent = `L${lineNo}`;
        button.addEventListener("click", () => inspectRawLine(lineNo));
        return button;
    }

    async function inspectRawLine(lineNo) {
        if (inspectedLines.has(lineNo)) return;
        try {
            const response = await fetch(`/api/jobs/${currentJobId}/lines/${lineNo}`);
            if (!response.ok) throw new Error(`Could not load line ${lineNo}.`);
            const line = await response.json();
            const item = document.createElement("div");
            item.className = "raw-line-item";
            const content = document.createElement("pre");
            content.textContent = `L${line.line_no}: ${line.raw}`;
            item.appendChild(content);
            rawLineViewer.appendChild(item);
            inspectedLines.add(lineNo);
        } catch (err) {
            const message = document.createElement("p");
            message.textContent = err.message;
            rawLineViewer.appendChild(message);
        }
    }
});