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

    let currentJobId = null;
    let pollInterval = null;
    let startTime = null;

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
        reportView.classList.add("hidden");
        uploadView.classList.remove("hidden");
        fileInput.value = "";
        currentJobId = null;
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

                // Update progress metrics
                const completed = job.completed_chunks;
                const total = job.total_chunks;
                completedChunksCount.textContent = completed;
                
                const pct = total > 0 ? (completed / total) * 100 : 0;
                progressBar.style.width = `${pct}%`;

                // Elapsed time
                const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
                document.getElementById("elapsed-time").textContent = `Elapsed: ${elapsedSec}s`;

                // Render chunk grid
                chunkGrid.innerHTML = "";
                job.chunks.forEach((chunk, index) => {
                    const box = document.createElement("div");
                    box.className = `chunk-box ${chunk.status}`;
                    box.textContent = index + 1;
                    chunkGrid.appendChild(box);
                });

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
            tdLines.textContent = f.evidence_lines.join(", ");
            tr.appendChild(tdLines);

            const tdReason = document.createElement("td");
            tdReason.textContent = f.reasoning;
            tr.appendChild(tdReason);

            tbody.appendChild(tr);
        });
    }
});