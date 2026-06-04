const elements = {
  healthDot: document.querySelector("#healthDot"),
  healthLabel: document.querySelector("#healthLabel"),
  llmProvider: document.querySelector("#llmProvider"),
  llmModel: document.querySelector("#llmModel"),
  embeddingModel: document.querySelector("#embeddingModel"),
  collectionName: document.querySelector("#collectionName"),
  runtimeNote: document.querySelector("#runtimeNote"),
  ingestForm: document.querySelector("#ingestForm"),
  ingestButton: document.querySelector("#ingestButton"),
  ingestResult: document.querySelector("#ingestResult"),
  queryForm: document.querySelector("#queryForm"),
  queryButton: document.querySelector("#queryButton"),
  streamButton: document.querySelector("#streamButton"),
  questionInput: document.querySelector("#questionInput"),
  timeline: document.querySelector("#timeline"),
  answerBody: document.querySelector("#answerBody"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setHealth(state, label) {
  elements.healthDot.classList.remove("online", "offline");
  if (state) {
    elements.healthDot.classList.add(state);
  }
  elements.healthLabel.textContent = label;
}

function setBusy(isBusy) {
  elements.ingestButton.disabled = isBusy;
  elements.queryButton.disabled = isBusy;
  elements.streamButton.disabled = isBusy;
}

function providerLabel(provider) {
  const labels = {
    openai: "OpenAI",
    huggingface: "Hugging Face",
    ollama: "Ollama",
  };
  return labels[provider] || provider || "-";
}

function formatMs(value) {
  const numeric = Number(value || 0);
  if (numeric >= 1000) {
    return `${(numeric / 1000).toFixed(1)}s`;
  }
  return `${Math.round(numeric)}ms`;
}

async function readError(response) {
  try {
    const payload = await response.json();
    return payload.detail || JSON.stringify(payload);
  } catch {
    return await response.text();
  }
}

async function loadRuntime() {
  try {
    const [healthResponse, configResponse] = await Promise.all([
      fetch("/health"),
      fetch("/api/config"),
    ]);

    if (!healthResponse.ok) {
      throw new Error(await readError(healthResponse));
    }
    if (!configResponse.ok) {
      throw new Error(await readError(configResponse));
    }

    const config = await configResponse.json();
    setHealth("online", "Service online");
    elements.llmProvider.textContent = providerLabel(config.llm_provider);
    elements.llmModel.textContent = config.llm_model;
    elements.embeddingModel.textContent = `${providerLabel(config.embedding_provider)}: ${config.embedding_model}`;
    elements.collectionName.textContent = config.qdrant_collection;
    elements.runtimeNote.textContent = config.hf_endpoint_mode
      ? `Hugging Face endpoint mode: ${config.hf_endpoint_mode}. Max hops: ${config.max_hops}.`
      : `Max hops: ${config.max_hops}. Provider selection is controlled by the server environment.`;
  } catch (error) {
    setHealth("offline", "Service needs attention");
    elements.runtimeNote.textContent = error.message || "Could not load runtime configuration.";
  }
}

function selectedSourceType() {
  return document.querySelector('input[name="sourceType"]:checked')?.value || "arxiv";
}

function syncSourcePanels() {
  const active = selectedSourceType();
  document.querySelectorAll("[data-source-panel]").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.sourcePanel !== active);
  });
}

function buildIngestPayload() {
  const sourceType = selectedSourceType();
  const payload = new FormData();

  if (sourceType === "arxiv") {
    const value = document.querySelector("#arxivInput").value.trim();
    if (!value) {
      throw new Error("Enter an arXiv ID.");
    }
    payload.append("arxiv_id", value);
  }

  if (sourceType === "url") {
    const value = document.querySelector("#urlInput").value.trim();
    if (!value) {
      throw new Error("Enter a URL.");
    }
    payload.append("url", value);
  }

  if (sourceType === "file") {
    const file = document.querySelector("#fileInput").files[0];
    if (!file) {
      throw new Error("Choose a PDF or DOCX file.");
    }
    payload.append("file", file);
  }

  return payload;
}

async function ingestSource(event) {
  event.preventDefault();
  elements.ingestResult.className = "result-box";
  elements.ingestResult.textContent = "Ingesting source and updating indexes...";
  setBusy(true);

  try {
    const response = await fetch("/ingest", {
      method: "POST",
      body: buildIngestPayload(),
    });

    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const payload = await response.json();
    elements.ingestResult.className = "result-box success";
    elements.ingestResult.innerHTML = `
      <strong>Indexed ${escapeHtml(payload.source_doc)}</strong><br />
      Created ${escapeHtml(payload.chunks_created)} chunks for retrieval.
    `;
  } catch (error) {
    elements.ingestResult.className = "result-box error";
    elements.ingestResult.textContent = error.message || "Ingestion failed.";
  } finally {
    setBusy(false);
  }
}

function resetResults(message = "Streaming reasoning will appear here.") {
  elements.timeline.innerHTML = `<li class="empty-state">${escapeHtml(message)}</li>`;
  elements.answerBody.className = "answer-body";
  elements.answerBody.textContent = "Waiting for the final answer...";
}

function appendTimelineStep(step) {
  if (elements.timeline.querySelector(".empty-state")) {
    elements.timeline.innerHTML = "";
  }

  const item = document.createElement("li");
  item.innerHTML = `
    <h3>${escapeHtml(stepTitle(step))}</h3>
    <p>${stepSummary(step)}</p>
  `;
  elements.timeline.appendChild(item);
}

function stepTitle(step) {
  const labels = {
    start: "Question received",
    decompose: "Query decomposition",
    retrieve: "Evidence retrieval",
    reflect: "Coverage reflection",
    synthesize: "Citation synthesis",
  };
  return labels[step.step] || step.event || "Agent event";
}

function stepSummary(step) {
  if (step.step === "start") {
    return `Question: <code>${escapeHtml(step.query)}</code>`;
  }

  if (step.step === "decompose") {
    const queries = Array.isArray(step.sub_queries) ? step.sub_queries : [];
    if (!queries.length) {
      return "No sub-queries were generated, so the original question will be used.";
    }
    return `Sub-queries: ${queries.map((query) => `<code>${escapeHtml(query)}</code>`).join(" ")}`;
  }

  if (step.step === "retrieve") {
    return `Query <code>${escapeHtml(step.query)}</code> returned ${escapeHtml(step.retrieved)} candidates, with ${escapeHtml(step.new_chunks_added)} new chunks added.`;
  }

  if (step.step === "reflect") {
    const followUp = step.follow_up_query ? `<code>${escapeHtml(step.follow_up_query)}</code>` : "none";
    return `Hop ${escapeHtml(step.hop)}. Evidence sufficient: ${escapeHtml(step.sufficient)}. Follow-up: ${followUp}. ${escapeHtml(step.rationale || "")}`;
  }

  if (step.step === "synthesize") {
    const count = Array.isArray(step.cited_chunks) ? step.cited_chunks.length : 0;
    return `Synthesized the answer using ${count} cited chunks.`;
  }

  return escapeHtml(JSON.stringify(step));
}

function renderAnswer(payload) {
  const metrics = payload.metrics || {};
  const citations = Array.isArray(payload.citations) ? payload.citations : [];
  const subQueries = Array.isArray(payload.sub_queries) ? payload.sub_queries : [];

  const subQueryHtml = subQueries.length
    ? `<div class="citation-list">${subQueries.map((query) => `<code>${escapeHtml(query)}</code>`).join(" ")}</div>`
    : "";

  const citationsHtml = citations.length
    ? `<div class="citation-list">
        ${citations
          .map(
            (citation) => `
              <article class="citation-card">
                <h3>${escapeHtml(citation.source_doc)}</h3>
                <code>${escapeHtml(citation.chunk_id)}</code>
                <p>${escapeHtml(citation.excerpt)}</p>
                <p>${citation.page ? `Page ${escapeHtml(citation.page)}. ` : ""}${citation.section ? `Section: ${escapeHtml(citation.section)}.` : ""}</p>
              </article>
            `,
          )
          .join("")}
      </div>`
    : "<p>No citations were returned.</p>";

  elements.answerBody.className = "answer-body";
  elements.answerBody.innerHTML = `
    <div class="answer-text">${escapeHtml(payload.answer || "No answer returned.")}</div>
    <div class="metrics-row">
      <span class="metric-pill">Total ${formatMs(metrics.total_latency_ms)}</span>
      <span class="metric-pill">Retrieval ${formatMs(metrics.retrieval_latency_ms)}</span>
      <span class="metric-pill">Rerank ${formatMs(metrics.reranking_latency_ms)}</span>
      <span class="metric-pill">LLM ${formatMs(metrics.llm_latency_ms)}</span>
      <span class="metric-pill">Hops ${escapeHtml(metrics.hops ?? 0)}</span>
      <span class="metric-pill">Chunks ${escapeHtml(metrics.retrieved_chunks ?? 0)}</span>
    </div>
    ${subQueryHtml ? `<h3>Sub-queries</h3>${subQueryHtml}` : ""}
    <h3>Citations</h3>
    ${citationsHtml}
  `;
}

function questionPayload() {
  const question = elements.questionInput.value.trim();
  if (question.length < 5) {
    throw new Error("Enter a question with at least 5 characters.");
  }
  return { question };
}

async function askOnce() {
  resetResults("A non-streaming query is running.");
  setBusy(true);

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(questionPayload()),
    });

    if (!response.ok) {
      throw new Error(await readError(response));
    }

    const payload = await response.json();
    elements.timeline.innerHTML = "";
    (payload.reasoning_trace || []).forEach(appendTimelineStep);
    renderAnswer(payload);
  } catch (error) {
    showAnswerError(error.message || "Query failed.");
  } finally {
    setBusy(false);
  }
}

async function askWithStream(event) {
  event.preventDefault();
  resetResults();
  setBusy(true);

  try {
    const response = await fetch("/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(questionPayload()),
    });

    if (!response.ok || !response.body) {
      throw new Error(await readError(response));
    }

    await readSseStream(response.body);
  } catch (error) {
    showAnswerError(error.message || "Streaming query failed.");
  } finally {
    setBusy(false);
  }
}

async function readSseStream(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    blocks.forEach(handleSseBlock);
  }

  if (buffer.trim()) {
    handleSseBlock(buffer);
  }
}

function handleSseBlock(block) {
  const lines = block.split("\n");
  let eventName = "message";
  let dataText = "";

  lines.forEach((line) => {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    }
    if (line.startsWith("data:")) {
      dataText += line.slice(5).trim();
    }
  });

  if (!dataText) {
    return;
  }

  const data = JSON.parse(dataText);
  if (eventName === "start") {
    elements.timeline.innerHTML = "";
    appendTimelineStep({ event: "Start", step: "start", query: data.question });
    return;
  }
  if (eventName === "reasoning_step") {
    appendTimelineStep(data);
    return;
  }
  if (eventName === "final_answer") {
    renderAnswer(data);
    return;
  }
  if (eventName === "error") {
    showAnswerError(data.detail || "The streaming query failed.");
  }
}

function showAnswerError(message) {
  elements.answerBody.className = "answer-body error";
  elements.answerBody.textContent = message;
}

document.querySelectorAll('input[name="sourceType"]').forEach((input) => {
  input.addEventListener("change", syncSourcePanels);
});

elements.ingestForm.addEventListener("submit", ingestSource);
elements.queryForm.addEventListener("submit", askWithStream);
elements.queryButton.addEventListener("click", askOnce);

syncSourcePanels();
loadRuntime();
