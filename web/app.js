/**
 * web/app.js
 * Frontend JavaScript logic for Hiver Twitter AI Support Agent Dashboard.
 */

const API_BASE = "http://127.0.0.1:8000";

const presets = {
    tracking: "Where is my order #12345? Tracking shows delayed since yesterday.",
    security: "Someone logged into my account from abroad and changed my email address!",
    ambiguous: "Hi help please",
    billing: "I was charged twice for my subscription this month."
};

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
});

async function checkHealth() {
    const statusText = document.getElementById("api-status-text");
    const statusDot = document.querySelector(".status-dot");
    const providerText = document.getElementById("provider-mode-text");

    try {
        const resp = await fetch(`${API_BASE}/api/v1/health`);
        if (resp.ok) {
            const data = await resp.json();
            statusText.textContent = `API: Healthy (${data.index_items.toLocaleString()} vectors)`;
            statusDot.className = "status-dot healthy";
            providerText.textContent = `Provider: ${data.generator_provider.toUpperCase()} (Mock)`;
        } else {
            statusText.textContent = "API: Degraded";
            statusDot.className = "status-dot";
        }
    } catch (e) {
        statusText.textContent = "API: Offline (Server Not Started)";
        statusDot.className = "status-dot";
    }
}

function loadPreset(type) {
    const textarea = document.getElementById("inquiry-text");
    if (presets[type]) {
        textarea.value = presets[type];
    }
}

async function analyzeInquiry() {
    const textarea = document.getElementById("inquiry-text");
    const query = textarea.value.trim();
    if (!query) {
        alert("Please enter a customer message or select a test scenario.");
        return;
    }

    const btnText = document.getElementById("btn-text");
    const btnSpinner = document.getElementById("btn-spinner");
    btnText.textContent = "Analyzing...";
    btnSpinner.classList.remove("hidden");

    try {
        const resp = await fetch(`${API_BASE}/api/v1/inquire`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ customer_message: query })
        });

        if (!resp.ok) {
            const errData = await resp.json();
            throw new Error(errData.detail || "API Error");
        }

        const data = await resp.json();
        updateUI(data);

    } catch (e) {
        alert(`Inquiry Analysis Error: ${e.message}`);
    } finally {
        btnText.textContent = "Analyze Inquiry";
        btnSpinner.classList.add("hidden");
    }
}

function updateUI(data) {
    // 1. Update Triage Decision Banner
    const banner = document.getElementById("decision-banner");
    const icon = document.getElementById("decision-icon");
    const title = document.getElementById("decision-title");
    const subtitle = document.getElementById("decision-subtitle");

    if (data.decision === "AUTO") {
        banner.className = "decision-banner auto";
        icon.textContent = "🟢";
        title.textContent = "AUTO HANDLE";
        subtitle.textContent = "Query cleared for automated AI response generation.";
    } else {
        banner.className = "decision-banner escalate";
        icon.textContent = "🔴";
        title.textContent = "ESCALATED TO HUMAN QUEUE";
        subtitle.textContent = `Halted response generation. Reason: ${data.escalation_reason}`;
    }

    // 2. Update Classification Metrics
    document.getElementById("intent-val").textContent = data.predicted_intent;
    const confPct = Math.round(data.intent_confidence * 100);
    document.getElementById("confidence-val").textContent = `${confPct}%`;
    document.getElementById("confidence-fill").style.width = `${confPct}%`;
    document.getElementById("reason-val").textContent = data.escalation_reason;
    document.getElementById("similarity-val").textContent = data.retrieval_similarity.toFixed(4);

    // 3. Update Response Content
    const responseBox = document.getElementById("response-content");
    if (data.decision === "AUTO") {
        responseBox.className = "response-content";
        responseBox.textContent = data.generated_response || "No reply generated.";
    } else {
        responseBox.className = "response-content halted";
        responseBox.textContent = "[HALTED - Ticket escalated to human support queue]";
    }

    // 4. Update Top-3 RAG Evidence List
    const evidenceList = document.getElementById("evidence-list");
    evidenceList.innerHTML = "";

    if (data.retrieved_evidence && data.retrieved_evidence.length > 0) {
        data.retrieved_evidence.forEach((item, idx) => {
            const card = document.createElement("div");
            card.className = "evidence-card-item";
            card.innerHTML = `
                <span class="evidence-id">#${idx + 1} (${item.interaction_id})</span>
                <span class="evidence-text">"${item.brand_response}"</span>
                <span class="evidence-score">$S_{cos}$: ${item.similarity_score.toFixed(4)}</span>
            `;
            evidenceList.appendChild(card);
        });
    } else {
        evidenceList.innerHTML = '<div class="empty-evidence">No evidence cases retrieved.</div>';
    }
}
