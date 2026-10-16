(() => {
  const form = document.getElementById("riskForm");
  const submitBtn = document.getElementById("submitBtn");
  const errorNote = document.getElementById("errorNote");
  const verdict = document.getElementById("verdict");

  const incomeInput = document.getElementById("person_income");
  const amountInput = document.getElementById("loan_amnt");
  const gradeInput = document.getElementById("loan_grade");
  const rateInput = document.getElementById("loan_int_rate");
  const percentInput = document.getElementById("loan_percent_income");
  const percentHint = document.getElementById("percentHint");

  const gaugeFill = document.getElementById("gaugeFill");
  const gaugeThreshold = document.getElementById("gaugeThreshold");
  const probNumber = document.getElementById("probNumber");
  const stampBadge = document.getElementById("stampBadge");
  const stampText = document.getElementById("stampText");

  const factProb = document.getElementById("factProb");
  const factThreshold = document.getElementById("factThreshold");
  const factResult = document.getElementById("factResult");

  const apiDot = document.getElementById("apiDot");
  const apiStatusText = document.getElementById("apiStatusText");

  const GAUGE_CIRCUMFERENCE = 540.35; // 2 * PI * 86, matches CSS

  // Typical market interest rate benchmarks by grade to avoid unrealistic inputs (e.g. Grade A at 19%)
  const GRADE_TYPICAL_RATES = {
    A: 7.5,
    B: 10.5,
    C: 13.5,
    D: 15.5,
    E: 17.5,
    F: 19.5,
    G: 21.0,
  };

  if (gradeInput && rateInput) {
    gradeInput.addEventListener("change", () => {
      const benchmarkRate = GRADE_TYPICAL_RATES[gradeInput.value];
      if (benchmarkRate !== undefined) {
        rateInput.value = benchmarkRate.toFixed(1);
      }
    });
  }

  // ---------- Auto-calculate loan-to-income ratio ----------
  function recalcPercent() {
    const income = parseFloat(incomeInput.value) || 0;
    const amount = parseFloat(amountInput.value) || 0;
    if (income > 0 && amount >= 0) {
      const ratio = amount / income;
      percentInput.value = ratio.toFixed(2);
      if (percentHint) {
        percentHint.textContent = `Auto-calculated: loan amount ÷ annual income ($${amount.toLocaleString()} ÷ $${income.toLocaleString()} = ${ratio.toFixed(2)}).`;
      }
    }
  }
  incomeInput.addEventListener("input", recalcPercent);
  amountInput.addEventListener("input", recalcPercent);
  recalcPercent();

  // ---------- Real service status check via /health ----------
  fetch("/health", { method: "GET" })
    .then((res) => {
      if (!res.ok) throw new Error("bad status");
      return res.json();
    })
    .then((data) => {
      if (data.status === "ready" && data.model_loaded && data.explainer_loaded) {
        apiDot.className = "dot ok";
        apiStatusText.textContent = "service ready";
        if (data.threshold !== undefined && data.threshold !== null) {
          const tPct = (data.threshold * 100).toFixed(1) + "%";
          if (factThreshold) factThreshold.textContent = tPct;
          if (gaugeThreshold) gaugeThreshold.style.transform = `rotate(${data.threshold * 360}deg)`;
        }
      } else {
        apiDot.className = "dot down";
        apiStatusText.textContent = "service degraded";
      }
    })
    .catch(() => {
      apiDot.className = "dot down";
      apiStatusText.textContent = "service unreachable";
    });

  // ---------- Helpers ----------
  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.classList.toggle("loading", isLoading);
    submitBtn.querySelector(".btn-label").textContent = isLoading
      ? "Reviewing file…"
      : "Assess risk";
  }

  function showError(message) {
    errorNote.textContent = message;
    errorNote.hidden = false;
  }

  function clearError() {
    errorNote.hidden = true;
    errorNote.textContent = "";
  }

  function animateNumber(el, from, to, duration) {
    const start = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const value = from + (to - from) * eased;
      el.textContent = value.toFixed(1);
      if (t < 1) requestAnimationFrame(tick);
      else el.textContent = to.toFixed(1);
    }
    requestAnimationFrame(tick);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ---------- Render SHAP Factors ----------
  function renderShap(shapData) {
    const shapList = document.getElementById("shapList");
    const riskDownCount = document.getElementById("riskDownCount");
    const riskUpCount = document.getElementById("riskUpCount");

    if (!shapList || !shapData || !Array.isArray(shapData.factors)) return;

    if (riskDownCount) riskDownCount.textContent = shapData.risk_decreasing_count || 0;
    if (riskUpCount) riskUpCount.textContent = shapData.risk_increasing_count || 0;

    shapList.innerHTML = "";

    const maxImpact = Math.max(...shapData.factors.map((f) => f.abs_impact || 0), 0.01);

    shapData.factors.forEach((factor, idx) => {
      const isIncrease = factor.direction === "increase";
      const barPct = Math.min(100, Math.max(8, (factor.abs_impact / maxImpact) * 100)).toFixed(1);

      const item = document.createElement("div");
      item.className = "shap-item";
      item.innerHTML = `
        <div class="shap-item__top">
          <span class="shap-item__label">${escapeHtml(factor.feature)}</span>
          <span class="shap-item__val">${escapeHtml(factor.value || "—")}</span>
        </div>
        <div class="shap-item__bar-container">
          <div class="shap-item__bar ${isIncrease ? "bar--increase" : "bar--reduce"}" style="width: 0%"></div>
        </div>
        <div class="shap-item__meta">
          <span class="shap-item__effect ${isIncrease ? "effect--increase" : "effect--reduce"}">
            ${isIncrease ? "▲ Increases Risk" : "▼ Reduces Risk"}
          </span>
          <span class="shap-item__score">
            Impact (log-odds): ${factor.shap_value > 0 ? "+" : ""}${factor.shap_value.toFixed(3)}
          </span>
        </div>
      `;

      shapList.appendChild(item);

      setTimeout(() => {
        const bar = item.querySelector(".shap-item__bar");
        if (bar) bar.style.width = `${barPct}%`;
      }, 70 + idx * 30);
    });
  }

  function renderVerdict(data) {
    const probabilityPct = data.default_probability * 100;
    const thresholdPct = data.threshold * 100;
    const isHighRisk = data.Result === "High Risk";
    const isMediumRisk = data.Result === "Medium Risk";

    verdict.hidden = false;
    verdict.scrollIntoView({ behavior: "smooth", block: "nearest" });

    // Gauge fill color based on 3-tier risk band
    const offset = GAUGE_CIRCUMFERENCE * (1 - probabilityPct / 100);
    if (isHighRisk) {
      gaugeFill.style.stroke = "var(--risk-red)";
    } else if (isMediumRisk) {
      gaugeFill.style.stroke = "var(--brass)";
    } else {
      gaugeFill.style.stroke = "var(--risk-green)";
    }

    requestAnimationFrame(() => {
      gaugeFill.style.strokeDashoffset = offset;
    });

    // Threshold tick mark on gauge
    gaugeThreshold.style.transform = `rotate(${thresholdPct * 3.6}deg)`;

    // Number readout
    animateNumber(probNumber, 0, probabilityPct, 1000);

    // Stamp: 3-tier risk band
    stampBadge.classList.remove("stamp--in", "risk-high", "risk-medium");
    void stampBadge.offsetWidth; // restart animation
    if (isHighRisk) {
      stampBadge.classList.add("risk-high");
      stampText.textContent = "HIGH RISK";
    } else if (isMediumRisk) {
      stampBadge.classList.add("risk-medium");
      stampText.textContent = "MEDIUM RISK";
    } else {
      stampText.textContent = "LOW RISK";
    }
    requestAnimationFrame(() => stampBadge.classList.add("stamp--in"));

    // Ledger facts
    factProb.textContent = `${probabilityPct.toFixed(1)}%`;
    factThreshold.textContent = `${thresholdPct.toFixed(1)}%`;
    factResult.textContent = data.Result;

    // SHAP Explainability Breakdown
    if (data.shap_analysis) {
      renderShap(data.shap_analysis);
    }
  }

  // ---------- Submit ----------
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();
    setLoading(true);

    const payload = {
      person_age: parseInt(document.getElementById("person_age").value, 10),
      person_income: parseFloat(incomeInput.value),
      person_home_ownership: document.getElementById("person_home_ownership").value,
      person_emp_length: parseFloat(document.getElementById("person_emp_length").value),
      loan_intent: document.getElementById("loan_intent").value,
      loan_grade: document.getElementById("loan_grade").value,
      loan_amnt: parseFloat(amountInput.value),
      loan_int_rate: parseFloat(document.getElementById("loan_int_rate").value),
      loan_percent_income: parseFloat(percentInput.value),
      cb_person_default_on_file: document.getElementById("cb_person_default_on_file").value,
      cb_person_cred_hist_length: parseInt(document.getElementById("cb_person_cred_hist_length").value, 10),
    };

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => null);
        const detail = body && body.detail ? JSON.stringify(body.detail) : `HTTP ${res.status}`;
        throw new Error(detail);
      }

      const data = await res.json();
      renderVerdict(data);
    } catch (err) {
      showError(`Could not reach the ledger. ${err.message || "Check the service is running."}`);
    } finally {
      setLoading(false);
    }
  });
})();
