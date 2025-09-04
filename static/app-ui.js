/* ==== expose globals for inline onclick ==== */
window.openFileDialog = function () {
  const input = document.getElementById('fileInput');
  if (input) input.click();
};

window.clearFile = function () {
  const input = document.getElementById('fileInput');
  if (input) input.value = '';
  const info = document.getElementById('fileInfo');
  if (info) info.style.display = 'none';
  const btn = document.getElementById('uploadBtn');
  if (btn) btn.disabled = true;
};

// static/chart.js
(() => {
  'use strict';
// --- small utils -------------------------------------------------------------
const fmtNum = (v) =>
  typeof v === "number" && Number.isFinite(v) ? v.toLocaleString() : v;

const safe = (v) => {
  // 0은 유효값, null/undefined/빈문자/NaN/Infinity만 N/A 처리
  if (v === null || v === undefined || v === "") return "N/A";
  if (typeof v === "number" && !Number.isFinite(v)) return "N/A";
  return v;
};

const wrap = document.getElementById('pieWrap');
if (wrap) wrap.style.height = '260px';

// IP 프로토콜 번호 → 이름
const protoMap = { 6: "tcp", 17: "udp", 1: "icmp" };
const mapProto = (v) => {
  if (v === null || v === undefined) return "N/A";
  const n = Number(v);
  if (Number.isFinite(n)) return protoMap[n] ?? String(n);
  return String(v).toLowerCase();
};

// 전역 차트 레지스트리(재렌더 시 destroy)
window.__charts = window.__charts || {};
function _mountChart(id, cfg) {
  const el = document.getElementById(id);
  if (!el) return null;
  if (window.__charts[id]) {
    try { window.__charts[id].destroy(); } catch {}
  }
  const chart = new Chart(el, cfg);
  window.__charts[id] = chart;
  return chart;
}

// --- charts ------------------------------------------------------------------
 function renderPieChart(id, normal, malicious) {
  const total = (normal ?? 0) + (malicious ?? 0);
  return _mountChart(id, {
    type: "doughnut",
    data: {
      labels: ["Normal", "Malicious"],
      datasets: [
        {
          data: [normal ?? 0, malicious ?? 0],
          backgroundColor: ["#00ff41", "#ff0040"],
          borderColor: ["#0a7b3d", "#a0002a"],
          borderWidth: 1.5,
          hoverOffset: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: {
          callbacks: {
            label(ctx) {
              const val = ctx.raw ?? 0;
              const pct = total ? ((val / total) * 100).toFixed(1) : "0.0";
              return `${ctx.label}: ${fmtNum(val)} (${pct}%)`;
            },
          },
        },
        legend: { labels: { color: "#00ff41" } },
      },
    },
  });
}

 function renderClassBar(id, byClass) {
  // byClass = { "DDoS": 5123, "PortScan": 233, "Benign": 1200, ... }
  if (!byClass || typeof byClass !== "object") return;

  const entries = Object.entries(byClass)
    .map(([k, v]) => [String(k), Number(v) || 0])
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8); // 상위 8개만

  const labels = entries.map((e) => e[0]);
  const data = entries.map((e) => e[1]);

  return _mountChart(id, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Count",
          data,
          backgroundColor: "#ff0040",
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: "#00ff41" },
          grid: { color: "rgba(0,255,65,0.1)" },
        },
        y: { ticks: { color: "#00ff41" } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label(ctx) {
              return ` ${fmtNum(ctx.raw ?? 0)}`;
            },
          },
        },
      },
    },
  });
}

 function renderTrendLine(id, points) {
  // points = [{ ts, malicent }, ...]  (관리자 대시보드에서 사용)
  return _mountChart(id, {
    type: "line",
    data: {
      labels: (points ?? []).map((p) => new Date(p.ts).toLocaleString()),
      datasets: [
        {
          label: "Malicious %",
          data: (points ?? []).map((p) => p.malicent),
          borderColor: "#ff0040",
          fill: false,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#00ff41" } },
      },
      scales: {
        x: { ticks: { color: "#00ff41" }, grid: { color: "rgba(0,255,65,0.1)" } },
        y: { ticks: { color: "#00ff41" }, grid: { color: "rgba(0,255,65,0.1)" } },
      },
    },
  });
}

// --- table -------------------------------------------------------------------
 function renderTopMaliciousTable(tbodySelector, rows) {
// app-ui.js 안 displayResults()의 테이블 렌더 부분만 교체
  const tbody = document.querySelector('#malTable tbody');
  if (tbody) {
    tbody.innerHTML = '';

    const safeNum = (v) => {
      if (v === null || v === undefined) return 'N/A';
      const n = Number(v);
      if (!Number.isFinite(n)) return 'N/A';
      // 정수는 그대로, 실수는 소수 3자리
      return Number.isInteger(n) ? n : n.toFixed(3);
    };
    const safeStr = (v) => {
      if (v === null || v === undefined) return 'N/A';
      const s = String(v).trim();
      return s === '' || s.toLowerCase() === 'nan' ? 'N/A' : s;
    };

    const pick = (row, keys, fallback = null) => {
      for (const k of keys) {
        if (k in row && row[k] !== undefined && row[k] !== null) return row[k];
      }
      return fallback;
    };

    // 백엔드가 주는 top_malicious 사용
    if (Array.isArray(data.top_malicious) && data.top_malicious.length > 0) {
      data.top_malicious.forEach((row) => {
        // 후보 컬럼(서로 다른 데이터셋 대응)
        const duration = pick(row, ['duration', 'dur', 'Flow Duration'], 0);
        const protocol = pick(row, ['protocol', 'proto', 'Protocol', 'protocol_type'], 'N/A');
        const srcBytes = pick(
          row,
          ['src_bytes', 'sbytes', 'Total Length of Fwd Packets', 'Fwd Packets Length Total'],
          0
        );
        const dstBytes = pick(
          row,
          ['dst_bytes', 'dbytes', 'Total Length of Bwd Packets', 'Bwd Packets Length Total'],
          0
        );

        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${safeNum(duration)}</td>
          <td>${safeStr(protocol).toLowerCase()}</td>
          <td>${safeNum(srcBytes)}</td>
          <td>${safeNum(dstBytes)}</td>
        `;
        tbody.appendChild(tr);
      });
    } else {
      tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No malicious entries found</td></tr>';
    }
  }
}

// --- helper: 간이 byClass 추정 (백엔드가 안 주는 경우) ------------------------
 function inferByClassFallback(data) {
  // 1순위: data.by_class 사용
  if (data && data.by_class && typeof data.by_class === "object") return data.by_class;

  // 2순위: top_malicious의 prediction 집계(대표성 낮음)
  if (Array.isArray(data?.top_malicious)) {
    const c = {};
    for (const r of data.top_malicious) {
      const k = String(r.prediction ?? "Malicious");
      c[k] = (c[k] ?? 0) + 1;
    }
    // Normal이 떨어져 있으면 추가
    if (data?.normal) c["Benign"] = data.normal;
    return c;
  }

  // 3순위: Normal/Malicious만 막대(최소 표시)
  if (typeof data?.normal === "number" && typeof data?.malicious === "number") {
    return { Benign: data.normal, Malicious: data.malicious };
  }
  return null;
}

if (typeof initializeElements === "function") {
  initializeElements();
}

window.openFileDialog = openFileDialog;
window.initializeElements = initializeElements;
window.displayResults = displayResults;
window.showNotification = showNotification;

// Chart helpers를 전역으로 쓰면 아래도 노출
window.renderPieChart = renderPieChart;
window.renderBarChart = renderBarChart;
window.renderLineChart = renderLineChart;
})();