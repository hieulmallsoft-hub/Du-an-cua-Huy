from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from groundwater.inference import GroundwaterService, load_service

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = Path(os.getenv("GROUNDWATER_ARTIFACT", BASE_DIR / "artifacts" / "model.pkl"))

app = FastAPI(
    title="Groundwater Level Prediction API",
    description="Predict next groundwater level and short-term forecast from VAR/VECM/LSTM/hybrid model.",
    version="1.0.0",
)

SERVICE: GroundwaterService | None = None


class PredictRequest(BaseModel):
    history_levels: Optional[List[float]] = Field(
        default=None,
        description="Recent groundwater levels (oldest to newest). If omitted, service uses artifact default history.",
    )
    exogenous_values: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional exogenous feature values, for example rainfall, temperature, pumping_rate.",
    )


class ForecastRequest(BaseModel):
    steps: int = Field(default=7, ge=1, le=90)
    history_levels: Optional[List[float]] = Field(
        default=None,
        description="Recent groundwater levels (oldest to newest).",
    )
    exogenous_values: Optional[Dict[str, float]] = Field(
        default=None,
        description="One exogenous dictionary reused for every forecast step.",
    )
    exogenous_sequence: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Optional list of exogenous dictionaries for each forecast step.",
    )


def get_service() -> GroundwaterService:
    global SERVICE
    if SERVICE is None:
        if not ARTIFACT_PATH.exists():
            raise HTTPException(status_code=503, detail="Artifact not found. Train model first.")
        SERVICE = load_service(ARTIFACT_PATH)
    return SERVICE


def read_dataset_meta(source_data: str | None) -> Dict[str, Any] | None:
    if not source_data:
        return None

    source_path = Path(source_data)
    if not source_path.is_absolute():
        source_path = BASE_DIR / source_path
    meta_path = source_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None

    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "artifact_path": str(ARTIFACT_PATH),
        "model_loaded": ARTIFACT_PATH.exists(),
    }


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dự đoán mực nước ngầm</title>
  <style>
    :root {
      --bg: #f3f7fb;
      --panel: #ffffff;
      --ink: #0b1f3a;
      --muted: #607086;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --border: #d5deea;
      --soft: #eef6f6;
      --soft-blue: #eef5ff;
      --error: #b42318;
      --success: #087443;
      --warning: #b45309;
      --shadow: 0 18px 45px rgba(15, 35, 65, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background:
        linear-gradient(180deg, rgba(15, 118, 110, 0.10), rgba(37, 99, 235, 0.05) 42%, transparent 78%),
        var(--bg);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }
    .hero {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: clamp(1.9rem, 3vw, 3rem);
      line-height: 1.05;
      letter-spacing: 0;
    }
    .eyebrow {
      display: inline-flex;
      width: fit-content;
      margin-bottom: 10px;
      color: #0f766e;
      background: #dff7f3;
      border: 1px solid #bce8e0;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    p.meta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 1rem;
      max-width: 760px;
      line-height: 1.55;
    }
    .hero-aside {
      min-width: 210px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(213, 222, 234, 0.9);
      border-radius: 12px;
      padding: 14px;
      box-shadow: var(--shadow);
    }
    .hero-aside .label {
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      font-weight: 700;
    }
    .hero-aside .value {
      margin-top: 6px;
      font-size: 1.3rem;
      font-weight: 800;
      color: var(--accent);
    }
    .info-strip {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin: 16px 0 14px;
    }
    .info-item {
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
    }
    .info-item b {
      display: block;
      margin-bottom: 4px;
      color: var(--ink);
    }
    .info-item span {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }
    .data-info {
      margin-top: 10px;
      font-size: 0.92rem;
      color: #1e3a5f;
      background: var(--soft-blue);
      border: 1px solid #cfe0ff;
      border-radius: 10px;
      padding: 12px 14px;
      line-height: 1.5;
    }
    .model-grid {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }
    .metric-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      box-shadow: 0 10px 28px rgba(15, 35, 65, 0.06);
    }
    .metric-card .k {
      font-size: 0.8rem;
      color: var(--muted);
      text-transform: uppercase;
      font-weight: 700;
    }
    .metric-card .v {
      margin-top: 6px;
      font-weight: 700;
      color: #0b3b60;
      font-size: 1.12rem;
    }
    .card {
      margin-top: 16px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
      box-shadow: var(--shadow);
    }
    .section-title {
      margin: 0 0 14px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      border-bottom: 1px solid #e3e9f2;
      padding-bottom: 12px;
    }
    .section-title h2 {
      margin: 0;
      font-size: 1.08rem;
    }
    .section-title span {
      color: var(--muted);
      font-size: 0.88rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 14px;
    }
    label {
      font-weight: 600;
      font-size: 0.92rem;
      display: block;
      margin-bottom: 7px;
    }
    input, textarea {
      width: 100%;
      border: 1px solid #c5cfdd;
      border-radius: 10px;
      padding: 11px 12px;
      font-size: 0.95rem;
      background: #fff;
      color: var(--ink);
      outline: none;
      transition: border-color 0.15s, box-shadow 0.15s;
    }
    input:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14);
    }
    textarea { min-height: 92px; resize: vertical; line-height: 1.5; }
    .actions {
      display: flex;
      gap: 12px;
      margin-top: 16px;
      flex-wrap: wrap;
    }
    button {
      border: none;
      border-radius: 9px;
      padding: 11px 16px;
      font-weight: 700;
      cursor: pointer;
      color: #fff;
      background: var(--accent);
      box-shadow: 0 10px 22px rgba(15, 118, 110, 0.22);
    }
    button.secondary {
      background: var(--accent-2);
      box-shadow: 0 10px 22px rgba(37, 99, 235, 0.18);
    }
    #status {
      margin-top: 14px;
      font-size: 0.9rem;
      color: var(--success);
      font-weight: 700;
    }
    #status.error { color: var(--error); font-weight: 700; }
    .trend {
      margin-top: 10px;
      font-size: 0.92rem;
      font-weight: 700;
      color: var(--success);
    }
    .trend.down { color: #842029; }
    .trend.flat { color: #4b5563; }
    .result {
      margin-top: 12px;
      background: #f7fafc;
      border: 1px solid #d9e2ee;
      border-radius: 10px;
      padding: 14px;
      font-family: Consolas, monospace;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.88rem;
      min-height: 92px;
    }
    .chart-wrap {
      margin-top: 14px;
      background: #fff;
      border: 1px solid #d8dee9;
      border-radius: 10px;
      padding: 10px;
    }
    #forecastChart {
      width: 100%;
      height: 260px;
      display: block;
      background: #fbfdff;
      border-radius: 8px;
    }
    @media (max-width: 760px) {
      .hero { grid-template-columns: 1fr; }
      .hero-aside { min-width: 0; }
      .info-strip { grid-template-columns: 1fr; }
      .section-title { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <header class="hero">
      <div>
        <div class="eyebrow">Đồ án tốt nghiệp</div>
        <h1>Hệ thống dự đoán mực nước ngầm</h1>
        <p class="meta">Ứng dụng mô phỏng quy trình dự báo biến <b>groundwater_level</b> cho một trạm quan trắc theo chuỗi thời gian, hỗ trợ dự đoán bước kế tiếp và dự báo nhiều bước trong tương lai.</p>
      </div>
      <aside class="hero-aside">
        <div class="label">Trạng thái mô hình</div>
        <div class="value" id="modelStatus">Đang tải</div>
      </aside>
    </header>

    <div class="info-strip">
      <div class="info-item">
        <b>Dự đoán t+1</b>
        <span>Ước lượng giá trị mực nước ngầm ở kỳ tiếp theo dựa trên chuỗi lịch sử.</span>
      </div>
      <div class="info-item">
        <b>Dự báo N bước</b>
        <span>Sinh chuỗi dự báo ngắn hạn cho 1 đến 90 kỳ tiếp theo.</span>
      </div>
      <div class="info-item">
        <b>Phạm vi dữ liệu</b>
        <span>Kết quả áp dụng cho trạm quan trắc đã huấn luyện, không đại diện cho toàn bộ khu vực.</span>
      </div>
    </div>

    <div id="dataInfo" class="data-info">Đang tải thông tin mô hình và bộ dữ liệu...</div>
    <div id="modelCards" class="model-grid"></div>

    <section class="card">
      <div class="section-title">
        <h2>Thông tin đầu vào dự báo</h2>
        <span>Nhập chuỗi theo thứ tự từ cũ đến mới</span>
      </div>
      <div class="grid">
        <div style="grid-column: 1 / -1;">
          <label for="historyLevels">Chuỗi lịch sử groundwater_level (cách nhau bằng dấu phẩy)</label>
          <textarea id="historyLevels">14.2,14.1,14.0,13.9,13.8,13.7,13.6,13.5,13.4,13.3,13.2,13.1,13.0,12.9,12.8,12.7,12.6,12.5,12.4,12.3,12.2,12.1,12.0,11.9,11.8,11.7,11.6,11.5,11.4,11.3</textarea>
        </div>
        <div>
          <label for="steps">Số bước dự báo N (1-90)</label>
          <input id="steps" type="number" min="1" max="90" value="7" />
        </div>
        <div>
          <label for="exogenousJson">Biến ngoại sinh JSON (tùy chọn)</label>
          <input id="exogenousJson" type="text" value='{"rainfall":12.5,"temperature":31.0,"pumping_rate":2.4}' />
        </div>
      </div>

      <div class="actions">
        <button id="btnPredict">Dự đoán t+1</button>
        <button id="btnForecast" class="secondary">Dự báo N bước</button>
      </div>
      <div id="status"></div>
      <div id="trend" class="trend"></div>
      <pre id="result" class="result">Sẵn sàng nhận dữ liệu đầu vào.</pre>
      <div class="chart-wrap">
        <canvas id="forecastChart" width="920" height="260"></canvas>
      </div>
    </section>
  </main>

  <script>
    const statusEl = document.getElementById("status");
    const trendEl = document.getElementById("trend");
    const resultEl = document.getElementById("result");
    const dataInfoEl = document.getElementById("dataInfo");
    const modelCardsEl = document.getElementById("modelCards");
    const modelStatusEl = document.getElementById("modelStatus");
    const canvas = document.getElementById("forecastChart");
    const ctx = canvas.getContext("2d");

    function asNumber(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) {
        throw new Error("Dữ liệu đầu vào phải là số hợp lệ.");
      }
      return n;
    }

    function parseHistory() {
      const raw = document.getElementById("historyLevels").value.trim();
      if (!raw) return null;
      const parts = raw.split(",").map(x => Number(x.trim())).filter(x => !Number.isNaN(x));
      if (!parts.length) {
        throw new Error("Chuỗi lịch sử không hợp lệ.");
      }
      return parts.length ? parts : null;
    }

    function parseExogenous() {
      const raw = document.getElementById("exogenousJson").value.trim();
      if (!raw) return null;
      try {
        const obj = JSON.parse(raw);
        if (!obj || typeof obj !== "object" || Array.isArray(obj)) {
          throw new Error("JSON biến ngoại sinh phải là một object.");
        }
        const normalized = {};
        for (const [k, v] of Object.entries(obj)) {
          normalized[String(k)] = asNumber(v);
        }
        return Object.keys(normalized).length ? normalized : null;
      } catch (_) {
        throw new Error("JSON biến ngoại sinh không hợp lệ.");
      }
    }

    async function callApi(path, payload) {
      const res = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Request failed");
      }
      return data;
    }

    function trendText(values, unit) {
      if (!values || values.length < 2) {
        trendEl.className = "trend flat";
        return "Xu hướng: chưa đủ dữ liệu để đánh giá.";
      }
      const first = Number(values[0]);
      const last = Number(values[values.length - 1]);
      const delta = last - first;
      const suffix = ` (${delta.toFixed(4)} ${unit}/N bước)`;
      if (delta > 0.02) {
        trendEl.className = "trend";
        return "Xu hướng: TĂNG" + suffix;
      }
      if (delta < -0.02) {
        trendEl.className = "trend down";
        return "Xu hướng: GIẢM" + suffix;
      }
      trendEl.className = "trend flat";
      return "Xu hướng: ỔN ĐỊNH" + suffix;
    }

    function setStatus(message, isError = false) {
      statusEl.className = isError ? "error" : "";
      statusEl.textContent = message;
    }

    function clearChart() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#52606d";
      ctx.font = "13px Segoe UI";
      ctx.fillText("Biểu đồ lịch sử và kết quả dự báo sẽ hiển thị tại đây.", 16, 24);
    }

    function line(series, color, x0, y0, w, h, minV, maxV, totalPoints) {
      if (!series.length) return;
      const scaleY = (v) => y0 + h - ((v - minV) / Math.max(1e-9, maxV - minV)) * h;
      ctx.beginPath();
      for (let i = 0; i < series.length; i++) {
        const x = x0 + (i / Math.max(1, totalPoints - 1)) * w;
        const y = scaleY(series[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    function drawForecastChart(history, forecast) {
      if (!history || !history.length) {
        clearChart();
        return;
      }

      const histTail = history.slice(-30);
      const all = histTail.concat(forecast || []);
      const minV = Math.min(...all);
      const maxV = Math.max(...all);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const pad = { left: 48, right: 18, top: 20, bottom: 32 };
      const x0 = pad.left;
      const y0 = pad.top;
      const w = canvas.width - pad.left - pad.right;
      const h = canvas.height - pad.top - pad.bottom;

      ctx.strokeStyle = "#d6deea";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = y0 + (i / 4) * h;
        ctx.beginPath();
        ctx.moveTo(x0, y);
        ctx.lineTo(x0 + w, y);
        ctx.stroke();
      }
      for (let i = 0; i <= 6; i++) {
        const x = x0 + (i / 6) * w;
        ctx.beginPath();
        ctx.moveTo(x, y0);
        ctx.lineTo(x, y0 + h);
        ctx.stroke();
      }

      const totalPoints = histTail.length + (forecast ? forecast.length : 0);
      line(histTail, "#1d4ed8", x0, y0, w, h, minV, maxV, totalPoints);
      if (forecast && forecast.length) {
        const merged = histTail.concat(forecast);
        line(merged, "#f97316", x0, y0, w, h, minV, maxV, totalPoints);
      }

      ctx.fillStyle = "#1f2937";
      ctx.font = "12px Segoe UI";
      ctx.fillText(`Min: ${minV.toFixed(2)}`, 12, canvas.height - 14);
      ctx.fillText(`Max: ${maxV.toFixed(2)}`, 120, canvas.height - 14);

      ctx.fillStyle = "#1d4ed8";
      ctx.fillRect(canvas.width - 210, 12, 12, 12);
      ctx.fillStyle = "#1f2937";
      ctx.fillText("Lịch sử", canvas.width - 192, 22);

      ctx.fillStyle = "#f97316";
      ctx.fillRect(canvas.width - 120, 12, 12, 12);
      ctx.fillStyle = "#1f2937";
      ctx.fillText("Dự báo", canvas.width - 102, 22);
    }

    async function loadModelInfo() {
      try {
        const res = await fetch("/model-info");
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Không lấy được thông tin mô hình");

        const meta = data.dataset_meta || {};
        const parts = [];
        parts.push(`Mô hình: ${data.model_name}`);
        parts.push(`Biến mục tiêu: ${data.target_col}`);
        if (meta.site_no) parts.push(`Trạm quan trắc: ${meta.site_no}`);
        if (meta.state_cd) parts.push(`Khu vực: ${meta.state_cd}`);
        if (meta.rows) parts.push(`Số dòng dữ liệu: ${meta.rows}`);
        if (meta.start_dt && meta.end_dt) parts.push(`Khoảng dữ liệu: ${meta.start_dt} đến ${meta.end_dt}`);
        if (Array.isArray(data.exogenous_cols) && data.exogenous_cols.length) {
          parts.push(`Biến ngoại sinh: ${data.exogenous_cols.join(", ")}`);
        }
        dataInfoEl.textContent = parts.join(" | ");
        modelStatusEl.textContent = data.model_name || "Sẵn sàng";

        const metrics = data.training_metrics || {};
        const selected = metrics[data.model_name] || {};
        const cards = [
          { key: "Mô hình", value: data.model_name || "-" },
          { key: "RMSE", value: selected.rmse != null ? Number(selected.rmse).toFixed(4) : "-" },
          { key: "MAE", value: selected.mae != null ? Number(selected.mae).toFixed(4) : "-" },
          { key: "R2", value: selected.r2 != null ? Number(selected.r2).toFixed(4) : "-" },
        ];
        modelCardsEl.innerHTML = cards
          .map((c) => `<div class="metric-card"><div class="k">${c.key}</div><div class="v">${c.value}</div></div>`)
          .join("");
      } catch (err) {
        dataInfoEl.textContent = "Không tải được thông tin mô hình hoặc bộ dữ liệu.";
        modelStatusEl.textContent = "Lỗi";
        modelCardsEl.innerHTML = "";
      }
    }

    async function doPredict() {
      try {
        setStatus("Đang thực hiện dự đoán t+1...");
        trendEl.textContent = "";
        trendEl.className = "trend";
        const payload = { history_levels: parseHistory(), exogenous_values: parseExogenous() };
        const data = await callApi("/predict-next", payload);
        const value = Number(data.prediction_next).toFixed(4);
        resultEl.textContent = `Kết quả dự đoán t+1\\n- Giá trị dự báo: ${value} ${data.unit}\\n- Mô hình sử dụng: ${data.model_name}`;
        const history = payload.history_levels || [];
        drawForecastChart(history, [Number(data.prediction_next)]);
        setStatus("Hoàn tất dự đoán.");
      } catch (err) {
        setStatus(err.message, true);
      }
    }

    async function doForecast() {
      try {
        setStatus("Đang thực hiện dự báo N bước...");
        const steps = Number(document.getElementById("steps").value || "7");
        if (!Number.isInteger(steps) || steps < 1 || steps > 90) {
          throw new Error("Số bước N phải nằm trong khoảng từ 1 đến 90.");
        }
        const payload = { steps, history_levels: parseHistory(), exogenous_values: parseExogenous() };
        const data = await callApi("/forecast", payload);
        const lines = (data.forecast || []).map((v, idx) => `t+${idx + 1}: ${Number(v).toFixed(4)} ${data.unit}`);
        resultEl.textContent = `Kết quả dự báo ${data.steps} bước\\n` + lines.join("\\n");
        trendEl.textContent = trendText(data.forecast || [], data.unit || "unit");
        drawForecastChart(payload.history_levels || [], (data.forecast || []).map(Number));
        setStatus("Hoàn tất dự báo.");
      } catch (err) {
        setStatus(err.message, true);
      }
    }

    document.getElementById("btnPredict").addEventListener("click", doPredict);
    document.getElementById("btnForecast").addEventListener("click", doForecast);
    loadModelInfo();
    clearChart();
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    svc = get_service()
    dataset_meta = read_dataset_meta(svc.source_data)
    return {
        "model_name": svc.model_name,
        "target_col": svc.target_col,
        "target_description": "Predicts groundwater_level for one observation site over time.",
        "prediction_scope": {
            "predict_next": "t+1 next step forecast",
            "forecast": "multi-step future forecast for N steps",
        },
        "source_data": svc.source_data,
        "dataset_meta": dataset_meta,
        "horizon": svc.horizon,
        "lags": svc.lags,
        "rolling_windows": svc.rolling_windows,
        "exogenous_cols": svc.exogenous_cols,
        "trained_at": svc.trained_at,
        "split": svc.split,
        "training_metrics": svc.training_metrics,
    }


@app.post("/predict-next")
def predict_next(payload: PredictRequest) -> Dict[str, Any]:
    svc = get_service()
    try:
        pred = svc.predict_next(
            history_levels=payload.history_levels,
            exogenous_values=payload.exogenous_values,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "model_name": svc.model_name,
        "prediction_next": pred,
        "unit": "mét",
    }


@app.post("/forecast")
def forecast(payload: ForecastRequest) -> Dict[str, Any]:
    svc = get_service()
    try:
        preds = svc.forecast(
            history_levels=payload.history_levels,
            steps=payload.steps,
            exogenous_values=payload.exogenous_values,
            exogenous_sequence=payload.exogenous_sequence,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "model_name": svc.model_name,
        "steps": payload.steps,
        "forecast": preds,
        "unit": "mét",
    }
