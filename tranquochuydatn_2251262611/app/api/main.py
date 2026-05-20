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
  <title>Demo du doan muc nuoc ngam</title>
  <style>
    :root {
      --bg: #f6f7fb;
      --card: #ffffff;
      --ink: #14213d;
      --muted: #52606d;
      --accent: #007f5f;
      --accent-2: #2b9348;
      --border: #d8dee9;
      --error: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      color: var(--ink);
      background: linear-gradient(160deg, #ecfdf3 0%, var(--bg) 45%, #eef2ff 100%);
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 24px 14px 36px;
    }
    h1 {
      margin: 0;
      font-size: clamp(1.5rem, 2.8vw, 2.3rem);
    }
    p.meta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .notes {
      margin-top: 10px;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.5;
      font-size: 0.92rem;
    }
    .data-info {
      margin-top: 10px;
      font-size: 0.9rem;
      color: #1e3a5f;
      background: #eef6ff;
      border: 1px solid #d5e6ff;
      border-radius: 8px;
      padding: 8px 10px;
    }
    .model-grid {
      margin-top: 10px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
    }
    .metric-card {
      background: #f8fafc;
      border: 1px solid #dbe3f0;
      border-radius: 8px;
      padding: 8px 10px;
    }
    .metric-card .k {
      font-size: 0.8rem;
      color: #52606d;
    }
    .metric-card .v {
      margin-top: 2px;
      font-weight: 700;
      color: #0b3b60;
    }
    .card {
      margin-top: 14px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
    }
    label {
      font-weight: 600;
      font-size: 0.92rem;
      display: block;
      margin-bottom: 5px;
    }
    input, textarea {
      width: 100%;
      border: 1px solid #c7d0dd;
      border-radius: 9px;
      padding: 9px 10px;
      font-size: 0.94rem;
      background: #fff;
    }
    textarea { min-height: 84px; resize: vertical; }
    .actions {
      display: flex;
      gap: 10px;
      margin-top: 12px;
      flex-wrap: wrap;
    }
    button {
      border: none;
      border-radius: 10px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
      color: #fff;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
    }
    #status {
      margin-top: 10px;
      font-size: 0.9rem;
      color: var(--muted);
    }
    #status.error { color: var(--error); font-weight: 700; }
    .trend {
      margin-top: 10px;
      font-size: 0.92rem;
      font-weight: 700;
      color: #0f5132;
    }
    .trend.down { color: #842029; }
    .trend.flat { color: #4b5563; }
    .result {
      margin-top: 12px;
      background: #f8fafc;
      border: 1px dashed #b8c3d0;
      border-radius: 10px;
      padding: 12px;
      font-family: Consolas, monospace;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 0.88rem;
    }
    .chart-wrap {
      margin-top: 12px;
      background: #fff;
      border: 1px solid #d8dee9;
      border-radius: 10px;
      padding: 8px;
    }
    #forecastChart {
      width: 100%;
      height: 240px;
      display: block;
      background: #fbfdff;
      border-radius: 8px;
    }
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Demo du doan muc nuoc ngam</h1>
    <p class="meta">He thong nay du doan bien <b>groundwater_level</b> cho <b>1 tram quan trac</b> theo chuoi thoi gian.</p>
    <ul class="notes">
      <li><b>Du doan t+1</b>: gia tri ngay (hoac ky) tiep theo.</li>
      <li><b>Forecast N buoc</b>: gia tri N ngay (hoac N ky) tiep theo.</li>
      <li>Khong phai du doan cho toan tinh/toan quoc, ma la cho tram du lieu dang huan luyen.</li>
    </ul>
    <div id="dataInfo" class="data-info">Dang tai thong tin model...</div>
    <div id="modelCards" class="model-grid"></div>

    <section class="card">
      <div class="grid">
        <div style="grid-column: 1 / -1;">
          <label for="historyLevels">Chuoi lich su groundwater_level (tach boi dau phay, cu -> moi)</label>
          <textarea id="historyLevels">14.2,14.1,14.0,13.9,13.8,13.7,13.6,13.5,13.4,13.3,13.2,13.1,13.0,12.9,12.8,12.7,12.6,12.5,12.4,12.3,12.2,12.1,12.0,11.9,11.8,11.7,11.6,11.5,11.4,11.3</textarea>
        </div>
        <div>
          <label for="steps">So buoc du doan N (1-90)</label>
          <input id="steps" type="number" min="1" max="90" value="7" />
        </div>
        <div>
          <label for="exogenousJson">Bien ngoai sinh JSON (tuy chon)</label>
          <input id="exogenousJson" type="text" value='{"rainfall":12.5,"temperature":31.0,"pumping_rate":2.4}' />
        </div>
      </div>

      <div class="actions">
        <button id="btnPredict">Du doan t+1</button>
        <button id="btnForecast">Du doan N buoc</button>
      </div>
      <div id="status"></div>
      <div id="trend" class="trend"></div>
      <pre id="result" class="result">San sang.</pre>
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
    const canvas = document.getElementById("forecastChart");
    const ctx = canvas.getContext("2d");

    function asNumber(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) {
        throw new Error("Du lieu dau vao phai la so hop le.");
      }
      return n;
    }

    function parseHistory() {
      const raw = document.getElementById("historyLevels").value.trim();
      if (!raw) return null;
      const parts = raw.split(",").map(x => Number(x.trim())).filter(x => !Number.isNaN(x));
      if (!parts.length) {
        throw new Error("Chuoi lich su khong hop le.");
      }
      return parts.length ? parts : null;
    }

    function parseExogenous() {
      const raw = document.getElementById("exogenousJson").value.trim();
      if (!raw) return null;
      try {
        const obj = JSON.parse(raw);
        if (!obj || typeof obj !== "object" || Array.isArray(obj)) {
          throw new Error("JSON bien ngoai sinh phai la object.");
        }
        const normalized = {};
        for (const [k, v] of Object.entries(obj)) {
          normalized[String(k)] = asNumber(v);
        }
        return Object.keys(normalized).length ? normalized : null;
      } catch (_) {
        throw new Error("JSON bien ngoai sinh khong hop le.");
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
        return "Xu huong: khong du du lieu.";
      }
      const first = Number(values[0]);
      const last = Number(values[values.length - 1]);
      const delta = last - first;
      const suffix = ` (${delta.toFixed(4)} ${unit}/N buoc)`;
      if (delta > 0.02) {
        trendEl.className = "trend";
        return "Xu huong: TANG" + suffix;
      }
      if (delta < -0.02) {
        trendEl.className = "trend down";
        return "Xu huong: GIAM" + suffix;
      }
      trendEl.className = "trend flat";
      return "Xu huong: ON DINH" + suffix;
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
      ctx.fillText("Bieu do lich su va du doan se hien thi o day.", 16, 24);
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
      ctx.fillText("Lich su", canvas.width - 192, 22);

      ctx.fillStyle = "#f97316";
      ctx.fillRect(canvas.width - 120, 12, 12, 12);
      ctx.fillStyle = "#1f2937";
      ctx.fillText("Du doan", canvas.width - 102, 22);
    }

    async function loadModelInfo() {
      try {
        const res = await fetch("/model-info");
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Khong lay duoc model info");

        const meta = data.dataset_meta || {};
        const parts = [];
        parts.push(`Model: ${data.model_name}`);
        parts.push(`Target: ${data.target_col}`);
        if (meta.site_no) parts.push(`Tram: ${meta.site_no}`);
        if (meta.state_cd) parts.push(`State: ${meta.state_cd}`);
        if (meta.rows) parts.push(`So dong du lieu: ${meta.rows}`);
        if (meta.start_dt && meta.end_dt) parts.push(`Khoang du lieu: ${meta.start_dt} -> ${meta.end_dt}`);
        if (Array.isArray(data.exogenous_cols) && data.exogenous_cols.length) {
          parts.push(`Bien ngoai sinh: ${data.exogenous_cols.join(", ")}`);
        }
        dataInfoEl.textContent = parts.join(" | ");

        const metrics = data.training_metrics || {};
        const selected = metrics[data.model_name] || {};
        const cards = [
          { key: "Model", value: data.model_name || "-" },
          { key: "RMSE", value: selected.rmse != null ? Number(selected.rmse).toFixed(4) : "-" },
          { key: "MAE", value: selected.mae != null ? Number(selected.mae).toFixed(4) : "-" },
          { key: "R2", value: selected.r2 != null ? Number(selected.r2).toFixed(4) : "-" },
        ];
        modelCardsEl.innerHTML = cards
          .map((c) => `<div class="metric-card"><div class="k">${c.key}</div><div class="v">${c.value}</div></div>`)
          .join("");
      } catch (err) {
        dataInfoEl.textContent = "Khong tai duoc thong tin model/data.";
        modelCardsEl.innerHTML = "";
      }
    }

    async function doPredict() {
      try {
        setStatus("Dang du doan t+1...");
        trendEl.textContent = "";
        trendEl.className = "trend";
        const payload = { history_levels: parseHistory(), exogenous_values: parseExogenous() };
        const data = await callApi("/predict-next", payload);
        const value = Number(data.prediction_next).toFixed(4);
        resultEl.textContent = `Ket qua du doan t+1\\n- Gia tri: ${value} ${data.unit}\\n- Model: ${data.model_name}`;
        const history = payload.history_levels || [];
        drawForecastChart(history, [Number(data.prediction_next)]);
        setStatus("Thanh cong");
      } catch (err) {
        setStatus(err.message, true);
      }
    }

    async function doForecast() {
      try {
        setStatus("Dang du doan N buoc...");
        const steps = Number(document.getElementById("steps").value || "7");
        if (!Number.isInteger(steps) || steps < 1 || steps > 90) {
          throw new Error("So buoc N phai nam trong [1, 90].");
        }
        const payload = { steps, history_levels: parseHistory(), exogenous_values: parseExogenous() };
        const data = await callApi("/forecast", payload);
        const lines = (data.forecast || []).map((v, idx) => `t+${idx + 1}: ${Number(v).toFixed(4)} ${data.unit}`);
        resultEl.textContent = `Ket qua du doan ${data.steps} buoc\\n` + lines.join("\\n");
        trendEl.textContent = trendText(data.forecast || [], data.unit || "unit");
        drawForecastChart(payload.history_levels || [], (data.forecast || []).map(Number));
        setStatus("Thanh cong");
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
        "unit": "meters",
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
        "unit": "meters",
    }
