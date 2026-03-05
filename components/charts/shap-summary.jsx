"use client";

import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

// SHAP feature importance data (from the summary plot)
const shapData = [
  { feature: "ca", importance: 0.85, direction: "positive" },
  { feature: "cp_4.0", importance: 0.72, direction: "positive" },
  { feature: "age", importance: 0.55, direction: "mixed" },
  { feature: "chol", importance: 0.48, direction: "mixed" },
  { feature: "oldpeak", importance: 0.62, direction: "positive" },
  { feature: "thal_7.0", importance: 0.58, direction: "positive" },
  { feature: "thal_3.0", importance: 0.45, direction: "negative" },
  { feature: "sex_0.0", importance: 0.4, direction: "mixed" },
  { feature: "thalach", importance: 0.42, direction: "negative" },
  { feature: "restecg_0.0", importance: 0.35, direction: "mixed" },
  { feature: "slope_1.0", importance: 0.33, direction: "positive" },
  { feature: "trestbps", importance: 0.3, direction: "mixed" },
].sort((a, b) => b.importance - a.importance);

function getBarColor(direction) {
  switch (direction) {
    case "positive":
      return "#f0503c";
    case "negative":
      return "#0d7377";
    default:
      return "#f59e0b";
  }
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-border bg-card p-3 shadow-lg">
      <p className="text-sm font-semibold text-foreground">{d.feature}</p>
      <p className="text-sm text-muted-foreground">
        Mean |SHAP|:{" "}
        <span className="font-mono font-medium text-foreground">
          {d.importance.toFixed(3)}
        </span>
      </p>
      <p className="text-xs text-muted-foreground mt-1">
        Direction:{" "}
        <span className="capitalize font-medium">{d.direction}</span>
      </p>
    </div>
  );
}

export default function ShapSummary() {
  return (
    <div>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-foreground">
          SHAP Feature Importance
        </h3>
        <p className="text-sm text-muted-foreground">
          Global feature attribution using TreeExplainer on the XGBoost
          component
        </p>
      </div>

      <div className="h-80 sm:h-[420px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={shapData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--border)"
              horizontal={false}
            />
            <XAxis
              type="number"
              tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              label={{
                value: "Mean |SHAP Value|",
                position: "insideBottom",
                offset: -2,
                style: { fill: "var(--muted-foreground)", fontSize: 12 },
              }}
            />
            <YAxis
              type="category"
              dataKey="feature"
              tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              width={75}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={18}>
              {shapData.map((entry) => (
                <Cell
                  key={entry.feature}
                  fill={getBarColor(entry.direction)}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap items-center justify-center gap-4">
        {[
          { label: "Increases Risk", color: "#f0503c" },
          { label: "Decreases Risk", color: "#0d7377" },
          { label: "Mixed Impact", color: "#f59e0b" },
        ].map((item) => (
          <div key={item.label} className="flex items-center gap-2 text-sm">
            <div
              className="h-3 w-3 rounded-full"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-muted-foreground">{item.label}</span>
          </div>
        ))}
      </div>

      {/* Original SHAP plot image */}
      <div className="mt-6 rounded-xl border border-border overflow-hidden">
        <div className="bg-muted/30 px-4 py-2 border-b border-border">
          <p className="text-sm font-medium text-foreground">
            Original SHAP Beeswarm Plot
          </p>
        </div>
        <img
          src="/graphs/shap_summary_plot.png"
          alt="SHAP summary beeswarm plot showing feature importance and direction of impact for heart disease prediction"
          className="w-full"
        />
      </div>
    </div>
  );
}
