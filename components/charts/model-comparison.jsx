"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";

const data = [
  {
    name: "Random Forest",
    Accuracy: 0.8,
    Recall: 0.79,
    "F1-Score": 0.81,
  },
  {
    name: "XGBoost",
    Accuracy: 0.85,
    Recall: 0.79,
    "F1-Score": 0.83,
  },
  {
    name: "Stacked Ensemble",
    Accuracy: 0.88,
    Recall: 0.85,
    "F1-Score": 0.87,
  },
];

const COLORS = {
  Accuracy: "#0d7377",
  Recall: "#f59e0b",
  "F1-Score": "#10b981",
};

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload) return null;
  return (
    <div className="rounded-lg border border-border bg-card p-3 shadow-lg">
      <p className="text-sm font-semibold text-foreground mb-2">{label}</p>
      {payload.map((entry) => (
        <div key={entry.name} className="flex items-center gap-2 text-sm">
          <div
            className="h-2.5 w-2.5 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-muted-foreground">{entry.name}:</span>
          <span className="font-mono font-medium text-foreground">
            {(entry.value * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

export default function ModelComparisonChart() {
  return (
    <div>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-foreground">
          Model Performance Comparison
        </h3>
        <p className="text-sm text-muted-foreground">
          Cleveland Dataset -- Accuracy, Recall, and F1-Score across three
          model architectures
        </p>
      </div>
      <div className="h-80 sm:h-96">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 10, right: 20, left: 0, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="name"
              tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 12, color: "var(--muted-foreground)" }}
            />
            <Bar
              dataKey="Accuracy"
              fill={COLORS.Accuracy}
              radius={[4, 4, 0, 0]}
            />
            <Bar
              dataKey="Recall"
              fill={COLORS.Recall}
              radius={[4, 4, 0, 0]}
            />
            <Bar
              dataKey="F1-Score"
              fill={COLORS["F1-Score"]}
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Summary cards */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        {data.map((model) => (
          <div
            key={model.name}
            className="rounded-lg border border-border bg-muted/30 p-3 text-center"
          >
            <p className="text-xs font-medium text-muted-foreground">
              {model.name}
            </p>
            <p className="mt-1 text-xl font-bold font-mono text-primary">
              {(model.Accuracy * 100).toFixed(0)}%
            </p>
            <p className="text-xs text-muted-foreground">Accuracy</p>
          </div>
        ))}
      </div>
    </div>
  );
}
