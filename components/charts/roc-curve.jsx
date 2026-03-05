"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

// Approximated ROC curve data points from the graph
const rocData = [
  { fpr: 0, tpr: 0 },
  { fpr: 0.005, tpr: 0.1 },
  { fpr: 0.01, tpr: 0.35 },
  { fpr: 0.015, tpr: 0.6 },
  { fpr: 0.02, tpr: 0.72 },
  { fpr: 0.025, tpr: 0.78 },
  { fpr: 0.03, tpr: 0.85 },
  { fpr: 0.04, tpr: 0.95 },
  { fpr: 0.05, tpr: 0.96 },
  { fpr: 0.07, tpr: 0.97 },
  { fpr: 0.1, tpr: 0.975 },
  { fpr: 0.15, tpr: 0.98 },
  { fpr: 0.2, tpr: 0.985 },
  { fpr: 0.3, tpr: 0.99 },
  { fpr: 0.4, tpr: 0.995 },
  { fpr: 0.5, tpr: 0.998 },
  { fpr: 0.7, tpr: 0.999 },
  { fpr: 1.0, tpr: 1.0 },
];

const randomGuess = [
  { fpr: 0, tpr: 0 },
  { fpr: 1, tpr: 1 },
];

function CustomTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-border bg-card p-3 shadow-lg">
      <p className="text-sm font-medium text-foreground mb-1">
        ROC Curve Point
      </p>
      <div className="flex flex-col gap-1 text-sm">
        <span className="text-muted-foreground">
          FPR:{" "}
          <span className="font-mono font-medium text-foreground">
            {d.fpr.toFixed(3)}
          </span>
        </span>
        <span className="text-muted-foreground">
          TPR:{" "}
          <span className="font-mono font-medium text-foreground">
            {d.tpr.toFixed(3)}
          </span>
        </span>
      </div>
    </div>
  );
}

export default function RocCurveChart() {
  return (
    <div>
      <div className="mb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h3 className="text-lg font-semibold text-foreground">ROC Curve</h3>
          <p className="text-sm text-muted-foreground">
            Receiver Operating Characteristic -- Cleveland Heart Disease Dataset
          </p>
        </div>
        <div className="rounded-lg border border-primary/30 bg-secondary px-3 py-1.5">
          <span className="text-sm font-medium text-secondary-foreground">
            AUC = <span className="font-mono font-bold">0.990</span>
          </span>
        </div>
      </div>
      <div className="h-80 sm:h-96">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="fpr"
              type="number"
              domain={[0, 1]}
              tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              label={{
                value: "False Positive Rate",
                position: "insideBottom",
                offset: -2,
                style: {
                  fill: "var(--muted-foreground)",
                  fontSize: 12,
                },
              }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              label={{
                value: "True Positive Rate",
                angle: -90,
                position: "insideLeft",
                style: {
                  fill: "var(--muted-foreground)",
                  fontSize: 12,
                },
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 12, color: "var(--muted-foreground)" }}
            />
            <Line
              data={randomGuess}
              type="linear"
              dataKey="tpr"
              name="Random Guess"
              stroke="var(--muted-foreground)"
              strokeDasharray="5 5"
              strokeWidth={1.5}
              dot={false}
            />
            <Line
              data={rocData}
              type="monotone"
              dataKey="tpr"
              name="Fusion Model (AUC = 0.990)"
              stroke="var(--primary)"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 5, fill: "var(--primary)" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
