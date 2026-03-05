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
} from "recharts";

// Approximated Precision-Recall data from the graph
const prData = [
  { recall: 0.0, precision: 1.0 },
  { recall: 0.1, precision: 1.0 },
  { recall: 0.2, precision: 1.0 },
  { recall: 0.3, precision: 1.0 },
  { recall: 0.4, precision: 1.0 },
  { recall: 0.5, precision: 1.0 },
  { recall: 0.55, precision: 0.995 },
  { recall: 0.6, precision: 0.98 },
  { recall: 0.65, precision: 0.975 },
  { recall: 0.7, precision: 0.97 },
  { recall: 0.75, precision: 0.965 },
  { recall: 0.8, precision: 0.96 },
  { recall: 0.85, precision: 0.97 },
  { recall: 0.9, precision: 0.97 },
  { recall: 0.95, precision: 0.91 },
  { recall: 0.98, precision: 0.82 },
  { recall: 0.99, precision: 0.74 },
  { recall: 1.0, precision: 0.46 },
];

function CustomTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-border bg-card p-3 shadow-lg">
      <p className="text-sm font-medium text-foreground mb-1">PR Curve Point</p>
      <div className="flex flex-col gap-1 text-sm">
        <span className="text-muted-foreground">
          Recall:{" "}
          <span className="font-mono font-medium text-foreground">
            {d.recall.toFixed(3)}
          </span>
        </span>
        <span className="text-muted-foreground">
          Precision:{" "}
          <span className="font-mono font-medium text-foreground">
            {d.precision.toFixed(3)}
          </span>
        </span>
      </div>
    </div>
  );
}

export default function PrecisionRecallChart() {
  return (
    <div>
      <div className="mb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h3 className="text-lg font-semibold text-foreground">
            Precision-Recall Curve
          </h3>
          <p className="text-sm text-muted-foreground">
            Cleveland Heart Disease Dataset
          </p>
        </div>
        <div className="rounded-lg border border-primary/30 bg-secondary px-3 py-1.5">
          <span className="text-sm font-medium text-secondary-foreground">
            AP = <span className="font-mono font-bold">0.989</span>
          </span>
        </div>
      </div>
      <div className="h-80 sm:h-96">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={prData}
            margin={{ top: 10, right: 20, left: 0, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="recall"
              type="number"
              domain={[0, 1]}
              tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              label={{
                value: "Recall",
                position: "insideBottom",
                offset: -2,
                style: { fill: "var(--muted-foreground)", fontSize: 12 },
              }}
            />
            <YAxis
              domain={[0.4, 1]}
              tick={{ fill: "var(--muted-foreground)", fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: "var(--border)" }}
              label={{
                value: "Precision",
                angle: -90,
                position: "insideLeft",
                style: { fill: "var(--muted-foreground)", fontSize: 12 },
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: 12, color: "var(--muted-foreground)" }}
            />
            <Line
              type="monotone"
              dataKey="precision"
              name="Fusion Model (AP = 0.989)"
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
