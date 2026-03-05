"use client";

import { motion } from "framer-motion";

// Correlation data from the heatmap image
const features = [
  "age", "sex", "cp", "trestbps", "chol", "fbs",
  "restecg", "thalach", "exang", "oldpeak", "slope",
  "ca", "thal", "target",
];

const correlations = [
  [1.0, -0.09, 0.11, 0.29, 0.2, 0.13, 0.15, -0.39, 0.1, 0.2, 0.16, 0.36, 0.13, 0.23],
  [-0.09, 1.0, 0.01, -0.07, -0.2, 0.04, 0.03, -0.06, 0.14, 0.11, 0.03, 0.09, 0.38, 0.28],
  [0.11, 0.01, 1.0, -0.04, 0.07, -0.06, 0.06, -0.34, 0.38, 0.2, 0.15, 0.24, 0.27, 0.41],
  [0.29, -0.07, -0.04, 1.0, 0.13, 0.18, 0.15, -0.05, 0.07, 0.19, 0.12, 0.1, 0.14, 0.15],
  [0.2, -0.2, 0.07, 0.13, 1.0, 0.01, 0.17, -0.0, 0.06, 0.04, -0.01, 0.12, 0.01, 0.08],
  [0.13, 0.04, -0.06, 0.18, 0.01, 1.0, 0.07, -0.01, -0.0, 0.01, 0.05, 0.15, 0.06, 0.0],
  [0.15, 0.03, 0.06, 0.15, 0.17, 0.07, 1.0, -0.07, 0.08, 0.11, 0.14, 0.13, 0.02, 0.17],
  [-0.39, -0.06, -0.34, -0.05, -0.0, -0.01, -0.07, 1.0, -0.38, -0.35, -0.39, -0.27, -0.27, -0.42],
  [0.1, 0.14, 0.38, 0.07, 0.06, -0.0, 0.08, -0.38, 1.0, 0.29, 0.25, 0.15, 0.33, 0.42],
  [0.2, 0.11, 0.2, 0.19, 0.04, 0.01, 0.11, -0.35, 0.29, 1.0, 0.58, 0.29, 0.34, 0.42],
  [0.16, 0.03, 0.15, 0.12, -0.01, 0.05, 0.14, -0.39, 0.25, 0.58, 1.0, 0.11, 0.28, 0.33],
  [0.36, 0.09, 0.24, 0.1, 0.12, 0.15, 0.13, -0.27, 0.15, 0.29, 0.11, 1.0, 0.26, 0.46],
  [0.13, 0.38, 0.27, 0.14, 0.01, 0.06, 0.02, -0.27, 0.33, 0.34, 0.28, 0.26, 1.0, 0.53],
  [0.23, 0.28, 0.41, 0.15, 0.08, 0.0, 0.17, -0.42, 0.42, 0.42, 0.33, 0.46, 0.53, 1.0],
];

function getColor(val) {
  // Color scale: blue (negative) -> white (0) -> red (positive)
  const abs = Math.abs(val);
  if (val > 0) {
    const r = 220;
    const g = Math.round(220 - abs * 160);
    const b = Math.round(220 - abs * 180);
    return `rgb(${r}, ${g}, ${b})`;
  }
  const r = Math.round(220 - abs * 180);
  const g = Math.round(220 - abs * 140);
  const b = 220;
  return `rgb(${r}, ${g}, ${b})`;
}

function getTextColor(val) {
  return Math.abs(val) > 0.35 ? "#ffffff" : "var(--foreground)";
}

export default function CorrelationHeatmap() {
  return (
    <div>
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-foreground">
          Feature Correlation Heatmap
        </h3>
        <p className="text-sm text-muted-foreground">
          Pearson correlation between all clinical features and target in the
          Cleveland dataset
        </p>
      </div>

      {/* Interactive heatmap */}
      <div className="overflow-x-auto">
        <div className="min-w-[640px]">
          {/* Column labels */}
          <div className="flex">
            <div className="w-16 shrink-0" />
            {features.map((f) => (
              <div
                key={`col-${f}`}
                className="flex-1 text-center"
              >
                <span
                  className="text-[10px] font-medium text-muted-foreground"
                  style={{
                    writingMode: "vertical-lr",
                    transform: "rotate(180deg)",
                    display: "inline-block",
                    height: 60,
                  }}
                >
                  {f}
                </span>
              </div>
            ))}
          </div>

          {/* Matrix rows */}
          {correlations.map((row, ri) => (
            <div key={ri} className="flex">
              <div className="w-16 shrink-0 flex items-center justify-end pr-2">
                <span className="text-[10px] font-medium text-muted-foreground">
                  {features[ri]}
                </span>
              </div>
              {row.map((val, ci) => (
                <motion.div
                  key={`${ri}-${ci}`}
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: (ri + ci) * 0.005 }}
                  className="flex-1 aspect-square flex items-center justify-center border border-background/50 text-[9px] font-mono font-medium cursor-default transition-transform hover:scale-110 hover:z-10"
                  style={{
                    backgroundColor: getColor(val),
                    color: getTextColor(val),
                  }}
                  title={`${features[ri]} vs ${features[ci]}: ${val.toFixed(2)}`}
                >
                  {val.toFixed(2)}
                </motion.div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Color scale legend */}
      <div className="mt-4 flex items-center justify-center gap-2">
        <span className="text-xs text-muted-foreground">-1.0</span>
        <div
          className="h-3 w-48 rounded-full"
          style={{
            background:
              "linear-gradient(to right, rgb(40, 80, 220), rgb(220, 220, 220), rgb(220, 60, 40))",
          }}
        />
        <span className="text-xs text-muted-foreground">+1.0</span>
      </div>

      {/* Key insights */}
      <div className="mt-6 rounded-lg bg-muted/30 p-4">
        <h4 className="text-sm font-semibold text-foreground mb-2">
          Key Correlations with Target
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {[
            { feature: "thal", value: 0.53, dir: "positive" },
            { feature: "ca", value: 0.46, dir: "positive" },
            { feature: "thalach", value: -0.42, dir: "negative" },
            { feature: "exang", value: 0.42, dir: "positive" },
            { feature: "oldpeak", value: 0.42, dir: "positive" },
            { feature: "cp", value: 0.41, dir: "positive" },
          ].map((item) => (
            <div
              key={item.feature}
              className="flex items-center justify-between rounded-md bg-card px-3 py-2 border border-border text-sm"
            >
              <span className="font-medium text-foreground">
                {item.feature}
              </span>
              <span
                className={`font-mono font-medium ${
                  item.dir === "positive" ? "text-accent" : "text-primary"
                }`}
              >
                {item.value > 0 ? "+" : ""}
                {item.value.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
