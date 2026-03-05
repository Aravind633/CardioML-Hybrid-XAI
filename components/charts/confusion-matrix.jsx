"use client";

import { motion } from "framer-motion";

const matrix = [
  [156, 4],
  [5, 132],
];

const labels = ["No Disease", "Disease"];

const total = matrix.flat().reduce((a, b) => a + b, 0);
const accuracy = ((matrix[0][0] + matrix[1][1]) / total * 100).toFixed(1);
const precision = (matrix[1][1] / (matrix[0][1] + matrix[1][1]) * 100).toFixed(1);
const recall = (matrix[1][1] / (matrix[1][0] + matrix[1][1]) * 100).toFixed(1);

function getCellColor(row, col) {
  // Diagonal = correct, off-diagonal = errors
  if (row === col) {
    return "bg-primary/20 text-primary border-primary/30";
  }
  return "bg-accent/10 text-accent border-accent/20";
}

export default function ConfusionMatrixChart() {
  return (
    <div>
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-foreground">
          Confusion Matrix
        </h3>
        <p className="text-sm text-muted-foreground">
          Cleveland Heart Disease Dataset -- Full dataset evaluation
        </p>
      </div>

      <div className="flex flex-col items-center gap-6">
        {/* Matrix */}
        <div className="flex flex-col items-center">
          {/* Column headers */}
          <div className="flex items-center mb-2">
            <div className="w-24 sm:w-32" />
            {labels.map((label) => (
              <div
                key={`col-${label}`}
                className="w-24 sm:w-32 text-center text-sm font-medium text-muted-foreground"
              >
                {label}
              </div>
            ))}
          </div>

          <p className="text-xs text-muted-foreground mb-2 font-medium">
            Predicted Label
          </p>

          {/* Rows */}
          <div className="flex items-stretch">
            {/* Row label for "True Label" */}
            <div className="flex items-center mr-2">
              <span
                className="text-xs text-muted-foreground font-medium"
                style={{
                  writingMode: "vertical-lr",
                  transform: "rotate(180deg)",
                }}
              >
                True Label
              </span>
            </div>

            <div className="flex flex-col gap-2">
              {matrix.map((row, ri) => (
                <div key={ri} className="flex items-center gap-2">
                  <div className="w-20 sm:w-28 text-right text-sm font-medium text-muted-foreground pr-2">
                    {labels[ri]}
                  </div>
                  {row.map((val, ci) => (
                    <motion.div
                      key={`${ri}-${ci}`}
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{
                        delay: ri * 0.2 + ci * 0.2,
                        duration: 0.4,
                        type: "spring",
                      }}
                      className={`flex h-24 w-24 sm:h-32 sm:w-32 items-center justify-center rounded-xl border-2 ${getCellColor(
                        ri,
                        ci
                      )}`}
                    >
                      <span className="text-2xl sm:text-3xl font-bold font-mono">
                        {val}
                      </span>
                    </motion.div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Metrics summary */}
        <div className="grid grid-cols-3 gap-4 w-full max-w-md">
          {[
            { label: "Accuracy", value: `${accuracy}%` },
            { label: "Precision", value: `${precision}%` },
            { label: "Recall", value: `${recall}%` },
          ].map((metric) => (
            <div
              key={metric.label}
              className="rounded-lg border border-border bg-muted/30 p-3 text-center"
            >
              <p className="text-xs font-medium text-muted-foreground">
                {metric.label}
              </p>
              <p className="mt-1 text-xl font-bold font-mono text-primary">
                {metric.value}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
