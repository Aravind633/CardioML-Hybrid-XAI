"use client";

import { motion } from "framer-motion";
import {
  ShieldAlert,
  ShieldCheck,
  TrendingUp,
  TrendingDown,
  BrainCircuit,
  BarChart3,
} from "lucide-react";

export default function PredictionResult({ result }) {
  if (!result) return null;

  const { probability, shapValues } = result;
  const isHighRisk = probability >= 0.5;
  const pct = (probability * 100).toFixed(1);

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="flex flex-col gap-6"
    >
      {/* Risk Score Card */}
      <div
        className={`rounded-xl border-2 p-6 ${
          isHighRisk
            ? "border-accent/50 bg-accent/5"
            : "border-success/50 bg-success/5"
        }`}
      >
        <div className="flex items-center gap-3 mb-4">
          {isHighRisk ? (
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-accent/10 text-accent">
              <ShieldAlert className="h-6 w-6" />
            </div>
          ) : (
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-success/10 text-success">
              <ShieldCheck className="h-6 w-6" />
            </div>
          )}
          <div>
            <h3
              className={`text-lg font-bold ${
                isHighRisk ? "text-accent" : "text-success"
              }`}
            >
              {isHighRisk ? "High Risk Detected" : "Low Risk Detected"}
            </h3>
            <p className="text-sm text-muted-foreground">
              Fusion Model Prediction
            </p>
          </div>
        </div>

        {/* Risk gauge */}
        <div className="flex flex-col gap-2">
          <div className="flex items-end justify-between">
            <span className="text-sm font-medium text-muted-foreground">
              Risk Probability
            </span>
            <span
              className={`text-3xl font-bold font-mono ${
                isHighRisk ? "text-accent" : "text-success"
              }`}
            >
              {pct}%
            </span>
          </div>
          <div className="h-3 w-full rounded-full bg-muted overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${pct}%` }}
              transition={{ duration: 1.2, ease: "easeOut" }}
              className={`h-full rounded-full ${
                isHighRisk ? "bg-accent" : "bg-success"
              }`}
            />
          </div>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>0% - Low Risk</span>
            <span>50% Threshold</span>
            <span>100% - High Risk</span>
          </div>
        </div>
      </div>

      {/* SHAP Explanation */}
      <div className="rounded-xl border border-border bg-card p-6">
        <div className="flex items-center gap-3 mb-5">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-secondary text-primary">
            <BrainCircuit className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">
              Why this prediction?
            </h3>
            <p className="text-sm text-muted-foreground">
              SHAP-based Explainable AI
            </p>
          </div>
        </div>

        {/* Feature importance bars */}
        <div className="flex flex-col gap-3">
          {shapValues.map((item, i) => {
            const maxAbsImpact = Math.max(
              ...shapValues.map((s) => Math.abs(s.impact))
            );
            const barWidth = (Math.abs(item.impact) / maxAbsImpact) * 100;
            const isPositive = item.impact > 0;

            return (
              <motion.div
                key={item.feature}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1, duration: 0.4 }}
                className="flex flex-col gap-1"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">
                    {item.feature}
                  </span>
                  <div className="flex items-center gap-1">
                    {isPositive ? (
                      <TrendingUp className="h-3.5 w-3.5 text-accent" />
                    ) : (
                      <TrendingDown className="h-3.5 w-3.5 text-success" />
                    )}
                    <span
                      className={`text-xs font-mono font-medium ${
                        isPositive ? "text-accent" : "text-success"
                      }`}
                    >
                      {isPositive ? "+" : ""}
                      {item.impact.toFixed(3)}
                    </span>
                  </div>
                </div>
                <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${barWidth}%` }}
                    transition={{ delay: 0.3 + i * 0.1, duration: 0.6 }}
                    className={`h-full rounded-full ${
                      isPositive ? "bg-accent/70" : "bg-success/70"
                    }`}
                  />
                </div>
              </motion.div>
            );
          })}
        </div>

        {/* Clinical Interpretation */}
        <div className="mt-6 rounded-lg bg-muted/50 p-4">
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-foreground">
              Clinical Interpretation
            </span>
          </div>
          <div className="flex flex-col gap-2">
            {shapValues.slice(0, 3).map((item) => {
              const direction =
                item.impact > 0 ? "increases" : "reduces";
              return (
                <p
                  key={item.feature}
                  className="text-sm text-muted-foreground leading-relaxed"
                >
                  <span className="font-medium text-foreground">
                    {item.feature}
                  </span>{" "}
                  {direction} the predicted risk.
                </p>
              );
            })}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
