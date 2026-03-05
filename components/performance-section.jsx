"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart3,
  Grid3X3,
  TrendingUp,
  Target,
  BrainCircuit,
  Layers,
} from "lucide-react";
import ModelComparisonChart from "./charts/model-comparison";
import RocCurveChart from "./charts/roc-curve";
import PrecisionRecallChart from "./charts/precision-recall";
import ConfusionMatrixChart from "./charts/confusion-matrix";
import ShapSummary from "./charts/shap-summary";
import CorrelationHeatmap from "./charts/correlation-heatmap";

const tabs = [
  { id: "comparison", label: "Model Comparison", icon: BarChart3 },
  { id: "roc", label: "ROC Curve", icon: TrendingUp },
  { id: "pr", label: "Precision-Recall", icon: Target },
  { id: "confusion", label: "Confusion Matrix", icon: Grid3X3 },
  { id: "shap", label: "SHAP Summary", icon: BrainCircuit },
  { id: "correlation", label: "Correlation", icon: Layers },
];

export default function PerformanceSection() {
  const [activeTab, setActiveTab] = useState("comparison");

  const renderChart = () => {
    switch (activeTab) {
      case "comparison":
        return <ModelComparisonChart />;
      case "roc":
        return <RocCurveChart />;
      case "pr":
        return <PrecisionRecallChart />;
      case "confusion":
        return <ConfusionMatrixChart />;
      case "shap":
        return <ShapSummary />;
      case "correlation":
        return <CorrelationHeatmap />;
      default:
        return <ModelComparisonChart />;
    }
  };

  return (
    <section id="performance" className="py-20 lg:py-28 bg-muted/30">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Section header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="inline-flex items-center gap-2 rounded-full bg-secondary px-4 py-1.5 text-sm font-medium text-secondary-foreground mb-4">
            <BarChart3 className="h-3.5 w-3.5" />
            Model Evaluation
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground text-balance">
            Performance Dashboard
          </h2>
          <p className="mt-3 mx-auto max-w-2xl text-muted-foreground leading-relaxed">
            Comprehensive evaluation metrics for the hybrid CardioML fusion
            model trained on the Cleveland Heart Disease dataset.
          </p>
        </motion.div>

        {/* Tab navigation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="flex flex-wrap justify-center gap-2 mb-8"
        >
          {tabs.map((tab) => (
            <button
              type="button"
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
                activeTab === tab.id
                  ? "bg-primary text-primary-foreground shadow-md shadow-primary/20"
                  : "bg-card text-muted-foreground border border-border hover:text-foreground hover:bg-muted"
              }`}
            >
              <tab.icon className="h-4 w-4" />
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          ))}
        </motion.div>

        {/* Chart area */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3, duration: 0.6 }}
          className="rounded-2xl border border-border bg-card p-4 sm:p-6 shadow-sm"
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              {renderChart()}
            </motion.div>
          </AnimatePresence>
        </motion.div>
      </div>
    </section>
  );
}
