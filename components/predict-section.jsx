"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Activity } from "lucide-react";
import EcgUpload from "./ecg-upload";
import ClinicalForm from "./clinical-form";
import PredictionResult from "./prediction-result";

export default function PredictSection() {
  const [ecgScore, setEcgScore] = useState(null);
  const [result, setResult] = useState(null);

  return (
    <section id="predict" className="py-20 lg:py-28">
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
            <Activity className="h-3.5 w-3.5" />
            Risk Assessment Tool
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground text-balance">
            Heart Disease Risk Prediction
          </h2>
          <p className="mt-3 mx-auto max-w-2xl text-muted-foreground leading-relaxed">
            Upload an ECG signal and enter clinical parameters to receive an
            AI-powered risk assessment with full SHAP explainability.
          </p>
        </motion.div>

        {/* Two column layout */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left: Inputs */}
          <div className="flex flex-col gap-8">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="rounded-2xl border border-border bg-card p-6 shadow-sm"
            >
              <EcgUpload onScoreChange={setEcgScore} />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2, duration: 0.6 }}
              className="rounded-2xl border border-border bg-card p-6 shadow-sm"
            >
              <ClinicalForm ecgScore={ecgScore} onPredict={setResult} />
            </motion.div>
          </div>

          {/* Right: Results */}
          <div className="lg:sticky lg:top-24 lg:self-start">
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
              className="rounded-2xl border border-border bg-card p-6 shadow-sm min-h-[400px]"
            >
              {result ? (
                <PredictionResult result={result} />
              ) : (
                <div className="flex flex-col items-center justify-center h-full py-16 text-center">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted mb-4">
                    <Activity className="h-7 w-7 text-muted-foreground" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    Awaiting Analysis
                  </h3>
                  <p className="text-sm text-muted-foreground max-w-xs">
                    Upload an ECG signal and fill in clinical data, then click
                    &quot;Predict Risk&quot; to see your results here.
                  </p>

                  {/* Animated pulse ring */}
                  <div className="relative mt-8 h-20 w-20">
                    <motion.div
                      animate={{ scale: [1, 1.5, 1], opacity: [0.4, 0, 0.4] }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeOut",
                      }}
                      className="absolute inset-0 rounded-full border-2 border-primary/30"
                    />
                    <motion.div
                      animate={{ scale: [1, 1.3, 1], opacity: [0.6, 0, 0.6] }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeOut",
                        delay: 0.5,
                      }}
                      className="absolute inset-0 rounded-full border-2 border-primary/20"
                    />
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="h-3 w-3 rounded-full bg-primary/40" />
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
}
