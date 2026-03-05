"use client";

import { motion } from "framer-motion";
import {
  FlaskConical,
  Database,
  Cpu,
  Layers,
  BrainCircuit,
  HeartPulse,
  ArrowRight,
  CheckCircle2,
} from "lucide-react";

const pipeline = [
  {
    step: 1,
    title: "Data Collection",
    icon: Database,
    desc: "Cleveland Heart Disease dataset (297 samples, 13 clinical features) combined with MIT-BIH ECG arrhythmia dataset.",
    details: [
      "13 clinical features: age, sex, chest pain, BP, cholesterol, etc.",
      "ECG signals: 187-point heartbeat waveforms",
      "Binary classification: Heart Disease vs. No Disease",
    ],
  },
  {
    step: 2,
    title: "ECG Deep Learning",
    icon: HeartPulse,
    desc: "CNN-LSTM neural network processes raw ECG signals to generate an arrhythmia-based risk score.",
    details: [
      "Conv1D feature extraction (64 filters, kernel=6)",
      "Dual LSTM layers (64 + 32 units) with dropout",
      "5-class softmax output mapped to risk score",
    ],
  },
  {
    step: 3,
    title: "Feature Engineering",
    icon: Cpu,
    desc: "Clinical features undergo preprocessing with OneHotEncoding for categorical and passthrough for numerical variables.",
    details: [
      "7 categorical features: sex, cp, fbs, restecg, exang, slope, thal",
      "6 numerical features: age, trestbps, chol, thalach, oldpeak, ca",
      "ColumnTransformer-based pipeline integration",
    ],
  },
  {
    step: 4,
    title: "Stacked Ensemble",
    icon: Layers,
    desc: "Two-layer stacking: Random Forest (500 trees) and XGBoost (400 estimators) as base learners, Logistic Regression as meta-learner.",
    details: [
      "Random Forest: 500 estimators, max_depth=12",
      "XGBoost: 400 estimators, lr=0.05, max_depth=5",
      "5-fold cross-validated stacking with LR meta-learner",
    ],
  },
  {
    step: 5,
    title: "Explainable AI (XAI)",
    icon: BrainCircuit,
    desc: "SHAP TreeExplainer provides feature-level attribution scores for transparent, clinician-friendly explanations.",
    details: [
      "TreeExplainer on XGBoost component",
      "Per-patient feature attribution (SHAP values)",
      "Global importance via summary beeswarm plots",
    ],
  },
];

export default function MethodologySection() {
  return (
    <section id="methodology" className="py-20 lg:py-28">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <span className="inline-flex items-center gap-2 rounded-full bg-secondary px-4 py-1.5 text-sm font-medium text-secondary-foreground mb-4">
            <FlaskConical className="h-3.5 w-3.5" />
            Research Methodology
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground text-balance">
            How CardioML Works
          </h2>
          <p className="mt-3 mx-auto max-w-2xl text-muted-foreground leading-relaxed">
            A five-stage hybrid pipeline combining deep learning ECG analysis
            with clinical machine learning for explainable cardiac risk
            prediction.
          </p>
        </motion.div>

        {/* Pipeline steps */}
        <div className="flex flex-col gap-6">
          {pipeline.map((item, i) => (
            <motion.div
              key={item.step}
              initial={{ opacity: 0, x: i % 2 === 0 ? -40 : 40 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1, duration: 0.6 }}
              className="group rounded-2xl border border-border bg-card p-6 transition-all hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5"
            >
              <div className="flex flex-col lg:flex-row gap-6">
                {/* Step indicator */}
                <div className="flex items-start gap-4 lg:w-64 shrink-0">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary text-primary-foreground text-lg font-bold shrink-0 transition-transform group-hover:scale-110">
                    {item.step}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <item.icon className="h-5 w-5 text-primary" />
                      <h3 className="text-lg font-semibold text-foreground">
                        {item.title}
                      </h3>
                    </div>
                    <p className="mt-1 text-sm text-muted-foreground leading-relaxed">
                      {item.desc}
                    </p>
                  </div>
                </div>

                {/* Details */}
                <div className="flex-1 flex flex-col gap-2 lg:border-l lg:border-border lg:pl-6">
                  {item.details.map((detail, di) => (
                    <div
                      key={di}
                      className="flex items-start gap-2 text-sm text-muted-foreground"
                    >
                      <CheckCircle2 className="h-4 w-4 text-primary shrink-0 mt-0.5" />
                      <span>{detail}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Arrow connector */}
              {i < pipeline.length - 1 && (
                <div className="flex justify-center mt-4 lg:hidden">
                  <ArrowRight className="h-5 w-5 text-muted-foreground rotate-90" />
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
