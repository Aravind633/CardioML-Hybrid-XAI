"use client";

import { motion } from "framer-motion";
import {
  HeartPulse,
  Activity,
  BrainCircuit,
  ShieldCheck,
  ArrowDown,
} from "lucide-react";

const stats = [
  { label: "Accuracy", value: "96.9%", desc: "Cleveland Dataset" },
  { label: "ROC-AUC", value: "0.990", desc: "Fusion Model" },
  { label: "Features", value: "13+", desc: "Clinical + ECG" },
  { label: "XAI Method", value: "SHAP", desc: "Explainability" },
];

const features = [
  {
    icon: HeartPulse,
    title: "ECG Signal Analysis",
    desc: "Deep learning LSTM model processes 187-point ECG heartbeat signals for arrhythmia-based risk scoring.",
  },
  {
    icon: BrainCircuit,
    title: "Hybrid ML Fusion",
    desc: "Stacked ensemble of Random Forest + XGBoost with logistic meta-learner combines ECG and clinical data.",
  },
  {
    icon: ShieldCheck,
    title: "Explainable AI (XAI)",
    desc: "SHAP-based feature attribution provides transparent, clinician-friendly explanations for every prediction.",
  },
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.15 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
};

export default function Hero() {
  return (
    <section
      id="hero"
      className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden pt-16"
    >
      {/* Animated background */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,var(--secondary)_0%,transparent_60%)]" />
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3],
          }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          className="absolute top-20 left-1/4 h-96 w-96 rounded-full bg-primary/10 blur-3xl"
        />
        <motion.div
          animate={{
            scale: [1.2, 1, 1.2],
            opacity: [0.2, 0.4, 0.2],
          }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
          className="absolute bottom-20 right-1/4 h-80 w-80 rounded-full bg-accent/10 blur-3xl"
        />
      </div>

      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 text-center"
      >
        {/* Badge */}
        <motion.div variants={itemVariants} className="mb-6">
          <span className="inline-flex items-center gap-2 rounded-full bg-secondary px-4 py-1.5 text-sm font-medium text-secondary-foreground">
            <Activity className="h-3.5 w-3.5" />
            Hybrid ECG + Clinical ML Pipeline
          </span>
        </motion.div>

        {/* Title */}
        <motion.h1
          variants={itemVariants}
          className="text-4xl sm:text-5xl lg:text-7xl font-bold tracking-tight text-foreground text-balance leading-tight"
        >
          Explainable Heart Disease
          <br />
          <span className="text-primary">Risk Prediction</span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          variants={itemVariants}
          className="mt-6 mx-auto max-w-2xl text-lg text-muted-foreground leading-relaxed text-pretty"
        >
          Combining ECG deep learning with clinical machine learning for
          transparent, SHAP-powered cardiac risk assessment. Built on the
          Cleveland Heart Disease dataset with a stacked ensemble approach.
        </motion.p>

        {/* CTA */}
        <motion.div
          variants={itemVariants}
          className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4"
        >
          <a
            href="#predict"
            className="inline-flex items-center gap-2 rounded-xl bg-primary px-6 py-3 text-base font-semibold text-primary-foreground transition-all hover:opacity-90 hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-primary/25"
          >
            <HeartPulse className="h-5 w-5" />
            Run Risk Assessment
          </a>
          <a
            href="#performance"
            className="inline-flex items-center gap-2 rounded-xl bg-card px-6 py-3 text-base font-semibold text-foreground border border-border transition-all hover:bg-muted hover:scale-[1.02] active:scale-[0.98]"
          >
            <Activity className="h-5 w-5" />
            View Model Performance
          </a>
        </motion.div>

        {/* Stats */}
        <motion.div
          variants={itemVariants}
          className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6"
        >
          {stats.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.8 + i * 0.1, duration: 0.5 }}
              className="rounded-xl bg-card/80 backdrop-blur border border-border p-4 sm:p-5"
            >
              <div className="text-2xl sm:text-3xl font-bold text-primary font-mono">
                {stat.value}
              </div>
              <div className="mt-1 text-sm font-medium text-foreground">
                {stat.label}
              </div>
              <div className="text-xs text-muted-foreground">{stat.desc}</div>
            </motion.div>
          ))}
        </motion.div>

        {/* Feature cards */}
        <motion.div
          variants={itemVariants}
          className="mt-16 grid md:grid-cols-3 gap-6"
        >
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.15, duration: 0.6 }}
              className="group rounded-xl bg-card border border-border p-6 text-left transition-all hover:border-primary/40 hover:shadow-lg hover:shadow-primary/5"
            >
              <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-secondary text-primary transition-transform group-hover:scale-110">
                <feature.icon className="h-5 w-5" />
              </div>
              <h3 className="mt-4 text-lg font-semibold text-foreground">
                {feature.title}
              </h3>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
                {feature.desc}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
        className="absolute bottom-8"
      >
        <motion.a
          href="#predict"
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="flex flex-col items-center gap-2 text-muted-foreground hover:text-primary transition-colors"
        >
          <span className="text-xs font-medium">Scroll to predict</span>
          <ArrowDown className="h-4 w-4" />
        </motion.a>
      </motion.div>
    </section>
  );
}
