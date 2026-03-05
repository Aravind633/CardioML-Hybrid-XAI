"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Stethoscope,
  User,
  HeartPulse,
  Droplets,
  Gauge,
  Zap,
} from "lucide-react";

const FIELD_GROUPS = [
  {
    title: "Demographics",
    icon: User,
    fields: [
      {
        name: "age",
        label: "Age",
        type: "number",
        min: 20,
        max: 100,
        default: 50,
      },
      {
        name: "sex",
        label: "Sex",
        type: "select",
        options: [
          { label: "Male", value: "Male" },
          { label: "Female", value: "Female" },
        ],
        default: "Male",
      },
    ],
  },
  {
    title: "Cardiac Symptoms",
    icon: HeartPulse,
    fields: [
      {
        name: "cp",
        label: "Chest Pain Type",
        type: "select",
        options: [
          { label: "Typical Angina (0)", value: 0 },
          { label: "Atypical Angina (1)", value: 1 },
          { label: "Non-anginal (2)", value: 2 },
          { label: "Asymptomatic (3)", value: 3 },
        ],
        default: 0,
      },
      {
        name: "exang",
        label: "Exercise Angina",
        type: "select",
        options: [
          { label: "No (0)", value: 0 },
          { label: "Yes (1)", value: 1 },
        ],
        default: 0,
      },
    ],
  },
  {
    title: "Vital Signs",
    icon: Gauge,
    fields: [
      {
        name: "trestbps",
        label: "Resting Blood Pressure",
        type: "number",
        min: 80,
        max: 200,
        default: 120,
        unit: "mmHg",
      },
      {
        name: "thalach",
        label: "Max Heart Rate Achieved",
        type: "number",
        min: 60,
        max: 220,
        default: 150,
        unit: "bpm",
      },
    ],
  },
  {
    title: "Blood Work",
    icon: Droplets,
    fields: [
      {
        name: "chol",
        label: "Serum Cholesterol",
        type: "number",
        min: 100,
        max: 400,
        default: 200,
        unit: "mg/dl",
      },
      {
        name: "fbs",
        label: "Fasting Blood Sugar > 120",
        type: "select",
        options: [
          { label: "No (0)", value: 0 },
          { label: "Yes (1)", value: 1 },
        ],
        default: 0,
      },
    ],
  },
  {
    title: "ECG & Tests",
    icon: Zap,
    fields: [
      {
        name: "restecg",
        label: "Resting ECG Result",
        type: "select",
        options: [
          { label: "Normal (0)", value: 0 },
          { label: "ST-T Abnormality (1)", value: 1 },
          { label: "LV Hypertrophy (2)", value: 2 },
        ],
        default: 0,
      },
      {
        name: "oldpeak",
        label: "ST Depression (Oldpeak)",
        type: "number",
        min: 0,
        max: 6,
        step: 0.1,
        default: 1.0,
      },
      {
        name: "slope",
        label: "Slope of Peak ST",
        type: "select",
        options: [
          { label: "Upsloping (0)", value: 0 },
          { label: "Flat (1)", value: 1 },
          { label: "Downsloping (2)", value: 2 },
        ],
        default: 0,
      },
      {
        name: "ca",
        label: "Major Vessels (Fluoroscopy)",
        type: "select",
        options: [
          { label: "0", value: 0 },
          { label: "1", value: 1 },
          { label: "2", value: 2 },
          { label: "3", value: 3 },
        ],
        default: 0,
      },
      {
        name: "thal",
        label: "Thalassemia",
        type: "select",
        options: [
          { label: "Normal (3)", value: 3 },
          { label: "Fixed Defect (6)", value: 6 },
          { label: "Reversible Defect (7)", value: 7 },
        ],
        default: 3,
      },
    ],
  },
];

export default function ClinicalForm({ ecgScore, onPredict }) {
  const initialValues = {};
  FIELD_GROUPS.forEach((group) => {
    group.fields.forEach((field) => {
      initialValues[field.name] = field.default;
    });
  });

  const [values, setValues] = useState(initialValues);
  const [predicting, setPredicting] = useState(false);

  const handleChange = (name, value) => {
    setValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setPredicting(true);

    // Simulate prediction delay (in production, call Python API)
    setTimeout(() => {
      const input = {
        age: Number(values.age),
        sex: values.sex === "Male" ? 1.0 : 0.0,
        cp: Number(values.cp),
        trestbps: Number(values.trestbps),
        chol: Number(values.chol),
        fbs: Number(values.fbs),
        restecg: Number(values.restecg),
        thalach: Number(values.thalach),
        exang: Number(values.exang),
        oldpeak: Number(values.oldpeak),
        slope: Number(values.slope),
        ca: Number(values.ca),
        thal: Number(values.thal),
      };

      // Simulated risk probability based on clinical features
      let riskBase = 0.3;
      if (input.age > 55) riskBase += 0.1;
      if (input.sex === 1) riskBase += 0.05;
      if (input.cp >= 2) riskBase += 0.1;
      if (input.trestbps > 140) riskBase += 0.08;
      if (input.chol > 240) riskBase += 0.07;
      if (input.fbs === 1) riskBase += 0.03;
      if (input.thalach < 120) riskBase += 0.1;
      if (input.exang === 1) riskBase += 0.12;
      if (input.oldpeak > 2) riskBase += 0.1;
      if (input.slope === 2) riskBase += 0.05;
      if (input.ca >= 2) riskBase += 0.1;
      if (input.thal === 7) riskBase += 0.08;

      const prob = Math.min(0.98, Math.max(0.02, riskBase));

      // Simulated SHAP values
      const shapValues = [
        {
          feature: "ca (Major Vessels)",
          impact: input.ca >= 1 ? 0.35 : -0.12,
        },
        {
          feature: "cp (Chest Pain Type)",
          impact: input.cp >= 2 ? 0.28 : -0.15,
        },
        {
          feature: "age",
          impact: input.age > 55 ? 0.18 : -0.08,
        },
        {
          feature: "thalach (Max Heart Rate)",
          impact: input.thalach < 130 ? 0.22 : -0.18,
        },
        {
          feature: "oldpeak (ST Depression)",
          impact: input.oldpeak > 1.5 ? 0.2 : -0.06,
        },
        {
          feature: "thal (Thalassemia)",
          impact: input.thal === 7 ? 0.15 : -0.1,
        },
      ].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

      onPredict({
        probability: prob,
        shapValues: shapValues,
        input: input,
      });
      setPredicting(false);
    }, 2000);
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-secondary text-primary">
          <Stethoscope className="h-5 w-5" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">
            Clinical Information
          </h3>
          <p className="text-sm text-muted-foreground">
            Enter patient clinical parameters
          </p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-6">
        {FIELD_GROUPS.map((group, gi) => (
          <motion.div
            key={group.title}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: gi * 0.1, duration: 0.4 }}
            className="rounded-xl border border-border bg-card/50 p-4"
          >
            <div className="mb-3 flex items-center gap-2">
              <group.icon className="h-4 w-4 text-primary" />
              <span className="text-sm font-semibold text-foreground">
                {group.title}
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {group.fields.map((field) => (
                <div key={field.name} className="flex flex-col gap-1.5">
                  <label
                    htmlFor={field.name}
                    className="text-xs font-medium text-muted-foreground"
                  >
                    {field.label}
                    {field.unit && (
                      <span className="ml-1 text-muted-foreground/60">
                        ({field.unit})
                      </span>
                    )}
                  </label>
                  {field.type === "select" ? (
                    <select
                      id={field.name}
                      value={values[field.name]}
                      onChange={(e) =>
                        handleChange(
                          field.name,
                          isNaN(e.target.value)
                            ? e.target.value
                            : Number(e.target.value)
                        )
                      }
                      className="h-9 rounded-lg border border-border bg-background px-3 text-sm text-foreground outline-none transition-colors focus:border-primary focus:ring-1 focus:ring-primary"
                    >
                      {field.options.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={field.name}
                      type="number"
                      min={field.min}
                      max={field.max}
                      step={field.step || 1}
                      value={values[field.name]}
                      onChange={(e) =>
                        handleChange(field.name, Number(e.target.value))
                      }
                      className="h-9 rounded-lg border border-border bg-background px-3 text-sm text-foreground outline-none transition-colors focus:border-primary focus:ring-1 focus:ring-primary"
                    />
                  )}
                </div>
              ))}
            </div>
          </motion.div>
        ))}

        <button
          type="submit"
          disabled={ecgScore === null || predicting}
          className="flex items-center justify-center gap-2 rounded-xl bg-primary px-6 py-3 text-base font-semibold text-primary-foreground transition-all hover:opacity-90 hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
        >
          {predicting ? (
            <>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="h-5 w-5 rounded-full border-2 border-primary-foreground/30 border-t-primary-foreground"
              />
              Analyzing...
            </>
          ) : (
            <>
              <HeartPulse className="h-5 w-5" />
              Predict Risk
            </>
          )}
        </button>

        {ecgScore === null && (
          <p className="text-center text-xs text-muted-foreground">
            Please upload an ECG signal first to enable prediction.
          </p>
        )}
      </form>
    </div>
  );
}
