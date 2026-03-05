"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileCheck, AlertCircle, Activity, X } from "lucide-react";

export default function EcgUpload({ onScoreChange }) {
  const [file, setFile] = useState(null);
  const [ecgData, setEcgData] = useState(null);
  const [score, setScore] = useState(null);
  const [error, setError] = useState(null);
  const [processing, setProcessing] = useState(false);

  const processFile = useCallback(
    (uploadedFile) => {
      setError(null);
      setProcessing(true);

      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const text = e.target.result;
          const values = text
            .split(/[,\n\r]+/)
            .map((v) => v.trim())
            .filter((v) => v !== "")
            .map(Number);

          if (values.length !== 187) {
            setError(
              `ECG must have exactly 187 values. Found ${values.length}.`
            );
            setProcessing(false);
            return;
          }

          if (values.some(isNaN)) {
            setError("ECG file contains non-numeric values.");
            setProcessing(false);
            return;
          }

          setEcgData(values);

          // Simulate ECG risk score computation (frontend demo)
          // In production, this would call the Python backend
          setTimeout(() => {
            const maxVal = Math.max(...values);
            const variance =
              values.reduce((sum, v) => sum + Math.pow(v - maxVal / 2, 2), 0) /
              values.length;
            const simulatedScore = Math.min(
              0.95,
              Math.max(0.05, 0.3 + variance * 2)
            );
            const rounded = Math.round(simulatedScore * 1000) / 1000;
            setScore(rounded);
            onScoreChange(rounded);
            setProcessing(false);
          }, 1500);
        } catch {
          setError("Failed to parse ECG file.");
          setProcessing(false);
        }
      };
      reader.readAsText(uploadedFile);
    },
    [onScoreChange]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      const dropped = e.dataTransfer.files[0];
      if (dropped && dropped.name.endsWith(".csv")) {
        setFile(dropped);
        processFile(dropped);
      } else {
        setError("Please upload a .csv file.");
      }
    },
    [processFile]
  );

  const handleFileSelect = useCallback(
    (e) => {
      const selected = e.target.files[0];
      if (selected) {
        setFile(selected);
        processFile(selected);
      }
    },
    [processFile]
  );

  const reset = () => {
    setFile(null);
    setEcgData(null);
    setScore(null);
    setError(null);
    onScoreChange(null);
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-secondary text-primary">
          <Activity className="h-5 w-5" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-foreground">
            ECG Signal Input
          </h3>
          <p className="text-sm text-muted-foreground">
            Upload a single ECG heartbeat (.csv, 187 values)
          </p>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {!file ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="relative flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed border-border bg-muted/30 px-6 py-10 transition-colors hover:border-primary/50 hover:bg-secondary/30 cursor-pointer"
          >
            <input
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              className="absolute inset-0 opacity-0 cursor-pointer"
              aria-label="Upload ECG CSV file"
            />
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-secondary text-primary">
              <Upload className="h-5 w-5" />
            </div>
            <div className="text-center">
              <p className="text-sm font-medium text-foreground">
                Drop your ECG file here or click to browse
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                CSV format with 187 numeric values (MIT-BIH compatible)
              </p>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="result"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="rounded-xl border border-border bg-card p-4"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-secondary text-primary">
                  <FileCheck className="h-4 w-4" />
                </div>
                <div>
                  <p className="text-sm font-medium text-foreground">
                    {file.name}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={reset}
                className="flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground hover:bg-muted transition-colors"
                aria-label="Remove file"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {processing && (
              <div className="mt-4 flex items-center gap-3">
                <div className="h-1.5 flex-1 rounded-full bg-muted overflow-hidden">
                  <motion.div
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 1.5, ease: "easeInOut" }}
                    className="h-full rounded-full bg-primary"
                  />
                </div>
                <span className="text-xs text-muted-foreground">
                  Processing...
                </span>
              </div>
            )}

            {score !== null && !processing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-4 flex items-center gap-3 rounded-lg bg-secondary/50 p-3"
              >
                <Activity className="h-5 w-5 text-primary" />
                <div>
                  <p className="text-sm font-medium text-foreground">
                    ECG Signal Risk Score
                  </p>
                  <p className="text-xl font-bold font-mono text-primary">
                    {score.toFixed(3)}
                  </p>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: -5 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2 rounded-lg bg-accent/10 px-3 py-2 text-sm text-accent"
        >
          <AlertCircle className="h-4 w-4 shrink-0" />
          {error}
        </motion.div>
      )}

      {/* ECG Waveform Preview */}
      {ecgData && !processing && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          className="rounded-xl border border-border bg-card p-4"
        >
          <p className="mb-3 text-sm font-medium text-foreground">
            ECG Waveform Preview
          </p>
          <div className="h-24 w-full">
            <svg
              viewBox="0 0 374 100"
              className="w-full h-full"
              preserveAspectRatio="none"
            >
              <polyline
                fill="none"
                stroke="var(--primary)"
                strokeWidth="1.5"
                points={ecgData
                  .map(
                    (v, i) =>
                      `${(i / 186) * 374},${100 - v * 90 - 5}`
                  )
                  .join(" ")}
              />
            </svg>
          </div>
        </motion.div>
      )}
    </div>
  );
}
