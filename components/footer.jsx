"use client";

import { HeartPulse, Github, ExternalLink } from "lucide-react";

const footerLinks = [
  { label: "Home", href: "#hero" },
  { label: "Predict", href: "#predict" },
  { label: "Performance", href: "#performance" },
  { label: "Methodology", href: "#methodology" },
  { label: "About", href: "#about" },
];

export default function Footer() {
  return (
    <footer className="border-t border-border bg-card">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-12">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-8">
          {/* Brand */}
          <div className="flex flex-col gap-3">
            <a href="#hero" className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <HeartPulse className="h-4 w-4" />
              </div>
              <span className="text-lg font-bold text-foreground">
                CardioML
              </span>
            </a>
            <p className="text-sm text-muted-foreground max-w-xs">
              Explainable ECG-Aware Heart Disease Prediction using Hybrid Machine Learning and SHAP-based XAI.
            </p>
          </div>

          {/* Links */}
          <div className="flex flex-wrap gap-4">
            {footerLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                {link.label}
              </a>
            ))}
          </div>

          {/* External */}
          <div className="flex items-center gap-3">
            <a
              href="https://github.com/Aravind633/CardioML-Hybrid-XAI"
              target="_blank"
              rel="noopener noreferrer"
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-border text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              aria-label="GitHub Repository"
            >
              <Github className="h-4 w-4" />
            </a>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-muted-foreground">
            CardioML - Academic Research Project. Not intended for medical
            diagnosis.
          </p>
          <p className="text-xs text-muted-foreground">
            Built with Next.js, Tailwind CSS, and Recharts
          </p>
        </div>
      </div>
    </footer>
  );
}
