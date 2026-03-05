import Navigation from "@/components/navigation";
import Hero from "@/components/hero";
import PredictSection from "@/components/predict-section";
import PerformanceSection from "@/components/performance-section";
import MethodologySection from "@/components/methodology-section";
import AboutSection from "@/components/about-section";
import Footer from "@/components/footer";

export default function Home() {
  return (
    <main className="min-h-screen">
      <Navigation />
      <Hero />
      <PredictSection />
      <PerformanceSection />
      <MethodologySection />
      <AboutSection />
      <Footer />
    </main>
  );
}
