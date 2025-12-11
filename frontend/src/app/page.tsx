"use client";

import { useState, useCallback } from "react";
import { AttentionHeatmap } from "@/components/AttentionHeatmap";
import { ImageUpload } from "@/components/ImageUpload";
import { AdapterSelector } from "@/components/AdapterSelector";

interface AttentionEntry {
  layer: number;
  head: number;
  attention: number[];
}

interface AnalysisResult {
  output_text: string;
  attention_data: AttentionEntry[];
  num_layers: number;
  num_heads: number;
  tokens: string[];
  vision_token_range: (number | null)[];
  image_size: number[];
  adapter_name: string | null;
  coordinates: [number, number] | null;
  generated_tokens: string[];
  coordinate_token_indices: number[];
  vision_grid: [number, number] | null;
}

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [query, setQuery] = useState("What action should I take?");
  const [adapter, setAdapter] = useState<string>("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelect = useCallback((file: File) => {
    setImage(file);
    setImageUrl(URL.createObjectURL(file));
    setResult(null);
  }, []);

  const handleAnalyze = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", image);
      formData.append("query", query);
      if (adapter) {
        formData.append("adapter_name", adapter);
      }

      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.detail || response.statusText;
        throw new Error(`Analysis failed: ${errorMsg}`);
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800 px-6 py-4">
        <h1 className="text-2xl font-bold">LoRA Attention Visualizer</h1>
        <p className="text-sm text-zinc-400">
          See what the model attends to when generating click coordinates
        </p>
      </header>

      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel - Controls */}
          <div className="space-y-6">
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6">
              <h2 className="text-lg font-semibold mb-4">Input</h2>

              <ImageUpload onImageSelect={handleImageSelect} />

              {imageUrl && (
                <div className="mt-4">
                  <img
                    src={imageUrl}
                    alt="Selected"
                    className="w-full rounded-lg border border-zinc-700"
                  />
                </div>
              )}

              <div className="mt-4">
                <label className="block text-sm font-medium text-zinc-400 mb-2">
                  Query
                </label>
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="w-full rounded-md border border-zinc-700 bg-zinc-800 px-4 py-2 text-zinc-100 focus:border-blue-500 focus:outline-none"
                  placeholder="What action should I take?"
                />
              </div>

              <div className="mt-4">
                <label className="block text-sm font-medium text-zinc-400 mb-2">
                  LoRA Adapter
                </label>
                <AdapterSelector value={adapter} onChange={setAdapter} />
              </div>

              <button
                onClick={handleAnalyze}
                disabled={!image || loading}
                className="mt-6 w-full rounded-md bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? "Analyzing..." : "Analyze"}
              </button>

              {error && (
                <div className="mt-4 rounded-md bg-red-900/50 border border-red-700 px-4 py-2 text-red-200">
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Visualization */}
          <div className="space-y-6">
            {result && (
              <>
                <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6">
                  <h2 className="text-lg font-semibold mb-2">
                    Coordinate Attention
                  </h2>
                  <p className="text-sm text-zinc-400 mb-4">
                    Bright areas = what the model looked at when outputting the click location
                  </p>
                  <AttentionHeatmap
                    imageUrl={imageUrl!}
                    attentionData={result.attention_data}
                    selectedLayer={0}
                    selectedHead={-1}
                    coordinates={result.coordinates}
                    visionGrid={result.vision_grid}
                    imageSize={result.image_size as [number, number] | null}
                  />
                </div>

                <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6">
                  <h2 className="text-lg font-semibold mb-4">Model Output</h2>
                  <div className="rounded-md bg-zinc-800 px-4 py-3 font-mono text-sm whitespace-pre-wrap">
                    {result.output_text}
                  </div>
                  <div className="mt-4 text-sm text-zinc-400 space-y-1">
                    <p><span className="text-zinc-500">Adapter:</span> {result.adapter_name || "Base Model"}</p>
                    <p><span className="text-zinc-500">Click target:</span> {result.coordinates ? `[${result.coordinates.join(", ")}]` : "No coordinates found"}</p>
                    <p><span className="text-zinc-500">Image size:</span> {result.image_size ? `${result.image_size[0]}x${result.image_size[1]}` : "Unknown"}</p>
                    <p><span className="text-zinc-500">Vision grid:</span> {result.vision_grid ? `${result.vision_grid[0]}x${result.vision_grid[1]}` : "None"}</p>
                  </div>
                </div>
              </>
            )}

            {!result && !loading && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6 text-center text-zinc-500">
                Upload an image and click Analyze to see what the model attends to when generating coordinates
              </div>
            )}

            {loading && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6 text-center">
                <div className="animate-pulse text-zinc-400">
                  Running inference with attention capture...
                  <p className="text-xs mt-2 text-zinc-500">This takes 30-60 seconds (custom generation loop)</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
