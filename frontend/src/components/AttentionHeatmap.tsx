"use client";

import { useEffect, useRef, useMemo } from "react";

interface AttentionData {
  layer: number;
  attention: number[][][]; // [heads, seq_len, seq_len]
  shape: number[];
}

interface AttentionHeatmapProps {
  imageUrl: string;
  attentionData: AttentionData[];
  selectedLayer: number;
  selectedHead: number;
  visionTokenRange: (number | null)[];
  imageSize: number[]; // [width, height]
}

// Viridis-like colormap
function getColor(value: number): string {
  // Clamp value between 0 and 1
  const v = Math.max(0, Math.min(1, value));

  // Simple viridis approximation
  const r = Math.round(255 * (0.267 + 0.329 * v + 2.566 * v * v - 2.762 * v * v * v));
  const g = Math.round(255 * (0.004 + 1.416 * v - 0.766 * v * v));
  const b = Math.round(255 * (0.329 + 1.442 * v - 1.631 * v * v + 0.859 * v * v * v));

  return `rgba(${Math.min(255, r)}, ${Math.min(255, g)}, ${Math.min(255, b)}, 0.7)`;
}

export function AttentionHeatmap({
  imageUrl,
  attentionData,
  selectedLayer,
  selectedHead,
  visionTokenRange,
  imageSize,
}: AttentionHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Extract attention for selected layer/head
  const attentionMap = useMemo(() => {
    const layerData = attentionData.find((d) => d.layer === selectedLayer);
    if (!layerData) return null;

    // Get attention for selected head
    // Shape: [num_heads, seq_len, seq_len]
    const headAttn = layerData.attention[selectedHead];
    if (!headAttn) return null;

    // Extract vision-to-vision attention (image self-attention)
    // visionTokenRange = [start, end]
    const [vStart, vEnd] = visionTokenRange;
    if (vStart === null || vEnd === null) return null;

    // Get average attention from all tokens to vision tokens
    // This shows what parts of the image are most attended to
    const numVisionTokens = vEnd - vStart;
    const visionAttention: number[] = new Array(numVisionTokens).fill(0);

    // Sum attention to each vision token across all query positions
    for (let q = 0; q < headAttn.length; q++) {
      for (let v = vStart; v < vEnd; v++) {
        visionAttention[v - vStart] += headAttn[q][v];
      }
    }

    // Normalize
    const maxAttn = Math.max(...visionAttention);
    const minAttn = Math.min(...visionAttention);
    const range = maxAttn - minAttn || 1;

    return visionAttention.map((v) => (v - minAttn) / range);
  }, [attentionData, selectedLayer, selectedHead, visionTokenRange]);

  // Draw heatmap overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !attentionMap) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      // Set canvas size to match container
      const containerWidth = container.clientWidth;
      const scale = containerWidth / img.width;
      const canvasHeight = img.height * scale;

      canvas.width = containerWidth;
      canvas.height = canvasHeight;

      // Draw original image
      ctx.drawImage(img, 0, 0, containerWidth, canvasHeight);

      // Calculate grid dimensions for attention map
      // Qwen typically uses 14x14 or similar patch grid
      // We need to figure out the spatial arrangement
      const numTokens = attentionMap.length;

      // Try to find a reasonable grid size
      // Common sizes: 14x14=196, 16x16=256, etc.
      let gridSize = Math.ceil(Math.sqrt(numTokens));

      // Adjust if not square
      const gridWidth = gridSize;
      const gridHeight = Math.ceil(numTokens / gridWidth);

      const cellWidth = containerWidth / gridWidth;
      const cellHeight = canvasHeight / gridHeight;

      // Draw attention heatmap
      for (let i = 0; i < numTokens; i++) {
        const row = Math.floor(i / gridWidth);
        const col = i % gridWidth;

        const x = col * cellWidth;
        const y = row * cellHeight;

        ctx.fillStyle = getColor(attentionMap[i]);
        ctx.fillRect(x, y, cellWidth, cellHeight);
      }

      // Add colorbar legend
      const legendWidth = 20;
      const legendHeight = canvasHeight * 0.6;
      const legendX = containerWidth - legendWidth - 10;
      const legendY = (canvasHeight - legendHeight) / 2;

      // Draw gradient
      const gradient = ctx.createLinearGradient(0, legendY + legendHeight, 0, legendY);
      for (let i = 0; i <= 10; i++) {
        gradient.addColorStop(i / 10, getColor(i / 10));
      }
      ctx.fillStyle = gradient;
      ctx.fillRect(legendX, legendY, legendWidth, legendHeight);

      // Draw border
      ctx.strokeStyle = "white";
      ctx.lineWidth = 1;
      ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

      // Labels
      ctx.fillStyle = "white";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("High", legendX + legendWidth + 4, legendY + 10);
      ctx.fillText("Low", legendX + legendWidth + 4, legendY + legendHeight);
    };

    img.src = imageUrl;
  }, [imageUrl, attentionMap]);

  if (!attentionMap) {
    return (
      <div className="text-zinc-500 text-center py-8">
        No attention data available for this layer/head
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative">
      <canvas ref={canvasRef} className="w-full rounded-lg" />
      <div className="absolute bottom-2 left-2 bg-black/70 px-2 py-1 rounded text-xs">
        Layer {selectedLayer}, Head {selectedHead}
      </div>
    </div>
  );
}
