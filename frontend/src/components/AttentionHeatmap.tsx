"use client";

import { useEffect, useRef, useMemo } from "react";

interface AttentionEntry {
  layer: number;
  head: number;
  attention: number[];
}

interface AttentionHeatmapProps {
  imageUrl: string;
  attentionData: AttentionEntry[];
  selectedLayer: number;
  selectedHead: number;
  coordinates?: [number, number] | null;
  visionGrid?: [number, number] | null;  // [height, width] from Qwen's image_grid_thw
  imageSize?: [number, number] | null;   // [width, height] of original image
}

// Viridis-like colormap
function getColor(value: number): string {
  const v = Math.max(0, Math.min(1, value));
  const r = Math.round(255 * (0.267 + 0.329 * v + 2.566 * v * v - 2.762 * v * v * v));
  const g = Math.round(255 * (0.004 + 1.416 * v - 0.766 * v * v));
  const b = Math.round(255 * (0.329 + 1.442 * v - 1.631 * v * v + 0.859 * v * v * v));
  return `rgba(${Math.min(255, Math.max(0, r))}, ${Math.min(255, Math.max(0, g))}, ${Math.min(255, Math.max(0, b))}, 0.6)`;
}

export function AttentionHeatmap({
  imageUrl,
  attentionData,
  selectedLayer,
  selectedHead,
  coordinates,
  visionGrid,
  imageSize,
}: AttentionHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Find attention data for selected layer/head
  const attentionMap = useMemo(() => {
    const entry = attentionData.find(
      (d) => d.layer === selectedLayer && d.head === selectedHead
    );
    if (!entry || !entry.attention || entry.attention.length === 0) {
      return null;
    }

    const values = entry.attention;
    const maxAttn = Math.max(...values);
    const minAttn = Math.min(...values);
    const range = maxAttn - minAttn || 1;

    return values.map((v) => (v - minAttn) / range);
  }, [attentionData, selectedLayer, selectedHead]);

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
      const containerWidth = container.clientWidth;

      // Use backend imageSize for aspect ratio if available (what model actually saw)
      // Fall back to loaded image dimensions
      const refWidth = imageSize ? imageSize[0] : img.width;
      const refHeight = imageSize ? imageSize[1] : img.height;

      // Canvas height based on reference aspect ratio
      const canvasHeight = containerWidth * (refHeight / refWidth);

      canvas.width = containerWidth;
      canvas.height = canvasHeight;

      // Draw original image stretched to canvas (may distort if aspect ratios differ)
      ctx.drawImage(img, 0, 0, containerWidth, canvasHeight);

      // Calculate grid dimensions for attention map
      // Use actual grid from Qwen's image_grid_thw if available
      const numTokens = attentionMap.length;
      let gridHeight: number;
      let gridWidth: number;

      if (visionGrid && visionGrid[0] * visionGrid[1] === numTokens) {
        // Use exact grid dimensions from model
        gridHeight = visionGrid[0];
        gridWidth = visionGrid[1];
      } else {
        // Fallback: try to infer square-ish grid
        gridWidth = Math.ceil(Math.sqrt(numTokens));
        gridHeight = Math.ceil(numTokens / gridWidth);
        while (gridWidth * gridHeight < numTokens) {
          gridHeight++;
        }
      }

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

      // Draw click location crosshair if coordinates exist
      if (coordinates) {
        const [clickX, clickY] = coordinates;
        // RU coordinates: 0-1000 maps to full dimension
        const pixelX = (clickX / 1000) * canvas.width;
        const pixelY = (clickY / 1000) * canvas.height;
        console.log("canvas dims:", canvas.width, canvas.height, "coords:", clickX, clickY, "pixel:", pixelX, pixelY);

        // Draw crosshair
        ctx.strokeStyle = "#ff0000";
        ctx.lineWidth = 2;

        // Horizontal line
        ctx.beginPath();
        ctx.moveTo(pixelX - 20, pixelY);
        ctx.lineTo(pixelX + 20, pixelY);
        ctx.stroke();

        // Vertical line
        ctx.beginPath();
        ctx.moveTo(pixelX, pixelY - 20);
        ctx.lineTo(pixelX, pixelY + 20);
        ctx.stroke();

        // Circle around target
        ctx.beginPath();
        ctx.arc(pixelX, pixelY, 15, 0, 2 * Math.PI);
        ctx.stroke();

        // White outline for visibility
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(pixelX, pixelY, 16, 0, 2 * Math.PI);
        ctx.stroke();

        // Label with background for readability
        const label = `[${clickX}, ${clickY}]`;
        ctx.font = "bold 12px monospace";
        const textWidth = ctx.measureText(label).width;

        // Position label - try right side, fall back to left if too close to edge
        let labelX = pixelX + 22;
        if (labelX + textWidth + 4 > containerWidth) {
          labelX = pixelX - textWidth - 26;
        }

        // Background
        ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
        ctx.fillRect(labelX - 2, pixelY - 8, textWidth + 4, 16);

        // Text
        ctx.fillStyle = "#ff0000";
        ctx.textAlign = "left";
        ctx.fillText(label, labelX, pixelY + 4);
      }

      // Add colorbar legend
      const legendWidth = 20;
      const legendHeight = canvasHeight * 0.5;
      const legendX = containerWidth - legendWidth - 10;
      const legendY = (canvasHeight - legendHeight) / 2;

      const gradient = ctx.createLinearGradient(0, legendY + legendHeight, 0, legendY);
      for (let i = 0; i <= 10; i++) {
        gradient.addColorStop(i / 10, getColor(i / 10));
      }
      ctx.fillStyle = gradient;
      ctx.fillRect(legendX, legendY, legendWidth, legendHeight);

      ctx.strokeStyle = "white";
      ctx.lineWidth = 1;
      ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

      ctx.fillStyle = "white";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("High", legendX + legendWidth + 4, legendY + 10);
      ctx.fillText("Low", legendX + legendWidth + 4, legendY + legendHeight);
    };

    img.src = imageUrl;
  }, [imageUrl, attentionMap, coordinates, visionGrid, imageSize]);

  if (!attentionMap) {
    return (
      <div className="text-zinc-500 text-center py-8">
        No attention data available for Layer {selectedLayer}
        <div className="text-xs mt-2">
          Available: {attentionData.length} entries
        </div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="relative">
      <canvas ref={canvasRef} className="w-full rounded-lg" />
      <div className="absolute bottom-2 left-2 bg-black/70 px-2 py-1 rounded text-xs">
        Layer {selectedLayer}{coordinates ? ` • Coord attention` : ""}
      </div>
      <div className="absolute bottom-2 right-2 bg-black/70 px-2 py-1 rounded text-xs">
        {attentionMap.length} vision tokens
      </div>
    </div>
  );
}
