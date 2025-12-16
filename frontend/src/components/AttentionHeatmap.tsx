"use client";

import { useEffect, useRef, useMemo, useState, useCallback } from "react";

interface AttentionEntry {
  layer: number;
  head: number;
  attention: number[];
  type?: string;  // "layer_4", "layer_9", "decision_focus"
}

interface AttentionHeatmapProps {
  imageUrl: string;
  attentionData: AttentionEntry[];
  coordinates?: [number, number] | null;
  visionGrid?: [number, number] | null;
  imageSize?: [number, number] | null;
}

// Generate a unique color based on layer index using HSL color wheel
// Early layers = cool colors (cyan/blue), late layers = warm colors (orange/magenta)
function getLayerHSL(layerIdx: number, totalLayers: number): { h: number; s: number; l: number } {
  // Map layer 0 to hue 180 (cyan), last layer to hue 330 (magenta)
  const hue = 180 + (layerIdx / Math.max(totalLayers - 1, 1)) * 150;
  return { h: hue % 360, s: 100, l: 50 };
}

// Generate color function for a specific layer with dynamic total
function getLayerColor(layerIdx: number, totalLayers: number, value: number): string {
  const v = Math.max(0, Math.min(1, value));
  const alpha = v > 0.2 ? (v - 0.2) * 0.7 : 0;
  const { h, s, l } = getLayerHSL(layerIdx, totalLayers);
  return `hsla(${h}, ${s}%, ${l}%, ${alpha.toFixed(2)})`;
}

// Get solid color for toggle button
function getLayerButtonColor(layerIdx: number, totalLayers: number): string {
  const { h, s, l } = getLayerHSL(layerIdx, totalLayers);
  return `hsl(${h}, ${s}%, ${l}%)`;
}

// Decision focus color (red/yellow gradient)
function getDecisionFocusColor(value: number): string {
  const v = Math.max(0, Math.min(1, value));
  const alpha = v > 0.2 ? (v - 0.2) * 0.8 : 0;
  // Red to yellow gradient based on intensity
  const hue = 0 + v * 60;  // 0 (red) to 60 (yellow)
  return `hsla(${hue}, 100%, 50%, ${alpha.toFixed(2)})`;
}


export function AttentionHeatmap({
  imageUrl,
  attentionData,
  coordinates,
  visionGrid,
  imageSize,
}: AttentionHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Parse layer maps and decision focus from attention data
  const { layerMaps, decisionFocusMap } = useMemo(() => {
    const normalize = (values: number[]) => {
      const maxAttn = Math.max(...values);
      const minAttn = Math.min(...values);
      const range = maxAttn - minAttn || 1;
      return values.map((v) => (v - minAttn) / range);
    };

    const maps: Record<number, number[]> = {};
    let decisionMap: number[] | null = null;

    for (const entry of attentionData) {
      if (entry.type === "decision_focus" && entry.attention?.length) {
        decisionMap = normalize(entry.attention);
      } else {
        // Parse type like "layer_4", "layer_9", etc.
        const match = entry.type?.match(/^layer_(\d+)$/);
        if (match && entry.attention?.length) {
          const layerIdx = parseInt(match[1], 10);
          maps[layerIdx] = normalize(entry.attention);
        }
      }
    }
    return { layerMaps: maps, decisionFocusMap: decisionMap };
  }, [attentionData]);

  // Track which layers are enabled (all on by default)
  const [enabledLayers, setEnabledLayers] = useState<Set<number>>(new Set());
  const [showDecisionFocus, setShowDecisionFocus] = useState(true);
  const [isAnimating, setIsAnimating] = useState(false);
  const [windowSize, setWindowSize] = useState(3);
  const animationRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize enabled layers when data changes
  useEffect(() => {
    const layers = Object.keys(layerMaps).map(Number);
    setEnabledLayers(new Set(layers));
  }, [layerMaps]);

  const toggleLayer = (layer: number) => {
    setEnabledLayers((prev) => {
      const next = new Set(prev);
      if (next.has(layer)) {
        next.delete(layer);
      } else {
        next.add(layer);
      }
      return next;
    });
  };

  // Animation logic - sliding window through layers
  const sortedLayerList = useMemo(() =>
    Object.keys(layerMaps).map(Number).sort((a, b) => a - b),
    [layerMaps]
  );

  const startAnimation = useCallback(() => {
    if (sortedLayerList.length === 0) return;

    setIsAnimating(true);
    let currentStart = 0;

    const animate = () => {
      const windowLayers = sortedLayerList.slice(currentStart, currentStart + windowSize);
      setEnabledLayers(new Set(windowLayers));

      currentStart++;
      if (currentStart + windowSize > sortedLayerList.length) {
        currentStart = 0;
      }

      animationRef.current = setTimeout(animate, 400);
    };

    // Start with all off, then begin
    setEnabledLayers(new Set());
    animationRef.current = setTimeout(animate, 200);
  }, [sortedLayerList, windowSize]);

  const stopAnimation = useCallback(() => {
    setIsAnimating(false);
    if (animationRef.current) {
      clearTimeout(animationRef.current);
      animationRef.current = null;
    }
    // Show all layers when stopping
    setEnabledLayers(new Set(sortedLayerList));
  }, [sortedLayerList]);

  // Cleanup animation on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        clearTimeout(animationRef.current);
      }
    };
  }, []);

  // Draw heatmap overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const layerCount = Object.keys(layerMaps).length;
    if (!canvas || !container || layerCount === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const containerWidth = container.clientWidth;

      const refWidth = imageSize ? imageSize[0] : img.width;
      const refHeight = imageSize ? imageSize[1] : img.height;
      const canvasHeight = containerWidth * (refHeight / refWidth);

      canvas.width = containerWidth;
      canvas.height = canvasHeight;

      // Draw original image
      ctx.drawImage(img, 0, 0, containerWidth, canvasHeight);

      // Helper to draw a heatmap overlay
      const drawHeatmap = (attnMap: number[], layerIdx: number, total: number) => {
        const numTokens = attnMap.length;
        let gridHeight: number;
        let gridWidth: number;

        if (visionGrid && visionGrid[0] * visionGrid[1] === numTokens) {
          gridHeight = visionGrid[0];
          gridWidth = visionGrid[1];
        } else {
          gridWidth = Math.ceil(Math.sqrt(numTokens));
          gridHeight = Math.ceil(numTokens / gridWidth);
          while (gridWidth * gridHeight < numTokens) {
            gridHeight++;
          }
        }

        const cellWidth = containerWidth / gridWidth;
        const cellHeight = canvasHeight / gridHeight;

        for (let i = 0; i < numTokens; i++) {
          const row = Math.floor(i / gridWidth);
          const col = i % gridWidth;
          const x = col * cellWidth;
          const y = row * cellHeight;
          ctx.fillStyle = getLayerColor(layerIdx, total, attnMap[i]);
          ctx.fillRect(x, y, cellWidth, cellHeight);
        }
      };

      // Draw enabled vision layers in order (early to late)
      const layersList = Object.keys(layerMaps).map(Number).sort((a, b) => a - b);
      const totalLayers = layersList.length;
      for (const layerIdx of layersList) {
        if (enabledLayers.has(layerIdx)) {
          drawHeatmap(layerMaps[layerIdx], layerIdx, totalLayers);
        }
      }

      // Draw decision focus overlay (red/yellow) - LM attention uses 2x2 pooled grid
      if (showDecisionFocus && decisionFocusMap) {
        const numTokens = decisionFocusMap.length;
        let gridHeight: number;
        let gridWidth: number;

        // LM sees 2x2 pooled vision tokens, so grid is half the size
        if (visionGrid) {
          const pooledH = Math.ceil(visionGrid[0] / 2);
          const pooledW = Math.ceil(visionGrid[1] / 2);
          if (Math.abs(pooledH * pooledW - numTokens) <= 2) {
            gridHeight = pooledH;
            gridWidth = pooledW;
          } else {
            // Fallback to square grid
            gridWidth = Math.ceil(Math.sqrt(numTokens));
            gridHeight = Math.ceil(numTokens / gridWidth);
          }
        } else {
          gridWidth = Math.ceil(Math.sqrt(numTokens));
          gridHeight = Math.ceil(numTokens / gridWidth);
        }

        const cellWidth = containerWidth / gridWidth;
        const cellHeight = canvasHeight / gridHeight;

        for (let i = 0; i < numTokens; i++) {
          const row = Math.floor(i / gridWidth);
          const col = i % gridWidth;
          const x = col * cellWidth;
          const y = row * cellHeight;
          ctx.fillStyle = getDecisionFocusColor(decisionFocusMap[i]);
          ctx.fillRect(x, y, cellWidth, cellHeight);
        }
      }

      // Draw click location crosshair (dark blue)
      if (coordinates) {
        const [clickX, clickY] = coordinates;
        const pixelX = (clickX / 1000) * canvas.width;
        const pixelY = (clickY / 1000) * canvas.height;

        // Crosshair - dark blue
        ctx.strokeStyle = "#1e3a5f";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(pixelX - 20, pixelY);
        ctx.lineTo(pixelX + 20, pixelY);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pixelX, pixelY - 20);
        ctx.lineTo(pixelX, pixelY + 20);
        ctx.stroke();

        // Circle
        ctx.beginPath();
        ctx.arc(pixelX, pixelY, 15, 0, 2 * Math.PI);
        ctx.stroke();

        // White outline
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(pixelX, pixelY, 16, 0, 2 * Math.PI);
        ctx.stroke();

        // Label
        const label = `[${clickX}, ${clickY}]`;
        ctx.font = "bold 12px monospace";
        const textWidth = ctx.measureText(label).width;
        let labelX = pixelX + 22;
        if (labelX + textWidth + 4 > containerWidth) {
          labelX = pixelX - textWidth - 26;
        }
        ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
        ctx.fillRect(labelX - 2, pixelY - 8, textWidth + 4, 16);
        ctx.fillStyle = "#1e3a5f";
        ctx.textAlign = "left";
        ctx.fillText(label, labelX, pixelY + 4);
      }
    };

    img.src = imageUrl;
  }, [imageUrl, layerMaps, enabledLayers, coordinates, visionGrid, imageSize, showDecisionFocus, decisionFocusMap]);

  const sortedLayers = Object.keys(layerMaps).map(Number).sort((a, b) => a - b);
  const numTokens = sortedLayers.length > 0 ? layerMaps[sortedLayers[0]]?.length || 0 : 0;

  if (sortedLayers.length === 0) {
    return (
      <div className="text-zinc-500 text-center py-8">
        No attention data available
        <div className="text-xs mt-2">
          Available: {attentionData.length} entries
        </div>
      </div>
    );
  }

  const totalLayers = sortedLayers.length;

  return (
    <div className="space-y-2">
      {/* Controls row */}
      <div className="flex flex-wrap gap-2 p-2 bg-zinc-800 rounded-lg items-center">
        {/* Vision layer toggles */}
        <span className="text-xs text-zinc-400">Vision:</span>
        {sortedLayers.map((layerIdx) => {
          const isEnabled = enabledLayers.has(layerIdx);
          const btnColor = getLayerButtonColor(layerIdx, totalLayers);
          return (
            <button
              key={layerIdx}
              onClick={() => !isAnimating && toggleLayer(layerIdx)}
              disabled={isAnimating}
              className={`w-7 h-6 rounded text-xs font-mono transition-all ${
                isEnabled ? "ring-2 ring-white" : "opacity-30"
              } ${isAnimating ? "cursor-not-allowed" : ""}`}
              style={{
                backgroundColor: isEnabled ? btnColor : "transparent",
                border: `1px solid ${btnColor}`,
                color: isEnabled ? "black" : btnColor,
              }}
            >
              {layerIdx}
            </button>
          );
        })}

        <div className="ml-1 flex gap-1 border-l border-zinc-600 pl-2">
          <button
            onClick={() => setEnabledLayers(new Set(sortedLayers))}
            disabled={isAnimating}
            className="px-2 py-1 rounded text-xs bg-zinc-600 hover:bg-zinc-500 disabled:opacity-50"
          >
            All
          </button>
          <button
            onClick={() => setEnabledLayers(new Set())}
            disabled={isAnimating}
            className="px-2 py-1 rounded text-xs bg-zinc-600 hover:bg-zinc-500 disabled:opacity-50"
          >
            None
          </button>
        </div>

        {/* Animation controls */}
        <div className="ml-1 flex gap-1 items-center border-l border-zinc-600 pl-2">
          <button
            onClick={isAnimating ? stopAnimation : startAnimation}
            className={`px-2 py-1 rounded text-xs font-medium ${
              isAnimating ? "bg-red-600 hover:bg-red-500" : "bg-green-600 hover:bg-green-500"
            }`}
          >
            {isAnimating ? "Stop" : "Animate"}
          </button>
          <input
            type="number"
            min={1}
            max={10}
            value={windowSize}
            onChange={(e) => setWindowSize(Math.max(1, Math.min(10, parseInt(e.target.value) || 3)))}
            className="w-10 px-1 py-1 rounded text-xs bg-zinc-700 text-center"
            title="Window size"
          />
        </div>

        {/* Decision focus toggle */}
        {decisionFocusMap && (
          <div className="ml-1 flex gap-1 items-center border-l border-zinc-600 pl-2">
            <button
              onClick={() => setShowDecisionFocus(!showDecisionFocus)}
              className={`px-2 py-1 rounded text-xs font-medium ${
                showDecisionFocus
                  ? "bg-orange-600 hover:bg-orange-500"
                  : "bg-zinc-600 hover:bg-zinc-500"
              }`}
            >
              Decision
            </button>
          </div>
        )}

      </div>

      {/* Canvas */}
      <div ref={containerRef} className="relative">
        <canvas ref={canvasRef} className="w-full rounded-lg" />
        <div className="absolute bottom-2 left-2 bg-black/70 px-2 py-1 rounded text-xs">
          {enabledLayers.size} of {sortedLayers.length} layers
        </div>
        <div className="absolute bottom-2 right-2 bg-black/70 px-2 py-1 rounded text-xs">
          {numTokens} vision tokens
        </div>
      </div>
    </div>
  );
}
