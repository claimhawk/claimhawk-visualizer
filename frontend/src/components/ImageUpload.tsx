"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
}

export function ImageUpload({ onImageSelect }: ImageUploadProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onImageSelect(acceptedFiles[0]);
      }
    },
    [onImageSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg", ".webp"],
    },
    multiple: false,
  });

  return (
    <div
      {...getRootProps()}
      className={`
        border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
        transition-colors
        ${
          isDragActive
            ? "border-blue-500 bg-blue-500/10"
            : "border-zinc-700 hover:border-zinc-500"
        }
      `}
    >
      <input {...getInputProps()} />
      <div className="text-zinc-400">
        {isDragActive ? (
          <p>Drop the image here...</p>
        ) : (
          <>
            <p>Drag & drop an image here</p>
            <p className="text-sm mt-1">or click to select</p>
          </>
        )}
      </div>
    </div>
  );
}
