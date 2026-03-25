import { useEffect, useCallback } from "react";
import { WindowSize } from "../types";

interface UseKeyboardControlsProps {
  isRunning: boolean;
  isComplete: boolean;
  windowSize: WindowSize;
  availableTimeKeys: string[];
  loadedTimeKeys: Set<string>;
  onAddTimeKeyLayer: (timeKey: string) => void;
  onRemoveTimeKeyLayer: () => void;
  onSetWindowSize: (size: WindowSize) => void;
  onConfirmSegment: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}

export function useKeyboardControls({
  isRunning,
  isComplete,
  windowSize,
  availableTimeKeys,
  loadedTimeKeys,
  onAddTimeKeyLayer,
  onRemoveTimeKeyLayer,
  onSetWindowSize,
  onConfirmSegment,
  onZoomIn,
  onZoomOut,
}: UseKeyboardControlsProps) {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!isRunning || isComplete) return;

      switch (event.key) {
        case "ArrowRight": {
          // Add next time key layer (for window_size=7)
          if (windowSize === 7) {
            const loadedArray = Array.from(loadedTimeKeys);
            const currentIndex = loadedArray.length - 1;
            if (currentIndex < availableTimeKeys.length - 1) {
              onAddTimeKeyLayer(availableTimeKeys[currentIndex + 1]);
            }
          }
          break;
        }

        case "ArrowLeft": {
          // Remove most recent time key layer
          if (windowSize === 7 && loadedTimeKeys.size > 1) {
            onRemoveTimeKeyLayer();
          }
          break;
        }

        case "ArrowUp": {
          // Zoom in (centered on aircraft) - prevent page scroll
          event.preventDefault();
          onZoomIn();
          break;
        }

        case "ArrowDown": {
          // Zoom out - prevent page scroll
          event.preventDefault();
          onZoomOut();
          break;
        }

        case "Enter": {
          // Confirm current segment
          onConfirmSegment();
          break;
        }

        case "1": {
          onSetWindowSize(1);
          break;
        }

        case "7": {
          onSetWindowSize(7);
          break;
        }
      }
    },
    [isRunning, isComplete, windowSize, availableTimeKeys, loadedTimeKeys, onAddTimeKeyLayer, onRemoveTimeKeyLayer, onSetWindowSize, onConfirmSegment, onZoomIn, onZoomOut]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);
}
