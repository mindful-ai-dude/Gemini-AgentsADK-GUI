// src/utils/modelOptions.js
// Updated May 2025 to match google/adk-python model identifiers

export const modelOptions = [
  { value: 'gemini-2.5-flash-preview-04-17', label: 'Gemini 2.5 Flash (Prev)' },
  { value: 'gemini‑2.5‑pro‑exp‑05‑06', label: 'Gemini 2.5 Pro (Prev)' },
  { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
  { value: 'gemini-2.0-flash-lite', label: 'Gemini 2.0 Flash Lite' }
  // add any newly released preview or production models here
];

// Default model settings - these might need adjustment based on Google model specifics
export const modelSettingsDefaults = {
  temperature: 0.7,
  topP: 0.95, // Common default for Gemini
  topK: 40,   // Common default for Gemini
  // frequencyPenalty: 0, // Less common in Gemini API, check ADK specifics
  // presencePenalty: 0,  // Less common in Gemini API, check ADK specifics
  maxOutputTokens: 8192 // Example, adjust based on model and ADK
};