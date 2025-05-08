// src/utils/modelOptions.js

// Options based on common Google AI / Vertex AI models.
// Refine these based on specific models supported by the Google ADK for agent use.
export const modelOptions = [
  {
    value: "gemini-2.5-flash-preview-04-17", // Example model name
    label: "Gemini 2.5 Flash",
    description: "Fast and versatile multimodal model (recommended default)"
  },
  {
    value: "gemini-2.5-pro-preview-05-06", // Example model name
    label: "Gemini 2.5 Pro",
    description: "Worlds Best multimodal model for complex tasks"
  },
  {
    value: "gemini-2.0-flash-exp", // Image gen model
    label: "Gemini 2.0 Flash",
    description: "Balanced model for various tasks"
  }
  // Add other relevant Google models supported by ADK here
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