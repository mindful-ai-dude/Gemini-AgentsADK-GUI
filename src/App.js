// src/App.js
import React, { useState, useMemo, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Dashboard from './components/Dashboard';
import AgentBuilder from './components/AgentBuilder';
import AgentTester from './components/AgentTester';
import Navbar from './components/Navbar';
import ApiKeySetup from './components/ApiKeySetup'; // Will be modified later

// Main application component
function App() {
  // State to track if the Google API key is set and validated
  const [apiKeySet, setApiKeySet] = useState(() => {
    // Check if a key exists in local storage initially
    return !!localStorage.getItem('google_api_key'); // Use new key name
  });
  const [darkMode, setDarkMode] = useState(() => {
    const savedMode = localStorage.getItem('darkMode');
    return savedMode === 'true';
  });

  // Persist dark mode preference
  useEffect(() => {
    localStorage.setItem('darkMode', darkMode);
  }, [darkMode]);

  // Define the theme based on dark mode state
  const theme = useMemo(() =>
    createTheme({
      palette: {
        mode: darkMode ? 'dark' : 'light',
        primary: {
          // Google-like blue or keep the green? Let's keep green for now.
          main: '#10A37F', // Or use a Google blue like #4285F4
        },
        secondary: {
          main: '#0D8A6F', // Or use a Google secondary color
        },
        background: {
          default: darkMode ? '#121212' : '#F7F7F8',
          paper: darkMode ? '#1E1E1E' : '#FFFFFF',
        },
      },
      typography: {
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      },
      shape: {
        borderRadius: 8,
      },
    }), [darkMode]);

  // If API key is not set, render the setup component
  if (!apiKeySet) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {/* Pass the callback to update state once key is set */}
        <ApiKeySetup onApiKeySet={() => setApiKeySet(true)} />
      </ThemeProvider>
    );
  }

  // If API key is set, render the main application layout
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {/* Pass dark mode state and toggle function to Navbar */}
      <Navbar darkMode={darkMode} toggleDarkMode={() => setDarkMode(!darkMode)} />
      {/* Define application routes */}
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/build" element={<AgentBuilder />} />
        {/* Route for editing an existing agent */}
 spezifische Agenten-ID */}
        <Route path="/build/:agentId" element={<AgentBuilder />} />
        {/* Route for testing a specific agent */}
        <Route path="/test/:agentId" element={<AgentTester />} />
      </Routes>
    </ThemeProvider>
  );
}

export default App;