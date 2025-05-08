// src/components/ApiKeySetup.js
import React, { useState } from 'react';
import apiService from '../utils/apiService'; // Uses the refactored service
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import GoogleIcon from '@mui/icons-material/Google'; // Using Google icon
import KeyIcon from '@mui/icons-material/Key';
import Link from '@mui/material/Link'; // For linking to Google Cloud console

function ApiKeySetup({ onApiKeySet }) {
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');
  const [isValidating, setIsValidating] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    // Basic validation for non-empty key
    if (!apiKey.trim()) {
      setError('Please enter your Google Cloud API key.');
      return;
    }

    setIsValidating(true);
    setError('');

    try {
      // Call the refactored apiService method which hits the backend
      await apiService.validateApiKey(apiKey);

      // If validation succeeds (no error thrown), the key is saved in apiService/localStorage.
      // Notify the App component to proceed.
      onApiKeySet();

    } catch (validationError) {
      // Display the error message from the backend or a generic one
      setError(validationError.message || 'Failed to validate API key. Please check your key, permissions, and backend connection.');
      // Ensure key is removed from local storage if validation fails
      localStorage.removeItem('google_api_key');
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 8 }}>
      <Paper
        sx={{
          p: 4,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          borderRadius: 2,
        }}
        elevation={3}
      >
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          mb: 3,
          color: 'primary.main' // Keep theme color for now
        }}>
          {/* Using Google Icon */}
          <GoogleIcon sx={{ fontSize: 40, mr: 1, color: '#4285F4' }} />
          <Typography variant="h4" component="h1">
            Gemini Agent Builder
          </Typography>
        </Box>

        <Typography variant="h6" gutterBottom>
          Welcome to Gemini Agent Builder
        </Typography>

        <Typography variant="body1" sx={{ mb: 2, textAlign: 'center' }}>
          To get started, please enter your Google Cloud API key.
          This key will be stored locally in your browser and used by the backend service to interact with Google AI services.
        </Typography>
        <Typography variant="body2" sx={{ mb: 4, textAlign: 'center' }}>
           Ensure your key has the necessary permissions (e.g., for Vertex AI). Find or create keys in the{' '}
           <Link href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noopener noreferrer">
             Google Cloud Console
           </Link>.
        </Typography>

        {error && (
          <Alert severity="error" sx={{ width: '100%', mb: 3 }}>
            <AlertTitle>Validation Error</AlertTitle>
            {error}
          </Alert>
        )}

        <Box component="form" onSubmit={handleSubmit} sx={{ width: '100%' }}>
          <TextField
            fullWidth
            label="Google Cloud API Key" // UPDATED Label
            variant="outlined"
            type="password" // Hide the key by default
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Enter your Google Cloud API Key" // UPDATED Placeholder
            InputProps={{
              startAdornment: <KeyIcon sx={{ mr: 1, color: 'text.secondary' }} />,
            }}
            sx={{ mb: 3 }}
            required // Make field required
          />

          <Button
            type="submit"
            fullWidth
            variant="contained"
            size="large"
            disabled={isValidating || !apiKey.trim()} // Disable if validating or empty
            sx={{ py: 1.5 }}
          >
            {isValidating ? 'Validating...' : 'Continue'}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
}

export default ApiKeySetup;