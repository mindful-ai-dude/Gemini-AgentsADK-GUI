// src/components/AgentTester.js
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import apiService from '../utils/apiService'; // Uses refactored service
import TestAgent from './builder/TestAgent'; // The chat component

function AgentTester() {
  const { agentId } = useParams(); // ID of the agent *configuration*
  const navigate = useNavigate();
  const [agentConfig, setAgentConfig] = useState(null); // State holds the config
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAgentConfig = async () => {
      setError(null); // Clear previous errors
      setLoading(true);
      try {
        // Get all local configs and find the one with matching ID
        const agents = await apiService.getAgents();
        const foundAgentConfig = agents.find(a => a.id === agentId);

        if (foundAgentConfig) {
          setAgentConfig(foundAgentConfig);
        } else {
          setError(`Agent configuration with ID ${agentId} not found.`);
          setAgentConfig(null); // Explicitly set to null if not found
        }
      } catch (err) {
        console.error('Error fetching agent configuration:', err);
        setError('Failed to load agent configuration: ' + (err.message || ''));
        setAgentConfig(null);
      } finally {
        setLoading(false);
      }
    };

    if (agentId) {
      fetchAgentConfig();
    } else {
      setError('No agent configuration ID provided in URL.');
      setLoading(false);
      setAgentConfig(null);
    }
  }, [agentId]); // Re-fetch if agentId changes

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 8 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/')} // Navigate back to dashboard
          sx={{ mr: 2 }}
        >
          Back to Dashboard
        </Button>
        <Typography variant="h4" component="h1">
          Test Agent Configuration {/* UPDATED Title */}
        </Typography>
      </Box>

      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 8 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Alert severity="error" sx={{ mb: 4 }}>
          {error} - <Button onClick={() => navigate('/')} size="small">Go to Dashboard</Button>
        </Alert>
      ) : agentConfig ? (
        // Pass the loaded agent configuration to the TestAgent chat component
        <TestAgent agentData={agentConfig} />
      ) : (
        // Should ideally not happen if error state is handled, but as a fallback
        <Alert severity="warning">
          Agent configuration not found or failed to load. Please select an agent from the dashboard.
           - <Button onClick={() => navigate('/')} size="small">Go to Dashboard</Button>
        </Alert>
      )}
    </Container>
  );
}

export default AgentTester;