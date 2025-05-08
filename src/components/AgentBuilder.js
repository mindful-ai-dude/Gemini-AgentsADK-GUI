// src/components/AgentBuilder.js
import React, { useState, useEffect } from 'react';
import apiService from '../utils/apiService'; // Uses refactored service
import { useNavigate, useParams } from 'react-router-dom';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Paper from '@mui/material/Paper';
import Stepper from '@mui/material/Stepper';
import Step from '@mui/material/Step';
import StepLabel from '@mui/material/StepLabel';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import SaveIcon from '@mui/icons-material/Save';
// import PlayArrowIcon from '@mui/icons-material/PlayArrow'; // Removed unused import
import CodeIcon from '@mui/icons-material/Code';

// Import builder steps
import BasicDetails from './builder/BasicDetails';
import InstructionsEditor from './builder/InstructionsEditor';
import ToolsSelector from './builder/ToolsSelector';
import SubAgentsSelector from './builder/SubAgentsSelector'; // RENAMED import
import GuardrailsConfigurator from './builder/GuardrailsConfigurator';
import CodePreview from './builder/CodePreview';
import TestAgent from './builder/TestAgent';

// Updated step names
const steps = [
  'Basic Details',
  'Instructions',
  'Tools',
  'Sub-Agents', // UPDATED Step Name
  'Guardrails',
  'Code Preview',
  'Test Agent Config' // UPDATED Step Name
];

// Default agent configuration structure for Google ADK
const defaultAgentData = {
  name: '',
  description: '',
  model: 'gemini-1.5-flash', // Default Google model
  instructions: 'You are a helpful Google AI assistant.',
  modelSettings: { // Using defaults from modelOptions.js
    temperature: 0.7,
    topP: 0.95,
    topK: 40,
    maxOutputTokens: 8192
  },
  tools: [], // List of tool configurations
  handoffs: [], // Conceptually maps to sub_agents in ADK
  inputGuardrails: [], // Configuration for input guardrails
  outputGuardrails: [], // Configuration for output guardrails
  outputType: null, // For structured output configuration
  // Timestamps managed by apiService
};


function AgentBuilder() {
  const navigate = useNavigate();
  const { agentId } = useParams(); // ID of the agent *configuration*
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // State holds the agent *configuration* being built/edited
  const [agentData, setAgentData] = useState(defaultAgentData);

  // Load agent config data if editing an existing one
  useEffect(() => {
    const loadAgentConfigData = async () => {
      if (agentId) {
        try {
          setLoading(true);
          // apiService.getAgents gets all local configs
          const agents = await apiService.getAgents();
          const foundAgentConfig = agents.find(a => a.id === agentId);

          if (foundAgentConfig) {
            // Ensure all expected fields exist, merging with defaults
            setAgentData(prev => ({ ...defaultAgentData, ...prev, ...foundAgentConfig }));
            setError(null);
          } else {
            setError(`Agent configuration with ID ${agentId} not found. Creating new.`);
            setAgentData(defaultAgentData); // Reset to default if not found
          }
        } catch (err) {
          console.error('Error loading agent configuration:', err);
          setError('Failed to load agent configuration: ' + (err.message || ''));
          setAgentData(defaultAgentData); // Reset on error
        } finally {
          setLoading(false);
        }
      } else {
        // Check session storage for cloning
        const storedAgentData = sessionStorage.getItem('agent_to_edit');
        if (storedAgentData) {
          try {
            const parsedData = JSON.parse(storedAgentData);
            // Create a new config based on cloned data, ensuring defaults
            setAgentData({ ...defaultAgentData, ...parsedData, id: undefined, created_at: undefined, updated_at: undefined, last_used: undefined, lastUsed: 'Never' });
            sessionStorage.removeItem('agent_to_edit');
            setError(null);
          } catch (err) {
            console.error('Error parsing stored agent data for cloning:', err);
            setError('Failed to load cloned data.');
            setAgentData(defaultAgentData);
          }
        } else {
          // Start with default data for a new agent config
          setAgentData(defaultAgentData);
          setError(null);
        }
      }
    };

    loadAgentConfigData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId]); // Dependency is agentId

  const handleNext = () => {
    setActiveStep((prev) => prev + 1);
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };

  // Update specific field in the agent configuration state
  const updateAgentData = (field, value) => {
    setAgentData((prev) => ({
      ...prev,
      [field]: value
    }));
  };

  // Save the agent configuration locally using apiService
  const handleSave = async () => {
    setError(null); // Clear previous errors
    // Basic validation
    if (!agentData.name || !agentData.name.trim()) {
        setError("Agent Name is required.");
        setActiveStep(0); // Go back to basic details step
        return null; // Indicate save failed
    }
     if (!agentData.instructions || !agentData.instructions.trim()) {
        setError("Agent Instructions are required.");
        setActiveStep(1); // Go back to instructions step
        return null; // Indicate save failed
    }

    let savedAgentConfig = null;
    try {
      setLoading(true);
      if (agentData.id) { // Check if it's an existing config by checking if id exists
        // Update existing agent config
        savedAgentConfig = await apiService.updateAgent(agentData.id, agentData);
        console.log("Agent configuration updated successfully:", savedAgentConfig.id);
      } else {
        // Create new agent config (apiService generates ID)
        savedAgentConfig = await apiService.createAgent(agentData);
        console.log("Agent configuration created successfully:", savedAgentConfig.id);
        // Update state with the newly created agent config including its ID
        setAgentData(savedAgentConfig);
      }
      return savedAgentConfig; // Return the saved/updated config
    } catch (error) {
      console.error("Error saving agent configuration:", error);
      setError("Failed to save agent configuration: " + (error.message || "Please try again."));
      return null; // Indicate save failed
    } finally {
      setLoading(false);
    }
  };

  // Removed unused function handleSaveAndTest
  // const handleSaveAndTest = async () => {
  //     const savedConfig = await handleSave();
  //     if (savedConfig && savedConfig.id) {
  //         navigate(`/test/${savedConfig.id}`);
  //     } else {
  //         // Error occurred during save, message already set by handleSave
  //         console.log("Save failed, not navigating to test page.");
  //     }
  // };


  // Render the content for the current step
  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return <BasicDetails agentData={agentData} updateAgentData={updateAgentData} />;
      case 1:
        return <InstructionsEditor agentData={agentData} updateAgentData={updateAgentData} />;
      case 2:
        return <ToolsSelector agentData={agentData} updateAgentData={updateAgentData} />;
      case 3:
        // Use the renamed component
        return <SubAgentsSelector agentData={agentData} updateAgentData={updateAgentData} />;
      case 4:
        return <GuardrailsConfigurator agentData={agentData} updateAgentData={updateAgentData} />;
      case 5:
        // Pass the current config state to CodePreview
        return <CodePreview agentData={agentData} />;
      case 6:
        // Pass the current config state to TestAgent
        return <TestAgent agentData={agentData} />;
      default:
        return 'Unknown step';
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 8 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/')}
          sx={{ mr: 2 }}
        >
          Back to Dashboard
        </Button>
        <Typography variant="h4" component="h1">
          {/* Check if agentData has an ID to determine edit/create mode */}
          {agentData.id ? 'Edit Agent Configuration' : 'Create Agent Configuration'}
        </Typography>
      </Box>

      {/* Display loading indicator */}
      {loading && !agentData.id && !sessionStorage.getItem('agent_to_edit') ? ( // Show loading only on initial load for new/clone
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 8 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Paper sx={{ p: 4, borderRadius: 2 }} elevation={2}>
          {/* Display error messages */}
          {error && (
            <Alert severity="error" sx={{ mb: 4 }}>
              {error}
            </Alert>
          )}
          <Stepper activeStep={activeStep} alternativeLabel sx={{ mb: 4 }}>
            {steps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>

          <Box>
            {/* Render the content for the active step */}
            {getStepContent(activeStep)}

            <Divider sx={{ my: 4 }} />

            {/* Navigation Buttons */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Button
                variant="outlined"
                disabled={activeStep === 0}
                onClick={handleBack}
              >
                Back
              </Button>

              <Box>
                {/* Final step buttons */}
                {activeStep === steps.length - 1 ? (
                  <>
                    <Button
                      variant="contained"
                      startIcon={<SaveIcon />}
                      onClick={async () => {
                          const saved = await handleSave();
                          if (saved) navigate('/'); // Navigate to dashboard on successful save
                      }}
                      disabled={loading}
                      sx={{ mr: 1 }}
                    >
                      {agentData.id ? 'Update Config' : 'Save Config'}
                    </Button>
                    {/* Removed Save & Run as TestAgent is the last step */}
                  </>
                ) : (
                  // Next/Generate Code button
                  <Button
                    variant="contained"
                    onClick={handleNext}
                    // Show Code icon only on the step before Code Preview
                    endIcon={activeStep === steps.length - 3 ? <CodeIcon /> : null}
                  >
                    {/* Adjust button text based on the next step */}
                    {activeStep === steps.length - 3 ? 'Generate Code' :
                     activeStep === steps.length - 2 ? 'Test Agent Config' : 'Next'}
                  </Button>
                )}
              </Box>
            </Box>
          </Box>
        </Paper>
      )}
    </Container>
  );
}

export default AgentBuilder;