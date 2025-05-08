// src/components/builder/BasicDetails.js
import React from 'react';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Slider from '@mui/material/Slider';
import InputAdornment from '@mui/material/InputAdornment';
import Tooltip from '@mui/material/Tooltip';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import { modelOptions, modelSettingsDefaults } from '../../utils/modelOptions'; // Uses updated options

function BasicDetails({ agentData, updateAgentData }) {

  // Ensure modelSettings exists and has defaults
  const modelSettings = { ...modelSettingsDefaults, ...(agentData.modelSettings || {}) };

  const handleModelSettingChange = (setting, value) => {
    // Ensure value is a number for sliders/numeric fields
    const numericValue = (typeof value === 'string' && !isNaN(parseFloat(value))) ? parseFloat(value) : value;
     // Handle potential NaN after parseFloat
     const finalValue = isNaN(numericValue) ? modelSettingsDefaults[setting] : numericValue;

    updateAgentData('modelSettings', {
      ...modelSettings, // Use the ensured modelSettings object
      [setting]: finalValue
    });
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Agent Configuration Details
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure the basic details and model settings for your agent configuration.
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Agent Config Name" // UPDATED Label
            value={agentData.name || ''} // Ensure controlled component
            onChange={(e) => updateAgentData('name', e.target.value)}
            placeholder="e.g., Customer Support Agent Config"
            helperText="A descriptive name for this configuration"
            required
            error={!agentData.name} // Add error state if name is empty
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <FormControl fullWidth required>
            <InputLabel id="model-select-label">Model</InputLabel>
            <Select
              labelId="model-select-label"
              value={agentData.model || modelOptions[0].value} // Ensure controlled component with default
              onChange={(e) => updateAgentData('model', e.target.value)}
              label="Model"
            >
              {modelOptions.map((option) => (
                <MenuItem key={option.value} value={option.value}>
                  {/* Improved display for model options */}
                  <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="body1" component="span">{option.label}</Typography>
                    <Typography variant="caption" color="text.secondary" component="span">
                      {option.description}
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
             <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
               Select the Google AI model to use.
             </Typography>
          </FormControl>
        </Grid>

        <Grid item xs={12}>
          <TextField
            fullWidth
            label="Agent Description"
            value={agentData.description || ''} // Ensure controlled component
            onChange={(e) => updateAgentData('description', e.target.value)}
            placeholder="e.g., Handles customer inquiries about Google Cloud products."
            helperText="A brief description of what this agent configuration is for."
            multiline
            rows={2}
          />
        </Grid>

        {/* --- Model Settings --- */}
        <Grid item xs={12}>
          <Typography variant="subtitle1" gutterBottom sx={{ mt: 2 }}>
            Model Settings (Gemini Defaults)
          </Typography>
        </Grid>

        <Grid item xs={12} md={6}>
          <Box sx={{ px: 1 }}> {/* Reduced padding slightly */}
            <Typography variant="body2" gutterBottom id="temperature-slider-label">
              Temperature
              <Tooltip title="Controls randomness. Lower values (e.g., 0.2) are more deterministic, higher values (e.g., 0.9) are more creative.">
                <InfoOutlinedIcon fontSize="small" sx={{ ml: 0.5, verticalAlign: 'middle', color: 'text.secondary' }} />
              </Tooltip>
            </Typography>
            <Slider
              aria-labelledby="temperature-slider-label"
              value={typeof modelSettings.temperature === 'number' ? modelSettings.temperature : modelSettingsDefaults.temperature}
              onChange={(_, value) => handleModelSettingChange('temperature', value)}
              min={0}
              max={1} // Gemini temperature typically 0-1
              step={0.1}
              marks={[{ value: 0, label: '0.0' }, { value: 0.5, label: '0.5' }, { value: 1, label: '1.0' }]}
              valueLabelDisplay="auto"
            />
          </Box>
        </Grid>

        <Grid item xs={12} md={6}>
          <Box sx={{ px: 1 }}>
            <Typography variant="body2" gutterBottom id="top-p-slider-label">
              Top P
              <Tooltip title="Nucleus sampling. Considers the smallest set of tokens whose probability adds up to top_p. (e.g., 0.95)">
                <InfoOutlinedIcon fontSize="small" sx={{ ml: 0.5, verticalAlign: 'middle', color: 'text.secondary' }} />
              </Tooltip>
            </Typography>
            <Slider
              aria-labelledby="top-p-slider-label"
              value={typeof modelSettings.topP === 'number' ? modelSettings.topP : modelSettingsDefaults.topP}
              onChange={(_, value) => handleModelSettingChange('topP', value)}
              min={0}
              max={1}
              step={0.05}
              marks={[{ value: 0, label: '0.0' }, { value: 0.5, label: '0.5' }, { value: 1, label: '1.0' }]}
              valueLabelDisplay="auto"
            />
          </Box>
        </Grid>

        <Grid item xs={12} md={6}>
           <Box sx={{ px: 1 }}>
            <Typography variant="body2" gutterBottom id="top-k-slider-label">
              Top K
              <Tooltip title="Selects the next token from the top K most likely tokens. (e.g., 40)">
                <InfoOutlinedIcon fontSize="small" sx={{ ml: 0.5, verticalAlign: 'middle', color: 'text.secondary' }} />
              </Tooltip>
            </Typography>
            <Slider
              aria-labelledby="top-k-slider-label"
              value={typeof modelSettings.topK === 'number' ? modelSettings.topK : modelSettingsDefaults.topK}
              onChange={(_, value) => handleModelSettingChange('topK', value)}
              min={1}
              max={100} // Example range for Top K
              step={1}
              marks={[{ value: 1, label: '1' }, { value: 50, label: '50' }, { value: 100, label: '100' }]}
              valueLabelDisplay="auto"
            />
          </Box>
        </Grid>

        <Grid item xs={12} md={6}>
          <TextField
            fullWidth
            label="Max Output Tokens"
            type="number"
            value={modelSettings.maxOutputTokens || ''} // Use the value from state
            onChange={(e) => handleModelSettingChange('maxOutputTokens', parseInt(e.target.value, 10) || modelSettingsDefaults.maxOutputTokens)} // Parse int, provide default on NaN
            InputProps={{
              inputProps: { min: 1, step: 1 },
              startAdornment: (
                <InputAdornment position="start">
                  <Tooltip title="Maximum number of tokens to generate in the response.">
                    <InfoOutlinedIcon fontSize="small" sx={{ color: 'text.secondary' }} />
                  </Tooltip>
                </InputAdornment>
              ),
            }}
            helperText="Limits the length of the generated response."
          />
        </Grid>

        {/* Frequency/Presence Penalty removed as they are less common for Gemini, check ADK docs if needed */}

      </Grid>
    </Box>
  );
}

export default BasicDetails;