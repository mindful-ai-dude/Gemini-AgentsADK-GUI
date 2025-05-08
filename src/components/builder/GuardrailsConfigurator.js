// src/components/builder/GuardrailsConfigurator.js
import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import FormGroup from '@mui/material/FormGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import Switch from '@mui/material/Switch';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Alert from '@mui/material/Alert';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import Stack from '@mui/material/Stack';
import AddIcon from '@mui/icons-material/Add';
import SecurityIcon from '@mui/icons-material/Security'; // Input guardrail icon
import VerifiedUserOutlinedIcon from '@mui/icons-material/VerifiedUserOutlined'; // Output guardrail icon
import SchemaIcon from '@mui/icons-material/Schema'; // Icon for structured output
import CodeIcon from '@mui/icons-material/Code';
import Editor from '@monaco-editor/react';
import { guardrailTemplate, structuredOutputTemplate } from '../../utils/codeTemplates'; // Use updated templates
import { ThemeProvider, useTheme } from '@mui/material/styles'; // To get theme mode for editor

function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`guardrails-tabpanel-${index}`}
      aria-labelledby={`guardrails-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}> {/* Increased padding */}
          {children}
        </Box>
      )}
    </div>
  );
}

function GuardrailsConfigurator({ agentData, updateAgentData }) {
  const theme = useTheme(); // Get theme for editor
  const [tabValue, setTabValue] = useState(0);
  const [openDialog, setOpenDialog] = useState(false);
  const [dialogMode, setDialogMode] = useState('input'); // 'input', 'output', or 'structured'
  const [customCode, setCustomCode] = useState('');
  const [currentGuardrailId, setCurrentGuardrailId] = useState(null); // For editing

  // Example state for predefined guardrails (adapt based on actual ADK features)
  const [enabledGuardrails, setEnabledGuardrails] = useState(() => {
      // Initialize from agentData or defaults
      const initial = {
          // Input
          googleSafetyFilter: agentData.inputGuardrails?.some(g => g.id === 'googleSafetyFilter') ?? true, // Example default
          customInputBlocker: agentData.inputGuardrails?.some(g => g.id === 'customInputBlocker') ?? false,
          // Output
          googleSafetyFilterOutput: agentData.outputGuardrails?.some(g => g.id === 'googleSafetyFilterOutput') ?? true, // Example default
          customOutputModifier: agentData.outputGuardrails?.some(g => g.id === 'customOutputModifier') ?? false,
      };
      return initial;
  });

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // --- Dialog Handling ---
  const handleOpenCodeDialog = (mode, guardrail = null) => {
    setDialogMode(mode);
    if (guardrail) {
      // Editing existing custom guardrail
      setCurrentGuardrailId(guardrail.id);
      setCustomCode(guardrail.code || `## Custom ${mode} guardrail: ${guardrail.name}\n\n# Add Python code here\n# Example (Input): \n# if 'bad word' in context.request:\n#   raise GuardrailViolation('Input contains bad word')\n\n# Example (Output):\n# if 'secret' in context.response:\n#   context.response = '[REDACTED]'\n`);
    } else if (mode === 'structured') {
        // Editing/Creating Structured Output Schema
        setCurrentGuardrailId(null); // Not editing a specific guardrail ID
        // Load existing schema or template
        setCustomCode(agentData.outputType?.schema || structuredOutputTemplate.replace('{{className}}', 'MyStructuredOutput').replace('{{fields}}', '  field1: str\n  field2: Optional[int] = None'));
    }
     else {
      // Adding new custom guardrail
      setCurrentGuardrailId(null);
      setCustomCode(guardrailTemplate.replace('{{name}}', `my_${mode}_guardrail`).replace('{{instructions}}', '# Define guardrail logic'));
    }
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setCustomCode('');
    setCurrentGuardrailId(null);
  };

  const handleSaveCustomConfig = () => {
    if (dialogMode === 'structured') {
        // Save structured output schema
        // TODO: Add validation for Pydantic model code if possible client-side
        updateAgentData('outputType', { schema: customCode }); // Store schema code
        console.log("Updated structured output schema.");
    } else {
        // Save custom input/output guardrail
        const guardrailListKey = dialogMode === 'input' ? 'inputGuardrails' : 'outputGuardrails';
        const existingList = agentData[guardrailListKey] || [];
        const newGuardrail = {
            id: currentGuardrailId || `custom-${dialogMode}-${Date.now()}`,
            name: `Custom ${dialogMode} Guardrail ${currentGuardrailId ? '(Edited)' : ''}`, // Simple naming
            type: 'custom', // Indicate it's custom code
            code: customCode,
        };

        let updatedList;
        if (currentGuardrailId) {
            // Update existing
            updatedList = existingList.map(g => g.id === currentGuardrailId ? newGuardrail : g);
        } else {
            // Add new
            updatedList = [...existingList, newGuardrail];
        }
        updateAgentData(guardrailListKey, updatedList);
        console.log(`Saved custom ${dialogMode} guardrail: ${newGuardrail.id}`);
    }
    handleCloseDialog();
  };

  // --- Predefined Guardrail Toggle ---
  const handleGuardrailToggle = (guardrailId, type) => (event) => {
    const isEnabled = event.target.checked;
    const guardrailListKey = type === 'input' ? 'inputGuardrails' : 'outputGuardrails';
    const existingList = agentData[guardrailListKey] || [];
    let updatedList;

    if (isEnabled) {
      // Add the predefined guardrail if not already present
      if (!existingList.some(g => g.id === guardrailId)) {
        updatedList = [...existingList, { id: guardrailId, name: guardrailId, type: 'predefined' }]; // Add predefined marker
      } else {
        updatedList = existingList; // Already exists
      }
    } else {
      // Remove the predefined guardrail
      updatedList = existingList.filter(g => g.id !== guardrailId);
    }

    setEnabledGuardrails(prev => ({ ...prev, [guardrailId]: isEnabled }));
    updateAgentData(guardrailListKey, updatedList);
  };

   // --- Custom Guardrail Deletion ---
   const handleDeleteCustomGuardrail = (guardrailId, type) => {
       const guardrailListKey = type === 'input' ? 'inputGuardrails' : 'outputGuardrails';
       const updatedList = (agentData[guardrailListKey] || []).filter(g => g.id !== guardrailId);
       updateAgentData(guardrailListKey, updatedList);
       console.log(`Deleted custom ${type} guardrail: ${guardrailId}`);
   };


  // Filter custom guardrails for display
  const customInputGuardrails = (agentData.inputGuardrails || []).filter(g => g.type === 'custom');
  const customOutputGuardrails = (agentData.outputGuardrails || []).filter(g => g.type === 'custom');

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Guardrails & Structured Output
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure safety filters, custom validation logic (guardrails), and define structured output schemas for your agent. (Note: Guardrail implementation is highly dependent on the Google ADK specifics).
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="Guardrails Configuration Tabs">
          <Tab label="Input Guardrails" icon={<SecurityIcon />} iconPosition="start" id="guardrails-tab-0" aria-controls="guardrails-tabpanel-0"/>
          <Tab label="Output Guardrails" icon={<VerifiedUserOutlinedIcon />} iconPosition="start" id="guardrails-tab-1" aria-controls="guardrails-tabpanel-1"/>
          <Tab label="Structured Output" icon={<SchemaIcon />} iconPosition="start" id="guardrails-tab-2" aria-controls="guardrails-tabpanel-2"/>
        </Tabs>
      </Box>

      {/* Input Guardrails Panel */}
      <TabPanel value={tabValue} index={0}>
        <Alert severity="info" sx={{ mb: 3 }}>
          Input guardrails check user input *before* it reaches the main agent logic. Use them to block harmful content or enforce input formats. Google's Safety Filters might be applied by default depending on the model and ADK settings.
        </Alert>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: '100%' }}>
              <Typography variant="subtitle1" gutterBottom>Predefined Filters (Example)</Typography>
              <FormGroup>
                <FormControlLabel
                  control={
                    <Switch
                      checked={enabledGuardrails.googleSafetyFilter}
                      onChange={handleGuardrailToggle('googleSafetyFilter', 'input')}
                    />
                  }
                  label="Google Safety Filter (Input)"
                />
                <Typography variant="caption" color="text.secondary" sx={{ pl: 4, mt: -1, mb: 1 }}>
                  Leverage Google's built-in safety classifiers (harmful content, etc.). Behavior depends on ADK/model configuration.
                </Typography>
                {/* Add more predefined input guardrails if ADK offers them */}
              </FormGroup>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Typography variant="subtitle1" gutterBottom>Custom Input Guardrails</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2, flexGrow: 1 }}>
                Define custom Python logic (e.g., using ADK callbacks) to validate or block input.
              </Typography>
               {customInputGuardrails.map(g => (
                   <Chip
                       key={g.id}
                       label={g.name}
                       onDelete={() => handleDeleteCustomGuardrail(g.id, 'input')}
                       onClick={() => handleOpenCodeDialog('input', g)}
                       sx={{ mb: 1, mr: 1 }}
                   />
               ))}
              <Button
                variant="outlined"
                startIcon={<AddIcon />}
                onClick={() => handleOpenCodeDialog('input')}
                sx={{ mt: 'auto' }} // Push button to bottom
              >
                Add Custom Input Guardrail
              </Button>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Output Guardrails Panel */}
      <TabPanel value={tabValue} index={1}>
        <Alert severity="info" sx={{ mb: 3 }}>
          Output guardrails check the agent's generated response *before* it's sent to the user. Use them to filter sensitive data, ensure policy adherence, or modify responses.
        </Alert>
         <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: '100%' }}>
              <Typography variant="subtitle1" gutterBottom>Predefined Filters (Example)</Typography>
              <FormGroup>
                 <FormControlLabel
                  control={
                    <Switch
                      checked={enabledGuardrails.googleSafetyFilterOutput}
                      onChange={handleGuardrailToggle('googleSafetyFilterOutput', 'output')}
                    />
                  }
                  label="Google Safety Filter (Output)"
                />
                 <Typography variant="caption" color="text.secondary" sx={{ pl: 4, mt: -1, mb: 1 }}>
                   Apply Google's safety classifiers to the agent's generated response.
                 </Typography>
                 {/* Add more predefined output guardrails if ADK offers them */}
              </FormGroup>
            </Paper>
          </Grid>
           <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Typography variant="subtitle1" gutterBottom>Custom Output Guardrails</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2, flexGrow: 1 }}>
                Define custom Python logic to validate or modify the agent's final response.
              </Typography>
               {customOutputGuardrails.map(g => (
                   <Chip
                       key={g.id}
                       label={g.name}
                       onDelete={() => handleDeleteCustomGuardrail(g.id, 'output')}
                       onClick={() => handleOpenCodeDialog('output', g)}
                       sx={{ mb: 1, mr: 1 }}
                   />
               ))}
              <Button
                variant="outlined"
                startIcon={<AddIcon />}
                onClick={() => handleOpenCodeDialog('output')}
                 sx={{ mt: 'auto' }}
              >
                Add Custom Output Guardrail
              </Button>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Structured Output Panel */}
      <TabPanel value={tabValue} index={2}>
        <Alert severity="info" sx={{ mb: 3 }}>
          Define a Pydantic model schema to enforce a specific JSON structure for the agent's final output. This requires the agent's instructions to guide it towards using this schema.
        </Alert>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                 <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <SchemaIcon color="primary" sx={{ mr: 1 }} />
                    <Typography variant="h6">
                    Structured Output Schema (Pydantic)
                    </Typography>
                 </Box>
                 <Button
                    variant="contained"
                    color="primary"
                    startIcon={<CodeIcon />}
                    onClick={() => handleOpenCodeDialog('structured')}
                 >
                    {agentData.outputType?.schema ? 'Edit Schema' : 'Define Schema'}
                 </Button>
              </Box>

              <Typography variant="body2" sx={{ mb: 3 }}>
                Define a Pydantic BaseModel class in Python code below. The agent will attempt to format its output according to this schema if instructed correctly.
              </Typography>

              <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1, mb: 3, minHeight: '200px', bgcolor: 'background.default' }}>
                <Editor
                  height="300px" // Fixed height for preview
                  language="python"
                  theme={theme.palette.mode === 'dark' ? 'vs-dark' : 'light'}
                  value={agentData.outputType?.schema || "# No schema defined yet. Click 'Define Schema' to add one."}
                  options={{
                    readOnly: true, // Preview is read-only here
                    minimap: { enabled: false },
                    scrollBeyondLastLine: false,
                    fontSize: 13,
                  }}
                />
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Dialog for Editing Code/Schema */}
      <Dialog
        open={openDialog}
        onClose={handleCloseDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {dialogMode === 'input' ? 'Edit Custom Input Guardrail' :
           dialogMode === 'output' ? 'Edit Custom Output Guardrail' :
           'Edit Structured Output Schema (Pydantic)'}
        </DialogTitle>
        <DialogContent dividers>
          <Typography variant="body2" paragraph>
            {dialogMode === 'structured' ?
             "Define your Pydantic BaseModel class here. Ensure necessary imports like 'BaseModel', 'Field', 'Optional', 'List' from 'pydantic' or 'typing'." :
             `Define your custom ${dialogMode} guardrail function using Python. Refer to Google ADK documentation for the expected function signature and context object structure (e.g., using callbacks).`}
          </Typography>
          {dialogMode !== 'structured' && (
              <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                <Chip label="Use context object" size="small"/>
                <Chip label={dialogMode === 'input' ? "Raise GuardrailViolation on block" : "Modify context.response or raise"} size="small"/>
              </Stack>
          )}

          <Box sx={{ height: 400 }}>
            <Editor
              height="100%"
              language="python"
              theme={theme.palette.mode === 'dark' ? 'vs-dark' : 'light'}
              value={customCode}
              onChange={(value) => setCustomCode(value || '')} // Update state on change
              options={{
                minimap: { enabled: true }, // Enable minimap in dialog
                scrollBeyondLastLine: false,
                fontSize: 13,
                tabSize: 4, // Standard Python tab size
                insertSpaces: true,
              }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>
            Cancel
          </Button>
          <Button
            onClick={handleSaveCustomConfig}
            variant="contained"
            disabled={!customCode.trim()} // Disable save if code is empty
          >
            Save {dialogMode === 'structured' ? 'Schema' : 'Guardrail'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default GuardrailsConfigurator;