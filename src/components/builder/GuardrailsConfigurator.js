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
import SecurityIcon from '@mui/icons-material/Security'; // Input Callbacks/Guardrails icon
import VerifiedUserOutlinedIcon from '@mui/icons-material/VerifiedUserOutlined'; // Output Callbacks/Guardrails icon
import SchemaIcon from '@mui/icons-material/Schema'; // Icon for structured output
import CodeIcon from '@mui/icons-material/Code';
import EditIcon from '@mui/icons-material/Edit'; // Added for editing chips
import DeleteIcon from '@mui/icons-material/Delete'; // Added for deleting chips
import Editor from '@monaco-editor/react';
import { guardrailTemplate, structuredOutputTemplate } from '../../utils/codeTemplates'; // Use updated templates (guardrailTemplate now represents callback examples)
import { useTheme } from '@mui/material/styles'; // To get theme mode for editor

function TabPanel(props) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`callbacks-tabpanel-${index}`} // Updated ID
      aria-labelledby={`callbacks-tab-${index}`} // Updated aria
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}> {/* Consistent padding */}
          {children}
        </Box>
      )}
    </div>
  );
}

function GuardrailsConfigurator({ agentData, updateAgentData }) {
  const theme = useTheme(); // Get theme for editor
  const [tabValue, setTabValue] = useState(0); // 0: Before Callbacks, 1: After Callbacks, 2: Structured Output
  const [openDialog, setOpenDialog] = useState(false);
  const [dialogMode, setDialogMode] = useState('before'); // 'before', 'after', or 'structured'
  const [customCode, setCustomCode] = useState('');
  const [currentCallbackId, setCurrentCallbackId] = useState(null); // For editing custom callback code snippets
  const [currentCallbackName, setCurrentCallbackName] = useState(''); // For dialog title/naming

  // Example state for predefined concepts (ADK uses model safety settings primarily)
  // These switches are more illustrative placeholders now.
  const [enabledSettings, setEnabledSettings] = useState(() => {
      // Initialize based on agentData if specific flags were stored, otherwise defaults
      return {
          googleSafetyFilterInput: agentData.inputGuardrails?.some(g => g.id === 'googleSafetyFilterInput') ?? true, // Default ON
          googleSafetyFilterOutput: agentData.outputGuardrails?.some(g => g.id === 'googleSafetyFilterOutput') ?? true, // Default ON
      };
  });

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // --- Dialog Handling ---
  const handleOpenCodeDialog = (mode, callbackConfig = null) => {
    setDialogMode(mode);
    if (callbackConfig) {
      // Editing existing custom callback code snippet
      setCurrentCallbackId(callbackConfig.id);
      setCurrentCallbackName(callbackConfig.name || `Custom ${mode} Callback`);
      // Provide a more relevant default/example if code is empty
      setCustomCode(callbackConfig.code || `# Python code for ${callbackConfig.name}\n\n# Example for ${mode} callback:\ndef ${callbackConfig.name}(context, data):\n    print("Callback triggered!")\n    # Modify 'data' or return specific object to override\n    return None`);
      setOpenDialog(true);
    } else if (mode === 'structured') {
        // Editing/Creating Structured Output Schema
        setCurrentCallbackId(null); // Not editing a specific callback ID
        setCurrentCallbackName('Structured Output Schema');
        // Load existing schema or template
        setCustomCode(agentData.outputType?.schema || structuredOutputTemplate
            .replace('{{className}}', 'MyStructuredOutput')
            .replace('{{class_description}}', 'Description for the output structure.')
            .replace('{{fields}}', '  field1: str = Field(description="Description of field1")\n  field2: Optional[int] = Field(default=None, description="Description of field2")')
            .replace('{{name}}', agentData.name || 'MyAgent')
            .replace('{{instructions}}', agentData.instructions || 'Provide structured output.')
            .replace('{{model}}', agentData.model || 'gemini-1.5-flash')
        );
        setOpenDialog(true);
    } else {
      // Adding new custom callback code snippet
      const callbackType = mode === 'before' ? 'before_model_callback' : 'after_model_callback'; // Example mapping
      const newName = `my_${callbackType}_${Date.now().toString().slice(-4)}`;
      setCurrentCallbackId(null);
      setCurrentCallbackName(`New Custom ${mode} Callback`);
      setCustomCode(guardrailTemplate // Use template which now contains callback examples
          .replace('{{name}}', agentData.name || 'MyAgent')
          .replace('{{instructions}}', `# Define Python function for ${callbackType}`)
          .replace('@input_guardrail', `# Register as ${callbackType} in Agent constructor`) // Adapt template comments
          .replace('content_filter', newName) // Use generated name
      );
      setOpenDialog(true);
    }
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setCustomCode('');
    setCurrentCallbackId(null);
    setCurrentCallbackName('');
  };

  const handleSaveCustomConfig = () => {
    if (dialogMode === 'structured') {
        // Save structured output schema
        // TODO: Add validation for Pydantic model code if possible client-side
        updateAgentData('outputType', { schema: customCode, name: 'PydanticSchema' }); // Store schema code and a type name
        console.log("Updated structured output schema.");
    } else {
        // Save custom callback code snippet
        // Determine which list to update based on dialogMode ('before' or 'after')
        // We'll store them generically for now, association with specific ADK callbacks happens in backend/code gen
        // Let's use inputGuardrails for 'before_*' and outputGuardrails for 'after_*' conceptually
        const callbackListKey = dialogMode === 'before' ? 'inputGuardrails' : 'outputGuardrails';
        const existingList = agentData[callbackListKey] || [];
        const newCallbackConfig = {
            id: currentCallbackId || `custom-${dialogMode}-${Date.now()}`,
            // Use the name from state, fallback if needed
            name: currentCallbackName || `custom_${dialogMode}_callback_${(existingList.length + 1)}`,
            type: 'custom_callback', // Indicate it's custom callback code
            hook_point: dialogMode, // Store whether it's intended for 'before' or 'after' hooks
            code: customCode,
        };

        let updatedList;
        if (currentCallbackId) {
            // Update existing
            updatedList = existingList.map(cb => cb.id === currentCallbackId ? newCallbackConfig : cb);
        } else {
            // Add new
            updatedList = [...existingList, newCallbackConfig];
        }
        updateAgentData(callbackListKey, updatedList);
        console.log(`Saved custom ${dialogMode} callback config: ${newCallbackConfig.id}`);
    }
    handleCloseDialog();
  };

  // --- Predefined Settings Toggle (Illustrative) ---
  const handleSettingToggle = (settingId, type) => (event) => {
    const isEnabled = event.target.checked;
    const listKey = type === 'input' ? 'inputGuardrails' : 'outputGuardrails';
    const existingList = agentData[listKey] || [];
    let updatedList;

    // Treat these as markers/flags rather than executable guardrails
    if (isEnabled) {
      if (!existingList.some(g => g.id === settingId)) {
        updatedList = [...existingList, { id: settingId, name: settingId, type: 'setting_flag' }];
      } else {
        updatedList = existingList;
      }
    } else {
      updatedList = existingList.filter(g => g.id !== settingId);
    }

    setEnabledSettings(prev => ({ ...prev, [settingId]: isEnabled }));
    updateAgentData(listKey, updatedList);
  };

   // --- Custom Callback Deletion ---
   const handleDeleteCustomCallback = (callbackId, type) => {
       const listKey = type === 'before' ? 'inputGuardrails' : 'outputGuardrails';
       const updatedList = (agentData[listKey] || []).filter(cb => cb.id !== callbackId);
       updateAgentData(listKey, updatedList);
       console.log(`Deleted custom ${type} callback config: ${callbackId}`);
   };

  // Filter custom callbacks for display
  const customBeforeCallbacks = (agentData.inputGuardrails || []).filter(g => g.type === 'custom_callback');
  const customAfterCallbacks = (agentData.outputGuardrails || []).filter(g => g.type === 'custom_callback');

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Callbacks & Structured Output {/* UPDATED Title */}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure ADK Callbacks (Python functions executed at specific points like before/after model/tool calls) to implement guardrails, logging, or custom logic. Define structured output schemas (Pydantic) if needed.
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="Callbacks Configuration Tabs">
          <Tab label="Before Callbacks" icon={<SecurityIcon />} iconPosition="start" id="callbacks-tab-0" aria-controls="callbacks-tabpanel-0"/>
          <Tab label="After Callbacks" icon={<VerifiedUserOutlinedIcon />} iconPosition="start" id="callbacks-tab-1" aria-controls="callbacks-tabpanel-1"/>
          <Tab label="Structured Output" icon={<SchemaIcon />} iconPosition="start" id="callbacks-tab-2" aria-controls="callbacks-tabpanel-2"/>
        </Tabs>
      </Box>

      {/* Before Callbacks Panel (e.g., before_model_callback, before_tool_callback) */}
      <TabPanel value={tabValue} index={0}>
        <Alert severity="info" sx={{ mb: 3 }}>
          Configure Python functions (defined below, executed by the backend) to run *before* core ADK actions like calling the LLM or executing a tool. Use these for input validation, request modification, caching, or access control.
        </Alert>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: '100%' }}>
              <Typography variant="subtitle1" gutterBottom>Model Safety Settings (Illustrative)</Typography>
              <FormGroup>
                <FormControlLabel
                  control={
                    <Switch
                      checked={enabledSettings.googleSafetyFilterInput}
                      onChange={handleSettingToggle('googleSafetyFilterInput', 'input')} // Stored conceptually with 'before' hooks
                    />
                  }
                  label="Use Google Safety Filters (Input)"
                />
                <Typography variant="caption" color="text.secondary" sx={{ pl: 4, mt: -1, mb: 1 }}>
                  Leverage Google's built-in safety classifiers. Configuration happens via `generate_content_config` in the agent definition. This switch is a reminder/placeholder.
                </Typography>
              </FormGroup>
            </Paper>
          </Grid>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Typography variant="subtitle1" gutterBottom>Custom 'Before' Callbacks</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2, flexGrow: 1 }}>
                Add Python code snippets for functions like `before_model_callback` or `before_tool_callback`. The exact hook point is determined by the function signature/registration in the backend.
              </Typography>
               {customBeforeCallbacks.map(cb => (
                   <Chip
                       key={cb.id}
                       label={cb.name}
                       onDelete={() => handleDeleteCustomCallback(cb.id, 'before')}
                       onClick={() => handleOpenCodeDialog('before', cb)}
                       icon={<EditIcon fontSize="small" />}
                       sx={{ mb: 1, mr: 1, justifyContent: 'space-between' }}
                       deleteIcon={<DeleteIcon />}
                   />
               ))}
              <Button
                variant="outlined"
                startIcon={<AddIcon />}
                onClick={() => handleOpenCodeDialog('before')}
                sx={{ mt: 'auto' }}
              >
                Add Custom 'Before' Callback
              </Button>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* After Callbacks Panel (e.g., after_model_callback, after_tool_callback) */}
      <TabPanel value={tabValue} index={1}>
        <Alert severity="info" sx={{ mb: 3 }}>
          Configure Python functions to run *after* core ADK actions. Use these for output validation/modification (e.g., PII filtering), logging results, or triggering actions based on the output.
        </Alert>
         <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: '100%' }}>
              <Typography variant="subtitle1" gutterBottom>Model Safety Settings (Illustrative)</Typography>
              <FormGroup>
                 <FormControlLabel
                  control={
                    <Switch
                      checked={enabledSettings.googleSafetyFilterOutput}
                      onChange={handleSettingToggle('googleSafetyFilterOutput', 'output')} // Stored conceptually with 'after' hooks
                    />
                  }
                  label="Use Google Safety Filters (Output)"
                />
                 <Typography variant="caption" color="text.secondary" sx={{ pl: 4, mt: -1, mb: 1 }}>
                   Apply Google's safety classifiers to the agent's generated response. Configured via `generate_content_config`.
                 </Typography>
              </FormGroup>
            </Paper>
          </Grid>
           <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
              <Typography variant="subtitle1" gutterBottom>Custom 'After' Callbacks</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2, flexGrow: 1 }}>
                 Add Python code snippets for functions like `after_model_callback` or `after_tool_callback`.
              </Typography>
               {customAfterCallbacks.map(cb => (
                   <Chip
                       key={cb.id}
                       label={cb.name}
                       onDelete={() => handleDeleteCustomCallback(cb.id, 'after')}
                       onClick={() => handleOpenCodeDialog('after', cb)}
                       icon={<EditIcon fontSize="small" />}
                       sx={{ mb: 1, mr: 1, justifyContent: 'space-between' }}
                       deleteIcon={<DeleteIcon />}
                   />
               ))}
              <Button
                variant="outlined"
                startIcon={<AddIcon />}
                onClick={() => handleOpenCodeDialog('after')}
                 sx={{ mt: 'auto' }}
              >
                Add Custom 'After' Callback
              </Button>
            </Paper>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Structured Output Panel */}
      <TabPanel value={tabValue} index={2}>
        <Alert severity="warning" sx={{ mb: 3 }}>
          Defining a structured output schema (using the `output_schema` parameter in ADK's `LlmAgent`) currently **disables tool usage** for that specific agent. Ensure your instructions clearly tell the agent to conform to the schema.
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
                Define a Pydantic `BaseModel` class below. The agent will attempt to format its output according to this schema if instructed correctly.
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
                 <Button
                    variant="outlined"
                    color="error"
                    disabled={!agentData.outputType?.schema}
                    onClick={() => {
                         if (window.confirm("Are you sure you want to remove the structured output schema?")) {
                             updateAgentData('outputType', null);
                         }
                    }}
                 >
                    Remove Schema
                 </Button>
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
          {dialogMode === 'before' ? `Edit Custom 'Before' Callback: ${currentCallbackName}` :
           dialogMode === 'after' ? `Edit Custom 'After' Callback: ${currentCallbackName}` :
           'Edit Structured Output Schema (Pydantic)'}
        </DialogTitle>
        <DialogContent dividers>
           {/* Input field for Callback Name when adding new */}
           {dialogMode !== 'structured' && !currentCallbackId && (
               <TextField
                 autoFocus
                 margin="dense"
                 label="Callback Function Name"
                 type="text"
                 fullWidth
                 variant="standard"
                 value={currentCallbackName.replace("New Custom before Callback", "my_before_callback").replace("New Custom after Callback", "my_after_callback")} // Provide default suggestion
                 onChange={(e) => setCurrentCallbackName(e.target.value)}
                 helperText="Enter a valid Python function name (e.g., check_user_input)."
                 sx={{ mb: 2 }}
               />
           )}
          <Typography variant="body2" paragraph>
            {dialogMode === 'structured' ?
             "Define your Pydantic BaseModel class here. Ensure necessary imports like 'BaseModel', 'Field', 'Optional', 'List' from 'pydantic' or 'typing'." :
             `Define the Python code for the '${currentCallbackName || 'callback function'}'. Refer to Google ADK documentation for expected function signatures (e.g., '(context: CallbackContext, data: LlmRequest | LlmResponse | dict)') and how to register it with the LlmAgent.`}
          </Typography>
          {dialogMode !== 'structured' && (
              <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                <Chip label="Access context via 'context'" size="small"/>
                <Chip label="Modify request/response or return object to override" size="small"/>
                <Chip label="Return None to continue" size="small"/>
              </Stack>
          )}

          <Box sx={{ height: 400, border: '1px solid', borderColor: 'divider' }}>
            <Editor
              height="100%"
              language="python"
              theme={theme.palette.mode === 'dark' ? 'vs-dark' : 'light'}
              value={customCode}
              onChange={(value) => setCustomCode(value || '')}
              options={{
                minimap: { enabled: true },
                scrollBeyondLastLine: false,
                fontSize: 13,
                tabSize: 4,
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
            disabled={!customCode.trim() || (dialogMode !== 'structured' && !currentCallbackId && !currentCallbackName.trim())}
          >
            Save {dialogMode === 'structured' ? 'Schema' : 'Callback Code'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default GuardrailsConfigurator;