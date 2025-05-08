// src/components/builder/ToolsSelector.js
import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import ListItemIcon from '@mui/material/ListItemIcon';
import Divider from '@mui/material/Divider';
import Chip from '@mui/material/Chip';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Alert from '@mui/material/Alert'; // Added Alert
import SearchIcon from '@mui/icons-material/Search';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import CodeIcon from '@mui/icons-material/Code';
import PublicIcon from '@mui/icons-material/Public'; // Google Search
import DescriptionIcon from '@mui/icons-material/Description'; // File Search (hypothetical)
import SmartToyIcon from '@mui/icons-material/SmartToy'; // Code Execution / Vertex AI Search
import FunctionsIcon from '@mui/icons-material/Functions'; // Custom Function
import TimerIcon from '@mui/icons-material/Timer'; // Long Running Function
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import EditIcon from '@mui/icons-material/Edit'; // For editing custom tools
import Editor from '@monaco-editor/react';
import { useTheme } from '@mui/material/styles';

// Import the combined/updated tools list and templates
import { allTools, functionToolTemplate, fullToolExampleTemplate } from '../../utils/tools'; // Assuming fullToolExampleTemplate is now in tools.js
import { modelSettingsDefaults } from '../../utils/modelOptions'; // For default settings reference

function ToolsSelector({ agentData, updateAgentData }) {
  const theme = useTheme();
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTab, setSelectedTab] = useState(0); // 0: Built-in, 1: Function Examples
  const [openDialog, setOpenDialog] = useState(false);
  const [currentToolConfig, setCurrentToolConfig] = useState(null); // Holds the config being edited/viewed
  const [customToolCode, setCustomToolCode] = useState(''); // Code for the editor
  const [isEditing, setIsEditing] = useState(false); // Track if dialog is for editing

  // --- Tool Management ---
  const handleAddTool = (toolToAdd) => {
    // Check for limitations (e.g., only one built-in tool)
    const isAddingBuiltIn = builtInTools.some(bt => bt.id === toolToAdd.id);
    const hasBuiltInAlready = agentData.tools?.some(t => builtInTools.some(bt => bt.id === t.id));

    if (isAddingBuiltIn && hasBuiltInAlready) {
      alert("Limitation: Only one built-in tool (Google Search, Code Execution, Vertex AI Search) can be added per agent currently. Use Agent-as-a-Tool pattern for multiple capabilities.");
      return;
    }

    // For tools requiring parameters (like Vertex AI Search), prompt if needed or add with defaults
    let finalToolToAdd = { ...toolToAdd };
    if (toolToAdd.id === 'vertexaisearch' && !toolToAdd.parameters?.data_store_id?.value) {
        const dsId = prompt(`Please enter the Vertex AI Search Datastore ID for '${toolToAdd.name}':\n(Format: projects/<PROJECT_ID>/locations/<LOCATION>/.../dataStores/<DATASTORE_ID>)`);
        if (!dsId) return; // User cancelled
        finalToolToAdd = {
            ...toolToAdd,
            parameters: {
                ...toolToAdd.parameters,
                data_store_id: { ...toolToAdd.parameters.data_store_id, value: dsId }
            }
        };
    }


    const updatedTools = [...(agentData.tools || []), finalToolToAdd];
    updateAgentData('tools', updatedTools);
  };

  const handleRemoveTool = (toolId) => {
    const updatedTools = (agentData.tools || []).filter(tool => tool.id !== toolId);
    updateAgentData('tools', updatedTools);
  };

  // --- Dialog Handling for Custom Tools ---
  const handleOpenToolDialog = (toolConfig = null) => {
    if (toolConfig && toolConfig.category === 'Function') { // Only allow editing/viewing custom functions for now
      setCurrentToolConfig(toolConfig);
      setCustomToolCode(toolConfig.code || '# Add your Python function code here');
      setIsEditing(true); // Editing existing
    } else {
      // Adding a new custom tool
      setCurrentToolConfig({ // Pre-fill with structure for a new tool
          id: `custom-${Date.now()}`,
          name: '',
          description: '',
          category: 'Function',
          parameters: {}, // User defines via docstring/signature
          code: functionToolTemplate // Start with the template
              .replace('{{name}}', 'my_custom_tool')
              .replace('{{parameters}}', 'param1: str, param2: int = 5')
              .replace('{{return_type}}', 'str')
              .replace('{{description}}', 'Description of what this tool does.')
              .replace('{{args_docs}}', 'param1: Description of param1.\n        param2: Description of param2 (default: 5).')
              .replace('{{return_doc}}', 'Description of the return value.')
              .replace('{{args_list}}', '{param1=, param2=}')
              .replace('result = "Simulated result from {{name}}"', 'result = f"Processed {param1} and {param2}"\n    return result')
      });
      setCustomToolCode(currentToolConfig?.code || ''); // Set editor code
      setIsEditing(false); // Adding new
    }
    setOpenDialog(true);
  };

  const handleCloseToolDialog = () => {
    setOpenDialog(false);
    setCurrentToolConfig(null);
    setCustomToolCode('');
    setIsEditing(false);
  };

  const handleSaveCustomTool = () => {
    if (!currentToolConfig || !customToolCode.trim()) {
        alert("Cannot save empty tool code.");
        return;
    }

    // TODO: Basic validation - try to extract function name and description from code?
    // This is complex client-side. For now, assume user provides name/desc separately or it's in the code.
    const updatedTool = {
        ...currentToolConfig,
        code: customToolCode,
        // Attempt to parse name/description from code if needed, or require manual input
        name: currentToolConfig.name || 'my_custom_tool', // Placeholder if not parsed
        description: currentToolConfig.description || 'Custom function tool.' // Placeholder
    };

    let updatedTools;
    if (isEditing) {
      // Update existing tool in the list
      updatedTools = (agentData.tools || []).map(t => t.id === updatedTool.id ? updatedTool : t);
    } else {
      // Add new tool to the list
      updatedTools = [...(agentData.tools || []), updatedTool];
    }
    updateAgentData('tools', updatedTools);
    handleCloseDialog();
  };


  // --- UI Helpers ---
  const getToolIcon = (category) => {
    switch (category) {
      case 'Built-in': return <SmartToyIcon color="primary"/>; // Or specific icons below
      case 'Web': return <PublicIcon color="info"/>; // Google Search
      case 'Code': return <CodeIcon color="secondary"/>; // Code Execution
      case 'File': return <DescriptionIcon color="action"/>; // Vertex AI Search (or File Search)
      case 'Function': return <FunctionsIcon color="success"/>;
      case 'Long Running Function': return <TimerIcon color="warning"/>;
      case 'Agent-as-Tool': return <AccountTreeIcon color="disabled"/>; // Example
      default: return <ExtensionIcon color="disabled"/>;
    }
  };

  // Filter tools based on search term and selected tab
  const getFilteredTools = () => {
      const sourceList = selectedTab === 0 ? builtInTools : exampleFunctionTools;
      if (!searchTerm) return sourceList;
      const lowerSearchTerm = searchTerm.toLowerCase();
      return sourceList.filter(
          tool => (tool.name?.toLowerCase().includes(lowerSearchTerm)) ||
                  (tool.description?.toLowerCase().includes(lowerSearchTerm)) ||
                  (tool.category?.toLowerCase().includes(lowerSearchTerm))
      );
  };

  const displayedTools = getFilteredTools();
  const activeTools = agentData.tools || [];

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Agent Tools Configuration
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Select built-in tools or define custom Python functions to give your agent capabilities. The agent's instructions should guide its tool usage.
      </Typography>
       <Alert severity="warning" sx={{ mb: 2 }}>
           **Limitation:** Currently, ADK supports only one built-in tool (Google Search, Code Execution, Vertex AI Search) per agent. Built-in tools cannot be used in sub-agents. Use the Agent-as-Tool pattern for more complex workflows involving multiple built-in capabilities.
       </Alert>

      <Grid container spacing={3}>
        {/* Left Panel: Available Tools */}
        <Grid item xs={12} md={7}>
          <Paper sx={{ height: '100%', p: 2, display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1, flexShrink: 0 }}>
              <TextField
                fullWidth
                size="small"
                placeholder="Search available tools..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: <SearchIcon color="action" sx={{ mr: 1 }} />,
                }}
                sx={{ mr: 2 }}
              />
              <Tooltip title="Define a new custom Python function tool">
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={() => handleOpenToolDialog()} // Open dialog for new custom tool
                  size="small"
                >
                  Custom Tool
                </Button>
              </Tooltip>
            </Box>

            <Tabs
              value={selectedTab}
              onChange={(_, newValue) => setSelectedTab(newValue)}
              sx={{ borderBottom: 1, borderColor: 'divider', mb: 1, flexShrink: 0 }}
              aria-label="Tool type tabs"
            >
              <Tab label="Built-in Tools" id="tool-tab-0" aria-controls="tool-tabpanel-0"/>
              <Tab label="Function Examples" id="tool-tab-1" aria-controls="tool-tabpanel-1"/>
            </Tabs>

            {/* Tool List Area */}
            <Box sx={{ overflowY: 'auto', flexGrow: 1 }}>
              <List dense> {/* Use dense list */}
                {displayedTools.map((tool) => {
                  const isAdded = activeTools.some(t => t.id === tool.id);
                  return (
                    <ListItem
                      key={tool.id}
                      className="tool-item"
                      secondaryAction={
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                           {/* Show Edit button only for added custom functions */}
                           {isAdded && tool.category === 'Function' && (
                               <Tooltip title="Edit Custom Tool Code">
                                   <IconButton edge="end" size="small" sx={{ mr: 0.5 }} onClick={() => handleOpenToolDialog(tool)}>
                                       <EditIcon fontSize="small" />
                                   </IconButton>
                               </Tooltip>
                           )}
                           {/* Add/Remove Button */}
                           {isAdded ? (
                             <Tooltip title="Remove this tool from the agent">
                               <IconButton edge="end" aria-label="remove" color="error" size="small" onClick={() => handleRemoveTool(tool.id)}>
                                 <DeleteIcon fontSize="small"/>
                               </IconButton>
                             </Tooltip>
                           ) : (
                             <Button variant="outlined" size="small" onClick={() => handleAddTool(tool)} sx={{ ml: 1 }}>
                               Add
                             </Button>
                           )}
                        </Box>
                      }
                      sx={{ py: 0.5 }} // Reduced padding
                    >
                      <ListItemIcon sx={{ minWidth: 32 }}> {/* Reduced icon margin */}
                        <Tooltip title={tool.category || 'Tool'}>
                            {getToolIcon(tool.category)}
                        </Tooltip>
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Typography variant="body2" component="span" sx={{ mr: 1 }}>{tool.name}</Typography>
                            {isAdded && (
                              <Chip label="Added" color="primary" size="small" variant="outlined"/>
                            )}
                          </Box>
                        }
                        secondary={
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', whiteSpace: 'normal' }}>
                                {tool.description}
                                {tool.requirements && ` (Requires: ${tool.requirements})`}
                            </Typography>
                        }
                        secondaryTypographyProps={{ component: 'div' }} // Allow secondary text to wrap
                      />
                    </ListItem>
                  );
                })}
                 {displayedTools.length === 0 && (
                    <ListItem>
                        <ListItemText primary="No tools match your search in this category." sx={{ textAlign: 'center', color: 'text.secondary' }}/>
                    </ListItem>
                 )}
              </List>
            </Box>
          </Paper>
        </Grid>

        {/* Right Panel: Selected Tools */}
        <Grid item xs={12} md={5}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Agent's Tools
              </Typography>
              <Tooltip title="Tools added here will be available for the agent to call based on its instructions.">
                <IconButton size="small">
                  <HelpOutlineIcon />
                </IconButton>
              </Tooltip>
            </Box>

            {activeTools.length === 0 ? (
              <Box sx={{ p: 3, textAlign: 'center', border: '1px dashed', borderColor: 'grey.300', borderRadius: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  No tools added yet. Add built-in or custom function tools from the left.
                </Typography>
              </Box>
            ) : (
              <List dense>
                {activeTools.map((tool, index) => (
                  <React.Fragment key={tool.id || index}>
                    {index > 0 && <Divider component="li" sx={{ my: 0.5 }}/>}
                    <ListItem
                      secondaryAction={
                        <Box>
                           {/* Edit button for custom functions */}
                           {tool.category === 'Function' && (
                               <Tooltip title="Edit Custom Tool Code">
                                   <IconButton edge="end" size="small" sx={{ mr: 0.5 }} onClick={() => handleOpenToolDialog(tool)}>
                                       <EditIcon fontSize="small" />
                                   </IconButton>
                               </Tooltip>
                           )}
                           <Tooltip title="Remove this tool">
                                <IconButton edge="end" aria-label="remove" color="error" size="small" onClick={() => handleRemoveTool(tool.id)}>
                                <DeleteIcon fontSize="small"/>
                                </IconButton>
                           </Tooltip>
                        </Box>
                      }
                       sx={{ py: 0.5 }}
                    >
                      <ListItemIcon sx={{ minWidth: 32 }}>
                         <Tooltip title={tool.category || 'Tool'}>
                            {getToolIcon(tool.category)}
                         </Tooltip>
                      </ListItemIcon>
                      <ListItemText
                        primary={<Typography variant="body2">{tool.name}</Typography>}
                        secondary={
                            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', whiteSpace: 'normal' }}>
                                {tool.description || tool.category}
                                {tool.id === 'vertexaisearch' && ` (Datastore: ${tool.parameters?.data_store_id?.value || 'Not Set'})`}
                            </Typography>
                        }
                         secondaryTypographyProps={{ component: 'div' }}
                      />
                    </ListItem>
                  </React.Fragment>
                ))}
              </List>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Dialog for Custom Tool Code */}
      <Dialog
        open={openDialog}
        onClose={handleCloseToolDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {isEditing ? `Edit Custom Tool: ${currentToolConfig?.name}` : 'Create New Custom Function Tool'}
        </DialogTitle>
        <DialogContent dividers>
           {/* Add fields for name and description if creating new */}
           {!isEditing && (
               <Grid container spacing={2} sx={{ mb: 2 }}>
                   <Grid item xs={6}>
                       <TextField
                           fullWidth
                           required
                           label="Function Name"
                           value={currentToolConfig?.name || ''}
                           onChange={(e) => setCurrentToolConfig(prev => ({...prev, name: e.target.value.replace(/\s+/g, '_')}))} // Basic sanitization
                           helperText="Python function name (e.g., my_tool_name)."
                       />
                   </Grid>
                    <Grid item xs={6}>
                       <TextField
                           fullWidth
                           required
                           label="Tool Description (for LLM)"
                           value={currentToolConfig?.description || ''}
                           onChange={(e) => setCurrentToolConfig(prev => ({...prev, description: e.target.value}))}
                            helperText="How the agent understands when to use this tool."
                       />
                   </Grid>
               </Grid>
           )}
          <Typography variant="body2" paragraph>
            Define the Python function below. Use type hints for parameters and the return value. The docstring will be used as the tool's description for the agent.
          </Typography>
          <Box sx={{ height: 400, border: '1px solid', borderColor: 'divider' }}>
            <Editor
              height="100%"
              language="python"
              theme={theme.palette.mode === 'dark' ? 'vs-dark' : 'light'}
              value={customToolCode}
              onChange={(value) => setCustomToolCode(value || '')}
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
          <Button onClick={handleCloseToolDialog}>
            Cancel
          </Button>
          <Button
            onClick={handleSaveCustomTool}
            variant="contained"
            disabled={!customToolCode.trim() || (!isEditing && (!currentToolConfig?.name || !currentToolConfig?.description))} // Basic validation
          >
            {isEditing ? 'Save Changes' : 'Add Custom Tool'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default ToolsSelector;