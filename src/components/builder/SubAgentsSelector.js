// src/components/builder/SubAgentsSelector.js

import React, { useState } from 'react'; // Removed useEffect
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
import Alert from '@mui/material/Alert';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import AccountTreeIcon from '@mui/icons-material/AccountTree'; // Icon for sub-agents
// import PersonIcon from '@mui/icons-material/Person'; // Removed unused import
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

// --- Sample Available Agent Configurations (Simulated) ---
// In a real system, these might be fetched from a registry or defined elsewhere.
// These represent other agent *configurations* that this agent can potentially call *as tools*.
// The distinction between "sub-agent" (delegation) and "agent-as-tool" depends on ADK implementation.
// We'll treat them as potential tools for now.
const sampleAvailableAgentConfigs = [
  {
    id: 'bqml_agent_config_1', // Unique ID for the config
    name: 'BQML Forecasting Agent', // Name of the agent config
    description: 'Trains forecasting models using BigQuery ML ARIMA_PLUS.',
    model: 'gemini-1.5-flash', // Model used by this agent config
    category: 'Data Science',
    // Define the tool interface this agent exposes
    toolName: 'call_bqml_agent',
    toolDescription: 'Use for BigQuery ML specific tasks like model training or forecasting.'
  },
  {
    id: 'db_agent_config_1',
    name: 'Database Query Agent (NL2SQL)',
    description: 'Translates natural language to SQL and queries BigQuery.',
    model: 'gemini-1.5-pro', // Might use a more powerful model for NL2SQL
    category: 'Database',
    toolName: 'call_db_agent',
    toolDescription: 'Use to query the database using natural language.'
  },
  // Add other potentially callable agent configurations
];

function SubAgentsSelector({ agentData, updateAgentData }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [openDialog, setOpenDialog] = useState(false);
  // State for creating a *reference* to another agent config as a tool
  const [newAgentToolRef, setNewAgentToolRef] = useState({
    id: '', // Will be the ID of the *target* agent config
    name: '', // Will be the *tool name* used to call the target agent
    description: '', // Description of *why* this agent calls the other
    targetAgentName: '', // Display name of the target agent
    targetAgentModel: '', // Model of the target agent (for info)
    category: 'Agent-as-Tool' // Indicate this is a reference
  });

  // 'handoffs' in agentData now conceptually stores agent-as-tool references
  const activeAgentTools = agentData.handoffs || [];

  const handleAddAgentAsTool = (targetAgentConfig) => {
    // Create the tool reference based on the selected agent config
    const agentToolReference = {
      id: targetAgentConfig.id, // Reference the target agent config ID
      name: targetAgentConfig.toolName, // The name used to *call* this agent
      description: targetAgentConfig.toolDescription, // How the parent agent should use it
      targetAgentName: targetAgentConfig.name, // Store for display
      targetAgentModel: targetAgentConfig.model, // Store for display
      category: 'Agent-as-Tool'
    };

    const updatedAgentTools = [...activeAgentTools, agentToolReference];
    updateAgentData('handoffs', updatedAgentTools); // Update the 'handoffs' field
  };

  const handleRemoveAgentTool = (agentToolId) => {
    // Filter based on the referenced agent ID ('id' field in our structure)
    const updatedAgentTools = activeAgentTools.filter(tool => tool.id !== agentToolId);
    updateAgentData('handoffs', updatedAgentTools);
  };

  // Dialog for manually defining an agent-as-tool reference (less common)
  const handleOpenNewDialog = () => {
     setNewAgentToolRef({
       id: '', // User needs to provide target agent ID
       name: '', // User defines the call name
       description: '',
       targetAgentName: '',
       targetAgentModel: '',
       category: 'Agent-as-Tool'
     });
     setOpenDialog(true);
   };

  const handleCloseDialog = () => {
    setOpenDialog(false);
  };

  const handleSaveManualAgentToolRef = () => {
    // Basic validation for manual entry
    if (newAgentToolRef.id && newAgentToolRef.name && newAgentToolRef.targetAgentName) {
      const updatedAgentTools = [...activeAgentTools, { ...newAgentToolRef }];
      updateAgentData('handoffs', updatedAgentTools);
      setOpenDialog(false);
    } else {
      alert("Please provide Target Agent ID, Tool Name, and Target Agent Name.");
    }
  };


  const filteredAvailableAgents = sampleAvailableAgentConfigs.filter(
    agentConfig => agentConfig.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                   agentConfig.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Group available agents by category
  const groupedAvailableAgents = filteredAvailableAgents.reduce((acc, agentConfig) => {
    const category = agentConfig.category || 'Other';
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(agentConfig);
    return acc;
  }, {});

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Agent Tools & Sub-Agents {/* UPDATED Title */}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure other agents that this agent can call as tools or delegate tasks to (sub-agents). This enables multi-agent workflows. The exact mechanism (tool vs. sub-agent) depends on the ADK implementation.
      </Typography>

      <Grid container spacing={3}>
        {/* Left Panel: Available Agent Configs to add as Tools */}
        <Grid item xs={12} md={7}>
          <Paper sx={{ height: '100%', p: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <TextField
                size="small"
                placeholder="Search available agents..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                sx={{ flexGrow: 1, mr: 2 }}
              />
              <Tooltip title="Manually define a reference to another agent as a tool (Advanced)">
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={handleOpenNewDialog}
                  size="small"
                >
                  Manual Ref
                </Button>
              </Tooltip>
            </Box>

            {Object.entries(groupedAvailableAgents).map(([category, agentConfigs]) => (
              <Box key={category} sx={{ mb: 3 }}>
                <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
                  {category} Agents
                </Typography>
                <List sx={{ bgcolor: 'background.paper', borderRadius: 1 }}>
                  {agentConfigs.map((agentConfig) => {
                    const isAdded = activeAgentTools.some(h => h.id === agentConfig.id);

                    return (
                      <ListItem
                        key={agentConfig.id}
                        className="tool-item" // Reuse styling
                        secondaryAction={
                          isAdded ? (
                            <IconButton
                              edge="end"
                              aria-label="remove"
                              color="error"
                              onClick={() => handleRemoveAgentTool(agentConfig.id)}
                              title="Remove this agent tool reference"
                            >
                              <DeleteIcon />
                            </IconButton>
                          ) : (
                            <Button
                              variant="outlined"
                              size="small"
                              onClick={() => handleAddAgentAsTool(agentConfig)}
                              title={`Add ${agentConfig.name} as a callable tool (${agentConfig.toolName})`}
                            >
                              Add as Tool
                            </Button>
                          )
                        }
                      >
                        <ListItemIcon>
                          {/* Use a different icon for agent references */}
                          <AccountTreeIcon color="action" />
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              {agentConfig.name}
                              {isAdded && (
                                <Chip
                                  label="Added as Tool"
                                  color="primary"
                                  size="small"
                                  sx={{ ml: 1 }}
                                />
                              )}
                            </Box>
                          }
                          secondary={
                            <>
                              {agentConfig.description}
                              <Chip
                                label={agentConfig.model}
                                size="small"
                                variant="outlined"
                                sx={{ ml: 1, mt: 0.5 }}
                                title={`Model used: ${agentConfig.model}`}
                              />
                               <Chip
                                label={`Callable as: ${agentConfig.toolName}`}
                                size="small"
                                variant="outlined"
                                color="secondary"
                                sx={{ ml: 1, mt: 0.5 }}
                                title={agentConfig.toolDescription}
                              />
                            </>
                          }
                        />
                      </ListItem>
                    );
                  })}
                </List>
              </Box>
            ))}

            {Object.keys(groupedAvailableAgents).length === 0 && (
              <Box sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  No available agent configurations found matching your search, or none are defined as samples.
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Right Panel: Active Agent Tool References */}
        <Grid item xs={12} md={5}>
          <Paper sx={{ p: 2, height: '100%' }}>
             <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Configured Agent Tools {/* UPDATED Title */}
                </Typography>
                 <Tooltip title="These are references to other agents that this agent can invoke, typically via a specific tool name.">
                    <IconButton size="small">
                    <HelpOutlineIcon />
                    </IconButton>
                </Tooltip>
            </Box>

            {activeAgentTools.length === 0 ? (
              <Box
                sx={{
                  p: 3,
                  textAlign: 'center',
                  border: '1px dashed',
                  borderColor: 'grey.300',
                  borderRadius: 1
                }}
              >
                <Typography variant="body2" color="text.secondary">
                  No other agents configured to be called yet. Add agent references from the left panel.
                </Typography>
              </Box>
            ) : (
              <>
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography variant="body2">
                    This agent can call the following agents using their specified tool name. Ensure the target agents are running and accessible by the backend.
                  </Typography>
                </Alert>

                <List>
                  {activeAgentTools.map((agentToolRef, index) => (
                    <React.Fragment key={agentToolRef.id || index}> {/* Use index as fallback key */}
                      {index > 0 && <Divider />}
                      <ListItem
                        secondaryAction={
                          <IconButton
                            edge="end"
                            aria-label="remove"
                            color="error"
                            onClick={() => handleRemoveAgentTool(agentToolRef.id)}
                            title="Remove this reference"
                          >
                            <DeleteIcon />
                          </IconButton>
                        }
                      >
                        <ListItemIcon>
                          <AccountTreeIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText
                          primary={agentToolRef.targetAgentName || agentToolRef.id} // Show target name or ID
                          secondary={`Callable via tool: ${agentToolRef.name}`} // Show the tool name used for calling
                          title={agentToolRef.description} // Show description on hover
                        />
                      </ListItem>
                    </React.Fragment>
                  ))}
                </List>
              </>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Dialog for Manual Agent-as-Tool Reference */}
       <Dialog
         open={openDialog}
         onClose={handleCloseDialog}
         maxWidth="sm"
         fullWidth
       >
         <DialogTitle>
           Manually Add Agent-as-Tool Reference
         </DialogTitle>
         <DialogContent dividers>
           <Typography variant="body2" color="text.secondary" paragraph>
             Define how this agent can call another agent. You need the target agent's unique ID (from its configuration) and a tool name to use for invocation.
           </Typography>
           <Grid container spacing={2} sx={{ mt: 1 }}>
             <Grid item xs={12}>
               <TextField
                 fullWidth
                 label="Target Agent Config ID"
                 value={newAgentToolRef.id}
                 onChange={(e) => setNewAgentToolRef({...newAgentToolRef, id: e.target.value})}
                 helperText="The unique ID of the agent configuration to call."
                 required
               />
             </Grid>
              <Grid item xs={12}>
               <TextField
                 fullWidth
                 label="Target Agent Name (for display)"
                 value={newAgentToolRef.targetAgentName}
                 onChange={(e) => setNewAgentToolRef({...newAgentToolRef, targetAgentName: e.target.value})}
                 helperText="The display name of the agent being called."
                 required
               />
             </Grid>
             <Grid item xs={12}>
               <TextField
                 fullWidth
                 label="Tool Name (for calling)"
                 value={newAgentToolRef.name}
                 onChange={(e) => setNewAgentToolRef({...newAgentToolRef, name: e.target.value})}
                 helperText="The name this agent will use to invoke the target agent (e.g., call_billing_agent)."
                 required
               />
             </Grid>
             <Grid item xs={12}>
               <TextField
                 fullWidth
                 label="Tool Description (for parent agent)"
                 value={newAgentToolRef.description}
                 onChange={(e) => setNewAgentToolRef({...newAgentToolRef, description: e.target.value})}
                 helperText="Describe when the parent agent should use this tool to call the target agent."
                 multiline
                 rows={2}
               />
             </Grid>
           </Grid>
         </DialogContent>
         <DialogActions>
           <Button onClick={handleCloseDialog}>
             Cancel
           </Button>
           <Button
             onClick={handleSaveManualAgentToolRef}
             variant="contained"
             disabled={!newAgentToolRef.id || !newAgentToolRef.name || !newAgentToolRef.targetAgentName}
           >
             Add Reference
           </Button>
         </DialogActions>
       </Dialog>
    </Box>
  );
}

export default SubAgentsSelector;