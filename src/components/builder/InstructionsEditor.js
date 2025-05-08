// src/components/builder/InstructionsEditor.js
import React, { useState } from 'react';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid'; // Added Grid import
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Chip from '@mui/material/Chip';
import Alert from '@mui/material/Alert';
import Editor from '@monaco-editor/react';
import CodeIcon from '@mui/icons-material/Code';
import TextFormatIcon from '@mui/icons-material/TextFormat';
import FormatListBulletedIcon from '@mui/icons-material/FormatListBulleted';
import Tooltip from '@mui/material/Tooltip'; // Added Tooltip
import { useTheme } from '@mui/material/styles'; // Ensure useTheme is imported

// Example templates updated for Google context
const instructionTemplates = [
  {
    id: 'google-cloud-support',
    name: 'Google Cloud Support',
    description: 'Handle inquiries about Google Cloud services and billing.',
    content: `You are a helpful assistant for Google Cloud Platform users. Your primary goal is to answer questions accurately regarding GCP services, billing, and basic troubleshooting.

Key Responsibilities:
1.  Provide clear explanations of GCP services (Compute Engine, Cloud Storage, BigQuery, etc.).
2.  Assist with understanding billing concepts and potential cost savings.
3.  Guide users towards relevant Google Cloud documentation or support channels for complex issues.
4.  Use available tools (like Google Search via google_search tool) to find up-to-date information when necessary.
5.  Maintain a professional, helpful, and patient tone.
6.  If the query involves specific BQML tasks, route to the 'bqml_agent'.
7.  If the query requires direct database interaction (SQL generation), route to the 'database_agent'.

Constraints:
- Do not provide specific security recommendations. Guide users to Security Command Center documentation.
- Do not ask for or handle sensitive credentials (API keys, passwords).
- Clearly state when information might be outdated and recommend checking official docs.
- Be precise. If the user asks for a dataset, provide the name. Don't call any additional agent if not absolutely necessary.`
  },
  {
    id: 'data-analysis-bq',
    name: 'Data Analysis (BigQuery)',
    description: 'Analyze data in BigQuery, provide insights, and generate Python code for plots.',
    content: `You are a data analyst assistant specializing in Google BigQuery. Your task is to help users understand their data stored in BigQuery, perform analysis, and visualize results.

Follow these guidelines:
1.  Understand the user's goal for data analysis. Ask clarifying questions about tables, columns, and desired outcomes.
2.  Use the 'database_agent' tool (call_db_agent) to query BigQuery for necessary data. Formulate precise questions for the database agent.
3.  Once data is retrieved, use your Python execution capabilities (via code interpreter extension) to perform analysis (e.g., calculate statistics, group data).
4.  Generate Python code (using libraries like Matplotlib or Seaborn) to create relevant plots based on the analysis. Ensure the code is executable and clearly commented.
5.  Present findings clearly, summarizing key insights from the data and visualizations.
6.  Explain the steps taken in your analysis.
7.  If the user asks for machine learning tasks within BigQuery (e.g., training a model), route the request to the 'bqml_agent'.

Important:
- Always prioritize using the provided tools ('call_db_agent', 'call_bqml_agent') for data retrieval or BQML tasks.
- Generate Python code for analysis and visualization *after* retrieving data.
- Ensure generated Python code is safe and relevant to the user's request.`
  },
  // Add more relevant templates if needed
];

function InstructionsEditor({ agentData, updateAgentData }) {
  const theme = useTheme(); // Get theme using the hook
  const [editorMode, setEditorMode] = useState('text');
  const [showTemplates, setShowTemplates] = useState(false);

  const handleInstructionsChange = (value) => {
    updateAgentData('instructions', value || ''); // Ensure value is not null/undefined
  };

  const applyTemplate = (template) => {
    updateAgentData('instructions', template.content);
    setShowTemplates(false); // Hide templates after applying one
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Agent Instructions (System Prompt)
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Define the agent's persona, capabilities, limitations, and how it should respond. This guides the underlying Google AI model's behavior.
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        Provide clear, specific instructions. Mention available tools (e.g., `google_search`, `call_db_agent`) and sub-agents (e.g., `bqml_agent`) and when to use them. Define the desired tone and constraints.
      </Alert>

      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Tooltip title="Toggle between simple text editing and code/markdown view">
          <ToggleButtonGroup
            value={editorMode}
            exclusive
            onChange={(_, newMode) => newMode && setEditorMode(newMode)}
            size="small"
            aria-label="Editor Mode"
          >
            <ToggleButton value="text" aria-label="Text Mode">
              <TextFormatIcon sx={{ mr: 0.5 }} />
              Text
            </ToggleButton>
            <ToggleButton value="code" aria-label="Code/Markdown Mode">
              <CodeIcon sx={{ mr: 0.5 }} />
              Code/Markdown
            </ToggleButton>
          </ToggleButtonGroup>
        </Tooltip>

        <Chip
          icon={<FormatListBulletedIcon />}
          label="Use Template"
          clickable
          color={showTemplates ? "primary" : "default"}
          onClick={() => setShowTemplates(!showTemplates)}
          aria-controls={showTemplates ? 'template-list' : undefined}
          aria-expanded={showTemplates}
        />
      </Box>

      {/* Template Selection Area */}
      {showTemplates && (
        <Box sx={{ mb: 3, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }} id="template-list">
          <Typography variant="subtitle2" gutterBottom>
            Select an Instruction Template:
          </Typography>
          <Grid container spacing={2}>
            {instructionTemplates.map((template) => (
              <Grid item xs={12} sm={6} md={4} key={template.id}>
                <Paper
                  elevation={1}
                  sx={{
                    p: 1.5,
                    cursor: 'pointer',
                    height: '100%',
                    '&:hover': {
                      boxShadow: 3,
                      borderColor: 'primary.main',
                    },
                    border: '1px solid transparent',
                    display: 'flex',
                    flexDirection: 'column'
                  }}
                  onClick={() => applyTemplate(template)}
                  tabIndex={0} // Make it focusable
                  onKeyPress={(e) => e.key === 'Enter' && applyTemplate(template)} // Allow keyboard activation
                >
                  <Typography variant="subtitle2" gutterBottom>{template.name}</Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ flexGrow: 1 }}>
                    {template.description}
                  </Typography>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* Editor Area */}
      <Box sx={{ height: 400, border: 1, borderColor: 'divider', borderRadius: 1, overflow: 'hidden' }}>
        {editorMode === 'text' ? (
          <TextField
            fullWidth
            multiline
            value={agentData.instructions || ''} // Ensure controlled component
            onChange={(e) => handleInstructionsChange(e.target.value)}
            placeholder="Enter detailed instructions for your agent..."
            variant="outlined"
            minRows={15} // Ensure minimum height
            sx={{
              height: '100%',
              '& .MuiOutlinedInput-root': {
                height: '100%',
                alignItems: 'flex-start',
                p: 1.5 // Add padding inside text field
              },
              '& .MuiOutlinedInput-notchedOutline': {
                border: 'none' // Remove inner border
              },
              '& .MuiInputBase-inputMultiline': { // Target multiline input specifically
                height: 'calc(100% - 24px)', // Adjust height considering padding
                overflowY: 'auto',
                fontSize: '0.9rem', // Slightly smaller font for more text
                lineHeight: 1.5
              }
            }}
            aria-label="Agent Instructions Text Editor"
          />
        ) : (
          <Editor
            height="100%" // Monaco editor takes height prop directly
            language="markdown" // Use markdown for better formatting potential
            theme={theme.palette.mode === 'dark' ? 'vs-dark' : 'light'} // Use theme from useTheme hook
            value={agentData.instructions || ''}
            onChange={handleInstructionsChange}
            options={{
              minimap: { enabled: false },
              lineNumbers: 'on',
              wordWrap: 'on',
              wrappingIndent: 'indent',
              scrollBeyondLastLine: false,
              fontSize: 13, // Consistent font size
              tabSize: 2,
              insertSpaces: true,
            }}
            aria-label="Agent Instructions Code/Markdown Editor"
          />
        )}
      </Box>
       <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
         Tip: Use Markdown for formatting lists, code blocks, etc., even in Text mode. Switch to Code/Markdown view for syntax highlighting.
       </Typography>
    </Box>
  );
}

export default InstructionsEditor;