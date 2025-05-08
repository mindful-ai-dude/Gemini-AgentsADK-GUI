// src/components/builder/CodePreview.js
import React, { useState, useEffect, useMemo } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Alert from '@mui/material/Alert';
import Editor from '@monaco-editor/react';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import CodeIcon from '@mui/icons-material/Code';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { ThemeProvider, useTheme } from '@mui/material/styles'; // For editor theme

// Import updated Google ADK code templates
import {
  agentTemplate,
  functionToolTemplate, // Keep for potential display if needed
  structuredOutputTemplate, // Keep for potential display if needed
  guardrailTemplate, // Keep for potential display if needed
  runnerTemplate,
  streamingTemplate,
  fullToolExampleTemplate // Keep for potential display if needed
} from '../../utils/codeTemplates';

// Helper to generate tool import statements
const generateToolImports = (tools) => {
    let imports = new Set(); // Use Set to avoid duplicates
    (tools || []).forEach(tool => {
        if (tool.id === 'googlesearch' || tool.name === 'google_search') {
            imports.add('from google.adk.tools import google_search');
        }
        // Add imports for other known built-in tools here
        // else if (tool.id === 'filesearch') {
        //     imports.add('from google.adk.tools import FileSearchTool'); // Hypothetical
        // }
        // Custom tools might be defined inline or imported from elsewhere (handled below)
    });
    return Array.from(imports).join('\n');
};

// Helper to generate custom tool definitions
const generateCustomToolDefs = (tools) => {
    return (tools || [])
        .filter(tool => tool.category === 'Function' && tool.code) // Only include functions with code
        .map(tool => `\n# --- Custom Tool: ${tool.name} ---\n${tool.code}\n# -------------------------------\n`)
        .join('\n');
};

// Helper to generate tool instantiation list
const generateToolInstantiation = (tools) => {
    return (tools || [])
        .map(tool => {
            if (tool.id === 'googlesearch' || tool.name === 'google_search') {
                return '    google_search(),'; // Assuming simple instantiation
            }
            // Add instantiation for other built-in tools
            // else if (tool.id === 'filesearch') {
            //     return `    FileSearchTool(...),`; // Add parameters if needed
            // }
            else if (tool.category === 'Function') {
                // Assume the function name is the variable name after definition
                return `    ${tool.name},`;
            }
            return `    # Tool not recognized: ${tool.name || tool.id}`;
        })
        .join('\n');
};

// Helper to generate sub-agent instantiation (Placeholder)
const generateSubAgentInstantiation = (handoffs) => {
     return (handoffs || [])
        .map(handoff => `    # ${handoff.targetAgentName || handoff.id}, # Requires import and instantiation`)
        .join('\n');
};


function CodePreview({ agentData }) {
  const theme = useTheme();
  const [tabValue, setTabValue] = useState(0);
  const [copied, setCopied] = useState(false);

  // Memoize code generation to avoid re-computation on every render
  const generatedCode = useMemo(() => {
    const toolImports = generateToolImports(agentData.tools);
    const customToolDefs = generateCustomToolDefs(agentData.tools);
    const toolInstantiation = generateToolInstantiation(agentData.tools);
    const subAgentInstantiation = generateSubAgentInstantiation(agentData.handoffs); // Placeholder

    // Populate Agent Template
    let agentCode = agentTemplate
      .replace('{{name}}', agentData.name || 'MyGeminiAgent')
      .replace('"""{{instructions}}"""', `"""${agentData.instructions || 'You are a helpful assistant.'}"""`) // Ensure multiline instructions work
      .replace('{{model}}', agentData.model || 'gemini-1.5-flash')
      .replace('{{tool_imports}}', toolImports)
      .replace('{{custom_tool_definitions}}', customToolDefs)
      .replace('{{tools_instantiation}}', toolInstantiation || '# No tools configured')
      .replace('{{sub_agents_instantiation}}', subAgentInstantiation || '# No sub-agents configured')
      // Inject model settings (example - adapt based on ADK Agent constructor)
      .replace('{{temperature}}', agentData.modelSettings?.temperature ?? modelSettingsDefaults.temperature)
      .replace('{{topP}}', agentData.modelSettings?.topP ?? modelSettingsDefaults.topP)
      .replace('{{topK}}', agentData.modelSettings?.topK ?? modelSettingsDefaults.topK)
      .replace('{{maxOutputTokens}}', agentData.modelSettings?.maxOutputTokens ?? modelSettingsDefaults.maxOutputTokens)
      // Inject current date (example)
      .replace('{{current_date}}', new Date().toISOString().split('T')[0]);


    // Populate Runner Template
    const runnerCode = runnerTemplate.replace('{{prompt}}', 'What can you tell me about Google Cloud Storage?');

    // Populate Streaming Template
    const streamingCode = streamingTemplate.replace('{{prompt}}', 'Summarize the main points about Google App Engine.');

    return { agentCode, runnerCode, streamingCode };
  }, [agentData]); // Recalculate only when agentData changes

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    setCopied(false); // Reset copied state on tab change
  };

  const handleCopyCode = () => {
    let codeToCopy;
    switch (tabValue) {
      case 0: codeToCopy = generatedCode.agentCode; break;
      case 1: codeToCopy = generatedCode.runnerCode; break;
      case 2: codeToCopy = generatedCode.streamingCode; break;
      default: codeToCopy = generatedCode.agentCode;
    }
    navigator.clipboard.writeText(codeToCopy).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(err => console.error('Failed to copy code: ', err));
  };

  const handleDownloadCode = () => {
    let codeToDownload;
    let filename;
    switch (tabValue) {
      case 0:
        codeToDownload = generatedCode.agentCode;
        filename = `${agentData.name.replace(/\s+/g, '_') || 'agent'}.py`;
        break;
      case 1:
        codeToDownload = generatedCode.runnerCode;
        filename = 'run_agent_example.py';
        break;
      case 2:
        codeToDownload = generatedCode.streamingCode;
        filename = 'stream_agent_example.py';
        break;
      default:
        codeToDownload = generatedCode.agentCode;
        filename = `${agentData.name.replace(/\s+/g, '_') || 'agent'}.py`;
    }

    const element = document.createElement('a');
    const file = new Blob([codeToDownload], { type: 'text/plain;charset=utf-8' });
    element.href = URL.createObjectURL(file);
    element.download = filename;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
    URL.revokeObjectURL(element.href); // Clean up blob URL
  };

  // Function to get the code for the current tab
  const getCurrentCode = () => {
      switch (tabValue) {
          case 0: return generatedCode.agentCode;
          case 1: return generatedCode.runnerCode;
          case 2: return generatedCode.streamingCode;
          default: return '';
      }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Generated Python Code (Google ADK)
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Review the generated Python code based on your agent configuration. This code uses the Google Agent Development Kit (ADK).
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        This code requires the `google-adk` Python package and appropriate Google Cloud authentication (API Key or Service Account credentials) configured in the execution environment. The runner/streaming examples are illustrative and depend on the final ADK API.
      </Alert>

      <Paper sx={{ mb: 3 }} elevation={3}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="Generated Code Tabs">
            <Tab label="Agent Definition" icon={<CodeIcon />} iconPosition="start" id="code-tab-0" aria-controls="code-tabpanel-0"/>
            <Tab label="Runner Example" icon={<PlayArrowIcon />} iconPosition="start" id="code-tab-1" aria-controls="code-tabpanel-1"/>
            <Tab label="Streaming Example" icon={<PlayArrowIcon />} iconPosition="start" id="code-tab-2" aria-controls="code-tabpanel-2"/>
          </Tabs>
        </Box>

        {/* Use a single Editor instance and update its value */}
        <Box sx={{ p: 0 }} role="tabpanel" hidden={false} id={`code-tabpanel-${tabValue}`} aria-labelledby={`code-tab-${tabValue}`}>
           <Editor
             height="400px"
             language="python"
             theme={theme.palette.mode === 'dark' ? 'vs-dark' : 'light'}
             value={getCurrentCode()} // Dynamically set value based on tab
             options={{
               readOnly: true,
               minimap: { enabled: true }, // Enable minimap for potentially long code
               scrollBeyondLastLine: false,
               fontSize: 13,
               wordWrap: 'on', // Enable word wrap
             }}
           />
        </Box>

        <Divider />

        <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Stack direction="row" spacing={1}>
            <Chip label="Python" color="primary" variant="outlined" size="small" />
            <Chip label="Google ADK" variant="outlined" size="small" />
          </Stack>

          <Box>
            <Button
              startIcon={copied ? <CheckCircleOutlineIcon /> : <ContentCopyIcon />}
              onClick={handleCopyCode}
              color={copied ? "success" : "primary"}
              sx={{ mr: 1 }}
              disabled={!getCurrentCode()} // Disable if no code
            >
              {copied ? "Copied!" : "Copy Code"}
            </Button>

            <Button
              startIcon={<FileDownloadIcon />}
              variant="outlined"
              onClick={handleDownloadCode}
              disabled={!getCurrentCode()} // Disable if no code
            >
              Download
            </Button>
          </Box>
        </Box>
      </Paper>

      <Typography variant="subtitle1" gutterBottom>
        Next Steps
      </Typography>
      <Typography variant="body2" paragraph>
        Use the 'Test Agent Config' tab to interact with your agent configuration via the backend service. You can also download the generated code to run it directly in a Python environment with the Google ADK installed and credentials configured.
      </Typography>
    </Box>
  );
}

export default CodePreview;