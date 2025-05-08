// src/components/builder/TestAgent.js
import React, { useState, useRef, useEffect } from 'react';
import apiService from '../../utils/apiService'; // Uses refactored service
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemAvatar from '@mui/material/ListItemAvatar'; // Use ListItemAvatar
import ListItemText from '@mui/material/ListItemText'; // Use ListItemText
import Avatar from '@mui/material/Avatar';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import SendIcon from '@mui/icons-material/Send';
import SmartToyOutlinedIcon from '@mui/icons-material/SmartToyOutlined'; // Agent icon
import PersonOutlineIcon from '@mui/icons-material/PersonOutline'; // User icon
import BuildIcon from '@mui/icons-material/Build'; // Tool icon
import CodeIcon from '@mui/icons-material/Code'; // Icon for code/structured output
import ReplayIcon from '@mui/icons-material/Replay';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { useTheme } from '@mui/material/styles'; // For theme access

function TestAgent({ agentData }) {
  const theme = useTheme();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null); // State for errors during run
  const messagesEndRef = useRef(null);
  const [conversationHistory, setConversationHistory] = useState([]); // Store backend interaction history if needed

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Clear chat and history when agentData changes (e.g., switching agents in builder)
   useEffect(() => {
       handleClearChat();
   // eslint-disable-next-line react-hooks/exhaustive-deps
   }, [agentData]);


  const handleSendMessage = async () => {
    if (!input.trim()) return;
    setError(null); // Clear previous errors

    const currentInput = input; // Capture input before clearing
    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: currentInput
    };

    // Add user message to UI immediately
    setMessages(prev => [...prev, userMessage]);
    setInput(''); // Clear input field
    setIsLoading(true);

    // Add user message to conversation history for backend (if needed by ADK)
    // const currentHistory = [...conversationHistory, { role: 'user', content: currentInput }];
    // setConversationHistory(currentHistory); // Update history state

    try {
      // Call the backend service to run the agent config
      // Pass the current agent configuration and the user input
      const response = await apiService.runAgent(agentData.id || 'temp', currentInput, agentData); // Pass full config if ID is missing (new/unsaved)

      console.log('Backend Response:', response);

      // Process the response from the backend
      processAgentResponse(response);

      // Add assistant response to conversation history (if needed)
      // if (response.final_output) {
      //    setConversationHistory(prev => [...prev, { role: 'assistant', content: response.final_output }]);
      // }

    } catch (runError) {
      console.error('Error running agent via backend:', runError);
      const errorMessage = {
        id: `error-${Date.now()}`,
        role: 'system', // Use system role for errors
        type: 'error',
        content: `Error: ${runError.message || 'Failed to get response from agent backend.'}`
      };
      setMessages(prev => [...prev, errorMessage]);
      setError(runError.message); // Set error state for potential display elsewhere
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Process the agent response received from the backend.
   */
  const processAgentResponse = (response) => {
    const backendMessages = [];

    // Process intermediate items (like tool calls/results) if backend provides them
    if (response.new_items && Array.isArray(response.new_items)) {
      response.new_items.forEach((item, index) => {
        const messageId = `${response.id}-item-${index}`;
        if (item.type === "tool_call_item" && item.raw_item) {
          backendMessages.push({
            id: messageId,
            role: 'assistant', // Tool calls originate from the assistant
            type: 'tool_call',
            tool: item.raw_item.name || 'Unknown Tool',
            content: `Using tool: ${item.raw_item.name}... (Args: ${item.raw_item.arguments || '{}'})`
          });
        } else if (item.type === "tool_call_output_item") {
           backendMessages.push({
            id: messageId,
            role: 'tool', // Role indicating tool output
            type: 'tool_result',
            tool: 'Tool', // We might not know which tool produced it without more context
            content: typeof item.output === 'string' ? item.output : JSON.stringify(item.output) // Handle string or object output
          });
        }
        // Add handling for other potential item types from ADK if needed
      });
    }

    // Add the final assistant response
    let finalContent = 'Agent did not provide a final text response.';
    if (response.final_output) {
       if (typeof response.final_output === 'string') {
           finalContent = response.final_output;
       } else {
           // If structured output, display as formatted JSON
           try {
               finalContent = "```json\n" + JSON.stringify(response.final_output, null, 2) + "\n```"; // Format as code block
           } catch {
               finalContent = String(response.final_output); // Fallback to string conversion
           }
       }
    } else if (response.output && typeof response.output === 'string') {
        // Fallback if final_output isn't present but output is
        finalContent = response.output;
    }


    backendMessages.push({
      id: `${response.id}-final`,
      role: 'assistant',
      content: finalContent
    });

    setMessages(prev => [...prev, ...backendMessages]);
  };

  const handleClearChat = () => {
    setMessages([]);
    setConversationHistory([]); // Clear history as well
    setError(null);
  };

  // Render messages with appropriate styling
  const renderMessage = (message) => {
    const isUser = message.role === 'user';
    const isAssistant = message.role === 'assistant';
    const isTool = message.role === 'tool';
    const isSystem = message.role === 'system';
    const isError = message.type === 'error';

    let avatarIcon = <PersonOutlineIcon />;
    let avatarBgColor = isUser ? theme.palette.primary.dark : theme.palette.grey[500]; // Default grey for system/tool
    let paperBgColor = isUser ? theme.palette.primary.main : theme.palette.background.paper;
    let textColor = isUser ? theme.palette.primary.contrastText : theme.palette.text.primary;
    let align = isUser ? 'flex-end' : 'flex-start';
    let borderRadius = isUser ? '12px 12px 0 12px' : '0 12px 12px 12px';

    if (isAssistant) {
      avatarIcon = <SmartToyOutlinedIcon />;
      avatarBgColor = theme.palette.secondary.main; // Use secondary color for assistant
      borderRadius = '0 12px 12px 12px';
       if (message.type === 'tool_call') {
           avatarIcon = <BuildIcon />;
           avatarBgColor = theme.palette.info.main;
           paperBgColor = theme.palette.mode === 'dark' ? theme.palette.grey[700] : theme.palette.grey[200];
           textColor = theme.palette.text.secondary;
       } else if (message.content?.startsWith('```json')) {
           avatarIcon = <CodeIcon />; // Icon for structured output
           avatarBgColor = theme.palette.success.main;
       }
    } else if (isTool) {
        avatarIcon = <BuildIcon />;
        avatarBgColor = theme.palette.info.dark;
        paperBgColor = theme.palette.mode === 'dark' ? theme.palette.grey : theme.palette.grey;
        textColor = theme.palette.text.secondary;
        borderRadius = '0 12px 12px 12px';
    } else if (isSystem) {
        // Center system messages/errors
        align = 'center';
        if (isError) {
            return ( // Render errors differently
                 <ListItem key={message.id} sx={{ justifyContent: 'center', my: 1 }}>
                    <Alert severity="error" sx={{ width: 'auto', maxWidth: '80%' }}>{message.content}</Alert>
                 </ListItem>
            );
        } else {
             // Simple system messages (like 'Processing...')
             return (
                 <ListItem key={message.id} sx={{ justifyContent: 'center', my: 0.5 }}>
                     <Chip label={message.content} size="small" />
                 </ListItem>
             );
        }
    }


    return (
      <ListItem key={message.id} sx={{ justifyContent: align, mb: 1 }}>
         {!isUser && (
             <ListItemAvatar sx={{ minWidth: 'auto', mr: 1.5 }}>
                <Avatar sx={{ bgcolor: avatarBgColor, width: 32, height: 32 }}>
                    {React.cloneElement(avatarIcon, { sx: { fontSize: 18 } })}
                </Avatar>
             </ListItemAvatar>
         )}
        <Paper
          elevation={1}
          sx={{
            p: 1.5, // Consistent padding
            bgcolor: paperBgColor,
            color: textColor,
            borderRadius: borderRadius,
            maxWidth: '80%',
            wordBreak: 'break-word', // Ensure long words wrap
            whiteSpace: 'pre-wrap' // Respect newlines and spaces in content
          }}
        >
          {/* Render content based on type */}
          {message.type === 'tool_call' ? (
              <Typography variant="body2" sx={{ fontStyle: 'italic', opacity: 0.8 }}>{message.content}</Typography>
          ) : message.type === 'tool_result' ? (
               <Box>
                   <Typography variant="caption" display="block" sx={{ fontWeight: 'bold', mb: 0.5 }}>Tool Result:</Typography>
                   <Typography variant="body2" component="pre" sx={{ fontFamily: 'monospace', fontSize: '0.8rem', bgcolor: theme.palette.action.hover, p: 1, borderRadius: 1, overflowX: 'auto' }}>
                       {message.content}
                   </Typography>
               </Box>
          ) : message.content?.startsWith('```json') ? (
              // Basic rendering for JSON code blocks
              <Typography variant="body2" component="pre" sx={{ fontFamily: 'monospace', fontSize: '0.8rem', bgcolor: theme.palette.action.hover, p: 1, borderRadius: 1, overflowX: 'auto' }}>
                  {message.content.replace(/```json\n|```/g, '')}
              </Typography>
          ) : (
              <Typography variant="body1">{message.content}</Typography>
          )}
        </Paper>
         {isUser && (
             <ListItemAvatar sx={{ minWidth: 'auto', ml: 1.5 }}>
                <Avatar sx={{ bgcolor: avatarBgColor, width: 32, height: 32 }}>
                    {React.cloneElement(avatarIcon, { sx: { fontSize: 18 } })}
                </Avatar>
             </ListItemAvatar>
         )}
      </ListItem>
    );
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Test Agent Configuration
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Interact with your agent configuration. Messages are sent to the backend service which runs the agent using Google ADK.
      </Typography>

      <Paper sx={{ mb: 3 }} elevation={3}>
        <Box sx={{ p: 1.5, bgcolor: 'primary.main', color: 'white', display: 'flex', alignItems: 'center', borderTopLeftRadius: theme.shape.borderRadius, borderTopRightRadius: theme.shape.borderRadius }}>
          <SmartToyOutlinedIcon sx={{ mr: 1 }} />
          <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
            {agentData.name || 'Agent Test'} - Chat via Backend
          </Typography>
          <Tooltip title="Clear Chat History">
            <IconButton
              size="small"
              color="inherit"
              onClick={handleClearChat}
              aria-label="clear chat"
            >
              <DeleteOutlineIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        <Box
          sx={{
            height: 450, // Increased height
            overflowY: 'auto',
            bgcolor: theme.palette.mode === 'dark' ? theme.palette.grey[900] : theme.palette.grey[100], // Adjusted background
            p: 2
          }}
        >
          {messages.length === 0 ? (
            <Box
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                color: 'text.secondary',
                textAlign: 'center'
              }}
            >
              <SmartToyOutlinedIcon sx={{ fontSize: 48, mb: 2, color: 'primary.light' }} />
              <Typography variant="body1" gutterBottom>
                Chat with your agent configuration
              </Typography>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Type a message below to start the interaction.
              </Typography>
              <Button
                variant="outlined"
                size="small"
                startIcon={<ReplayIcon />}
                onClick={() => setInput("What can you do?")}
              >
                Suggest Prompt
              </Button>
            </Box>
          ) : (
            <List sx={{ p: 0 }}>
              {messages.map(message => renderMessage(message))}
              {/* Loading indicator at the end */}
              {isLoading && (
                  <ListItem sx={{ justifyContent: 'flex-start' }}>
                      <ListItemAvatar sx={{ minWidth: 'auto', mr: 1.5 }}>
                          <Avatar sx={{ bgcolor: theme.palette.secondary.main, width: 32, height: 32 }}>
                              <CircularProgress size={18} color="inherit" />
                          </Avatar>
                      </ListItemAvatar>
                      <Paper elevation={1} sx={{ p: 1, borderRadius: '0 12px 12px 12px', bgcolor: 'background.paper', fontStyle: 'italic', opacity: 0.7 }}>
                          <Typography variant="body2">Agent is thinking...</Typography>
                      </Paper>
                  </ListItem>
              )}
              <div ref={messagesEndRef} />
            </List>
          )}
        </Box>

        <Divider />

        <Box sx={{ p: 2, display: 'flex', alignItems: 'center' }}>
          <TextField
            fullWidth
            placeholder="Type your message..."
            variant="outlined"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              // Send on Enter key press, unless Shift is held
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // Prevent default newline behavior
                handleSendMessage();
              }
            }}
            disabled={isLoading}
            size="small"
            sx={{ mr: 1 }}
            multiline // Allow multiline input
            maxRows={3} // Limit expansion
            aria-label="User input message"
          />
          <Tooltip title="Send Message">
             <span> {/* Span needed for tooltip when button is disabled */}
                <Button
                    variant="contained"
                    color="primary"
                    onClick={handleSendMessage}
                    disabled={!input.trim() || isLoading}
                    sx={{ minWidth: '48px', height: '40px', p: 1 }} // Fixed size
                    aria-label="Send message"
                >
                    {isLoading ? <CircularProgress size={24} color="inherit"/> : <SendIcon />}
                </Button>
             </span>
          </Tooltip>
        </Box>
      </Paper>

      <Alert severity="warning">
        <Typography variant="body2">
          This test environment sends your agent configuration and input to the Flask backend for execution with the Google ADK. Ensure the backend server (`python app.py`) is running.
        </Typography>
      </Alert>
    </Box>
  );
}

export default TestAgent;