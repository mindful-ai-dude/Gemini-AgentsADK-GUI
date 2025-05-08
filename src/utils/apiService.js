// src/utils/apiService.js

// Local storage keys remain the same for managing configurations client-side
const STORAGE_KEY_AGENTS = 'google_adk_agents';
const STORAGE_KEY_API_KEY = 'google_api_key';
const BACKEND_URL = 'http://localhost:5001'; // URL of the Flask backend

// Import model options for defaults
import { modelOptions, modelSettingsDefaults } from './modelOptions';

class ApiService {
  constructor() {
    if (!localStorage.getItem(STORAGE_KEY_AGENTS)) {
      localStorage.setItem(STORAGE_KEY_AGENTS, JSON.stringify([]));
    }
    // No client-side API client needed now, backend handles it.
  }

  /**
   * Validates a Google API key by calling the backend endpoint.
   * @param {string} apiKey - The API key to validate
   * @returns {Promise<boolean>} - Whether the key is valid
   */
  async validateApiKey(apiKey) {
    console.log("Calling backend to validate API Key...");
    try {
      const response = await fetch(`${BACKEND_URL}/api/validate-key`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ apiKey: apiKey }),
      });

      const result = await response.json();

      if (!response.ok) {
        // Throw error with message from backend if available
        throw new Error(result.error || `HTTP error! status: ${response.status}`);
      }

      if (result.success) {
        console.log("Backend validation successful:", result.message);
        // Save the validated key locally
        localStorage.setItem(STORAGE_KEY_API_KEY, apiKey);
        return true;
      } else {
        // Should have been caught by !response.ok, but handle just in case
        throw new Error(result.error || 'Validation failed on backend.');
      }
    } catch (error) {
      console.error('Error validating API key via backend:', error);
      localStorage.removeItem(STORAGE_KEY_API_KEY); // Remove potentially invalid key
      // Rethrow the error so the UI component can catch it
      throw error;
    }
  }

  /**
   * Create a new agent configuration locally.
   * @param {Object} agentData - The agent configuration from the GUI
   * @returns {Promise<Object>} - The created agent configuration object
   */
  async createAgent(agentData) {
    // Still managed locally
    const newAgentConfig = this._createLocalAgentConfig(agentData);
    this._saveAgentToLocalStorage(newAgentConfig);
    console.log("Saved new agent configuration locally:", newAgentConfig.id);
    return newAgentConfig;
  }

  /**
   * Update an existing agent configuration locally.
   * @param {string} agentId - The ID of the agent config to update
   * @param {Object} agentData - The updated agent configuration
   * @returns {Promise<Object>} - The updated agent configuration object
   */
  async updateAgent(agentId, agentData) {
    // Still managed locally
    const agents = this._getAgentsFromLocalStorage();
    const agentIndex = agents.findIndex(a => a.id === agentId);

    if (agentIndex === -1) {
      throw new Error(`Agent configuration with ID ${agentId} not found`);
    }

    const updatedAgentConfig = {
      ...agents[agentIndex],
      ...agentData,
      id: agentId,
      updated_at: new Date().toISOString(),
      lastUsed: agents[agentIndex].lastUsed
    };

    this._saveAgentToLocalStorage(updatedAgentConfig);
    console.log("Updated agent configuration locally:", updatedAgentConfig.id);
    return updatedAgentConfig;
  }

  /**
   * Get all agent configurations from local storage.
   * @returns {Promise<Array>} - List of agent configuration objects
   */
  async getAgents() {
    // Still managed locally
    return this._getAgentsFromLocalStorage();
  }

  /**
   * Delete an agent configuration from local storage.
   * @param {string} agentId - The ID of the agent config to delete
   * @returns {Promise<boolean>} - Whether the deletion was successful
   */
  async deleteAgent(agentId) {
    // Still managed locally
    try {
      const agents = this._getAgentsFromLocalStorage();
      const updatedAgents = agents.filter(agent => agent.id !== agentId);
      localStorage.setItem(STORAGE_KEY_AGENTS, JSON.stringify(updatedAgents));
      console.log("Deleted agent configuration locally:", agentId);
      return true;
    } catch (error) {
      console.error('Error deleting agent configuration:', error);
      return false;
    }
  }

  /**
   * Run an agent by sending its configuration and input to the backend.
   * @param {string} agentId - The ID of the agent configuration to run
   * @param {string} input - The user input message
   * @returns {Promise<Object>} - The run result from the backend
   */
  async runAgent(agentId, input) {
    console.log(`Calling backend to run agent config ID: ${agentId}`);
    const agents = this._getAgentsFromLocalStorage();
    const agentConfig = agents.find(a => a.id === agentId);

    if (!agentConfig) {
      throw new Error(`Agent configuration with ID ${agentId} not found`);
    }

    // Retrieve the current API key to potentially pass if needed,
    // although ideally, the backend uses its configured credentials.
    const apiKey = localStorage.getItem(STORAGE_KEY_API_KEY);
    // if (!apiKey) {
    //   throw new Error('Google API key not found in local storage.');
    // }

    try {
      const response = await fetch(`${BACKEND_URL}/api/run-agent`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Optionally include API key if backend needs it per-request
          // 'Authorization': `Bearer ${apiKey}` // Or a custom header
        },
        body: JSON.stringify({
          agentConfig: agentConfig, // Send the full configuration
          input: input
        }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || `Backend error! status: ${response.status}`);
      }

      // Update last used time locally after successful run
      const agentIndex = agents.findIndex(a => a.id === agentId);
       if (agentIndex !== -1) {
         agents[agentIndex].last_used = new Date().toISOString();
         // Don't update lastUsed here, getAgents will calculate it
         localStorage.setItem(STORAGE_KEY_AGENTS, JSON.stringify(agents));
       }

      console.log("Backend agent run successful.");
      return result; // Return the result from the backend

    } catch (error) {
      console.error('Error running agent via backend:', error);
      throw error; // Rethrow for the UI component
    }
  }

  // --- Helper Methods (Mostly unchanged, manage local config) ---

  _createLocalAgentConfig(agentData) {
    const timestamp = new Date().toISOString();
    return {
      id: `agent-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      name: agentData.name || 'Untitled Agent',
      description: agentData.description || '',
      model: agentData.model || modelOptions[0].value,
      instructions: agentData.instructions || 'You are a helpful Google AI assistant.',
      tools: agentData.tools || [],
      handoffs: agentData.handoffs || [],
      inputGuardrails: agentData.inputGuardrails || [],
      outputGuardrails: agentData.outputGuardrails || [],
      modelSettings: {
        ...modelSettingsDefaults,
        ...(agentData.modelSettings || {})
      },
      outputType: agentData.outputType || null,
      created_at: timestamp,
      updated_at: timestamp,
      last_used: null,
      lastUsed: 'Never'
    };
  }

  _saveAgentToLocalStorage(agentConfig) {
    const agents = this._getAgentsFromLocalStorage(false); // Get raw data without relative time
    const existingIndex = agents.findIndex(a => a.id === agentConfig.id);
    if (existingIndex >= 0) {
      agents[existingIndex] = agentConfig;
    } else {
      agents.push(agentConfig);
    }
    agents.sort((a, b) => a.name.localeCompare(b.name));
    localStorage.setItem(STORAGE_KEY_AGENTS, JSON.stringify(agents));
  }

  _getAgentsFromLocalStorage(addRelativeTime = true) { // Added flag
    const agentsJson = localStorage.getItem(STORAGE_KEY_AGENTS);
    try {
      const agents = agentsJson ? JSON.parse(agentsJson) : [];
      if (addRelativeTime) {
        return agents.map(agent => ({
          ...agent,
          lastUsed: agent.last_used ? this._getRelativeTimeString(new Date(agent.last_used)) : 'Never'
        }));
      }
      return agents; // Return raw data if flag is false
    } catch (error) {
      console.error("Error parsing agents from local storage:", error);
      localStorage.setItem(STORAGE_KEY_AGENTS, JSON.stringify([]));
      return [];
    }
  }

   _getRelativeTimeString(date) {
     if (!date || isNaN(date.getTime())) return 'Never';
     const now = new Date();
     const diffMs = now.getTime() - date.getTime();
     const diffSec = Math.floor(diffMs / 1000);
     const diffMin = Math.floor(diffSec / 60);
     const diffHour = Math.floor(diffMin / 60);
     const diffDay = Math.floor(diffHour / 24);
     const diffWeek = Math.floor(diffDay / 7);
     const diffMonth = Math.floor(diffDay / 30);
     const diffYear = Math.floor(diffDay / 365);

     if (diffSec < 60) return 'Just now';
     if (diffMin === 1) return '1 minute ago';
     if (diffMin < 60) return `${diffMin} minutes ago`;
     if (diffHour === 1) return '1 hour ago';
     if (diffHour < 24) return `${diffHour} hours ago`;
     if (diffDay === 1) return 'Yesterday';
     if (diffDay < 7) return `${diffDay} days ago`;
     if (diffWeek === 1) return '1 week ago';
     if (diffMonth < 1) return `${diffWeek} weeks ago`;
     if (diffMonth === 1) return '1 month ago';
     if (diffYear < 1) return `${diffMonth} months ago`;
     if (diffYear === 1) return '1 year ago';
     return `${diffYear} years ago`;
   }
}

const apiServiceInstance = new ApiService();
export default apiServiceInstance;