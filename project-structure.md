└── gemini-agentsdk-gui-app/       
    ├── README.md                   
    ├── CHANGELOG.md                
    ├── CODE_OF_CONDUCT.md           (Disclaimer)
    ├── config-overrides.js         <-- Potentially Unchanged (Check ADK browser needs)
    ├── CONTRIBUTING.md              (Disclaimer)
    ├── LICENSE                      (Copyright Holder/Year)
    ├── Google-ADK Reference Notes.md <-- NEW (or replace OpenAI one)
    ├── package.json                
    ├── SECURITY.md                  (Disclaimer)
    ├── test-google-api.bat         <-- Renamed & Modified
    ├── webpack.config.js           <-- Potentially Unchanged
    ├── docs/                        (Content & Naming)
    │   ├── deployment.md
    │   ├── example-agents.md
    │   ├── faq.md
    │   ├── getting-started.md
    │   └── index.md
    ├── public/                      (index.html, manifest.json)
    │   ├── index.html
    │   └── manifest.json
    ├── src/
    │   ├── App.js                   (Naming, API Key Logic)
    │   ├── index.js                <-- Unchanged
    │   ├── test-google-api.js      <-- Renamed & Modified
    │   ├── components/             <-- Sub-components will be modified
    │   │   ├── AgentBuilder.js
    │   │   ├── AgentTester.js
    │   │   ├── ApiKeySetup.js        
    │   │   ├── Dashboard.js
    │   │   ├── Navbar.js              (Naming)
    │   │   └── builder/
    │   │       ├── BasicDetails.js      (Models)
    │   │       ├── CodePreview.js       (Templates)
    │   │       ├── GuardrailsConfigurator.js  (ADK Guardrails)
    │   │       ├── SubAgentsSelector.js <-- Renamed & Modified (Handoffs -> SubAgents)
    │   │       ├── InstructionsEditor.js  (Templates/Text)
    │   │       ├── TestAgent.js         (API Calls)
    │   │       └── ToolsSelector.js     (ADK Tools)
    │   ├── styles/
    │   │   └── index.css             
    │   └── utils/
    │       ├── apiService.js         
    │       ├── codeTemplates.js      
    │       ├── modelOptions.js       
    │       └── tools.js              
    └── .github/                     
        ├── ISSUE_TEMPLATE/
        │   ├── bug_report.md
        │   └── feature_request.md
        └── PULL_REQUEST_TEMPLATE/
            └── pull_request_template.md