// Uses helpbox deepseek-coder-v2:16b as brain (UPGRADED from qwen2.5:14b)
// Uses helpbox deepseek-coder-v2:16b for code generation
// Monitors .ai/chats/codriver.txt for tasks

use chrono::Utc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};
use tokio::time::sleep;

const HOSTBOX_OLLAMA: &str = "http://192.168.12.106:11434";
const HELPBOX_OLLAMA: &str = "http://192.168.12.66:11434";
const WORKBOX_OLLAMA: &str = "http://192.168.12.136:11434";
const CALLBOX_OLLAMA: &str = "http://192.168.12.9:11434";
const BRAIN_MODEL: &str = "deepseek-coder-v2:16b";  // UPGRADED: Smarter than qwen2.5:14b
const CODE_MODEL: &str = "deepseek-coder-v2:16b";
const DUCKDB_MODEL: &str = "duckdb-nsql:7b";
const CHAT_FILE: &str = "/home/admin/freightdev/openhwy/.ai/chats/codriver.txt";
const PROJECTS_DIR: &str = "/home/admin/freightdev/openhwy/.ai/projects/complete";
const CONTEXT_LOG_DIR: &str = "/home/admin/freightdev/openhwy/.ai/logs";
const HELPBOX_CONTEXT_FILE: &str = "/home/admin/freightdev/openhwy/.ai/logs/helpbox_context.json";
const DUCKDB_INDEX_FILE: &str = "/home/admin/freightdev/openhwy/.ai/logs/codebase.duckdb";
const CHECK_INTERVAL_MS: u64 = 500;
const CONTEXT_TIMEOUT_SECS: u64 = 300; // 5 minutes
const MAX_CONTEXT_MESSAGES: usize = 20; // Keep last 20 messages

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContextMessage {
    role: String,      // "user" or "assistant"
    content: String,
    timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct HelpboxContext {
    messages: VecDeque<ContextMessage>,
    last_file: Option<String>,
    last_activity: String,
}

#[derive(Debug, Clone)]
struct OllamaNode {
    name: String,
    endpoint: String,
    models: Vec<String>,
}

struct CoDriver {
    client: Client,
    last_line_count: usize,
    helpbox_context: HelpboxContext,
    last_helpbox_activity: Instant,
    available_nodes: Vec<OllamaNode>,
}

impl CoDriver {
    fn new() -> Self {
        // Try to load existing context
        let helpbox_context = Self::load_helpbox_context().unwrap_or_else(|_| HelpboxContext {
            messages: VecDeque::new(),
            last_file: None,
            last_activity: Utc::now().to_rfc3339(),
        });

        Self {
            client: Client::new(),
            last_line_count: 0,
            helpbox_context,
            last_helpbox_activity: Instant::now(),
            available_nodes: Vec::new(),
        }
    }

    async fn discover_models(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Discovering available models across Ollama cluster...");

        let nodes = vec![
            ("hostbox", HOSTBOX_OLLAMA),
            ("helpbox", HELPBOX_OLLAMA),
            ("workbox", WORKBOX_OLLAMA),
            ("callbox", CALLBOX_OLLAMA),
        ];

        for (name, endpoint) in nodes {
            match self.client
                .get(format!("{}/api/tags", endpoint))
                .timeout(Duration::from_secs(3))
                .send()
                .await
            {
                Ok(response) => {
                    if let Ok(data) = response.json::<serde_json::Value>().await {
                        let models: Vec<String> = data["models"]
                            .as_array()
                            .unwrap_or(&vec![])
                            .iter()
                            .filter_map(|m| m["name"].as_str().map(|s| s.to_string()))
                            .collect();

                        println!("  {} ({} models): {}", name, models.len(), models.join(", "));

                        self.available_nodes.push(OllamaNode {
                            name: name.to_string(),
                            endpoint: endpoint.to_string(),
                            models,
                        });
                    }
                }
                Err(_) => {
                    println!("  {} - offline", name);
                }
            }
        }

        Ok(())
    }

    fn find_model(&self, model_name: &str) -> Option<String> {
        for node in &self.available_nodes {
            if node.models.iter().any(|m| m == model_name) {
                return Some(node.endpoint.clone());
            }
        }
        None
    }

    fn load_helpbox_context() -> Result<HelpboxContext, Box<dyn std::error::Error>> {
        if Path::new(HELPBOX_CONTEXT_FILE).exists() {
            let content = fs::read_to_string(HELPBOX_CONTEXT_FILE)?;
            let context: HelpboxContext = serde_json::from_str(&content)?;
            Ok(context)
        } else {
            Err("No context file found".into())
        }
    }

    fn save_helpbox_context(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure log directory exists
        fs::create_dir_all(CONTEXT_LOG_DIR)?;

        // Save context to JSON
        let json = serde_json::to_string_pretty(&self.helpbox_context)?;
        fs::write(HELPBOX_CONTEXT_FILE, json)?;

        Ok(())
    }

    fn add_to_helpbox_context(&mut self, role: &str, content: &str) {
        let msg = ContextMessage {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: Utc::now().to_rfc3339(),
        };

        self.helpbox_context.messages.push_back(msg);

        // Keep only last MAX_CONTEXT_MESSAGES
        while self.helpbox_context.messages.len() > MAX_CONTEXT_MESSAGES {
            self.helpbox_context.messages.pop_front();
        }

        self.helpbox_context.last_activity = Utc::now().to_rfc3339();
        self.last_helpbox_activity = Instant::now();

        // Save context
        let _ = self.save_helpbox_context();
    }

    fn clear_helpbox_context_if_timeout(&mut self) {
        if self.last_helpbox_activity.elapsed().as_secs() > CONTEXT_TIMEOUT_SECS {
            println!("⏰ Helpbox context timed out (5 min idle), saving and clearing...");

            // Save final context before clearing
            let _ = self.save_helpbox_context();

            // Clear context
            self.helpbox_context.messages.clear();
            self.helpbox_context.last_file = None;
            self.helpbox_context.last_activity = Utc::now().to_rfc3339();
            self.last_helpbox_activity = Instant::now();
        }
    }

    async fn execute_command(&self, command: &str) -> Result<String, Box<dyn std::error::Error>> {
        println!("🔧 Executing: {}", command);

        let output = Command::new("bash")
            .arg("-c")
            .arg(command)
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push_str("\nSTDERR:\n");
            }
            result.push_str(&stderr);
        }

        Ok(result)
    }

    async fn diagnose_error(&self, error: &str, context: &str) -> Result<String, Box<dyn std::error::Error>> {
        let diagnostic_prompt = format!(
            r#"You are CoDriver's troubleshooting system.

ERROR: {}

CONTEXT: {}

What bash commands should I run to diagnose this issue? Respond with ONLY the bash command, one per line.
If no diagnostic needed, respond "NO_DIAGNOSTICS_NEEDED".

Examples:
- ssh helpbox "ps aux | grep ollama"
- curl -s http://192.168.12.66:11434/api/tags
- ssh helpbox "systemctl status ollama""#,
            error, context
        );

        let response = self.query_brain(&diagnostic_prompt).await?;
        Ok(response)
    }

    async fn troubleshoot_helpbox_connection(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        println!("🔍 Diagnosing helpbox connection issue...");

        // Check if helpbox is running
        let check_cmd = r#"sshpass -p 'Ibsyconje3!' ssh -o StrictHostKeyChecking=no helpbox "ps aux | grep -i ollama | grep -v grep""#;
        let ps_output = self.execute_command(check_cmd).await?;

        if ps_output.is_empty() {
            println!("❌ Helpbox Ollama not running!");
            self.respond("Helpbox Ollama is not running. Attempting to start...")?;

            // Try to start Ollama
            let start_cmd = r#"sshpass -p 'Ibsyconje3!' ssh helpbox "systemctl start ollama""#;
            let _ = self.execute_command(start_cmd).await?;

            // Wait a bit
            sleep(Duration::from_secs(5)).await;

            // Check again
            let recheck_output = self.execute_command(check_cmd).await?;
            if recheck_output.is_empty() {
                println!("❌ Failed to start helpbox Ollama");
                self.respond("Could not start helpbox Ollama. Manual intervention needed.")?;
                return Ok(false);
            } else {
                println!("✓ Helpbox Ollama started successfully");
                self.respond("Helpbox Ollama started. Retrying...")?;
                return Ok(true);
            }
        }

        // Check if helpbox is overloaded
        println!("📊 Checking helpbox load...");
        let load_output = ps_output.to_lowercase();

        // Ask brain to analyze the load
        let analysis_prompt = format!(
            r#"You are analyzing server load.

PROCESS OUTPUT:
{}

Is the Ollama process overloaded or busy? Answer with:
- "OVERLOADED" if CPU is very high (>300%) or seems stuck
- "BUSY" if it's processing but might finish soon
- "NORMAL" if load looks reasonable

Respond with ONLY one word."#,
            ps_output
        );

        let load_status = self.query_brain(&analysis_prompt).await?;
        let load_status_lower = load_status.to_lowercase();

        if load_status_lower.contains("overloaded") {
            println!("⚠️  Helpbox is OVERLOADED");
            self.respond("Helpbox is overloaded. Waiting 30 seconds before retry...")?;
            sleep(Duration::from_secs(30)).await;
            return Ok(true);
        } else if load_status_lower.contains("busy") {
            println!("⏳ Helpbox is BUSY processing another task");
            self.respond("Helpbox is busy. Waiting 15 seconds before retry...")?;
            sleep(Duration::from_secs(15)).await;
            return Ok(true);
        } else {
            println!("❓ Helpbox appears normal but connection failed");
            self.respond("Helpbox looks normal but connection failed. Waiting 5 seconds...")?;
            sleep(Duration::from_secs(5)).await;
            return Ok(true);
        }
    }

    fn build_helpbox_prompt_with_context(&self, new_task: &str) -> String {
        // Check if task contains code block - if so, extract and use it directly
        if new_task.contains("```") {
            let start = new_task.find("```").unwrap();
            let end = new_task.rfind("```").unwrap();
            if start < end {
                // Extract code between triple backticks
                let code_section = &new_task[start..=end];
                // Remove first line (language marker) and last line (closing backticks)
                let lines: Vec<&str> = code_section.lines().collect();
                if lines.len() > 2 {
                    let actual_code = lines[1..lines.len()-1].join("\n");
                    return format!("Generate this EXACT code. Copy it verbatim:\n\n{}", actual_code);
                }
            }
        }

        // No code block found - build instructional prompt
        let mut prompt = String::new();

        // Task FIRST - most important
        prompt.push_str("TASK: ");
        prompt.push_str(new_task);
        prompt.push_str("\n\n");

        // Then requirements
        prompt.push_str(r#"MANDATORY REQUIREMENTS - YOU MUST FOLLOW THESE EXACTLY:

1. If the task specifies an API endpoint, use EXACTLY that endpoint
   Example: If task says "http://localhost:8000/api/leads" → use that EXACT URL
   DO NOT use "https://api.example.com" or any placeholder

2. If the task specifies interface fields, use EXACTLY those fields
   Example: If task says "budget_min and budget_max" → use those EXACT field names
   DO NOT use generic names like "budget"

3. If the task asks for a table, generate an HTML <table> element
   DO NOT generate a list or cards

4. STYLING - TAILWIND CSS ONLY:
   ✅ USE: className="..." with Tailwind classes
   ❌ NEVER: import anything from @mui/*, @material-ui/*, styled-components
   ❌ NEVER: Material-UI components like <TableHead>, <Paper>, etc.

5. Always include loading and error states with proper TypeScript types

GENERATE THE CODE NOW - FOLLOW THE TASK REQUIREMENTS EXACTLY:"#);

        prompt
    }

    async fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🤖 CoDriver v0.3 Online - Autonomous Mode");
        println!("📂 Monitoring: {}", CHAT_FILE);
        println!("💾 Context: {} messages loaded", self.helpbox_context.messages.len());
        println!();

        // Create chat file if it doesn't exist
        if !Path::new(CHAT_FILE).exists() {
            fs::write(CHAT_FILE, "# CoDriver Chat\n# Write messages as: user: \"your message\"\n\n")?;
        }

        // Count existing lines
        let content = fs::read_to_string(CHAT_FILE)?;
        self.last_line_count = content.lines().count();

        // Discover available models
        self.discover_models().await?;

        println!("\n✅ CoDriver ready! Autonomous operation enabled.\n");
        Ok(())
    }

    async fn verify_ollama(&self) -> Result<(), Box<dyn std::error::Error>> {
        let response = self.client
            .get(format!("{}/api/tags", HOSTBOX_OLLAMA))
            .timeout(Duration::from_secs(5))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("Ollama returned: {}", response.status()).into());
        }

        Ok(())
    }

    async fn check_new_messages(&mut self) -> Result<Option<String>, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(CHAT_FILE)?;
        let lines: Vec<&str> = content.lines().collect();

        if lines.len() > self.last_line_count {
            // Get new lines
            let new_lines: Vec<&str> = lines[self.last_line_count..].to_vec();
            self.last_line_count = lines.len();

            // Find last user message
            for line in new_lines.iter().rev() {
                if line.trim().starts_with("user:") {
                    let message = line.trim()
                        .strip_prefix("user:")
                        .unwrap_or("")
                        .trim()
                        .trim_matches('"')
                        .to_string();

                    if !message.is_empty() {
                        return Ok(Some(message));
                    }
                }
            }
        }

        Ok(None)
    }

    async fn query_brain(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let payload = json!({
            "model": BRAIN_MODEL,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.7,
                "num_predict": 500
            }
        });

        let response = self.client
            .post(format!("{}/api/generate", HELPBOX_OLLAMA))  // UPGRADED: Using helpbox (smarter brain)
            .json(&payload)
            .timeout(Duration::from_secs(120))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("Ollama query failed: {}", response.status()).into());
        }

        let result: serde_json::Value = response.json().await?;
        let answer = result["response"]
            .as_str()
            .unwrap_or("(no response)")
            .trim()
            .to_string();

        Ok(answer)
    }

    fn respond(&self, message: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Read current content
        let mut content = fs::read_to_string(CHAT_FILE)?;

        // Append codriver response
        content.push_str(&format!("codriver: \"{}\"\n", message));

        // Write back
        fs::write(CHAT_FILE, content)?;

        Ok(())
    }

    async fn query_helpbox(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let payload = json!({
            "model": CODE_MODEL,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.2,
                "num_predict": 2000
            }
        });

        let response = self.client
            .post(format!("{}/api/generate", HELPBOX_OLLAMA))
            .json(&payload)
            .timeout(Duration::from_secs(180))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("Helpbox query failed: {}", response.status()).into());
        }

        let result: serde_json::Value = response.json().await?;
        let code = result["response"]
            .as_str()
            .unwrap_or("// No code generated")
            .trim()
            .to_string();

        Ok(code)
    }

    async fn search_codebase(&self, query: &str) -> Result<String, Box<dyn std::error::Error>> {
        println!("🔍 Searching codebase with DuckDB...");

        // First, ask DuckDB model to generate SQL query
        let sql_prompt = format!(
            r#"Convert this search request to a SQL query for searching code files in DuckDB:

"{}"

Schema:
- files table: path (text), content (text), extension (text), size (integer)

Generate ONLY the SQL query, nothing else."#,
            query
        );

        let payload = json!({
            "model": DUCKDB_MODEL,
            "prompt": sql_prompt,
            "stream": false,
            "options": {
                "temperature": 0.1,
                "num_predict": 200
            }
        });

        let response = self.client
            .post(format!("{}/api/generate", WORKBOX_OLLAMA))
            .json(&payload)
            .timeout(Duration::from_secs(30))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(format!("DuckDB query failed: {}", response.status()).into());
        }

        let result: serde_json::Value = response.json().await?;
        let sql = result["response"]
            .as_str()
            .unwrap_or("SELECT * FROM files LIMIT 10")
            .trim()
            .to_string();

        println!("  Generated SQL: {}", sql);

        // Execute the SQL query via bash + duckdb
        let duckdb_cmd = format!(
            "duckdb {} -c \"{}\" 2>/dev/null || echo 'DuckDB not available'",
            DUCKDB_INDEX_FILE, sql
        );

        let output = self.execute_command(&duckdb_cmd).await?;

        Ok(output)
    }

    async fn index_codebase(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 Indexing codebase with DuckDB...");

        let index_script = format!(
            r#"
duckdb {} << 'EOF'
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    content TEXT,
    extension TEXT,
    size INTEGER,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index all Rust files
INSERT OR REPLACE INTO files (path, content, extension, size)
SELECT
    regexp_replace(filename, '.*/openhwy/', '') as path,
    content,
    regexp_extract(filename, '\.([^.]+)$', 1) as extension,
    length(content) as size
FROM read_text('/home/admin/freightdev/openhwy/**/*.rs', filename=true)
WHERE content IS NOT NULL;

-- Index TypeScript/JavaScript files
INSERT OR REPLACE INTO files (path, content, extension, size)
SELECT
    regexp_replace(filename, '.*/openhwy/', '') as path,
    content,
    regexp_extract(filename, '\.([^.]+)$', 1) as extension,
    length(content) as size
FROM read_text('/home/admin/freightdev/openhwy/**/*.{{ts,tsx,js,jsx}}', filename=true)
WHERE content IS NOT NULL;

SELECT 'Indexed ' || COUNT(*) || ' files' as result FROM files;
EOF
"#,
            DUCKDB_INDEX_FILE
        );

        let output = self.execute_command(&index_script).await?;
        println!("  {}", output.trim());

        Ok(())
    }

    async fn generate_component(&mut self, task: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Check for context timeout
        self.clear_helpbox_context_if_timeout();

        // Determine if we're editing an existing file or creating new
        let is_edit = task.to_lowercase().contains("edit")
            || task.to_lowercase().contains("update")
            || task.to_lowercase().contains("add to")
            || task.to_lowercase().contains("modify");

        let filepath = if is_edit && self.helpbox_context.last_file.is_some() {
            // Use the last file we worked on
            self.helpbox_context.last_file.clone().unwrap()
        } else {
            // Determine new filename
            let filename = self.extract_filename(task);
            format!("{}/{}", PROJECTS_DIR, filename)
        };

        // Check if file exists
        let file_exists = Path::new(&filepath).exists();

        if file_exists {
            println!("📝 Editing existing file: {}", filepath);
            self.respond(&format!("Editing {} using helpbox (L3)...", filepath))?;

            // Load existing file content for context
            let existing_content = fs::read_to_string(&filepath)?;
            self.add_to_helpbox_context("assistant", &format!("Current file content:\n{}", existing_content));
        } else {
            println!("🔧 Generating new component using helpbox...");
            self.respond("Routing to helpbox (L3) for code generation...")?;
        }

        // Build prompt with context
        self.add_to_helpbox_context("user", task);
        let code_prompt = self.build_helpbox_prompt_with_context(task);

        // Query helpbox with retry and troubleshooting
        let code = match self.query_helpbox(&code_prompt).await {
            Ok(code) => code,
            Err(e) => {
                // Connection failed - troubleshoot
                println!("❌ Helpbox query failed: {}", e);
                self.respond(&format!("Helpbox connection failed: {}. Troubleshooting...", e))?;

                // Try to troubleshoot
                match self.troubleshoot_helpbox_connection().await {
                    Ok(true) => {
                        // Troubleshooting suggested retry
                        println!("🔄 Retrying helpbox query after troubleshooting...");
                        self.respond("Retrying code generation...")?;

                        // Retry once
                        match self.query_helpbox(&code_prompt).await {
                            Ok(code) => code,
                            Err(e2) => {
                                println!("❌ Retry failed: {}", e2);
                                return Err(format!("Helpbox unavailable after troubleshooting: {}", e2).into());
                            }
                        }
                    }
                    Ok(false) => {
                        // Troubleshooting failed, cannot proceed
                        return Err("Helpbox unavailable and cannot be fixed automatically".into());
                    }
                    Err(e3) => {
                        println!("❌ Troubleshooting error: {}", e3);
                        return Err(format!("Troubleshooting failed: {}", e3).into());
                    }
                }
            }
        };

        // Add response to context
        self.add_to_helpbox_context("assistant", &code);

        // Ensure projects directory exists
        fs::create_dir_all(PROJECTS_DIR)?;

        // Save code
        fs::write(&filepath, &code)?;

        // Remember this file
        self.helpbox_context.last_file = Some(filepath.clone());
        let _ = self.save_helpbox_context();

        if file_exists {
            println!("✓ File updated: {}", filepath);
            self.respond(&format!("✅ File updated: {}", filepath))?;
        } else {
            println!("✓ Code generated: {}", filepath);
            self.respond(&format!("✅ Component ready: {}", filepath))?;
        }

        Ok(())
    }

    fn extract_filename(&self, task: &str) -> String {
        // Try to extract filename from task
        if task.to_lowercase().contains("component") {
            let words: Vec<&str> = task.split_whitespace().collect();
            let mut component_name = "Component";

            for (i, word) in words.iter().enumerate() {
                if word.to_lowercase() == "component" && i > 0 {
                    component_name = words[i - 1];
                    break;
                } else if word.ends_with("Card") || word.ends_with("Button") || word.ends_with("Form") {
                    component_name = word;
                    break;
                }
            }

            format!("{}.tsx", component_name)
        } else {
            "generated_code.tsx".to_string()
        }
    }

    async fn handle_task(&mut self, task: String) -> Result<(), Box<dyn std::error::Error>> {
        println!("📥 User task: {}", task);

        // Build decision prompt - let brain decide what to do
        let available_models: Vec<String> = self.available_nodes
            .iter()
            .map(|n| format!("{}: {}", n.name, n.models.join(", ")))
            .collect();

        let decision_prompt = format!(
            r#"You are CoDriver, an autonomous AI development assistant. Analyze the user's request and choose the BEST capability.

USER REQUEST: "{}"

AVAILABLE OLLAMA RESOURCES:
{}

YOUR CAPABILITIES:

1. **generate_component** - Build or edit code files
   USE WHEN: User says "build", "create", "generate", "fix code", "edit", "update code", "make component"
   EXAMPLES:
   - "Build a React component" → generate_component
   - "Fix this code to use Tailwind CSS" → generate_component
   - "Create LeadList.tsx" → generate_component
   - "Update the API endpoint in the code" → generate_component

2. **execute_command** - Run bash commands, tools, curl, scripts
   USE WHEN: User says "run", "execute", "call", "curl", "check", "test", "use tool"
   EXAMPLES:
   - "Run the check-ollama.sh tool" → execute_command
   - "Call HOSTBOX with curl" → execute_command
   - "Check if backend is running" → execute_command
   - "Clean whitespace with the tool" → execute_command
   - "Test the API with curl" → execute_command

3. **search_codebase** - Find code patterns with DuckDB
   USE WHEN: User says "find", "search", "locate", "where is"
   EXAMPLES:
   - "Find all error handlers" → search_codebase
   - "Where is the API endpoint defined?" → search_codebase
   - "Search for React components" → search_codebase

4. **index_codebase** - Index files into DuckDB
   USE WHEN: User says "index", "scan files", "build index"
   EXAMPLES:
   - "Index the codebase" → index_codebase
   - "Scan all files" → index_codebase

5. **find_leads** - Search for paid development work
   USE WHEN: User says "find work", "search jobs", "get leads"
   EXAMPLES:
   - "Find dev gigs" → find_leads
   - "Search for work" → find_leads

6. **respond_only** - Answer conversationally without action
   USE WHEN: User asks a question that doesn't need code/commands
   EXAMPLES:
   - "What did you learn?" → respond_only
   - "Explain this concept" → respond_only
   - "What's your status?" → respond_only

DECISION RULES:
- Code tasks (build/fix/edit) = generate_component
- Commands/tools (run/execute/curl/test) = execute_command
- Code search (find/locate) = search_codebase
- File indexing = index_codebase
- Work search = find_leads
- Questions only = respond_only

ANALYZE THE REQUEST:
Task keywords: "{}"

If the task mentions:
- "build", "create", "generate", "component", "fix code", "edit", "code" → generate_component
- "run", "execute", "curl", "tool", "command", "test", "check", "call" → execute_command
- "find", "search", "where", "locate" in code → search_codebase
- "index" → index_codebase
- Just asking questions → respond_only

YOUR CHOICE (respond with ONLY ONE capability name):"#,
            task, available_models.join("\n"), task
        );

        // Ask brain what to do
        let decision = self.query_brain(&decision_prompt).await?;
        let action = decision.lines().next().unwrap_or("respond_only").trim().to_lowercase();

        println!("🧠 Autonomous decision: {}", action);
        self.respond(&format!("Decision: {} - executing...", action))?;

        // Execute based on autonomous decision
        // More specific matches first to avoid conflicts
        match action.as_str() {
            s if s.contains("index_codebase") || s.contains("index") => {
                match self.index_codebase().await {
                    Ok(_) => {
                        println!("✓ Indexing complete");
                        self.respond("Codebase indexed successfully")?;
                    }
                    Err(e) => {
                        eprintln!("❌ Indexing failed: {}", e);
                        self.respond(&format!("Indexing failed: {}", e))?;
                    }
                }
            }

            s if s.contains("search_codebase") || s.contains("search") => {
                match self.search_codebase(&task).await {
                    Ok(results) => {
                        println!("✓ Search complete");
                        self.respond(&format!("Search results:\n{}", results.lines().take(20).collect::<Vec<_>>().join("\n")))?;
                    }
                    Err(e) => {
                        eprintln!("❌ Search failed: {}", e);
                        self.respond(&format!("Search failed: {}", e))?;
                    }
                }
            }

            s if s.contains("generate_component") || s.contains("generate") => {
                if let Err(e) = self.generate_component(&task).await {
                    eprintln!("❌ Code generation failed: {}", e);
                    self.respond(&format!("Failed: {}", e))?;
                } else {
                    println!("✓ Complete\n");
                }
            }

            s if s.contains("execute_command") => {
                // Extract command from task
                let cmd_prompt = format!(
                    r#"Extract the bash command to execute from this request: "{}".
Respond with ONLY the bash command, nothing else."#,
                    task
                );

                let cmd = self.query_brain(&cmd_prompt).await?;
                println!("🔧 Executing: {}", cmd);

                match self.execute_command(&cmd).await {
                    Ok(output) => {
                        self.respond(&format!("Command output:\n{}", output.lines().take(30).collect::<Vec<_>>().join("\n")))?;
                    }
                    Err(e) => {
                        self.respond(&format!("Command failed: {}", e))?;
                    }
                }
            }

            s if s.contains("find_leads") => {
                self.respond("Routing to Lead Agency API...")?;
                println!("⚠️  Lead generation not implemented yet\n");
            }

            _ => {
                // Default: respond conversationally
                let response_prompt = format!(
                    r#"You are CoDriver, an autonomous AI coordinator.

User: "{}"

Respond helpfully (1-3 sentences)."#,
                    task
                );

                let response = self.query_brain(&response_prompt).await?;
                self.respond(&response)?;
                println!("✓ Responded\n");
            }
        }

        Ok(())
    }

    async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            // Check for new user messages
            if let Some(task) = self.check_new_messages().await? {
                if let Err(e) = self.handle_task(task).await {
                    eprintln!("❌ Task failed: {}", e);
                    self.respond(&format!("Error: {}", e))?;
                }
            }

            // Sleep briefly
            sleep(Duration::from_millis(CHECK_INTERVAL_MS)).await;
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut codriver = CoDriver::new();
    codriver.initialize().await?;
    codriver.run().await?;

    Ok(())
}
