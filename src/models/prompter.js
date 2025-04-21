// --- START OF FILE prompter.js ---

import { readFileSync, mkdirSync, writeFileSync} from 'fs';
import path from 'path'; // Needed for path operations
import { Examples } from '../utils/examples.js';
import { getCommandDocs } from '../agent/commands/index.js';
import { SkillLibrary } from "../agent/library/skill_library.js";
import { stringifyTurns } from '../utils/text.js';
import { getCommand } from '../agent/commands/index.js';
import settings from '../../settings.js';
import { log, logVision } from '../../logger.js'; // <-- IMPORT log AND logVision

// --- Model Provider Imports ---
import { Gemini } from './gemini.js';
import { GPT } from './gpt.js';
import { Claude } from './claude.js';
import { Mistral } from './mistral.js';
import { ReplicateAPI } from './replicate.js';
import { Local } from './local.js';
import { Novita } from './novita.js';
import { GroqCloudAPI } from './groq.js';
import { HuggingFace } from './huggingface.js';
import { Qwen } from "./qwen.js";
import { Grok } from "./grok.js";
import { DeepSeek } from './deepseek.js';
import { Hyperbolic } from './hyperbolic.js';
import { GLHF } from './glhf.js';
import { OpenRouter } from './openrouter.js';

export class Prompter {
    constructor(agent, fp) {
        this.agent = agent;
        const profilePath = path.resolve(fp); // Ensure absolute path
        const defaultProfilePath = path.resolve('./profiles/defaults/_default.json');
        const baseProfilePath = path.resolve(settings.base_profile);

        // --- Load Profiles ---
        try {
            this.profile = JSON.parse(readFileSync(profilePath, 'utf8'));
            const default_profile = JSON.parse(readFileSync(defaultProfilePath, 'utf8'));
            const base_profile = JSON.parse(readFileSync(baseProfilePath, 'utf8'));

            // Merge profiles: default < base < individual
            this.profile = { ...default_profile, ...base_profile, ...this.profile };

        } catch (err) {
            console.error(`Error loading profiles (Default: ${defaultProfilePath}, Base: ${baseProfilePath}, Individual: ${profilePath}):`, err);
            throw new Error(`Failed to load profile configuration: ${err.message}`);
        }

        this.convo_examples = null;
        this.coding_examples = null;

        const name = this.profile.name || path.basename(fp, '.json'); // Use filename as fallback name
        this.agent.name = name; // Ensure agent name is set early

        this.cooldown = this.profile.cooldown ? parseInt(this.profile.cooldown, 10) : 0;
        if (isNaN(this.cooldown) || this.cooldown < 0) {
            console.warn(`[${name}] Invalid cooldown value '${this.profile.cooldown}', setting to 0.`);
            this.cooldown = 0;
        }
        this.last_prompt_time = 0;
        this.awaiting_coding = false;

        // --- Instantiate Models ---
        try {
            console.log(`[${name}] Initializing models...`);
            // Chat Model (Mandatory)
            if (!this.profile.model) throw new Error("`model` (for chat) is not defined in the profile.");
            let chat_model_profile = this._prepareProfile(this.profile.model);
            this.chat_model = this._createModel(chat_model_profile);
            console.log(`[${name}] Chat model initialized: API=${chat_model_profile.api}, Model=${chat_model_profile.model}`);

            // Code Model (Optional, fallback to chat)
            let code_model_profile;
            if (this.profile.code_model) {
                code_model_profile = this._prepareProfile(this.profile.code_model);
                this.code_model = this._createModel(code_model_profile);
                console.log(`[${name}] Code model initialized: API=${code_model_profile.api}, Model=${code_model_profile.model}`);
            } else {
                this.code_model = this.chat_model;
                 console.log(`[${name}] No specific code_model configured, using chat model.`);
            }

            // Vision Model (Optional, fallback to chat)
            let vision_model_profile;
            if (this.profile.vision_model) {
                vision_model_profile = this._prepareProfile(this.profile.vision_model);
                this.vision_model = this._createModel(vision_model_profile);
                 console.log(`[${name}] Vision model initialized: API=${vision_model_profile.api}, Model=${vision_model_profile.model}`);
            } else {
                this.vision_model = this.chat_model;
                 console.log(`[${name}] No specific vision_model configured, using chat model (vision may fail if chat model lacks capability).`);
            }
        } catch (error) {
             console.error(`[${name}] Critical error during model initialization: ${error.message}`);
             throw error; // Stop if models can't be created
        }


        // --- Embedding Model ---
        let embedding_profile = this.profile.embedding;
        // Determine default embedding API if not specified
        if (embedding_profile === undefined) {
            const chatApi = this._prepareProfile(this.profile.model).api; // Get chat API
            if (chatApi !== 'ollama' && chatApi !== 'replicate' && chatApi !== 'huggingface') { // APIs known to often have embeddings
                 embedding_profile = { api: chatApi }; // Default to chat API if suitable
                 console.log(`[${name}] Embedding profile not specified, defaulting to chat API '${chatApi}' for embeddings.`);
            } else {
                 embedding_profile = { api: 'none' }; // Default to none for others
                 console.log(`[${name}] Embedding profile not specified, defaulting to 'none'. Word overlap will be used.`);
            }
        } else if (typeof embedding_profile === 'string') {
            embedding_profile = { api: embedding_profile }; // Convert string shorthand
        }

        // Ensure embedding_profile is an object before proceeding
        if (!embedding_profile || typeof embedding_profile !== 'object') {
             console.warn(`[${name}] Invalid embedding configuration: ${embedding_profile}. Defaulting to 'none'.`);
             embedding_profile = { api: 'none' };
        }


        console.log(`[${name}] Using embedding settings:`, embedding_profile);
        this.embedding_model = null; // Initialize as null
        if (embedding_profile.api && embedding_profile.api !== 'none') {
            try {
                // Prepare the profile specifically for embedding (might need model name)
                let final_embedding_profile = this._prepareProfile(embedding_profile, true); // Pass true to indicate embedding context
                this.embedding_model = this._createModel(final_embedding_profile);
                console.log(`[${name}] Embedding model initialized: API=${final_embedding_profile.api}, Model=${final_embedding_profile.model}`);
            } catch (err) {
                console.warn(`[${name}] Warning: Failed to initialize embedding model (API: ${embedding_profile.api}): ${err.message}`);
                console.log(`[${name}] Continuing with word-overlap for skills/examples.`);
                this.embedding_model = null;
            }
        } else {
             console.log(`[${name}] Embedding API is 'none' or not specified. Using word-overlap.`);
        }


        // --- Initialize Skill Library & Save Profile ---
        this.skill_libary = new SkillLibrary(agent, this.embedding_model); // Pass agent and potentially null model
        const botDir = path.resolve(`./bots/${name}`);
        try {
            mkdirSync(botDir, { recursive: true });
            // Save the final *merged* profile for debugging/reference
            writeFileSync(path.join(botDir, '_last_merged_profile.json'), JSON.stringify(this.profile, null, 4));
            console.log(`[${name}] Merged profile saved to ${path.join(botDir, '_last_merged_profile.json')}`);
        } catch (err) {
             console.error(`[${name}] Failed to create bot directory or save merged profile:`, err);
        }
    }

    // --- Helper to prepare profile object (string or object) ---
    _prepareProfile(profileInput, isEmbedding = false) {
        let profile = {};
        if (typeof profileInput === 'string') {
            profile = { model: profileInput }; // Convert string to object
        } else if (profileInput && typeof profileInput === 'object') {
            profile = { ...profileInput }; // Copy object
        } else {
            throw new Error(`Invalid profile input type: ${typeof profileInput}`);
        }

        // Infer API if not present
        if (!profile.api) {
            const modelName = profile.model || '';
            if (modelName.includes('openrouter/')) profile.api = 'openrouter';
            else if (modelName.includes('ollama/')) profile.api = 'ollama';
            else if (modelName.includes('gemini')) profile.api = 'google';
            else if (modelName.includes('gpt') || modelName.includes('o1') || modelName.includes('o3')) profile.api = 'openai';
            else if (modelName.includes('claude')) profile.api = 'anthropic';
            else if (modelName.includes('huggingface/')) profile.api = "huggingface";
            else if (modelName.includes('replicate/')) profile.api = 'replicate';
            else if (modelName.includes('mistralai/') || modelName.includes("mistral/")) profile.api = 'mistral';
            else if (modelName.includes("groq/") || modelName.includes("groqcloud/")) profile.api = 'groq';
            else if (modelName.includes("glhf/")) profile.api = 'glhf';
            else if (modelName.includes("hyperbolic/")) profile.api = 'hyperbolic';
            else if (modelName.includes('novita/')) profile.api = 'novita';
            else if (modelName.includes('qwen')) profile.api = 'qwen';
            else if (modelName.includes('grok')) profile.api = 'xai';
            else if (modelName.includes('deepseek')) profile.api = 'deepseek';
            else if (modelName.includes('mistral')) profile.api = 'mistral'; // Handle non-prefixed mistral
            else {
                // If it's for embedding and API is still unknown, default might be needed
                if (isEmbedding) {
                     console.warn(`Could not infer API for embedding model '${modelName}'. Check profile.`);
                     profile.api = 'none'; // Fallback for embedding
                } else {
                    throw new Error(`Could not infer API for model: ${modelName}`);
                }
            }
            // console.log(`Inferred API '${profile.api}' for model '${modelName}'`);
        }

        // Set default embedding model name if needed
        if (isEmbedding && !profile.model && profile.api !== 'none') {
             switch (profile.api) {
                 case 'google': profile.model = 'text-embedding-004'; break;
                 case 'openai': profile.model = 'text-embedding-3-small'; break;
                 case 'mistral': profile.model = 'mistral-embed'; break;
                 case 'qwen': profile.model = 'text-embedding-v1'; break; // Or v2/v3 if available
                 case 'ollama': profile.model = 'nomic-embed-text'; break; // Common Ollama default
                 // Add other API defaults as needed
                 default: console.warn(`No default embedding model set for API '${profile.api}'.`);
             }
             if(profile.model) console.log(`Using default embedding model '${profile.model}' for API '${profile.api}'.`);
        }


        // Ensure model name is set if API is not 'none'
        if (profile.api !== 'none' && !profile.model) {
             throw new Error(`Model name is required for API '${profile.api}'.`);
        }

        return profile;
    }


    // --- Helper to create model instance ---
    _createModel(profile) {
        if (!profile || !profile.api || (profile.api !== 'none' && !profile.model)) {
           throw new Error(`Cannot create model: Invalid prepared profile: ${JSON.stringify(profile)}`);
        }

        let model = null;
        const api = profile.api.toLowerCase();
        const modelName = profile.model;
        const url = profile.url;
        const params = profile.params || {};

        // console.log(`[${this.getName()}] Creating model instance for API: ${api}, Model: ${modelName}`); // Less verbose

        try {
            switch (api) {
                case 'google': model = new Gemini(modelName, url, params); break;
                case 'openai': model = new GPT(modelName, url, params); break;
                case 'anthropic': model = new Claude(modelName, url, params); break;
                case 'replicate': model = new ReplicateAPI(modelName, url, params); break;
                case 'ollama': model = new Local(modelName.replace('ollama/', ''), url, params); break;
                case 'mistral': model = new Mistral(modelName, url, params); break;
                case 'groq': model = new GroqCloudAPI(modelName.replace(/^(groqcloud|groq)\//, ''), url, params); break;
                case 'huggingface': model = new HuggingFace(modelName, url, params); break;
                case 'glhf': model = new GLHF(modelName.replace('glhf/', ''), url, params); break;
                case 'hyperbolic': model = new Hyperbolic(modelName.replace('hyperbolic/', ''), url, params); break;
                case 'novita': model = new Novita(modelName.replace('novita/', ''), url, params); break;
                case 'qwen': model = new Qwen(modelName, url, params); break;
                case 'xai': model = new Grok(modelName, url, params); break;
                case 'deepseek': model = new DeepSeek(modelName, url, params); break;
                case 'openrouter': model = new OpenRouter(modelName, url, params); break;
                case 'none': model = null; break; // Explicitly handle 'none' for embedding
                default:
                    throw new Error(`Unsupported API specified: '${api}'`);
            }
        } catch (error) {
            console.error(`[${this.getName()}] Failed to instantiate model for API '${api}', Model '${modelName}':`, error);
            throw error;
        }

        // Don't throw if model is null for 'none' API
        if (api !== 'none' && !model) {
            throw new Error(`Model creation returned null for API '${api}' and model '${modelName}'.`);
        }
        return model;
    }

    // --- Getters ---
    getName() {
        return this.profile.name || 'UnnamedAgent';
    }

    getInitModes() {
        return this.profile.modes || [];
    }

    // --- Initialization ---
    async initExamples() {
        // Initialize example handlers (they handle null embedding model internally)
        this.convo_examples = new Examples(this.embedding_model, settings.num_examples);
        this.coding_examples = new Examples(this.embedding_model, settings.num_examples);

        try {
            // Load conversation examples
            const convoExamplePaths = this.profile.conversation_examples || [];
            if (convoExamplePaths.length > 0) {
                console.log(`[${this.getName()}] Loading conversation examples from: ${convoExamplePaths.join(', ')}`);
                await this.convo_examples.load(convoExamplePaths);
            } else {
                console.log(`[${this.getName()}] No conversation examples specified.`);
            }

            // Load coding examples
            const codingExamplePaths = this.profile.coding_examples || [];
            if (codingExamplePaths.length > 0) {
                console.log(`[${this.getName()}] Loading coding examples from: ${codingExamplePaths.join(', ')}`);
                await this.coding_examples.load(codingExamplePaths);
            } else {
                console.log(`[${this.getName()}] No coding examples specified.`);
            }

            // Initialize skill library (loads skills from disk)
            console.log(`[${this.getName()}] Initializing skill library...`);
            await this.skill_libary.initSkillLibrary(); // Assumes this method exists and handles loading

            console.log(`[${this.getName()}] Examples and skills initialized.`);

        } catch (error) {
            console.error(`[${this.getName()}] Failed to initialize examples or skills:`, error);
            console.warn(`[${this.getName()}] Continuing without fully loaded examples/skills due to error.`);
            // Don't re-throw, allow agent to potentially function without them
        }
    }

    // --- Prompt String Replacement ---
    async replaceStrings(prompt, messages, examples=null, to_summarize=[], last_goals=null) {
        prompt = String(prompt || ''); // Ensure prompt is a string

        prompt = prompt.replaceAll('$NAME', this.agent.name || 'Bot');

        // Use try-catch for async operations within replaceAll context
        if (prompt.includes('$STATS')) {
            try {
                let stats = await getCommand('!stats').perform(this.agent);
                prompt = prompt.replaceAll('$STATS', stats || 'Stats unavailable.');
            } catch (e) { console.error("Error getting stats:", e); prompt = prompt.replaceAll('$STATS', 'Error fetching stats.'); }
        }
        if (prompt.includes('$INVENTORY')) {
            try {
                let inventory = await getCommand('!inventory').perform(this.agent);
                prompt = prompt.replaceAll('$INVENTORY', inventory || 'Inventory unavailable.');
            } catch (e) { console.error("Error getting inventory:", e); prompt = prompt.replaceAll('$INVENTORY', 'Error fetching inventory.'); }
        }
        if (prompt.includes('$ACTION')) {
            prompt = prompt.replaceAll('$ACTION', this.agent.actions.currentActionLabel || 'idle');
        }
        if (prompt.includes('$COMMAND_DOCS')) {
            prompt = prompt.replaceAll('$COMMAND_DOCS', getCommandDocs() || 'No command docs available.');
        }
        if (prompt.includes('$CODE_DOCS')) {
            try {
                const code_task_content = messages?.slice().reverse().find(msg =>
                    msg.role !== 'system' && msg.content?.includes('!newAction(')
                )?.content?.match(/!newAction\((.*?)\)/)?.[1] || '';
                const relevantDocs = await this.skill_libary.getRelevantSkillDocs(code_task_content, settings.relevant_docs_count);
                prompt = prompt.replaceAll('$CODE_DOCS', relevantDocs || 'No relevant code docs found.');
            } catch (e) { console.error("Error getting code docs:", e); prompt = prompt.replaceAll('$CODE_DOCS', 'Error fetching code docs.'); }
        }
        if (prompt.includes('$EXAMPLES') && examples) {
            try {
                const exampleMessage = await examples.createExampleMessage(messages || []);
                prompt = prompt.replaceAll('$EXAMPLES', exampleMessage || 'No relevant examples found.');
            } catch (e) { console.error("Error creating example message:", e); prompt = prompt.replaceAll('$EXAMPLES', 'Error fetching examples.'); }
        }
        if (prompt.includes('$MEMORY')) {
            prompt = prompt.replaceAll('$MEMORY', this.agent.history.memory || 'No memory available.');
        }
        if (prompt.includes('$TO_SUMMARIZE')) {
            prompt = prompt.replaceAll('$TO_SUMMARIZE', stringifyTurns(to_summarize || []) || 'Nothing to summarize.');
        }
        if (prompt.includes('$CONVO')) {
            prompt = prompt.replaceAll('$CONVO', 'Recent conversation:\n' + (stringifyTurns(messages || []) || 'No conversation history.'));
        }
        if (prompt.includes('$SELF_PROMPT')) {
            let self_prompt = (this.agent.self_prompter && !this.agent.self_prompter.isStopped())
                ? `YOUR CURRENT ASSIGNED GOAL: "${this.agent.self_prompter.prompt || 'None'}"\n`
                : 'You currently have no assigned goal.\n';
            prompt = prompt.replaceAll('$SELF_PROMPT', self_prompt);
        }
        if (prompt.includes('$LAST_GOALS')) {
            let goal_text = 'No recent goal attempts recorded.\n';
            if (last_goals && Object.keys(last_goals).length > 0) {
                goal_text = '';
                for (let goal in last_goals) {
                    goal_text += last_goals[goal]
                        ? `You recently successfully completed the goal "${goal}".\n`
                        : `You recently failed to complete the goal "${goal}".\n`;
                }
            }
            prompt = prompt.replaceAll('$LAST_GOALS', goal_text.trim());
        }
        if (prompt.includes('$BLUEPRINTS')) {
            let blueprints = 'No blueprints available.';
            if (this.agent.npc?.constructions && Object.keys(this.agent.npc.constructions).length > 0) {
                blueprints = 'Available blueprints: ' + Object.keys(this.agent.npc.constructions).join(', ');
            }
            prompt = prompt.replaceAll('$BLUEPRINTS', blueprints);
        }

        // Final check for remaining placeholders
        let remaining = prompt.match(/\$[A-Z_]+/g);
        if (remaining !== null && remaining.length > 0) {
            console.warn(`[${this.getName()}] Unknown or unhandled prompt placeholders: ${remaining.join(', ')}`);
        }
        return prompt;
    }

    // --- Cooldown Logic ---
    async checkCooldown() {
        if (this.cooldown <= 0) return;
        const now = Date.now();
        const elapsed = now - this.last_prompt_time;
        const waitTime = this.cooldown - elapsed;
        if (waitTime > 0) {
            // console.log(`[${this.getName()}] Cooldown active. Waiting for ${waitTime}ms...`); // Less verbose
            await new Promise(resolve => setTimeout(resolve, waitTime));
        }
        this.last_prompt_time = Date.now();
    }

    // --- Prompting Methods ---

    async promptConvo(messages) {
        this.most_recent_msg_time = Date.now();
        let current_msg_time = this.most_recent_msg_time;
        const maxRetries = 3;

        for (let i = 0; i < maxRetries; i++) {
            await this.checkCooldown();
            if (current_msg_time !== this.most_recent_msg_time) {
                console.log(`[${this.getName()}] New message arrived during convo cooldown/prompt prep. Discarding.`);
                return '';
            }

            let prompt = this.profile.conversing;
            if (!prompt) {
                console.error(`[${this.getName()}] 'conversing' prompt template missing.`);
                return "Config error: Cannot generate response.";
            }
            prompt = await this.replaceStrings(prompt, messages, this.convo_examples);

            let generation = '';
            try {
                generation = await this.chat_model.sendRequest(messages, prompt);
                if (!generation || typeof generation !== 'string' || generation.includes("My brain disconnected")) {
                    console.warn(`[${this.getName()}] Convo model returned empty/error on attempt ${i + 1}. Retrying...`);
                    if (i < maxRetries - 1) continue;
                    else generation = "I'm having trouble responding.";
                }
            } catch (error) {
                console.error(`[${this.getName()}] Error during chat_model.sendRequest (convo):`, error);
                if (i < maxRetries - 1) continue;
                else generation = "Error during response generation.";
            }

            if (generation.includes('(FROM OTHER BOT)')) {
                console.warn(`[${this.getName()}] Hallucinated bot tag on attempt ${i + 1}. Retrying...`);
                if (i < maxRetries - 1) continue;
                else return "I seem confused.";
            }

            if (current_msg_time !== this.most_recent_msg_time) {
                console.log(`[${this.getName()}] New message arrived during convo generation. Discarding.`);
                return '';
            }
            return generation;
        }
        console.error(`[${this.getName()}] Failed convo generation after ${maxRetries} attempts.`);
        return "I'm having trouble responding right now.";
    }

    async promptCoding(messages) {
        if (this.awaiting_coding) {
            console.warn(`[${this.getName()}] Already awaiting coding response.`);
            return '```// Skipped: Already processing code.```';
        }
        this.awaiting_coding = true;

        try {
            await this.checkCooldown();
            let prompt = this.profile.coding;
            if (!prompt) {
                console.error(`[${this.getName()}] 'coding' prompt template missing.`);
                return "```// Error: Coding template missing.```";
            }
            prompt = await this.replaceStrings(prompt, messages, this.coding_examples);

            let resp = "```// Error: Failed to get response.```"; // Default error
            try {
                resp = await this.code_model.sendRequest(messages, prompt);
                if (!resp || typeof resp !== 'string') resp = "```// Error: Empty response from code model.```";
            } catch (error) {
                console.error(`[${this.getName()}] Error during code_model.sendRequest:`, error);
            }
            return resp;

        } catch (error) {
            console.error(`[${this.getName()}] Error in promptCoding prep:`, error);
            return "```// Error: Failed during coding prompt prep.```";
        } finally {
            this.awaiting_coding = false; // ALWAYS reset flag
        }
    }

    async promptMemSaving(to_summarize) {
        await this.checkCooldown();
        let prompt = this.profile.saving_memory;
        if (!prompt) {
            console.error(`[${this.getName()}] 'saving_memory' prompt template missing.`);
            return "Error: Cannot save memory.";
        }
        prompt = await this.replaceStrings(prompt, null, null, Array.isArray(to_summarize) ? to_summarize : []);

        try {
            const summary = await this.chat_model.sendRequest([], prompt);
            return summary || "Could not generate summary.";
        } catch (error) {
            console.error(`[${this.getName()}] Error during memory saving:`, error);
            return "Error saving memory.";
        }
    }

    async promptShouldRespondToBot(new_message) {
        await this.checkCooldown();
        let prompt = this.profile.bot_responder;
        if (!prompt) {
            console.warn(`[${this.getName()}] 'bot_responder' template missing. Defaulting to not respond.`);
            return false;
        }

        let messages = this.agent.history.getHistory(5);
        messages.push({ role: 'user', content: new_message });
        prompt = await this.replaceStrings(prompt, messages);

        try {
            let res = await this.chat_model.sendRequest([], prompt);
            const decision = res?.trim().toLowerCase();
            // console.log(`[${this.getName()}] Bot responder raw: '${res}', Decision: '${decision}'`); // Debugging
            return ['respond', 'yes', 'true'].includes(decision);
        } catch (error) {
            console.error(`[${this.getName()}] Error during bot responder check:`, error);
            return false;
        }
    }

    async promptVision(messages, imageBuffer) {
        if (!this.vision_model || typeof this.vision_model.sendVisionRequest !== 'function') {
            console.error(`[${this.getName()}] Vision model misconfigured or missing sendVisionRequest method.`);
            return "Error: Vision not available.";
        }

        await this.checkCooldown();
        let prompt = this.profile.image_analysis;
        if (!prompt) {
            console.error(`[${this.getName()}] 'image_analysis' template missing.`);
            return "Error: Cannot analyze image.";
        }
        prompt = await this.replaceStrings(prompt, messages); // Prepare text prompt

        let analysisResult = "Error: Analysis failed."; // Default error
        try {
            console.log(`[${this.getName()}] Sending image for analysis...`);
            // The vision model's sendVisionRequest handles the API call and *text* logging
            analysisResult = await this.vision_model.sendVisionRequest(messages, prompt, imageBuffer);

            // --- Log vision data AFTER getting the result ---
            // Check settings flag AND if the result seems valid (not an error message)
            if (settings.log_vision_data && analysisResult && typeof analysisResult === 'string' && !analysisResult.startsWith("Error:") && !analysisResult.includes("My brain disconnected")) {
                console.log(`[${this.getName()}] Logging vision data.`);
                logVision(imageBuffer, analysisResult); // Call the dedicated vision logger
            } else {
                 console.warn(`[${this.getName()}] Vision analysis failed or returned error, skipping vision log.`);
            }

            return analysisResult || "Could not analyze the image."; // Handle empty response

        } catch (error) {
            console.error(`[${this.getName()}] Error during vision request execution:`, error);
            // analysisResult remains the default error message
            return analysisResult;
        }
    }


    async promptGoalSetting(messages, last_goals) {
        let system_message = this.profile.goal_setting;
        if (!system_message) {
            console.error(`[${this.getName()}] 'goal_setting' template missing.`);
            return null;
        }
        system_message = await this.replaceStrings(system_message, messages); // Replace in system prompt

        let user_message_template = 'Use the below info to determine what goal to target next:\n\n$LAST_GOALS\n$STATS\n$INVENTORY\n$CONVO';
        let user_message = await this.replaceStrings(user_message_template, messages, null, null, last_goals || {});
        let requestMessages = [{ role: 'user', content: user_message }];

        try {
            await this.checkCooldown();
            const rawResponse = await this.chat_model.sendRequest(requestMessages, system_message);

            if (!rawResponse || typeof rawResponse !== 'string') {
                console.warn(`[${this.getName()}] Goal setting returned empty/invalid response.`);
                return null;
            }

            let goal = null;
            try {
                // Robust JSON extraction
                const jsonMatch = rawResponse.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
                let data = null;
                if (jsonMatch && jsonMatch[1]) {
                    data = jsonMatch[1].trim();
                } else if (rawResponse.trim().startsWith('{') && rawResponse.trim().endsWith('}')) {
                    data = rawResponse.trim(); // Assume whole response is JSON
                }

                if (data) {
                    goal = JSON.parse(data);
                } else {
                    console.warn(`[${this.getName()}] No JSON block found in goal response: ${rawResponse.substring(0, 100)}...`);
                }
            } catch (err) {
                console.error(`[${this.getName()}] Failed to parse goal JSON: ${rawResponse}`, err);
                return null;
            }

            // Validate goal structure
            if (!goal || typeof goal !== 'object' || !goal.name || typeof goal.name !== 'string' || goal.quantity === undefined || isNaN(parseInt(goal.quantity))) {
                console.warn(`[${this.getName()}] Invalid goal structure:`, goal);
                return null;
            }

            goal.quantity = parseInt(goal.quantity);
            console.log(`[${this.getName()}] Parsed goal:`, goal);
            return goal;

        } catch (error) {
            console.error(`[${this.getName()}] Error during goal setting request:`, error);
            return null;
        }
    }
}

// --- END OF FILE prompter.js ---