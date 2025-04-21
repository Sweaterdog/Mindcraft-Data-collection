const settings = {
    "minecraft_version": "1.21.1", // supports up to 1.21.1
    "host": "127.0.0.1", // or "localhost", "your.ip.address.here"
    "port": 55916,
    "auth": "offline", // or "microsoft"

    // --- Mindserver Configuration ---
    "host_mindserver": true, // if true, the mindserver will be hosted on this machine. otherwise, specify a public IP address
    "mindserver_host": "localhost",
    "mindserver_port": 8080,

    // --- Profile Configuration ---
    "base_profile": "./profiles/defaults/survival.json", // also see creative.json, god_mode.json
    "profiles": [
        "./andy.json",
        // "./profiles/gpt.json",
        // "./profiles/claude.json",
        // "./profiles/gemini.json",
        // "./profiles/llama.json",
        // "./profiles/qwen.json",
        // "./profiles/grok.json",
        // "./profiles/mistral.json",
        // "./profiles/deepseek.json",

        // using more than 1 profile requires you to /msg each bot indivually
        // individual profiles override values from the base profile
    ],

    // --- Agent Behavior & Memory ---
    "load_memory": false, // load memory from previous session
    "init_message": "Respond with hello world and your name", // sends to all on spawn
    "only_chat_with": [], // users that the bots listen to and send general messages to. if empty it will chat publicly
    "speak": false, // allows all bots to speak through system text-to-speech. works on windows, mac, on linux you need to `apt install espeak`
    "language": "en", // translate to/from this language. Supports these language names: https://cloud.google.com/translate/docs/languages
    "show_bot_views": false, // show bot's view in browser at localhost:3000, 3001...
    "narrate_behavior": true, // chat simple automatic actions ('Picking up item!')
    "chat_bot_messages": true, // publicly chat messages to other bots

    // --- Security & Execution ---
    "allow_insecure_coding": false, // allows newAction command and model can write/run code on your computer. enable at own risk
    "allow_vision": false, // allows vision model to interpret screenshots as inputs (enables vision capability)
    "blocked_actions" : [], // commands to disable and remove from docs. Ex: ["!setMode"]
    "code_timeout_mins": -1, // minutes code is allowed to run. -1 for no timeout

    // --- Prompting & Context ---
    "relevant_docs_count": 5, // number of relevant code function docs to select for prompting. -1 for all
    "max_messages": 15, // max number of messages to keep in context
    "num_examples": 2, // number of examples to give to the model
    "max_commands": -1, // max number of commands that can be used in consecutive responses. -1 for no limit
    "verbose_commands": true, // show full command syntax

    // --- Data Logging Settings ---
    "log_normal_data": true,    // Log standard input/output pairs to normal_logs.csv
    "log_reasoning_data": true, // Log input/output pairs where response contains <think> blocks to reasoning_logs.csv
    "log_vision_data": true     // Log vision requests (image path + transcribed text) to vision_logs.csv (requires allow_vision: true)

}; // End of settings object definition

// --- Environment Variable Overrides ---
// These allow changing settings without editing the file directly, useful for deployment/testing
if (process.env.MINECRAFT_PORT) {
    console.log(`Overriding Minecraft port from environment variable: ${process.env.MINECRAFT_PORT}`);
    settings.port = parseInt(process.env.MINECRAFT_PORT, 10); // Ensure it's an integer
}
if (process.env.MINDSERVER_PORT) {
    console.log(`Overriding Mindserver port from environment variable: ${process.env.MINDSERVER_PORT}`);
    settings.mindserver_port = parseInt(process.env.MINDSERVER_PORT, 10); // Ensure it's an integer
}
if (process.env.PROFILES) {
    try {
        const profilesFromEnv = JSON.parse(process.env.PROFILES);
        // Basic validation: check if it's an array and not empty
        if (Array.isArray(profilesFromEnv) && profilesFromEnv.length > 0) {
            console.log(`Overriding profiles from environment variable: ${process.env.PROFILES}`);
            settings.profiles = profilesFromEnv;
        } else {
             console.warn(`Environment variable PROFILES is set but not a valid non-empty JSON array. Using default profiles.`);
        }
    } catch (e) {
        console.error(`Error parsing PROFILES environment variable: ${e}. Using default profiles.`);
    }
}
// Add overrides for logging flags if desired
if (process.env.LOG_NORMAL_DATA) {
    settings.log_normal_data = process.env.LOG_NORMAL_DATA.toLowerCase() === 'true';
    console.log(`Overriding log_normal_data from environment: ${settings.log_normal_data}`);
}
if (process.env.LOG_REASONING_DATA) {
    settings.log_reasoning_data = process.env.LOG_REASONING_DATA.toLowerCase() === 'true';
    console.log(`Overriding log_reasoning_data from environment: ${settings.log_reasoning_data}`);
}
if (process.env.LOG_VISION_DATA) {
    settings.log_vision_data = process.env.LOG_VISION_DATA.toLowerCase() === 'true';
    console.log(`Overriding log_vision_data from environment: ${settings.log_vision_data}`);
}


export default settings;
