// --- START OF FILE openrouter.js ---

import OpenAIApi from 'openai';
import { getKey } from '../utils/keys.js'; // Removed hasKey as it wasn't used
import { strictFormat } from '../utils/text.js';
import { log } from '../../logger.js'; // <-- IMPORT log

export class OpenRouter {
    constructor(model_name, url, params) { // Added params
        this.model_name = model_name ? model_name.replace('openrouter/', '') : null; // Clean name
        this.url = url || 'https://openrouter.ai/api/v1';
        this.params = params || {}; // Store params

        let config = {};
        config.baseURL = this.url;

        const apiKey = getKey('OPENROUTER_API_KEY');
        if (!apiKey) {
             throw new Error('OPENROUTER_API_KEY not found in keys.');
        }
        config.apiKey = apiKey;

        // Add default headers expected by OpenRouter
        config.defaultHeaders = {
             'HTTP-Referer': 'https://github.com/askcaspar/mindcraft', // Replace with your app URL/name
             'X-Title': 'MindCraft', // Replace with your app title
        };


        this.openai = new OpenAIApi(config);
    }

    async sendRequest(turns, systemMessage, stop_seq='***') { // Default stop sequence changed
        let messages = [{ role: 'system', content: systemMessage }];
        messages.push(...strictFormat(turns)); // Apply strict format
        let modelInput = messages; // Capture input for logging

        // Ensure model name is provided
        if (!this.model_name) {
            console.warn("OpenRouter model name not specified in profile. Using default: 'openai/gpt-3.5-turbo'");
            this.model_name = 'openai/gpt-3.5-turbo'; // Provide a default if needed
        }

        // Build the request package, merging with constructor params
        const pack = {
            model: this.model_name, // Use the potentially cleaned name
            messages: modelInput,
            stop: stop_seq ? [stop_seq] : undefined, // API expects array or undefined
            // Add other parameters from constructor, e.g., temperature, max_tokens
            temperature: this.params.temperature,
            max_tokens: this.params.max_tokens,
            top_p: this.params.top_p,
            // frequency_penalty: this.params.frequency_penalty, // Example additional param
            // presence_penalty: this.params.presence_penalty,  // Example additional param
            ...(this.params || {}) // Spread remaining params cautiously
        };
        // Remove params explicitly handled above to avoid duplication
        delete pack.temperature;
        delete pack.max_tokens;
        delete pack.top_p;
        delete pack.stop;


        let rawResponse = null; // Store raw response content

        try {
            console.log(`Awaiting openrouter api response (model: ${pack.model})...`);
           // console.log("Payload:", JSON.stringify(pack)); // Debug payload if needed
            let completion = await this.openai.chat.completions.create(pack);

            if (!completion?.choices?.[0]?.message?.content) {
                // Handle cases where the structure is unexpected or content is missing
                console.error('Invalid response structure or missing content from OpenRouter:', JSON.stringify(completion));
                rawResponse = 'Received invalid response structure from OpenRouter.';
                // Still log this state
                 log(modelInput, rawResponse);
                 return rawResponse; // Return the error message directly
            }

            rawResponse = completion.choices[0].message.content; // Store raw response

            // --- Log successful raw response ---
            log(modelInput, rawResponse);

            // --- Post-logging checks ---
            if (completion.choices[0].finish_reason === 'length') {
                 console.warn("OpenRouter response may have been truncated due to token limits.");
                 // Don't throw, just warn
            }
            console.log('Received.');

        } catch (err) {
            console.error('[OpenRouter] Error:', err); // Log the actual error object
            let errorMsg = err.message || 'Unknown error';
             // Try to extract more details from the error response if available
             if (err.response && err.response.data && err.response.data.error) {
                 errorMsg = `API Error: ${err.response.data.error.message || JSON.stringify(err.response.data.error)}`;
             } else if (err.status) {
                  errorMsg = `HTTP Error: ${err.status}`;
             }

            // Handle context length error specifically for retry if needed
            if (errorMsg.includes('context length') && turns.length > 1) {
                 console.log('Context length exceeded, trying again with shorter context.');
                 rawResponse = 'Context length exceeded, retrying...';
                 log(modelInput, rawResponse);
                 // Recursive call for retry
                 return await this.sendRequest(turns.slice(1), systemMessage, stop_seq);
            }
             // Handle vision errors
            else if (errorMsg.includes("doesn't support image input") || errorMsg.includes("Invalid type for path `messages")) {
                rawResponse = 'Vision is only supported by certain models (or message format incorrect).';
            }
             // Generic fallback
            else {
                 rawResponse = `My brain disconnected, try again. Error: ${errorMsg}`;
            }
            log(modelInput, rawResponse); // Log the error state
        }

        // Post-processing after logging and error handling
         let finalRes = rawResponse;
         if (finalRes && typeof finalRes === 'string' && finalRes.includes('<think>') && finalRes.includes('</think>')) {
             //console.log("Removing think block from OpenRouter final response.");
             finalRes = finalRes.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
         }

        return finalRes; // Return potentially processed response or error message
    }

    // Handles vision request by formatting messages and calling sendRequest
    async sendVisionRequest(messages, systemMessage, imageBuffer) {
        const imageMessages = messages.filter(message => message.role !== 'system');
        const visionUserMessage = {
            role: "user",
            content: [
                { type: "text", text: systemMessage }, // Combine prompt text
                {
                    type: "image_url",
                    image_url: {
                        url: `data:image/jpeg;base64,${imageBuffer.toString('base64')}`
                    }
                }
            ]
        };
        imageMessages.push(visionUserMessage);

        // Call sendRequest with formatted messages. Logging happens inside.
        // System message likely empty. Ensure the selected OpenRouter model supports vision.
        console.warn(`Attempting vision request via OpenRouter model: ${this.model_name}. Ensure this model supports vision.`);
        return this.sendRequest(imageMessages, "");
    }

    async embed(text) {
        console.warn('Embeddings via OpenRouter depend on the specific model. Not directly implemented here.');
        throw new Error('Embeddings via OpenRouter require specifying a compatible model and are not directly supported by this provider.');
        // Implementation would involve calling openai.embeddings.create with a model like 'text-embedding-ada-002' etc.
    }
}
// --- END OF FILE openrouter.js ---