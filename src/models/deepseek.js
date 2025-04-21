// --- START OF FILE deepseek.js ---

import OpenAIApi from 'openai';
import { getKey } from '../utils/keys.js'; // Removed hasKey as it wasn't used
import { strictFormat } from '../utils/text.js';
import { log } from '../../logger.js'; // <-- IMPORT log

export class DeepSeek {
    constructor(model_name, url, params) {
        this.model_name = model_name;
        this.params = params;

        let config = {};

        config.baseURL = url || 'https://api.deepseek.com';
        config.apiKey = getKey('DEEPSEEK_API_KEY');
        if (!config.apiKey) {
             throw new Error('DEEPSEEK_API_KEY not found in keys.');
        }

        this.openai = new OpenAIApi(config);
    }

    async sendRequest(turns, systemMessage, stop_seq='***') {
        let messages = [{'role': 'system', 'content': systemMessage}].concat(turns);
        messages = strictFormat(messages); // Apply formatting
        let modelInput = messages; // Capture formatted input for logging

        const pack = {
            model: this.model_name || "deepseek-chat",
            messages: modelInput, // Use modelInput
            stop: stop_seq,
            ...(this.params || {})
        };

        let rawResponse = null; // To store the raw response for logging

        try {
            console.log('Awaiting deepseek api response...');
            let completion = await this.openai.chat.completions.create(pack);
            rawResponse = completion.choices[0].message.content; // Store raw response

            // --- Log successful raw response ---
            log(modelInput, rawResponse);

            // --- Post-logging checks ---
            if (completion.choices[0].finish_reason == 'length') {
                // Already logged, now throw for retry logic
                throw new Error('Context length exceeded');
            }
            console.log('Received.');

        } catch (err) {
            console.error("[DeepSeek] Error:", err); // Log the actual error
            if ((err.message == 'Context length exceeded' || (err.response && err.response.status === 400)) && turns.length > 1) { // Added status check for potential API errors indicating length issues
                console.log('Context length exceeded, trying again with shorter context.');
                 rawResponse = 'Context length exceeded, retrying...';
                 log(modelInput, rawResponse); // Log the specific state
                // Recursive call will handle its own logging
                return await this.sendRequest(turns.slice(1), systemMessage, stop_seq);
            } else {
                rawResponse = 'My brain disconnected, try again.'; // Assign generic error
                log(modelInput, rawResponse); // Log the generic error
            }
        }
        // Return the raw response (or error message)
        return rawResponse;
    }

    async embed(text) {
        console.warn('Embeddings are not supported by the Deepseek provider.');
        throw new Error('Embeddings are not supported by Deepseek.');
    }
}
// --- END OF FILE deepseek.js ---