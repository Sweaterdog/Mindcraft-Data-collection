// --- START OF FILE grok.js ---

import OpenAIApi from 'openai';
import { getKey } from '../utils/keys.js';
import { log } from '../../logger.js'; // <-- IMPORT log

// xAI Grok model using OpenAI compatible API
export class Grok {
    constructor(model_name, url, params) {
        this.model_name = model_name;
        this.url = url;
        this.params = params;

        let config = {};
        config.baseURL = url || "https://api.x.ai/v1"; // Default xAI endpoint

        config.apiKey = getKey('XAI_API_KEY');
        if (!config.apiKey) {
             throw new Error('XAI_API_KEY not found in keys.');
        }

        this.openai = new OpenAIApi(config);
    }

    async sendRequest(turns, systemMessage, stop_seq='***') {
        let messages = [{'role': 'system', 'content': systemMessage}].concat(turns);
        // Note: Grok might not need strictFormat, adjust if needed based on API behavior
        let modelInput = messages; // Capture input for logging

        const pack = {
            model: this.model_name || "grok-1", // Default to grok-1 if not specified
            messages: modelInput, // Use modelInput
            stop: stop_seq ? [stop_seq] : undefined, // Use array or undefined for stop
            ...(this.params || {})
        };

        let rawResponse = null; // To store the raw response

        try {
            console.log('Awaiting xai (Grok) api response...');
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
             console.error("[Grok] Error:", err); // Log the actual error
             // Handle context length error for retry
            if ((err.message == 'Context length exceeded' || (err.response && err.response.status === 400 && err.message.includes('context'))) && turns.length > 1) {
                console.log('Context length exceeded, trying again with shorter context.');
                rawResponse = 'Context length exceeded, retrying...';
                log(modelInput, rawResponse); // Log the specific state
                // Recursive call will handle its own logging
                return await this.sendRequest(turns.slice(1), systemMessage, stop_seq);
            }
            // Handle specific vision error
            else if (err.message && err.message.includes('expects a single `text` element per message')) {
                 rawResponse = 'Vision is only supported by certain models (or message format incorrect).';
                 log(modelInput, rawResponse);
            }
             // Handle other errors
            else {
                 rawResponse = 'My brain disconnected, try again.'; // Assign generic error
                 log(modelInput, rawResponse); // Log the generic error
            }
        }

        // Post-processing after potential errors handled and logging
        let finalRes = rawResponse;
        if (finalRes && typeof finalRes === 'string') {
             // Replace special token if it exists, *after* logging the raw response
            finalRes = finalRes.replace(/<\|separator\|>/g, '*no response*');
        }

        return finalRes; // Return potentially processed response or error message
    }

    // Handles vision request by formatting messages and calling sendRequest
    async sendVisionRequest(messages, systemMessage, imageBuffer) {
        const imageMessages = messages.filter(message => message.role !== 'system'); // Start with non-system messages
         const visionUserMessage = {
             role: "user",
             content: [
                 // Combine system prompt and image in one user message if needed by API
                 { type: "text", text: systemMessage }, // Adjust prompt text as needed
                 {
                     type: "image_url",
                     image_url: {
                         // Use standard base64 data URL format
                         url: `data:image/jpeg;base64,${imageBuffer.toString('base64')}`
                     }
                 }
             ]
         };
        imageMessages.push(visionUserMessage);

        // Call sendRequest with formatted messages; logging happens inside sendRequest
        // The system message for the sendRequest call itself might be empty now
        return this.sendRequest(imageMessages, ""); // Pass empty system message if prompt is in content
    }

    async embed(text) {
        console.warn('Embeddings are not supported by the Grok provider.');
        throw new Error('Embeddings are not supported by Grok.');
    }
}
// --- END OF FILE grok.js ---