// --- START OF FILE gpt.js ---

import OpenAIApi from 'openai';
import { getKey, hasKey } from '../utils/keys.js';
import { strictFormat } from '../utils/text.js';
import { log } from '../../logger.js'; // <-- IMPORT log

export class GPT {
    constructor(model_name, url, params) {
        this.model_name = model_name;
        this.params = params;

        let config = {};
        if (url)
            config.baseURL = url;

        if (hasKey('OPENAI_ORG_ID'))
            config.organization = getKey('OPENAI_ORG_ID');

        config.apiKey = getKey('OPENAI_API_KEY');
        if (!config.apiKey) {
            throw new Error('OPENAI_API_KEY not found in keys.');
        }


        this.openai = new OpenAIApi(config);
    }

    async sendRequest(turns, systemMessage, stop_seq='***') {
        let messages = [{'role': 'system', 'content': systemMessage}].concat(turns);
        let modelInput = strictFormat(messages); // Apply strict format here
        // Note: o1 models might require different formatting handled below

        const pack = {
            model: this.model_name || "gpt-3.5-turbo",
            messages: modelInput, // Use formatted input
            stop: stop_seq ? [stop_seq] : undefined, // API expects array or undefined
            ...(this.params || {})
        };

        // Specific handling for o1 models (potentially removing stop sequence)
        if (this.model_name && this.model_name.includes('o1')) {
            // o1 might use a different format; strictFormat might need adjustment
            // For now, just delete stop as before
             console.warn("Applying o1 model specific adjustments (removing stop sequence).");
             delete pack.stop;
             // Re-capture modelInput if formatting changes for o1
             // modelInput = modified_format_for_o1(messages);
             // pack.messages = modelInput;
        }

        let rawResponse = null; // To store raw response

        try {
            console.log(`Awaiting openai api response from model ${pack.model}...`);
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
            console.error("[GPT] Error:", err); // Log the actual error
            // Handle context length error for retry
            if ((err.message == 'Context length exceeded' || err.code == 'context_length_exceeded' || (err.response && err.response.status === 400 && err.message.includes('context'))) && turns.length > 1) {
                console.log('Context length exceeded, trying again with shorter context.');
                 rawResponse = 'Context length exceeded, retrying...';
                 log(modelInput, rawResponse); // Log the specific state
                // Recursive call will handle its own logging
                return await this.sendRequest(turns.slice(1), systemMessage, stop_seq);
            }
            // Handle vision format error
            else if (err.message && (err.message.includes('image_url') || err.message.includes('Invalid type for path `messages')) ) { // Broader check for vision errors
                 rawResponse = 'Vision is only supported by certain models (or message format incorrect).';
                 log(modelInput, rawResponse);
            }
            // Handle other errors
            else {
                 rawResponse = 'My brain disconnected, try again.'; // Assign generic error
                 log(modelInput, rawResponse); // Log the generic error
            }
        }
        // Return the raw response (or error message)
        return rawResponse;
    }

    // Handles vision request by formatting messages and calling sendRequest
    async sendVisionRequest(messages, systemMessage, imageBuffer) {
        const imageMessages = messages.filter(message => message.role !== 'system'); // Start with non-system
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

        // Call sendRequest with the formatted messages. Logging happens inside sendRequest.
        // Pass empty system message as the prompt is now in the user content.
        return this.sendRequest(imageMessages, "");
    }

    async embed(text) {
        if (!text || typeof text !== 'string') {
             console.error("Invalid input text for embedding:", text);
             throw new Error("Invalid input for embedding.");
        }
        // Truncate text if it exceeds the model's limit (e.g., 8191 tokens for some models)
        const MAX_EMBED_LENGTH = 8191; // Adjust if needed for the specific embedding model
        if (text.length > MAX_EMBED_LENGTH) {
             console.warn(`Embedding text truncated from ${text.length} to ${MAX_EMBED_LENGTH} characters.`);
            text = text.slice(0, MAX_EMBED_LENGTH);
        }
        try {
            const embedding = await this.openai.embeddings.create({
                model: this.params?.embedding_model || "text-embedding-3-small", // Allow overriding model via params
                input: text,
                encoding_format: "float",
            });
            if (embedding && embedding.data && embedding.data.length > 0 && embedding.data[0].embedding) {
                 return embedding.data[0].embedding;
            } else {
                 throw new Error("Invalid embedding response structure received.");
            }
        } catch(err) {
            console.error("[GPT Embed] Error creating embedding:", err);
             // Provide a more informative error message if possible
             const errorMsg = err.response ? `${err.response.status}: ${err.response.data?.error?.message}` : err.message;
             throw new Error(`Embedding creation failed: ${errorMsg}`);
        }
    }
}
// --- END OF FILE gpt.js ---