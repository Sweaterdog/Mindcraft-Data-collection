// --- START OF FILE glhf.js ---

import OpenAIApi from 'openai';
import { getKey } from '../utils/keys.js';
import { log } from '../../logger.js'; // <-- IMPORT log

export class GLHF {
    constructor(model_name, url) {
        this.model_name = model_name;
        const apiKey = getKey('GHLF_API_KEY');
        if (!apiKey) {
            throw new Error('API key not found. Please check keys.json and ensure GHLF_API_KEY is defined.');
        }
        this.openai = new OpenAIApi({
            apiKey,
            baseURL: url || "https://glhf.chat/api/openai/v1"
        });
    }

    async sendRequest(turns, systemMessage, stop_seq = '***') {
        // Construct the message array for the API request.
        let messages = [{ role: 'system', content: systemMessage }].concat(turns);
        let modelInput = messages; // Capture input for logging

        const pack = {
            model: this.model_name || "hf:meta-llama/Llama-3.1-405B-Instruct",
            messages: modelInput, // Use modelInput
            stop: [stop_seq]
        };

        const maxAttempts = 5; // Retry for partial <think>
        let attempt = 0;
        let finalRes = null; // Will hold the final result after processing
        let rawResponse = null; // Will hold the raw response before processing

        while (attempt < maxAttempts) {
            attempt++;
            console.log(`Awaiting glhf.chat API response... (attempt: ${attempt})`);
            rawResponse = null; // Reset raw response for each attempt
            try {
                let completion = await this.openai.chat.completions.create(pack);
                rawResponse = completion.choices[0].message.content; // Store raw response

                // --- Log successful raw response ---
                log(modelInput, rawResponse);

                // --- Post-logging checks and processing ---
                if (completion.choices[0].finish_reason === 'length') {
                     // Logged above, now handle the error for retry logic
                    throw new Error('Context length exceeded');
                }

                // Check for partial <think> block *after* logging
                if (rawResponse.includes("<think>") && !rawResponse.includes("</think>")) {
                    console.warn("Partial <think> block detected. Re-generating...");
                    finalRes = null; // Ensure we retry
                    continue; // Go to next attempt
                }

                 // If </think> exists without <think>, prepend it (less common)
                 if (rawResponse.includes("</think>") && !rawResponse.includes("<think>")) {
                     console.warn("Found </think> without <think>, prepending <think>.");
                     // Note: We logged the raw response already. The finalRes will be modified.
                     finalRes = "<think>" + rawResponse;
                 } else {
                     finalRes = rawResponse; // Assign raw response if no </think> issue
                 }

                // Replace separator token *after* think block handling
                finalRes = finalRes.replace(/<\|separator\|>/g, '*no response*');

                break; // Valid response obtained and processed.

            } catch (err) {
                 console.error("[GLHF] Error:", err); // Log the actual error
                if ((err.message === 'Context length exceeded' || err.code === 'context_length_exceeded') && turns.length > 1) {
                    console.log('Context length exceeded, trying again with shorter context.');
                    // Log the error state before retrying
                    rawResponse = 'Context length exceeded, retrying...';
                    log(modelInput, rawResponse); // Log the specific state
                    // Recursive call will handle its own logging
                    return await this.sendRequest(turns.slice(1), systemMessage, stop_seq);
                } else {
                    rawResponse = 'My brain disconnected, try again.'; // Assign generic error
                    log(modelInput, rawResponse); // Log the generic error
                    finalRes = rawResponse; // Assign error to final result
                    break; // Exit loop on other errors
                }
            }
        } // End while loop

        // If loop finished without a valid finalRes (e.g., max attempts on partial <think>)
        if (finalRes === null) {
            finalRes = "I thought too hard, sorry, try again";
            rawResponse = finalRes; // Set rawResponse for logging consistency
            log(modelInput, rawResponse); // Log the fallback error
        }

        return finalRes; // Return the final processed response or error
    }

    async embed(text) {
        console.warn('Embeddings are not supported by the glhf provider.');
        throw new Error('Embeddings are not supported by glhf.');
    }
}
// --- END OF FILE glhf.js ---