// --- START OF FILE hyperbolic.js ---

import { getKey } from '../utils/keys.js';
import { strictFormat } from '../utils/text.js'; // Import strictFormat
import { log } from '../../logger.js'; // <-- IMPORT log

export class Hyperbolic {
    constructor(modelName, apiUrl, params) { // Added params
        this.modelName = modelName || "deepseek-ai/DeepSeek-V3"; // Keep default or use provided
        this.apiUrl = apiUrl || "https://api.hyperbolic.xyz/v1/chat/completions";
        this.params = params || {}; // Store params

        // Retrieve the Hyperbolic API key
        this.apiKey = getKey('HYPERBOLIC_API_KEY');
        if (!this.apiKey) {
            throw new Error('HYPERBOLIC_API_KEY not found. Check your keys.js file.');
        }
    }

    async sendRequest(turns, systemMessage, stopSeq = '***') {
        // Prepare messages, applying strictFormat
        const messages = [{ role: 'system', content: systemMessage }];
        messages.push(...strictFormat(turns)); // Format turns
        let modelInput = messages; // Capture input for logging

        // Build the request payload, merging with constructor params
        const payload = {
            model: this.modelName,
            messages: modelInput,
            max_tokens: this.params.max_tokens || 8192, // Use param or default
            temperature: this.params.temperature ?? 0.7, // Use param or default (nullish coalescing)
            top_p: this.params.top_p ?? 0.9, // Use param or default
            stream: false, // Keep stream false for simplicity
            stop: stopSeq ? [stopSeq] : undefined, // Add stop sequence if provided
            ...(this.params || {}) // Spread other params cautiously
        };
        // Remove params already explicitly handled to avoid duplication if they exist in this.params
        delete payload.max_tokens;
        delete payload.temperature;
        delete payload.top_p;
        delete payload.stop;


        const maxAttempts = 5; // Retries for partial <think>
        let attempt = 0;
        let finalRes = null; // Final processed result
        let rawResponse = null; // Raw response content

        while (attempt < maxAttempts) {
            attempt++;
            rawResponse = null; // Reset raw response
            console.log(`Awaiting Hyperbolic API response... (attempt: ${attempt})`);
           // console.log('Payload:', JSON.stringify(payload)); // Debug payload if needed

            try {
                const response = await fetch(this.apiUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.apiKey}`
                    },
                    body: JSON.stringify(payload) // Send the constructed payload
                });

                if (!response.ok) {
                     // Try to get error details from response body
                     let errorBody = '';
                     try { errorBody = await response.text(); } catch (_) { /* ignore */ }
                     console.error(`Hyperbolic API Error: ${response.status} ${response.statusText}`, errorBody);
                    throw new Error(`HTTP error! status: ${response.status} - ${errorBody || response.statusText}`);
                }

                const data = await response.json();
                rawResponse = data?.choices?.[0]?.message?.content || ''; // Store raw response

                // --- Log successful raw response ---
                log(modelInput, rawResponse);

                // --- Post-logging checks ---
                if (data?.choices?.[0]?.finish_reason === 'length') {
                    console.warn("Hyperbolic response may have been truncated due to token limits.");
                    // Don't throw error, just warn
                }

                console.log('Received response from Hyperbolic.');

                // Check for <think> blocks *after* logging
                const hasOpenTag = rawResponse.includes("<think>");
                const hasCloseTag = rawResponse.includes("</think>");

                if (hasOpenTag && !hasCloseTag) {
                    console.warn("Partial <think> block detected. Re-generating...");
                    finalRes = null; // Ensure retry
                    continue; // Next attempt
                }

                 // Handle </think> without <think>
                 if (hasCloseTag && !hasOpenTag) {
                     console.warn("Found </think> without <think>, prepending <think>.");
                     finalRes = '<think>' + rawResponse;
                 } else {
                      finalRes = rawResponse; // Assign raw if no </think> issue
                 }

                 // Remove complete think blocks *after* handling partials
                 if (finalRes.includes("<think>") && finalRes.includes("</think>")) {
                    // console.log("Removing think block from Hyperbolic final response.");
                     finalRes = finalRes.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
                 }

                 // Replace separator token
                 finalRes = finalRes.replace(/<\|separator\|>/g, '*no response*');

                 break; // Valid response obtained and processed

            } catch (err) {
                console.error("[Hyperbolic] Error during request or processing:", err);
                 // Handle context length specifically if possible (might be in HTTP error message)
                if (err.message.includes('Context length exceeded') && turns.length > 1) {
                    console.log('Context length likely exceeded, trying again with shorter context...');
                     rawResponse = 'Context length exceeded, retrying...';
                     log(modelInput, rawResponse);
                    // Recursive call for retry
                    return await this.sendRequest(turns.slice(1), systemMessage, stopSeq);
                } else {
                     rawResponse = 'My brain disconnected, try again.'; // Generic error
                     log(modelInput, rawResponse); // Log the error state
                     finalRes = rawResponse; // Assign error to final result
                     break; // Exit loop on other errors
                }
            }
        } // End while loop

        // Fallback if loop finished without success
        if (finalRes == null) {
            console.warn("Could not get a valid response after max attempts.");
            finalRes = 'I thought too hard, sorry, try again.';
            log(modelInput, finalRes); // Log fallback error
        }

        return finalRes.trim(); // Return final processed result
    }

    async embed(text) {
        console.warn('Embeddings are not supported by the Hyperbolic provider.');
        throw new Error('Embeddings are not supported by Hyperbolic.');
    }
}
// --- END OF FILE hyperbolic.js ---