// --- START OF FILE local.js ---

import { strictFormat } from '../utils/text.js';
import { log } from '../../logger.js'; // <-- IMPORT log

export class Local {
    constructor(model_name, url, params) {
        this.model_name = model_name; // Keep provided name (e.g., llama3:instruct)
        this.params = params || {}; // Store params
        this.url = url || 'http://127.0.0.1:11434'; // Default Ollama URL
        this.chat_endpoint = '/api/chat';
        this.embedding_endpoint = '/api/embeddings';
    }

    async sendRequest(turns, systemMessage) {
        let model = this.model_name || 'llama3.1'; // Use provided name or default
        let messages = [{ role: 'system', content: systemMessage }];
        messages.push(...strictFormat(turns)); // Apply formatting
        let modelInput = messages; // Capture input for logging

        // Prepare payload, merging constructor params
        const body = {
            model: model,
            messages: modelInput,
            stream: false, // Keep stream false
            options: { // Ollama uses 'options' for parameters
                temperature: this.params.temperature,
                top_p: this.params.top_p,
                num_ctx: this.params.num_ctx || this.params.max_tokens || 4096, // Map max_tokens if present
                stop: this.params.stop ? [this.params.stop] : undefined, // Use stop param if provided
                // Add other common Ollama options from params if needed
                // num_predict: this.params.num_predict,
                // repeat_penalty: this.params.repeat_penalty,
            },
            ...(this.params.ollama_options || {}) // Allow passing raw Ollama options directly
        };
         // Remove params handled in 'options' from top level if they exist in this.params
         delete body.temperature;
         delete body.top_p;
         delete body.num_ctx;
         delete body.stop;


        const maxAttempts = 5; // Retries for partial <think> (if applicable)
        let attempt = 0;
        let finalRes = null; // Final processed result
        let rawResponse = null; // Raw response content

        while (attempt < maxAttempts) {
            attempt++;
            rawResponse = null; // Reset raw response
            console.log(`Awaiting local response (model: ${model}, attempt: ${attempt})...`);

            try {
                const responseData = await this.send(this.chat_endpoint, body); // Call helper method

                if (responseData && responseData.message && responseData.message.content) {
                    rawResponse = responseData.message.content; // Store raw response
                     // --- Log successful raw response ---
                     log(modelInput, rawResponse);

                    // --- Post-logging checks ---
                    // Check Ollama specific finish reasons if needed (e.g., 'length')
                    if (responseData.done_reason === 'length' || responseData.total_duration === 0) { // Check for length or potential timeout/error indicators
                          console.warn("Ollama response may have been truncated or generation failed.");
                          // Don't throw, just warn, we have the partial response
                    }
                     console.log('Received.');

                } else {
                     console.error("Invalid response structure from local API:", responseData);
                     rawResponse = 'Received invalid response structure from local API.';
                     log(modelInput, rawResponse); // Log the error state
                     finalRes = rawResponse; // Assign error to final result
                     break; // Exit loop on structure error
                }

                // Check for <think> blocks *after* logging (especially for reasoning models)
                const hasOpenTag = rawResponse.includes("<think>");
                const hasCloseTag = rawResponse.includes("</think>");

                // Check for partial mismatch and retry if needed
                if (hasOpenTag && !hasCloseTag) {
                     // Only retry if the model name suggests it might do reasoning
                     if (model.includes('reasoning') || model.includes('deepseek')) {
                         console.warn("Partial <think> block detected with reasoning model. Re-generating...");
                         finalRes = null; // Ensure retry
                         continue; // Next attempt
                     } else {
                          console.warn("Detected unterminated <think> block, but not retrying as model might not support it. Truncating.");
                          finalRes = rawResponse.substring(0, rawResponse.indexOf('<think>')).trim(); // Truncate before partial block
                     }

                } else if (hasCloseTag && !hasOpenTag) {
                     console.warn("Found </think> without <think>, prepending <think>.");
                     finalRes = '<think>' + rawResponse;
                 } else {
                      finalRes = rawResponse; // Assign raw if no </think> issue
                 }


                // Remove complete think blocks *after* handling partials
                 if (finalRes.includes("<think>") && finalRes.includes("</think>")) {
                    // console.log("Removing think block from local final response.");
                     finalRes = finalRes.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
                 }

                break; // Valid response obtained and processed.

            } catch (err) {
                // Error handling moved inside the send helper, but catch potential rethrows
                 console.error("[Local] Error during sendRequest processing:", err);
                 rawResponse = `My brain disconnected, try again. Error: ${err.message || 'Unknown fetch error'}`;
                 log(modelInput, rawResponse); // Log the error state
                 finalRes = rawResponse; // Assign error to final result
                 // Check for context length error to allow retry
                 if (err.message && err.message.toLowerCase().includes('context length') && turns.length > 1) {
                    console.log('Context length exceeded, trying again with shorter context.');
                    // Log specific state before retry
                    log(modelInput, 'Context length exceeded, retrying...');
                    // Recursive call handles its own logging
                    return await this.sendRequest(turns.slice(1), systemMessage); // Retry with sliced turns
                 }
                 break; // Exit loop on other errors
            }
        } // End while loop

        // Fallback if loop finished without success
        if (finalRes === null) {
            console.warn("Could not get a valid response after max attempts.");
            finalRes = 'I thought too hard, sorry, try again.';
            log(modelInput, finalRes); // Log fallback error
        }

        return finalRes.trim(); // Return final processed result
    }


    async embed(text) {
         if (!text || typeof text !== 'string') {
             console.error("Invalid input text for embedding:", text);
             throw new Error("Invalid input for embedding.");
         }
        // Use specified embedding model or a common default like nomic-embed-text
        let model = this.params?.embedding_model || 'nomic-embed-text';
        let body = {
             model: model,
             prompt: text // Ollama uses 'prompt' for embeddings, not 'input'
        };

        try {
             console.log(`Requesting local embedding (model: ${model})...`);
             let responseData = await this.send(this.embedding_endpoint, body);
             if (responseData && responseData.embedding && Array.isArray(responseData.embedding)) {
                 return responseData.embedding;
             } else {
                  console.error("Invalid embedding response structure from local API:", responseData);
                  throw new Error('Invalid embedding response structure received.');
             }
        } catch (err) {
             console.error("[Local Embed] Error creating embedding:", err);
             throw new Error(`Local embedding creation failed: ${err.message || 'Unknown fetch error'}`);
        }
    }

    // Helper function to send requests to Ollama endpoint
    async send(endpoint, body) {
        const url = new URL(endpoint, this.url);
        const method = 'POST';
        const headers = new Headers({ 'Content-Type': 'application/json' });

        try {
            const response = await fetch(url, {
                method,
                headers,
                body: JSON.stringify(body)
            });

            if (!response.ok) {
                 // Try to get error message from Ollama response body
                 let errorMsg = `Ollama Status: ${response.status} ${response.statusText}`;
                 try {
                     const errorData = await response.json();
                     errorMsg = errorData.error || errorMsg; // Ollama often returns { "error": "..." }
                 } catch (_) { /* Ignore if response is not JSON */ }

                 // Throw an error that includes the specific message
                 console.error("Ollama API Error:", errorMsg);
                 throw new Error(errorMsg);
            }

            // If response is OK, parse JSON
            const data = await response.json();
            return data;

        } catch (err) {
             // Catch fetch errors (network issues) or errors thrown above
             console.error('Failed to send Ollama request:', err);
             // Re-throw the error to be handled by the calling function (sendRequest/embed)
             throw err;
        }
    }
}
// --- END OF FILE local.js ---