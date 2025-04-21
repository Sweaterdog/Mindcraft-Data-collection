// --- START OF FILE qwen.js ---

import OpenAIApi from 'openai';
import { getKey } from '../utils/keys.js'; // Removed hasKey as it wasn't used
import { strictFormat } from '../utils/text.js';
import { log } from '../../logger.js'; // <-- IMPORT log

export class Qwen {
    constructor(model_name, url, params) {
        this.model_name = model_name;
        this.params = params || {}; // Store params
        let config = {};

        config.baseURL = url || 'https://dashscope.aliyuncs.com/compatible-mode/v1';
        config.apiKey = getKey('QWEN_API_KEY');
         if (!config.apiKey) {
             throw new Error('QWEN_API_KEY not found in keys.');
         }

        this.openai = new OpenAIApi(config);
    }

    async sendRequest(turns, systemMessage, stop_seq='***') {
        let messages = [{'role': 'system', 'content': systemMessage}];
        messages.push(...strictFormat(turns)); // Apply formatting
        let modelInput = messages; // Capture input for logging

        // Build request package, merging constructor params
        const pack = {
            model: this.model_name || "qwen-plus", // Default model
            messages: modelInput,
            stop: stop_seq ? [stop_seq] : undefined, // API expects array or undefined
            // Add other parameters from constructor
            temperature: this.params.temperature,
            max_tokens: this.params.max_tokens, // Qwen uses max_tokens
            top_p: this.params.top_p,
            ...(this.params || {}) // Spread remaining params
        };
         // Remove params explicitly handled
         delete pack.temperature;
         delete pack.max_tokens;
         delete pack.top_p;
         delete pack.stop;


        let rawResponse = null; // Store raw response

        try {
            console.log(`Awaiting Qwen api response (model: ${pack.model})...`);
            //console.log("Payload:", JSON.stringify(pack)); // Debug payload
            let completion = await this.openai.chat.completions.create(pack);

             if (!completion?.choices?.[0]?.message?.content) {
                console.error('Invalid response structure or missing content from Qwen:', JSON.stringify(completion));
                // Check for specific error messages in response if available
                 let errorDetail = completion?.error?.message || 'Unknown structure issue';
                 rawResponse = `Received invalid response structure from Qwen: ${errorDetail}`;
                 log(modelInput, rawResponse);
                 return rawResponse;
             }

            rawResponse = completion.choices[0].message.content; // Store raw response

            // --- Log successful raw response ---
            log(modelInput, rawResponse);

            // --- Post-logging checks ---
            if (completion.choices[0].finish_reason == 'length') {
                console.warn("Qwen response may have been truncated due to token limits.");
                // Don't throw, just warn
            }
            console.log('Received.');

        } catch (err) {
             console.error("[Qwen] Error:", err); // Log the actual error object
             let errorMsg = err.message || 'Unknown error';
              if (err.response && err.response.data && err.response.data.error) {
                  // Example: { "error": { "code": "InvalidParameter", "message": "...", "request_id": "..." } }
                  errorMsg = `API Error (${err.response.data.error.code}): ${err.response.data.error.message || JSON.stringify(err.response.data.error)}`;
              } else if (err.status) {
                   errorMsg = `HTTP Error: ${err.status}`;
              }


            // Handle context length error for retry
            if ((errorMsg.includes('context length') || (err.status === 400 && errorMsg.includes('maximum context length'))) && turns.length > 1) {
                 console.log('Context length exceeded, trying again with shorter context.');
                 rawResponse = 'Context length exceeded, retrying...';
                 log(modelInput, rawResponse);
                 // Recursive call for retry
                 return await this.sendRequest(turns.slice(1), systemMessage, stop_seq);
            }
             // Handle vision errors (if applicable)
            else if (errorMsg.includes("doesn't support image input") || errorMsg.includes("Invalid type for path `messages")) {
                rawResponse = 'Vision is only supported by certain models (or message format incorrect).';
            }
             // Handle rate limits (common with Alibaba APIs)
            else if (err.status === 429 || errorMsg.includes('Throttling')) {
                 rawResponse = 'Qwen API rate limit exceeded. Please try again later.';
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
             //console.log("Removing think block from Qwen final response.");
             finalRes = finalRes.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
         }

        return finalRes; // Return potentially processed response or error message
    }


    async embed(text) {
        if (!text || typeof text !== 'string') {
             console.error("Invalid input text for embedding:", text);
             throw new Error("Invalid input for embedding.");
        }
         const embeddingModel = this.params?.embedding_model || "text-embedding-v1"; // Qwen default (check latest)

        const maxRetries = 5; // Maximum number of retries for rate limits
        for (let retries = 0; retries < maxRetries; retries++) {
            try {
                 console.log(`Requesting Qwen embedding (model: ${embeddingModel}, attempt: ${retries + 1})...`);
                 const response = await this.openai.embeddings.create({
                    model: embeddingModel,
                    input: text, // Qwen API might expect single string or array based on model
                    // encoding_format: "float", // Check if Qwen supports/requires this
                });

                 // Adjust path based on actual Qwen response structure
                 // Example assumes OpenAI-like structure: response.data[0].embedding
                 if (response && response.data && response.data.length > 0 && response.data[0].embedding) {
                    return response.data[0].embedding;
                 } else {
                     // Handle potential different structures, e.g., response.output.embeddings[0].embedding
                      if (response?.output?.embeddings?.[0]?.embedding) {
                         return response.output.embeddings[0].embedding;
                      }
                      console.error("Unexpected embedding response structure from Qwen:", response);
                      throw new Error('Invalid embedding response structure received from Qwen.');
                 }

            } catch (err) {
                console.error(`[Qwen Embed] Error on attempt ${retries + 1}:`, err);
                 let errorMsg = err.message || 'Unknown error';
                 let status = err.status;
                  if (err.response && err.response.data && err.response.data.error) {
                      errorMsg = `API Error (${err.response.data.error.code}): ${err.response.data.error.message || JSON.stringify(err.response.data.error)}`;
                      status = err.response.status; // Get status from response if available
                  }

                // Check for rate limit error (e.g., 429 or specific code)
                if (status === 429 || errorMsg.includes('Throttling')) {
                    if (retries < maxRetries - 1) { // Only retry if not the last attempt
                        // Exponential backoff with jitter
                         const delay = Math.pow(2, retries) * 500 + Math.floor(Math.random() * 1000); // Shorter base delay
                         console.log(`Rate limit hit, retrying embedding in ${delay} ms...`);
                         await new Promise(resolve => setTimeout(resolve, delay)); // Wait
                         continue; // Retry the loop
                    } else {
                         console.error("Max retries reached for embedding due to rate limiting.");
                         throw new Error('Max retries reached for Qwen embedding (rate limit).');
                    }
                } else {
                     // For other errors, throw immediately
                     throw new Error(`Qwen embedding creation failed: ${errorMsg}`);
                }
            }
        }
         // This point should theoretically not be reached if error handling is correct
         throw new Error('Qwen embedding failed after exhausting retries or due to an unexpected issue.');
    }

}
// --- END OF FILE qwen.js ---