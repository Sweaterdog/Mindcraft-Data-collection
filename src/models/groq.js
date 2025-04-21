// --- START OF FILE groq.js ---

import Groq from 'groq-sdk'
import { getKey } from '../utils/keys.js';
import { strictFormat } from '../utils/text.js'; // Import strictFormat
import { log } from '../../logger.js'; // <-- IMPORT log

// GroqCloud API (Fast Llama/Mistral models)
export class GroqCloudAPI {

    constructor(model_name, url, params) {
        this.model_name = model_name;
        this.url = url; // Groq SDK doesn't use baseURL, but keep for potential future use/info
        this.params = params || {};

        // Remove or adjust deprecated parameters if present
        if (this.params.max_tokens && !this.params.max_completion_tokens) {
             console.warn("GROQCLOUD WARNING: Profile uses deprecated `max_tokens`. Using it for `max_completion_tokens` instead. Please update profile.");
             this.params.max_completion_tokens = this.params.max_tokens;
        }
        delete this.params.max_tokens; // Ensure deprecated param is removed

        // Remove tools if present (not used here)
        if (this.params.tools) {
            console.warn("GROQCLOUD WARNING: 'tools' parameter found in profile, removing as it's not used by this implementation.");
            delete this.params.tools;
        }


        if (this.url)
            console.warn("Groq Cloud SDK does not use a custom URL. The provided URL will be ignored.");

        const apiKey = getKey('GROQCLOUD_API_KEY');
        if (!apiKey) {
            throw new Error('GROQCLOUD_API_KEY not found in keys.');
        }
        this.groq = new Groq({ apiKey: apiKey });
    }

    async sendRequest(turns, systemMessage, stop_seq = null) {
        // Construct messages array, apply strict formatting
        let messages = [{"role": "system", "content": systemMessage}].concat(turns);
        messages = strictFormat(messages); // Apply formatting
        let modelInput = messages; // Capture formatted input for logging

        let rawResponse = null; // To store the raw response content

        try {
            console.log("Awaiting Groq response...");

            // Set default max completion tokens if not provided
            const maxCompletionTokens = this.params.max_completion_tokens || 4096; // Default value

            const pack = {
                "messages": modelInput, // Use formatted input
                "model": this.model_name || "llama3-70b-8192", // Updated default Groq model
                "stream": false, // Non-streaming for simpler handling
                "stop": stop_seq, // Pass stop sequence if provided
                "max_tokens": maxCompletionTokens, // Use the correct parameter name for Groq
                ...(this.params || {}) // Spread other parameters like temperature, top_p
            };
            // Remove max_completion_tokens from params if it was added manually above
             delete pack.max_completion_tokens;


            let completion = await this.groq.chat.completions.create(pack);

            if (completion && completion.choices && completion.choices.length > 0) {
                 rawResponse = completion.choices[0]?.message?.content; // Store raw response content
                 // --- Log successful raw response ---
                 log(modelInput, rawResponse);

                  // --- Post-logging checks ---
                  const finishReason = completion.choices[0]?.finish_reason;
                  if (finishReason === 'length' || finishReason === 'max_tokens') { // Check for length limit
                      console.warn("Groq response may have been truncated due to token limits.");
                      // Don't throw error, just warn, as we have the partial response
                  } else if (finishReason === 'stop') {
                      console.log("Groq response finished due to stop sequence.");
                  }

                 console.log('Received.');

            } else {
                 throw new Error("Invalid response structure received from Groq API.");
            }


        } catch(err) {
            console.error("[Groq] Error:", err); // Log the actual error
            // Handle specific errors (e.g., vision format)
            if (err.message && err.message.includes("content must be a string")) { // Likely vision format issue
                 rawResponse = "Vision format likely incorrect for this model.";
            }
            // Handle rate limits (Groq might return 429)
            else if (err.status === 429) {
                  rawResponse = "Groq API rate limit exceeded. Please try again later.";
            }
            // Handle context length errors (might be 400 Bad Request)
            else if (err.status === 400 && err.message && err.message.includes('context')) {
                // Groq might not support easy retry, just log error
                 rawResponse = "Context length likely exceeded for Groq model.";
            }
            // Generic fallback error
            else {
                rawResponse = "My brain disconnected, try again.";
            }
             // --- Log error response ---
            log(modelInput, rawResponse);
        }

         // Post-processing: Remove think blocks *after* logging raw response
         let finalRes = rawResponse;
         if (finalRes && typeof finalRes === 'string' && finalRes.includes('<think>') && finalRes.includes('</think>')) {
            // console.log("Removing think block from Groq final response.");
             finalRes = finalRes.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
         }

        return finalRes; // Return potentially processed response or error message
    }

    // Handles vision request by formatting messages and calling sendRequest
    async sendVisionRequest(messages, systemMessage, imageBuffer) {
        const imageMessages = messages.filter(message => message.role !== 'system'); // Start with non-system
        const visionUserMessage = {
            role: "user",
            // Groq likely expects OpenAI format for vision
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
        // System message likely empty as prompt is in user content.
        // WARNING: Groq models might not support vision. This will likely error.
        console.warn("Attempting vision request with Groq. Most Groq models do NOT support vision.");
        return this.sendRequest(imageMessages, "");
    }

    async embed(_) {
        console.warn('Embeddings are not supported by the Groq provider.');
        throw new Error('Embeddings are not supported by Groq.');
    }
}
// --- END OF FILE groq.js ---