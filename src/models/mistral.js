// --- START OF FILE mistral.js ---

import MistralClient from '@mistralai/mistralai'; // Correct import name convention
import { getKey } from '../utils/keys.js';
import { strictFormat } from '../utils/text.js';
import { log } from '../../logger.js'; // <-- IMPORT log

export class Mistral {
    #client; // Use private field syntax

    constructor(model_name, url, params) {
        this.model_name = model_name;
        this.params = params || {}; // Ensure params is an object

        if (url) {
            // Mistral client constructor does allow overriding baseURL
            console.warn("Providing a custom URL to Mistral client. Ensure it's compatible.");
        }

        const apiKey = getKey("MISTRAL_API_KEY");
        if (!apiKey) {
            throw new Error("Mistral API Key missing, make sure to set MISTRAL_API_KEY in keys.json");
        }

        try {
            this.#client = new MistralClient(apiKey, url ? { baseURL: url } : undefined); // Pass URL if provided
        } catch (err) {
             console.error("Failed to initialize Mistral client:", err);
             throw new Error(`Mistral client initialization failed: ${err.message}`);
        }


        // Clean up model name (remove prefix) if necessary
        if (this.model_name && (this.model_name.startsWith("mistral/") || this.model_name.startsWith("mistralai/"))) {
            this.model_name = this.model_name.split("/")[1];
             console.log(`Using cleaned Mistral model name: ${this.model_name}`);
        }
    }

    async sendRequest(turns, systemMessage) {
        let rawResponse = null; // Store raw response content
        let modelInput = null; // Store input used for logging

        try {
            const model = this.model_name || "mistral-large-latest"; // Default model

            // Prepare messages using strictFormat and adding system message
            const messages = [{ role: "system", content: systemMessage }];
            messages.push(...strictFormat(turns)); // Apply formatting to turns
            modelInput = messages; // Capture final input

            console.log(`Awaiting mistral api response (model: ${model})...`);
            const response = await this.#client.chat({ // Use chat method
                model: model,
                messages: modelInput,
                // Map params if needed, e.g., max_tokens
                maxTokens: this.params?.max_tokens || this.params?.maxTokens, // Use maxTokens for Mistral
                temperature: this.params?.temperature,
                topP: this.params?.top_p,
                // safePrompt: this.params?.safePrompt ?? false, // Example: control safe prompt
                ...(this.params || {}) // Spread remaining params cautiously
            });

             if (response && response.choices && response.choices.length > 0) {
                 rawResponse = response.choices[0]?.message?.content; // Store raw content
                 // --- Log successful raw response ---
                 log(modelInput, rawResponse);

                  // --- Post-logging checks ---
                  const finishReason = response.choices[0]?.finish_reason;
                  if (finishReason === 'length') {
                       console.warn("Mistral response may have been truncated due to token limits.");
                  }
                  console.log('Received.');

             } else {
                  throw new Error("Invalid response structure received from Mistral API.");
             }

        } catch (err) {
            console.error("[Mistral] Error:", err); // Log the actual error
            // Handle specific errors
            if (err.message && err.message.includes("A request containing images has been given")) {
                rawResponse = "Vision is only supported by certain models.";
            } else if (err.status === 429) { // Handle rate limits
                 rawResponse = "Mistral API rate limit exceeded.";
            } else if (err.status === 401) { // Handle auth errors
                 rawResponse = "Mistral API authentication failed. Check your API key.";
            }
            // Generic fallback
            else {
                 rawResponse = `My brain disconnected, try again. Error: ${err.message || 'Unknown'}`;
            }
            // --- Log error response ---
            log(modelInput || turns, rawResponse); // Log error, use original turns if input formatting failed
        }

         // Post-processing after logging
         let finalRes = rawResponse;
         if (finalRes && typeof finalRes === 'string' && finalRes.includes('<think>') && finalRes.includes('</think>')) {
             //console.log("Removing think block from Mistral final response.");
             finalRes = finalRes.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
         }

        return finalRes; // Return potentially processed response or error
    }


    async sendVisionRequest(messages, systemMessage, imageBuffer) {
        // Mistral vision models (like mistral-large) expect OpenAI-like format
        const visionModel = this.model_name || "mistral-large-latest"; // Or specify a known vision model
        console.warn(`Attempting vision request with Mistral model: ${visionModel}. Ensure it supports vision.`);

        // Prepare messages: system prompt first, then user message with text + image
        const formattedMessages = [];
        if (systemMessage) {
            formattedMessages.push({ role: "system", content: systemMessage });
        }
        // Add prior messages if needed, applying strictFormat
        formattedMessages.push(...strictFormat(messages.filter(m => m.role !== 'system')));

        // Add the vision user message
        const visionUserMessage = {
            role: "user",
            content: [
                { type: "text", text: "Analyze the attached image." }, // Adjust prompt text as needed
                {
                    type: "image_url",
                    // Mistral SDK might expect 'url' field directly
                    image_url: { url: `data:image/jpeg;base64,${imageBuffer.toString('base64')}` }
                }
            ]
        };
        formattedMessages.push(visionUserMessage);

        let modelInput = formattedMessages; // Input for logging
        let rawResponse = null;

        try {
             console.log(`Awaiting Mistral vision response (model: ${visionModel})...`);
             const response = await this.#client.chat({
                 model: visionModel,
                 messages: modelInput,
                 maxTokens: this.params?.max_tokens || this.params?.maxTokens || 1024,
                 temperature: this.params?.temperature,
                 topP: this.params?.top_p,
             });

             if (response && response.choices && response.choices.length > 0) {
                 rawResponse = response.choices[0]?.message?.content;
                 log(modelInput, rawResponse); // Log success
                 console.log('Received vision response.');
             } else {
                  throw new Error("Invalid vision response structure from Mistral.");
             }

        } catch (err) {
             console.error("[Mistral Vision] Error:", err);
             if (err.message && err.message.includes("model does not have the 'vision' capability")) {
                 rawResponse = "Vision is not supported by this specific Mistral model.";
             } else {
                 rawResponse = `Mistral vision request failed: ${err.message || 'Unknown'}`;
             }
             log(modelInput, rawResponse); // Log error
        }

        // Post-processing
         let finalRes = rawResponse;
         if (finalRes && typeof finalRes === 'string' && finalRes.includes('<think>') && finalRes.includes('</think>')) {
             finalRes = finalRes.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
         }
        return finalRes;
    }


    async embed(text) {
         if (!text || typeof text !== 'string') {
             console.error("Invalid input text for embedding:", text);
             throw new Error("Invalid input for embedding.");
         }
         const embeddingModel = this.params?.embedding_model || "mistral-embed"; // Default embedding model

         try {
            const response = await this.#client.embeddings({ // Use embeddings method
                model: embeddingModel,
                input: [text] // API expects an array of strings
            });

             if (response && response.data && response.data.length > 0 && response.data[0].embedding) {
                 return response.data[0].embedding;
             } else {
                  throw new Error("Invalid embedding response structure received from Mistral.");
             }
         } catch(err) {
              console.error("[Mistral Embed] Error creating embedding:", err);
              const errorMsg = err.message || 'Unknown error';
              throw new Error(`Mistral embedding creation failed: ${errorMsg}`);
         }
    }
}
// --- END OF FILE mistral.js ---