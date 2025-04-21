// --- START OF FILE claude.js ---

import Anthropic from '@anthropic-ai/sdk';
import { strictFormat } from '../utils/text.js';
import { getKey } from '../utils/keys.js';
import { log } from '../../logger.js'; // <-- IMPORT log

export class Claude {
    constructor(model_name, url, params) {
        this.model_name = model_name;
        this.params = params || {};

        let config = {};
        if (url)
            config.baseURL = url;

        config.apiKey = getKey('ANTHROPIC_API_KEY');

        this.anthropic = new Anthropic(config);
    }

    async sendRequest(turns, systemMessage) {
        const messages = strictFormat(turns);
        let modelInput = messages; // Capture input for logging
        let rawResponse = null; // Store raw response

        try {
            console.log('Awaiting anthropic api response...');
            if (!this.params.max_tokens) {
                // Default max_tokens if not set
                this.params.max_tokens = 4096;
            }
            const resp = await this.anthropic.messages.create({
                model: this.model_name || "claude-3-sonnet-20240229",
                system: systemMessage,
                messages: modelInput, // Use modelInput here
                ...(this.params || {})
            });

            console.log('Received.');
            // Extract text content for the response
            const textContent = resp.content.find(content => content.type === 'text');
            if (textContent) {
                rawResponse = textContent.text; // Store raw text response
            } else {
                console.warn('No text content found in the response.');
                rawResponse = 'No response text from Claude.'; // Assign placeholder
            }

            // --- Log successful response ---
            log(modelInput, rawResponse);

        } catch (err) {
            console.error("[Claude] Error:", err); // Log the actual error
            if (err.message && err.message.includes("does not support image input")) {
                rawResponse = "Vision is only supported by certain models.";
            } else if (err.status === 429) { // Handle rate limits specifically if needed
                 rawResponse = "Rate limit reached. Please try again later.";
            }
             else {
                rawResponse = "My brain disconnected, try again.";
            }
             // --- Log error response ---
             log(modelInput, rawResponse);
        }
        // Return the captured raw response (or error message)
        return rawResponse;
    }

    // sendVisionRequest now just prepares messages and calls sendRequest.
    // Logging happens inside sendRequest based on the response received.
    // Separate vision logging (image+text) should happen *outside* this class,
    // likely in prompter.js using logVision(imageBuffer, responseFromSendVisionRequest).
    async sendVisionRequest(turns, systemMessage, imageBuffer) {
        const imageMessages = [...turns]; // Copy existing turns
         // Format the new message with text and image
        const visionUserMessage = {
             role: "user",
             content: [
                 {
                     type: "text",
                     text: systemMessage // Or adjust based on how you want to prompt vision
                 },
                 {
                     type: "image",
                     source: {
                         type: "base64",
                         media_type: "image/jpeg", // Assuming JPEG
                         data: imageBuffer.toString('base64')
                     }
                 }
             ]
         };
        imageMessages.push(visionUserMessage);

        // The system message for the API call might be different or empty now
        // depending on whether the main prompt is in the user message content.
        // Using an empty system message here assuming prompt is in the content.
        const effectiveSystemMessage = ""; // Or pass the original systemMessage if needed

        // Call the standard sendRequest which handles the API call and logging
        return this.sendRequest(imageMessages, effectiveSystemMessage);
    }


    async embed(text) {
        // Claude does not currently support embeddings via this SDK approach easily.
        console.warn('Embeddings are not directly supported by the Claude provider in this implementation.');
        throw new Error('Embeddings are not supported by Claude.');
    }
}
// --- END OF FILE claude.js ---