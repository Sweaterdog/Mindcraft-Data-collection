// --- START OF FILE gemini.js ---

import { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } from '@google/generative-ai';
import { toSinglePrompt, strictFormat } from '../utils/text.js';
import { getKey } from '../utils/keys.js';
import { log } from '../../logger.js'; // <-- IMPORT log

export class Gemini {
    constructor(model_name, url, params) {
        this.model_name = model_name;
        this.params = params;
        this.url = url; // Base URL for potential proxy/alternative endpoint
        this.safetySettings = [ // Adjusted to use SDK enums
            { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
            { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
            { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
            { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
            // Note: HARM_CATEGORY_DANGEROUS is often implicitly covered by DANGEROUS_CONTENT
        ];

        const apiKey = getKey('GEMINI_API_KEY');
        if (!apiKey) {
            throw new Error('GEMINI_API_KEY not found in keys.');
        }
        // Initialize the SDK client
        // Base URL is handled differently in v1beta SDK - usually via GoogleAuth or direct endpoint specification if needed
        this.genAI = new GoogleGenerativeAI(apiKey);
        if (this.url) {
             console.warn("Custom URL with Gemini SDK might require specific configuration (e.g., custom transport) not fully handled here.");
        }
    }

    _getModelInstance() {
         // Helper to get model instance, applying safety settings
        const modelConfig = {
            model: this.model_name || "gemini-1.5-flash-latest", // Use latest flash as default
            safetySettings: this.safetySettings,
             // systemInstruction can be added here if model supports it
             // systemInstruction: { role: "system", parts: [{ text: systemMessage }] }
        };
        return this.genAI.getGenerativeModel(modelConfig);
    }


    async sendRequest(turns, systemMessage) {
        const model = this._getModelInstance(); // Get model with safety settings

        console.log('Awaiting Google API response...');

        // Prepare contents array using strictFormat and mapping roles
        // Gemini uses 'user' and 'model' roles
        let formattedTurns = strictFormat(turns); // Ensure consistent format first
        let contents = [];
        // Prepend system message as the first 'user' turn if systemInstruction isn't used/supported
        if (systemMessage) {
             contents.push({ role: 'user', parts: [{ text: systemMessage }] });
             // Gemini expects alternating roles, so add a placeholder model response if needed
             contents.push({ role: 'model', parts: [{ text: "Okay." }] }); // Simple acknowledgement
        }

        formattedTurns.forEach(turn => {
            contents.push({
                role: turn.role === 'assistant' ? 'model' : 'user',
                parts: [{ text: turn.content }]
            });
        });

        // Ensure roles alternate correctly (user, model, user, model...)
        // Simple check: remove consecutive turns with the same role (prefer 'user' if conflict)
        let finalContents = [];
        if (contents.length > 0) {
            finalContents.push(contents[0]);
            for (let i = 1; i < contents.length; i++) {
                if (contents[i].role !== finalContents[finalContents.length - 1].role) {
                    finalContents.push(contents[i]);
                } else {
                     console.warn(`[Gemini] Correcting consecutive roles. Discarding turn: ${JSON.stringify(contents[i])}`);
                     // If consecutive user roles, merge content? For now, just discard second one.
                }
            }
        }


        let modelInput = finalContents; // Capture final input for logging
        let rawResponse = null; // Store raw response text

        try {
            const result = await model.generateContent({
                contents: modelInput, // Use the role-corrected contents
                generationConfig: {
                    // Map common params if needed, e.g., max_tokens to maxOutputTokens
                    maxOutputTokens: this.params?.max_tokens || this.params?.maxOutputTokens || 2048,
                    temperature: this.params?.temperature,
                    topP: this.params?.top_p,
                    // stopSequences: stop_seq ? [stop_seq] : undefined, // Add stop sequence if provided
                    ...(this.params || {}) // Spread remaining params cautiously
                }
            });

            const response = result.response;
            if (response && response.text) {
                rawResponse = response.text(); // Get raw text
            } else if (response && response.candidates && response.candidates[0]?.content?.parts) {
                 // Handle cases where response might be structured differently (e.g., function calls, safety blocks)
                 rawResponse = response.candidates[0].content.parts.map(p => p.text || '').join(''); // Join text parts
            }
            else {
                 // Check for safety blocks or other reasons for no text
                 const blockReason = response?.promptFeedback?.blockReason;
                 const safetyRatings = response?.candidates?.[0]?.safetyRatings;
                 console.warn(`[Gemini] No text received. Block Reason: ${blockReason}, Safety Ratings: ${JSON.stringify(safetyRatings)}`);
                 rawResponse = `No response received from Gemini. (Reason: ${blockReason || 'Unknown'})`;
            }


            // --- Log successful raw response (or error placeholder) ---
            log(modelInput, rawResponse);
            console.log('Received.');

        } catch (err) {
            console.error("[Gemini] Error:", err); // Log the actual error
             // Check if it's an API error response with details
             let errorDetails = err.message;
             if (err.response && err.response.data) {
                 errorDetails = JSON.stringify(err.response.data);
             }
            rawResponse = `My brain disconnected, try again. Error: ${errorDetails}`; // Assign error
            log(modelInput, rawResponse); // Log the error state
        }

        // Post-processing (like removing think tags) after logging
         let finalRes = rawResponse;
         if (finalRes && typeof finalRes === 'string' && finalRes.includes('<think>') && finalRes.includes('</think>')) {
            // console.log("Removing think block from Gemini final response.");
             finalRes = finalRes.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
         }

        return finalRes; // Return potentially processed response or error
    }


    async sendVisionRequest(turns, systemMessage, imageBuffer) {
        // Use a model known to support vision, e.g., gemini-1.5-pro or flash
        const modelName = this.model_name && this.model_name.includes('pro') ? this.model_name : 'gemini-1.5-flash-latest';
         const modelConfig = {
             model: modelName,
             safetySettings: this.safetySettings,
         };
        const model = this.genAI.getGenerativeModel(modelConfig);


        // Prepare the prompt parts: text first, then image
        const textPart = { text: systemMessage || "Describe this image." }; // Use system message or default
        const imagePart = {
            inlineData: {
                data: imageBuffer.toString('base64'),
                mimeType: 'image/jpeg' // Assuming JPEG
            }
        };

        // Combine with previous turns if necessary (less common for simple vision prompts)
        // For now, sending only the system prompt + image
        const contents = [{ role: 'user', parts: [textPart, imagePart] }];
        let modelInput = contents; // Capture input for logging
        let rawResponse = null;

        try {
            console.log(`Awaiting Google API vision response (${modelName})...`);
            const result = await model.generateContent({
                 contents: modelInput,
                 generationConfig: { // Apply relevant params
                     maxOutputTokens: this.params?.max_tokens || this.params?.maxOutputTokens || 1024,
                     temperature: this.params?.temperature,
                     topP: this.params?.top_p,
                 }
            });
            const response = result.response;

            if (response && response.text) {
                 rawResponse = response.text();
            } else {
                 const blockReason = response?.promptFeedback?.blockReason;
                 console.warn(`[Gemini Vision] No text received. Block Reason: ${blockReason}`);
                 rawResponse = `No response received from Gemini Vision. (Reason: ${blockReason || 'Unknown'})`;
            }


            // --- Log successful raw response (or placeholder) ---
            log(modelInput, rawResponse);
            console.log('Received.');

        } catch (err) {
            console.error("[Gemini Vision] Error:", err);
             let errorDetails = err.message;
              if (err.toString().includes("400 Bad Request")) { // Check for specific error types
                  rawResponse = "Vision request failed (400 Bad Request). Check model support or image format.";
              } else if (err.message.includes("Image input modality is not enabled")) {
                  rawResponse = "Vision is only supported by certain models.";
              } else {
                 rawResponse = `Vision request failed: ${errorDetails}`;
              }
            log(modelInput, rawResponse); // Log error state
        }

        // Post-processing after logging
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

        // Use a dedicated embedding model
        const embeddingModelName = this.params?.embedding_model || "text-embedding-004"; // Allow override
        const model = this.genAI.getGenerativeModel({ model: embeddingModelName });

        try {
            const result = await model.embedContent(text);
            const embedding = result?.embedding;
            if (embedding && embedding.values) {
                return embedding.values;
            } else {
                 throw new Error("Invalid embedding response structure received.");
            }
        } catch (err) {
             console.error("[Gemini Embed] Error creating embedding:", err);
             const errorMsg = err.response ? `${err.response.status}: ${err.response.data?.error?.message}` : err.message;
             throw new Error(`Embedding creation failed: ${errorMsg}`);
        }
    }
}
// --- END OF FILE gemini.js ---