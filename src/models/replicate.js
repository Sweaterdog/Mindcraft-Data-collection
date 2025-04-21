// --- START OF FILE replicate.js ---

import Replicate from 'replicate';
import { toSinglePrompt } from '../utils/text.js';
import { getKey } from '../utils/keys.js';
import { log } from '../../logger.js'; // <-- IMPORT log

// Replicate API (often uses single prompt format)
export class ReplicateAPI {
	constructor(model_name, url, params) {
		// Model name often includes user/org, e.g., 'meta/meta-llama-3-70b-instruct'
		this.model_name = model_name;
		this.url = url; // Replicate library doesn't use baseURL, but keep for info
		this.params = params || {}; // Store params

		if (this.url) {
			console.warn('Replicate API client does not use a custom base URL. Ignoring provided URL.');
		}

         const apiKey = getKey('REPLICATE_API_KEY');
         if (!apiKey) {
             throw new Error('REPLICATE_API_KEY not found in keys.');
         }

		this.replicate = new Replicate({ auth: apiKey });
	}

	async sendRequest(turns, systemMessage) {
		const stop_seq = this.params?.stop || '***'; // Use stop sequence from params or default

		// Replicate often works best with a single combined prompt.
		// Adjust toSinglePrompt or formatting based on the specific Replicate model needs.
		const prompt = toSinglePrompt(turns, systemMessage, stop_seq); // Combine system message here
		let modelInput = prompt; // Log the combined prompt

		let model_identifier = this.model_name || 'meta/meta-llama-3-70b-instruct:31c44fac1604d0f493447c14c0fa74460544307f733c579d995017afb1f73090'; // Default with version hash

		// Prepare input payload, merging constructor params
		const input = {
			prompt: modelInput, // Use the combined prompt
			system_prompt: systemMessage, // Some models use this explicitly
			// Map common parameters
			max_new_tokens: this.params?.max_tokens || this.params?.max_new_tokens || 1024, // Replicate uses max_new_tokens
			temperature: this.params?.temperature,
			top_p: this.params?.top_p,
			stop_sequences: stop_seq ? [stop_seq] : undefined, // Use stop_sequences
			...(this.params || {}) // Spread other model-specific params cautiously
		};
         // Remove params handled explicitly to avoid duplication/conflict
         delete input.max_tokens;
         delete input.stop;


		let rawResponse = ''; // Accumulate raw response from stream
		let finalRes = null; // Final processed response

		try {
			console.log(`Awaiting Replicate API response (model: ${model_identifier})...`);
             //console.log("Input payload:", input); // Debug input
			let stream = this.replicate.stream(model_identifier, { input });

			for await (const event of stream) {
				// Accumulate text data from events (event might be string or object)
				if (typeof event === 'string') {
				    rawResponse += event;
				} else if (event.type === 'output') {
                    // Handle structured output if necessary, assume string for now
                     rawResponse += event.data;
                } else if (event.type === 'error') {
                     console.error("Replicate stream error event:", event.data);
                     throw new Error(`Replicate stream error: ${event.data?.detail || 'Unknown stream error'}`);
                }


				// Optional: Check for stop sequence *during* streaming if needed
				// if (stop_seq && rawResponse.includes(stop_seq)) {
				// 	rawResponse = rawResponse.split(stop_seq)[0];
				// 	break; // Stop accumulating
				// }
			}

             // --- Log the complete raw response after streaming ---
            log(modelInput, rawResponse);
            console.log('Received raw response length:', rawResponse.length);


            // --- Post-logging processing ---
             finalRes = rawResponse; // Assign accumulated response

             // Remove think blocks if present
             if (finalRes && typeof finalRes === 'string' && finalRes.includes('<think>') && finalRes.includes('</think>')) {
                 //console.log("Removing think block from Replicate final response.");
                 finalRes = finalRes.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
             }


		} catch (err) {
			console.error("[Replicate] Error:", err); // Log the actual error object
            let errorMsg = err.message || 'Unknown error';
            // Try to extract detail if available
             if (err.response && err.response.data && err.response.data.detail) {
                 errorMsg = `API Error: ${err.response.data.detail}`;
             } else if (err.status) {
                 errorMsg = `HTTP Error: ${err.status}`;
             }

			rawResponse = `My brain disconnected, try again. Error: ${errorMsg}`;
            log(modelInput, rawResponse); // Log the error state
            finalRes = rawResponse; // Assign error to final result
		}

		console.log('Final response length:', finalRes?.length ?? 0);
		return finalRes ? finalRes.trim() : ''; // Return final processed result, ensure trim returns string
	}

	async embed(text) {
         if (!text || typeof text !== 'string') {
             console.error("Invalid input text for embedding:", text);
             throw new Error("Invalid input for embedding.");
         }
         // Requires a specific embedding model identifier (including version hash)
         const embedding_model_identifier = this.params?.embedding_model || "replicate/all-mpnet-base-v2:b6b7585c9640cd7a857a73b9ae714248568083234737b2cf13b84e1449199657"; // Example default

         console.log(`Requesting embedding from Replicate model: ${embedding_model_identifier}`);
         try {
             const output = await this.replicate.run(
                 embedding_model_identifier,
                 { input: { text: text } } // Input format depends on the specific model
             );

             // Output structure also depends on the model - adjust access path as needed
             // Example assumes output is an array of floats or similar structure
             if (output && Array.isArray(output)) { // Simple check, might need refinement
                 return output;
             }
              // Another common structure: output.embedding or output[0] etc.
              else if (output && output.embedding && Array.isArray(output.embedding)) {
                  return output.embedding;
              }
              else {
                  console.error("Unexpected embedding output structure:", output);
                  throw new Error('Invalid embedding response structure received from Replicate.');
              }
         } catch(err) {
              console.error("[Replicate Embed] Error creating embedding:", err);
              const errorMsg = err.message || 'Unknown error';
              throw new Error(`Replicate embedding creation failed: ${errorMsg}`);
         }
	}
}
// --- END OF FILE replicate.js ---