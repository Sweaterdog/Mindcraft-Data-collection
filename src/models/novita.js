// --- START OF FILE novita.js ---

import OpenAIApi from 'openai';
import { getKey } from '../utils/keys.js';
import { strictFormat } from '../utils/text.js';
import { log } from '../../logger.js'; // <-- IMPORT log

// Novita AI (OpenAI Compatible Endpoint)
export class Novita {
	constructor(model_name, url, params) {
    this.model_name = model_name ? model_name.replace('novita/', '') : null; // Clean name
    this.url = url || 'https://api.novita.ai/v3/openai'; // Default Novita endpoint
    this.params = params || {}; // Store params

    let config = {
      baseURL: this.url
    };
    config.apiKey = getKey('NOVITA_API_KEY');
     if (!config.apiKey) {
        throw new Error('NOVITA_API_KEY not found in keys.');
     }

    this.openai = new OpenAIApi(config);
  }

	async sendRequest(turns, systemMessage, stop_seq='***') {
      let messages = [{'role': 'system', 'content': systemMessage}];
      messages.push(...strictFormat(turns)); // Apply formatting
      let modelInput = messages; // Capture input for logging

      // Build the request package merging constructor params
      const pack = {
          model: this.model_name || "meta-llama/Llama-3.1-70b-instruct", // Default model if needed
          messages: modelInput,
          stop: stop_seq ? [stop_seq] : undefined, // API expects array or undefined
          // Add other parameters from constructor
          temperature: this.params.temperature,
          max_tokens: this.params.max_tokens,
          top_p: this.params.top_p,
          ...(this.params || {}) // Spread remaining params cautiously
      };
       // Remove params explicitly handled to avoid duplication
       delete pack.temperature;
       delete pack.max_tokens;
       delete pack.top_p;
       delete pack.stop;


      let rawResponse = null; // Store raw response

      try {
          console.log(`Awaiting novita api response (model: ${pack.model})...`);
          //console.log("Payload:", JSON.stringify(pack)); // Debug payload
          let completion = await this.openai.chat.completions.create(pack);

           if (!completion?.choices?.[0]?.message?.content) {
                console.error('Invalid response structure or missing content from Novita:', JSON.stringify(completion));
                rawResponse = 'Received invalid response structure from Novita.';
                 log(modelInput, rawResponse);
                 return rawResponse;
            }

          rawResponse = completion.choices[0].message.content; // Store raw response

          // --- Log successful raw response ---
          log(modelInput, rawResponse);

          // --- Post-logging checks ---
          if (completion.choices[0].finish_reason == 'length') {
               console.warn("Novita response may have been truncated due to token limits.");
               // Don't throw, just warn
          }
          console.log('Received.');

      } catch (err) {
          console.error("[Novita] Error:", err); // Log the actual error object
           let errorMsg = err.message || 'Unknown error';
            if (err.response && err.response.data && err.response.data.error) {
                 errorMsg = `API Error: ${err.response.data.error.message || JSON.stringify(err.response.data.error)}`;
            } else if (err.status) {
                 errorMsg = `HTTP Error: ${err.status}`;
            }


          // Handle context length error specifically for retry
          if ((errorMsg.includes('context length') || (err.status === 400 && errorMsg.includes('maximum context length'))) && turns.length > 1) {
              console.log('Context length exceeded, trying again with shorter context.');
              rawResponse = 'Context length exceeded, retrying...';
              log(modelInput, rawResponse);
              // Recursive call for retry
              return await this.sendRequest(turns.slice(1), systemMessage, stop_seq); // Corrected recursive call
          }
           // Handle vision errors (if Novita endpoint supports it and format is wrong)
          else if (errorMsg.includes("doesn't support image input") || errorMsg.includes("Invalid type for path `messages")) {
             rawResponse = 'Vision is only supported by certain models (or message format incorrect).';
          }
          // Generic fallback
          else {
               rawResponse = `My brain disconnected, try again. Error: ${errorMsg}`;
          }
          log(modelInput, rawResponse); // Log the error state
      }

      // Post-processing *after* logging and error handling
      let finalRes = rawResponse;
      if (finalRes && typeof finalRes === 'string') {
            // Remove think blocks if present
           if (finalRes.includes('<think>') && finalRes.includes('</think>')) {
               //console.log("Removing think block from Novita final response.");
               finalRes = finalRes.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
           }
            // Handle potential partial think block (less common now with logger retries, but good safety)
            else if (finalRes.includes('<think>')) {
                console.warn("Detected potentially incomplete <think> block in final response. Truncating before it.");
                finalRes = finalRes.substring(0, finalRes.indexOf('<think>')).trim();
            }
      }

      return finalRes; // Return potentially processed response or error message
  }

	async embed(text) {
		console.warn('Embeddings are not directly supported by the Novita provider class.');
		throw new Error('Embeddings are not supported by Novita AI via this class.');
	}
}
// --- END OF FILE novita.js ---