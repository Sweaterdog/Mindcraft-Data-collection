// --- START OF FILE huggingface.js ---

import { toSinglePrompt } from '../utils/text.js';
import { getKey } from '../utils/keys.js';
import { HfInference } from "@huggingface/inference";
import { log } from '../../logger.js'; // <-- IMPORT log

export class HuggingFace {
  constructor(model_name, url, params) {
    // Remove 'huggingface/' prefix if present
    this.model_name = model_name ? model_name.replace('huggingface/', '') : null;
    this.url = url; // URL is typically not used with HfInference library directly
    this.params = params;

    if (this.url) {
      console.warn("Hugging Face Inference Client doesn't typically use a custom base URL. Ensure your setup supports this if provided.");
    }

    const apiKey = getKey('HUGGINGFACE_API_KEY');
     if (!apiKey) {
       // Allow initialization without key for local TGI maybe, but warn
       console.warn("HUGGINGFACE_API_KEY not found. Inference might fail if targeting HF API.");
       // this.huggingface = new HfInference(); // Initialize without key
     } //else {
        this.huggingface = new HfInference(apiKey); // Initialize with key
     //}
     // For now, assume API key is required for standard use.
     if (!this.huggingface) {
         throw new Error("Failed to initialize HuggingFace client. API Key might be missing.");
     }
  }

  async sendRequest(turns, systemMessage) {
    const stop_seq = this.params?.stop || '***'; // Use stop sequence from params or default
    // Build a single prompt (HF Inference often works better with single prompts)
    // Adjust `toSinglePrompt` if needed for specific HF model formats
    const prompt = toSinglePrompt(turns, systemMessage, stop_seq); // Combine system message here
    let modelInput = prompt; // Log the combined prompt

    // Fallback model if none was provided
    const model_name = this.model_name || 'meta-llama/Meta-Llama-3-8B-Instruct'; // Use a known good default

    const maxAttempts = 5; // Retries for partial <think>
    let attempt = 0;
    let finalRes = null; // Final processed result
    let rawResponse = ''; // Accumulate raw streaming response

    while (attempt < maxAttempts) {
      attempt++;
      console.log(`Awaiting Hugging Face API response... (model: ${model_name}, attempt: ${attempt})`);
      rawResponse = ''; // Reset raw response for each attempt

      try {
        // Using chatCompletionStream for potentially better compatibility
        const stream = this.huggingface.chatCompletionStream({
          model: model_name,
          messages: [{ role: "user", content: modelInput }], // Use the combined prompt as user message
          max_tokens: this.params?.max_tokens || 1024, // Add max_tokens from params or default
          temperature: this.params?.temperature, // Pass other params if available
          top_p: this.params?.top_p,
          stop: stop_seq ? [stop_seq] : undefined, // Pass stop sequence if defined
          ...(this.params || {}) // Spread remaining params cautiously
        });

        for await (const chunk of stream) {
           // Append content, handling potential nulls/undefined
           rawResponse += (chunk.choices[0]?.delta?.content || "");
           // Optional: Check for stop sequence within the stream if needed
           // if (rawResponse.includes(stop_seq)) {
           //    rawResponse = rawResponse.split(stop_seq)[0];
           //    break; // Stop accumulating if stop sequence found
           // }
        }

        // --- Log the complete raw response after streaming ---
        log(modelInput, rawResponse);
        console.log('Received raw response length:', rawResponse.length);

        // --- Post-logging checks and processing ---
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
             finalRes = "<think>" + rawResponse;
         } else {
             finalRes = rawResponse; // Assign raw if no </think> issue
         }

         // Remove think blocks *after* logging and handling partials
         if (finalRes.includes("<think>") && finalRes.includes("</think>")) {
            // console.log("Removing think block from final response.");
             finalRes = finalRes.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
         }


        break; // Valid response obtained and processed.

      } catch (err) {
        console.error("[HuggingFace] Error:", err);
        rawResponse = 'My brain disconnected, try again.'; // Assign error
        log(modelInput, rawResponse); // Log the error state
        finalRes = rawResponse; // Assign error to final result
        break; // Exit loop on error
      }
    } // End while loop

    // Fallback if loop finished without success
    if (finalRes === null) {
      console.warn("Could not get a valid response after max attempts.");
      finalRes = 'I thought too hard, sorry, try again.';
      log(modelInput, finalRes); // Log the fallback error
    }

    console.log('Final response length:', finalRes.length);
    return finalRes.trim(); // Return trimmed final result
  }

  async embed(text) {
    console.warn('Embeddings via HfInference library might require specific setup or model. Not fully implemented.');
    throw new Error('Embeddings are not directly supported by this HuggingFace provider implementation.');
    // To implement: use this.huggingface.featureExtraction({ model: 'embedding-model-name', inputs: text })
  }
}
// --- END OF FILE huggingface.js ---