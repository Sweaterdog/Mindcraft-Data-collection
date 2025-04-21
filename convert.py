# --- START OF FILE convert.py ---

import csv
import json
import logging
import sys
import os
import random
from typing import List, Dict, Tuple, Optional
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
import torch
import glob  # For finding files
import traceback # For detailed error logging

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---
LOGS_DIR = './logs'
NORMAL_LOG_FILE = os.path.join(LOGS_DIR, 'normal_logs.csv')
REASONING_LOG_FILE = os.path.join(LOGS_DIR, 'reasoning_logs.csv')
VISION_LOG_FILE = os.path.join(LOGS_DIR, 'vision_logs.csv') # Define path for checking

# Add USERNAMES.py or provide the list directly
try:
    # Ensure USERNAMES.py is in the same directory or Python path
    from USERNAMES import Get_Usernames
    MINECRAFT_USERNAMES_LIST = Get_Usernames()
    logger.info(f"Loaded {len(MINECRAFT_USERNAMES_LIST)} usernames from USERNAMES.py")
except ImportError:
    logger.warning("USERNAMES.py not found or Get_Usernames function missing. Using a default list of usernames.")
    MINECRAFT_USERNAMES_LIST = [f"Player{i:03d}" for i in range(1000)] # Default backup with padding
except Exception as e:
     logger.error(f"Error loading usernames from USERNAMES.py: {e}", exc_info=True)
     MINECRAFT_USERNAMES_LIST = [f"Player{i:03d}" for i in range(1000)]

ORIGINAL_USERNAMES = [
    "IMPORT YOUR OWN USERNAMES HERE"
]

BAD_OUTPUTS = {
    "My brain just kinda stopped working. Try again.",
    "My brain disconnected, try again.",
    "I thought too hard, sorry, try again.",
    "*no response*",
    "No response received.",
    "No response data.",
    # Add other known bad outputs if necessary - check logger.js for more examples
    "Vision is only supported", # Example
}

DEFAULT_TOKENIZER = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit" # Or choose another default

# --- Global Counters ---
username_replaced_count = 0
duplicate_username_count = 0
available_minecraft_usernames = []

# --- CSV Field Size Limit ---
def set_csv_field_size_limit():
    maxInt = sys.maxsize
    decrement = True
    while decrement:
        decrement = False
        try:
            csv.field_size_limit(maxInt)
            logger.debug(f"CSV field size limit set to {maxInt}")
        except OverflowError:
            maxInt = int(maxInt / 10)
            decrement = True
set_csv_field_size_limit()

# --- Username Handling ---
def initialize_usernames():
    global available_minecraft_usernames, duplicate_username_count
    # Ensure MINECRAFT_USERNAMES_LIST is actually a list
    if not isinstance(MINECRAFT_USERNAMES_LIST, list):
        logger.error("MINECRAFT_USERNAMES_LIST is not a list. Using default.")
        usernames_to_process = [f"Player{i:03d}" for i in range(1000)]
    else:
        usernames_to_process = MINECRAFT_USERNAMES_LIST

    unique_usernames = list(set(filter(None, usernames_to_process))) # Remove None/empty and duplicates
    duplicate_username_count = len(usernames_to_process) - len(unique_usernames)
    available_minecraft_usernames = unique_usernames # Use unique list directly
    random.shuffle(available_minecraft_usernames) # Shuffle for better randomness
    logger.info(f"Initialized {len(available_minecraft_usernames)} unique usernames (removed {duplicate_username_count} duplicates/empty).")

def get_replacement_username(conversation_replacements: Dict[str, str]) -> Optional[str]:
    """Gets a unique replacement username not already used in the conversation."""
    global available_minecraft_usernames
    if not available_minecraft_usernames:
        logger.warning("Ran out of unique Minecraft usernames! Resetting list. Duplicates may occur in subsequent conversations.")
        # Re-initialize (reshuffle) the full list
        initialize_usernames()
        if not available_minecraft_usernames: # Still empty after reset?
             logger.error("Username list is empty even after reset! Cannot replace.")
             return None

    # Try to find a name not used *in this specific conversation yet*
    used_in_this_convo = set(conversation_replacements.values())
    potential_replacements = [u for u in available_minecraft_usernames if u not in used_in_this_convo]

    if not potential_replacements:
        # If all available names are somehow already used in this convo (highly unlikely unless few usernames)
        # Fallback: just pick one from the available list, even if used for a different original name
        if not available_minecraft_usernames: return None # Should not happen after reset logic
        replacement = random.choice(available_minecraft_usernames)
        logger.debug(f"Could not find a fully unique username for this turn within the conversation, reusing '{replacement}'.")
    else:
         replacement = random.choice(potential_replacements)

    # We don't remove from the global list here anymore to allow reuse across different conversations easily.
    # The uniqueness check is per-conversation.
    return replacement

def replace_usernames_in_text(text: str, conversation_replacements: Dict[str, str]) -> Tuple[str, bool]:
    """Replaces original usernames with Minecraft usernames, tracking replacements per conversation."""
    global username_replaced_count
    replaced_in_this_call = False
    modified_text = text

    if not available_minecraft_usernames and len(MINECRAFT_USERNAMES_LIST) > 0:
        initialize_usernames()

    if not MINECRAFT_USERNAMES_LIST or not available_minecraft_usernames: # No usernames loaded or list exhausted
        if len(ORIGINAL_USERNAMES) > 0 and any(orig in text for orig in ORIGINAL_USERNAMES):
             logger.warning("Cannot replace usernames: No replacement names available.")
        return text, False

    # Iterate through original names that *might* be in the text
    names_in_text = [name for name in ORIGINAL_USERNAMES if name in modified_text]

    for orig_name in names_in_text:
        if orig_name not in conversation_replacements:
            replacement = get_replacement_username(conversation_replacements)
            if replacement:
                conversation_replacements[orig_name] = replacement
                logger.debug(f"Mapping original name '{orig_name}' to '{replacement}' for this conversation.")
            else:
                logger.warning(f"Could not find replacement username for '{orig_name}'. Skipping.")
                continue # Skip replacement if no name available

        # Perform replacement using the assigned name for this conversation
        if orig_name in conversation_replacements:
            target_replacement = conversation_replacements[orig_name]
            # Use regex for safer replacement (whole word boundary might be too strict)
            # Simple replace should be okay if names are distinct enough
            if orig_name in modified_text: # Check again in case previous replacement affected it
               modified_text = modified_text.replace(orig_name, target_replacement)
               if not replaced_in_this_call:
                   username_replaced_count += 1 # Count conversation-level replacement once
                   replaced_in_this_call = True

    return modified_text, replaced_in_this_call

# --- JSON Parsing ---
def parse_json_safely(text: str) -> List[Dict[str, str]]:
    """Safely parses the stringified JSON from the 'input' column."""
    parsed_data = []
    if not text or not isinstance(text, str):
        logger.debug("Input text is empty or not a string, cannot parse.")
        return parsed_data

    original_text = text # Keep for fallback

    try:
        # Handle outer quotes and escaped quotes if present
        if text.startswith('"') and text.endswith('"'):
            # Attempt to remove outer quotes and unescape internal ones
            try:
                # This handles JSON strings that were stringified *twice*
                 text = json.loads(text)
                 if not isinstance(text, str): # If loads() gives back non-string, revert
                      text = original_text
                      if text.startswith('"') and text.endswith('"'):
                           text = text[1:-1].replace('""', '"')
                 else: # If it was doubly stringified, loads() gives the inner string
                      # Now we need to parse the actual JSON list/object within
                      pass # Continue to the main json.loads below
            except json.JSONDecodeError:
                 # If it wasn't doubly stringified, treat as single quotes
                 text = text[1:-1].replace('""', '"')


        data = json.loads(text)

        # Expecting a list of dicts like [{'role': '...', 'content': '...'}, ...]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'role' in item and 'content' in item:
                    role = item['role']
                    content = item.get('content', "") # Use .get for safety

                    # Handle cases where content might be a list (e.g., vision models)
                    # For text conversion, we join list parts or take the first text part.
                    final_content = ""
                    if isinstance(content, list):
                         text_parts = []
                         for part in content:
                             if isinstance(part, dict) and part.get("type") == "text":
                                 text_parts.append(str(part.get("text", "")))
                             elif isinstance(part, str): # Handle simple string lists
                                 text_parts.append(part)
                         final_content = " ".join(text_parts).strip()
                         if not final_content:
                              logger.debug(f"Content list in role '{role}' had no text parts: {content}")

                    elif content is not None:
                         final_content = str(content)
                    else:
                         logger.debug(f"Content for role '{role}' is None.")


                    # Map roles to 'human'/'gpt' format
                    role_map = {"system": "human", "user": "human", "assistant": "gpt", "model": "gpt"}
                    from_role = role_map.get(str(role).lower(), "human") # Default to human, ensure role is string

                    parsed_data.append({
                        "from": from_role,
                        "value": final_content
                    })
                else:
                    logger.debug(f"Skipping invalid item in input JSON list: {item}")
        else:
            logger.warning(f"Unexpected JSON structure in input column (expected list): {type(data)}. Treating as single message.")
            # Fallback: treat the whole input as a single human message
            parsed_data.append({"from": "human", "value": str(data)})

    except json.JSONDecodeError as e:
        logger.warning(f"Input column contains non-JSON text or invalid JSON: {e}. Treating as single human message. Text: '{original_text[:100]}...'")
        parsed_data.append({"from": "human", "value": original_text}) # Use original text on failure
    except Exception as e:
        logger.error(f"Unexpected error parsing input JSON: {e}\nText: '{original_text[:100]}...'", exc_info=False)
        logger.debug(traceback.format_exc()) # Log full traceback for debugging
        parsed_data.append({"from": "human", "value": original_text}) # Fallback

    return parsed_data


# --- Conversation Processing ---
def create_conversation_thread(row: Dict[str, str]) -> Optional[List[Dict[str, str]]]:
    """Creates a single conversation thread from a CSV row."""
    messages = []
    conversation_replacements = {} # Track username swaps for THIS conversation

    # 1. Process Input Column (contains conversation history)
    input_text = str(row.get("input", "")).strip()
    if not input_text:
        # logger.debug("Skipping row with empty input.") # Too verbose maybe
        return None # Skip rows with no input

    parsed_input_messages = parse_json_safely(input_text)
    if not parsed_input_messages:
        # logger.debug("Skipping row with input that couldn't be parsed into messages.")
        return None # Skip if parsing yielded nothing


    for msg in parsed_input_messages:
        # Skip messages with empty value after parsing
        if not msg.get("value", "").strip():
            continue
        replaced_value, _ = replace_usernames_in_text(msg["value"], conversation_replacements)
        messages.append({
            "from": msg["from"],
            "value": replaced_value.strip() # Ensure stripped value
        })

    # 2. Process Output Column (the model's response)
    output_text = str(row.get("output", "")).strip()
    if not output_text:
        # logger.debug("Skipping row with empty output.")
        return None # Skip rows with no output response

    # Check against BAD_OUTPUTS (case-insensitive partial match might be better)
    is_bad_output = False
    for bad_out in BAD_OUTPUTS:
        if bad_out.lower() in output_text.lower():
            is_bad_output = True
            break
    if is_bad_output:
        logger.debug(f"Skipping row due to bad output indicator: {output_text[:60]}...")
        return None

    # Replace usernames in the output, using the *same* replacements as the input
    replaced_output, _ = replace_usernames_in_text(output_text, conversation_replacements)

    # Clean residual <think>undefined</think> blocks specifically
    # Also clean potential incomplete <think> tags if logger didn't catch them
    cleaned_output = replaced_output.replace("<think>\nundefined</think>\n", "").replace("<think>\nundefined</think>", "").strip()
    # Simple cleanup for potentially broken think tags - might need refinement
    if "<think>" in cleaned_output and "</think>" not in cleaned_output:
        cleaned_output = cleaned_output.split("<think>", 1)[0].strip()
    # Remove complete think blocks if they weren't handled earlier (should be rare now)
    cleaned_output = re.sub(r'<think>.*?</think>', '', cleaned_output, flags=re.DOTALL).strip()


    if not cleaned_output: # If cleaning results in empty output
         logger.debug("Skipping row because output became empty after cleaning.")
         return None

    # Add the processed output as a 'gpt' message
    messages.append({
        "from": "gpt",
        "value": cleaned_output
    })

    # Filter out conversations that became empty after processing
    messages = [m for m in messages if m.get("value", "").strip()]

    # Final check: ensure conversation has at least one human and one gpt message
    has_human = any(m['from'] == 'human' for m in messages)
    has_gpt = any(m['from'] == 'gpt' for m in messages)
    if not (has_human and has_gpt):
        # logger.debug("Skipping conversation without both human and gpt turns after processing.")
        return None

    return messages

def extract_conversations_from_csv(csv_filepath: str) -> List[List[Dict[str, str]]]:
    """Reads a CSV log file and extracts conversation threads."""
    if not os.path.isfile(csv_filepath):
        logger.warning(f"CSV log file not found: {csv_filepath}")
        return []

    logger.info(f"Reading conversations from: {csv_filepath}")
    conversations = []
    processed_rows = 0
    skipped_rows = 0
    expected_headers = ["input", "output"] # Expecting these from text logs

    try:
        with open(csv_filepath, newline='', encoding="utf-8") as csvfile:
            # Sniff to check delimiter, handle potential BOM
            try:
                # Read a sample to detect dialect and check header
                sample = csvfile.read(4096)
                dialect = csv.Sniffer().sniff(sample)
                has_header = csv.Sniffer().has_header(sample)
                csvfile.seek(0) # Rewind after sniffing

                if not has_header:
                     logger.error(f"CSV file {csv_filepath} is missing a header row. Skipping file.")
                     return []

                reader = csv.DictReader(csvfile, dialect=dialect)

                # Verify header columns after loading DictReader
                if not reader.fieldnames or not all(h in reader.fieldnames for h in expected_headers):
                    logger.error(f"CSV file {csv_filepath} has missing headers. Expected: {expected_headers}. Found: {reader.fieldnames}. Skipping file.")
                    return []

            except csv.Error as sniff_err:
                 logger.error(f"Could not determine CSV dialect or header for {csv_filepath}: {sniff_err}. Skipping file.")
                 return []
            except Exception as e:
                 logger.error(f"Error during CSV sniffing/header check for {csv_filepath}: {e}", exc_info=True)
                 return []


            for row_num, row_dict in enumerate(tqdm(reader, desc=f"Processing {os.path.basename(csv_filepath)}", unit="row", leave=False), start=2):
                 processed_rows += 1
                 try:
                    # Handle potential null bytes in fields
                    cleaned_row = {k: v.replace('\x00', '') if isinstance(v, str) else v for k, v in row_dict.items()}
                    conv = create_conversation_thread(cleaned_row)
                    if conv:
                        conversations.append(conv)
                    else:
                         skipped_rows += 1
                 except Exception as e:
                    logger.error(f"Error processing row {row_num} in {csv_filepath}: {e}", exc_info=False) # Less verbose traceback for row errors
                    logger.debug(traceback.format_exc())
                    skipped_rows += 1

    except Exception as e:
        logger.error(f"Failed to read or process CSV file {csv_filepath}: {e}", exc_info=True)
        return [] # Return empty list on major read error

    logger.info(f"Finished {os.path.basename(csv_filepath)}. Extracted {len(conversations)} conversations from {processed_rows} rows (skipped {skipped_rows}).")
    return conversations


# --- Tokenization ---
def load_tokenizer(model_name: str) -> Optional[PreTrainedTokenizer]:
    """Loads the tokenizer."""
    try:
        logger.info(f"Loading tokenizer '{model_name}'...")
        # Trust remote code if necessary for some tokenizers, but be cautious
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{model_name}': {e}", exc_info=True)
        return None

def tokenize_conversations(conversations: List[List[Dict[str, str]]], tokenizer: PreTrainedTokenizer, device: str, desc: str) -> List[Tuple[int, List[Dict[str, str]]]]:
    """Tokenizes conversations and returns counts."""
    conv_token_counts = []
    logger.info(f"Tokenizing conversations ({desc}) with progress bar...")
    for conv in tqdm(conversations, desc=desc, unit="conv", leave=False):
        # Simple concatenation for token count estimation
        full_text = "\n".join(msg["value"] for msg in conv if msg.get("value")).strip()
        if not full_text:
             conv_token_counts.append((0, conv))
             continue
        try:
            # Tokenize without padding/truncation to get actual length
            encoded = tokenizer(full_text, return_tensors="pt", padding=False, truncation=False)
            input_ids = encoded["input_ids"].to(device)
            token_count = input_ids.shape[-1]
            conv_token_counts.append((token_count, conv))
        except Exception as e:
            logger.warning(f"Error tokenizing conversation (hash: {hash(full_text)}): {e}. Assigning 0 tokens.", exc_info=False) # Less verbose logging
            conv_token_counts.append((0, conv)) # Add with 0 count on error
    return conv_token_counts

# --- Deduplication and Filtering ---
def deduplicate_conversations(conversations: List[List[Dict[str, str]]]) -> List[List[Dict[str, str]]]:
    """Removes duplicate conversations based on first human and last gpt message content."""
    unique_conversations = []
    seen_keys = set()
    duplicates_found = 0
    logger.info("Deduplicating conversations...")
    for conv in tqdm(conversations, desc="Deduplicating", unit="conv", leave=False):
        if not conv: continue # Skip empty conversations

        # Create a unique key for the conversation
        first_human = next((msg["value"].strip() for msg in conv if msg["from"] == "human" and msg.get("value")), None)
        last_gpt = next((msg["value"].strip() for msg in reversed(conv) if msg["from"] == "gpt" and msg.get("value")), None)

        if not first_human or not last_gpt: # Handle edge cases or single valid message convos
             key_tuple = tuple( (m["from"], m["value"].strip()) for m in conv if m.get("value"))
             key = hash(key_tuple) # Hash the tuple of (from, value) pairs
        else:
            key = (first_human, last_gpt) # Use tuple of strings

        # Add if key is new
        if key not in seen_keys:
            seen_keys.add(key)
            unique_conversations.append(conv)
        else:
            duplicates_found += 1

    logger.info(f"Removed {duplicates_found} duplicate conversations. Remaining: {len(unique_conversations)}")
    return unique_conversations

def filter_code_conversations(conversations: List[List[Dict[str, str]]], code_ratio: float = 0.15) -> List[List[Dict[str, str]]]:
    """Filters conversations to maintain a ratio of coding to non-coding examples."""
    logger.info(f"Filtering conversations for '--codeOnly' flag with ~{code_ratio*100:.1f}% non-coding examples.")
    coding = []
    noncoding = []
    for conv in conversations:
        if not conv: continue
        # Check if any message contains ``` or the last GPT message contains !newAction(
        has_code_block = any("```" in msg.get("value","") for msg in conv)
        has_new_action = (conv[-1]["from"] == "gpt" and "!newAction(" in conv[-1].get("value",""))
        is_coding_example = has_code_block or has_new_action

        if is_coding_example:
            coding.append(conv)
        else:
            noncoding.append(conv)

    logger.info(f"Found {len(coding)} coding examples and {len(noncoding)} non-coding examples.")

    if len(coding) == 0 and len(noncoding) > 0:
        logger.warning("No coding examples found, but non-coding examples exist. '--codeOnly' flag results in empty dataset.")
        return []
    elif len(coding) == 0 and len(noncoding) == 0:
         logger.warning("No coding or non-coding examples found to filter.")
         return []

    # Determine how many non-coding examples to keep
    noncoding_target_count = int(round(code_ratio * len(coding)))
    noncoding_actual_count = min(noncoding_target_count, len(noncoding)) # Can't keep more than available

    selected_noncoding = []
    if noncoding_actual_count > 0:
         logger.info(f"Selecting {noncoding_actual_count} non-coding examples to include.")
         if len(noncoding) > 0 : # Ensure noncoding list is not empty
            selected_noncoding = random.sample(noncoding, noncoding_actual_count)
         else:
             logger.warning("Attempted to select non-coding examples, but the list is empty.")
    else:
         logger.info("No non-coding examples will be included based on the ratio and availability.")

    final_conversations = coding + selected_noncoding
    random.shuffle(final_conversations) # Shuffle the combined list
    logger.info(f"Final dataset size after code filtering: {len(final_conversations)}")
    return final_conversations

# --- Vision Data Handling (Placeholder/Structure) ---
# This function is NOT used by default but shows how you might process vision data
def extract_vision_data(csv_filepath: str) -> pd.DataFrame:
    """Reads the vision CSV and prepares a DataFrame."""
    if not os.path.isfile(csv_filepath):
        logger.warning(f"Vision log file not found: {csv_filepath}")
        return pd.DataFrame({'image_path': [], 'text': []}) # Return empty DataFrame

    logger.info(f"Reading vision data from: {csv_filepath}")
    expected_headers = ["image_path", "text"]
    rows = []
    try:
        with open(csv_filepath, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames or not all(h in reader.fieldnames for h in expected_headers):
                logger.error(f"Vision CSV {csv_filepath} missing headers. Expected: {expected_headers}. Found: {reader.fieldnames}. Skipping.")
                return pd.DataFrame({'image_path': [], 'text': []})

            for row_num, row_dict in enumerate(tqdm(reader, desc=f"Processing {os.path.basename(csv_filepath)}", unit="row", leave=False), start=2):
                 # Basic validation
                 img_path = row_dict.get("image_path", "").strip()
                 text = row_dict.get("text", "").strip()
                 # Ensure image path is relative to LOGS_DIR and exists (optional check)
                 full_img_path = os.path.join(LOGS_DIR, img_path)
                 if img_path and text:
                     # if not os.path.exists(full_img_path):
                     #     logger.warning(f"Image path specified in vision log does not exist: {full_img_path} (Row {row_num})")
                     #     # Decide whether to skip or include anyway
                     #     continue
                     rows.append({"image_path": img_path, "text": text})
                 # else: skip row if path or text is missing
    except Exception as e:
        logger.error(f"Failed to read or process vision CSV {csv_filepath}: {e}", exc_info=True)
        return pd.DataFrame({'image_path': [], 'text': []})

    logger.info(f"Finished {os.path.basename(csv_filepath)}. Extracted {len(rows)} vision entries.")
    return pd.DataFrame(rows)


# --- Main Execution ---
if __name__ == "__main__":
    # Argument Parsing (More robust)
    args = sys.argv[1:]
    do_tokenize_all = '--tokenize' in args
    do_tokenize_largest = '--tokenize_largest' in args
    is_code_only = '--codeOnly' in args
    is_vision_output = '--vision' in args # Flag to generate vision-specific output format

    tokenizer_name = DEFAULT_TOKENIZER
    for arg in args:
        if arg.startswith('--tokenizer='):
            tokenizer_name = arg.split('=', 1)[1]
            break # Take the first one found

    # --- Initialization ---
    initialize_usernames() # Load usernames early

    # Identify log files to process (ONLY text logs for conversation format)
    log_files_to_process = []
    if os.path.exists(NORMAL_LOG_FILE):
        log_files_to_process.append(NORMAL_LOG_FILE)
    if os.path.exists(REASONING_LOG_FILE):
        log_files_to_process.append(REASONING_LOG_FILE)

    # Check for vision log but DO NOT process it for the conversation output
    if os.path.exists(VISION_LOG_FILE):
        logger.info(f"Found vision log file ({os.path.basename(VISION_LOG_FILE)}). It will NOT be included in the conversation output unless '--vision' flag is used for separate vision processing.")

    if not log_files_to_process and not (is_vision_output and os.path.exists(VISION_LOG_FILE)):
        logger.error(f"No text log files ({os.path.basename(NORMAL_LOG_FILE)}, {os.path.basename(REASONING_LOG_FILE)}) found in {LOGS_DIR}, and not running in dedicated vision mode. Exiting.")
        sys.exit(1)
    elif not log_files_to_process and is_vision_output and os.path.exists(VISION_LOG_FILE):
        logger.info("Running in dedicated vision output mode. Only processing vision log.")
    elif log_files_to_process:
        logger.info(f"Found text log files to process: {', '.join(os.path.basename(f) for f in log_files_to_process)}")


    # --- Dedicated Vision Output Mode ---
    if is_vision_output:
        logger.info("--- Running in Vision Output Mode ---")
        vision_df = extract_vision_data(VISION_LOG_FILE)
        if vision_df.empty:
            logger.error("No valid data extracted from vision log file. Exiting.")
            sys.exit(1)

        # Add 'messages' column with None (or potentially a structured prompt?)
        # For simplicity, keeping it minimal: just image path and text
        # vision_df['messages'] = [None] * len(vision_df) # Or format text as message

        # Output format might need adjustment based on training requirements
        # Option 1: Simple image path and text
        output_df = vision_df[['image_path', 'text']]
        output_parquet = "Andy_vision_data.parquet"

        # Option 2: Structure similar to text data but with image ref
        # output_df = pd.DataFrame({
        #     "image_path": vision_df["image_path"],
        #     "messages": [[{"from": "human", "value": "Describe this image."}, {"from": "gpt", "value": text}] for text in vision_df["text"]]
        # })
        # output_parquet = "Andy_conversations_vision_structured.parquet"


        logger.info(f"Writing {len(output_df)} vision entries to {output_parquet}")
        try:
            output_df.to_parquet(output_parquet, index=False)
            logger.info("Successfully wrote Vision Parquet file.")
            logger.info("--- Vision Output Mode Complete ---")
            sys.exit(0) # Exit after vision processing
        except Exception as e:
            logger.error(f"Error writing Vision Parquet file '{output_parquet}': {e}", exc_info=True)
            sys.exit(1)
    # --- End of Dedicated Vision Output Mode ---


    # --- Standard Conversation Processing ---
    logger.info("--- Running in Conversation Output Mode ---")

    # --- Load Tokenizer (if needed) ---
    tokenizer = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if do_tokenize_all or do_tokenize_largest:
        tokenizer = load_tokenizer(tokenizer_name)
        if not tokenizer:
            logger.warning("Tokenizer failed to load. Tokenization steps will be skipped.")
            do_tokenize_all = False
            do_tokenize_largest = False
        else:
             logger.info(f"Using device: {device} for tokenization.")


    # --- Extract Conversations ---
    combined_conversations = []
    file_conversation_counts = {}

    for log_file in log_files_to_process:
        convs = extract_conversations_from_csv(log_file)
        file_conversation_counts[os.path.basename(log_file)] = len(convs)
        combined_conversations.extend(convs)

    initial_total_count = len(combined_conversations)
    if initial_total_count == 0:
        logger.error("No valid conversations extracted from any text log file. Exiting.")
        sys.exit(1)

    logger.info(f"Initially extracted {initial_total_count} conversations from {len(log_files_to_process)} text file(s).")

    # --- Deduplicate ---
    combined_conversations = deduplicate_conversations(combined_conversations)
    dedup_count = len(combined_conversations)
    if not combined_conversations:
        logger.error("No conversations remaining after deduplication. Exiting.")
        sys.exit(1)

    # --- Tokenization Analysis (Optional) ---
    if tokenizer:
        if do_tokenize_all:
            all_token_counts = tokenize_conversations(combined_conversations, tokenizer, device, "Tokenizing all data")
            total_tokens = sum(count for count, _ in all_token_counts)
            if combined_conversations:
                 avg_tokens = total_tokens / len(combined_conversations)
                 logger.info(f"Total tokens across {len(combined_conversations)} unique conversations: {total_tokens:,} (Avg: {avg_tokens:.2f} tokens/conv)")
            else:
                 logger.info("No conversations to calculate total tokens.")


        if do_tokenize_largest:
            conv_token_counts = tokenize_conversations(combined_conversations, tokenizer, device, "Tokenizing for largest")
            conv_token_counts.sort(key=lambda x: x[0], reverse=True)
            top_n = 5
            top_convs = conv_token_counts[:top_n]
            if top_convs:
                 max_tokens_overall = top_convs[0][0] # The highest count
                 logger.info(f"--- Top {top_n} Largest Conversations (Tokens) ---")
                 for idx, (count, _) in enumerate(top_convs, 1):
                     logger.info(f"Top {idx}: {count:,} tokens")
                 logger.info(f"Maximum tokens found: {max_tokens_overall:,}")
            else:
                 logger.info("No conversations available to determine largest token counts.")

    # --- Apply Filters ---
    random.shuffle(combined_conversations) # Shuffle before potential filtering

    if is_code_only:
        combined_conversations = filter_code_conversations(combined_conversations)
        if not combined_conversations:
             logger.error("No conversations remaining after '--codeOnly' filtering. Exiting.")
             sys.exit(1)

    # --- Prepare Final DataFrame ---
    final_data = {}
    output_filename_suffix = ""

    # NOTE: '--vision' flag now triggers a separate mode above. This section only handles text data.
    if is_code_only:
        final_data["conversations"] = combined_conversations
        output_filename_suffix = "_codeOnly"
    else:
        # Default output format for conversations
        final_data["conversations"] = combined_conversations
        output_filename_suffix = "" # Default

    df_final = pd.DataFrame(final_data)
    output_parquet = f"Andy_conversations{output_filename_suffix}.parquet"

    # --- Save Output ---
    logger.info(f"Attempting to write {len(df_final)} final conversations to {output_parquet}")
    try:
        df_final.to_parquet(output_parquet, index=False, engine='pyarrow') # Specify engine
        logger.info("Successfully wrote Parquet file.")
    except ImportError:
         logger.warning("pyarrow not installed. Trying fastparquet. Install with: pip install pyarrow")
         try:
             df_final.to_parquet(output_parquet, index=False, engine='fastparquet')
             logger.info("Successfully wrote Parquet file using fastparquet.")
         except ImportError:
              logger.error("fastparquet not installed either. Cannot write Parquet file. Install pyarrow or fastparquet.")
              sys.exit(1)
         except Exception as e:
             logger.error(f"Error writing Parquet file '{output_parquet}' with fastparquet: {e}", exc_info=True)
             sys.exit(1)
    except Exception as e:
        logger.error(f"Error writing Parquet file '{output_parquet}': {e}", exc_info=True)
        sys.exit(1)

    # --- Final Summary ---
    final_count = len(df_final)
    logger.info(
        f"\n"
        f"================== Conversation Conversion Summary ==================\n"
        f"Processed text log files: {', '.join(os.path.basename(f) for f in log_files_to_process)}\n"
        f"Initial conversations extracted: {initial_total_count}\n"
        f"Conversations after deduplication: {dedup_count}\n"
        f"Final conversations after filtering (if any): {final_count}\n"
        f"----------------------------------------------------------------------\n"
        f"Username replacements performed: {username_replaced_count} (approx count across conversations)\n"
        f"Unique Minecraft usernames available: {len(set(MINECRAFT_USERNAMES_LIST))} (removed {duplicate_username_count} duplicates/empty)\n"
        f"----------------------------------------------------------------------\n"
        f"Output file: {output_parquet}\n"
        f"======================================================================"
    )

    # Log conversation counts per source file
    logger.info("--- Source File Contributions (Initial Extraction) ---")
    for file, count in file_conversation_counts.items():
        logger.info(f"File '{file}' contributed: {count} conversations")
    logger.info("======================================================================")

# --- END OF FILE convert.py ---