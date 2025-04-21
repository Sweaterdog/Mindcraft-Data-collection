// --- START OF FILE logger.js ---

import { writeFileSync, mkdirSync, existsSync, appendFileSync, readFileSync } from 'fs';
import { join } from 'path';
import settings from '../../settings.js'; // Import settings
import path from 'path'; // Needed for path operations

// --- Configuration ---
const LOGS_DIR = './logs';
const VISION_IMAGES_DIR = join(LOGS_DIR, 'images'); // Directory to optionally store logged images

// --- Log File Paths ---
const REASONING_LOG_FILE = join(LOGS_DIR, 'reasoning_logs.csv');
const NORMAL_LOG_FILE = join(LOGS_DIR, 'normal_logs.csv');
const VISION_LOG_FILE = join(LOGS_DIR, 'vision_logs.csv'); // For vision data

// --- Log Headers ---
const TEXT_LOG_HEADER = 'input,output\n';
// Storing relative path to image and the transcribed text
const VISION_LOG_HEADER = 'image_path,text\n';

// --- Log Counters ---
let logCounts = {
    normal: 0,
    reasoning: 0,
    vision: 0,
    total: 0,
    skipped_disabled: 0,
    skipped_empty: 0,
    vision_images_saved: 0,
};

// --- Helper Functions ---
function ensureDirectoryExistence(dirPath) {
    if (!existsSync(dirPath)) {
        try {
            mkdirSync(dirPath, { recursive: true });
            console.log(`[Logger] Created directory: ${dirPath}`);
        } catch (error) {
            console.error(`[Logger] Error creating directory ${dirPath}:`, error);
            return false;
        }
    }
    return true;
}

function countLogEntries(logFile) {
    if (!existsSync(logFile)) return 0;
    try {
        const data = readFileSync(logFile, 'utf8');
        const lines = data.split('\n').filter(line => line.trim());
        // Check if the first line looks like a header before subtracting
        const hasHeader = lines.length > 0 && lines[0].includes(',');
        return Math.max(0, hasHeader ? lines.length - 1 : lines.length);
    } catch (err) {
        console.error(`[Logger] Error reading log file ${logFile}:`, err);
        return 0;
    }
}


function ensureLogFile(logFile, header) {
     if (!ensureDirectoryExistence(path.dirname(logFile))) return false; // Ensure parent dir exists

     if (!existsSync(logFile)) {
        try {
            writeFileSync(logFile, header);
            console.log(`[Logger] Created log file: ${logFile}`);
        } catch (error) {
            console.error(`[Logger] Error creating log file ${logFile}:`, error);
            return false;
        }
    } else {
         try {
            const content = readFileSync(logFile, 'utf-8');
            const headerLine = header.split('\n')[0];
            // If file is empty or header doesn't match, overwrite/create header
            if (!content.trim() || !content.startsWith(headerLine)) {
                 // Attempt to prepend header if file has content but wrong/no header
                 if(content.trim() && !content.startsWith(headerLine)) {
                    console.warn(`[Logger] Log file ${logFile} seems to be missing or has an incorrect header. Prepending correct header.`);
                    writeFileSync(logFile, header + content);
                 } else {
                    // File is empty or correctly headed, just ensure header is there
                     writeFileSync(logFile, header);
                 }
                 console.log(`[Logger] Ensured header in log file: ${logFile}`);
            }
        } catch (error) {
            console.error(`[Logger] Error checking/writing header for log file ${logFile}:`, error);
            // Proceed cautiously, maybe log an error and continue?
        }
    }
    return true;
}


function writeToLogFile(logFile, csvEntry) {
    try {
        appendFileSync(logFile, csvEntry);
        // console.log(`[Logger] Logged data to ${logFile}`); // Keep console less noisy
    } catch (error) {
        console.error(`[Logger] Error writing to CSV log file ${logFile}:`, error);
    }
}

// --- Auto-Detection for Log Type (Based on Response Content) ---
function determineLogType(response) {
    // Reasoning check: needs <think>...</think> but ignore the specific 'undefined' placeholder
    const isReasoning = response.includes('<think>') && response.includes('</think>') && !response.includes('<think>\nundefined</think>');

    if (isReasoning) {
        return 'reasoning';
    } else {
        return 'normal';
    }
}

function sanitizeForCsv(value) {
    if (typeof value !== 'string') {
        value = String(value);
    }
    // Escape double quotes by doubling them and enclose the whole string in double quotes
    return `"${value.replace(/"/g, '""')}"`;
}


// --- Main Logging Function (for text-based input/output) ---
export function log(input, response) {
    const trimmedInputStr = input ? (typeof input === 'string' ? input.trim() : JSON.stringify(input)) : "";
    const trimmedResponse = response ? String(response).trim() : ""; // Ensure response is a string

    // Basic filtering
    if (!trimmedInputStr && !trimmedResponse) {
        logCounts.skipped_empty++;
        return;
    }
    if (trimmedInputStr === trimmedResponse) {
         logCounts.skipped_empty++;
        return;
    }
     // Avoid logging common error messages that aren't useful training data
    const errorMessages = [
        "My brain disconnected, try again.",
        "My brain just kinda stopped working. Try again.",
        "I thought too hard, sorry, try again.",
        "*no response*",
        "No response received.",
        "No response data.",
        "Failed to send", // Broader match
        "Error:", // Broader match
        "Vision is only supported",
        "Context length exceeded",
        "Image input modality is not enabled",
        "An unexpected error occurred",
        // Add more generic errors/placeholders as needed
    ];
    // Also check for responses that are just the input repeated (sometimes happens with errors)
    if (errorMessages.some(err => trimmedResponse.includes(err)) || trimmedResponse === trimmedInputStr) {
        logCounts.skipped_empty++;
        // console.warn(`[Logger] Skipping log due to error/placeholder/repeat: "${trimmedResponse.substring(0, 70)}..."`);
        return;
    }


    const logType = determineLogType(trimmedResponse);
    let logFile;
    let header;
    let settingFlag;

    switch (logType) {
        case 'reasoning':
            logFile = REASONING_LOG_FILE;
            header = TEXT_LOG_HEADER;
            settingFlag = settings.log_reasoning_data;
            break;
        case 'normal':
        default:
            logFile = NORMAL_LOG_FILE;
            header = TEXT_LOG_HEADER;
            settingFlag = settings.log_normal_data;
            break;
    }

    // Check if logging for this type is enabled
    if (!settingFlag) {
        logCounts.skipped_disabled++;
        return;
    }

    // Ensure directory and file exist
    if (!ensureLogFile(logFile, header)) return; // ensureLogFile now checks parent dir too

    // Prepare the CSV entry using the sanitizer
    const safeInput = sanitizeForCsv(trimmedInputStr);
    const safeResponse = sanitizeForCsv(trimmedResponse);
    const csvEntry = `${safeInput},${safeResponse}\n`;

    // Write to the determined log file
    writeToLogFile(logFile, csvEntry);

    // Update counts
    logCounts[logType]++;
    logCounts.total++; // Total here refers to text logs primarily

    // Display summary periodically (based on total text logs)
    if (logCounts.normal + logCounts.reasoning > 0 && (logCounts.normal + logCounts.reasoning) % 20 === 0) {
       printSummary();
    }
}

// --- Dedicated Vision Logging Function ---
// Call this function from your Mindcraft code AFTER you receive the transcription
// Example call: logVision(imageBuffer, transcribedText);
export function logVision(imageBuffer, transcribedText) {
    if (!settings.log_vision_data) {
         logCounts.skipped_disabled++;
        return;
    }

    const trimmedText = transcribedText ? String(transcribedText).trim() : "";

    if (!imageBuffer || !trimmedText) {
        logCounts.skipped_empty++;
        console.warn("[Logger] Skipping vision log: Image buffer or transcribed text is empty.");
        return;
    }

    // Ensure log file and image directory exist
    if (!ensureLogFile(VISION_LOG_FILE, VISION_LOG_HEADER)) return;
    if (!ensureDirectoryExistence(VISION_IMAGES_DIR)) return;

    // Generate a unique filename for the image (e.g., using timestamp and random number)
    const timestamp = Date.now();
    const randomSuffix = Math.random().toString(36).substring(2, 8);
    const imageFilename = `vision_${timestamp}_${randomSuffix}.jpg`; // Assuming JPEG
    const relativeImagePath = path.join('images', imageFilename); // Relative path for CSV
    const fullImagePath = join(LOGS_DIR, relativeImagePath);

    try {
        // Save the image buffer to the file
        writeFileSync(fullImagePath, imageBuffer);
        logCounts.vision_images_saved++;
         // console.log(`[Logger] Saved vision image: ${fullImagePath}`); // Less verbose

        // Prepare CSV entry with the *relative* path and sanitized text
        const safeRelativePath = sanitizeForCsv(relativeImagePath);
        const safeText = sanitizeForCsv(trimmedText);
        const csvEntry = `${safeRelativePath},${safeText}\n`;

        // Log the entry
        writeToLogFile(VISION_LOG_FILE, csvEntry);
        logCounts.vision++;

        // Optional: Trigger summary display on vision logs too
        if (logCounts.vision > 0 && logCounts.vision % 10 === 0) { // More frequent summary for vision initially
            printSummary();
        }

    } catch (error) {
        console.error(`[Logger] Error saving vision image or writing vision log: ${error}`);
        // Decide if you still want to log the CSV entry even if image saving failed
        // Maybe log the text entry with an indicator that the image is missing?
        // For now, we just log the error and don't write the CSV row if image saving fails.
    }
}

function printSummary() {
    const totalStored = logCounts.normal + logCounts.reasoning + logCounts.vision;
    console.log('\n' + '='.repeat(60));
    console.log('LOGGER SUMMARY');
    console.log('-'.repeat(60));
    console.log(`Normal logs stored:    ${logCounts.normal}`);
    console.log(`Reasoning logs stored: ${logCounts.reasoning}`);
    console.log(`Vision logs stored:    ${logCounts.vision} (Images saved: ${logCounts.vision_images_saved})`);
    console.log(`Skipped (disabled):    ${logCounts.skipped_disabled}`);
    console.log(`Skipped (empty/err):   ${logCounts.skipped_empty}`);
    console.log('-'.repeat(60));
    console.log(`Total logs stored:     ${totalStored}`);
    console.log('='.repeat(60) + '\n');
}

// Initialize counts at startup
function initializeCounts() {
    logCounts.normal = countLogEntries(NORMAL_LOG_FILE);
    logCounts.reasoning = countLogEntries(REASONING_LOG_FILE);
    logCounts.vision = countLogEntries(VISION_LOG_FILE);
    // Total count will be accumulated during runtime
    console.log(`[Logger] Initialized log counts: Normal=${logCounts.normal}, Reasoning=${logCounts.reasoning}, Vision=${logCounts.vision}`);
}

initializeCounts();

// --- END OF FILE logger.js ---