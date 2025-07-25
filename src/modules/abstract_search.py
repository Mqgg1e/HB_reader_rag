from typing import List
from src.modules.document import Document
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
# --- New Function to Generate Summaries ---

# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
GLOBAL_TOKENIZER = None
GLOBAL_MODEL = None


def _load_global_llm_model(model_name: str = "/kaggle/input/mistral-7b-instruct-v0.2/pytorch/mistral-7b/1"):
    """
    Helper function to load the LLM model and tokenizer globally.
    This ensures the model is loaded only once. Adapted for P100 GPU (CUDA).
    Includes optional 4-bit quantization for memory optimization.
    """
    global GLOBAL_TOKENIZER, GLOBAL_MODEL
    if GLOBAL_TOKENIZER is None or GLOBAL_MODEL is None:
        print(f"Loading LLM model and tokenizer '{model_name}' for the first time. This may take a while...")
        try:
            # Determine the device (CUDA if available, else CPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"DEBUG_MODEL_LOAD: Device detected: {device}")

            # --- Quantization Configuration for GPU (P100) ---
            # P100 GPUs often have limited VRAM, so 4-bit quantization is highly recommended.
            # Make sure 'bitsandbytes' and 'accelerate' libraries are installed:
            # !pip install -U bitsandbytes accelerate

            # Uncomment the following lines to enable 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # Enable 4-bit quantization
                bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit quantization
                bnb_4bit_use_double_quant=True,  # Double quantization for more precision
                bnb_4bit_compute_dtype=torch.float16  # Compute in float16 for P100 compatibility
            )
            # --- End Quantization Configuration ---

            GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
                model_name,
                # For P100, float16 (half-precision) is generally better for memory if bfloat16 is not supported or causes issues.
                # If using 4-bit quantization, bnb_4bit_compute_dtype will handle the compute precision.
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                quantization_config=quantization_config,  # Uncomment this line to use quantization
                # device_map="auto" is often good for GPUs, but for explicit control, we'll manually move.
                device_map="auto"  # Let transformers handle device mapping for GPU, often more robust
            )
            # With device_map="auto", explicit .to(device) might not be strictly necessary
            # but leaving it for consistency if needed for specific setups.
            # GLOBAL_MODEL.to(device) # Model is typically already on GPU if device_map="auto" is used
            GLOBAL_MODEL.eval()  # Set model to evaluation mode

            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(model_name)

            # --- CRITICAL FIX FOR "tokenizer does not have a padding token" ---
            if GLOBAL_TOKENIZER.pad_token is None:
                if GLOBAL_TOKENIZER.eos_token is not None:
                    GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token
                    print(
                        f"Setting pad_token_id to eos_token_id:{GLOBAL_TOKENIZER.eos_token_id} for open-end generation.")
                elif GLOBAL_TOKENIZER.unk_token is not None:
                    GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.unk_token
                    print(f"Setting pad_token_id to unk_token_id:{GLOBAL_TOKENIZER.unk_token_id}.")
                else:
                    # Fallback: add a new special token if neither EOS nor UNK exists
                    GLOBAL_TOKENIZER.add_special_tokens({'pad_token': '[PAD]'})
                    print("Added new [PAD] token as tokenizer's pad_token.")
            # --- END CRITICAL FIX ---
            print(f"Successfully loaded LLM model '{model_name}'.")
            # --- LLM Warm-up / Pre-run ---
            print("Performing LLM warm-up run...")
            warmup_prompt = "Hello, what is your name?"  # Simple, short prompt
            warmup_messages = [{"role": "user", "content": warmup_prompt}]

            # Apply chat template for warm-up prompt
            warmup_input_ids = GLOBAL_TOKENIZER.apply_chat_template(
                warmup_messages,
                return_tensors="pt",
                add_generation_prompt=True,
                truncation=True,
                max_length=GLOBAL_TOKENIZER.model_max_length  # Or a smaller safe value like 512 for warm-up
            ).to(GLOBAL_MODEL.device)

            with torch.no_grad():
                # Perform a small, quick generation for warm-up
                _ = GLOBAL_MODEL.generate(
                    warmup_input_ids,
                    max_new_tokens=50,  # Generate a small number of tokens
                    pad_token_id=GLOBAL_TOKENIZER.pad_token_id,
                    do_sample=False  # Keep it deterministic for warm-up
                )
            print("LLM warm-up complete.")
            # --- End LLM Warm-up ---
        except Exception as e:
            print(f"Error loading LLM model '{model_name}': {e}")
            print(
                "Please ensure you have enough VRAM/RAM and that 'transformers', 'torch', 'accelerate', 'bitsandbytes' (if using quantization) are installed correctly.")
            print("If you have a GPU, ensure CUDA is properly set up for PyTorch.")
            raise  # Re-raise the exception to stop execution if model loading fails


# --- Updated generate_summary_with_llm Function ---

def generate_summary_with_llm(text_content: str) -> str:
    """
    Generates a summary for the given text using a locally loaded LLM (Mistral 7B Instruct v0.2).

    Args:
        text_content (str): The full text of the chapter to summarize.

    Returns:
        str: The generated summary.
    """
    _load_global_llm_model()  # Ensure model is loaded

    # Reference global model and tokenizer
    tokenizer = GLOBAL_TOKENIZER
    model = GLOBAL_MODEL

    # Prompt engineering: Crafting a clear instruction for summarization
    # prompt = f"Please summarize the following Bible chapters so that they are concise and highlight the core points:\n\n{text_content}\n\nsummary："
    prompt = (
        "You are a biblical scholar. Summarize the following chapter from the Bible in 2-3 sentences, "
        "highlighting the key people, events, and theological significance. Use clear and concise English.\n\n"
        f"{text_content}\n\nSummary:"
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    print(f"  Generating summary for a chapter ({len(text_content)} characters)...")
    try:
        # --- CRITICAL CHANGE: Reduce max_length to fit GPU memory ---
        # 512 tokens is a reasonable starting point for P100 with 4-bit quantization.
        # You can experiment with 256, 768, 1024 based on results.
        MAX_INPUT_LENGTH = 8192

        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            truncation=True,  # Crucial for long texts to avoid exceeding model's context window
            max_length=MAX_INPUT_LENGTH  # Use the reduced max_length here
        )

        # --- NEW DEBUGGING FOR INPUTS TENSOR ---
        print(f"DEBUG_INPUTS_TENSOR: Type of 'input_ids' from apply_chat_template: {type(input_ids)}")
        if isinstance(input_ids, torch.Tensor):
            print(f"DEBUG_INPUTS_TENSOR: Shape of 'input_ids': {input_ids.shape}")
            print(f"DEBUG_INPUTS_TENSOR: Dims of 'input_ids': {input_ids.ndim}")
        else:
            print("DEBUG_INPUTS_TENSOR: 'input_ids' is NOT a torch.Tensor as expected!")
            raise TypeError("Expected input_ids to be a torch.Tensor after apply_chat_template.")
        # --- END NEW DEBUGGING ---

        # Manually create attention_mask. It's 1 for non-pad tokens, 0 for pad tokens.
        # Ensure pad_token_id is set for the tokenizer (handled in _load_global_llm_model)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # Determine the device (CUDA if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        print(
            f"DEBUG_PRE_GENERATE: input_ids device: {input_ids.device}, shape: {input_ids.shape}, dtype: {input_ids.dtype}")
        print(
            f"DEBUG_PRE_GENERATE: attention_mask device: {attention_mask.device}, shape: {attention_mask.shape}, dtype: {attention_mask.dtype}")

        generation_kwargs = {
            "max_new_tokens": 768,  # Keep reduced for now
            "do_sample": False,  # Keep deterministic generation for summarization
            "pad_token_id": tokenizer.pad_token_id,
            "return_dict_in_generate": True,  # Ensure outputs is a ModelOutput object
            # "num_beams": 4, # Commented out for simpler greedy decoding, less memory
            # "early_stopping": True, # Only relevant with beam search
            # "output_scores": True, # Can be useful for debugging, but not strictly needed
        }

        with torch.no_grad():  # Disable gradient calculation for inference to save memory and speed up
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )

        print(f"DEBUG_POST_GENERATE: Type of outputs.sequences: {type(outputs.sequences)}")
        print(f"DEBUG_POST_GENERATE: Shape of outputs.sequences: {outputs.sequences.shape}")
        print(f"DEBUG_POST_GENERATE: outputs.sequences.ndim: {outputs.sequences.ndim}")

        outputs_tensor = outputs.sequences

        print(f"DEBUG_STEP_1: Type of outputs: {type(outputs)}")
        print(f"DEBUG_STEP_2: Type of outputs.sequences: {type(outputs.sequences)}")
        print(f"DEBUG_STEP_3: Shape of outputs.sequences: {outputs.sequences.shape}")
        print(f"DEBUG_STEP_4: outputs.sequences.ndim: {outputs.sequences.ndim}")

        outputs_cpu = outputs_tensor.cpu()
        print(f"DEBUG_STEP_5: Shape of outputs_cpu (after .cpu()): {outputs_cpu.shape}")
        print(f"DEBUG_STEP_6: Number of dimensions of outputs_cpu: {outputs_cpu.ndim}")

        # --- NEW ROBUST INDEXING ---
        # Flatten the tensor into a 1D tensor, which tokenizer.decode expects for a single sequence.
        # This handles cases where outputs_cpu might be (1, N) or even (N,)
        generated_tokens = outputs_cpu.view(-1)
        print(f"DEBUG_STEP_7: Shape of generated_tokens (after view(-1)): {generated_tokens.shape}")
        print(f"DEBUG_STEP_8: Number of dimensions of generated_tokens: {generated_tokens.ndim}")
        # --- END NEW ROBUST INDEXING ---

        input_ids_cpu = input_ids.cpu()
        prompt_length = input_ids_cpu.shape[1]
        print(f"DEBUG_STEP_9: prompt_length: {prompt_length}")
        print(f"DEBUG_STEP_10: generated_tokens.shape[0]: {generated_tokens.shape[0]}")  # generated_tokens is now 1D

        # Slice the generated tokens to remove the prompt part
        if prompt_length >= generated_tokens.shape[0]:
            print(
                f"Warning: Prompt length ({prompt_length}) is greater than or equal to generated sequence length ({generated_tokens.shape[0]}). Skipping prompt removal.")
            summary_tokens = generated_tokens
        else:
            summary_tokens = generated_tokens[prompt_length:]
            print(f"DEBUG_STEP_11: Shape of summary_tokens (after slicing): {summary_tokens.shape}")

        summary = tokenizer.decode(summary_tokens, skip_special_tokens=True).strip()
        print(f"DEBUG_STEP_12: Summary generated successfully.")
        return summary
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "out of memory" in str(e):  # Changed for CUDA OOM
            print(
                f"Error during summary generation: GPU out of memory. Try reducing max_new_tokens or input text length.")
            return "Summary generation failed due to insufficient GPU memory."
        else:
            print(f"Error during summary generation: {e}")
            return "Summary generation failed."
    except Exception as e:
        print(f"An unexpected error occurred during summary generation: {e}")
        return "Summary generation failed due to an unexpected error."



def create_summarized_chapter_documents(
    chapter_documents: List[Document],
    output_jsonl_file: str = "/kaggle/working/summary.jsonl"
) -> List[Document]:
    """
    Generates summaries for each chapter document using an LLM and creates new documents
    with the summary as page_content and original chapter as metadata.
    Optionally writes each summarized chapter to a JSONL file after processing.

    Args:
        chapter_documents (List[Document]): A list of Document objects, where each represents a full chapter.
        output_jsonl_file (str, optional): Path to a JSONL file to append each summarized chapter.

    Returns:
        List[Document]: A list of new Document objects. Each new document will have:
                        - page_content: The generated summary of the chapter.
                        - metadata: Contains the original chapter's metadata, plus the full chapter content
                                    under a new key (e.g., 'original_chapter_content').
    """
    summarized_chunks = []
    print(f"Starting summary generation for {len(chapter_documents)} chapters...")
    for i, chapter_doc in enumerate(chapter_documents):
        print(
            f"  Processing chapter {i + 1}/{len(chapter_documents)}: {chapter_doc.metadata.get('location', 'Unknown Location')}")

        # Get the full chapter content
        full_chapter_content = chapter_doc.page_content

        # Generate summary using the LLM (or simulated LLM)
        chapter_summary = generate_summary_with_llm(full_chapter_content)

        # Create new metadata for the summarized chunk
        # It should contain all original chapter metadata PLUS the original full content
        new_metadata = chapter_doc.metadata.copy()  # Copy existing metadata
        new_metadata["original_chapter_content"] = full_chapter_content  # Store original content
        new_metadata["chunk_type"] = "chapter_summary"  # Indicate chunk type

        # The page_content of this new Document will be the summary
        summarized_doc = Document(page_content=chapter_summary, metadata=new_metadata)
        summarized_chunks.append(summarized_doc)

        # Write to file after each chapter if output_jsonl_file is provided
        if output_jsonl_file:
            with open(output_jsonl_file, "a", encoding="utf-8") as f:
                json.dump({
                    "page_content": summarized_doc.page_content,
                    "metadata": summarized_doc.metadata
                }, f, ensure_ascii=False)
                f.write("\n")

    print("Summary generation complete.")
    return summarized_chunks



# # --- End-to-End Usage Example with Summary Generation ---
#
# # 1. Define file paths
# bible_raw_txt_file = os.path.join("data", "raw_data", "AKJV.txt")
# processed_verses_jsonl_file = os.path.join("processed_data", "akjv_verses.jsonl")
# processed_chapters_jsonl_file = os.path.join("processed_data", "akjv_chapters.jsonl")
# # New file for summarized chapters
# processed_summaries_jsonl_file = os.path.join("processed_data", "akjv_chapter_summaries.jsonl")
#
# # For demonstration, we'll use a simulated raw text content
# # This `sample_cleaned_bible_text_for_demo` should reflect content after header cleaning for parse_bible_verses
# sample_cleaned_bible_text_for_demo = """
# Genesis
#
# Gen.1:1 In the beginning God created the heaven and the earth.
# Gen.1:2 And the earth was without form, and void; and darkness was on the face of the deep. And the Spirit of God moved on the face of the waters.
# Gen.1:3 And God said, Let there be light: and there was light.
# Gen.1:4 And God saw the light, that it was good: and God divided the light from the darkness.
# Gen.1:5 And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day.
#
# Gen.2:1 Thus the heavens and the earth were finished, and all the host of them.
# Gen.2:2 And on the seventh day God ended his work which he had made; and he rested on the seventh day from all his work which he had made.
#
# Exodus
#
# Exo.1:1 Now these are the names of the children of Israel, which came into Egypt with Jacob: every man and his household came with them.
# Exo.1:2 Reuben, Simeon, Levi, and Judah, Issachar, Zebulun, and Benjamin,
# Exo.1:3 Dan, and Naphtali, Gad, and Asher.
# """
#
# try:
#     # --- Part 1: Initial Processing (Verse Level) ---
#     # raw_bible_text_content = read_text_file_robustly(bible_raw_txt_file, relative_to_script=True)
#     # cleaned_text = clean_bible_text_header(raw_bible_text_content)
#     # Using the sample for demo:
#     cleaned_text = sample_cleaned_bible_text_for_demo
#
#     parsed_verse_documents = parse_bible_verses(cleaned_text)
#     print("\n--- Verses Parsed ---")
#
#     # --- Part 2: Chunking by Chapter ---
#     chapter_documents = chunk_bible_by_chapter(parsed_verse_documents)
#     print("\n--- Chapters Chunked ---")
#
#     # --- Part 3: Generate Summaries for Chapters ---
#     summarized_chapter_documents = create_summarized_chapter_documents(chapter_documents)
#     print("\n--- Summaries Generated ---")
#     print(
#         f"Example summarized chapter: {summarized_chapter_documents[0].metadata['location']} - Summary: {summarized_chapter_documents[0].page_content}")
#     print(
#         f"Original content for first summarized chapter: {summarized_chapter_documents[0].metadata['original_chapter_content'][:100]}...")
#
#     # Save the summarized chapter documents
#     save_documents_to_jsonl(summarized_chapter_documents, processed_summaries_jsonl_file)
#     print("\n--- Summarized Chapters Saved ---")
#
#     # (Optional) Load the summarized documents back to verify
#     loaded_summaries = load_documents_from_jsonl(processed_summaries_jsonl_file)
#     print("\n--- Summarized Chapters Loaded ---")
#     print(f"Loaded {len(loaded_summaries)} summarized documents. First loaded summary: {loaded_summaries[0]}")
#
# except (FileNotFoundError, IOError, json.JSONDecodeError) as e:
#     print(f"An error occurred during the workflow: {e}")
