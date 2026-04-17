from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os
from sentence_transformers import SentenceTransformer
SAVE_PATH = "models_weights/trocr"
os.makedirs(SAVE_PATH, exist_ok=True)



SENTENCE_PATH = "models_weights/sentence_transformer"
os.makedirs(SENTENCE_PATH, exist_ok=True)
print("⏳ Downloading Sentence Transformer...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
sentence_model.save(SENTENCE_PATH)
print("✅ Sentence Transformer saved!")