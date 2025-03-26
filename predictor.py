import re
import torch
import pickle
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

class DiagnosisPredictor:
    def __init__(self, model_dir, embeddings_path, dataset):
        # Load the IndoBERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertModel.from_pretrained(model_dir)
        
        # Load the precomputed embeddings dataset
        with open(embeddings_path, "rb") as f:
            self.dataset_embeddings = pickle.load(f)
        
        # Store the dataset containing diagnoses (DataFrame format expected)
        self.df = dataset
    
    def preprocess_text(self, text):
        """Preprocess input text."""
        text = text.lower()
        
        # Replace (+) and (-) within parentheses with words
        text = re.sub(r'\(\+\)', ' positif ', text)
        text = re.sub(r'\(-\)', ' negatif ', text)
        
        # Separate '/' between words but keep decimal numbers intact
        text = re.sub(r'(?<=\w)/(?=\w)', ' ', text)  # Replace '/' between words with space
        text = re.sub(r'[^\w\s\.\-]', ' ', text)      # Remove special characters except numbers and periods
        
        return text

    def get_bert_embeddings(self, text):
        """Generate IndoBERT embeddings for the input text."""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize and convert text to tensor
        inputs = self.tokenizer(processed_text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean pooling of the last hidden layer for the full representation
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.squeeze().numpy()
    
    def predict_diagnosis(self, user_ases_soap, user_diagnosis_dokter, top_n=1):
        """Predict diagnosis based on user input."""
        # Combine user inputs into one text
        user_text = user_ases_soap + " " + user_diagnosis_dokter
        
        # Get embeddings for the combined user input
        user_embedding = self.get_bert_embeddings(user_text)
        
        # Calculate cosine similarity between user embedding and dataset embeddings
        similarities = cosine_similarity([user_embedding], self.dataset_embeddings).flatten()
        
        # Get indices of the top-N highest similarities
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Retrieve primary and secondary diagnosis predictions
        prediksi_diagnosa_primer = self.df.iloc[top_indices]['diagnosa_primer_klaim'].values
        prediksi_diagnosa_sekunder = self.df.iloc[top_indices]['diagnosa_sekunder_klaim'].values
        
        return prediksi_diagnosa_primer, prediksi_diagnosa_sekunder