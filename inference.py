import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
from Bio import Entrez
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os

# Configure logging
logger = logging.getLogger(__name__)

# Set email for PubMed (required)
Entrez.email = Entrez.email = os.environ.get('PUBMED_EMAIL', "your-email@example.com") # Use environment variable

class BioBERTSimilarity:
    """Lightweight BioBERT similarity calculator"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize BioBERT with error handling"""
        try:
            model_name = "dmis-lab/biobert-base-cased-v1.2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            logger.info("BioBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"BioBERT model could not be loaded: {e}")
            logger.info("Continuing without BioBERT - scores will be based on other metrics")
    
    def get_embedding(self, text):
        """Get BioBERT embedding for text"""
        if self.model is None or self.tokenizer is None:
            return np.random.randn(768)  # Fallback random embedding
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                  max_length=512, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        except Exception as e:
            logger.error(f"Error getting BioBERT embedding: {e}")
            return np.random.randn(768)
    
    def calculate_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        try:
            emb1 = self.get_embedding(text1).reshape(1, -1)
            emb2 = self.get_embedding(text2).reshape(1, -1)
            return cosine_similarity(emb1, emb2)[0][0]
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

class DrugAnalyzer:
    """Main drug analysis class combining GNN and BioBERT"""
    
    def __init__(self, model_components):
        """Initialize with loaded model components"""
        self.model = model_components['model']
        self.hetero_data = model_components['hetero_data']
        self.entity_to_id = model_components['entity_to_id']
        self.filtered_node_types = model_components['filtered_node_types']
        
        # Load DrugBank mapping
        self.drugbank_mapping = self.load_drugbank_mapping()
        
        # Initialize BioBERT
        self.biobert = BioBERTSimilarity()
        
        # Set device
        self.device = torch.device('cpu')  # Use CPU for Render.com
        self.model.to(self.device)
        
        logger.info(f"DrugAnalyzer initialized with {len(self.entity_to_id)} entities")
    
    def load_drugbank_mapping(self):
        """Load DrugBank ID to name mapping"""
        mapping = {}
        try:
            if os.path.exists('drugbank_vocabulary.csv'):
                df = pd.read_csv('drugbank_vocabulary.csv', delimiter=';', engine='python')
                for _, row in df.iterrows():
                    drug_id = str(row['Column1']).strip()
                    drug_name = str(row['Column3']).strip()
                    if drug_id and drug_name and drug_id != 'nan' and drug_name != 'nan':
                        mapping[drug_id] = drug_name
                logger.info(f"Loaded {len(mapping)} drugs from DrugBank vocabulary")
            else:
                logger.warning("DrugBank vocabulary file not found")
        except Exception as e:
            logger.error(f"Error loading DrugBank mapping: {e}")
        
        return mapping
    
    def get_drug_name(self, drug_id):
        """Convert DrugBank ID to drug name"""
        clean_id = drug_id.replace("Compound::", "").strip()
        return self.drugbank_mapping.get(clean_id, f"Unknown_{clean_id}")
    
    def check_drug_exists(self, drug_id):
        """Check if drug exists in knowledge graph"""
        if not drug_id.startswith('Compound::'):
            drug_id = f'Compound::{drug_id}'
        return drug_id in self.entity_to_id
    
    def calculate_gnn_score(self, drug_id, alzheimer_id="Disease::MESH:D000544"):
        """Calculate GNN-based repurposing score"""
        try:
            # Ensure proper format
            if not drug_id.startswith('Compound::'):
                drug_id = f'Compound::{drug_id}'
            
            # Check if entities exist
            if drug_id not in self.entity_to_id or alzheimer_id not in self.entity_to_id:
                return 0.0, "Entity not found in knowledge graph"
            
            # Get embeddings
            self.model.eval()
            with torch.no_grad():
                node_embeddings = self.model(self.hetero_data.x_dict, self.hetero_data.edge_index_dict)
            
            # Get disease embedding
            disease_local_idx = self.hetero_data['Disease'].global_to_local[self.entity_to_id[alzheimer_id]]
            disease_embedding = node_embeddings['Disease'][disease_local_idx]
            
            # Get drug embedding
            drug_local_idx = self.hetero_data['Compound'].global_to_local[self.entity_to_id[drug_id]]
            drug_embedding = node_embeddings['Compound'][drug_local_idx]
            
            # Calculate similarity
            similarity = F.cosine_similarity(
                disease_embedding.unsqueeze(0), 
                drug_embedding.unsqueeze(0)
            ).item()
            
            # Normalize to 0-100 scale
            score = max(0, min(100, (similarity + 1) * 50))
            
            return score, "success"
            
        except Exception as e:
            logger.error(f"Error calculating GNN score for {drug_id}: {e}")
            return 0.0, f"GNN calculation error: {str(e)}"
    
    def fetch_pubmed_evidence(self, drug_name, max_refs=3):
        """Fetch PubMed evidence with BioBERT scoring"""
        query = f'("{drug_name}"[Title/Abstract]) AND ("Alzheimer"[Title/Abstract])'
        evidence = []
        
        try:
            # Search PubMed
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_refs)
            record = Entrez.read(handle)
            pmids = record["IdList"]
            handle.close()
            
            if not pmids:
                return evidence
            
            # Fetch article details
            handle = Entrez.efetch(db="pubmed", id=pmids, retmode="xml")
            articles = Entrez.read(handle)
            handle.close()
            
            for article in articles["PubmedArticle"]:
                try:
                    article_info = article["MedlineCitation"]["Article"]
                    
                    title = article_info.get("ArticleTitle", "N/A")
                    if title == "N/A" or len(title) < 10:
                        continue
                    
                    # Extract metadata
                    pub_date = article_info.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                    year = pub_date.get("Year", "N/A")
                    
                    authors = []
                    for a in article_info.get("AuthorList", []):
                        last, first = a.get("LastName", ""), a.get("ForeName", "")
                        if last and first:
                            authors.append(f"{first} {last}")
                    authors_str = ", ".join(authors[:3]) if authors else "N/A"
                    
                    journal = article_info.get("Journal", {}).get("Title", "N/A")
                    
                    # Get abstract
                    abstract_parts = []
                    if "Abstract" in article_info:
                        for ab in article_info["Abstract"].get("AbstractText", []):
                            if isinstance(ab, dict) and "#text" in ab:
                                abstract_parts.append(ab["#text"])
                            elif isinstance(ab, str):
                                abstract_parts.append(ab)
                    abstract = " ".join(abstract_parts)
                    
                    # Calculate BioBERT score
                    biobert_score = 0
                    if self.biobert.model is not None:
                        alzheimer_concepts = [
                            "treatment of Alzheimer's disease",
                            "cognitive improvement", 
                            "dementia therapy"
                        ]
                        scores = []
                        for concept in alzheimer_concepts:
                            sim = self.biobert.calculate_similarity(
                                f"{title} {abstract[:300]}", concept
                            )
                            scores.append(sim)
                        biobert_score = np.mean(scores) * 100
                    
                    evidence.append({
                        "title": title,
                        "year": year,
                        "authors": authors_str,
                        "journal": journal,
                        "biobert_score": round(biobert_score, 2),
                        "abstract": abstract[:200] + "..." if len(abstract) > 200 else abstract
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing article: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching PubMed evidence: {e}")
        
        return evidence
    
    def analyze_drug(self, drug_id):
        """Main drug analysis function"""
        try:
            # Get drug name
            drug_name = self.get_drug_name(drug_id)
            
            # Check if drug exists in KG
            exists_in_kg = self.check_drug_exists(drug_id)
            
            result = {
                'drug_id': drug_id,
                'drug_name': drug_name,
                'exists_in_kg': exists_in_kg,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            if exists_in_kg:
                # Calculate GNN score
                gnn_score, gnn_status = self.calculate_gnn_score(drug_id)
                result.update({
                    'gnn_score': round(gnn_score, 2),
                    'gnn_status': gnn_status,
                })
                
                # Get PubMed evidence
                evidence = self.fetch_pubmed_evidence(drug_name)
                result.update({
                    'evidence_count': len(evidence),
                    'evidence': evidence,
                    'analysis_status': 'completed'
                })
                
            else:
                result.update({
                    'gnn_score': 0,
                    'gnn_status': 'Drug not found in knowledge graph',
                    'evidence_count': 0,
                    'evidence': [],
                    'analysis_status': 'not_found'
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in analyze_drug for {drug_id}: {e}")
            return {
                'drug_id': drug_id,
                'error': str(e),
                'analysis_status': 'failed',
                'timestamp': pd.Timestamp.now().isoformat()
            }