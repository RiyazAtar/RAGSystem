"""
RAG System with Conversation History
Implements vector store, retrieval, and LLM-based Q&A with session management
"""

import os
import logging
import json
import pickle
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class VectorStore:
    """FAISS-based vector store"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: Optional[np.ndarray] = None
        self.documents: List[Dict] = []
        self.index = None
        
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """Add documents with embeddings"""
        self.documents.extend(documents)
        
        if self.vectors is None:
            self.vectors = embeddings
        else:
            self.vectors = np.vstack([self.vectors, embeddings])
        
        if FAISS_AVAILABLE:
            self._build_index()
    
    def _build_index(self):
        """Build FAISS index"""
        if self.vectors is None:
            return
        
        vectors = self.vectors.astype('float32')
        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar documents"""
        if self.vectors is None or len(self.vectors) == 0:
            return []
        
        query = query_embedding.reshape(1, -1).astype('float32')
        
        if FAISS_AVAILABLE and self.index is not None:
            faiss.normalize_L2(query)
            scores, indices = self.index.search(query, min(top_k, len(self.documents)))
            results = [(self.documents[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
        else:
            # NumPy fallback
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            normalized = self.vectors / (norms + 1e-10)
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            similarities = np.dot(normalized, query_norm.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [(self.documents[idx], float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'dimension': self.dimension, 'vectors': self.vectors, 'documents': self.documents}, f)
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.dimension = state['dimension']
        self.vectors = state['vectors']
        self.documents = state['documents']
        if FAISS_AVAILABLE and self.vectors is not None:
            self._build_index()


class EmbeddingModel:
    """Sentence embedding model"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384
        
    def load(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            logger.warning("Using mock embeddings")
    
    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)
        else:
            # Mock embeddings
            return np.array([np.random.randn(self.dimension).astype('float32') for _ in texts])


class ConversationManager:
    """Manages conversation history for multiple sessions with persistent storage"""

    def __init__(self, max_history: int = 10, db_path: str = "conversations.db", use_db: bool = True):
        self.max_history = max_history
        self.use_db = use_db
        self.db_path = db_path

        # In-memory cache
        self.sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.session_metadata: Dict[str, Dict] = {}

        # Initialize database
        if self.use_db:
            self._init_database()
            self._load_recent_sessions()
        
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to session history"""
        timestamp = datetime.now().isoformat()

        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }

        # Add to in-memory cache
        self.sessions[session_id].append(message)

        # Trim if too long
        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][-(self.max_history * 2):]

        # Update metadata
        if session_id not in self.session_metadata:
            self.session_metadata[session_id] = {"created": timestamp}
        self.session_metadata[session_id]["last_activity"] = timestamp
        self.session_metadata[session_id]["message_count"] = len(self.sessions[session_id])

        # Save to database
        if self.use_db:
            self._save_message_to_db(session_id, role, content, timestamp)
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get session history"""
        return self.sessions.get(session_id, [])
    
    def get_context(self, session_id: str, last_n: int = 6) -> str:
        """Get formatted conversation context"""
        history = self.get_history(session_id)
        if not history:
            return ""
        
        context_parts = ["Previous conversation:"]
        for msg in history[-last_n:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def clear_session(self, session_id: str):
        """Clear session history"""
        self.sessions[session_id] = []
        if session_id in self.session_metadata:
            self.session_metadata[session_id]["message_count"] = 0

        # Clear from database
        if self.use_db:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error clearing session from database: {e}")
    
    def get_all_sessions(self) -> Dict[str, Dict]:
        """Get all session info"""
        return {
            sid: {
                "message_count": len(msgs),
                "turns": len(msgs) // 2,
                **self.session_metadata.get(sid, {})
            }
            for sid, msgs in self.sessions.items()
        }
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'sessions': dict(self.sessions), 'metadata': self.session_metadata}, f)
    
    def load(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.sessions = defaultdict(list, state['sessions'])
            self.session_metadata = state['metadata']

    # ==================== Database Methods ====================

    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0
                )
            """)

            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_messages
                ON messages(session_id, timestamp)
            """)

            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def _save_message_to_db(self, session_id: str, role: str, content: str, timestamp: str):
        """Save a single message to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert or update session metadata
            cursor.execute("""
                INSERT INTO sessions (session_id, created_at, last_activity, message_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_activity = ?,
                    message_count = message_count + 1
            """, (session_id, timestamp, timestamp, timestamp))

            # Insert message
            cursor.execute("""
                INSERT INTO messages (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, timestamp))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving message to database: {e}")

    def _load_recent_sessions(self, limit: int = 100):
        """Load recent sessions from database into memory"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get recent session IDs
            cursor.execute("""
                SELECT session_id FROM sessions
                ORDER BY last_activity DESC
                LIMIT ?
            """, (limit,))

            session_ids = [row[0] for row in cursor.fetchall()]

            # Load messages for each session
            for session_id in session_ids:
                cursor.execute("""
                    SELECT role, content, timestamp
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """, (session_id,))

                messages = []
                for role, content, timestamp in cursor.fetchall():
                    messages.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })

                self.sessions[session_id] = messages

                # Load metadata
                cursor.execute("""
                    SELECT created_at, last_activity, message_count
                    FROM sessions
                    WHERE session_id = ?
                """, (session_id,))

                row = cursor.fetchone()
                if row:
                    self.session_metadata[session_id] = {
                        "created": row[0],
                        "last_activity": row[1],
                        "message_count": row[2]
                    }

            conn.close()
            logger.info(f"Loaded {len(session_ids)} sessions from database")
        except Exception as e:
            logger.error(f"Error loading sessions from database: {e}")

    def get_session_from_db(self, session_id: str) -> List[Dict[str, str]]:
        """Load a specific session from database"""
        if not self.use_db:
            return self.sessions.get(session_id, [])

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT role, content, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))

            messages = []
            for role, content, timestamp in cursor.fetchall():
                messages.append({
                    "role": role,
                    "content": content,
                    "timestamp": timestamp
                })

            conn.close()

            # Update in-memory cache
            if messages:
                self.sessions[session_id] = messages

            return messages
        except Exception as e:
            logger.error(f"Error loading session from database: {e}")
            return self.sessions.get(session_id, [])


class ResponseGenerator:
    """Generate responses using templates or LLM"""
    
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.llm = None
        self.tokenizer = None
        
    def load_llm(self, model_path: str = None):
        """Load LLM for response generation"""
        if not self.use_llm:
            logger.info("Using template-based responses")
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            model_name = model_path or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            logger.info(f"Loading LLM: {model_name}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
            
            self.llm = pipeline("text-generation", model=model, tokenizer=self.tokenizer,
                               max_new_tokens=256, temperature=0.7, top_p=0.95,
                               pad_token_id=self.tokenizer.eos_token_id)
            logger.info("LLM loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load LLM: {e}. Using templates.")
            self.use_llm = False
    
    def generate(self, question: str, context: str, conv_history: str = "") -> str:
        """Generate response"""
        if self.use_llm and self.llm:
            return self._llm_generate(question, context, conv_history)
        return self._template_generate(question, context)
    
    def _llm_generate(self, question: str, context: str, conv_history: str) -> str:
        """Generate using LLM"""
        prompt = f"""<|system|>
You are a helpful e-commerce assistant. Answer questions about products based on the context.
Be concise and accurate. Use the conversation history for context.
</s>
<|user|>
{conv_history}

Product Information:
{context}

Question: {question}
</s>
<|assistant|>
"""
        try:
            result = self.llm(prompt)
            response = result[0]['generated_text']
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._template_generate(question, context)
    
    def _template_generate(self, question: str, context: str) -> str:
        """Generate using templates"""
        q_lower = question.lower()
        products = self._parse_products(context)
        
        if not products:
            return "I couldn't find relevant products. Please try a different question."
        
        if any(w in q_lower for w in ['price', 'cost', 'cheap', 'expensive', 'budget']):
            return self._price_response(products, q_lower)
        elif any(w in q_lower for w in ['best', 'recommend', 'suggest', 'top']):
            return self._recommend_response(products)
        elif any(w in q_lower for w in ['discount', 'offer', 'deal', 'sale']):
            return self._discount_response(products)
        elif any(w in q_lower for w in ['rating', 'review', 'popular']):
            return self._rating_response(products)
        else:
            return self._general_response(products)
    
    def _parse_products(self, context: str) -> List[Dict]:
        """Parse products from context"""
        products = []
        current = {}
        
        for line in context.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                current[key.strip().lower().replace(' ', '_')] = value.strip()
            elif not line and current:
                products.append(current)
                current = {}
        
        if current:
            products.append(current)
        
        return products
    
    def _price_response(self, products: List[Dict], question: str) -> str:
        """Price-focused response"""
        sorted_prods = sorted(products, key=lambda x: self._extract_number(x.get('sale_price', '999999')))
        
        if 'cheap' in question or 'under' in question or 'budget' in question:
            p = sorted_prods[0]
            return f"The most affordable option is **{p.get('product', 'Unknown')[:50]}** at {p.get('sale_price', 'N/A')} ({p.get('discount', '0%')} off). Rating: {p.get('rating', 'N/A')}"
        
        resp = "Here are products sorted by price:\n\n"
        for i, p in enumerate(sorted_prods[:5], 1):
            resp += f"{i}. **{p.get('product', 'Unknown')[:40]}** - {p.get('sale_price', 'N/A')}\n"
        return resp
    
    def _recommend_response(self, products: List[Dict]) -> str:
        """Recommendation response"""
        sorted_prods = sorted(products, key=lambda x: self._extract_number(x.get('rating', '0')), reverse=True)
        
        if sorted_prods:
            top = sorted_prods[0]
            resp = f"Based on ratings, I recommend **{top.get('product', 'Unknown')}**.\n\n"
            resp += f"• Price: {top.get('sale_price', 'N/A')} ({top.get('discount', 'N/A')})\n"
            resp += f"• Rating: {top.get('rating', 'N/A')}\n"
            resp += f"• Brand: {top.get('brand', 'N/A')}\n"
            
            if len(sorted_prods) > 1:
                resp += "\n**Other options:**\n"
                for p in sorted_prods[1:3]:
                    resp += f"• {p.get('product', 'Unknown')[:40]} - {p.get('rating', 'N/A')}\n"
            return resp
        
        return "I couldn't find products to recommend."
    
    def _discount_response(self, products: List[Dict]) -> str:
        """Discount-focused response"""
        sorted_prods = sorted(products, key=lambda x: self._extract_number(x.get('discount', '0%')), reverse=True)
        
        resp = "Best discount deals:\n\n"
        for i, p in enumerate(sorted_prods[:5], 1):
            resp += f"{i}. **{p.get('product', 'Unknown')[:40]}** - {p.get('discount', 'N/A')} off (Now {p.get('sale_price', 'N/A')})\n"
        return resp
    
    def _rating_response(self, products: List[Dict]) -> str:
        """Rating-focused response"""
        sorted_prods = sorted(products, key=lambda x: self._extract_number(x.get('rating', '0')), reverse=True)
        
        resp = "Top-rated products:\n\n"
        for i, p in enumerate(sorted_prods[:5], 1):
            resp += f"{i}. **{p.get('product', 'Unknown')[:40]}** - ⭐ {p.get('rating', 'N/A')}\n"
        return resp
    
    def _general_response(self, products: List[Dict]) -> str:
        """General response"""
        # Sort products by rating (highest first)
        sorted_prods = sorted(products, key=lambda x: self._extract_number(x.get('rating', '0')), reverse=True)
        resp = f"I found {len(products)} relevant products (sorted by reviews):\n\n"
        for i, p in enumerate(sorted_prods[:5], 1):
            resp += f"**{i}. {p.get('product', 'Unknown')}**\n"
            resp += f"   • Price: {p.get('sale_price', 'N/A')} ({p.get('discount', 'N/A')})\n"
            resp += f"   • Rating: {p.get('rating', 'N/A')}\n\n"
        return resp
    
    def _extract_number(self, s: str) -> float:
        """Extract number from string"""
        import re
        nums = re.findall(r'[\d.]+', str(s).replace(',', ''))
        return float(nums[0]) if nums else 0


class RAGSystem:
    """Complete RAG system with conversation history"""
    
    def __init__(self, config: Dict = None):
        config = config or {}
        self.embedding_model = EmbeddingModel(config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'))
        self.vector_store: Optional[VectorStore] = None

        # Initialize conversation manager with persistent storage
        db_path = config.get('db_path', 'conversations/conversations.db')
        use_db = config.get('use_persistent_storage', True)
        self.conversation_manager = ConversationManager(
            max_history=config.get('max_history', 10),
            db_path=db_path,
            use_db=use_db
        )

        self.response_generator = ResponseGenerator(config.get('use_llm', False))
        
        self.top_k = config.get('top_k', 5)
        self.similarity_threshold = config.get('similarity_threshold', 0.3)
        
        # Metrics
        self.query_count = 0
        self.query_history: List[Dict] = []
    
    def initialize(self, load_llm: bool = False):
        """Initialize components"""
        logger.info("Initializing RAG system...")
        self.embedding_model.load()
        self.vector_store = VectorStore(self.embedding_model.dimension)
        
        if load_llm:
            self.response_generator.load_llm()
    
    def index_products(self, df: pd.DataFrame, text_column: str = "rag_document"):
        """Index products"""
        logger.info(f"Indexing {len(df)} products...")
        
        documents = []
        texts = []
        
        for _, row in df.iterrows():
            doc = {
                'id': row.get('product_id', ''),
                'text': row[text_column],
                'product_name': row.get('product_name', ''),
                'brand': row.get('brand', ''),
                'category': row.get('main_category', ''),
                'price': row.get('discounted_price', ''),
                'actual_price': row.get('actual_price', ''),
                'rating': row.get('rating', ''),
                'discount': f"{row.get('discount_percentage', 0):.0f}%"
            }
            documents.append(doc)
            texts.append(row[text_column])
        
        embeddings = self.embedding_model.encode(texts, show_progress=True)
        self.vector_store.add_documents(documents, embeddings)
        logger.info(f"Indexed {len(documents)} products")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant documents"""
        if self.vector_store is None:
            return []
        
        k = top_k or self.top_k
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.vector_store.search(query_embedding, k)
        
        return [{'document': doc, 'score': score} for doc, score in results if score >= self.similarity_threshold]
    
    def answer(self, question: str, session_id: str = "default", include_sources: bool = True) -> Dict[str, Any]:
        """Answer question with conversation context"""
        start_time = datetime.now()
        self.query_count += 1
        
        # Get conversation context
        conv_context = self.conversation_manager.get_context(session_id)
        
        # Enhance query with context
        enhanced_query = question
        history = self.conversation_manager.get_history(session_id)
        if history:
            recent = [m['content'] for m in history[-2:] if m['role'] == 'user']
            if recent:
                enhanced_query = f"{recent[-1]} {question}"
        
        # Retrieve
        retrieved = self.retrieve(enhanced_query)
        
        if not retrieved:
            answer = "I couldn't find relevant products. Please try a different question."
            self.conversation_manager.add_message(session_id, "user", question)
            self.conversation_manager.add_message(session_id, "assistant", answer)
            
            return {
                'answer': answer,
                'confidence': 'low',
                'sources': [],
                'session_id': session_id,
                'conversation_turns': len(self.conversation_manager.get_history(session_id)) // 2,
                'retrieval_time': (datetime.now() - start_time).total_seconds()
            }
        
        # Build context
        context_parts = [f"Product {i}:\n{item['document']['text']}" for i, item in enumerate(retrieved[:5], 1)]
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self.response_generator.generate(question, context, conv_context)
        
        # Update conversation history
        self.conversation_manager.add_message(session_id, "user", question)
        self.conversation_manager.add_message(session_id, "assistant", answer)
        
        # Calculate confidence
        avg_score = np.mean([r['score'] for r in retrieved])
        confidence = 'high' if avg_score > 0.7 else 'medium' if avg_score > 0.5 else 'low'
        
        response = {
            'answer': answer,
            'confidence': confidence,
            'session_id': session_id,
            'conversation_turns': len(self.conversation_manager.get_history(session_id)) // 2,
            'retrieval_time': (datetime.now() - start_time).total_seconds(),
            'num_sources': len(retrieved)
        }
        
        if include_sources:
            response['sources'] = [{
                'product_id': r['document']['id'],
                'product_name': r['document']['product_name'],
                'price': r['document']['price'],
                'rating': r['document']['rating'],
                'score': round(r['score'], 4)
            } for r in retrieved]
        
        # Log query
        self.query_history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'session_id': session_id,
            'confidence': confidence
        })
        
        return response
    
    def get_conversation(self, session_id: str) -> Dict:
        """Get conversation history"""
        return {
            'session_id': session_id,
            'messages': self.conversation_manager.get_history(session_id),
            'turns': len(self.conversation_manager.get_history(session_id)) // 2
        }
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history"""
        self.conversation_manager.clear_session(session_id)
    
    def get_all_sessions(self) -> Dict:
        """Get all sessions"""
        return self.conversation_manager.get_all_sessions()
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'total_documents': len(self.vector_store.documents) if self.vector_store else 0,
            'embedding_dimension': self.embedding_model.dimension,
            'total_queries': self.query_count,
            'active_sessions': len(self.conversation_manager.sessions),
            'llm_enabled': self.response_generator.use_llm
        }
    
    def save(self, directory: str):
        """Save RAG system"""
        os.makedirs(directory, exist_ok=True)
        
        if self.vector_store:
            self.vector_store.save(os.path.join(directory, "vector_store.pkl"))
        
        self.conversation_manager.save(os.path.join(directory, "conversations.pkl"))
        
        with open(os.path.join(directory, "config.json"), 'w') as f:
            json.dump(self.get_stats(), f, indent=2)
        
        logger.info(f"RAG system saved to {directory}")
    
    def load(self, directory: str):
        """Load RAG system"""
        self.initialize()
        
        vs_path = os.path.join(directory, "vector_store.pkl")
        if os.path.exists(vs_path):
            self.vector_store.load(vs_path)
        
        conv_path = os.path.join(directory, "conversations.pkl")
        if os.path.exists(conv_path):
            self.conversation_manager.load(conv_path)
        
        logger.info(f"RAG system loaded from {directory}")
