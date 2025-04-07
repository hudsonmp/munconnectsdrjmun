#!/usr/bin/env python3
"""
MUN Conference Research Web Scraper

This script scrapes web sources for Model UN conference research based on topics in agenda_topics.json.
It processes and stores information from various sources including UN sites, news outlets, and academic sources.

Usage:
    python mun_scraper.py [options]

Author: Hudson Mitchell-Pullman
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse

import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from requests.exceptions import RequestException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mun_scraper.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mun_scraper")

# Constants
MIN_CONTENT_LENGTH = 500  # Minimum text length to consider a page relevant
MAX_PAGES_PER_SOURCE = 15  # Maximum pages to scrape per source
REQUEST_DELAY = 1.5  # Delay between requests in seconds
MAX_DEPTH = 3  # Maximum crawl depth from starting URL
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
]
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}

# Directory structure
DATA_DIR = "mun_data"
ARTICLES_DIR = os.path.join(DATA_DIR, "articles")
SUMMARIES_DIR = os.path.join(DATA_DIR, "summaries")
INDEX_DIR = os.path.join(DATA_DIR, "index")

# Content filtering keywords
CONTENT_FILTERS = {
    "issues": ["issue", "problem", "challenge", "concern", "crisis", "situation", "matter"],
    "countries": ["country", "nation", "state", "government", "region", "territory"],
    "solutions": ["solution", "resolution", "approach", "strategy", "policy", "initiative", "program", "action"],
    "history": ["history", "background", "context", "origin", "development", "evolution"],
    "humanitarian": ["humanitarian", "human rights", "aid", "assistance", "relief", "welfare"],
    "economic": ["economic", "economy", "financial", "trade", "market", "industry", "investment"],
    "un_interaction": ["united nations", "un", "resolution", "security council", "general assembly"],
    "ngo_involvement": ["ngo", "non-governmental organization", "organization", "agency", "foundation"]
}


class MUNScraper:
    """Main scraper class for the MUN conference research tool."""
    
    def __init__(self, topics_file: str, sources_file: str, output_dir: str = DATA_DIR):
        """
        Initialize the MUN scraper.
        
        Args:
            topics_file: Path to the agenda_topics.json file
            sources_file: Path to the sources.json file
            output_dir: Directory to store scraped data
        """
        self.topics_file = topics_file
        self.sources_file = sources_file
        self.output_dir = output_dir
        self.articles_dir = os.path.join(output_dir, "articles")
        self.summaries_dir = os.path.join(output_dir, "summaries")
        self.index_dir = os.path.join(output_dir, "index")
        
        # Create necessary directories
        os.makedirs(self.articles_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Download NLTK resources if not already available
        self._download_nltk_resources()
        
        self.topics = {}
        self.sources = {}
        self.visited_urls = set()
        self.scrape_count = 0
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.document_urls = []
        
        # Load topics and sources
        self._load_topics()
        self._load_sources()
        
        # Initialize stemmer and stopwords
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def _download_nltk_resources(self):
        """Download required NLTK resources."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def _load_topics(self):
        """Load topics from the topics file."""
        try:
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                self.topics = json.load(f)
            logger.info(f"Loaded {sum(len(topics) for topics in self.topics.values())} topics from {self.topics_file}")
        except Exception as e:
            logger.error(f"Failed to load topics from {self.topics_file}: {e}")
            sys.exit(1)
    
    def _load_sources(self):
        """Load sources from the sources file."""
        try:
            with open(self.sources_file, 'r', encoding='utf-8') as f:
                self.sources = json.load(f)
            logger.info(f"Loaded sources for {len(self.sources)} committees from {self.sources_file}")
        except Exception as e:
            logger.error(f"Failed to load sources from {self.sources_file}: {e}")
            sys.exit(1)
    
    def get_random_user_agent(self) -> str:
        """Return a random user agent string."""
        return random.choice(USER_AGENTS)
    
    def get_session(self) -> requests.Session:
        """Create and return a requests Session with appropriate headers."""
        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)
        session.headers.update({'User-Agent': self.get_random_user_agent()})
        return session
    
    def is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid and should be crawled."""
        parsed = urlparse(url)
        
        # Check for valid scheme
        if parsed.scheme not in ('http', 'https'):
            return False
        
        # Avoid certain file types
        ignored_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip', '.rar', '.jpg', '.jpeg', '.png', '.gif']
        if any(parsed.path.lower().endswith(ext) for ext in ignored_extensions):
            return False
        
        # Check for URL fragments
        if '#' in url:
            return False
        
        return True
    
    def fetch_url(self, url: str, depth: int = 0) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a URL.
        
        Args:
            url: URL to fetch
            depth: Current depth in the crawl
            
        Returns:
            BeautifulSoup object or None if fetch failed
        """
        if url in self.visited_urls or not self.is_valid_url(url) or depth > MAX_DEPTH:
            return None
        
        self.visited_urls.add(url)
        
        try:
            session = self.get_session()
            response = session.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                return None
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type.lower():
                logger.debug(f"Skipping non-HTML content: {url}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        
        except RequestException as e:
            logger.warning(f"Error fetching {url}: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict:
        """
        Extract content from a BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object to extract content from
            url: URL of the page
            
        Returns:
            Dictionary with extracted content
        """
        # Initialize content dictionary
        content = {
            "url": url,
            "title": "",
            "text": "",
            "date": "",
            "source_type": self.determine_source_type(url),
            "extracted_at": datetime.now().isoformat(),
            "metadata": {}
        }
        
        # Extract title
        if soup.title:
            content["title"] = soup.title.text.strip()
        
        # Extract main content
        main_content = []
        
        # Remove script, style, and other unwanted elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        # Extract paragraphs
        for p in soup.find_all(['p', 'article', 'section', 'div', 'main']):
            if p.text.strip():
                main_content.append(p.text.strip())
        
        content["text"] = "\n\n".join(main_content)
        
        # Extract date if available
        date_tags = soup.find_all(['time', 'meta'], attrs={'itemprop': 'datePublished'})
        if date_tags:
            content["date"] = date_tags[0].get('datetime', date_tags[0].text.strip())
        
        # Extract additional metadata
        meta_tags = soup.find_all('meta')
        metadata = {}
        for tag in meta_tags:
            name = tag.get('name', tag.get('property', ''))
            if name and name in ['description', 'keywords', 'author', 'og:type', 'og:site_name']:
                metadata[name] = tag.get('content', '')
        
        content["metadata"] = metadata
        
        return content
    
    def determine_source_type(self, url: str) -> str:
        """Determine the type of source based on the URL."""
        hostname = urlparse(url).netloc.lower()
        
        # UN organizations
        if '.un.org' in hostname or 'unhcr.org' in hostname or 'who.int' in hostname or 'unesco.org' in hostname:
            return "UN_organization"
        
        # News outlets
        if any(domain in hostname for domain in ['bbc.', 'reuters.', 'aljazeera.', 'guardian.', 'nytimes.']):
            return "news_outlet"
        
        # NGOs
        if any(domain in hostname for domain in ['hrw.org', 'amnesty.org', 'rescue.org', 'oxfam.org']):
            return "NGO"
        
        # Academic sources
        if any(domain in hostname for domain in ['scholar.google.', 'jstor.org', 'researchgate.net']):
            return "academia"
        
        # Default
        return "other"
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract links from a BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object to extract links from
            base_url: Base URL to resolve relative links
            
        Returns:
            List of absolute URLs
        """
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            
            # Skip empty links, anchors, and javascript links
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
            
            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)
            
            # Check if URL is valid
            if self.is_valid_url(absolute_url):
                links.append(absolute_url)
        
        return links 
    
    def calculate_relevance_score(self, content: Dict, topic: str, committee: str) -> float:
        """
        Calculate relevance score of content to a specific topic.
        
        Args:
            content: Content dictionary
            topic: Topic to check relevance against
            committee: Committee the topic belongs to
            
        Returns:
            Relevance score between 0 and 1
        """
        text = f"{content['title']} {content['text']}"
        text = text.lower()
        
        # Check if topic keywords are in the content
        topic_keywords = self.get_topic_keywords(topic)
        committee_keywords = committee.lower().split()
        
        # Count occurrences of topic keywords
        topic_matches = sum(1 for keyword in topic_keywords if keyword.lower() in text)
        committee_matches = sum(1 for keyword in committee_keywords if keyword.lower() in text)
        
        # Check for content filter matches
        filter_matches = 0
        for category, keywords in CONTENT_FILTERS.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    filter_matches += 1
                    break  # Only count one match per category
        
        # Calculate score components
        topic_score = min(topic_matches / max(1, len(topic_keywords)), 1.0) * 0.6
        committee_score = min(committee_matches / max(1, len(committee_keywords)), 1.0) * 0.2
        filter_score = min(filter_matches / len(CONTENT_FILTERS), 1.0) * 0.2
        
        # Combine scores
        total_score = topic_score + committee_score + filter_score
        
        return total_score
    
    def get_topic_keywords(self, topic: str) -> List[str]:
        """Extract keywords from a topic string."""
        # Tokenize and remove stopwords
        tokens = word_tokenize(topic.lower())
        keywords = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        
        # Add the original multi-word phrases
        phrases = [phrase for phrase in re.findall(r'\b\w+\s+\w+\b', topic.lower())]
        keywords.extend(phrases)
        
        return keywords
    
    def is_content_relevant(self, content: Dict, topic: str, committee: str) -> Tuple[bool, float]:
        """
        Determine if content is relevant to a topic.
        
        Args:
            content: Content dictionary
            topic: Topic to check relevance against
            committee: Committee the topic belongs to
            
        Returns:
            Tuple of (is_relevant, relevance_score)
        """
        # Check if content is too short
        if len(content['text']) < MIN_CONTENT_LENGTH:
            return False, 0
        
        relevance_score = self.calculate_relevance_score(content, topic, committee)
        return relevance_score >= 0.3, relevance_score
    
    def summarize_content(self, content: Dict) -> str:
        """
        Generate a summary of the content.
        
        Args:
            content: Content dictionary
            
        Returns:
            Summary text
        """
        text = content['text']
        
        # Simple extractive summarization by selecting key sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # If text is short, return as is
        if len(sentences) <= 5:
            return text
        
        # Calculate sentence scores based on position and keyword frequency
        word_freq = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word.isalnum() and word not in self.stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Normalize word frequency
        max_freq = max(word_freq.values()) if word_freq else 1
        word_freq = {word: freq/max_freq for word, freq in word_freq.items()}
        
        # Score sentences
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            # Position score (first and last sentences are important)
            if i < 3 or i >= len(sentences) - 3:
                score += 0.3
            
            # Word frequency score
            for word in word_tokenize(sentence.lower()):
                if word.isalnum() and word not in self.stop_words:
                    score += word_freq.get(word, 0)
            
            sentence_scores.append((i, score))
        
        # Select top sentences (about 20% of the original content)
        num_summary_sentences = max(3, int(len(sentences) * 0.2))
        top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_summary_sentences]
        top_sentences = sorted(top_sentences, key=lambda x: x[0])  # Restore original order
        
        summary_sentences = [sentences[i] for i, _ in top_sentences]
        summary = ' '.join(summary_sentences)
        
        return summary
    
    def post_process_article(self, article: Dict) -> Dict:
        """
        Apply post-processing to an article.
        
        Args:
            article: Article dictionary
            
        Returns:
            Processed article dictionary
        """
        # Generate summary
        article['summary'] = self.summarize_content(article)
        
        # Extract entities (simplified version)
        text = article['text'].lower()
        entities = {
            'countries': [],
            'organizations': [],
            'key_issues': []
        }
        
        # Extract countries (very simplified - would use NER in production)
        common_countries = ['afghanistan', 'australia', 'brazil', 'canada', 'china', 'france', 
                            'germany', 'india', 'iran', 'iraq', 'israel', 'japan', 'mexico', 
                            'north korea', 'pakistan', 'russia', 'south korea', 'syria', 
                            'united kingdom', 'united states', 'yemen']
        
        for country in common_countries:
            if country in text:
                entities['countries'].append(country)
        
        # Extract organizations
        org_patterns = ['united nations', 'un', 'who', 'unesco', 'unicef', 'world bank', 
                        'amnesty international', 'human rights watch', 'red cross', 
                        'doctors without borders', 'oxfam']
        
        for org in org_patterns:
            if org in text:
                entities['organizations'].append(org)
        
        # Extract key issues from content filters
        for category, keywords in CONTENT_FILTERS.items():
            for keyword in keywords:
                if keyword in text and keyword not in entities['key_issues']:
                    entities['key_issues'].append(keyword)
        
        article['entities'] = entities
        
        return article
    
    def save_article(self, article: Dict, committee: str, topic: str) -> str:
        """
        Save an article to disk.
        
        Args:
            article: Article dictionary
            committee: Committee the article is relevant to
            topic: Topic the article is relevant to
            
        Returns:
            Path to the saved article
        """
        # Create committee and topic directories
        committee_dir = os.path.join(self.articles_dir, self._sanitize_filename(committee))
        topic_dir = os.path.join(committee_dir, self._sanitize_filename(topic))
        os.makedirs(topic_dir, exist_ok=True)
        
        # Create a unique filename based on URL
        url_hash = hashlib.md5(article['url'].encode('utf-8')).hexdigest()
        filename = f"{url_hash}.json"
        filepath = os.path.join(topic_dir, filename)
        
        # Save article metadata
        article['committee'] = committee
        article['topic'] = topic
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False)
            
        return filepath
    
    def _sanitize_filename(self, name: str) -> str:
        """Convert a string to a valid filename."""
        return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
    
    def crawl_url(self, url: str, committee: str, topic: str, depth: int = 0) -> List[str]:
        """
        Crawl a URL and its links up to the specified depth.
        
        Args:
            url: URL to crawl
            committee: Committee the URL is relevant to
            topic: Topic the URL is relevant to
            depth: Current depth in the crawl
            
        Returns:
            List of URLs that were processed
        """
        # Check if we've reached the max crawl depth or max pages per source
        if depth > MAX_DEPTH or self.scrape_count >= MAX_PAGES_PER_SOURCE:
            return []
        
        # Fetch and parse the URL
        soup = self.fetch_url(url, depth)
        if not soup:
            return []
        
        # Random delay between requests
        time.sleep(REQUEST_DELAY + random.uniform(0, 0.5))
        
        # Extract content
        content = self.extract_content(soup, url)
        
        # Check if content is relevant
        is_relevant, relevance_score = self.is_content_relevant(content, topic, committee)
        
        processed_urls = [url]
        
        if is_relevant:
            # Post-process and save the article
            content['relevance_score'] = relevance_score
            article = self.post_process_article(content)
            self.save_article(article, committee, topic)
            self.scrape_count += 1
            
            logger.info(f"Saved article: {content['title']} - Relevance: {relevance_score:.2f}")
        
        # Extract and follow links if we haven't reached the max depth
        if depth < MAX_DEPTH and self.scrape_count < MAX_PAGES_PER_SOURCE:
            links = self.extract_links(soup, url)
            
            # Limit the number of links to follow
            links = links[:5]  # Only follow a few links per page
            
            for link in links:
                if link not in self.visited_urls:
                    child_urls = self.crawl_url(link, committee, topic, depth + 1)
                    processed_urls.extend(child_urls)
                    
                    # Check if we've reached the maximum number of pages
                    if self.scrape_count >= MAX_PAGES_PER_SOURCE:
                        break
        
        return processed_urls
    
    def crawl_source(self, source_url: str, committee: str, topic: str) -> List[str]:
        """
        Crawl a specific source for a topic.
        
        Args:
            source_url: Source URL to crawl
            committee: Committee the source is relevant to
            topic: Topic the source is relevant to
            
        Returns:
            List of URLs that were processed
        """
        logger.info(f"Crawling source: {source_url} for {committee} - {topic}")
        
        # Reset the scrape count for this source
        self.scrape_count = 0
        
        # Crawl the source
        processed_urls = self.crawl_url(source_url, committee, topic)
        
        logger.info(f"Finished crawling {source_url}. Processed {len(processed_urls)} URLs.")
        return processed_urls
    
    def create_csv_summaries(self):
        """Create CSV summaries of the scraped articles by committee and topic."""
        logger.info("Creating CSV summaries...")
        
        # Create committee-level summary
        committee_data = []
        
        for committee_dir in os.listdir(self.articles_dir):
            committee_path = os.path.join(self.articles_dir, committee_dir)
            
            if not os.path.isdir(committee_path):
                continue
            
            # Get topic-level summary
            for topic_dir in os.listdir(committee_path):
                topic_path = os.path.join(committee_path, topic_dir)
                
                if not os.path.isdir(topic_path):
                    continue
                
                topic_articles = []
                
                # Process articles in this topic
                for article_file in os.listdir(topic_path):
                    if not article_file.endswith('.json'):
                        continue
                    
                    article_path = os.path.join(topic_path, article_file)
                    
                    with open(article_path, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                    
                    # Add to topic articles
                    topic_articles.append({
                        'title': article.get('title', 'No title'),
                        'url': article.get('url', ''),
                        'source_type': article.get('source_type', 'other'),
                        'relevance_score': article.get('relevance_score', 0),
                        'date': article.get('date', ''),
                        'summary': article.get('summary', '')
                    })
                    
                    # Add to committee data
                    committee_data.append({
                        'committee': committee_dir.replace('_', ' '),
                        'topic': topic_dir.replace('_', ' '),
                        'title': article.get('title', 'No title'),
                        'url': article.get('url', ''),
                        'source_type': article.get('source_type', 'other'),
                        'relevance_score': article.get('relevance_score', 0)
                    })
                
                # Create topic CSV
                if topic_articles:
                    topic_df = pd.DataFrame(topic_articles)
                    topic_csv_path = os.path.join(self.summaries_dir, f"{committee_dir}_{topic_dir}.csv")
                    topic_df.to_csv(topic_csv_path, index=False)
                    logger.info(f"Created topic summary: {topic_csv_path}")
        
        # Create committee CSV
        if committee_data:
            committee_df = pd.DataFrame(committee_data)
            committee_csv_path = os.path.join(self.summaries_dir, "all_committees.csv")
            committee_df.to_csv(committee_csv_path, index=False)
            logger.info(f"Created committee summary: {committee_csv_path}")
    
    def build_search_index(self):
        """Build a TF-IDF search index for the scraped articles."""
        logger.info("Building search index...")
        
        documents = []
        document_paths = []
        
        # Collect all articles
        for root, _, files in os.walk(self.articles_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                    
                    # Combine title and text for indexing
                    doc_text = f"{article.get('title', '')} {article.get('text', '')}"
                    documents.append(doc_text)
                    document_paths.append(file_path)
        
        if not documents:
            logger.warning("No documents found for indexing.")
            return
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        
        # Fit and transform documents
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        # Save the document paths
        self.document_urls = document_paths
        
        # Save the index
        index_data = {
            'vectorizer': self.tfidf_vectorizer,
            'matrix': self.tfidf_matrix,
            'document_paths': document_paths
        }
        
        index_path = os.path.join(self.index_dir, "search_index.pkl")
        
        import pickle
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"Search index built and saved to {index_path}")
        
        # Create a feature names file for reference
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        with open(os.path.join(self.index_dir, "feature_names.txt"), 'w', encoding='utf-8') as f:
            f.write("\n".join(feature_names))
    
    def search(self, query: str, top_n: int = 10) -> List[Dict]:
        """
        Search the indexed articles.
        
        Args:
            query: Search query
            top_n: Number of top results to return
            
        Returns:
            List of search results
        """
        if not self.tfidf_vectorizer or not self.tfidf_matrix:
            logger.error("Search index not built. Call build_search_index() first.")
            return []
        
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top N results
        top_indices = similarity_scores.argsort()[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            if similarity_scores[idx] > 0:
                document_path = self.document_urls[idx]
                
                with open(document_path, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                
                results.append({
                    'score': float(similarity_scores[idx]),
                    'title': article.get('title', 'No title'),
                    'url': article.get('url', ''),
                    'committee': article.get('committee', ''),
                    'topic': article.get('topic', ''),
                    'summary': article.get('summary', ''),
                    'source_type': article.get('source_type', 'other')
                })
        
        return results 

    def run(self, committees: Optional[List[str]] = None, topics: Optional[List[str]] = None):
        """
        Run the scraper for specified committees and topics.
        
        Args:
            committees: List of committees to scrape, or None for all
            topics: List of topics to scrape, or None for all
        """
        start_time = time.time()
        logger.info("Starting MUN scraper")
        
        # Process all committees or only specified ones
        processed_committees = []
        
        for committee, committee_topics in self.topics.items():
            # Skip if we're only processing specific committees and this one isn't in the list
            if committees and committee not in committees:
                continue
            
            logger.info(f"Processing committee: {committee}")
            
            # Find sources for this committee
            committee_sources = None
            for src_committee, src_data in self.sources.items():
                if src_committee == committee:
                    committee_sources = src_data
                    break
            
            if not committee_sources:
                logger.warning(f"No sources found for committee: {committee}")
                continue
            
            # Process each topic in this committee
            for topic_idx, topic in enumerate(committee_topics):
                # Skip if we're only processing specific topics and this one isn't in the list
                if topics and topic not in topics:
                    continue
                
                logger.info(f"Processing topic {topic_idx+1}/{len(committee_topics)}: {topic}")
                
                # Find source data for this topic
                topic_sources = None
                for src_topic_data in committee_sources:
                    if src_topic_data.get('topic') == topic:
                        topic_sources = src_topic_data.get('sources', {})
                        break
                
                if not topic_sources:
                    logger.warning(f"No sources found for topic: {topic}")
                    continue
                
                # Process each source category for this topic
                for category, sources in topic_sources.items():
                    logger.info(f"Processing source category: {category}")
                    
                    for source in sources:
                        source_url = source.get('url')
                        if not source_url:
                            continue
                        
                        # Crawl this source
                        self.crawl_source(source_url, committee, topic)
            
            processed_committees.append(committee)
        
        # Create summaries and build search index
        self.create_csv_summaries()
        self.build_search_index()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Scraping completed in {duration:.2f} seconds")
        logger.info(f"Processed {len(processed_committees)} committees")
        logger.info(f"Results stored in {self.output_dir}")


def generate_react_search_app(output_dir: str):
    """
    Generate a simple React app with shadcn/ui for searching the scraped content.
    
    Args:
        output_dir: Directory to store the React app
    """
    app_dir = os.path.join(output_dir, "mun_search_app")
    os.makedirs(app_dir, exist_ok=True)
    
    logger.info(f"Generating React search app in {app_dir}")
    
    # Create README with setup instructions
    readme_content = """# MUN Research Search App

This is a React application for searching MUN research content.

## Setup Instructions

1. Make sure you have Node.js and npm installed
2. Navigate to this directory in your terminal
3. Run the following commands:

```bash
# Install dependencies
npm install

# Start the development server
npm run dev
```

4. Open http://localhost:3000 in your browser

## Features

- Topic-specific and keyword search
- Filtering by committee, topic, and source type
- Content preview and summary display
- Link to original sources
- Responsive design with shadcn/ui
"""
    
    with open(os.path.join(app_dir, "README.md"), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # Create package.json with dependencies
    package_json = {
        "name": "mun-search-app",
        "version": "0.1.0",
        "private": True,
        "scripts": {
            "dev": "next dev",
            "build": "next build",
            "start": "next start",
            "lint": "next lint"
        },
        "dependencies": {
            "next": "^14.0.0",
            "react": "^18.2.0",
            "react-dom": "^18.2.0",
            "@radix-ui/react-select": "^2.0.0",
            "@radix-ui/react-slot": "^1.0.2",
            "class-variance-authority": "^0.7.0",
            "clsx": "^2.0.0",
            "lucide-react": "^0.294.0",
            "tailwind-merge": "^2.1.0",
            "tailwindcss-animate": "^1.0.7"
        },
        "devDependencies": {
            "@types/node": "^20.8.10",
            "@types/react": "^18.2.35",
            "@types/react-dom": "^18.2.14",
            "autoprefixer": "^10.4.16",
            "eslint": "^8.53.0",
            "eslint-config-next": "^14.0.0",
            "postcss": "^8.4.31",
            "tailwindcss": "^3.3.5",
            "typescript": "^5.2.2"
        }
    }
    
    with open(os.path.join(app_dir, "package.json"), 'w', encoding='utf-8') as f:
        json.dump(package_json, f, indent=2)
    
    # Instructions for creating a Flask API to serve the search results
    api_instructions = """# Setting up the Search API

To connect the React app to your search index, create a simple Flask API:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load the search index
with open('../mun_data/index/search_index.pkl', 'rb') as f:
    index_data = pickle.load(f)

vectorizer = index_data['vectorizer']
matrix = index_data['matrix']
document_paths = index_data['document_paths']

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    committee = request.args.get('committee', '')
    topic = request.args.get('topic', '')
    
    if not query:
        return jsonify({'results': []})
    
    # Transform query
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(query_vector, matrix)[0]
    
    # Get top results
    top_n = 20
    top_indices = similarity_scores.argsort()[::-1][:top_n*2]  # Get more than needed for filtering
    
    results = []
    for idx in top_indices:
        if similarity_scores[idx] > 0:
            document_path = document_paths[idx]
            
            with open(document_path, 'r', encoding='utf-8') as f:
                article = json.load(f)
            
            # Apply filters if specified
            if committee and article.get('committee', '') != committee:
                continue
                
            if topic and article.get('topic', '') != topic:
                continue
            
            results.append({
                'score': float(similarity_scores[idx]),
                'title': article.get('title', 'No title'),
                'url': article.get('url', ''),
                'committee': article.get('committee', ''),
                'topic': article.get('topic', ''),
                'summary': article.get('summary', ''),
                'source_type': article.get('source_type', 'other')
            })
            
            # Stop once we have enough filtered results
            if len(results) >= top_n:
                break
    
    return jsonify({'results': results})

@app.route('/metadata', methods=['GET'])
def metadata():
    # Get available committees and topics
    committees = set()
    topics = {}
    
    for path in document_paths:
        with open(path, 'r', encoding='utf-8') as f:
            article = json.load(f)
            
        committee = article.get('committee', '')
        topic = article.get('topic', '')
        
        if committee:
            committees.add(committee)
            if committee not in topics:
                topics[committee] = set()
            if topic:
                topics[committee].add(topic)
    
    return jsonify({
        'committees': list(committees),
        'topics': {c: list(t) for c, t in topics.items()}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

Save this as `api.py` and install dependencies:

```bash
pip install flask flask-cors
python api.py
```
"""
    
    with open(os.path.join(app_dir, "api_instructions.md"), 'w', encoding='utf-8') as f:
        f.write(api_instructions)
    
    logger.info("React search app generated successfully")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MUN Conference Research Web Scraper")
    
    parser.add_argument(
        "--topics-file",
        default="agenda_topics.json",
        help="Path to the agenda topics JSON file"
    )
    
    parser.add_argument(
        "--sources-file",
        default="sources.json",
        help="Path to the sources JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        default=DATA_DIR,
        help="Directory to store scraped data"
    )
    
    parser.add_argument(
        "--committees",
        nargs="+",
        help="Specific committees to scrape (default: all committees)"
    )
    
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Specific topics to scrape (default: all topics)"
    )
    
    parser.add_argument(
        "--max-pages",
        type=int,
        default=MAX_PAGES_PER_SOURCE,
        help=f"Maximum pages to scrape per source (default: {MAX_PAGES_PER_SOURCE})"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help=f"Delay between requests in seconds (default: {REQUEST_DELAY})"
    )
    
    parser.add_argument(
        "--depth",
        type=int,
        default=MAX_DEPTH,
        help=f"Maximum crawl depth (default: {MAX_DEPTH})"
    )
    
    parser.add_argument(
        "--min-content",
        type=int,
        default=MIN_CONTENT_LENGTH,
        help=f"Minimum content length to consider (default: {MIN_CONTENT_LENGTH})"
    )
    
    parser.add_argument(
        "--generate-ui",
        action="store_true",
        help="Generate React search UI"
    )
    
    parser.add_argument(
        "--search",
        type=str,
        help="Search the scraped content (requires completed scrape)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the scraper."""
    args = parse_arguments()
    
    # Update global constants
    global MAX_PAGES_PER_SOURCE, REQUEST_DELAY, MAX_DEPTH, MIN_CONTENT_LENGTH
    MAX_PAGES_PER_SOURCE = args.max_pages
    REQUEST_DELAY = args.delay
    MAX_DEPTH = args.depth
    MIN_CONTENT_LENGTH = args.min_content
    
    # Create the scraper
    scraper = MUNScraper(
        topics_file=args.topics_file,
        sources_file=args.sources_file,
        output_dir=args.output_dir
    )
    
    # Search mode
    if args.search:
        try:
            # Try to load existing search index
            import pickle
            index_path = os.path.join(scraper.index_dir, "search_index.pkl")
            
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            scraper.tfidf_vectorizer = index_data['vectorizer']
            scraper.tfidf_matrix = index_data['matrix']
            scraper.document_urls = index_data['document_paths']
            
            results = scraper.search(args.search, top_n=10)
            
            print(f"\nSearch results for '{args.search}':\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (Score: {result['score']:.2f})")
                print(f"   Committee: {result['committee']}, Topic: {result['topic']}")
                print(f"   Source Type: {result['source_type']}")
                print(f"   URL: {result['url']}")
                print(f"   Summary: {result['summary'][:150]}...\n")
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            print("Could not search. Make sure you've run the scraper first.")
    else:
        # Run the scraper
        scraper.run(committees=args.committees, topics=args.topics)
        
        # Generate UI if requested
        if args.generate_ui:
            generate_react_search_app(args.output_dir)


if __name__ == "__main__":
    main() 