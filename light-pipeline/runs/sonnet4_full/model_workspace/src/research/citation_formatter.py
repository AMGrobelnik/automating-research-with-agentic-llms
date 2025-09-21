"""APA-style citation formatter with URL validation and metadata extraction."""

import re
import requests
from urllib.parse import urlparse, urljoin
from datetime import datetime, date
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
from loguru import logger
import asyncio
import aiohttp

from ..schemas.citation_schemas import (
    CitationSource, FormattedCitation, CitationType, SearchResult
)
from ..utils.validators import CitationValidator
from ..config.config_manager import get_config_manager

# Define color constants
BLUE, GREEN, YELLOW, CYAN, RED, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[0m"


@dataclass
class ExtractedMetadata:
    """Metadata extracted from web page."""
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    publication_name: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class MetadataExtractor:
    """Extract metadata from web pages for citation purposes."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    async def extract_metadata(self, url: str, timeout: int = 30) -> ExtractedMetadata:
        """
        Extract metadata from a web page.
        
        Args:
            url: URL to extract metadata from
            timeout: Request timeout in seconds
            
        Returns:
            ExtractedMetadata object with extracted information
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    if response.status != 200:
                        logger.warning(f"{YELLOW}Failed to fetch {url}: HTTP {response.status}{END}")
                        return ExtractedMetadata()
                    
                    content_type = response.headers.get('content-type', '').lower()
                    if 'html' not in content_type:
                        logger.debug(f"{CYAN}Non-HTML content type for {url}: {content_type}{END}")
                        return ExtractedMetadata()
                    
                    html_content = await response.text()
                    return self._parse_html_metadata(html_content, url)
        
        except Exception as e:
            logger.warning(f"{YELLOW}Error extracting metadata from {url}: {e}{END}")
            return ExtractedMetadata()
    
    def _parse_html_metadata(self, html_content: str, url: str) -> ExtractedMetadata:
        """Parse HTML content to extract metadata."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            metadata = ExtractedMetadata()
            
            # Extract title
            metadata.title = self._extract_title(soup)
            
            # Extract author
            metadata.author = self._extract_author(soup)
            
            # Extract publication date
            metadata.publication_date = self._extract_publication_date(soup)
            
            # Extract publication name
            metadata.publication_name = self._extract_publication_name(soup, url)
            
            # Extract description
            metadata.description = self._extract_description(soup)
            
            # Extract keywords
            metadata.keywords = self._extract_keywords(soup)
            
            return metadata
        
        except Exception as e:
            logger.warning(f"{YELLOW}Error parsing HTML metadata: {e}{END}")
            return ExtractedMetadata()
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title from various sources."""
        # Try Open Graph title
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()
        
        # Try Twitter title
        twitter_title = soup.find('meta', {'name': 'twitter:title'})
        if twitter_title and twitter_title.get('content'):
            return twitter_title['content'].strip()
        
        # Try regular title tag
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Try h1 tag
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return None
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author information."""
        # Try meta author tag
        author_meta = soup.find('meta', {'name': 'author'})
        if author_meta and author_meta.get('content'):
            return author_meta['content'].strip()
        
        # Try JSON-LD structured data
        json_ld = soup.find('script', {'type': 'application/ld+json'})
        if json_ld:
            try:
                import json
                data = json.loads(json_ld.get_text())
                if isinstance(data, dict):
                    author = data.get('author')
                    if author:
                        if isinstance(author, dict):
                            return author.get('name', '').strip()
                        elif isinstance(author, str):
                            return author.strip()
            except:
                pass
        
        # Try schema.org microdata
        author_elem = soup.find(attrs={'itemprop': 'author'})
        if author_elem:
            return author_elem.get_text().strip()
        
        # Try common author class names
        author_selectors = [
            '.author', '.byline', '.author-name', 
            '[class*="author"]', '[class*="byline"]'
        ]
        
        for selector in author_selectors:
            author_elem = soup.select_one(selector)
            if author_elem:
                text = author_elem.get_text().strip()
                # Clean up "By Author Name" format
                text = re.sub(r'^by\s+', '', text, flags=re.IGNORECASE)
                if text and len(text) < 100:  # Reasonable author name length
                    return text
        
        return None
    
    def _extract_publication_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract publication date."""
        # Try meta tags
        date_selectors = [
            ('meta', {'name': 'article:published_time'}),
            ('meta', {'name': 'published_time'}),
            ('meta', {'name': 'publication_date'}),
            ('meta', {'name': 'date'}),
            ('meta', {'property': 'article:published_time'}),
        ]
        
        for tag, attrs in date_selectors:
            elem = soup.find(tag, attrs)
            if elem and elem.get('content'):
                date_str = elem['content']
                parsed_date = self._parse_date_string(date_str)
                if parsed_date:
                    return parsed_date
        
        # Try schema.org microdata
        date_elem = soup.find(attrs={'itemprop': 'datePublished'})
        if date_elem:
            date_str = date_elem.get('datetime') or date_elem.get_text()
            parsed_date = self._parse_date_string(date_str)
            if parsed_date:
                return parsed_date
        
        # Try time tags
        time_tags = soup.find_all('time')
        for time_tag in time_tags:
            date_str = time_tag.get('datetime') or time_tag.get_text()
            parsed_date = self._parse_date_string(date_str)
            if parsed_date:
                return parsed_date
        
        return None
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats."""
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        # Common date patterns
        patterns = [
            r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})',  # ISO 8601
            r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
            r'(\d{2})/(\d{2})/(\d{4})',  # MM/DD/YYYY
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # M/D/YYYY
            r'(\d{4})/(\d{2})/(\d{2})',  # YYYY/MM/DD
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) >= 3:
                        if len(groups) >= 6:  # Has time
                            return datetime(int(groups[0]), int(groups[1]), int(groups[2]),
                                          int(groups[3]), int(groups[4]), int(groups[5]))
                        else:  # Date only
                            if '/' in date_str and groups[2].startswith('20'):  # MM/DD/YYYY format
                                return datetime(int(groups[2]), int(groups[0]), int(groups[1]))
                            else:  # YYYY-MM-DD format
                                return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                except ValueError:
                    continue
        
        return None
    
    def _extract_publication_name(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract publication/website name."""
        # Try Open Graph site name
        og_site = soup.find('meta', property='og:site_name')
        if og_site and og_site.get('content'):
            return og_site['content'].strip()
        
        # Try meta application-name
        app_name = soup.find('meta', {'name': 'application-name'})
        if app_name and app_name.get('content'):
            return app_name['content'].strip()
        
        # Extract from domain name
        try:
            domain = urlparse(url).netloc
            domain = re.sub(r'^www\.', '', domain)
            # Capitalize domain name as fallback
            return domain.split('.')[0].capitalize()
        except:
            return None
    
    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page description."""
        # Try meta description
        desc_meta = soup.find('meta', {'name': 'description'})
        if desc_meta and desc_meta.get('content'):
            return desc_meta['content'].strip()
        
        # Try Open Graph description
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            return og_desc['content'].strip()
        
        return None
    
    def _extract_keywords(self, soup: BeautifulSoup) -> List[str]:
        """Extract keywords from the page."""
        keywords = []
        
        # Try meta keywords
        keywords_meta = soup.find('meta', {'name': 'keywords'})
        if keywords_meta and keywords_meta.get('content'):
            keywords.extend([k.strip() for k in keywords_meta['content'].split(',')])
        
        return [k for k in keywords if k]


class CitationFormatter:
    """APA-style citation formatter with URL validation and metadata extraction."""
    
    def __init__(self):
        self.metadata_extractor = MetadataExtractor()
        self.config_manager = get_config_manager()
        self.citation_validator = CitationValidator()
    
    async def create_citation_from_search_result(self, search_result: SearchResult) -> FormattedCitation:
        """
        Create a formatted citation from a search result.
        
        Args:
            search_result: SearchResult to create citation from
            
        Returns:
            FormattedCitation with APA-style formatting
        """
        # Extract metadata from URL
        metadata = await self.metadata_extractor.extract_metadata(str(search_result.url))
        
        # Create citation source
        citation_source = CitationSource(
            url=search_result.url,
            title=metadata.title or search_result.title or "Untitled",
            author=metadata.author,
            publication_date=metadata.publication_date,
            publication_name=metadata.publication_name,
            citation_type=self._determine_citation_type(str(search_result.url), metadata),
            access_date=datetime.now()
        )
        
        # Format citation in APA style
        formatted_text = self._format_apa_citation(citation_source, search_result.snippet)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(citation_source, metadata)
        
        return FormattedCitation(
            formatted_text=formatted_text,
            source=citation_source,
            confidence_score=confidence_score
        )
    
    async def create_citation_from_url(self, url: str, custom_title: Optional[str] = None) -> FormattedCitation:
        """
        Create a formatted citation from a URL.
        
        Args:
            url: URL to create citation from
            custom_title: Optional custom title to use instead of extracted title
            
        Returns:
            FormattedCitation with APA-style formatting
        """
        # Validate URL first
        try:
            self.citation_validator.validate_url(url)
        except Exception as e:
            logger.warning(f"{YELLOW}URL validation failed: {e}{END}")
        
        # Extract metadata
        metadata = await self.metadata_extractor.extract_metadata(url)
        
        # Create citation source
        citation_source = CitationSource(
            url=url,
            title=custom_title or metadata.title or "Untitled",
            author=metadata.author,
            publication_date=metadata.publication_date,
            publication_name=metadata.publication_name,
            citation_type=self._determine_citation_type(url, metadata),
            access_date=datetime.now()
        )
        
        # Format citation
        formatted_text = self._format_apa_citation(citation_source)
        
        # Calculate confidence
        confidence_score = self._calculate_confidence_score(citation_source, metadata)
        
        return FormattedCitation(
            formatted_text=formatted_text,
            source=citation_source,
            confidence_score=confidence_score
        )
    
    def _determine_citation_type(self, url: str, metadata: ExtractedMetadata) -> CitationType:
        """Determine the type of citation based on URL and metadata."""
        url = url.lower()
        
        # Academic/journal sites
        if any(domain in url for domain in ['doi.org', 'pubmed.gov', 'ncbi.nlm.nih.gov', 'scholar.google']):
            return CitationType.ACADEMIC_PAPER
        
        # News sites
        if any(domain in url for domain in ['cnn.com', 'bbc.com', 'reuters.com', 'ap.org', 'nytimes.com', 'washingtonpost.com']):
            return CitationType.NEWS_ARTICLE
        
        # Government sites
        if '.gov' in url:
            return CitationType.GOVERNMENT_REPORT
        
        # Check for keywords in title or content
        if metadata.title:
            title_lower = metadata.title.lower()
            if any(word in title_lower for word in ['journal', 'research', 'study', 'paper']):
                return CitationType.JOURNAL_ARTICLE
        
        # Default to website
        return CitationType.WEBSITE
    
    def _format_apa_citation(self, source: CitationSource, snippet: Optional[str] = None) -> str:
        """
        Format citation in APA style.
        
        APA Website Citation Format:
        Author, A. A. (Year, Month Day). Title of webpage. Website Name. URL
        """
        parts = []
        
        # Author
        if source.author:
            # Format author name (assume "First Last" format)
            author_formatted = self._format_author_apa(source.author)
            parts.append(author_formatted)
        else:
            parts.append("[No author]")
        
        # Date
        if source.publication_date:
            date_str = self._format_date_apa(source.publication_date)
            parts.append(f"({date_str})")
        else:
            parts.append("(n.d.)")
        
        # Title
        title = source.title
        if not title.endswith('.'):
            title += '.'
        parts.append(f"*{title}*")
        
        # Publication name
        if source.publication_name:
            parts.append(f"{source.publication_name}.")
        
        # URL and access date
        access_date_str = self._format_date_apa(source.access_date) if source.access_date else datetime.now().strftime("%B %d, %Y")
        parts.append(f"Retrieved {access_date_str}, from {source.url}")
        
        citation = " ".join(parts)
        
        # Clean up extra spaces
        citation = re.sub(r'\s+', ' ', citation)
        
        return citation
    
    def _format_author_apa(self, author: str) -> str:
        """Format author name in APA style."""
        # Handle multiple authors
        if ',' in author or ' and ' in author.lower():
            # For now, just take the first author
            first_author = author.split(',')[0].split(' and ')[0].strip()
            author = first_author
        
        # Remove common prefixes/suffixes
        author = re.sub(r'^(by|author:?)\s+', '', author, flags=re.IGNORECASE)
        author = author.strip()
        
        # Basic format: "Last, F. M."
        parts = author.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            first_names = parts[:-1]
            
            # Create initials
            initials = []
            for name in first_names:
                if name and name[0].isalpha():
                    initials.append(f"{name[0].upper()}.")
            
            if initials:
                return f"{last_name}, {' '.join(initials)}"
        
        # If parsing fails, return as is
        return f"{author}."
    
    def _format_date_apa(self, date_obj: datetime) -> str:
        """Format date in APA style."""
        if isinstance(date_obj, datetime):
            return date_obj.strftime("%B %d, %Y")
        elif isinstance(date_obj, date):
            return date_obj.strftime("%B %d, %Y")
        else:
            return str(date_obj)
    
    def _calculate_confidence_score(self, source: CitationSource, metadata: ExtractedMetadata) -> float:
        """Calculate confidence score for the citation."""
        score = 0.0
        
        # Title present and reasonable length
        if source.title and len(source.title) > 5:
            score += 0.3
        
        # Author present
        if source.author:
            score += 0.2
        
        # Publication date present
        if source.publication_date:
            score += 0.2
        
        # Publication name present
        if source.publication_name:
            score += 0.15
        
        # URL is valid and accessible
        if source.url:
            score += 0.1
        
        # Description/metadata richness
        if metadata.description:
            score += 0.05
        
        return min(1.0, score)
    
    async def format_multiple_citations(self, search_results: List[SearchResult]) -> List[FormattedCitation]:
        """Format multiple search results into citations."""
        tasks = [self.create_citation_from_search_result(result) for result in search_results]
        citations = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_citations = []
        for i, citation in enumerate(citations):
            if isinstance(citation, Exception):
                logger.error(f"{RED}Error formatting citation {i}: {citation}{END}")
            else:
                valid_citations.append(citation)
        
        # Sort by confidence score
        valid_citations.sort(key=lambda c: c.confidence_score, reverse=True)
        
        return valid_citations
    
    def validate_citation_quality(self, citation: FormattedCitation, min_confidence: float = 0.7) -> bool:
        """
        Validate citation quality based on confidence score and content.
        
        Args:
            citation: Citation to validate
            min_confidence: Minimum confidence score required
            
        Returns:
            True if citation meets quality standards
        """
        if citation.confidence_score < min_confidence:
            return False
        
        # Check for minimum required elements
        if not citation.source.title or len(citation.source.title) < 5:
            return False
        
        if not citation.source.url:
            return False
        
        # Validate formatted text
        try:
            self.citation_validator.validate_citation_format(citation.formatted_text)
            return True
        except:
            return False
    
    def clean_citation_text(self, citation_text: str) -> str:
        """Clean and normalize citation text."""
        # Remove extra whitespace
        citation_text = re.sub(r'\s+', ' ', citation_text).strip()
        
        # Ensure proper punctuation
        if not citation_text.endswith('.'):
            citation_text += '.'
        
        # Remove duplicate punctuation
        citation_text = re.sub(r'\.+', '.', citation_text)
        
        return citation_text