"""Web search API with multi-provider integration and fallback system."""

import asyncio
import time
from typing import List, Dict, Optional, Any
import requests
from urllib.parse import quote, urljoin
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import aiohttp
from bs4 import BeautifulSoup

from ..schemas.citation_schemas import (
    SearchQuery, SearchResponse, SearchResult, SearchProvider, 
    APIError, RateLimitInfo, SearchMetrics
)
from ..config.config_manager import get_config_manager

# Define color constants
BLUE, GREEN, YELLOW, CYAN, RED, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[91m", "\033[0m"


@dataclass
class ProviderCredentials:
    """Credentials for search providers."""
    api_key: str
    search_engine_id: Optional[str] = None  # For Google Custom Search
    subscription_key: Optional[str] = None  # For Bing


class BaseSearchProvider:
    """Base class for search providers."""
    
    def __init__(self, provider: SearchProvider, credentials: Optional[ProviderCredentials] = None):
        self.provider = provider
        self.credentials = credentials
        self.rate_limit_info = RateLimitInfo(provider=provider)
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Perform search using this provider."""
        raise NotImplementedError("Subclasses must implement search method")
    
    def can_search(self) -> bool:
        """Check if provider can perform search (rate limits, credentials, etc.)."""
        return self.rate_limit_info.can_make_request()


class GoogleSearchProvider(BaseSearchProvider):
    """Google Custom Search API provider."""
    
    def __init__(self, credentials: ProviderCredentials):
        super().__init__(SearchProvider.GOOGLE, credentials)
        self.base_url = "https://customsearch.googleapis.com/customsearch/v1"
        
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Search using Google Custom Search API."""
        start_time = time.time()
        
        if not self.credentials or not self.credentials.api_key:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=0.0,
                provider_used=self.provider,
                error="Google API key not configured"
            )
        
        try:
            params = {
                'key': self.credentials.api_key,
                'cx': self.credentials.search_engine_id or '017576662512468239146:omuauf_lfve',  # Default search engine
                'q': query.query,
                'num': min(query.max_results, 10),  # Google limits to 10 per request
                'safe': 'active'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, timeout=query.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = self._parse_google_results(data, query)
                        
                        search_time = time.time() - start_time
                        total_results = int(data.get('searchInformation', {}).get('totalResults', '0'))
                        
                        self.rate_limit_info.requests_made += 1
                        
                        return SearchResponse(
                            query=query,
                            results=results,
                            total_results=total_results,
                            search_time=search_time,
                            provider_used=self.provider,
                            error=None
                        )
                    else:
                        error_msg = f"Google search failed: HTTP {response.status}"
                        logger.error(f"{RED}{error_msg}{END}")
                        
                        return SearchResponse(
                            query=query,
                            results=[],
                            total_results=0,
                            search_time=time.time() - start_time,
                            provider_used=self.provider,
                            error=error_msg
                        )
                        
        except Exception as e:
            error_msg = f"Google search error: {str(e)}"
            logger.error(f"{RED}{error_msg}{END}")
            
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=time.time() - start_time,
                provider_used=self.provider,
                error=error_msg
            )
    
    def _parse_google_results(self, data: Dict[str, Any], query: SearchQuery) -> List[SearchResult]:
        """Parse Google search results."""
        results = []
        items = data.get('items', [])
        
        for i, item in enumerate(items[:query.max_results]):
            try:
                result = SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    provider=self.provider,
                    relevance_score=1.0 - (i * 0.1),  # Decreasing relevance by rank
                    rank=i + 1
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"{YELLOW}Error parsing Google result {i}: {e}{END}")
                continue
        
        return results


class BingSearchProvider(BaseSearchProvider):
    """Bing Web Search API provider."""
    
    def __init__(self, credentials: ProviderCredentials):
        super().__init__(SearchProvider.BING, credentials)
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Search using Bing Web Search API."""
        start_time = time.time()
        
        if not self.credentials or not self.credentials.subscription_key:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=0.0,
                provider_used=self.provider,
                error="Bing API key not configured"
            )
        
        try:
            headers = {
                'Ocp-Apim-Subscription-Key': self.credentials.subscription_key
            }
            
            params = {
                'q': query.query,
                'count': min(query.max_results, 50),  # Bing allows up to 50
                'offset': 0,
                'mkt': 'en-US',
                'safeSearch': 'Moderate'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, headers=headers, timeout=query.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = self._parse_bing_results(data, query)
                        
                        search_time = time.time() - start_time
                        total_results = data.get('webPages', {}).get('totalEstimatedMatches', 0)
                        
                        self.rate_limit_info.requests_made += 1
                        
                        return SearchResponse(
                            query=query,
                            results=results,
                            total_results=total_results,
                            search_time=search_time,
                            provider_used=self.provider,
                            error=None
                        )
                    else:
                        error_msg = f"Bing search failed: HTTP {response.status}"
                        logger.error(f"{RED}{error_msg}{END}")
                        
                        return SearchResponse(
                            query=query,
                            results=[],
                            total_results=0,
                            search_time=time.time() - start_time,
                            provider_used=self.provider,
                            error=error_msg
                        )
                        
        except Exception as e:
            error_msg = f"Bing search error: {str(e)}"
            logger.error(f"{RED}{error_msg}{END}")
            
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=time.time() - start_time,
                provider_used=self.provider,
                error=error_msg
            )
    
    def _parse_bing_results(self, data: Dict[str, Any], query: SearchQuery) -> List[SearchResult]:
        """Parse Bing search results."""
        results = []
        web_pages = data.get('webPages', {}).get('value', [])
        
        for i, item in enumerate(web_pages[:query.max_results]):
            try:
                result = SearchResult(
                    title=item.get('name', ''),
                    url=item.get('url', ''),
                    snippet=item.get('snippet', ''),
                    provider=self.provider,
                    relevance_score=1.0 - (i * 0.1),  # Decreasing relevance by rank
                    rank=i + 1
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"{YELLOW}Error parsing Bing result {i}: {e}{END}")
                continue
        
        return results


class DuckDuckGoSearchProvider(BaseSearchProvider):
    """DuckDuckGo search provider (no API key required)."""
    
    def __init__(self):
        super().__init__(SearchProvider.DUCKDUCKGO)
        self.base_url = "https://html.duckduckgo.com/html/"
        # Set higher rate limit for DDG since it's free
        self.rate_limit_info.requests_limit = 50
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Search using DuckDuckGo HTML scraping."""
        start_time = time.time()
        
        try:
            # DuckDuckGo HTML search
            params = {
                'q': query.query,
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, headers=headers, timeout=query.timeout) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        results = self._parse_duckduckgo_results(html_content, query)
                        
                        search_time = time.time() - start_time
                        
                        self.rate_limit_info.requests_made += 1
                        
                        return SearchResponse(
                            query=query,
                            results=results,
                            total_results=len(results),
                            search_time=search_time,
                            provider_used=self.provider,
                            error=None
                        )
                    else:
                        error_msg = f"DuckDuckGo search failed: HTTP {response.status}"
                        logger.error(f"{RED}{error_msg}{END}")
                        
                        return SearchResponse(
                            query=query,
                            results=[],
                            total_results=0,
                            search_time=time.time() - start_time,
                            provider_used=self.provider,
                            error=error_msg
                        )
                        
        except Exception as e:
            error_msg = f"DuckDuckGo search error: {str(e)}"
            logger.error(f"{RED}{error_msg}{END}")
            
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=time.time() - start_time,
                provider_used=self.provider,
                error=error_msg
            )
    
    def _parse_duckduckgo_results(self, html_content: str, query: SearchQuery) -> List[SearchResult]:
        """Parse DuckDuckGo HTML results."""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            result_divs = soup.find_all('div', class_='result')
            
            for i, div in enumerate(result_divs[:query.max_results]):
                try:
                    # Extract title and URL
                    title_link = div.find('a', class_='result__a')
                    if not title_link:
                        continue
                    
                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '')
                    
                    # Extract snippet
                    snippet_div = div.find('a', class_='result__snippet')
                    snippet = snippet_div.get_text(strip=True) if snippet_div else ''
                    
                    if title and url:
                        result = SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            provider=self.provider,
                            relevance_score=1.0 - (i * 0.1),
                            rank=i + 1
                        )
                        results.append(result)
                
                except Exception as e:
                    logger.debug(f"{YELLOW}Error parsing DDG result {i}: {e}{END}")
                    continue
        
        except Exception as e:
            logger.warning(f"{YELLOW}Error parsing DuckDuckGo HTML: {e}{END}")
        
        return results


class WebSearchAPI:
    """Multi-provider web search API with fallback system."""
    
    def __init__(self):
        """Initialize WebSearchAPI with configuration."""
        self.config_manager = get_config_manager()
        self.search_config = self.config_manager.get_search_config()
        self.api_keys = self.config_manager.get_api_keys()
        
        self.providers: Dict[SearchProvider, BaseSearchProvider] = {}
        self.metrics = SearchMetrics()
        
        self._setup_providers()
    
    def _setup_providers(self) -> None:
        """Set up available search providers."""
        # Setup Google
        if self.api_keys.google_search_key:
            credentials = ProviderCredentials(
                api_key=self.api_keys.google_search_key,
                search_engine_id=None  # Use default
            )
            self.providers[SearchProvider.GOOGLE] = GoogleSearchProvider(credentials)
            logger.info(f"{GREEN}Google search provider initialized{END}")
        
        # Setup Bing
        if self.api_keys.bing_search_key:
            credentials = ProviderCredentials(
                api_key=self.api_keys.bing_search_key,
                subscription_key=self.api_keys.bing_search_key
            )
            self.providers[SearchProvider.BING] = BingSearchProvider(credentials)
            logger.info(f"{GREEN}Bing search provider initialized{END}")
        
        # Setup DuckDuckGo (always available)
        self.providers[SearchProvider.DUCKDUCKGO] = DuckDuckGoSearchProvider()
        logger.info(f"{GREEN}DuckDuckGo search provider initialized{END}")
        
        logger.info(f"{BLUE}WebSearchAPI initialized with {len(self.providers)} providers{END}")
    
    async def search(self, query: str, max_results: int = 10, providers: Optional[List[SearchProvider]] = None) -> SearchResponse:
        """
        Perform web search with fallback across providers.
        
        Args:
            query: Search query text
            max_results: Maximum number of results to return
            providers: Specific providers to use (None for auto-selection)
            
        Returns:
            SearchResponse with results from the first successful provider
        """
        search_query = SearchQuery(
            query=query,
            max_results=max_results,
            providers=providers or self.search_config.providers,
            timeout=self.search_config.timeout
        )
        
        # Try providers in order
        provider_order = providers or self._get_provider_fallback_order()
        
        last_error = None
        
        for provider in provider_order:
            if provider not in self.providers:
                logger.warning(f"{YELLOW}Provider {provider} not available{END}")
                continue
            
            search_provider = self.providers[provider]
            
            if not search_provider.can_search():
                logger.warning(f"{YELLOW}Provider {provider} rate limited{END}")
                continue
            
            try:
                logger.debug(f"{CYAN}Trying search with {provider}...{END}")
                response = await search_provider.search(search_query)
                
                self.metrics.total_queries += 1
                
                if response.error is None and response.results:
                    self.metrics.successful_queries += 1
                    self.metrics.total_results_found += len(response.results)
                    self.metrics.providers_used[provider] = self.metrics.providers_used.get(provider, 0) + 1
                    
                    # Update average response time
                    self.metrics.average_response_time = (
                        (self.metrics.average_response_time * (self.metrics.successful_queries - 1) + response.search_time)
                        / self.metrics.successful_queries
                    )
                    
                    logger.success(f"{GREEN}Search successful with {provider}: {len(response.results)} results{END}")
                    return response
                else:
                    self.metrics.failed_queries += 1
                    last_error = response.error or f"No results from {provider}"
                    logger.warning(f"{YELLOW}Search failed with {provider}: {last_error}{END}")
            
            except Exception as e:
                self.metrics.failed_queries += 1
                last_error = str(e)
                logger.error(f"{RED}Error with {provider}: {e}{END}")
        
        # All providers failed
        logger.error(f"{RED}All search providers failed{END}")
        return SearchResponse(
            query=search_query,
            results=[],
            total_results=0,
            search_time=0.0,
            provider_used=SearchProvider.DUCKDUCKGO,  # Default fallback
            error=last_error or "All search providers failed"
        )
    
    def _get_provider_fallback_order(self) -> List[SearchProvider]:
        """Get provider fallback order based on configuration and availability."""
        # Start with configured providers
        order = self.search_config.providers.copy()
        
        # Add any additional available providers not in config
        for provider in self.providers.keys():
            if provider not in order:
                order.append(provider)
        
        # Always ensure DuckDuckGo is last as fallback
        if SearchProvider.DUCKDUCKGO in order:
            order.remove(SearchProvider.DUCKDUCKGO)
        order.append(SearchProvider.DUCKDUCKGO)
        
        return order
    
    async def search_multiple_queries(self, queries: List[str], max_results: int = 10) -> List[SearchResponse]:
        """Search multiple queries concurrently."""
        tasks = [self.search(query, max_results) for query in queries]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"{RED}Error searching query {i}: {response}{END}")
                # Create error response
                error_response = SearchResponse(
                    query=SearchQuery(query=queries[i], max_results=max_results),
                    results=[],
                    total_results=0,
                    search_time=0.0,
                    provider_used=SearchProvider.DUCKDUCKGO,
                    error=str(response)
                )
                results.append(error_response)
            else:
                results.append(response)
        
        return results
    
    def get_metrics(self) -> SearchMetrics:
        """Get search API metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset search metrics."""
        self.metrics = SearchMetrics()
        logger.info(f"{BLUE}Search metrics reset{END}")
    
    def get_provider_status(self) -> Dict[SearchProvider, Dict[str, Any]]:
        """Get status of all providers."""
        status = {}
        
        for provider, search_provider in self.providers.items():
            rate_limit = search_provider.rate_limit_info
            status[provider] = {
                "available": search_provider.can_search(),
                "requests_made": rate_limit.requests_made,
                "requests_limit": rate_limit.requests_limit,
                "has_credentials": bool(search_provider.credentials),
            }
        
        return status