#!/usr/bin/env python3
"""
Professional Web Scraper for ComfyUI MCP Server v2.0
Handles various content sources including arXiv papers and GitHub repositories
"""

import asyncio
import logging
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ScrapedContent:
    """Structured scraped content"""
    url: str
    title: str
    content: str
    content_type: str
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None

class ContentScraper:
    """Professional content scraper with multiple source support"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.timeout = self.config.get("timeout", 30)
        self.user_agent = self.config.get("user_agent", "ComfyUI-MCP-Server/2.0")
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    
    async def scrape_url(self, url: str, method: str = "auto") -> ScrapedContent:
        """Scrape content from URL with automatic method detection"""
        
        if not HTTPX_AVAILABLE:
            return ScrapedContent(
                url=url,
                title="Error",
                content="httpx not available for web scraping",
                content_type="error",
                metadata={},
                success=False,
                error="httpx dependency not installed"
            )
        
        try:
            # Determine scraping method based on URL
            if method == "auto":
                method = self._detect_method(url)
            
            logger.info(f"Scraping {url} using method: {method}")
            
            if method == "arxiv":
                return await self._scrape_arxiv(url)
            elif method == "github":
                return await self._scrape_github(url)
            else:
                return await self._scrape_generic(url)
                
        except Exception as e:
            error_msg = f"Scraping failed for {url}: {str(e)}"
            logger.error(error_msg)
            return ScrapedContent(
                url=url,
                title="Error",
                content=f"Scraping failed: {str(e)}",
                content_type="error",
                metadata={},
                success=False,
                error=error_msg
            )
    
    def _detect_method(self, url: str) -> str:
        """Detect appropriate scraping method based on URL"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        if "arxiv.org" in domain:
            return "arxiv"
        elif "github.com" in domain:
            return "github"
        else:
            return "generic"
    
    async def _scrape_arxiv(self, url: str) -> ScrapedContent:
        """Scrape arXiv paper information"""
        try:
            # Extract paper ID from URL
            paper_id = self._extract_arxiv_id(url)
            if not paper_id:
                raise ValueError("Could not extract arXiv paper ID from URL")
            
            # Get paper abstract page
            abstract_url = f"https://arxiv.org/abs/{paper_id}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(abstract_url, headers=self.headers)
                response.raise_for_status()
                
                if not BS4_AVAILABLE:
                    # Fallback without BeautifulSoup
                    content = response.text
                    title = self._extract_title_regex(content)
                    abstract = self._extract_abstract_regex(content)
                else:
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = self._extract_arxiv_title(soup)
                    abstract = self._extract_arxiv_abstract(soup)
                
                # Try to get additional metadata
                metadata = await self._get_arxiv_metadata(paper_id, client)
                
                content = f"""# {title}

## Abstract
{abstract}

## Paper Information
- arXiv ID: {paper_id}
- URL: {url}
- Abstract URL: {abstract_url}
- PDF URL: https://arxiv.org/pdf/{paper_id}.pdf

## Metadata
{self._format_metadata(metadata)}
"""
                
                return ScrapedContent(
                    url=url,
                    title=title,
                    content=content,
                    content_type="arxiv_paper",
                    metadata={
                        "paper_id": paper_id,
                        "abstract_url": abstract_url,
                        "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
                        **metadata
                    },
                    success=True
                )
                
        except Exception as e:
            raise Exception(f"arXiv scraping failed: {str(e)}")
    
    async def _scrape_github(self, url: str) -> ScrapedContent:
        """Scrape GitHub repository information"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                
                if not BS4_AVAILABLE:
                    # Fallback without BeautifulSoup
                    content = response.text
                    title = self._extract_title_regex(content)
                    description = "GitHub repository"
                else:
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = self._extract_github_title(soup)
                    description = self._extract_github_description(soup)
                    readme_content = await self._get_github_readme(url, client)
                
                # Extract repository information
                repo_info = self._parse_github_url(url)
                
                content = f"""# {title}

## Description
{description}

## Repository Information
- Owner: {repo_info.get('owner', 'unknown')}
- Repository: {repo_info.get('repo', 'unknown')}
- URL: {url}

## README Content
{readme_content if BS4_AVAILABLE else 'README content not available (BeautifulSoup not installed)'}
"""
                
                return ScrapedContent(
                    url=url,
                    title=title,
                    content=content,
                    content_type="github_repository",
                    metadata=repo_info,
                    success=True
                )
                
        except Exception as e:
            raise Exception(f"GitHub scraping failed: {str(e)}")
    
    async def _scrape_generic(self, url: str) -> ScrapedContent:
        """Scrape generic web content"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                
                if not BS4_AVAILABLE:
                    # Fallback without BeautifulSoup
                    content = response.text[:5000]  # Limit content
                    title = self._extract_title_regex(content)
                else:
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = soup.title.string if soup.title else "Unknown Title"
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    content = soup.get_text()
                    
                    # Clean up content
                    lines = (line.strip() for line in content.splitlines())
                    content = '\n'.join(line for line in lines if line)
                    
                    # Limit content length
                    if len(content) > 10000:
                        content = content[:10000] + "\n\n[Content truncated...]"
                
                return ScrapedContent(
                    url=url,
                    title=title,
                    content=content,
                    content_type="web_page",
                    metadata={"content_length": len(content)},
                    success=True
                )
                
        except Exception as e:
            raise Exception(f"Generic scraping failed: {str(e)}")
    
    # Helper methods
    def _extract_arxiv_id(self, url: str) -> Optional[str]:
        """Extract arXiv paper ID from URL"""
        patterns = [
            r'arxiv\.org/abs/([0-9]+\.[0-9]+)',
            r'arxiv\.org/pdf/([0-9]+\.[0-9]+)',
            r'([0-9]+\.[0-9]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _extract_title_regex(self, html: str) -> str:
        """Extract title using regex (fallback method)"""
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        return title_match.group(1).strip() if title_match else "Unknown Title"
    
    def _extract_abstract_regex(self, html: str) -> str:
        """Extract abstract using regex (fallback method)"""
        abstract_match = re.search(r'<blockquote[^>]*class="abstract"[^>]*>.*?<span[^>]*class="descriptor"[^>]*>Abstract:</span>\s*([^<]+)', html, re.DOTALL | re.IGNORECASE)
        return abstract_match.group(1).strip() if abstract_match else "Abstract not found"
    
    def _extract_arxiv_title(self, soup) -> str:
        """Extract title from arXiv page"""
        title_elem = soup.find('h1', class_='title')
        if title_elem:
            # Remove "Title:" prefix if present
            title = title_elem.get_text().strip()
            return title.replace('Title:', '').strip()
        return "Unknown Title"
    
    def _extract_arxiv_abstract(self, soup) -> str:
        """Extract abstract from arXiv page"""
        abstract_elem = soup.find('blockquote', class_='abstract')
        if abstract_elem:
            # Remove "Abstract:" prefix
            abstract = abstract_elem.get_text().strip()
            return abstract.replace('Abstract:', '').strip()
        return "Abstract not found"
    
    def _extract_github_title(self, soup) -> str:
        """Extract title from GitHub page"""
        title_elem = soup.find('strong', {'itemprop': 'name'})
        if title_elem:
            return title_elem.get_text().strip()
        
        # Fallback to page title
        title_elem = soup.find('title')
        if title_elem:
            return title_elem.get_text().strip()
        
        return "Unknown Repository"
    
    def _extract_github_description(self, soup) -> str:
        """Extract description from GitHub page"""
        desc_elem = soup.find('p', class_='f4')
        if desc_elem:
            return desc_elem.get_text().strip()
        return "No description available"
    
    def _parse_github_url(self, url: str) -> Dict[str, str]:
        """Parse GitHub URL to extract owner and repo"""
        match = re.search(r'github\.com/([^/]+)/([^/]+)', url)
        if match:
            return {
                "owner": match.group(1),
                "repo": match.group(2),
                "full_name": f"{match.group(1)}/{match.group(2)}"
            }
        return {}
    
    async def _get_arxiv_metadata(self, paper_id: str, client) -> Dict[str, Any]:
        """Get additional metadata for arXiv paper"""
        try:
            # This would typically call arXiv API
            # For now, return basic metadata
            return {
                "source": "arxiv",
                "paper_id": paper_id,
                "scraped_at": "2025-06-27"
            }
        except:
            return {}
    
    async def _get_github_readme(self, repo_url: str, client) -> str:
        """Get README content from GitHub repository"""
        try:
            repo_info = self._parse_github_url(repo_url)
            if not repo_info:
                return "README not available"
            
            readme_url = f"https://raw.githubusercontent.com/{repo_info['full_name']}/main/README.md"
            response = await client.get(readme_url)
            
            if response.status_code == 200:
                return response.text[:5000]  # Limit README length
            else:
                # Try master branch
                readme_url = f"https://raw.githubusercontent.com/{repo_info['full_name']}/master/README.md"
                response = await client.get(readme_url)
                if response.status_code == 200:
                    return response.text[:5000]
            
            return "README not found"
        except:
            return "README not available"
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for display"""
        if not metadata:
            return "No additional metadata available"
        
        formatted = []
        for key, value in metadata.items():
            formatted.append(f"- {key}: {value}")
        
        return "\n".join(formatted)

# Factory function
def create_content_scraper(config: Optional[Dict[str, Any]] = None) -> ContentScraper:
    """Create and configure content scraper"""
    return ContentScraper(config)
