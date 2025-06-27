"""
Advanced Web Scraping Module for MCP Multi-Agent Framework
Provides intelligent content extraction from various web sources
"""

import os
import re
import time
import logging
import requests
from urllib.parse import urljoin, urlparse, parse_qs
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


@dataclass
class ScrapedContent:
    """Container for scraped content with metadata"""
    url: str
    title: str
    content: str
    abstract: str = ""
    authors: List[str] = None
    publication_date: str = ""
    content_type: str = "webpage"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.metadata is None:
            self.metadata = {}


class AdvancedWebScraper:
    """Advanced web scraper with multiple extraction strategies"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = requests.Session()
        
        # Configure session
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Rate limiting
        self.request_delay = self.config.get('request_delay', 1.0)
        self.last_request_time = 0
    
    def scrape_url(self, url: str, method: str = "auto") -> ScrapedContent:
        """
        Scrape content from URL using specified method
        
        Args:
            url: URL to scrape
            method: Extraction method ('auto', 'trafilatura', 'newspaper', 'selenium', 'arxiv')
        
        Returns:
            ScrapedContent object with extracted data
        """
        self.logger.info(f"Scraping URL: {url} using method: {method}")
        
        try:
            # Determine content type and best extraction method
            if method == "auto":
                method = self._determine_best_method(url)
            
            # Route to appropriate scraper
            if method == "arxiv" and self._is_arxiv_url(url):
                return self._scrape_arxiv(url)
            elif method == "trafilatura":
                return self._scrape_with_trafilatura(url)
            elif method == "newspaper":
                return self._scrape_with_newspaper(url)
            elif method == "selenium":
                return self._scrape_with_selenium(url)
            else:
                return self._scrape_with_requests(url)
                
        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {str(e)}")
            return ScrapedContent(
                url=url,
                title="Scraping Failed",
                content=f"Error scraping content: {str(e)}",
                content_type="error"
            )
    
    def _determine_best_method(self, url: str) -> str:
        """Determine the best scraping method for a URL"""
        
        if self._is_arxiv_url(url):
            return "arxiv"
        elif self._is_github_url(url):
            return "trafilatura"  # Try trafilatura first for GitHub, fallback to selenium if needed
        elif self._is_academic_site(url):
            return "trafilatura"  # Good for academic content
        elif self._is_news_site(url):
            return "newspaper"  # Optimized for news articles
        else:
            return "trafilatura"  # Good general-purpose choice
    
    def _is_arxiv_url(self, url: str) -> bool:
        """Check if URL is from arXiv"""
        return "arxiv.org" in url.lower()
    
    def _is_github_url(self, url: str) -> bool:
        """Check if URL is from GitHub"""
        return "github.com" in url.lower()
    
    def _is_academic_site(self, url: str) -> bool:
        """Check if URL is from an academic site"""
        academic_domains = [
            'ieee.org', 'acm.org', 'springer.com', 'elsevier.com',
            'nature.com', 'science.org', 'pnas.org', 'arxiv.org',
            'researchgate.net', 'scholar.google.com', 'semanticscholar.org'
        ]
        return any(domain in url.lower() for domain in academic_domains)
    
    def _is_news_site(self, url: str) -> bool:
        """Check if URL is from a news site"""
        news_domains = [
            'cnn.com', 'bbc.com', 'reuters.com', 'ap.org',
            'nytimes.com', 'washingtonpost.com', 'theguardian.com'
        ]
        return any(domain in url.lower() for domain in news_domains)
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _scrape_arxiv(self, url: str) -> ScrapedContent:
        """Scrape arXiv papers using the arxiv API"""
        
        if not ARXIV_AVAILABLE:
            self.logger.warning("arxiv package not available, falling back to web scraping")
            return self._scrape_with_trafilatura(url)
        
        try:
            # Extract arXiv ID from URL
            arxiv_id = self._extract_arxiv_id(url)
            if not arxiv_id:
                raise ValueError("Could not extract arXiv ID from URL")
            
            # Search for paper
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            
            # Download PDF content if available
            pdf_content = ""
            try:
                pdf_path = paper.download_pdf(dirpath="/tmp")
                pdf_content = self._extract_pdf_text(pdf_path)
                os.remove(pdf_path)  # Clean up
            except Exception as e:
                self.logger.warning(f"Could not download PDF: {str(e)}")
            
            return ScrapedContent(
                url=url,
                title=paper.title,
                content=pdf_content or paper.summary,
                abstract=paper.summary,
                authors=[author.name for author in paper.authors],
                publication_date=paper.published.strftime("%Y-%m-%d") if paper.published else "",
                content_type="arxiv_paper",
                metadata={
                    "arxiv_id": arxiv_id,
                    "categories": [cat for cat in paper.categories],
                    "doi": paper.doi,
                    "journal_ref": paper.journal_ref,
                    "comment": paper.comment
                }
            )
            
        except Exception as e:
            self.logger.error(f"arXiv scraping failed: {str(e)}")
            return self._scrape_with_trafilatura(url)
    
    def _extract_arxiv_id(self, url: str) -> Optional[str]:
        """Extract arXiv ID from URL"""
        patterns = [
            r'arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})',
            r'arxiv\.org/abs/([a-z-]+/[0-9]{7})',
            r'arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5})',
            r'arxiv\.org/pdf/([a-z-]+/[0-9]{7})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _scrape_with_trafilatura(self, url: str) -> ScrapedContent:
        """Scrape using trafilatura for clean content extraction"""
        
        if not TRAFILATURA_AVAILABLE:
            return self._scrape_with_requests(url)
        
        try:
            self._rate_limit()
            
            # Download page
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                raise ValueError("Could not download page")
            
            # Extract content
            content = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            if not content:
                raise ValueError("Could not extract content")
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(downloaded)
            
            title = metadata.title if metadata and metadata.title else "Unknown Title"
            authors = metadata.author.split(", ") if metadata and metadata.author else []
            pub_date = metadata.date if metadata and metadata.date else ""
            
            return ScrapedContent(
                url=url,
                title=title,
                content=content,
                authors=authors,
                publication_date=pub_date,
                content_type="webpage",
                metadata={
                    "description": metadata.description if metadata else "",
                    "sitename": metadata.sitename if metadata else "",
                    "categories": metadata.categories if metadata else []
                }
            )
            
        except Exception as e:
            self.logger.error(f"Trafilatura scraping failed: {str(e)}")
            return self._scrape_with_requests(url)
    
    def _scrape_with_newspaper(self, url: str) -> ScrapedContent:
        """Scrape using newspaper3k for article extraction"""
        
        if not NEWSPAPER_AVAILABLE:
            return self._scrape_with_requests(url)
        
        try:
            self._rate_limit()
            
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()  # Extract keywords and summary
            
            return ScrapedContent(
                url=url,
                title=article.title or "Unknown Title",
                content=article.text,
                abstract=article.summary,
                authors=article.authors,
                publication_date=article.publish_date.strftime("%Y-%m-%d") if article.publish_date else "",
                content_type="article",
                metadata={
                    "keywords": article.keywords,
                    "top_image": article.top_image,
                    "movies": article.movies
                }
            )
            
        except Exception as e:
            self.logger.error(f"Newspaper scraping failed: {str(e)}")
            return self._scrape_with_requests(url)
    
    def _scrape_with_selenium(self, url: str) -> ScrapedContent:
        """Scrape using Selenium for JavaScript-heavy sites"""

        if not SELENIUM_AVAILABLE:
            self.logger.warning("Selenium not available, falling back to requests")
            return self._scrape_with_requests(url)

        driver = None
        try:
            self._rate_limit()

            # Configure Chrome options for better stability
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")
            chrome_options.add_argument("--disable-javascript")  # For basic content scraping
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")

            # Reduce timeouts for faster failure
            chrome_options.add_argument("--timeout=15000")
            chrome_options.add_argument("--page-load-strategy=eager")

            # Create driver with shorter timeouts
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(15)  # Reduced from 30
            driver.implicitly_wait(5)  # Reduced implicit wait

            # Load page with timeout handling
            self.logger.info(f"Loading page with Selenium: {url}")
            driver.get(url)

            # Wait for basic content with shorter timeout
            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except Exception as wait_e:
                self.logger.warning(f"Timeout waiting for body element: {wait_e}")
                # Continue anyway, might still have content

            # Extract content
            title = driver.title or "No title"

            # Try to find main content with GitHub-specific selectors
            content_selectors = [
                "readme", ".markdown-body", ".repository-content",  # GitHub specific
                "main", "article", ".content", "#content",
                ".post-content", ".entry-content", ".article-body"
            ]

            content = ""
            for selector in content_selectors:
                try:
                    element = driver.find_element(By.CSS_SELECTOR, selector)
                    content = element.text
                    if content.strip():  # Only use if we got actual content
                        break
                except:
                    continue

            # Fallback to body content
            if not content.strip():
                try:
                    content = driver.find_element(By.TAG_NAME, "body").text
                except:
                    content = "No content extracted"

            self.logger.info(f"Selenium extracted {len(content)} characters")

            return ScrapedContent(
                url=url,
                title=title,
                content=content,
                content_type="webpage_js",
                metadata={"scraped_with": "selenium", "content_length": len(content)}
            )

        except Exception as e:
            self.logger.error(f"Selenium scraping failed: {str(e)}")
            # Always fallback to requests for reliability
            self.logger.info("Falling back to requests-based scraping")
            return self._scrape_with_requests(url)
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass  # Ignore cleanup errors
    
    def _scrape_with_requests(self, url: str) -> ScrapedContent:
        """Basic scraping using requests and BeautifulSoup"""
        
        try:
            self._rate_limit()
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            if not BS4_AVAILABLE:
                return ScrapedContent(
                    url=url,
                    title="Raw HTML",
                    content=response.text,
                    content_type="raw_html"
                )
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "Unknown Title"
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            content = soup.get_text()
            
            # Clean up content
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)
            
            return ScrapedContent(
                url=url,
                title=title,
                content=content,
                content_type="webpage",
                metadata={"scraped_with": "requests_bs4"}
            )
            
        except Exception as e:
            self.logger.error(f"Basic scraping failed: {str(e)}")
            return ScrapedContent(
                url=url,
                title="Scraping Failed",
                content=f"Error: {str(e)}",
                content_type="error"
            )
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        
        if not PDF_AVAILABLE:
            return ""
        
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if text.strip():
                    return text
        except Exception as e:
            self.logger.warning(f"pdfplumber failed: {str(e)}")
        
        try:
            # Fallback to PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {str(e)}")
            return ""
    
    def scrape_multiple_urls(self, urls: List[str], method: str = "auto") -> List[ScrapedContent]:
        """Scrape multiple URLs with rate limiting"""
        
        results = []
        for i, url in enumerate(urls):
            self.logger.info(f"Scraping {i+1}/{len(urls)}: {url}")
            result = self.scrape_url(url, method)
            results.append(result)
            
            # Add delay between requests
            if i < len(urls) - 1:
                time.sleep(self.request_delay)
        
        return results
