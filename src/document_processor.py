import fitz
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import os
import json

# Configure logging
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("logs/document_processor.log")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    
class DocumentProcessor:
    def __init__(self):
        self.documents: List[Document] = []
        
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF and extract text, tables, and links."""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract regular text
                text = page.get_text()
                
                # Extract tables
                tables = self._extract_tables(page)
                tables_text = self._format_tables(tables)
                
                # Extract links
                links = page.get_links()
                link_texts = []
                
                for link in links:
                    if "uri" in link:
                        try:
                            link_content = self._scrape_link(link["uri"])
                            link_texts.append(link_content)
                        except Exception as e:
                            logger.error(f"Error scraping link {link['uri']}: {str(e)}")
                
                # Combine all content
                combined_content = f"{text}\n\nTABLES:\n{tables_text}\n\nLINKED CONTENT:\n" + "\n".join(link_texts)
                
                # Create structured metadata
                metadata = {
                    "source": pdf_path,
                    "page": page_num,
                    "type": "statement",
                    "has_tables": bool(tables),
                    "table_count": len(tables),
                    "link_count": len(links)
                }
                
                # Add table metadata if present
                if tables:
                    metadata["tables"] = [
                        {
                            "row_count": len(table),
                            "col_count": len(table[0]) if table else 0
                        }
                        for table in tables
                    ]
                
                document = Document(
                    content=combined_content,
                    metadata=metadata
                )
                self.documents.append(document)
                
            logger.info(f"Successfully processed PDF with {len(doc)} pages")
            return self.documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
            
    def _extract_tables(self, page: fitz.Page) -> List[List[List[str]]]:
        """Extract tables from a PDF page using PyMuPDF."""
        try:
            # Find tables on the page
            tables = page.find_tables(
                vertical_strategy="lines",
                horizontal_strategy="lines",
                snap_tolerance=3,
                snap_x_tolerance=3,
                snap_y_tolerance=3,
                join_tolerance=3,
                edge_min_length=3,
                min_words_vertical=2,
                min_words_horizontal=2
            )
            
            extracted_tables = []
            for table in tables:
                # Extract cells from the table
                cells = table.extract()
                # Clean and format cell content
                cleaned_table = [
                    [
                        str(cell).strip() if cell is not None else ""
                        for cell in row
                    ]
                    for row in cells
                ]
                extracted_tables.append(cleaned_table)
                
            return extracted_tables
            
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return []
            
    def _format_tables(self, tables: List[List[List[str]]]) -> str:
        """Format extracted tables into a readable string."""
        if not tables:
            return ""
            
        formatted_text = []
        for i, table in enumerate(tables, 1):
            formatted_text.append(f"Table {i}:")
            
            # Get maximum column widths
            col_widths = [
                max(len(str(row[i])) for row in table)
                for i in range(len(table[0]))
            ]
            
            # Create formatted rows
            for row in table:
                formatted_row = " | ".join(
                    str(cell).ljust(width)
                    for cell, width in zip(row, col_widths)
                )
                formatted_text.append(formatted_row)
            
            formatted_text.append("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
            formatted_text.append("")
            
        return "\n".join(formatted_text)
            
    def _scrape_link(self, url: str) -> str:
        """Scrape content from a link."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return "" 
