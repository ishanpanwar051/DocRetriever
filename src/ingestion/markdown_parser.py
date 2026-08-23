import re
from pathlib import Path

class MarkdownParser:
    """
    Parses Markdown files into logical sections based on headings.
    WHY: Splitting on headings creates meaningful semantic boundaries for retrieval. 
    A single heading and its content usually cover a single topic, making it an 
    ideal candidate for a RAG chunk before further splitting.
    """
    
    def parse_file(self, path: Path) -> list[dict]:
        """
        Parses a .md file into a list of section dictionaries.
        
        Args:
            path: Path to the markdown file.
            
        Returns:
            list[dict]: List of sections with keys: content, section_title, source_file
        """
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Regex to match H1 or H2 headings
        # Note: In a production setting, this regex might be expanded to handle 
        # code blocks robustly, but for this parser, we assume standard well-formatted md.
        heading_pattern = re.compile(r'^(#{1,2})\s+(.*)$', re.MULTILINE)
        
        sections = []
        last_pos = 0
        current_title = "Introduction"
        
        # Iteratively find headings and capture the content in between
        for match in heading_pattern.finditer(content):
            start = match.start()
            
            # Content before this heading
            section_content = content[last_pos:start].strip()
            if section_content:
                sections.append({
                    "content": section_content,
                    "section_title": current_title,
                    "source_file": str(path)  # Storing relative path or absolute depending on caller
                })
            
            # Update title and position for next section
            current_title = match.group(2).strip()
            last_pos = match.end()
            
        # Add the final section
        final_content = content[last_pos:].strip()
        if final_content:
            sections.append({
                "content": final_content,
                "section_title": current_title,
                "source_file": str(path)
            })
            
        return sections
