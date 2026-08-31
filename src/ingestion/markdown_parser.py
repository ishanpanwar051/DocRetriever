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
        Safely ignores headings inside fenced code blocks (```).
        
        Args:
            path: Path to the markdown file.
            
        Returns:
            list[dict]: List of sections with keys: content, section_title, source_file
        """
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        sections = []
        current_title = "Introduction"
        current_lines: list[str] = []
        in_code_block = False
        
        heading_re = re.compile(r'^(#{1,2})\s+(.+)$')
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                current_lines.append(line)
                continue
                
            # Only match headings outside of code blocks
            if not in_code_block:
                match = heading_re.match(line)
                if match:
                    section_content = "".join(current_lines).strip()
                    if section_content:
                        sections.append({
                            "content": section_content,
                            "section_title": current_title,
                            "source_file": str(path)
                        })
                    current_title = match.group(2).strip()
                    current_lines = []
                    continue
                    
            current_lines.append(line)
            
        # Add remaining content
        final_content = "".join(current_lines).strip()
        if final_content:
            sections.append({
                "content": final_content,
                "section_title": current_title,
                "source_file": str(path)
            })
            
        return sections
