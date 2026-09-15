"""
src/analytics/doc_insights.py — Document Intelligence, Risk Analysis & Analytics Engine

Features:
- Automated Executive Summary Generation (3-5 core takeaways)
- Risk & Compliance Clause Detection (Liabilities, Penalties, Indemnities, Regulations)
- Named Entity Extraction (Organizations, Money Amounts, Dates, Roles)
- Lexical KPIs & Telemetry (Word Count, Reading Time, Lexical Diversity)
- Topic & Keyword Frequency Distribution for Visual Plotly Dashboards
"""

import re
from typing import Dict, List, Any, Optional
from collections import Counter


RISK_PATTERNS = [
    {
        "category": "Liability & Indemnity",
        "severity": "HIGH",
        "patterns": [
            r"\b(indemnif\w+|hold harmless|unlimited liability|aggregate liability|direct damages|indirect damages)\b",
            r"\b(consequential damages|punitive damages|loss of profit|loss of data)\b",
        ],
        "description": "Indemnification obligations or liability caps identified in contractual text.",
    },
    {
        "category": "Breach & Penalties",
        "severity": "HIGH",
        "patterns": [
            r"\b(material breach|termination for cause|liquidated damages|penalty fee|default interest)\b",
            r"\b(cure period|remedy breach|forfeiture|suspension of service)\b",
        ],
        "description": "Contractual breach triggers, termination clauses, or financial penalties.",
    },
    {
        "category": "Regulatory & Privacy Compliance",
        "severity": "MEDIUM",
        "patterns": [
            r"\b(gdpr|hipaa|soc\s*2|pci[\s-]?dss|iso\s*27001|ccpa|data protection|pii|ph\b)",
            r"\b(audit rights|sub-processor|data breach notification|security standards)\b",
        ],
        "description": "Data privacy, security standards, or regulatory compliance mandates.",
    },
    {
        "category": "Financial & Deficit Warnings",
        "severity": "MEDIUM",
        "patterns": [
            r"\b(operating loss|net loss|ebitda deficit|debt obligation|default on payment)\b",
            r"\b(liquidity risk|going concern|impairment charge|credit downgrade)\b",
        ],
        "description": "Fiscal deficit, liquidity risks, or negative financial projections.",
    },
    {
        "category": "Warranty & SLA Disclaimers",
        "severity": "LOW",
        "patterns": [
            r"\b(as is|without warranty|sla penalty|service credit|disclaimer of warranty)\b",
            r"\b(force majeure|acts of god|reasonable efforts)\b",
        ],
        "description": "Warranty exclusions or SLA performance disclaimer clauses.",
    },
]


STOPWORDS = {
    "the", "and", "to", "of", "a", "in", "is", "that", "for", "it", "as", "was", "with",
    "be", "by", "on", "not", "he", "i", "this", "are", "or", "an", "they", "from", "at",
    "which", "you", "more", "can", "if", "has", "but", "all", "we", "will", "one", "all",
    "page", "table", "section", "document", "their", "have", "been", "would", "shall"
}


class DocumentInsightsEngine:
    """
    Analyzes document chunks and generates comprehensive intelligence reports.
    """

    def analyze_document(
        self,
        text_content: str,
        filename: str = "document.pdf",
        chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete document intelligence pipeline.
        """
        if not text_content or not text_content.strip():
            text_content = "Empty document."

        # 1. Lexical KPIs
        words = re.findall(r"\b[A-Za-z0-9_$%\.-]+\b", text_content)
        total_words = len(words)
        unique_words = len(set(w.lower() for w in words))
        reading_time_min = round(max(0.5, total_words / 200), 1)
        lexical_diversity = round(unique_words / max(1, total_words) * 100, 1)

        # Count tables and pages from chunks metadata if provided
        total_chunks = len(chunks) if chunks else max(1, total_words // 400)
        table_count = sum(1 for c in (chunks or []) if c.get("is_table") or "Table" in c.get("section_title", ""))
        page_count = max([c.get("page_number", 1) for c in (chunks or [])] + [1])

        # 2. Risk & Compliance Flags
        risk_flags = self._extract_risk_flags(text_content)

        # 3. Named Entity Recognition (NER)
        entities = self._extract_entities(text_content)

        # 4. Executive Summary
        executive_summary = self._generate_executive_summary(text_content, entities, risk_flags)

        # 5. Topic & Keyword Frequency Distribution
        topic_distribution = self._compute_topic_distribution(words)

        return {
            "filename": filename,
            "kpis": {
                "total_words": total_words,
                "unique_words": unique_words,
                "reading_time_min": reading_time_min,
                "lexical_diversity_pct": lexical_diversity,
                "total_chunks": total_chunks,
                "table_count": table_count,
                "page_count": page_count,
            },
            "executive_summary": executive_summary,
            "risk_flags": risk_flags,
            "risk_summary": {
                "high_count": sum(1 for r in risk_flags if r["severity"] == "HIGH"),
                "medium_count": sum(1 for r in risk_flags if r["severity"] == "MEDIUM"),
                "low_count": sum(1 for r in risk_flags if r["severity"] == "LOW"),
                "total_risks": len(risk_flags),
            },
            "entities": entities,
            "topic_distribution": topic_distribution,
        }

    def _extract_risk_flags(self, text: str) -> List[Dict[str, Any]]:
        flags = []
        sentences = re.split(r"(?<=[.!?\n])\s+", text)

        for sentence in sentences:
            s_clean = sentence.strip()
            if len(s_clean) < 20 or len(s_clean) > 500:
                continue

            for risk_def in RISK_PATTERNS:
                for pat in risk_def["patterns"]:
                    match = re.search(pat, s_clean, re.IGNORECASE)
                    if match:
                        matched_term = match.group(0)
                        flags.append({
                            "category": risk_def["category"],
                            "severity": risk_def["severity"],
                            "matched_term": matched_term.title(),
                            "excerpt": s_clean,
                            "description": risk_def["description"],
                        })
                        break  # Match once per sentence per category

        # Limit to top 10 unique flags
        unique_flags = []
        seen = set()
        for f in flags:
            key = (f["category"], f["matched_term"].lower())
            if key not in seen:
                seen.add(key)
                unique_flags.append(f)

        return unique_flags[:10]

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []

        # 1. Money & Financial Amounts
        money_matches = re.findall(r"(\$|€|₹|£|USD|INR|EUR)\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|trillion|M|B|k))?", text, re.IGNORECASE)
        money_full = re.findall(r"(?:[\$€₹£]\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?(?:million|billion|trillion|M|B|k))?|\b\d+(?:,\d{3})*(?:\.\d+)?\s?(?:USD|INR|EUR|dollars|rupees)\b)", text, re.IGNORECASE)
        for m in set(money_full[:15]):
            if len(m.strip()) > 1:
                entities.append({"entity": m.strip(), "type": "MONEY", "label": "Financial Amount"})

        # 2. Dates & Fiscal Periods
        dates = re.findall(r"\b(?:Q[1-4]|FY\s?\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b", text, re.IGNORECASE)
        for d in set(dates[:15]):
            entities.append({"entity": d.strip(), "type": "DATE", "label": "Fiscal Period / Date"})

        # 3. Organizations & Technical Symbols
        org_pattern = re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s+(?:Inc|LLC|Corp|Ltd|Technologies|Systems|Solutions|Group|Bank|AI|LLM|FastAPI|PostgreSQL)))\b", text)
        for org in set(org_pattern[:15]):
            entities.append({"entity": org.strip(), "type": "ORG", "label": "Organization / Entity"})

        # 4. Roles / People
        roles = re.findall(r"\b(?:Chief\s+[A-Z][a-z]+|CEO|CTO|CFO|Director|Vice\s+President|Architect|Lead\s+Engineer|Auditor|Counsel)\b", text, re.IGNORECASE)
        for r in set(roles[:10]):
            entities.append({"entity": r.strip().title(), "type": "ROLE", "label": "Executive / Role"})

        return entities[:25]

    def _generate_executive_summary(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        risks: List[Dict[str, Any]]
    ) -> List[str]:
        bullets = []

        # Bullet 1: Core Subject / Nature of Document
        first_few = text[:600].strip().replace("\n", " ")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", first_few) if len(s.strip()) > 30]
        if sentences:
            bullets.append(f"**Document Focus:** {sentences[0]}")
        else:
            bullets.append("**Document Focus:** Enterprise technical or financial reference documentation.")

        # Bullet 2: Key Metrics & Financials
        money_items = [e["entity"] for e in entities if e["type"] == "MONEY"]
        date_items = [e["entity"] for e in entities if e["type"] == "DATE"]
        if money_items or date_items:
            m_str = ", ".join(money_items[:3]) if money_items else "Standard transaction terms"
            d_str = ", ".join(date_items[:2]) if date_items else "current reporting cycle"
            bullets.append(f"**Key Metrics & Milestones:** Features figures ({m_str}) referenced across {d_str}.")
        else:
            bullets.append("**Key Metrics & Milestones:** Contains structured domain guidelines and specifications.")

        # Bullet 3: Risk & Compliance Overview
        if risks:
            top_risk = risks[0]
            bullets.append(f"**Risk Profile:** Identified {len(risks)} compliance/risk clauses, notably relating to *{top_risk['category']}* ({top_risk['matched_term']}).")
        else:
            bullets.append("**Risk Profile:** No critical liability, penalty, or compliance red flags detected.")

        return bullets

    def _compute_topic_distribution(self, words: List[str]) -> List[Dict[str, Any]]:
        clean_words = [
            w.lower() for w in words
            if len(w) > 3 and w.lower() not in STOPWORDS and not w.isdigit()
        ]
        counts = Counter(clean_words)
        top_terms = counts.most_common(12)

        return [{"topic": term.capitalize(), "frequency": count} for term, count in top_terms]


# Global Singleton Instance
insights_engine = DocumentInsightsEngine()
