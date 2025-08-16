#!/usr/bin/env python3
"""
Process Wikipedia XML dump to clean JSON files for retrieval.
Replaces WikiExtractor with modern mwxml-based processing.
"""

import argparse
import bz2

import json
import mwxml
import os
import re


def clean_text(text):
    """Clean Wikipedia markup from article text with aggressive cleaning."""
    if not text:
        return ""

    # Remove HTML comments first
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove references and citations
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/?>", "", text)

    # Remove ALL HTML/XML tags
    text = re.sub(r"<[^>]+>", "", text)

    # More aggressive template removal
    def remove_templates(text):
        while "{{" in text:
            start = text.find("{{")
            if start == -1:
                break
            brace_count = 0
            i = start
            while i < len(text):
                if text[i : i + 2] == "{{":
                    brace_count += 1
                    i += 2
                elif text[i : i + 2] == "}}":
                    brace_count -= 1
                    i += 2
                    if brace_count == 0:
                        text = text[:start] + text[i:]
                        break
                else:
                    i += 1
            else:
                # No closing braces found, remove from start to end
                text = text[:start]
                break
        return text

    text = remove_templates(text)

    # Remove table markup completely
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)

    # Remove thumb|, left|, right|, etc. image positioning markup
    text = re.sub(
        r"\b(thumb|left|right|center|upright|frame)\|[^|]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\|[^|]*\|[^|]*px\|", "", text)  # Image size markup

    # Remove file and image references completely
    text = re.sub(r"\[\[(File|Image):[^\]]+\]\]", "", text, flags=re.IGNORECASE)

    # Remove categories
    text = re.sub(r"\[\[Category:[^\]]+\]\]", "", text, flags=re.IGNORECASE)

    # Clean wiki links - keep only the display text
    text = re.sub(r"\[\[([^|]+\|)?([^\]]+)\]\]", r"\2", text)

    # Remove external links
    text = re.sub(r"\[[^\] ]+ ([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[([^\] ]+)\]", "", text)
    text = re.sub(r"https?://[^\s]*", "", text)

    # Remove section headers
    text = re.sub(r"^=+[^=]*=+$", "", text, flags=re.MULTILINE)

    # Remove wiki markup for formatting
    text = re.sub(r"'''([^']+)'''", r"\1", text)  # Bold
    text = re.sub(r"''([^']+)''", r"\1", text)  # Italic

    # Remove citations and reference markers
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[clarification needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[when\?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[who\?\]", "", text, flags=re.IGNORECASE)

    # Remove table syntax remnants aggressively
    text = re.sub(r"\|\+.*?\|-", "", text, flags=re.DOTALL)
    text = re.sub(r"\|-[^|]*", "", text)
    text = re.sub(r"\![^|]*\|", "", text)
    text = re.sub(r"\|[^|]*\|\|", "", text)
    text = re.sub(r'class="[^"]*"', "", text)
    text = re.sub(r'style="[^"]*"', "", text)
    text = re.sub(r'scope="[^"]*"', "", text)
    text = re.sub(r'width="[^"]*"', "", text)
    text = re.sub(r'cellspacing="[^"]*"', "", text)
    text = re.sub(r'cellpadding="[^"]*"', "", text)

    # Remove remaining template-like structures
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"\{[^}]*\}", "", text)

    # Remove pipe symbols that are markup remnants
    text = re.sub(r"\|\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|", "", text, flags=re.MULTILINE)

    # Clean up multiple whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", text)  # Multiple newlines to double
    text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces to single
    text = re.sub(r"^\s+", "", text, flags=re.MULTILINE)  # Leading spaces

    # Remove lines that are clearly markup remnants
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        line = line.strip()
        # Skip lines that are mostly markup or very short
        if (
            line
            and not line.startswith("|")
            and not line.startswith("!")
            and not line.startswith("=")
            and not re.match(r"^[|\-\s]*$", line)
            and not re.match(
                r"^(thumb|left|right|center|upright|frame)", line, re.IGNORECASE
            )
            and len(line) > 15
        ):  # Skip very short lines that might be markup
            clean_lines.append(line)

    text = "\n".join(clean_lines)
    text = text.strip()

    return text


def process_dump(dump_file: str, output_dir: str, max_articles: int = None) -> None:
    """
    Process Wikipedia XML dump to JSON files.

    Args:
        dump_file: Path to the .xml.bz2 dump file
        output_dir: Output directory for JSON files
        max_articles: Maximum number of articles to process (for testing)
    """
    print(f"Processing Wikipedia dump: {dump_file}")
    print(f"Output directory: {output_dir}")

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)

    # Open the bz2 file
    with bz2.open(dump_file, "rt", encoding="utf-8") as f:
        dump = mwxml.Dump.from_file(f)

        articles_processed = 0
        articles_per_file = 1000
        file_count = 0
        current_file = None

        for page in dump:
            # Skip redirects and non-main namespace pages
            if page.redirect is not None:
                continue
            if page.namespace != 0:  # Only main namespace (articles)
                continue

            # Get the latest revision
            if not page:
                continue

            revision = None
            for rev in page:
                revision = rev
                break  # Get the first (latest) revision

            if not revision or not revision.text:
                continue

            # Clean the article text
            clean_article_text = clean_text(revision.text)

            # Skip very short articles
            if len(clean_article_text) < 500:
                continue

            # Create article record
            article = {
                "title": page.title,
                "text": clean_article_text,
                "url": f"https://en.wikipedia.org/wiki/{page.title.replace(' ', '_')}",
                "id": page.id,
            }

            # Open new file if needed
            if articles_processed % articles_per_file == 0:
                if current_file:
                    current_file.close()

                # Create subdirectory
                subdir = f"A{file_count // 100:02d}"
                subdir_path = os.path.join(output_dir, subdir)
                os.makedirs(subdir_path, exist_ok=True)

                # Open new file
                filename = f"wiki_{file_count % 100:02d}.json"
                filepath = os.path.join(subdir_path, filename)
                current_file = open(filepath, "w", encoding="utf-8")
                file_count += 1

            # Write article as JSON line
            current_file.write(json.dumps(article, ensure_ascii=False) + "\n")

            articles_processed += 1

            if articles_processed % 1000 == 0:
                print(f"Processed {articles_processed} articles...")

            # Stop if we've reached the limit
            if max_articles and articles_processed >= max_articles:
                break

        if current_file:
            current_file.close()

    print(f"Processing complete! {articles_processed} articles written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Process Wikipedia XML dump to JSON")
    parser.add_argument("dump_file", help="Path to the .xml.bz2 dump file")
    parser.add_argument("output_dir", help="Output directory for JSON files")
    parser.add_argument(
        "--max-articles", type=int, help="Maximum number of articles to process"
    )

    args = parser.parse_args()

    if not os.path.exists(args.dump_file):
        print(f"Error: Dump file not found: {args.dump_file}")
        return 1

    process_dump(args.dump_file, args.output_dir, args.max_articles)
    return 0


if __name__ == "__main__":
    exit(main())
