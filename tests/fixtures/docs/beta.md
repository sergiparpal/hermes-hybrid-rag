A markdown file with no level-two headings.

It exists to force extract_md to fall back to extract_txt: paragraph-group
parents instead of section parents. The first paragraph and this paragraph
should land in a single paragraph-group parent given their small sizes.

Another paragraph here, separated by a blank line. The fallback path packs
paragraphs greedily until the target character count is reached, so a tiny
file like this one becomes exactly one parent.
