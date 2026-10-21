Most of the previous formatting fixes are now correct. Do not change them. Fix only the tables below. Do not change any table content, caption, number, heading, figure, citation or body text.

1. SINGLE LINE SPACING INSIDE TABLES
The tables inherit the document's 1.5 line spacing, so Table 4.4 now runs over three pages (pages 54-56) and Table 4.5 over two. Typeset the inside of every table (Tables 3.1-3.3 and 4.1-4.10, both table and longtable environments) with single line spacing, as in the sample template, while keeping 1.5 spacing in the body text. For example, wrap each table body in \begin{singlespace} ... \end{singlespace} (setspace package), or put \renewcommand{\arraystretch}{1.15}\setstretch{1.0} at the start of each table. Use \small for the cell text of Tables 4.4 and 4.5, and \footnotesize for their file-path cells in \texttt.

2. TABLES THAT STILL SPLIT
After step 1, these tables should fit on one page. Make them normal floating tables (table environment, [htbp], \centering), not longtable, so they never split:
- Table 4.1 (Source Datasets Used in the Project): currently split across pages 58-59.
- Table 4.9 (Linear Probe (B-1) and Control Reasoning Transformer (B-3)): currently leaves only its last two rows (2-lead and 1-lead) on page 70.
If Table 4.4 or Table 4.5 still cannot fit on one page after step 1, keep it as a longtable but make sure the continued part has at least four rows (start the table on a new page if necessary).

3. "Source:" LINES
After Table 4.9 (and similar), the word "Source:" sits alone on one line and the file path on the next. Put each source line in \small, and let long \texttt paths break (e.g. use \path{...} from the url package or allow breaks after "/"), so "Source:" and the path start on the same line.

FINAL CHECKS (report back)
1. Table 4.4 takes at most two pages, and Tables 4.1 and 4.9 are not split.
2. No continued part of a table has fewer than four rows.
3. No "??" after compiling; the List of Tables is unchanged (3.1-3.3, 4.1-4.10).
4. Report the new page count. It should be below the current 91.
