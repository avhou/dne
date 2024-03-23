# de citeproc is nodig om referenties te kunnen resolven
# de link-citations metadata kan je ofwel command line, ofwel via een extension meegeven, dwz --from markdown+yaml_metadata_block als extra optie aan de pandoc binary meegeven


rm dne-report.pdf;docker run --rm --volume "$(pwd):/opt/docs" avhconsult/pandoc pandoc --filter pandoc-fignos --from markdown+smart+table_captions --citeproc --bibliography /opt/docs/dne-report.bib --csl /opt/docs/vancouver.sty -M link-citations=true dne-report.md -o dne-report.pdf;open dne-report.pdf
