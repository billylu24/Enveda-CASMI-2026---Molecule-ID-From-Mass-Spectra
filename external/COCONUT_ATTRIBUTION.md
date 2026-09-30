# COCONUT structure data attribution

Source: COCONUT — Collection of Open Natural Products, Steinbeck Lab and contributors.
Website: https://coconut.naturalproducts.net/
Release: September 2026, official CSV Lite.
Downloaded from https://coconut.s3.uni-jena.de/prod/downloads/2026-09/coconut_csv_lite-09-2026.zip

Data license: Creative Commons Attribution 4.0 International (CC BY 4.0).
Official notice: https://steinbeck-lab.github.io/coconut/license.html
License: https://creativecommons.org/licenses/by/4.0/

Modifications: selected identifier, canonical SMILES, exact molecular weight,
InChIKey, formula and annotation-level columns; converted to Parquet;
excluded rows lacking valid structure text or finite positive exact mass.
Downstream inference may filter by precursor mass and canonicalize molecular
tautomers with RDKit. COCONUT contributors do not endorse these predictions.

Retain this attribution with any redistribution of the derived catalog.
