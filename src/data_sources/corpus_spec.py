"""The demo corpus definition: exactly which drugs and topics we ingest.

Kept as data (not buried in a script) so the corpus is reproducible and so the
evaluation set in Phase 9 can be written against a known, fixed set of documents.

Drug selection is deliberate: the list covers several well-documented interaction
pairs (warfarin/aspirin, simvastatin/clarithromycin, sertraline/tramadol) plus a
control pair with no established interaction, which Phase 7 needs in order to test
the "no known interaction found" path rather than only the positive path.
"""

# FDA labels -- generic names.
DRUGS = [
    "metformin", "warfarin", "aspirin", "ibuprofen", "lisinopril",
    "atorvastatin", "simvastatin", "clarithromycin", "amoxicillin",
    "omeprazole", "levothyroxine", "metoprolol", "sertraline", "tramadol",
    "amlodipine", "clopidogrel", "prednisone", "furosemide", "gabapentin",
    "ciprofloxacin",
]

# Interaction pairs used by Phase 7 tests. `expected` documents what the corpus
# should support -- not a clinical assertion by this system.
INTERACTION_PAIRS = [
    ("warfarin", "aspirin", "interaction_expected"),
    ("simvastatin", "clarithromycin", "interaction_expected"),
    ("sertraline", "tramadol", "interaction_expected"),
    ("warfarin", "ciprofloxacin", "interaction_expected"),
    ("metformin", "levothyroxine", "no_major_interaction_expected"),
    ("amoxicillin", "gabapentin", "no_major_interaction_expected"),
]

# PubMed queries -- disease / symptom / treatment literature.
PUBMED_QUERIES = [
    "type 2 diabetes mellitus treatment guidelines",
    "metformin adverse effects review",
    "hypertension management first line therapy",
    "warfarin drug interactions bleeding risk",
    "statin associated muscle symptoms",
    "community acquired pneumonia antibiotic therapy",
    "asthma inhaled corticosteroid management",
    "hypothyroidism levothyroxine dosing",
    "major depressive disorder SSRI treatment",
    "serotonin syndrome clinical features",
    "chronic kidney disease staging management",
    "atrial fibrillation anticoagulation stroke prevention",
]

# PMC open-access full-text queries. The `open access[filter]` clause matters:
# without it, efetch returns metadata-only stubs for most hits.
PMC_QUERIES = [
    "metformin mechanism of action review open access[filter]",
    "drug drug interaction pharmacokinetics review open access[filter]",
    "hypertension guideline review open access[filter]",
]

# MedlinePlus health topics (plain-language disease content). Every English topic is
# kept: MedlinePlus has no single "diseases" group -- conditions are spread across
# Infections, Blood/Heart, Brain and Nerves, etc. -- so a group filter would be
# arbitrary. Set MEDLINEPLUS_GROUPS to a set of group names to narrow it.
MEDLINEPLUS_LANGUAGE = "English"
MEDLINEPLUS_GROUPS: set[str] | None = None
