Source Data for "PubMed-wide Classification and Disease-level Mapping of Peer-reviewed Literature for
Genetic Developmental Disorders in Gene2Phenotype" (Yates et al., Nature Communications NCOMMS-26-018174A-Z).

One file per display item, numbered as in the revised manuscript.

Fig2_pipeline_counts.csv                     counts shown at each stage of Figure 2 (unique PMIDs; final map in mappings)
Fig3_papers_per_DDG2P_entry.csv              automated (LitDD) and manually curated paper counts per DDG2P entry (n = 2,861)
Fig4_records_per_gene_by_search_strategy.csv records returned per DDG2P gene (n = 2,552) by each simulated search and by LitDD
Fig6_map_coordinates_and_cluster_labels.csv  2D map coordinates, HDBSCAN cluster (-1 = unclustered), the cluster's Mondo
                                             label and the label displayed (30 most frequent; MONDO:OTHER; NOISE) per paper (n = 72,939)
Table1.csv - Table4.csv                      the values of manuscript Tables 1-4 (Table3: titles in **bold** are in the
                                             existing manual curation set; Table2: NA = not applicable)
SupplementaryFig1_prevalence_ladder.csv      held-out F1, corpus positive rate and external recall per training arm

Figures 1 and 2 are schematics and Figure 5 is a screenshot of the G2P website; Figure 2's counts are given above.
Supplementary Tables present their values directly in the Supplementary Information.
Files are UTF-8 CSV with a byte-order mark, so that spreadsheet software shows symbols such as ≥ and – correctly.
