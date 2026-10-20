# SmartBin Classifier Notes

Catalogue: top 15 products by frequency. 20+ products would leave the rarer ones with too few photos to learn from and split.

Every selected bin contains at least one catalogue product. The model has not seen bins with none of them, so "empty bin" behaviour is untested.

Catalogue products almost never share a bin (2,006 appearances in 2,000 bins), so nearly every bin has one catalogue product. Multi-item PASS cases can't be built from real bins.

Test orders: PASS = one catalogue item present in the bin with its recorded quantity. FAIL = add an absent item, or request more than the recorded quantity. Reason: catalogue products almost never share a bin (1,994 of 2,000 bins have exactly one). Approved by Satwik.

Failure analysis (ResNet-18, test set): undercounting identical or hidden units (AuYou, Great Lakes), near misses at cut-off 0.5 (Bluedio 0.46, Microfiber 0.49), complete miss from occlusion or label noise (RTIC), overcount false PASS (Lifetime Warranty, raw 1.53).