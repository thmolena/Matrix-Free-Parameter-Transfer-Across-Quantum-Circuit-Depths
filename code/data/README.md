# Data provenance

The package authenticates eight files from Yinyu Ye's Stanford Gset MaxCut
collection: G1, G6, G11, G14, G18, G22, G32, and G43.

Original collection: <https://web.stanford.edu/~yyye/yyye/Gset/>

`checksums.sha256` pins every byte. The program also hashes unweighted edge
topology and proves that G6 duplicates G1 and G18 duplicates G14 after weights
are discarded. Those duplicate topologies are excluded; six sources contribute
24 deterministic ten-vertex neighborhoods each.

Verify bundled files without network access:

```bash
python code/scripts/download_data.py --verify-only
```

Without the bundled mirrors, the same script downloads each registered URL and
rejects any digest mismatch.
