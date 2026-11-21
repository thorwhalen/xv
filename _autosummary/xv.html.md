# xv

Access to arxiv resources

```pycon
>>> from xv import raw_sources
>>> list(raw_sources)
['titles', 'abstracts']
>>> from xv import sources  # raw store + wrapper. See parquet_codec code.
>>> titles_tables = sources['titles']
abstract_tables = sources['abstracts']
>>> print(list(titles_tables))
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
>>> titles_df = titles_tables[1]
```

etc.

### Modules

| [`data_access`](xv.data_access.html.md#module-xv.data_access)   | Accessing and preparing arxiv data for the xv project.   |
|--------------------------------------------------------------------------------------|----------------------------------------------------------|
| [`util`](xv.util.html.md#module-xv.util)                 | Utils                                                    |
