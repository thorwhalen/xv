# xv.data_access

Accessing and preparing arxiv data for the xv project.

The data was found at [https://alex.macrocosm.so/download](https://alex.macrocosm.so/download)

At the point of writing this, my attempts enable `graze` to automatically confirm
download in the googledrive downloads (which, when downloading too-big files,
will tell the user it can’t scan the file and ask the user to confirm the download).

Therefore, the following files need to be downloaded manually:

* **titles**: [https://drive.google.com/file/d/1Ul5mPePtoPKHZkH5Rm6dWKAO11dG98GN/view?usp=share_link](https://drive.google.com/file/d/1Ul5mPePtoPKHZkH5Rm6dWKAO11dG98GN/view?usp=share_link)
* **abstracts**: [https://drive.google.com/file/d/1g3K-wlixFxklTSUQNZKpEgN4WNTFTPIZ/view?usp=share_link](https://drive.google.com/file/d/1g3K-wlixFxklTSUQNZKpEgN4WNTFTPIZ/view?usp=share_link)

(If those urls don’t work, perhaps they were updated: See here: [https://alex.macrocosm.so/download](https://alex.macrocosm.so/download) .)

You can then copy them over to the place graze will look for by doing:

```python
from pathlib import Path
from imbed.util import Graze
from imbed.mdat.arxiv import urls

g[urls['titles']] = Path('TITLES_DATA_LOCAL_FILEPATH').read_bytes()
g[urls['abstracts']] = Path('ABSTRACTS_DATA_LOCAL_FILEPATH').read_bytes()
```

### Functions

| [`arxiv_url`](#xv.data_access.arxiv_url)(uri[, resource, if_unparsable])   | Return the URL for the given URI translated to the specified resource.                                    |
|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `compile_arxiv_uri`(data)                                                                    |                                                                                                           |
| [`extract_arxiv_doi`](#xv.data_access.extract_arxiv_doi)(s)                        | Extracts and returns a valid arXiv-assigned DOI from the input string if present, otherwise returns None. |
| [`extract_doi`](#xv.data_access.extract_doi)(s)                              | Extracts and returns a valid DOI from the input string if present, otherwise returns None.                |
| `parse_arxiv_uri`(uri)                                                                       |                                                                                                           |
| `return_input`(x)                                                                            |                                                                                                           |
| `return_none`(x)                                                                             |                                                                                                           |

### xv.data_access.arxiv_url(uri, resource='abs', \*, if_unparsable=<function return_input>)

Return the URL for the given URI translated to the specified resource.

* **Parameters:**
  * **uri** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The URI to be parsed ({doi}, [https://arxiv.org](https://arxiv.org)/{resource}/{doi}, or [https://ar5iv.labs.arxiv.org/html](https://ar5iv.labs.arxiv.org/html)/{doi})
  * **resource** ([`Literal`](https://docs.python.org/3/library/typing.html#typing.Literal)[`'abs'`, `'pdf'`, `'format'`, `'src'`, `'cits'`, `'html'`]) – The desired resource type.
  * **if_unparsable** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Function to call if the URI cannot be parsed.
    If None, it will raise a ValueError.
* **Returns:**
  The compiled URL for the specified resource.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

```pycon
>>> arxiv_url('https://arxiv.org/abs/10.3233', 'pdf')
'https://arxiv.org/pdf/10.3233'
>>> arxiv_url('https://arxiv.org/abs/10.3233', 'html')
'https://ar5iv.labs.arxiv.org/html/10.3233'
>>> arxiv_url('10.3233', 'abs')
'10.3233'
```

### xv.data_access.extract_arxiv_doi(s)

Extracts and returns a valid arXiv-assigned DOI from the input string if present,
otherwise returns None. Supports DOIs of the form:

> - “10.48550/arXiv.2202.01037”
> - “doi:10.48550/arXiv.2202.01037”
> - “doi://10.48550/arXiv.2202.01037”

The DOI always uses the prefix 10.48550 and the base arXiv identifier (no version).

```pycon
>>> extract_arxiv_doi("10.48550/arXiv.2505.07987")
'10.48550/arXiv.2505.07987'
>>> extract_arxiv_doi("doi:10.48550/arXiv.2302.11894")
'10.48550/arXiv.2302.11894'
>>> extract_arxiv_doi("doi://10.48550/arXiv.1234.56789")
'10.48550/arXiv.1234.56789'
>>> extract_arxiv_doi("10.48550/arXiv.2202.01037v2") is None
True
>>> extract_arxiv_doi("arXiv:2202.01037") is None
True
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)

### xv.data_access.extract_doi(s)

Extracts and returns a valid DOI from the input string if present,
otherwise returns None. Supports strings of the form:

> - “10.1234/abcd”
> - “doi:10.1234/abcd”
> - “doi://10.1234/abcd”
* **Parameters:**
  **s** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Input string potentially containing a DOI.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)
* **Returns:**
  The DOI without any protocol prefix, or None if not a valid DOI.

```pycon
>>> extract_doi("10.1000/182")
'10.1000/182'
>>> extract_doi("doi:10.1234/ABC-123")
'10.1234/ABC-123'
>>> extract_doi("random string") is None
True
```
