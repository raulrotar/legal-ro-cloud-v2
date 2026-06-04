# Corpus Gaps — Missing Gazette Issues

Questions Q16 and Q17 in the cloud QA set cannot be answered because the
relevant gazettes were never ingested. `laws/` currently covers only:
1989 / 2007 / 2017 / 2026.

## Missing issues

| Act | Expected gazette | Why needed |
|---|---|---|
| HG nr. 1.908/2006 (Zona liberă Galați) | Late-2006 Monitorul Oficial | Q16, Q17 — who signed and founding HG |
| HG nr. 190/1994 (înființare Zona liberă Galați) | MO nr. 120/13.V.1994 | Referenced by HG 1908/2006 |

## Action required

Download the relevant gazette PDFs from `monitoruloficial.ro` and name them
following the convention:

```
MO_PI_{issue_number}_{YYYY-MM-DD}.pdf
```

Place under `laws/{year}/{month}/{day}/`, then run:

```bash
uv run legalro-process extract --root laws/2006/ --out out/ --config config/cloud.yaml
uv run legalro-process load --root out/ --mongo "$MONGODB_URI"
```

Until these issues are ingested, Q16 and Q17 will always return
"Informația solicitată nu se regăsește în documentele furnizate."
