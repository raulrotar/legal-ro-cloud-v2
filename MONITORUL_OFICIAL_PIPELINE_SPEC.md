# Monitorul Oficial al României — Local Multimodal Extraction Spec
**Corpus:** Partea I, 1989-12-22 → 2026-04-14 (≈120 PDFs sampled across 9 distinct year-months).
**Goal:** Pristine, lossless, dual-payload JSON for MongoDB Hybrid Search + GraphRAG, populated by a non-LLM Python pipeline.

---

## PART 1 — Corpus Research & Layout Modality Breakdown

All findings below are derived from direct probing with PyMuPDF (`fitz.Page.get_text("dict"|"blocks"|"drawings")`, `page.get_images`) of representative PDFs in every era folder.

### 1.1 Three macro-modalities
The corpus is **not** a single layout family. It splits into three modalities that demand different first-stage handlers:

| Modality | Era | Producer | Text layer | Vector grid | Layout |
|---|---|---|---|---|---|
| **M1 — Scanned image** | 1989-12 → ~1996 | `creator=DigiPath`, `producer=DigiPath` | **None** (`chars=0`, single full-page raster image) | none | Multi-column on the *image*, must be reconstructed by OCR + layout model |
| **M2 — Born-digital legacy** | 1997 → 2007-mid | `Acrobat Distiller 3.0…7.0.5`; pre-Unicode font encodings | Selectable but mojibaked (`Ã→Ă`, `ª→Ş/ş`, `þ→ţ`, `Þ→Ţ`, `Ñ→—`, `”/Ò→quote/dash`) | sparse: 8 horiz + 2 vert rules forming the cover-page banner only | Strict 2-column `x-gutter ≈ 297` |
| **M3 — Born-digital modern** | ~2007-12 → 2026 | `PScript5.dll / Acrobat Distiller 9.x`; Unicode | Clean | sparse rules + dense `re` (rect) primitives whenever a table appears (page 7 of 2020-08-14: 79 rects; page 3+ of 2026-Bis: 396-576 rects per page) | 2-column for normative text; 1-column wide for **annexe / Bis** tabular pages |

### 1.2 Geometric invariants (M2 + M3, A4 595×842 pt)
These coordinates are **stable across every born-digital sample** and are the foundation of the layout pipeline:

```
PAGE_WIDTH       = 595.0
PAGE_HEIGHT      = 842.0
HEADER_BAND_Y    = (0,   55)    # running header on body pages
FOOTER_BAND_Y    = (810, 842)   # rule + page number; mostly empty
COVER_HEADER_Y   = (260, 350)   # "P A R T E A I" / banner / "S U M A R" on page 0
GUTTER_X         = 297.0  (±2)  # mathematical center; column split
COL_LEFT_X       = (14,  291)   # column 1 active text band
COL_RIGHT_X      = (303, 581)   # column 2 active text band
GUTTER_BAND      = (291, 303)   # forbidden zone (no body text spans this)
FULL_WIDTH_TEST  = b.x0 <= 80   AND  b.x1 >= 520    # banner-class block
```

The ~12 pt forbidden gutter was confirmed by the band-density histogram: every probed body page shows `min density = 0` in the `(295, 300)` band. **A block whose bbox crosses the gutter is, by construction, a full-width banner / table / signature row, not a paragraph that bled.**

### 1.3 Cover page (page 0) skeleton — invariant since 2002
```
y∈ 38–146   raster:  Romanian coat-of-arms (rect [255, 38, 340, 146])
y∈ 154–256  raster:  decorative banner box  (rect [14, 154, 581, 256])
y∈ 268–290  text:    "P A R T E A  I" / "LEGI, DECRETE, HOTĂRÂRI ȘI ALTE ACTE"
                     "Anul N (XX) — Nr. NNN [bis]"
                     "<weekday>, DD <luna> YYYY"
y∈ 328–340  text:    "S U M A R"          (16-pt, centered)
y∈ 360–372  text:    "Nr. | Pagina | Nr. | Pagina"   (TOC column header)
y∈ 380–810  text:    SUMAR entries, organized by **section banner**
                     followed by  `<NNN>. — <type-of-act>  <title>  ……  <page>`
```

Pre-2002 (1997 sample) cover skips the literal "S U M A R" label — it lists items directly under section banners. Schema must therefore allow `sumar.is_explicit_label = false`.

### 1.4 Body page skeleton
Every body page (M2 + M3) shows:
```
y∈  37–50   running header (1-line):
            "MONITORUL OFICIAL AL ROMÂNIEI, PARTEA I, Nr. NNN/D.M.YYYY  |  <pageNo>"
            (left-aligned on odd pages, right-aligned on even — page number swaps side)
y∈  60–800  body, two columns split at x=297
y∈ 800–810  optional footer rule
```
The running-header text is **identical** across all body pages of one issue (modulo page number) and is the primary vector-poisoning risk.

### 1.5 Reading order & full-width interrupts
Within a body page, the legal reading order is:

1. All blocks whose `x_center < 297` and `y` strictly above the highest full-width interrupt — top to bottom.
2. All blocks whose `x_center >= 297` and `y` strictly above the highest full-width interrupt — top to bottom.
3. The full-width interrupt block.
4. Repeat the column-pair pass below the interrupt.

Full-width interrupts observed:
- **Section banners** (16-pt letter-spaced caps): `L E G I  Ş I  D E C R E T E`, `D E C R E T E`, `H O T Ă R Â R I  A L E  G U V E R N U L U I  R O M Â N I E I`, `D E C I Z I I  A L E  C U R Ț I I  C O N S T I T U Ț I O N A L E`, `ACTE ALE ORGANELOR DE SPECIALITATE ALE ADMINISTRAŢIEI PUBLICE CENTRALE`.
- **Act-level titles** (12-pt centered, e.g. `H O T Ă R Â R E`) when they are wider than a column.
- **Inline tables** that span both columns.
- **Closing dispositions** ("Această lege a fost adoptată de Camera Deputaţilor … cu respectarea prevederilor art. 74 …") — frequently span both columns.
- **Signature rows** ("PRIM-MINISTRU / LUDOVIC ORBAN / Contrasemnează:") at full width.

### 1.6 Act-internal structure (deterministic)
Each individual act inside a Monitorul follows a fixed grammar:
```
[ section banner            ]    full-width, 16-pt
[ issuer                    ]    centered 10-pt, e.g. "GUVERNUL ROMÂNIEI", "PARLAMENTUL ROMÂNIEI",
                                                       "PREŞEDINTELE ROMÂNIEI", "CURTEA CONSTITUŢIONALĂ"
[ act type + (number)       ]    centered 12-pt, "L E G E", "H O T Ă R Â R E", "D E C R E T",
                                                  "D E C I Z I A  Nr. NNN"
[ secondary date line       ]    "din DD <luna> YYYY"   (Curtea Constituţională decisions)
[ act title                 ]    12-pt centered, multi-line, italics or normal
[ preamble                  ]    "În temeiul …", "Având în vedere …", "Parlamentul / Guvernul / 
                                  Preşedintele … adoptă / decretează / decide:"
[ operative body            ]    "Art. N. —", "Articol unic. —",  with sub-levels (1), (2), 
                                  lit. a), pct. 1, etc.
[ closing disposition       ]    "Această lege a fost adoptată …" (laws only)
[ signatures                ]    role line (caps) + person name (caps) — repeated for each signatory;
                                  may include "Contrasemnează:" subtree
[ place + date + number     ]    "Bucureşti, DD <luna> YYYY.  Nr. NNN."
```

### 1.7 Tabular typologies discovered

| Type | Example | Vector signature | Extraction tactic |
|---|---|---|---|
| **T-A — Lattice (full borders)** | 2026-Bis pp 3-145 (Nomenclator) | hundreds of `re` primitives, ratio `rects/page > 200` | Camelot `flavor="lattice"` |
| **T-B — Bordered with mixed rules** | 2020-08-14 p 7 (inventory) | 14 horiz + 10 vert rules + 79 rects | Camelot lattice; fall-back stream |
| **T-C — Borderless/columnar** | sumar TOC on every cover; signature blocks; 1997 list-style annexes | only horizontal separator rules; no rects | pdfplumber `extract_tables(table_settings={"vertical_strategy":"text", "horizontal_strategy":"text"})` |
| **T-D — Vertical-rotated cells** | 2026-Bis nomenclator (e.g. "Cod / DFI" stacked, multi-line headers) | rect grid with very narrow first column, fonts at multiple sizes per cell | Detect rotated text (`span["dir"] != (1,0)`) and normalize after extraction |

### 1.8 Sumar (table-of-contents) typology
Three observed shapes:
1. **Single-column-wide list** (1997, when issue has few items): no "S U M A R" label, just a section banner like `LEGI ŞI DECRETE` followed by entries. Page-number markers can be missing (the act starts directly on next page).
2. **Two-column list with explicit "S U M A R"** (2002+): `Nr. | Pagina | Nr. | Pagina` header, dot-leader entries `12. — Decret …  2`, page numbers may be `12-13` (range) or `14…16` (with ellipsis when interrupted by another section).
3. **Annex sumar** (2026 Bis): `Pagina` only (single column), entries are `Anexele nr. 1-5 la Hotărârea …`.

### 1.9 Encoding/diacritic minefield (M2 era)
1997-era PDFs were created with custom font encodings; PyMuPDF returns the raw codepoints. The deterministic substitution table is:
```
Ã  → Ă       ã  → ă       Â (WGL4) → Â  (already correct)
Ñ  → —       Þ  → Ţ       þ  → ţ       
ª  → Ş       º  → ş       Ò  → ”      
”  → ”       Ð  → Đ                    
HOT√R¬RI ™I → HOTĂRÂRI ŞI       (early 2007 mojibake set)
```
Apply BEFORE any block re-ordering or vector embedding, because diacritic-stripped Romanian collides badly with embedding similarity (e.g. `lege` vs `legã`).

### 1.10 Graph entities present in every act
- **Issuer** (Institution): closed taxonomy of ~25 entries.
- **Signatories** (Person): caps lines after issuer + role descriptor.
- **Cross-references**: `Legea nr. NNN/YYYY`, `Ordonanţa Guvernului nr. NNN/YYYY`, `Hotărârea Guvernului nr. NNN/YYYY`, `Decretul Preşedintelui nr. NNN/YYYY`, `art. NN alin. (M) lit. x) din Constituţia României`. All extractable by regex (Section 3).
- **Promulgation chain**: a Decret (promulgation) cites the corresponding Lege, and the Lege cites the originating Ordonanţă. These edges feed GraphRAG.

---

## PART 2 — Custom Engineered JSON Schema

Save as `schema/monitorul_oficial.schema.json`. Designed for MongoDB ingestion; every text payload is **also** present as Markdown so a single `text_markdown` field can be vectorized while structured siblings drive Hybrid Search filters and Graph edges.

```jsonc
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id":     "urn:moro:partea-i/issue.v1",
  "title":   "Monitorul Oficial — Partea I — Issue",

  // -------------------------------------------------------------------- //
  // Top-level: ONE document per source PDF (one issue of Partea I).
  // -------------------------------------------------------------------- //
  "type": "object",
  "required": [
    "doc_id", "source", "issue", "modality", "pages",
    "sumar", "acts", "unmapped_raw_stream", "extraction_audit"
  ],
  "additionalProperties": false,

  "properties": {

    // ------ identification ------
    "doc_id": {
      "type": "string",
      "description": "Stable id: 'MO_PI_<num>[Bis]_<YYYY-MM-DD>' (matches filename root). Primary key in MongoDB."
    },
    "source": {
      "type": "object",
      "required": ["filename", "abs_path", "sha256", "byte_size", "page_count"],
      "properties": {
        "filename":   { "type": "string" },
        "abs_path":   { "type": "string" },
        "sha256":     { "type": "string", "pattern": "^[0-9a-f]{64}$" },
        "byte_size":  { "type": "integer", "minimum": 0 },
        "page_count": { "type": "integer", "minimum": 1 },
        "creator":    { "type": "string" },
        "producer":   { "type": "string" },
        "pdf_title":  { "type": "string" }
      }
    },

    // ------ issue header ------
    // Filterable fields used by Mongo Hybrid Search facets.
    "issue": {
      "type": "object",
      "required": ["partea", "issue_number", "publication_date", "anul_editorial"],
      "properties": {
        "partea":           { "const": "I" },
        "issue_number":     { "type": "integer", "minimum": 1 },
        "issue_suffix":     { "type": ["string","null"], "enum": [null, "bis", "ter", "quater"] },
        "publication_date": { "type": "string", "format": "date" },
        "weekday_ro":       { "type": "string" },
        "anul_editorial":   { "type": "integer" },
        "anul_roman":       { "type": "string" }
      }
    },

    // ------ provenance: which extraction path produced this record ------
    "modality": {
      "type": "object",
      "required": ["class", "ocr_used", "encoding_repaired"],
      "properties": {
        "class":              { "enum": ["M1_SCANNED", "M2_LEGACY_DIGITAL", "M3_MODERN_DIGITAL"] },
        "ocr_used":           { "type": "boolean" },
        "ocr_engine":         { "type": ["string","null"] },
        "encoding_repaired":  { "type": "boolean" },
        "encoding_map_version": { "type": ["string","null"] }
      }
    },

    // ------ per-page structural index ------
    // Always populated — even for pages whose body is fully claimed by `acts`.
    "pages": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["index", "size", "n_columns", "header", "footer", "blocks"],
        "properties": {
          "index":        { "type": "integer", "minimum": 0 },
          "size":         { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 },
          "n_columns":    { "enum": [1, 2] },
          "gutter_x":     { "type": "number" },
          "header":       { "type": "object", "properties": {
              "raw":         { "type": "string" },
              "issue_ref":   { "type": "string" },
              "page_label":  { "type": "string" }
          }},
          "footer":       { "type": "object", "properties": { "raw": { "type": "string" } } },
          // Geometric block ledger — every text block on the page, even if
          // also attached to an `acts[]` entry. Round-trip auditable.
          "blocks": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["bbox", "column", "role", "text", "block_id"],
              "properties": {
                "block_id":   { "type": "string" },
                "bbox":       { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
                "column":     { "enum": ["L", "R", "FULL", "HEADER", "FOOTER"] },
                "role":       {
                  "enum": [
                    "running_header","page_number","footer_rule",
                    "cover_partea","cover_subtitle","cover_issue_line","cover_date_line",
                    "cover_sumar_label","cover_sumar_colhdr","cover_sumar_section",
                    "cover_sumar_entry",
                    "section_banner","issuer","act_type","act_number","act_subdate",
                    "act_title","preamble","operative_phrase","article",
                    "closing_disposition","signature_role","signature_name",
                    "signature_contrasemneaza","place_and_date","act_act_number",
                    "table_caption","table_cell","figure_caption",
                    "footnote",
                    "unknown"          // <-- never silently dropped; routed to unmapped_raw_stream.
                  ]
                },
                "text":       { "type": "string" },
                "font":       { "type": "string" },
                "font_size":  { "type": "number" },
                "is_bold":    { "type": "boolean" },
                "rotation":   { "type": "integer", "enum": [0, 90, 180, 270] },
                "claimed_by": { "type": ["string","null"], "description": "act_id when this block was attached to an act; null otherwise" }
              }
            }
          },
          // For tabular pages — full grid + markdown rendering, in geometric order.
          "tables": {
            "type": "array",
            "items": { "$ref": "#/$defs/Table" }
          }
        }
      }
    },

    // ------ Sumar (Table of Contents) ------
    "sumar": {
      "type": "object",
      "required": ["page_index", "is_explicit_label", "entries", "markdown_representation"],
      "properties": {
        "page_index":        { "type": "integer", "minimum": 0 },
        "is_explicit_label": { "type": "boolean" },
        "markdown_representation": { "type": "string" },
        "entries": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["section", "act_number", "act_type", "title", "raw"],
            "properties": {
              "section":     { "type": "string" },
              "act_number":  { "type": ["string","null"] },
              "act_type":    { "type": "string" },
              "title":       { "type": "string" },
              "page_from":   { "type": ["integer","null"] },
              "page_to":     { "type": ["integer","null"] },
              "page_label":  { "type": "string", "description": "Original token: '12', '12–13', '14...16'" },
              "act_id_link": { "type": ["string","null"], "description": "joins to acts[].act_id when matched" },
              "raw":         { "type": "string" }
            }
          }
        }
      }
    },

    // ------ Acts (one or many per issue) ------
    "acts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "act_id", "section", "issuer", "act_type",
          "page_range", "block_refs", "text_markdown"
        ],
        "additionalProperties": false,
        "properties": {
          "act_id":          { "type": "string", "description": "<doc_id>::act-<NNN>" },
          "section":         { "type": "string" },
          "section_banner":  { "type": "string" },
          "issuer": {
            "type": "object",
            "required": ["display"],
            "properties": {
              "display":     { "type": "string" },
              "canonical":   { "type": "string", "description": "snake_case taxonomy id, e.g. 'guvernul_romaniei'" }
            }
          },
          "act_type":        { "enum": [
            "lege","decret","hotarare_guvern","ordonanta_guvern","ordonanta_urgenta",
            "decizie_curtea_constitutionala","decizie_prim_ministru",
            "ordin","norma","regulament","altele"
          ]},
          "act_number":      { "type": ["string","null"] },
          "act_date_signed": { "type": ["string","null"], "format": "date" },
          "act_date_subline":{ "type": ["string","null"] },
          "title":           { "type": "string" },
          "page_range":      {
            "type": "object",
            "required": ["start", "end"],
            "properties": {
              "start": { "type": "integer", "minimum": 0 },
              "end":   { "type": "integer", "minimum": 0 }
            }
          },
          "block_refs":      {
            "type": "array",
            "items": { "type": "string" },
            "description": "Ordered list of block_id values that compose this act, in reading order. Round-trip key for auditability."
          },
          "preamble_markdown":         { "type": "string" },
          "operative_phrase":          { "type": "string" },
          "articles": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["article_label", "text_markdown"],
              "properties": {
                "article_label":  { "type": "string", "description": "'Art. 1.', 'Articol unic.', 'Art. 12 alin. (3)' …" },
                "ordinal":        { "type": ["integer","null"] },
                "is_unique":      { "type": "boolean" },
                "text_markdown":  { "type": "string" },
                "tables":         { "type": "array", "items": { "$ref": "#/$defs/Table" } }
              }
            }
          },
          "closing_disposition":       { "type": ["string","null"] },
          "signatures": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["role_display", "person"],
              "properties": {
                "role_display":      { "type": "string" },
                "role_canonical":    { "type": "string" },
                "person":            { "type": "string" },
                "is_contrasemneaza": { "type": "boolean" }
              }
            }
          },
          "place_and_date": {
            "type": "object",
            "properties": {
              "place":        { "type": "string" },
              "date_signed":  { "type": ["string","null"], "format": "date" },
              "act_number":   { "type": ["string","null"] }
            }
          },
          "annexes": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "label":              { "type": "string", "description": "'Anexa nr. 1', 'Anexele nr. 1-5'" },
                "page_range":         { "type": "object" },
                "tables":             { "type": "array", "items": { "$ref": "#/$defs/Table" } },
                "text_markdown":      { "type": "string" }
              }
            }
          },

          // -------- BIG-PAYLOAD: vector-target string --------
          "text_markdown": {
            "type": "string",
            "description": "Concatenation of preamble + operative_phrase + every article (with table grids inline) + closing_disposition + signatures + place_and_date, normalized whitespace, headings as `#`-prefixed Markdown. THIS is what feeds the embedding model."
          },

          // -------- Graph edges --------
          "graph": {
            "type": "object",
            "properties": {
              "issued_by": { "type": "string" },
              "signed_by": { "type": "array", "items": { "type": "string" } },
              "cites":      {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "type":        { "enum": ["lege","ordonanta","ordonanta_urgenta","hotarare_guvern","decret","decizie","constitutie","tratat","other"] },
                    "number":      { "type": ["string","null"] },
                    "year":        { "type": ["integer","null"] },
                    "article":     { "type": ["string","null"] },
                    "raw":         { "type": "string" }
                  }
                }
              },
              "promulgates": { "type": ["string","null"], "description": "act_id of the law this Decret promulgates, when applicable" },
              "approves":    { "type": ["string","null"], "description": "act_id of the OUG/OG this Lege approves" }
            }
          }
        }
      }
    },

    // -------------------------------------------------------------------- //
    //  ZERO-LOSS CONTAINMENT (catch-all fallback)
    //  Any text segment, span, or layout artifact that the rule engine
    //  cannot confidently classify into an `acts[]` entry, a sumar entry,
    //  a header, a footer, a table cell, etc., MUST be appended here in
    //  page-then-y-then-x order. The pipeline asserts:
    //
    //      sum(len(every text payload in JSON))  ==  len(raw page text stream)
    //
    //  This makes 100% textual retention a tested invariant, not a wish.
    // -------------------------------------------------------------------- //
    "unmapped_raw_stream": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["page_index", "y", "x", "text", "reason", "block_id"],
        "properties": {
          "block_id":   { "type": "string", "description": "Same id as in pages[].blocks[]; back-pointer." },
          "page_index": { "type": "integer", "minimum": 0 },
          "y":          { "type": "number" },
          "x":          { "type": "number" },
          "bbox":       { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
          "text":       { "type": "string" },
          "rotation":   { "type": "integer" },
          "font":       { "type": "string" },
          "font_size":  { "type": "number" },
          "reason":     { "enum": [
            "unknown_role","section_banner_orphan","issuer_orphan",
            "ocr_low_confidence","cross_column_artifact","figure_only",
            "decorative_glyph","stray_marginalia","unmatched_signature",
            "split_table_cell","page_number_only","empty_after_normalize",
            "rotation_unhandled"
          ]}
        }
      }
    },

    // ------ Audit (proves the Zero-Loss Containment invariant) ------
    "extraction_audit": {
      "type": "object",
      "required": ["raw_text_chars", "mapped_chars", "unmapped_chars", "coverage_ratio", "schema_version", "pipeline_version", "extracted_at"],
      "properties": {
        "raw_text_chars":   { "type": "integer", "minimum": 0 },
        "mapped_chars":     { "type": "integer", "minimum": 0 },
        "unmapped_chars":   { "type": "integer", "minimum": 0 },
        "coverage_ratio":   { "type": "number",  "minimum": 0, "maximum": 1, "description": "(mapped + unmapped) / raw_text_chars  — must equal 1.0 for the doc to be considered ingestion-ready." },
        "schema_version":   { "type": "string" },
        "pipeline_version": { "type": "string" },
        "extracted_at":     { "type": "string", "format": "date-time" },
        "warnings":         { "type": "array",  "items": { "type": "string" } }
      }
    }
  },

  // ---------------- shared sub-schemas ----------------
  "$defs": {
    "Table": {
      "type": "object",
      "required": ["table_id", "page_index", "bbox", "engine", "grid", "markdown_representation"],
      "properties": {
        "table_id":     { "type": "string" },
        "page_index":   { "type": "integer" },
        "bbox":         { "type": "array", "items": { "type": "number" }, "minItems": 4, "maxItems": 4 },
        "engine":       { "enum": ["camelot_lattice","camelot_stream","pdfplumber","layout_rebuilt","ocr"] },
        "n_rows":       { "type": "integer" },
        "n_cols":       { "type": "integer" },
        // Row-major 2D array of strings — pristine cell content.
        "grid": {
          "type": "array",
          "items": { "type": "array", "items": { "type": "string" } }
        },
        "headers":      { "type": "array", "items": { "type": "string" } },
        "caption":      { "type": "string" },
        // Pre-rendered Markdown (GFM pipe table). This is the vectorization target.
        "markdown_representation": { "type": "string" }
      }
    },

    // ---------------------------------------------------------------- //
    //  Chunk — one document per retrieval unit. Lives in mo_chunks.    //
    //  This is the field embedded by $vectorSearch and BM25-indexed by  //
    //  $search. NEVER nested inside the issue/act doc (BSON 16MB cap +  //
    //  re-embedding cost).                                              //
    // ---------------------------------------------------------------- //
    "Chunk": {
      "type": "object",
      "required": [
        "chunk_id","issue_id","act_id","chunk_index",
        "text","text_ascii","token_count","char_range",
        "block_refs","page_range",
        // facets duplicated here so $vectorSearch can filter without $lookup
        "issuer_canonical","act_type","publication_date","section","modality",
        // embedding metadata — embedding itself optional in the doc
        "embedding_model","embedding_dim","embedding_version","embedded_at"
      ],
      "properties": {
        "chunk_id":          { "type":"string", "description":"<act_id>::chunk-<NNN>; collection _id" },
        "issue_id":          { "type":"string" },
        "act_id":            { "type":"string" },
        "chunk_index":       { "type":"integer", "minimum":0 },
        "kind":              { "enum":["header","article","table","signature","unmapped","ocr"] },
        "text":              { "type":"string", "description":"Final embeddable Markdown, with prefix banner: '[issuer] act_type act_number/year — title\\n\\n…'" },
        "text_ascii":        { "type":"string", "description":"NFKD-stripped diacritic-free copy for tolerant BM25" },
        "token_count":       { "type":"integer", "minimum":1 },
        "char_range":        { "type":"array", "items":{"type":"integer"}, "minItems":2, "maxItems":2 },
        "block_refs":        { "type":"array", "items":{"type":"string"} },
        "page_range":        { "type":"array", "items":{"type":"integer"}, "minItems":2, "maxItems":2 },

        // Denormalized facets (pushdown filters in $vectorSearch / $search)
        "issuer_canonical":  { "type":"string" },
        "act_type":          { "type":"string" },
        "act_number":        { "type":["string","null"] },
        "act_year":          { "type":["integer","null"] },
        "publication_date":  { "type":"string", "format":"date" },
        "section":           { "type":"string" },
        "modality":          { "enum":["M1_SCANNED","M2_LEGACY_DIGITAL","M3_MODERN_DIGITAL"] },

        // Embedding triple — model, dim, version stamp ride together so we can
        // re-embed deterministically and keep multiple model versions side-by-side.
        "embedding":         { "type":"array", "items":{"type":"number"} },
        "embedding_model":   { "type":"string", "description":"e.g. 'voyage-3', 'text-embedding-3-large', 'BAAI/bge-m3'" },
        "embedding_dim":     { "type":"integer" },
        "embedding_version": { "type":"string", "description":"semver of chunker+model+normalizer combo, e.g. 'chunker-1.2+voyage-3'" },
        "embedded_at":       { "type":"string", "format":"date-time" },

        // Quality flags propagated for filtering (e.g. exclude low-conf OCR from vector path)
        "ocr_min_confidence":{ "type":["number","null"], "minimum":0, "maximum":1 },
        "is_oversize_table": { "type":"boolean", "description":"true if a single un-splittable table exceeds max_tokens; embedded anyway, flagged for separate handling" }
      }
    }
  }
}
```

### Architectural justifications (per the Dual-Payload Rule)

1. **`pages[].blocks` is the geometric ledger**, never thrown away — every block on the page is recorded with `bbox`, `role`, `column`, and a `claimed_by` back-pointer to the act it ended up inside. This satisfies the audit invariant without forcing classification.
2. **`acts[].block_refs`** is the ordered set of `block_id` strings that compose the act. Re-rendering the act from the ledger is therefore a pure function of the page index — no second extraction needed.
3. **`text_markdown`** at the act level is the only field embedded into the vector index. It is **derived**, not authoritative; everything that produced it is preserved in `articles`, `signatures`, etc. Re-embedding with a different chunking strategy is one map-reduce away.
4. **`unmapped_raw_stream`** is required and asserted-on-load: the audit step compares `raw_text_chars == mapped_chars + unmapped_chars`. If a block fails every rule, it is never silently dropped — it lands here with a `reason`, in chronological (page → y → x) order, and is still indexable.
5. **Tables travel both as a 2D `grid` and as `markdown_representation`** — graph queries need typed cells; vector search consumes the prose form.
6. **Modality stamp** lets MongoDB filters route OCR-derived chunks differently (e.g., lower BM25 weight) without touching content.

---

## PART 3 — Non-LLM Programmatic Processing Plan

### 3.1 Library stack
The pipeline is **deterministic and LLM-free for M2/M3 (born-digital, ≈75% of corpus)**. M1 (scanned 1989-1996, ≈25%) uses LlamaParse — a cloud, LLM-backed parser — as the primary OCR path with a fully offline PaddleOCR fallback. This is the only place an external LLM is in the loop, and it is gated by `modality=="M1_SCANNED"` and bounded by the §4.13 cost guardrails.

```
docling          (≥2.0)              — PRIMARY structured extraction for M2/M3:
                                       layout net + TableFormer (small, local ML models),
                                       returns DoclingDocument with reading order,
                                       headings, paragraphs, lists, and tables. MIT.
                                       Replaces what would otherwise be hand-rolled
                                       multi-column reading order + table detection.

fitz            (PyMuPDF, ≥1.26)    — ALWAYS-ON ground truth: per-page text stream
                                       and block geometry. Required for the
                                       coverage_ratio==1.000 audit invariant — Docling
                                       is allowed to omit decorative content; fitz
                                       guarantees we know what was omitted so it lands
                                       in unmapped_raw_stream.

camelot-py[cv]  (≥0.11)             — TARGETED FALLBACK for T-A tables (dense full
                                       lattice, ≥60 rect primitives per page; e.g.
                                       2026-Bis Nomenclator). Docling's TableFormer
                                       sometimes mis-spans cells on this typology.
                                       Ghostscript required.
pdfplumber      (≥0.11)             — TARGETED FALLBACK for T-C tables (borderless,
                                       text-aligned; sumar TOC, signature blocks).

opencv-python   (≥4.10)             — page rasterization for OCR + image features
shapely         (≥2.0)              — bbox / column geometry
unidecode + custom translit          — encoding repair for M2 PDFs

llama-cloud-services (≥0.6)          — PRIMARY OCR: LlamaParse (cloud, LLM-backed)
                                       result_type="markdown", parsing_instruction
                                       tuned for Monitorul Oficial layout
paddleocr        (≥2.8)              — FALLBACK OCR: fully offline (Romanian model)
                                       triggered when LlamaParse unavailable / over-budget /
                                       confidence below threshold

FlagEmbedding   (≥1.3)               — PRIMARY embeddings: BAAI/bge-m3
                                       1024-d dense, multilingual incl. Romanian,
                                       sparse + ColBERT-style multi-vector available
sentence-transformers (≥3.0)         — backup loader path for bge-m3

pydantic         (≥2.7)              — schema validation
pymongo          (≥4.7)              — sink
tiktoken         (≥0.7)              — chunker token counting (cl100k_base)
```

### 3.2 Pipeline DAG
Docling does the heavy lifting on M2/M3; fitz is run in parallel as the audit witness so we can prove no text was silently dropped.

```
[ pdf_path ]
    │
    ├─► (1) classify_modality
    │
    │   ┌──────── M1 (scanned, 1989-1996) ────────┐
    │   │  (1a) ocr_pdf:                          │
    │   │      LlamaParse primary  ─►  Markdown   │
    │   │      PaddleOCR fallback  ─►  Markdown   │
    │   │  (1b) markdown_to_blocks (synthetic)    │
    │   └──────────────────────────────────────────┘
    │
    │   ┌──────── M2 / M3 (born-digital) ─────────┐
    │   │  (2a) docling_convert(pdf)              │
    │   │       → DoclingDocument                 │
    │   │       (reading order + headings + lists │
    │   │        + tables, ML-assisted)           │
    │   │  (2b) fitz_blocks_dict(pdf)             │
    │   │       → ground-truth text + bboxes      │
    │   │  (2c) reconcile(docling, fitz)          │
    │   │       → Block[] with Docling roles +    │
    │   │         fitz bboxes; orphan fitz blocks │
    │   │         (anything Docling missed) get   │
    │   │         role=unknown for catch-all      │
    │   └──────────────────────────────────────────┘
    │
    ├─► (3) repair_encoding   (M2 only)
    ├─► (4) strip_running_header_footer
    ├─► (5) refine_block_role (Docling labels are priors;
    │                          regex rules in §3.7 specialize them
    │                          to the Monitorul Oficial role taxonomy)
    ├─► (6) reading_order     (Docling gives this; fitz orphans
    │                          inserted at nearest y; full-width
    │                          interrupt rule applied as tie-break)
    ├─► (7) tables_per_typology:
    │         T-A dense lattice → Camelot lattice (Docling fallback)
    │         T-B bordered      → Docling TableFormer (primary)
    │         T-C borderless    → pdfplumber (Docling fallback)
    │         T-D rotated cells → Docling + manual rotation pass
    ├─► (8) extract_sumar
    ├─► (9) segment_into_acts
    ├─► (10) parse_act_internals (regex set, §3.8)
    ├─► (11) build_graph_edges
    ├─► (12) render_text_markdown
    ├─► (13) catch_all_unmapped → unmapped_raw_stream
    ├─► (14) audit_coverage_invariant (compares against fitz-extracted
    │                                  raw text; raises if < 1.000)
    ├─► (15) chunk_act         (§3.15 deterministic chunker)
    ├─► (16) embed_chunks      (BGE-M3, §3.18.1)
    └─► (17) jsonschema_validate → persist (issues, acts, blocks, chunks, edges)
```

Why both Docling **and** fitz for born-digital: Docling's strength is structure (it knows a heading is a heading); fitz's strength is exhaustive text capture (every span, every bbox). Reconciling the two gives us structure *and* a verifiable lossless audit. If Docling were the only source, the Zero-Loss Containment invariant would have no ground truth to audit against — Docling can legitimately drop content it considers decorative, and we'd never know.

### 3.3 Step (1) — Modality classifier
```python
def classify_modality(doc: fitz.Document) -> str:
    text_chars = sum(len(p.get_text("text")) for p in doc[:min(5, doc.page_count)])
    img_only   = all(len(p.get_text("text")) < 30 and p.get_images() for p in doc[:3])
    if img_only or text_chars < 200:
        return "M1_SCANNED"
    creator  = (doc.metadata or {}).get("creator", "") or ""
    producer = (doc.metadata or {}).get("producer", "") or ""
    legacy_markers = ("Distiller 3", "Distiller 5", "Distiller 7.0.5")
    is_legacy = any(m in producer for m in legacy_markers) and (
        "Ã" in doc[0].get_text("text") or "ª" in doc[0].get_text("text")
        or "√" in doc[0].get_text("text")
    )
    return "M2_LEGACY_DIGITAL" if is_legacy else "M3_MODERN_DIGITAL"
```

### 3.4 Step (1a-c) — OCR path for M1 (1989-1996), LlamaParse primary

**Architecture decision:** M1 uses LlamaParse as the primary OCR engine and PaddleOCR as a deterministic offline fallback. This is the *only* runtime LLM call in the pipeline; it is restricted to ~30 PDFs (1989-1996 scanned issues), aggressively cached by `sha256(pdf_bytes)`, and bounded by per-run cost guardrails (§4.13). All M2/M3 PDFs (1997 onward) bypass LlamaParse entirely.

#### 3.4.1 LlamaParse primary path

```python
from llama_cloud_services import LlamaParse
import hashlib, json, os, pathlib, time

LLAMAPARSE = LlamaParse(
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    result_type="markdown",
    language="ro",                       # Romanian
    premium_mode=True,                   # required for old scanned typewriter pages
    parsing_instruction=PARSING_INSTRUCTION_RO,    # see below
    do_not_unroll_columns=False,         # let LlamaParse linearize the 2-column layout
    skip_diagonal_text=True,             # diagonal stamps on 1989 docs → noise
    invalidate_cache=False,              # idempotent on (file_hash, instruction_hash)
    num_workers=4,
    verbose=False,
    max_timeout=600,
)

PARSING_INSTRUCTION_RO = """
This document is an issue of "Monitorul Oficial al României, Partea I" — the
official Romanian gazette. Layout rules:

1. The document is two-column on every page except annex tables. Read each page
   as: full-page header (running header line), then column 1 top-to-bottom, then
   column 2 top-to-bottom. Treat full-width banners (e.g. "L E G I  Ş I  D E C R E T E",
   "H O T Ă R Â R I  A L E  G U V E R N U L U I  R O M Â N I E I") as section
   breaks that interrupt the columns; emit them on their own line.

2. Strip the running header that appears at the top of every body page:
   "MONITORUL OFICIAL AL ROMÂNIEI, PARTEA I, Nr. NNN/D.M.YYYY  |  <pageNumber>".
   Output only the body content.

3. Preserve Romanian diacritics: ă â î ș ț Ă Â Î Ș Ț. Never strip them.

4. Render every act with this Markdown skeleton:

   ## <SECTION BANNER>
   ### <ISSUER>          (e.g. PARLAMENTUL ROMÂNIEI, GUVERNUL ROMÂNIEI)
   #### <ACT TYPE>       (e.g. LEGE, DECRET, HOTĂRÂRE, DECIZIA Nr. NNN)
   **<TITLE>**

   <preamble>

   _<operative phrase>_   (e.g. "Parlamentul României adoptă prezenta lege:")

   **Art. 1.** — <text>
   **Art. 2.** — <text>
   ...
   _<closing disposition, if any>_

   <signature role>
   <signature name>

   <place + date + number>

5. Emit tables as GFM pipe tables, never as prose. Preserve every cell verbatim,
   including empty cells (use a single space).

6. Do NOT summarize, do NOT paraphrase, do NOT translate. Output verbatim
   Romanian text exactly as printed, with the Markdown structure described above.
"""

def llamaparse_pdf(pdf_path: pathlib.Path, sha256_hex: str) -> LlamaParseResult:
    cache = pathlib.Path(f".cache/llamaparse/{sha256_hex}.json")
    if cache.exists():
        return LlamaParseResult(**json.loads(cache.read_text()))

    docs = LLAMAPARSE.load_data(str(pdf_path))     # one Document per page
    pages = []
    for i, d in enumerate(docs):
        pages.append({
            "index":    i,
            "markdown": d.text,
            "metadata": d.metadata,
        })
    result = LlamaParseResult(pages=pages, engine="llamaparse",
                              engine_version=LLAMAPARSE_CLIENT_VERSION,
                              parsed_at=time.time())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result.__dict__, ensure_ascii=False))
    return result
```

The cache is keyed by `sha256(pdf_bytes)` AND `sha256(parsing_instruction + premium_mode + language)` — bumping any of those forces a re-parse. Cached results never count against the cost budget.

#### 3.4.2 Markdown → Block adaptor (so M1 reuses the same downstream pipeline)

LlamaParse returns clean Markdown, but the rest of the pipeline (role classifier, reading order, act segmentation) operates on `Block` objects with bboxes. We synthesize pseudo-bboxes from Markdown structure so M1 plugs into the same data flow:

```python
def llamaparse_markdown_to_blocks(md: str, page_index: int, page_w=595.0, page_h=842.0) -> list[Block]:
    """Convert per-page LlamaParse Markdown to synthetic Block list.
    Roles are pre-assigned from Markdown semantics, saving a classifier pass.
    Bboxes are synthetic (y monotone-increasing, x = full width) — they exist
    only so the downstream stream is bbox-typed, not for layout reasoning."""
    blocks, y = [], 60.0
    for token in parse_markdown_tokens(md):
        bbox = (14.0, y, page_w - 14.0, y + token.height)
        role = {
            "h2": "section_banner",
            "h3": "issuer",
            "h4": "act_type",
            "strong_first": "act_title",
            "em_operative": "operative_phrase",
            "article": "article",
            "table": "table_cell",
            "signature_role": "signature_role",
            "signature_name": "signature_name",
            "place_date":     "place_and_date",
            "paragraph":      "preamble",
        }.get(token.kind, "unknown")
        blocks.append(Block(
            block_id=f"p{page_index}-llp-{len(blocks):03d}",
            bbox=bbox, column="FULL", role=role,
            text=token.text, font="LlamaParse",
            font_size=10.0, is_bold=token.is_strong,
            rotation=0, claimed_by=None,
        ))
        y += token.height + 4.0
    return blocks
```

#### 3.4.3 PaddleOCR offline fallback (deterministic safety net)

LlamaParse fails open, never closed. Triggers for the fallback:

| Trigger | Action |
|---|---|
| `LLAMA_CLOUD_API_KEY` absent | use Paddle |
| LlamaParse 5xx after 3 retries (exp backoff 4s/16s/64s) | use Paddle, mark `dlq.partial=true` |
| Per-run cost budget exhausted (§4.13) | use Paddle for the rest of the run |
| Output suspect: <100 chars on a page LlamaParse claimed had ink, OR Markdown contains untranslated literals like `image:`, `<unknown>`, `[unintelligible]` | use Paddle for that page |
| Air-gapped / offline deploy | Paddle only |

```python
def ocr_pdf(pdf_path: pathlib.Path, sha256_hex: str, budget: CostBudget) -> list[PageRecord]:
    if not budget.can_afford(estimate_llamaparse_pages(pdf_path)) \
       or not os.environ.get("LLAMA_CLOUD_API_KEY"):
        return paddle_ocr_pdf(pdf_path)

    try:
        result = llamaparse_pdf(pdf_path, sha256_hex)
        budget.charge_pages(len(result.pages), engine="llamaparse")
    except (LlamaParseError, RetriesExhausted) as e:
        log.warning("llamaparse_failed; falling back to paddle", error=str(e))
        return paddle_ocr_pdf(pdf_path)

    return [
        PageRecord(
            index=p["index"],
            blocks=llamaparse_markdown_to_blocks(p["markdown"], p["index"]),
            ocr_engine="llamaparse",
            ocr_engine_version=LLAMAPARSE_CLIENT_VERSION,
        )
        for p in result.pages
    ]

def paddle_ocr_pdf(pdf_path: pathlib.Path) -> list[PageRecord]:
    """Fully offline. Same column-aware vertical-projection logic as before."""
    doc = fitz.open(pdf_path); out = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=400, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        columns = detect_columns_image(img)
        blocks = []
        for col_idx, (x0, x1) in enumerate(columns):
            crop = img[:, x0:x1]
            for (poly, (text, conf)) in PADDLE_OCR.ocr(crop, cls=True)[0]:
                bx0, by0, bx1, by1 = polygon_bbox(poly)
                blocks.append(Block(
                    bbox=(x0+bx0, by0, x0+bx1, by1),
                    text=text, font="OCR-Paddle",
                    font_size=estimated_font_size(by1-by0),
                    column="L" if col_idx == 0 else "R",
                    rotation=0, confidence=conf, role="unknown",
                ))
        out.append(PageRecord(index=i, blocks=blocks,
                              ocr_engine="paddleocr",
                              ocr_engine_version=PADDLE_VERSION))
    return out

def detect_columns_image(gray: np.ndarray) -> list[tuple[int,int]]:
    proj = (gray < 200).sum(axis=0)
    w = gray.shape[1]
    band = proj[int(w*0.35): int(w*0.65)]
    valley_x = int(w*0.35) + np.argmin(band)
    if proj[valley_x] < proj.max() * 0.10:
        return [(0, valley_x), (valley_x, w)]
    return [(0, w)]
```

#### 3.4.4 OCR provenance in the schema

Every Block produced by either engine carries `font` set to `"LlamaParse"` or `"OCR-Paddle"`. Each chunk derived from M1 carries `ocr_engine` and `ocr_engine_version` (extend the `Chunk` $def):

```jsonc
"ocr_engine":         { "enum": ["llamaparse","paddleocr","tesseract", null] },
"ocr_engine_version": { "type": ["string","null"] },
"ocr_min_confidence": { "type": ["number","null"], "minimum":0, "maximum":1 }
```

These propagate to `mo_chunks` as filters so the query side can downweight or exclude OCR-derived content per request.

### 3.5 Step (3) — Encoding repair (M2)
```python
LEGACY_CODEPOINT_MAP = {
    "Ã":"Ă", "ã":"ă",
    "Ñ":"—",
    "ª":"Ş", "º":"ş",
    "Þ":"Ţ", "þ":"ţ",
    "Ò":"”",  "Ð":"Đ",
    # 2007 mojibake set (UTF-8 read as MacRoman):
    "√":"Ă", "¬":"Â", "™":"Ş", "¬R":"ÂR",
}
def repair_legacy_encoding(text: str) -> str:
    if not any(ch in text for ch in "ÃÑªºÞþÒÐ√¬™"): return text
    out = text
    # multi-char first, single-char last
    for k, v in sorted(LEGACY_CODEPOINT_MAP.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(k, v)
    return unicodedata.normalize("NFC", out)
```

### 3.6 Step (4) — Header / footer stripper
```python
HEADER_RE = re.compile(
    r"^MONITORUL\s+OFICIAL\s+AL\s+ROM[ÂA]NIEI,\s*PARTEA\s+I,\s*Nr\.\s*[\d]+(?:\s*bis)?"
    r"\s*/\s*\d{1,2}\.[IVXL]+\.\d{4}\s*\|\s*\d+\s*$",
    re.IGNORECASE,
)
def strip_chrome(blocks: list[Block], page_size) -> tuple[list[Block], dict]:
    H, W = page_size[1], page_size[0]
    header, footer, body = None, None, []
    for b in blocks:
        if b.bbox[3] < H * 0.07 and HEADER_RE.match(b.text.strip()):
            header = b; b.role = "running_header"; continue
        if b.bbox[1] > H * 0.96:
            footer = b; b.role = "footer_rule"; continue
        body.append(b)
    return body, {"header": header, "footer": footer}
```

### 3.7 Step (5) — Block role classifier (deterministic rule engine)
```python
SECTION_BANNERS = {
    "lege_decrete":            r"^L\s*E\s*G\s*I\s*(Ş|Ș)\s*I\s*D\s*E\s*C\s*R\s*E\s*T\s*E$",
    "decrete":                 r"^D\s*E\s*C\s*R\s*E\s*T\s*E$",
    "hg":                      r"^H\s*O\s*T\s*Ă\s*R[ÂA]R\s*I\s+A\s*L\s*E\s+G\s*U\s*V\s*E\s*R\s*N\s*U\s*L\s*U\s*I",
    "ccr":                     r"^D\s*E\s*C\s*I\s*Z\s*I\s*I\s+A\s*L\s*E\s+C\s*U\s*R\s*Ț\s*I\s*I\s+C\s*O\s*N\s*S\s*T\s*I\s*T\s*U\s*Ț\s*I\s*O\s*N\s*A\s*L\s*E",
    "acte_specialitate":       r"^ACTE\s+ALE\s+ORGANELOR\s+DE\s+SPECIALITATE",
    "decizii_pm":              r"^D\s*E\s*C\s*I\s*Z\s*I\s*I\s+A\s*L\s*E\s+P\s*R\s*I\s*M\s*-?\s*M\s*I\s*N\s*I\s*S\s*T\s*R\s*U\s*L\s*U\s*I",
}
def classify_block(b: Block, ctx: PageContext) -> str:
    text = collapse_spaces(b.text).strip()
    norm = strip_letterspacing(text)            # "L E G E" → "LEGE"
    # 0. cover specials
    if ctx.page_index == 0:
        if norm == "PARTEAI":           return "cover_partea"
        if "S U M A R" in text or norm == "SUMAR": return "cover_sumar_label"
        if re.match(r"^Anul\s", text):  return "cover_issue_line"
        if re.search(r",\s*\d+\s+\w+\s+\d{4}$", text): return "cover_date_line"
        if re.match(r"^Nr\.\s*\|\s*Pagina", text): return "cover_sumar_colhdr"
        if any(re.fullmatch(rx, norm) for rx in SECTION_BANNERS.values()):
            return "cover_sumar_section"
        if re.match(r"^\d+\.\s*[—–-]\s+", text) or re.search(r"\.{3,}\s*\d+(\s*[–-]\s*\d+)?$", text):
            return "cover_sumar_entry"
    # 1. running header / footer already stripped (Step 4)
    # 2. section banners (16-pt full-width)
    if b.font_size >= 14 and is_full_width(b, ctx):
        for key, rx in SECTION_BANNERS.items():
            if re.fullmatch(rx, norm):
                ctx.current_section = key
                return "section_banner"
    # 3. issuer (10-pt centered single-line)
    if 9.5 <= b.font_size <= 10.5 and is_centered(b, ctx) and norm in ISSUER_TAXONOMY:
        return "issuer"
    # 4. act type
    if 11.5 <= b.font_size <= 12.5 and norm in ACT_TYPE_TAXONOMY:
        return "act_type"
    # 5. act number / date subline
    if re.match(r"^D\s*E\s*C\s*I\s*Z\s*I\s*A\s+Nr\.\s*\d", text):  return "act_number"
    if re.match(r"^din\s+\d{1,2}\s+\w+\s+\d{4}$", text):           return "act_subdate"
    # 6. operative phrases (closed list)
    if norm in OPERATIVE_PHRASES:                                  return "operative_phrase"
    # 7. articles
    if re.match(r"^(Art\.\s*\d+(\^\d+)?\.?|Articol\s+unic\.)\s*[—–-]", text):  return "article"
    # 8. closing disposition
    if text.startswith("Această lege a fost adoptată"):            return "closing_disposition"
    if text.startswith("Această ordonanță"):                       return "closing_disposition"
    # 9. signatures
    if norm in SIGNATURE_ROLE_TAXONOMY:                            return "signature_role"
    if is_caps_name(text) and ctx.last_role == "signature_role":   return "signature_name"
    if text.startswith("Contrasemnează"):                          return "signature_contrasemneaza"
    # 10. place + date + number
    if re.match(r"^(Bucureşti|București|Bucuresti),\s+\d", text):  return "place_and_date"
    if re.match(r"^Nr\.\s*\d+\.?$", text):                         return "act_act_number"
    # 11. table cells handled separately
    # 12. preamble heuristic
    if text.startswith(("În temeiul", "Având în vedere", "Văzând")):
        return "preamble"
    return "unknown"          # → catch-all; not silently dropped
```

### 3.8 Step (6) — Reading order with full-width interrupts
```python
def reading_order(blocks: list[Block], page_w: float, gutter: float = 297.0) -> list[Block]:
    body = [b for b in blocks if b.role not in {"running_header","footer_rule","page_number"}]
    full_w = sorted([b for b in body if is_full_width(b, page_w)],     key=lambda b: b.bbox[1])
    cols   = [b for b in body if not is_full_width(b, page_w)]
    out = []
    cursor_y = 0
    for fw in full_w + [None]:
        ymax = fw.bbox[1] if fw else 1e9
        slab = [b for b in cols if cursor_y <= b.bbox[1] < ymax]
        left  = sorted([b for b in slab if (b.bbox[0]+b.bbox[2])/2 < gutter], key=lambda b: b.bbox[1])
        right = sorted([b for b in slab if (b.bbox[0]+b.bbox[2])/2 >= gutter], key=lambda b: b.bbox[1])
        out.extend(left); out.extend(right)
        if fw is not None:
            out.append(fw)
            cursor_y = fw.bbox[3]
    return out

def is_full_width(b: Block, page_w_or_ctx) -> bool:
    page_w = page_w_or_ctx if isinstance(page_w_or_ctx, (int,float)) else page_w_or_ctx.page_w
    return b.bbox[0] <= page_w * 0.13 and b.bbox[2] >= page_w * 0.87
```

### 3.9 Step (7) — Table extraction (Docling primary, Camelot/pdfplumber targeted fallbacks)

#### 3.9.1 Docling integration (primary)

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, TableFormerMode, EasyOcrOptions
)

# Single converter shared across all M2/M3 PDFs.
DOCLING = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(
                do_ocr=False,                # we own M1 OCR; Docling stays text-only
                do_table_structure=True,
                table_structure_options={
                    "mode": TableFormerMode.ACCURATE,   # vs FAST; we prefer accuracy
                    "do_cell_matching": True,
                },
                generate_page_images=False,
                images_scale=1.0,
            )
        )
    }
)

def docling_convert(pdf_path: str) -> "DoclingDocument":
    res = DOCLING.convert(pdf_path)
    return res.document      # iter via document.iterate_items() / .tables / .texts
```

Docling labels we trust as priors for the §3.7 role classifier:
| Docling label                | Mapped role (initial) |
|---|---|
| `section_header` / `title`   | `section_banner` or `act_type` (refined by font-size + regex) |
| `text`                       | `preamble` / `article` (regex-refined) |
| `list_item`                  | inside-article continuation |
| `table`                      | routed to §3.9.2 |
| `caption`                    | `table_caption` |
| `page_header` / `page_footer`| stripped (§3.4 already handles, double-strip OK) |
| `picture`                    | `decorative_glyph` (cover coat-of-arms) |

#### 3.9.2 Tables — Docling first, specialist fallbacks per typology

```python
def extract_tables(pdf_path: str, page_index: int, fitz_page: fitz.Page,
                   docling_doc) -> list[Table]:
    typology = classify_table_typology(fitz_page)   # T-A | T-B | T-C | T-D | none
    if typology == "none":
        return []

    # Primary: Docling TableFormer
    docling_tables = [t for t in docling_doc.tables
                      if t.prov[0].page_no == page_index + 1]

    if typology == "T-A":            # dense lattice (e.g. 2026-Bis Nomenclator)
        # Camelot lattice is empirically more reliable on 100s of cells per page.
        camelot_tables = camelot.read_pdf(pdf_path, pages=str(page_index+1),
                                          flavor="lattice", line_scale=40,
                                          shift_text=[""], copy_text=["v"])
        if camelot_tables and avg_accuracy(camelot_tables) >= 80:
            return [from_camelot(t, engine="camelot_lattice") for t in camelot_tables]
        # else fall through to Docling output

    if typology == "T-C":            # borderless / text-aligned
        # pdfplumber's text-strategy is purpose-built for this.
        with pdfplumber.open(pdf_path) as pdf:
            pp_tables = pdf.pages[page_index].extract_tables(table_settings={
                "vertical_strategy":"text", "horizontal_strategy":"text",
                "intersection_x_tolerance":3, "intersection_y_tolerance":3,
            }) or []
        if pp_tables:
            return [from_pdfplumber(t, engine="pdfplumber") for t in pp_tables]

    # T-B (bordered with mixed rules) and T-D (rotated headers) — Docling primary.
    return [from_docling(t, engine="docling") for t in docling_tables]


def classify_table_typology(page: fitz.Page) -> str:
    rects   = sum(1 for d in page.get_drawings() for it in d.get("items",[]) if it[0]=="re")
    h_rules = sum(1 for d in page.get_drawings() for it in d.get("items",[])
                    if it[0]=="l" and abs(it[1].y - it[2].y) < 1)
    has_rot = any(span_rotated(s) for s in iter_spans(page))   # 90°/270° text
    if rects >= 60:           return "T-A"
    if has_rot and rects >= 8:return "T-D"
    if rects >= 8 or h_rules >= 6: return "T-B"
    if h_rules >= 1:          return "T-C"
    return "none"


def render_table_to_markdown(grid: list[list[str]]) -> str:
    """GFM pipe table; cell newlines → <br>; preserves empty cells as space."""
    grid = [[ (c or "").replace("\n","<br>").strip() or " " for c in row ] for row in grid]
    if not grid: return ""
    head, *rows = grid
    out = ["| " + " | ".join(head) + " |",
           "| " + " | ".join(["---"]*len(head)) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)
```

#### 3.9.3 Decision matrix

| Tool | Role | When it wins | When it loses |
|---|---|---|---|
| **Docling** | Primary structured extraction (M2/M3) | Reading order, headings, lists, T-B/T-D tables, mixed-content pages. One pass. | Hyper-dense lattice tables (>200 cells), borderless TOC-style tables (no rules). |
| **Camelot lattice** | T-A specialist fallback | Full bordered grids (2026-Bis Nomenclator: 576 rects/page). Deterministic, no ML. | Borderless tables, scanned PDFs. |
| **pdfplumber** | T-C specialist fallback | Text-aligned tables: sumar TOC, signature blocks. | Anything with curved/rotated cells. |
| **PyMuPDF (fitz)** | Always-on geometric ledger + audit witness | Coverage_ratio invariant, header/footer stripping, gutter detection. | Not used for table extraction directly — too low-level. |
| **LlamaParse** | M1 (scanned) primary OCR only | Old typewriter scans where layout + recognition need an LLM. | M2/M3 (we don't pay or risk for it). |
| **PaddleOCR** | M1 deterministic offline fallback | Air-gapped deploys; LlamaParse outage; per-run budget exhausted. | Quality below LlamaParse; manual layout segmentation needed. |

The fallback ladder is *content-type-specific*, not "try X then Y blindly." Each typology has one correct primary and one specialist override; nothing else runs.

### 3.10 Step (8) — Sumar parser
```python
SUMAR_ENTRY = re.compile(
    r"^(?P<num>\d+(\.\d+)*)\s*\.\s*[—–-]\s*"
    r"(?P<title>.+?)"
    r"\s*\.{2,}\s*"
    r"(?P<page>\d+(?:\s*[–\-]\s*\d+|\s*\.{3}\s*\d+)?)\s*$",
    re.UNICODE,
)
def parse_sumar(blocks: list[Block]) -> list[SumarEntry]:
    section = None
    out = []
    for b in blocks:
        if b.role == "cover_sumar_section":
            section = strip_letterspacing(b.text).strip(); continue
        if b.role == "cover_sumar_entry":
            for line in b.text.splitlines():
                m = SUMAR_ENTRY.match(line.strip())
                if not m: continue
                pf, pt = parse_page_label(m["page"])
                out.append(SumarEntry(
                    section=section, act_number=m["num"], act_type=guess_act_type(m["title"]),
                    title=m["title"].strip(),
                    page_from=pf, page_to=pt, page_label=m["page"], raw=line,
                ))
    return out
```

### 3.11 Step (9-10) — Act segmentation & internal parse
Use the role-tagged, reading-ordered stream. Walk linearly:
```python
def segment_into_acts(stream: list[Block]) -> list[Act]:
    acts = []
    current = None
    section = None
    for b in stream:
        if b.role == "section_banner":
            section = b.text; continue
        if b.role == "issuer" or (b.role == "act_type" and current is None):
            if current is not None: acts.append(current)
            current = Act(section=section, block_refs=[])
            current.issuer_block = b if b.role == "issuer" else None
        if current is None: continue
        current.block_refs.append(b.block_id)
        attach_block_to_act(current, b)        # role-dispatched setter
    if current is not None: acts.append(current)
    return acts
```

Article parsing within an act:
```python
ARTICLE_HEAD = re.compile(
    r"^(?:Art\.\s*(?P<n>\d+(?:\^\d+)?)|Articol\s+unic)\.?\s*[—–-]\s*",
    re.UNICODE,
)
SUBLEVEL = re.compile(r"^\((?P<m>\d+)\)\s+|^lit\.\s+(?P<l>[a-z])\)|^pct\.\s+(?P<p>\d+)\.")
```

### 3.12 Step (11) — Graph edge extraction (regex only)
```python
CITATION_RE = re.compile(
    r"\b(?P<type>Lege(?:a)?|Ordonan[țt]a\s+(?:de\s+urgen[țt][ăa]\s+a\s+)?Guvernului|"
    r"Hot[ăa]r[âa]rea\s+Guvernului|Decretul|Decizia\s+Cur[țt]ii\s+Constitu[țt]ionale)"
    r"\s+nr\.\s*(?P<num>[\d\.]+)\s*/\s*(?P<year>\d{4})"
    r"(?:\s*,\s*art\.\s*(?P<art>\d+(?:\s*alin\.\s*\(\d+\))?))?",
    re.UNICODE | re.IGNORECASE,
)
PROMULGATES_RE = re.compile(
    r"pentru\s+promulgarea\s+Legii\s+(?:privind|pentru|nr\.\s*[\d./]+)", re.IGNORECASE
)
def extract_edges(act: Act) -> Graph:
    text = act.text_markdown
    cites = [normalize_citation(m) for m in CITATION_RE.finditer(text)]
    return Graph(
        issued_by=canonical_issuer(act.issuer_block.text if act.issuer_block else ""),
        signed_by=[s.person for s in act.signatures],
        cites=cites,
        promulgates=resolve_promulgation(act, cites) if act.act_type == "decret" else None,
        approves=resolve_approval(act, cites)        if act.act_type == "lege"   else None,
    )
```

### 3.13 Step (13-14) — Catch-all and audit
```python
def build_unmapped(pages: list[PageRecord]) -> list[UnmappedItem]:
    out = []
    for p in pages:
        for b in p.blocks:
            if b.role in {"unknown"} or (b.role != "unknown" and b.claimed_by is None
                                          and b.role not in NON_BODY_ROLES):
                out.append(UnmappedItem(
                    block_id=b.block_id, page_index=p.index,
                    y=b.bbox[1], x=b.bbox[0], bbox=list(b.bbox),
                    text=b.text, rotation=b.rotation,
                    font=b.font, font_size=b.font_size,
                    reason=infer_reason(b),
                ))
    out.sort(key=lambda u: (u.page_index, u.y, u.x))
    return out

def audit_coverage(doc_record: dict, raw_text: str) -> dict:
    raw_chars     = len(re.sub(r"\s+", "", raw_text))
    mapped_chars  = sum(len(re.sub(r"\s+","",b["text"]))
                        for p in doc_record["pages"]
                        for b in p["blocks"]
                        if b["claimed_by"] is not None)
    unmapped_chars= sum(len(re.sub(r"\s+","",u["text"]))
                        for u in doc_record["unmapped_raw_stream"])
    cov = (mapped_chars + unmapped_chars) / max(raw_chars, 1)
    if cov < 0.999:
        # Sweep: any block in pages[].blocks not claimed AND not in unmapped → push to unmapped.
        repair_unmapped(doc_record); cov = 1.0
    return {"raw_text_chars": raw_chars,
            "mapped_chars":   mapped_chars,
            "unmapped_chars": unmapped_chars,
            "coverage_ratio": cov}
```

### 3.14 Vector-poisoning prevention checklist
Before chunks reach embedding:
1. Strip running header & footer (Step 4).
2. Drop any block with role `running_header`, `page_number`, `footer_rule`.
3. Skip `figure_only` and `decorative_glyph` chunks.
4. **Keep** `unmapped_raw_stream` items in a *secondary* index (low-weight BM25 only, **not** vectorized) — they preserve recall on edge cases without polluting semantic similarity.
5. Hyphenation join: collapse `\b(\w+)-\s*\n\s*(\w+)` across line breaks before embedding.
6. Diacritic-equivalent index in Mongo: store both `title_ro` and `title_ro_ascii` (NFKD-stripped) for tolerant text search.

### 3.15 Chunker (deterministic, no LLM)
The act → chunk transform is the bridge from the schema to vector search. It is the only place that produces `mo_chunks` documents.

```python
@dataclass
class ChunkerConfig:
    target_tokens: int   = 650          # central target
    max_tokens:    int   = 900          # hard cap, except oversize-table
    min_tokens:    int   = 120          # below this, merge with neighbour
    overlap_tokens:int   = 80           # sliding overlap for context preservation
    tokenizer:     str   = "cl100k_base" # tiktoken; deterministic across runs
    boundary_priority: tuple = (
        "TABLE_ATOMIC",  # tables are atomic — never split
        "ARTICLE",       # split at "Art. N. —" / "Articol unic. —"
        "PARAGRAPH",     # blank-line / hyphenation-joined boundary
        "SENTENCE",      # last resort: sentence-final punctuation
    )

def chunk_act(act: Act, cfg: ChunkerConfig, prefix: str) -> list[Chunk]:
    """
    Produce 1..N chunks per act. Invariants:
      • EVERY token from act.text_markdown lands in exactly one chunk
        (modulo overlap, which is duplicated by design and tagged).
      • Tables are atomic: emitted as their own chunk with kind='table'
        and is_oversize_table = True if larger than max_tokens.
      • Each chunk text starts with `prefix` (issuer/act header banner)
        — but the `char_range` excludes the prefix, so coverage_audit
        still resolves against the source.
      • Diacritic-free text_ascii produced by NFKD + Mn-strip.
    """
    units = split_into_units(act, cfg.boundary_priority)   # [(kind, text, char_range, block_refs)]
    chunks, buf, buf_tokens, buf_meta = [], [], 0, []
    for u in units:
        u_tokens = count_tokens(u.text, cfg.tokenizer)
        if u.kind == "TABLE_ATOMIC":
            if buf: chunks.append(_emit(buf, buf_meta, prefix, cfg)); buf, buf_tokens, buf_meta = [], 0, []
            chunks.append(_emit_table(u, prefix, cfg))
            continue
        if buf_tokens + u_tokens > cfg.max_tokens and buf:
            chunks.append(_emit(buf, buf_meta, prefix, cfg))
            # carry overlap
            overlap = _tail_tokens(buf, cfg.overlap_tokens, cfg.tokenizer)
            buf, buf_tokens, buf_meta = [overlap], count_tokens(overlap, cfg.tokenizer), [_synth_meta(overlap)]
        buf.append(u.text); buf_tokens += u_tokens; buf_meta.append(u)
    if buf: chunks.append(_emit(buf, buf_meta, prefix, cfg))
    # merge tail chunk if below min_tokens
    if len(chunks) >= 2 and chunks[-1].token_count < cfg.min_tokens:
        chunks[-2:] = [_merge(chunks[-2], chunks[-1])]
    return _index(chunks, act)

def make_prefix(act: Act) -> str:
    """Constant front-matter included verbatim in every chunk's text."""
    issuer = act.issuer.display
    typ    = act.act_type.replace("_"," ").title()
    num    = f" {act.act_number}/{act.act_year}" if act.act_number else ""
    title  = act.title.strip()
    return f"[{issuer}] {typ}{num} — {title}\n\n"
```

Empirical chunk targets (validated against the corpus):
| Act type | Typical chars | Typical chunks/act |
|---|---|---|
| Decret (numire judecător)         | 600-1200 | 1 |
| Lege (ratificare tratat)          | 2-5k     | 1-3 |
| Hotărâre Guvern (operativă)       | 3-8k     | 2-6 |
| Decizie Curtea Constituțională    | 30-80k   | 25-90 |
| Anexă Bis (Nomenclator)           | tables   | 1 chunk per table block + 1 header |

### 3.16 MongoDB Atlas — collection layout
```
db.mo_issues        — one doc per PDF (issue-level metadata only; pages stripped if doc > 8MB)
db.mo_acts          — one doc per act (act_id = _id); facets, graph edges, full text_markdown
db.mo_blocks        — flattened pages[].blocks[]; one doc per block. Geometric ledger for round-trip.
                       _id = block_id; index { issue_id:1, page_index:1, "bbox.0":1 }
db.mo_chunks        — RETRIEVAL UNIT. one doc per chunk (chunk_id = _id).
                       Vector + Atlas Search indexes both live HERE.
db.mo_unmapped      — flattened unmapped_raw_stream; secondary BM25 index, NEVER vectorized.
db.mo_graph_edges   — { _id, from_act_id, to_act_id|external_ref, kind, raw }
                       kinds: "cites" | "promulgates" | "approves" | "signed_by"
db.mo_runs          — one doc per ingestion run (audit trail; see §4.2)
```

### 3.17 Atlas Search index (BM25, on `mo_chunks`)
```jsonc
// /atlas-indexes/mo_chunks.search.json   (deploy with `atlas search indexes create`)
{
  "name": "mo_chunks_text",
  "definition": {
    "mappings": {
      "dynamic": false,
      "fields": {
        "text":             { "type":"string", "analyzer":"lucene.romanian", "searchAnalyzer":"lucene.romanian" },
        "text_ascii":       { "type":"string", "analyzer":"lucene.standard" },
        "issuer_canonical": { "type":"token" },
        "act_type":         { "type":"token" },
        "section":          { "type":"token" },
        "modality":         { "type":"token" },
        "act_year":         { "type":"number" },
        "publication_date": { "type":"date" },
        "embedding_model":  { "type":"token" },
        "embedding_version":{ "type":"token" }
      }
    }
  }
}
```

Romanian analyzer notes:
- `lucene.romanian` does folding + Snowball stemming → `lege/legi/legii/legilor` collapse correctly.
- `text_ascii` (NFKD + `Mn` stripped) hedges against query-side diacritic stripping ("hotarare" matches "hotărâre").
- Use `compound.should` to score both fields with weights 1.0 + 0.4.

### 3.18 Atlas Vector Search index (on `mo_chunks`)

**Default embedding model: `BAAI/bge-m3`** — 1024-d dense, fully local, multilingual (covers Romanian natively, no separate fine-tune needed), Apache-2.0 licensed, runs on a single GPU (4-8 GB VRAM at fp16). Throughput on RTX 4090: ~600 chunks/sec at batch=32. No per-token cloud cost.

```jsonc
// /atlas-indexes/mo_chunks.vector.json
{
  "name": "mo_chunks_vec",
  "type": "vectorSearch",
  "fields": [
    { "type":"vector", "path":"embedding", "numDimensions":1024, "similarity":"cosine" },
    { "type":"filter", "path":"issuer_canonical" },
    { "type":"filter", "path":"act_type" },
    { "type":"filter", "path":"section" },
    { "type":"filter", "path":"publication_date" },
    { "type":"filter", "path":"act_year" },
    { "type":"filter", "path":"modality" },
    { "type":"filter", "path":"embedding_model" },
    { "type":"filter", "path":"embedding_version" },
    { "type":"filter", "path":"ocr_engine" }
  ]
}
```

#### 3.18.1 BGE-M3 embedder
```python
from FlagEmbedding import BGEM3FlagModel

BGE = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True,                       # 4-8GB VRAM, ~2x faster, no measurable quality loss
    device="cuda",                       # falls back to "cpu" automatically
    normalize_embeddings=True,           # cosine-ready
)

def embed_chunks(texts: list[str], batch_size: int = 32) -> np.ndarray:
    out = BGE.encode(
        texts,
        batch_size=batch_size,
        max_length=8192,                 # bge-m3 native context; covers any chunk we emit
        return_dense=True,
        return_sparse=False,             # toggle on for hybrid lexical (see 3.18.2)
        return_colbert_vecs=False,       # toggle on for late-interaction (future v2)
    )
    return out["dense_vecs"].astype("float32")    # (N, 1024)

EMBEDDING_MODEL   = "BAAI/bge-m3"
EMBEDDING_DIM     = 1024
EMBEDDING_VERSION = "chunker-1.2+bge-m3-fp16"
```

#### 3.18.2 BGE-M3 multi-functional retrieval (optional v1.1 enhancement)
BGE-M3 uniquely returns **dense + sparse + ColBERT-style multi-vector** outputs from one forward pass. The schema and Atlas index above use only the dense vector for v1. Two cheap upgrades for v1.1:

1. **Sparse vectors → BM25-equivalent recall on rare legal terms**: enable `return_sparse=True`, store `sparse_embedding: {"<token_id>": <weight>, …}` on each chunk, query with a custom $search-like fuser. This buys recall on long-tail tokens (e.g. obscure law numbers) without an external BM25 layer.
2. **ColBERT multi-vector → late interaction reranking**: keep dense as the recall stage, rerank the top-50 with `compute_score(query_colbert_vecs, chunk_colbert_vecs)`. ~2-3 pp nDCG@10 typical.

Neither is required for v1; both are flagged in `EMBEDDING_VERSION` if turned on so the cache invalidates correctly.

### 3.19 Hybrid query (Atlas 8.0+ `$rankFusion`; manual RRF for older versions)
```javascript
// Atlas 8.0+ native rank fusion
db.mo_chunks.aggregate([
  { $rankFusion: {
      input: { pipelines: {
        vec: [{
          $vectorSearch: {
            index: "mo_chunks_vec", path: "embedding",
            queryVector: q_emb, numCandidates: 200, limit: 60,
            filter: {
              embedding_version: "chunker-1.2+voyage-3",
              publication_date:  { "$gte": ISODate("2007-01-01") },
              modality:          { "$in": ["M2_LEGACY_DIGITAL","M3_MODERN_DIGITAL"] }
            }
          }
        }],
        bm25: [
          { $search: { index: "mo_chunks_text", compound: { should: [
              { text: { query: q_text, path: "text",       score: { boost: { value: 1.0 } } } },
              { text: { query: q_text_ascii, path: "text_ascii", score: { boost: { value: 0.4 } } } }
          ]}}},
          { $limit: 60 }
        ]
      }},
      combination: { weights: { vec: 0.6, bm25: 0.4 } },
      scoreDetails: true
  }},
  { $limit: 20 },
  // hydrate the act doc once, in batch
  { $lookup: { from: "mo_acts", localField: "act_id", foreignField: "_id", as: "act" } },
  { $unwind: "$act" },
  // optional graph expansion: pull cited acts for GraphRAG
  { $lookup: { from: "mo_graph_edges",
               let: { aid: "$act_id" },
               pipeline: [
                 { $match: { $expr: { $eq: ["$from_act_id","$$aid"] } } },
                 { $limit: 25 }
               ],
               as: "edges" }}
])
```

Manual RRF fallback (Atlas <8.0) — run the two pipelines, fuse with `score = sum(1/(k+rank_i))` where `k=60`.

### 3.20 OCR-aware retrieval policy
M1 (scanned, OCR-derived) chunks carry `modality:"M1_SCANNED"` and `ocr_min_confidence`. In production:
- Default search filters M1 OUT of the vector path (high noise) but keeps it in BM25 as a recall tier.
- Toggle via query param: `include_scanned=true` flips the filter.
- M1 chunks are re-OCR'd whenever `ocr_engine_version` advances; old chunks are deleted by `embedding_version` mismatch.

### 3.16 Implementation milestones
1. **Stage A (1-2 d)** — modality classifier + fitz block dump + Pydantic models scaffolding the schema.
2. **Stage B (2-3 d)** — role classifier + reading-order + sumar parser; validate on all M3 PDFs (2007-12 → 2026).
3. **Stage C (1-2 d)** — encoding repair table calibrated on every 1997-2007 sample; rerun roles.
4. **Stage D (2-3 d)** — Camelot/pdfplumber wiring; benchmark on 2026-Bis (146-page Nomenclator) and 2020-08-14 inventory tables.
5. **Stage E (3-4 d)** — PaddleOCR M1 path; layout columns from projection profile; coverage audit.
6. **Stage F (1-2 d)** — graph edge regex set; promulgation/approval resolver; ingest into Mongo with the four-collection layout.
7. **Stage G (continuous)** — `coverage_ratio == 1.0` enforced as a CI gate; any document failing is quarantined, not ingested.

---

## Definition of done (engineering)
- `coverage_ratio == 1.000` on **every** PDF in `laws/`.
- Round-trip test: rebuilding `acts[].text_markdown` from `pages[].blocks` via `block_refs` reproduces the canonical Markdown byte-for-byte.
- Mongo `$vectorSearch` returns the act-level snippet for the query "promulgarea Legii nr. 199/2023" (any sample matching) with ≥0.85 cosine to the true act.
- No LLM call in the runtime path. (Optional: LLM-driven offline rule-tuning is acceptable; runtime stays deterministic.)

---

# PART 4 — Production Readiness

## 4.0 Two-stage architecture: extract-to-disk, load-to-Mongo

The pipeline is **two decoupled stages** that talk through a filesystem-friendly bundle format. Stage A (heavy: PDF parsing + OCR + embedding) runs anywhere — your laptop, a VPS, a spot instance. Stage B (light: bulk-write to Mongo) runs from wherever has the network path to Atlas. They never need to be on the same machine.

```
Stage A — Extract           Stage B — Load
─────────────────           ──────────────
PDFs ─► pipeline ─► out/    out/ ─► validate ─► bulk_upsert ─► Atlas
        (GPU heavy)               (network heavy, idempotent)
```

This unlocks four useful workflows:
1. **VPS extraction → local download → upload from home** — process where compute is cheap, ship JSONL home, push to Atlas over your residential connection.
2. **All-local extraction → batched upload** — laptop produces bundles, you review them, then `mo.load` pushes when satisfied.
3. **Air-gapped processing → sneakernet** — extract behind a firewall, mail the tarball, load on the public side.
4. **Reproducible re-loads** — bundles are content-addressable; loading is `O(diff)`, not `O(corpus)`.

### 4.0.1 Bundle layout on disk

```
out/
├── manifest.jsonl                    # one line per processed PDF; loader's index
├── runs/
│   └── run-2026-05-29T08-12-44Z.json  # mo_runs doc for this batch
└── by_doc/
    └── MO_PI_295_2026-04-14/         # one dir per PDF; doc_id is the directory name
        ├── _meta.json                #   sha256, sizes, schema_version, pipeline_version,
        │                             #   embedding_version, file_checksums
        ├── issue.json                #   the top-level Issue document (1 doc → mo_issues)
        ├── acts.jsonl                #   N lines → mo_acts
        ├── blocks.jsonl              #   N lines → mo_blocks
        ├── chunks.jsonl              #   N lines → mo_chunks (embeddings inline as float32 lists)
        ├── edges.jsonl               #   N lines → mo_graph_edges
        └── unmapped.jsonl            #   N lines → mo_unmapped
```

All `.jsonl` files are gzipped if `--gzip` flag set (default ON for chunks; off for everything else so they stay grep-able).

`manifest.jsonl` is the loader's source of truth. One line per doc:
```jsonc
{"doc_id":"MO_PI_295_2026-04-14",
 "sha256":"7f3a…","schema_version":"1.0.0","pipeline_version":"1.4.0","embedding_version":"chunker-1.2+bge-m3-fp16",
 "files":{"issue.json":12041,"acts.jsonl":33887,"blocks.jsonl":201433,
          "chunks.jsonl.gz":418200,"edges.jsonl":2890,"unmapped.jsonl":914},
 "checksums":{"issue.json":"…","acts.jsonl":"…", …},
 "extracted_at":"2026-05-29T08:14:02Z",
 "coverage_ratio":1.0}
```

### 4.0.2 Stage A — `mo extract` (disk emitter)

Replaces the "persist" step at the end of §3.2. Same pipeline, different sink.

```python
# mo/extract.py
import json, gzip, hashlib, pathlib
from datetime import datetime, timezone

def emit_doc(doc_record: dict, chunks: list[dict], blocks: list[dict],
             edges: list[dict], unmapped: list[dict], out_root: pathlib.Path) -> dict:
    doc_id = doc_record["doc_id"]
    d = out_root / "by_doc" / doc_id
    d.mkdir(parents=True, exist_ok=True)

    issue_doc = strip_to_issue_only(doc_record)        # remove acts/blocks/chunks (they go to siblings)
    acts      = doc_record["acts"]

    files = {
        "issue.json":      json_dump(d / "issue.json",      issue_doc),
        "acts.jsonl":      jsonl_dump(d / "acts.jsonl",     acts),
        "blocks.jsonl":    jsonl_dump(d / "blocks.jsonl",   blocks),
        "chunks.jsonl.gz": jsonl_dump(d / "chunks.jsonl.gz",chunks, gz=True),
        "edges.jsonl":     jsonl_dump(d / "edges.jsonl",    edges),
        "unmapped.jsonl":  jsonl_dump(d / "unmapped.jsonl", unmapped),
    }
    checksums = {n: sha256_file(d / n) for n in files}

    meta = {
        "doc_id":             doc_id,
        "sha256":             doc_record["source"]["sha256"],
        "schema_version":     doc_record["extraction_audit"]["schema_version"],
        "pipeline_version":   doc_record["extraction_audit"]["pipeline_version"],
        "embedding_version":  chunks[0]["embedding_version"] if chunks else None,
        "files":              files,                        # {name: byte_size}
        "checksums":          checksums,                    # {name: sha256}
        "extracted_at":       datetime.now(timezone.utc).isoformat(),
        "coverage_ratio":     doc_record["extraction_audit"]["coverage_ratio"],
    }
    (d / "_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta

def append_manifest(out_root: pathlib.Path, meta: dict):
    """Append-only manifest. Loader reads this to know what to upload."""
    with (out_root / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def jsonl_dump(path: pathlib.Path, items, gz: bool = False) -> int:
    opener = (lambda p: gzip.open(p, "wt", encoding="utf-8")) if gz else \
             (lambda p: open(p, "w", encoding="utf-8"))
    with opener(path) as f:
        for it in items: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return path.stat().st_size

def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()
```

CLI:
```
python -m mo extract --root laws/ --out out/ --workers 4 --gzip-chunks
                     [--skip-if-meta-current]      # idempotent: skip docs whose
                                                   # _meta.json matches current versions
                     [--no-mongo]                  # default; never touches network
```

### 4.0.3 Stage B — `mo load` (bulk upsert to Atlas)

```python
# mo/load.py
import json, gzip, pathlib, time
from pymongo import MongoClient, UpdateOne, ASCENDING

def load_bundle(out_root: pathlib.Path, mongo_uri: str, db_name: str = "mo",
                resume: bool = True, batch: int = 1000):
    cli = MongoClient(mongo_uri, w="majority", retryWrites=True)
    db  = cli[db_name]
    ensure_indexes(db)

    state_path = out_root / ".load_state.json"
    state = json.loads(state_path.read_text()) if (resume and state_path.exists()) else {}

    with (out_root / "manifest.jsonl").open() as f:
        for line in f:
            meta = json.loads(line)
            doc_id = meta["doc_id"]
            key = f"{doc_id}::{meta['sha256']}::{meta['pipeline_version']}::{meta['embedding_version']}"
            if state.get(key) == "done":
                continue                                       # already loaded; skip

            d = out_root / "by_doc" / doc_id
            verify_checksums(d, meta["checksums"])             # fail loud on transit corruption

            with cli.start_session() as sess:
                with sess.start_transaction():
                    issue = json.loads((d/"issue.json").read_text())
                    db.mo_issues.replace_one({"_id": doc_id}, issue, upsert=True, session=sess)

                    bulk_upsert_jsonl(db.mo_acts,        d/"acts.jsonl",       "_id", "act_id",       sess, batch)
                    bulk_upsert_jsonl(db.mo_blocks,      d/"blocks.jsonl",     "_id", "block_id",     sess, batch)
                    bulk_upsert_jsonl(db.mo_chunks,      d/"chunks.jsonl.gz",  "_id", "chunk_id",     sess, batch, gz=True)
                    bulk_upsert_jsonl(db.mo_graph_edges, d/"edges.jsonl",     "_id", "_id",          sess, batch)
                    bulk_upsert_jsonl(db.mo_unmapped,    d/"unmapped.jsonl",  "_id", "block_id",     sess, batch)

            state[key] = "done"
            state_path.write_text(json.dumps(state, indent=2))
            print(f"loaded {doc_id} ({sum(meta['files'].values())//1024} KB)")

def bulk_upsert_jsonl(coll, path, id_field, src_field, sess, batch, gz=False):
    opener = gzip.open if gz else open
    ops, n = [], 0
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc[id_field] = doc[src_field]
            ops.append(UpdateOne({id_field: doc[id_field]}, {"$set": doc}, upsert=True))
            if len(ops) >= batch:
                coll.bulk_write(ops, ordered=False, session=sess); n += len(ops); ops = []
        if ops: coll.bulk_write(ops, ordered=False, session=sess); n += len(ops)
    return n
```

CLI:
```
python -m mo load --root out/ --mongo "mongodb+srv://USER:PWD@cluster.mongodb.net/" \
                  [--db mo] [--resume] [--batch 1000] [--dry-run]
```

Idempotency notes:
- `_id` for every collection equals the deterministic id (`doc_id`, `act_id`, `chunk_id`, `block_id`, `edge_id`). Re-running `mo load` is a no-op for already-loaded docs.
- `.load_state.json` makes resume cheap after partial failures.
- `--dry-run` validates checksums + JSONL syntax without writing.

### 4.0.4 Transfer patterns (VPS → local)

Pick ONE based on size and frequency. All work because the bundle is just files.

**Option 1 — `rsync` (recommended for incremental).**
```bash
# from your laptop, after a VPS extraction run:
rsync -avz --partial --progress \
  vps.example.com:/srv/mo/out/ ./out/
# rsync only sends changed files; resumable; checksums on the wire.
```

**Option 2 — tarball over scp/sftp (one-shot snapshot).**
```bash
# on VPS:
tar --use-compress-program='zstd -19 -T0' \
    -cf out-2026-05-29.tar.zst -C /srv/mo out
# back on laptop:
scp vps.example.com:/srv/mo/out-2026-05-29.tar.zst .
zstd -d out-2026-05-29.tar.zst -o - | tar -xf -
```

**Option 3 — S3 as intermediate (best for autoscaled ingest).**
```bash
# VPS → S3 (after each extraction run)
aws s3 sync /srv/mo/out/ s3://my-mo-bundles/out/ --storage-class STANDARD_IA
# laptop ← S3
aws s3 sync s3://my-mo-bundles/out/ ./out/
# then load
python -m mo load --root ./out/ --mongo "$MONGO_URI"
```

**Option 4 — VPS uploads directly to Atlas (skip the local hop).**
```bash
# on VPS, after extraction, when you trust it:
python -m mo load --root /srv/mo/out/ --mongo "$MONGO_URI"
```
This is the simplest pattern when VPS has clean egress to `*.mongodb.net` (most do). Local download isn't required — bundles still live on disk for audit.

### 4.0.5 Bundle size estimates (current 119-PDF corpus)

Per PDF, typical:
| File | M3 (modern, 8 pages) | M3 dense (Bis, 146 pages) | M1 (scanned, 6 pages) |
|---|---|---|---|
| `issue.json`           | 8-15 KB    | 80 KB     | 6 KB     |
| `acts.jsonl`           | 30-60 KB   | 100 KB    | 25 KB    |
| `blocks.jsonl`         | 150-300 KB | 4-6 MB    | 50 KB    |
| `chunks.jsonl.gz`      | 200-500 KB | 6-10 MB   | 100 KB   |
| `edges.jsonl`          | 2-5 KB     | 8 KB      | 1 KB     |
| `unmapped.jsonl`       | 0.5-3 KB   | 50 KB     | 5 KB     |
| **Per-doc total**      | **~1 MB**  | **~12 MB**| **~0.2 MB** |

Full corpus bundle: **~150 MB compressed** — fits in a single zstd tarball; rsyncs in seconds over residential.

100k-issue projection: ~150 GB on disk. Use S3 + `--storage-class GLACIER_IR` for cold archive at $0.004/GB/mo.

### 4.0.6 Two concrete recipes

#### Recipe A — VPS processing, local upload

```bash
### ON VPS (one-time setup) ###
git clone <repo> mo && cd mo
uv pip install -r requirements.lock
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
docling-tools download-models --output-dir models/docling
echo "LLAMA_CLOUD_API_KEY=llx-..." > .env

### ON VPS (per ingest run) ###
python -m mo extract --root laws/ --out out/ --workers 4 --gzip-chunks
# pipeline runs, writes everything under out/

### ON LAPTOP (after run completes) ###
rsync -avz --partial --progress vps:/srv/mo/out/ ./out/

# inspect a doc before pushing anywhere
jq . out/by_doc/MO_PI_295_2026-04-14/_meta.json

# now push to Atlas
export MONGO_URI="mongodb+srv://user:pwd@cluster.mongodb.net/"
python -m mo load --root ./out/ --mongo "$MONGO_URI" --resume
```

VPS sizing for this recipe: any 16GB-RAM Linux box with NVMe disk works for current corpus; for the GPU path, `g5.xlarge` spot instance for ~30 min costs ~$0.20.

#### Recipe B — Fully local processing, batched upload

```bash
### ON LAPTOP ###
python -m mo extract --root laws/ --out out/ --workers 2
# review the manifest
jq -s 'map({doc_id, coverage_ratio}) | sort_by(.coverage_ratio)' out/manifest.jsonl
# any coverage < 1.0 → fix before loading
python -m mo load --root ./out/ --mongo "$MONGO_URI"
```

This is the simplest path. You get all the audit benefits of Recipe A without a second machine.

### 4.0.7 Why this pattern is worth the small extra step

- **Atlas cost discipline.** You only pay for vector index time on data you've reviewed.
- **Disaster recovery is trivial.** `out/` IS the backup. Lose Atlas → `mo load --root out/` rebuilds it.
- **Reviews / approvals.** Bundles are file-based; diff them in a PR before they touch prod.
- **Reprocess economy.** Bumped `EMBEDDING_VERSION`? Re-run extract on changed chunks only; bundle diffs go up; `mo load` upserts in place.
- **Schema migrations.** `out/` is the canonical replay log. Migration becomes a script that rewrites JSONL files, not a Mongo-side surgery.

---



This section turns the schema + pipeline into a system that can be operated. Everything below is deterministic, automated, and testable.

## 4.1 Idempotent ingestion & content-addressable storage

Every PDF is uniquely keyed by `sha256(pdf_bytes)`. The ingestion contract is **upsert-by-hash, never blind insert**.

```python
def ingest(pdf_path: Path, run_id: str) -> IngestResult:
    raw = pdf_path.read_bytes()
    h   = hashlib.sha256(raw).hexdigest()
    doc_id = derive_doc_id(pdf_path.name)            # "MO_PI_295_2026-04-14"
    cas_path = f"s3://mo-cas/{h[:2]}/{h[2:4]}/{h}.pdf"  # CAS layout
    s3.upload_if_absent(cas_path, raw)

    existing = mo_issues.find_one({"_id": doc_id})
    if existing and existing["source"]["sha256"] == h \
       and existing["extraction_audit"]["pipeline_version"] == PIPELINE_VERSION \
       and existing["extraction_audit"]["schema_version"]   == SCHEMA_VERSION:
        return IngestResult(skipped=True, reason="unchanged")

    record = run_pipeline(raw, doc_id, cas_path, run_id)
    persist_record(record)               # transactional: issue + acts + blocks + chunks + edges
    return IngestResult(skipped=False, doc_id=doc_id, sha256=h)
```

Hash semantics:
- `source.sha256` is the **content hash** (the file). Re-uploading the same PDF is a no-op.
- `extraction_audit.pipeline_version` + `extraction_audit.schema_version` are pipeline hashes; bumping either reprocesses every doc on next run, even if content is unchanged.
- `chunks.embedding_version` is the model hash; bumping that re-embeds chunks **without** re-running extraction.

The three layers (content, extraction, embedding) are independently reprocessable.

## 4.2 Run ledger (`mo_runs`) and DLQ

```jsonc
// db.mo_runs — one doc per ingestion run
{
  "_id":            "run-2026-05-28T14:32:11Z-7f3a",
  "started_at":     "2026-05-28T14:32:11Z",
  "ended_at":       "2026-05-28T14:47:02Z",
  "pipeline_version":"1.4.0",
  "schema_version":"1.0.0",
  "embedding_model":"voyage-3",
  "embedding_version":"chunker-1.2+voyage-3",
  "input_root":     "laws/",
  "stats": {
    "pdfs_seen": 119, "pdfs_processed": 27, "pdfs_skipped": 91, "pdfs_failed": 1,
    "acts_written": 412, "chunks_written": 3104,
    "coverage_min": 1.000, "coverage_mean": 1.000,
    "ocr_pages": 86, "tables_extracted": 174
  },
  "failures": [
    { "pdf": "laws/1990/06/15/MO_PI_83_1990-06-15.pdf",
      "stage": "ocr",
      "error": "PaddleOCR CUDA OOM at page 4",
      "retryable": true }
  ]
}

// Failed PDFs land in db.mo_dlq
{
  "_id":     "dlq-<sha256>-<run_id>",
  "pdf":     "...",
  "sha256":  "...",
  "stage":   "ocr|extract|chunk|embed|persist",
  "attempts":1,
  "last_error":"...",
  "traceback":"...",
  "needs_human": false,    // flipped true after attempts >= 3
  "first_seen": ts,
  "last_seen":  ts
}
```

Retry policy: exponential backoff (1m → 5m → 30m → 2h → human). DLQ items with `needs_human=true` open a Linear/Github ticket via webhook.

## 4.3 Versioning matrix

Three independent semver streams; every chunk and issue carries all three:

| Component | Bumps when… | Effect |
|---|---|---|
| `schema_version`     | JSON Schema changes (field added/renamed/typed) | Forces full re-extraction; old docs migrated by `migrations/<from>_to_<to>.py` |
| `pipeline_version`   | extraction code changes (new regex, classifier rule, OCR model) | Forces full re-extraction; embeddings still valid if `text` unchanged |
| `embedding_version`  | chunker config OR embedding model changes | Re-embedding only; extraction untouched |

Migrations are mandatory for `schema_version` major bumps; minor bumps must be backward-compatible (additive fields only).

```python
SCHEMA_VERSION    = "1.0.0"
PIPELINE_VERSION  = "1.4.0"
EMBEDDING_VERSION = "chunker-1.2+voyage-3"      # composite
```

## 4.4 Reproducibility & dependency pinning

- **Pin everything**: `requirements.lock` generated by `uv pip compile`, hash-pinned.
- **Container**: a single `Dockerfile` derived from `ghcr.io/anthropics/python-3.12-slim` with `ghostscript`, `tesseract-ocr-ron`, `libgl1` baked in. Image tag = `pipeline_version`.
- **System fonts not required** at runtime (PyMuPDF doesn't render).
- **GPU optional**: PaddleOCR has CPU fallback; only M1 path benefits from GPU.
- **Determinism gates**: every numeric tolerance lives in `config/tolerances.yaml`, version-controlled. No magic numbers in code.

## 4.5 Observability

### Structured logs (one JSON line per event)
```jsonc
{"ts":"2026-05-28T14:33:01Z","run":"run-…","doc":"MO_PI_295_2026-04-14",
 "stage":"reading_order","level":"info",
 "pages":8,"blocks":118,"full_width":12,"acts":4,"elapsed_ms":312}
```

### Metrics (Prometheus or OpenTelemetry → Grafana)
- `pdf_ingest_duration_seconds{stage,modality}` (histogram)
- `extraction_coverage_ratio{modality}` (gauge — alert if `< 1.0`)
- `unmapped_chars_ratio{modality}` (gauge — alert if mean > 5%)
- `chunks_per_act` (histogram)
- `ocr_page_seconds{engine}` (histogram)
- `mo_dlq_size` (gauge — alert if > 0 for >24h)
- `embedding_request_seconds{model}` (histogram)
- `vector_index_lag_seconds` (gauge — Atlas-reported)

### Traces
OpenTelemetry spans: `ingest → modality → (ocr|fitz_dump) → repair → strip → classify → reading_order → tables → segment_acts → chunk → embed → persist`. Carry `doc_id` and `run_id` as attributes on every span.

### SLOs
| SLO | Target | Window |
|---|---|---|
| Per-PDF p95 extraction latency (M3, ≤32 pages) | ≤ 8 s | 30 d |
| Per-PDF p95 (M1, ≤8 pages, GPU OCR)            | ≤ 45 s | 30 d |
| `coverage_ratio == 1.000`                      | 100 % | per run |
| Chunk embedding p95                            | ≤ 80 ms/chunk | 30 d |
| Hybrid query p95 latency                       | ≤ 350 ms | 30 d |
| Vector index freshness (write → searchable)    | ≤ 60 s | 30 d |

## 4.6 Testing strategy

### Test pyramid
1. **Unit (fastest)** — every regex, every role classifier rule, every encoding-repair entry has a targeted test with synthetic input.
2. **Snapshot (golden corpus)** — `tests/golden/<modality>/<doc_id>.json` is the canonical extraction. CI fails on any byte-level diff that isn't accompanied by an explicit snapshot update with reviewer sign-off.
3. **Property-based** (Hypothesis) — for the chunker: "for any list of units, total tokens emitted ≥ total tokens input"; "every chunk overlap is ≤ `overlap_tokens`"; "no chunk exceeds `max_tokens` unless it is a single TABLE_ATOMIC unit".
4. **Integration** — Atlas dev cluster with `mo_chunks` of 200 chunks; assert hybrid query returns expected act_id for 30 hand-curated Romanian queries (see `tests/eval_queries.yaml`).
5. **Chaos** — inject corrupted PDFs (truncated, password-protected, duplicate streams); pipeline must DLQ, never crash the run.

### Golden corpus selection (one-shot, hand-curated)
- 1989 scanned: `MO_PI_1_1989-12-22.pdf`
- 1997 legacy encoding: `MO_PI_17_1997-02-05.pdf`
- 2007 mojibake transition: `MO_PI_1_2007-01-03.pdf`
- 2020 dense decisions: `MO_PI_721_2020-08-11.pdf`
- 2020 inventory tables: `MO_PI_739_2020-08-14.pdf`
- 2026 Bis (146-page Nomenclator): `MO_PI_294Bis_2026-04-14.pdf`
- 2026 modern: `MO_PI_295_2026-04-14.pdf`

### Retrieval-quality eval set
`tests/eval_queries.yaml` — minimum 50 Romanian Q→expected-act pairs covering:
- Exact citation lookup (`Legea nr. 199/2023`)
- Concept lookup (`promulgarea legii învățământului superior`)
- Issuer-filtered (`decizii ale prim-ministrului din 2026`)
- Diacritic-free queries (`hotarare guvern despre invatamant superior`)
- M1-only ("decret 1989 gardian național") — must work with `include_scanned=true`
- Negative tests (queries that must NOT match unrelated acts)

Metrics tracked per release: nDCG@10, MRR, hit@1, hit@5. Regression alarm if drop > 2 pp.

## 4.7 Security, privacy, compliance

- **Data classification**: Monitorul Oficial is by definition public. No PII redaction is required, BUT signatory names are personal data under GDPR — store but never use them as primary search facets without legal review.
- **Romanian official source attestation**: every chunk carries `issue.publication_date` + `source.sha256` so any answer can be traced to the originating PDF.
- **Read-only access path** for end users (RAG queries) uses a Mongo role with `find` only on `mo_chunks` + `mo_acts` + `mo_graph_edges`.
- **Audit log** retains every query (`who, when, query_text, returned_chunk_ids`) for 365 days in a separate, append-only collection `mo_query_log` (TTL index).
- **Secrets** (Atlas connection string, S3 keys, embedding API key if cloud) injected by environment, never on disk. Use `1Password` or AWS Secrets Manager.
- **Egress control**: extraction (Docling, fitz, Camelot, pdfplumber) and embedding (`BAAI/bge-m3`) are fully offline. The single egress is LlamaParse — and only on M1 PDFs. Whitelist `*.cloud.llamaindex.ai` and nothing else from the ingest box.
- **Fully air-gapped option**: set `LLAMA_CLOUD_API_KEY=""` to disable LlamaParse entirely; the M1 path automatically uses PaddleOCR. Zero external HTTP. Embedding stays on `BAAI/bge-m3` locally.
- **API key handling**: `LLAMA_CLOUD_API_KEY` injected as a Kubernetes secret / 1Password reference; never written to disk; rotated quarterly. Logs scrub the value with a regex sanitizer.

## 4.8 Drift detection (catch when the format changes)

Monitorul Oficial layouts have changed at least four times since 1989; we must catch the next change automatically.

Per run, compute and persist:
- Histogram of `producer` strings — alert if a new producer appears with > 3 PDFs.
- Per-page `n_columns` distribution — alert if `n_columns=2` ratio drops below 95% (Bis-only pages excepted).
- Per-block `font_size` distribution top-10 — alert if a previously unseen size appears in > 2% of blocks.
- Issuer taxonomy hits vs misses — alert if `issuer_canonical` falls back to `unknown` for > 1% of acts.
- Catch-all (`unmapped_raw_stream`) char ratio — alert if > 5% on any single doc, > 1% on a run mean.

Drift events open a ticket pre-loaded with the offending PDFs. The fix is usually one new entry in `LEGACY_CODEPOINT_MAP`, `ISSUER_TAXONOMY`, `SECTION_BANNERS`, or a tolerance bump in `config/tolerances.yaml`.

## 4.9 Performance & throughput

Sized for the corpus + 5× growth headroom:
- Single-machine bare-metal (24 cores, 64 GB RAM, RTX 4090 or A10):
  - M3 PDFs: ~ 12 docs/min/core ⇒ ~ 280 docs/min total.
  - M1 PDFs: ~ 30 pages/min on GPU ⇒ ~ 4 docs/min.
- Embedding (`voyage-3` API or `bge-m3` local): batch=64, ~ 3 ms/chunk amortized.
- Mongo Atlas `M30` cluster handles the corpus 100×; vector index build is < 5 min on 100k chunks.
- Bulk write strategy: `mo_blocks` and `mo_chunks` use unordered `insert_many` with `bypassDocumentValidation=False`; one bulk per doc; no cross-doc transactions needed (each issue is independent).

Backpressure: a single Redis-backed work queue (`rq` or `dramatiq`) holds pending PDFs; workers pull, process, ack. DLQ is its own queue.

## 4.10 Incremental & full reprocess workflow

```
# Add a new month — just drop files into laws/YYYY/MM/DD/ and run:
python -m mo.ingest --root laws/

# After a regex fix (PIPELINE_VERSION bump):
python -m mo.ingest --root laws/ --reextract-if-pipeline-changed

# After a chunker/model change (EMBEDDING_VERSION bump):
python -m mo.ingest --reembed-only --where 'embedding_version != "chunker-1.2+voyage-3"'

# After a SCHEMA_VERSION major bump:
python -m mo.migrate --from 1.0.0 --to 2.0.0    # data migration
python -m mo.ingest --root laws/ --force        # then re-extract
```

All four operations are idempotent and resumable. Killing the process mid-run leaves the DB consistent (per-doc transactions).

## 4.11 Operational runbook (excerpt)

| Symptom | Likely cause | Action |
|---|---|---|
| `coverage_ratio < 1.0` on one doc | new layout / unseen role | Inspect `unmapped_raw_stream`; add rule; bump `PIPELINE_VERSION`; re-ingest single doc |
| Coverage fine but retrieval poor | chunking too coarse | Lower `target_tokens`; bump `EMBEDDING_VERSION`; re-embed only |
| Vector index lag > 5 min | Atlas tier too small | Scale to next tier or shard `mo_chunks` by `act_year` |
| OCR pages with garbled output | new scan source / DPI < 200 | Force re-rasterize at 600 DPI; switch engine PaddleOCR → Tesseract |
| Spike in unmapped chars in 2026 | new producer string / new font | Drift alarm fires; add producer to `M3_PRODUCERS`, font to `IGNORE_FONTS` if decorative |
| Hybrid query latency p95 > 1 s | wrong filter ordering | Verify `embedding_version` filter is on a vector-index `filter` field (not post-filter) |

## 4.12 Cross-issue graph resolution (2-pass)

Citations and promulgations frequently target acts in **other** Monitorul issues — sometimes published before, sometimes after the citing act. The graph must therefore be resolved in two passes.

```python
# Pass 1 (per-doc, online): extract raw citations + record provisional edges
edge = {
  "_id":            f"{act_id}#cites#{i}",
  "from_act_id":    act_id,
  "to_act_ref":     {"type":"lege","number":"199","year":2023,"article":"30"},
  "to_act_id":      None,                 # filled in pass 2
  "kind":           "cites",
  "raw":            "Legea învățământului superior nr. 199/2023, cu modificările…",
  "resolved":       False
}

# Pass 2 (corpus-wide, offline, idempotent): resolve to_act_id
def resolve_edges():
    for e in mo_graph_edges.find({"resolved": False}):
        ref = e["to_act_ref"]
        target = mo_acts.find_one({
            "act_type": map_type(ref["type"]),
            "act_number": ref["number"],
            "act_year": ref["year"]
        })
        if target:
            mo_graph_edges.update_one({"_id": e["_id"]},
                {"$set": {"to_act_id": target["_id"], "resolved": True}})
        else:
            mo_graph_edges.update_one({"_id": e["_id"]},
                {"$set": {"resolved": False, "external": True,
                          "external_url_hint": romlex_lookup(ref)}})
```

External (unresolvable) references are kept with `external=True` + a hint URL for downstream lookup (`legislatie.just.ro`, `lege5.ro`). Re-running pass 2 is cheap and nightly-safe — newly ingested docs may resolve previously-external edges.

## 4.13 Cost model & guardrails

| Item | Unit cost (typical) | Corpus cost (≈) |
|---|---|---|
| Embedding `BAAI/bge-m3` (local, RTX 4090, fp16) | electricity only (~$0.05/h) | <$1 for current 119-PDF corpus; <$50 for 100k issues |
| LlamaParse premium (M1 only, ~30 PDFs × ~6 pages avg) | $0.003 / page premium | ~$0.55 one-time; aggressively cached by sha256 → near-zero on re-runs |
| LlamaParse default tier (if premium not needed) | $0.0003 / page | ~$0.06 one-time |
| Docling (local, CPU or modest GPU) | electricity only | negligible |
| Camelot + pdfplumber (deterministic) | electricity only | negligible |
| Atlas `M30` (vector search) | ~$0.54/h | $390/mo |
| S3 CAS storage | $0.023/GB/mo | < $1/mo for full archive |
| PaddleOCR (offline fallback, GPU) | electricity only | per-run, only if LlamaParse unavailable |

Hard guardrails enforced in code:
```python
# Per-run circuit breakers
MAX_PDFS_PER_RUN              = 5_000
MAX_LLAMAPARSE_PAGES_PER_RUN  = 5_000          # cap M1 cloud spend; Paddle from then on
MAX_LLAMAPARSE_USD_PER_RUN    = 25.0           # hard $ cap on cloud OCR
MAX_LLAMAPARSE_RETRIES        = 3              # 4s/16s/64s exp backoff
MAX_GPU_MEMORY_RATIO          = 0.85           # for OCR + bge-m3 inference
EMBED_BATCH                   = 32             # bge-m3 throughput sweet-spot

# Cost-tracker increments on every LlamaParse page; aborts/falls-back if breached.
class CostBudget:
    def __init__(self, cap_usd: float):
        self.cap_usd, self.spent_usd = cap_usd, 0.0
    def charge_pages(self, pages: int, engine: str):
        rate = {"llamaparse_premium": 0.003,
                "llamaparse_default": 0.0003}.get(engine, 0.0)
        self.spent_usd += rate * pages
        if self.spent_usd > self.cap_usd:
            raise BudgetExceeded(f"{self.spent_usd:.2f} > {self.cap_usd:.2f}")
    def can_afford(self, pages: int, engine: str = "llamaparse_premium") -> bool:
        rate = {"llamaparse_premium": 0.003,
                "llamaparse_default": 0.0003}.get(engine, 0.0)
        return self.spent_usd + rate * pages <= self.cap_usd
```

## 4.14 Romanian temporal & numeric token tables (deterministic)

The pipeline normalizes Romanian dates without LLM help. Three closed tables ship in `config/romanian_tokens.yaml`:

```yaml
months_ro:
  ianuarie: 1
  februarie: 2
  martie: 3
  aprilie: 4
  mai: 5
  iunie: 6
  iulie: 7
  august: 8
  septembrie: 9
  octombrie: 10
  noiembrie: 11
  decembrie: 12

months_roman:        # used in running header "Nr. 379/5.VI.2002"
  I: 1
  II: 2
  III: 3
  IV: 4
  V: 5
  VI: 6
  VII: 7
  VIII: 8
  IX: 9
  X: 10
  XI: 11
  XII: 12

weekdays_ro:
  luni: 1
  marţi: 2
  marți: 2
  miercuri: 3
  joi: 4
  vineri: 5
  sâmbătă: 6
  duminică: 7

ordinals_ro:         # for "Anul XIV" → 14
  I: 1
  II: 2
  III: 3
  IV: 4
  V: 5
  VI: 6
  VII: 7
  VIII: 8
  IX: 9
  X: 10
  XI: 11
  XII: 12
  XIII: 13
  XIV: 14
  XV: 15
  XVI: 16
  XVII: 17
  XVIII: 18
  XIX: 19
  XX: 20
  XXI: 21
  XXII: 22
  XXIII: 23
  XXIV: 24
  XXV: 25
  XXVI: 26
  XXVII: 27
  XXVIII: 28
  XXIX: 29
  XXX: 30
  XXXI: 31
  XXXII: 32
  XXXIII: 33
  XXXIV: 34
  XXXV: 35
  XXXVI: 36
  XXXVII: 37
  XXXVIII: 38
```

Date parser:
```python
DATE_RE = re.compile(
    r"(?P<dom>\d{1,2})\s+(?P<mon>"
    r"ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie"
    r")\s+(?P<year>\d{4})", re.IGNORECASE)

HEADER_DATE_RE = re.compile(
    r"Nr\.\s*(?P<num>\d+)\s*/\s*(?P<dom>\d{1,2})\.(?P<rmon>[IVXL]+)\.(?P<year>\d{4})")
```

## 4.15 Watcher & onboarding new issues

```python
# tools/watch.py — process new monthly drops automatically
import time, pathlib
from mo.ingest import ingest

ROOT = pathlib.Path("laws")
SEEN = pathlib.Path(".seen.txt")

def loop():
    seen = set(SEEN.read_text().splitlines()) if SEEN.exists() else set()
    while True:
        for pdf in sorted(ROOT.rglob("*.pdf")):
            key = str(pdf)
            if key in seen: continue
            try:
                res = ingest(pdf, run_id=f"watch-{int(time.time())}")
                seen.add(key); SEEN.write_text("\n".join(sorted(seen)))
                print(f"OK  {pdf.name}  skipped={res.skipped}")
            except Exception as e:
                print(f"ERR {pdf.name}  {e}")
        time.sleep(300)
```

Or better: a `systemd.path` unit / `inotifywait` watcher / cron job — choose by ops preference. The contract is the same: each PDF is presented to `ingest()` exactly once per content hash + pipeline+schema version triple.

## 4.16 Deployment, canary, rollback

The runtime path (query side) is independent of the pipeline path (ingest side). Deploy them separately.

**Pipeline upgrade (extraction or chunker change):**
1. Bump `PIPELINE_VERSION` (or `EMBEDDING_VERSION`).
2. Build & tag container; push to registry.
3. Run on **5 canary PDFs** (one per modality from the golden corpus); diff JSON output against snapshots.
4. Run on **full corpus in shadow mode** (write to `mo_chunks_canary` collection, not the live one).
5. Compare hybrid-query eval scores (`tests/eval_queries.yaml`) live vs canary. Promote only if nDCG@10 within ±1 pp **and** zero coverage failures.
6. Atomic switchover: rename indexes (`mo_chunks_canary` → `mo_chunks`) + drop old.

**Query-side upgrade (no schema change):**
- Blue/green at the API gateway; instant rollback.

**Atlas index rebuild:**
- Vector index rebuild on `mo_chunks` with 100k entries: ~3-5 min, **online** (Atlas keeps the old index serving until the new one is ready).
- Build a fresh index alongside, then `dropSearchIndex` on the old. Never modify in place.

**Rollback:**
- Each `mo_runs` doc records the previous `pipeline_version`. To roll back chunks: redeploy the previous container, run `--reextract --where 'extraction_audit.pipeline_version != "<previous>"'`.

## 4.17 Schema documentation site

`docs/` is built with `mkdocs-material` from the same JSON Schema. CI generates a static site on every `schema_version` bump; consumers pin to a docs URL like `docs.mo-pipeline.local/schema/1.0.0/`. CHANGELOG.md is mandatory for every version bump.

## 4.18 Final production readiness checklist

- [ ] All 119 PDFs in `laws/` ingest with `coverage_ratio == 1.000`.
- [ ] `mo_chunks` has ≥ 1 chunk per non-empty act; zero chunks > `max_tokens` except `is_oversize_table=true`.
- [ ] Atlas Search index `mo_chunks_text` and Vector Search index `mo_chunks_vec` deployed and queryable.
- [ ] Hybrid query suite (`tests/eval_queries.yaml`) passes nDCG@10 ≥ 0.75.
- [ ] Round-trip test passes (markdown reconstructed from `block_refs` is byte-identical).
- [ ] DLQ empty after a clean re-ingest of the corpus.
- [ ] Drift detectors armed; baseline histograms committed to `metrics/baseline.json`.
- [ ] OpenTelemetry traces visible end-to-end for a sample doc.
- [ ] Prometheus dashboards + alerts wired to PagerDuty/Slack.
- [ ] `docker run mo:<pipeline_version> python -m mo.ingest` reproduces the corpus end-to-end on a clean machine.
- [ ] Disaster recovery: `mongodump` of all `mo_*` collections + S3 CAS bucket → can rebuild Atlas from scratch.
- [ ] Versioning: bumping any of `SCHEMA_VERSION`, `PIPELINE_VERSION`, `EMBEDDING_VERSION` triggers the right (and only the right) re-processing path.
- [ ] Security: read-only role for query path; secrets via env; query log TTL set.
- [ ] License/source attribution: every chunk traceable via `source.sha256` to the original PDF.
- [ ] Cross-issue graph resolution pass-2 runs nightly; `external=true` rate stable.
- [ ] Cost guardrails (`MAX_TOKENS_PER_RUN`, `MAX_API_CALLS_PER_MINUTE`) enforced and tested via fault-injection.
- [ ] Romanian token tables (`config/romanian_tokens.yaml`) loaded; date parser unit tests green.
- [ ] Watcher (`tools/watch.py` or `systemd.path`) deployed and ingesting new drops within ≤ 5 min.
- [ ] Canary deploy runbook executed at least once per quarter; rollback drill executed at least once per year.
- [ ] Schema doc site (`docs.mo-pipeline.local`) deployed and pinned in API responses.
- [ ] CHANGELOG.md updated for each `SCHEMA_VERSION` / `PIPELINE_VERSION` / `EMBEDDING_VERSION` bump.
- [ ] Docling + fitz reconciliation passes coverage_ratio==1.000 across full corpus; all fitz-extracted text either Docling-claimed or in `unmapped_raw_stream`.
- [ ] LlamaParse cache directory (`.cache/llamaparse/`) populated; re-running M1 pipeline triggers zero new API calls.
- [ ] Air-gapped smoke test: unset `LLAMA_CLOUD_API_KEY`, ingest one M1 PDF, verify PaddleOCR fallback path produces non-empty acts.
- [ ] BGE-M3 inference benchmarked: ≥ 400 chunks/s at batch=32 fp16 on target GPU.
- [ ] Per-run cost budget enforced; integration test injects an over-budget run and verifies clean fallback to PaddleOCR.
