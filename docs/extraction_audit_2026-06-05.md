# Extraction Audit Report
**Date:** 2026-06-05  
**Pipeline:** Option C (Docling → MD → llama3.1:8b → JSON)  
**Gazettes audited:** 21  
**Total acts in PDFs:** ~220  
**Total extracted acts:** 179 (before fallback merge)  

---

## Summary by Era

| Era | Gazettes | Real acts | Extracted acts | Phantom acts | Missing acts | Accuracy |
|-----|----------|-----------|----------------|--------------|--------------|----------|
| 1989 (scanned) | 6 | 32 | 22 | — | 10 | 93.2% field / 81.8% per-act |
| 2007-Jan (broken encoding) | 3 | 47 | 57 | 10 | 3+ | 76.6% field / ~45% per-act |
| 2007-Dec (modern) | 5 | ~28 | 29 | ~6 | 0 | ~85% |
| 2017 (modern) | 3 | ~42 | 49 | ~12 | 3 | ~80% |
| 2026 (modern) | 4 | ~19 | 21 | 2 | 0 | ~82% |

---

## 1989 Era (Scanned PDFs)

### MO_PI_1_1989-12-22
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| act[0] | title | "" (empty) | "COMUNICATUL CATRE TARA al Consiliului Frontului Salvarii Nationale" | MISSING |
| act[0] | act_number | "1" | No formal number (LLM grabbed gazette issue number) | SUSPICIOUS |
| act[0] | full_text | Truncated mid-sentence | OCR page-edge truncation in source PDF | SUSPICIOUS |

### MO_PI_2_1989-12-25
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| act[0] | full_text | Truncated at "in-" | OCR page-edge truncation | SUSPICIOUS |

**Clean otherwise.**

### MO_PI_3_1989-12-26
**No issues — cleanest extraction in the entire corpus.**

### MO_PI_4_1989-12-27
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| MISSING | — | — | **DECRET-LEGE Nr. 1** "privind abrogarea unor legi, decrete si alte acte normative" (signed Ion Iliescu, page 1) | **MISSING (critical)** |
| act[0] | all fields | DECRET-LEGE Nr. 2 ✓ | correct | OK |
| act[1] | all fields | COMUNICAT ✓ | correct | OK |

**Root cause:** Act boundary failure — the first page of the gazette contains DECRET-LEGE Nr. 1 but the LLM started extracting from Nr. 2. Nr. 1 abrogated large portions of communist-era law and is historically significant.

### MO_PI_5_1989-12-27
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| MISSING | — | — | COMUNICAT (page 1) | **MISSING** |
| MISSING | — | — | **DECRET Nr. 1** "privind numirea primului-ministru" (Petre Roman) | **MISSING (critical)** |
| MISSING | — | — | DECRET Nr. 2 "privind rechemarea generalului Militaru" | MISSING |
| act[1] | act_number | "2" | "4" (DECRET Nr. 4 = securitate stat) | WRONG |
| act[1] | title | "rechemarea unui general" | "trecerea în componenta MApN a Departamentului Securității Statului" | WRONG |
| act[1] | full_text | Merges DECRET Nr. 4 + Nr. 2 bodies | Should be two separate acts | WRONG (structural) |

**Root cause:** LLM failed to segment short decrees on the same page. DECRET Nr. 1 (first Romanian post-communist PM appointment) is entirely absent.

### MO_PI_6_1989-12-29
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| MISSING (4 acts) | — | — | DECRET Nr. 5, Nr. 6, Nr. 7, Nr. 13 entirely absent | **MISSING** |
| act[0] full_text | — | Contains Nr. 5 text appended | Nr. 5 should be a separate act | WRONG |
| act[1] | act_number | "8" | "6" (primary act is Nr. 6 — Min. Economiei) | WRONG |
| act[6] | act_number | "14" | Nr. 13 absorbed into act[6] (Radio TV) | WRONG |
| act[7] | act_number | "16" | Nr. 15 and Nr. 16 merged — Nr. 15 missing | WRONG |
| act[9] | act_number | "19" | Nr. 18 and Nr. 19 merged — Nr. 18 missing | WRONG |

**1989 pattern:** Consecutive short decrees on the same column/page are merged into one act. Longer multi-article decrees extracted correctly.

---

## 2007 January Era (Broken Encoding)

> **Note:** This era has systematic MD encoding corruption — closing signature lines (`Nr. NNNN.`) are often absent or garbled in the Docling output. This is the primary root cause of the errors below.

### MO_PI_1_2007-01-03
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| act[1] | act_number | "162 din 20 decembrie 2006" | "1416" (DECRET 1.416/2006) | WRONG (preamble reference) |
| act[3] | act_number | "162 din 20 decembrie 2006" | "1418" | WRONG |
| act[7] | act_number | "162 din 20 decembrie 2006" | "1422" | WRONG |
| act[9] | act_number | "1422" | "1428" | WRONG (Nr. consumed by act[7]) |
| act[17] | doc_type | DECIZIE | **DCC** (Decizia Curții Constituționale Nr. 831) | WRONG |
| act[17] | issuing_authority | Guvernul României | Curtea Constituțională | WRONG |
| act[17] | act_number | "1816" | "831" | WRONG (HG 1816 header appended by Docling) |
| act[18] | act_number | "1858" | "1816" | WRONG (shift caused by act[17] mis-numbering) |
| act[19] | act_number | "1911" | "1858" | WRONG |
| act[20] | act_number | "1919" | "1911" | WRONG |
| act[21] | act_number | "831" | "1919" | WRONG (DCC number reused) |
| acts[22–28] | all | 7 phantom acts | do not exist | **PHANTOM** (HG body fragments) |

**Act count:** 22 real, 29 extracted, 7 phantom.

### MO_PI_2_2007-01-03
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| act[1] | act_number | "1440" | "1439" (closing of 1440 consumed by adjacent act) | WRONG |
| act[2] | act_number | "E 356 din 13 decembrie 2006" | "1440" | WRONG (preamble reference) |
| acts[6–12] | issuing_authority | "Guvernul României" | "Primul-ministru" (PM decisions) | WRONG (7 acts) |
| act[14] | act_number | "231" | "551/1.475/2006" (joint ORDIN) | WRONG |
| act[14] | issuing_authority | garbled (`ûsi`) | "Min. Agriculturii + MAI" | WRONG (encoding) |
| act[14] | title | garbled | correct title | WRONG (encoding) |
| act[15] | act_number | "232" | no number (COMUNICAT) | WRONG |
| act[15] | issuing_authority | "Parlamentul României – Camera Deputaților" | "Fondul de Garantare a Depozitelor" | WRONG |
| act[16] | all | phantom duplicate of joint ORDIN | does not exist | **PHANTOM** |

**Act count:** 16 real, 17 extracted, 1 phantom.

### MO_PI_3_2007-01-03
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| act[2] | act_number | "1234" | "234" | WRONG (hallucination) |
| acts[1,2] | issuing_authority | "Guvernul României" | "Primul-ministru" | WRONG |
| act[5] | all | phantom (premature ORDIN 1540 extraction) | does not exist | **PHANTOM** |
| acts[9,10] | act_number | "1908" / "2198" | both should be "2224" | WRONG |
| act[10] | all | phantom fragment of ORDIN 2224 | does not exist | **PHANTOM** |

**Act count:** 9 real, 11 extracted, 2 phantom.

---

## 2007 December Era

### MO_PI_820_2007-12-03
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| act[1] | act_number | "1450" | "1449" (swapped with act[2]) | SUSPICIOUS |
| act[2] | act_number | "1449" | "1450" | SUSPICIOUS |

**Act count:** 7 real, 7 extracted. Correct count; numbers for HG 1449 and 1450 swapped.

### MO_PI_821_2007-12-03
**All 9 DCC acts correct — no issues.**  
DCCs 920, 922, 939, 944, 949, 975, 977, 1027, 1028 all extracted correctly.

### MO_PI_822_2007-12-03
**All 2 acts correct — no issues.**  
HG 1418 and ORDIN 2027 extracted correctly.

### MO_PI_823_2007-12-03
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| act[1] | act_number | "1447" | "100" (ANSPDCP DECIZIE) | WRONG (borrowed from HG) |
| act[1] | all | phantom duplicate of DECIZIE 100 | does not exist as separate act | **PHANTOM** |
| act[2] | act_number | "100" | "128" (EXIMBANK HG) | WRONG |
| act[2] | all | phantom duplicate of HG 128 | does not exist as separate act | **PHANTOM** |
| act[3] | DECIZIE 100 | ANSPDCP ✓ | correct | OK |
| act[4] | HG 128 | Comitetul Interministerial ✓ | correct | OK |

**Act count:** 3-4 real (HG 1447, DECIZIE 100, HG 128, + Rectificări), 5 extracted, 2 phantom.

### MO_PI_824_2007-12-03
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| act[1] | act_number | "330" | "1287/330" (joint ORDIN — two ministries) | WRONG (partial number) |

**Act count:** 2 real, 2 extracted. Correct count; joint ORDIN number incomplete.

---

## 2017 Era

### MO_PI_74_2017-01-30
> Gazette contains 30 presidential decrees (Nr. 20–49) for judicial appointments.

| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| MISSING | — | — | **DECRET Nr. 23** | MISSING |
| MISSING | — | — | **DECRET Nr. 29** | MISSING |
| MISSING | — | — | **DECRET Nr. 39** | MISSING |
| act[21] | act_number | "5 din 12 ianuarie 2017" | unknown (likely 23 or 39) | WRONG (CSM resolution ref) |
| act[27] | act_number | "5 din 12 ianuarie 2017" | unknown (likely 29 or 39) | WRONG |
| act[25] | act_number | "45" | should be unique — already assigned to act[17] | WRONG (duplicate) |
| act[30] | act_number | "0" | unrecoverable | UNRECOVERABLE |
| act[30] | doc_type | UNKNOWN | — | UNRECOVERABLE |

**Root cause:** All decrees share an identical preamble "având în vedere Hotărârea CSM nr. 5 din 12 ianuarie 2017" — the LLM grabbed this CSM resolution number instead of the decree number for acts without a visible closing block.

**Act count:** 30 real, 31 extracted (including 1 unrecoverable phantom).

### MO_PI_75_2017-01-30
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| act[2] | act_number | "2.412/C" | "115/C" (this order AMENDS 2412/C — it is not 2412/C) | WRONG |
| acts[3,4] | issuing_authority | Ministerul Educației | Ministerul Justiției | WRONG |
| act[3] | all | likely phantom (duplicate partial extraction of ORDIN 115/C) | does not exist separately | SUSPICIOUS |

**Act count:** 3 real, 5 extracted, ~2 phantom/duplicate.

### MO_PI_76_2017-01-30
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| acts[6,9,10,11,12] | all | 5 phantom duplicate acts | do not exist | **PHANTOM** |
| act[8] | doc_type | LEGE | ORDIN | WRONG |
| act[6] | act_number | "1643" | already assigned to act[3] | WRONG (duplicate) |
| act[9] | act_number | "1641" | already assigned to act[1] | WRONG (duplicate) |

**Act count:** 8 real (ORDIN 1607, 1641–1647), 13 extracted, 5 phantom duplicates. Each real ORDIN was split by the LLM into header + body as separate acts.

---

## 2026 Era

### MO_PI_294_2026-04-14
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| act[4] | act_number | "726" | "206" (HG 206/2026) | WRONG |
| act[0] | title | "Decizia nr. 598" | Full reference clause re art. 328 CPP | SUSPICIOUS |
| act[5] | act_number | "1.605/875" | New order numbers (amending 1605/875/2014) | WRONG |
| act[6] | all | phantom act | does not exist | **PHANTOM** |

### MO_PI_294Bis_2026-04-14
**No issues — single-act gazette (HG 191) extracted correctly including 441K char full_text.**

### MO_PI_295_2026-04-14
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| act[0] | title | "DECIZIA Nr. 576" | Full clause "referitoare la excepția de neconstituționalitate a dispozițiilor art. 973 din Codul civil" | SUSPICIOUS |
| acts[1,2,3] | issuing_authority | "Guvernul României" | "Prim-ministrul" (3 PM decisions) | WRONG |
| act[5] | act_number | "699" | "408" (LLM grabbed number of order being amended) | WRONG |
| act[6] | act_number | "408" | No number (RECTIFICARE) | SUSPICIOUS |

### MO_PI_311_2026-04-20
| Act | Field | Extracted | Expected | Severity |
|-----|-------|-----------|----------|----------|
| act[0] | title | "DECIZIA Nr. 384" | Full clause re OG 7/2010 | SUSPICIOUS |
| act[3] | issuing_authority | ANSVSA | ANMCS (two consecutive ORDINs had authority swapped) | WRONG |
| act[5] | act_number | "20625" | "20022" (AEP RAPORT — Partidul Realitatea) | WRONG |
| act[6] | all | phantom duplicate of Partidul Realitatea RAPORT | does not exist | **PHANTOM** |
| act[7] | act_number | "3428" | "20625" (AEP RAPORT — Partidul Satului Românesc) | WRONG |
| act[8] | doc_type | DECIZIE | RAPORT | WRONG |
| act[8] | act_number | "20625" | "20644" (AEP RAPORT — Partidul Revoluției) | WRONG |

---

## Cross-Cutting Patterns

### 1. Act boundary detection failure (1989, 2007-Jan, 2017)
Short acts on the same page/column are merged into one act object. The LLM assigns the last-seen act number to the merged block. Affected: all 1989 decrees short-form, 2007-Jan HG blocks, 2017 ORDIN blocks in MO_PI_76.

### 2. Preamble reference grabbed instead of act number (2007-Jan, 2017)
When the closing `Nr. NNNN.` block is absent from the MD (broken encoding or Docling layout failure), the LLM falls back to the first reference number in the preamble text. Examples: "162 din 20 decembrie 2006" (CSAP decision), "5 din 12 ianuarie 2017" (CSM resolution).

### 3. Act_number = number of referenced/amended order (2026)
For amendment orders, the LLM extracts the number of the order being amended rather than the new order's own number. Examples: ORDIN 115/C → extracted "2412/C"; ORDIN 408 → extracted "699/2024".

### 4. Phantom acts from long-act re-splitting (2007-Dec, 2017, 2026)
When an act has a long body (annexes, tables), the LLM re-extracts sections as separate acts. MO_PI_76 has 5 phantom duplicates from this pattern; MO_PI_823 has 2; MO_PI_311 has 1.

### 5. DCC title truncation (2007-Dec, 2026)
Constitutional Court decisions frequently get title = "DECIZIA Nr. NNN" instead of the full "referitoare la excepția de neconstituționalitate..." clause. Affects MO_PI_294 act[0], MO_PI_295 act[0], MO_PI_311 act[0], MO_PI_1_2007 act[17].

### 6. Issuing authority: "Guvernul României" for PM decisions (2007-Jan, 2026)
The heading pattern "GUVERNUL ROMÂNIEI / PRIMUL-MINISTRU / DECIZIE" causes the LLM to pick "Guvernul României" instead of "Primul-ministru". Systematic in 2007-Jan G2 (7 acts) and 2026 MO_PI_295 (3 acts).

### 7. Authority swap between consecutive acts (2026)
Two consecutive acts from different authorities (ANMCS/ANSVSA in MO_PI_311) had their issuing_authority values swapped.

### 8. Publisher credit misread as issuing authority (2007-Jan)
The gazette colophon "EDITOR: PARLAMENTUL ROMÂNIEI – CAMERA DEPUTAȚILOR" is misread as the issuing authority for COMUNICAT acts.

---

## Priority Issues (by impact)

| Priority | Gazette | Description |
|----------|---------|-------------|
| P1 | MO_PI_5_1989 | DECRET Nr. 1 (Petre Roman appointed PM) entirely missing |
| P1 | MO_PI_4_1989 | DECRET-LEGE Nr. 1 (abrogare legi comuniste) entirely missing |
| P1 | MO_PI_1_2007 | DCC 831 misclassified as DECIZIE + wrong authority + wrong number |
| P2 | MO_PI_1_2007 | 7 phantom acts (HG fragments), 4 wrong act_numbers |
| P2 | MO_PI_6_1989 | 4 missing decrees (Nr. 5,6,7,13); 4 merged/wrong number pairs |
| P2 | MO_PI_74_2017 | 3 missing decrees, 2 malformed act_numbers, 1 UNKNOWN act |
| P2 | MO_PI_76_2017 | 5 phantom duplicates, 1 wrong doc_type |
| P3 | MO_PI_75_2017 | ORDIN 115/C extracted as 2412/C; wrong authority on 2 acts |
| P3 | MO_PI_295_2026 | ORDIN 408 extracted as 699; 3 PM decisions with wrong authority |
| P3 | MO_PI_311_2026 | AEP RAPORT section scrambled (3 wrong numbers, 1 phantom, 1 wrong type) |
| P3 | MO_PI_823_2007 | 2 phantom duplicates, wrong nr on phantom |
| P4 | MO_PI_2_2007 | Joint ORDIN 551/1475 — wrong number + garbled encoding |
| P4 | MO_PI_294_2026 | HG 206 extracted as 726; phantom act |

---

## Recommendations for gemma4:12b Switch

All P1/P2 issues fall into categories where Gemma 4 12B's improvements are directly relevant:
- **Better instruction following** → fixes phantom act over-splitting and preamble number confusion
- **256K context** → eliminates pressure on long-act truncation 
- **Stronger reasoning** → better distinction between "this is the act number" vs "this is a referenced number"
- **Romanian-native tokenizer** → better handling of diacritic-heavy authority names

The broken_2007 encoding corruption (2007-Jan era) is an OCR problem upstream of the LLM — a model switch will not fix it. A sumar-based fallback for `Nr.` extraction would be needed there.
