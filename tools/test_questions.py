#!/usr/bin/env python3
"""Sequential test of all questions against the LegalRo RAG system."""
import subprocess
import re
import time

QUESTIONS = [
    ("Q4",  "Ce județe sunt vizate de ordinele ANCPI nr. 1.642 și nr. 1.643 din decembrie 2016 privind închiderea vechilor evidențe de cadastru și publicitate imobiliară, publicate în MO nr. 76/30.I.2017?",
             "Vâlcea și Mureș"),
    ("Q5",  "Cine a fost numit secretar de stat la Ministerul Sănătății Publice prin Decizia prim-ministrului nr. 226 din 28 decembrie 2006 și cine a contrasemnat decizia?",
             "Armean Petru; contrasemnat de Ion-Mircea Plângu"),
    ("Q6",  "Cine a fost eliberat din funcția de președinte, cu rang de secretar de stat, al Oficiului pentru Licență Industrială prin Decizia prim-ministrului nr. 117/2026?",
             "Ionuț-Mihai Rădoi"),
    ("Q7",  "Care era suma contribuției anuale a României la grupul de lucru OCDE privind investițiile internaționale, aprobată prin HG nr. 420/2005, și din ce buget se asigura contravaloarea în lei?",
             "3.000 euro, din bugetul Agenției Române pentru Investiții Străine"),
    ("Q8",  "Ce articol din Legea nr. 47/1992 a fost contestat prin excepția de neconstituționalitate soluționată prin Decizia CCR nr. 922/2007?",
             "Art. 29 alin. (6) din Legea nr. 47/1992"),
    ("Q9",  "Cine a ridicat excepția de neconstituționalitate soluționată prin Decizia CCR nr. 922/2007 și la ce instanță?",
             "George Daniel Subțirelu, Tribunalul București, Secția a V-a civilă, Dosar nr. 22.190/299/2006"),
    ("Q10", "Cine era președintele Curții Constituționale la data pronunțării Deciziei nr. 922/2007?",
             "Ioan Vida"),
    ("Q11", "Ce reglementează Ordinul ministrului internelor și reformei administrative nr. 346/2007, publicat în Monitorul Oficial nr. 820/3.XII.2007?",
             "Normele metodologice pentru Legea nr. 38/2003 privind transportul în regim de taxi și în regim de închiriere"),
    ("Q12", "Ce ordin anterior abrogă Ordinul nr. 346/2007 privind normele metodologice pentru taxi, conform art. 2 alin. (2)?",
             "Ordinul ministrului administrației publice nr. 275/2003"),
    ("Q13", "Cine a fost numit în funcția de secretar de stat la Ministerul Economiei, Digitalizării, Antreprenoriatului și Turismului prin Decizia nr. 116/2026?",
             "Dan-Adrian Pop"),
    ("Q14", "Ce s-a întâmplat cu Gabriel-Bogdan Ștețco în urma Deciziilor prim-ministrului nr. 115 și nr. 118 din aprilie 2026?",
             "Eliberat prin Decizia nr. 115 (secretar de stat la ME); numit prin Decizia nr. 118 (președinte OLI)"),
    ("Q15", "Ce grad militar a fost acordat colonelului Rus Iosif Alexandru prin Decretul prezidențial nr. 1.418/2006?",
             "General de brigadă cu o stea"),
    ("Q16", "Ce hotărâre a Guvernului a înființat Zona liberă Galați și Regia Autonomă Administrația Zonei Libere Galați, modificată prin HG nr. 1.908/2006?",
             "HG nr. 190/1994, publicată în MO nr. 120 din 13 mai 1994"),
    ("Q17", "Cine a semnat HG nr. 1.908/2006 privind extinderea regimului de zonă liberă în porturile Galați, Brăila și Constanța?",
             "Călin Popescu-Tăriceanu"),
    ("Q18", "Cine a fost numit în funcția de președinte al Autorității Naționale pentru Reglementare în Comunicații și Tehnologia Informației prin Decizia prim-ministrului nr. 234/2006?",
             "Dan Cristian Georgescu"),
    ("Q19", "Cine a fost eliberat din funcția de secretar de stat la Ministerul Economiei și Comerțului prin Decizia prim-ministrului nr. 233/2006?",
             "Kramer Alpar"),
    ("Q20", "Ce articol legal a fost contestat prin excepția de neconstituționalitate soluționată prin Decizia CCR nr. 920/2007?",
             "Art. II alin. (3) din Legea nr. 219/2005"),
    ("Q21", "Cine a ridicat excepția de neconstituționalitate soluționată prin Decizia CCR nr. 920/2007 și la ce instanță?",
             "Maximilian Muntean, Curtea de Apel București, Secția a IV-a civilă, Dosar nr. 20.683/1/2005"),
    ("Q22", "Cine a semnat HG nr. 1.447/2007 privind aprobarea Normelor financiare pentru activitatea sportivă și ce hotărâre a abrogat?",
             "Călin Popescu-Tăriceanu; a abrogat HG nr. 484/2003"),
    ("Q23", "Care este suma maximă per persoană decontabilă pentru o masă oficială la încheierea competițiilor sportive internaționale desfășurate în țară, conform Normelor financiare aprobate prin HG nr. 1.447/2007?",
             "90 lei de persoană"),
    ("Q24", "Care este suma maximă per persoană per joc sau reuniune pentru băuturi răcoritoare asigurate sportivilor la competițiile sportive organizate în țară, conform HG nr. 1.447/2007?",
             "5 lei/persoană/joc sau reuniune"),
    ("Q25", "Ce lege stă la baza Normelor metodologice aprobate prin Ordinul nr. 346/2007 privind transportul în regim de taxi, și ce act normativ abrogă ordinul la data intrării sale în vigoare?",
             "Legea nr. 38/2003; abrogă Ordinul nr. 275/2003"),
    ("Q26", "Ce documente trebuie să prezinte un operator de transport persoană juridică pentru eliberarea autorizației de transport în regim de taxi, conform Ordinului nr. 346/2007?",
             "Cerere, copie licență transport, copie certificat înregistrare, certificat constatator registrul comerțului, declarație pe propria răspundere"),
    ("Q27", "Ce act normativ aprobă Nomenclatorul domeniilor și al specializărilor/programelor de studii universitare și structura instituțiilor de învățământ superior pentru anul universitar 2026-2027, publicat în MO nr. 294 bis/14.IV.2026?",
             "HG nr. 191/2026"),
    ("Q28", "Pe ce temei legal a fost adoptată HG nr. 191/2026 privind Nomenclatorul domeniilor universitare pentru 2026-2027?",
             "Art. 108 din Constituție și art. 30 alin. (8) din Legea nr. 199/2023"),
    ("Q29", "Ce județe sunt vizate de ordinele ANCPI nr. 1.644 și nr. 1.645 din decembrie 2016 privind închiderea vechilor evidențe de cadastru, publicate în MO nr. 76/30.I.2017?",
             "Hunedoara și Călărași"),
    ("Q30", "Ce act normativ constituie temeiul legal pentru închiderea vechilor evidențe de cadastru și publicitate imobiliară de către ANCPI, invocat în ordinele din MO nr. 75/30.I.2017?",
             "Legea cadastrului nr. 7/1996 (art. 11 alin. (2) lit. p), art. 15 alin. (1), art. 3 alin. (13))"),
    ("Q31", "Ce județ vizează Ordinul directorului general al ANCPI nr. 1.642/2016 privind închiderea evidențelor de cadastru publicat în MO nr. 76/30.I.2017?",
             "Județul Vâlcea"),

    # ── MO_PI_1_1989 ─────────────────────────────────────────────────────────
    ("Q32", "Prin ce act publicat în Monitorul Oficial nr. 1 din 22 decembrie 1989 a anunțat Consiliul Frontului Salvării Naționale că toate ministerele și organele centrale trebuie să i se subordoneze?",
             "Comunicatul Consiliului Frontului Salvării Naționale"),
    ("Q33", "Ce organ era chemat să asigure ordinea publică împreună cu comitetele cetățenești, conform comunicatului FSN publicat în MO nr. 1/22.XII.1989?",
             "Miliția"),

    # ── MO_PI_2_1989 ─────────────────────────────────────────────────────────
    ("Q34", "Ce mesaj a transmis Consiliul Frontului Salvării Naționale în comunicatul din 25 decembrie 1989, publicat în MO nr. 2/1989?",
             "Revoluția a învins"),
    ("Q35", "Ce s-a hotărât în comunicatul FSN din 25 decembrie 1989 privind Ministerul de Interne, publicat în MO nr. 2/1989?",
             "Unitățile Ministerului de Interne se vor integra Ministerului Apărării Naționale"),

    # ── MO_PI_3_1989 ─────────────────────────────────────────────────────────
    ("Q36", "Ce capete de acuzare au fost reținute la procesul lui Nicolae și Elena Ceaușescu din 25 decembrie 1989, menționat în comunicatul publicat în MO nr. 3/26.XII.1989?",
             "Genocid peste 60000 victime; subminarea puterii de stat; distrugerea bunurilor obștești; subminarea economiei naționale; tentativă de fugă cu fonduri de peste 1 miliard dolari"),

    # ── MO_PI_4_1989 ─────────────────────────────────────────────────────────
    ("Q37", "Din câți membri era compus Consiliul Frontului Salvării Naționale conform Decretului-lege publicat în MO nr. 4/27.XII.1989?",
             "145 membri"),
    ("Q38", "Cum a fost redenumită Miliția conform decretului-lege publicat în Monitorul Oficial nr. 4 din 27 decembrie 1989?",
             "Poliție"),

    # ── MO_PI_5_1989 ─────────────────────────────────────────────────────────
    ("Q39", "Cine a fost rechmat în cadrele active ale armatei prin Decretul nr. 2 semnat de Ion Iliescu și publicat în MO nr. 5/27.XII.1989?",
             "General-colonel în rezervă Nicolae Militaru"),
    ("Q40", "Ce funcție a primit general-colonelul Nicolae Militaru prin Decretul nr. 3 din 26 decembrie 1989, publicat în MO nr. 5/1989?",
             "Ministru al apărării naționale"),

    # ── MO_PI_6_1989 ─────────────────────────────────────────────────────────
    ("Q41", "Ce minister a fost înființat prin decret publicat în MO nr. 6/29.XII.1989 pentru gestionarea apelor, pădurilor și mediului înconjurător?",
             "Ministerul Apelor, Pădurilor și Mediului Înconjurător"),
    ("Q42", "Ce minister nou a fost înființat prin decret publicat în MO nr. 6/29.XII.1989 pentru economia națională?",
             "Ministerul Economiei Naționale"),

    # ── MO_PI_2_2007-01-03 ───────────────────────────────────────────────────
    ("Q43", "Ce tratat internațional a fost supus aprobării Parlamentului prin Decretul nr. 1440/2006, publicat în MO nr. 2/3.I.2007?",
             "Amendamentul la Convenția privind comerțul internațional cu specii sălbatice CITES adoptat la Gaborone 30 aprilie 1983"),
    ("Q44", "Prin ce decret prezidențial din 2006 publicat în MO nr. 2/3.I.2007 s-a conferit Ordinul național Steaua României în grad de Cavaler, la propunerea guvernatorului BNR?",
             "Decretul nr. 1441"),

    # ── MO_PI_822_2007-12-03 ─────────────────────────────────────────────────
    ("Q45", "Ce hotărâre de guvern privind securitatea aeronautică a fost modificată prin HG nr. 1448/2007 publicată în MO nr. 822/3.XII.2007?",
             "HG nr. 443/2005"),
    ("Q46", "Ce număr are hotărârea de guvern privind modificarea Programului național de securitate aeronautică, publicată în MO nr. 822/3.XII.2007?",
             "HG nr. 1448"),

    # ── MO_PI_824_2007-12-03 ─────────────────────────────────────────────────
    ("Q47", "Ce lege a stat la baza Ordinului nr. 353/2007 al Ministerului Internelor privind normele de aplicare a serviciilor de transport public local, publicat în MO nr. 824/3.XII.2007?",
             "Legea serviciilor de transport public local nr. 92/2007"),
    ("Q48", "Ce normative aprobă Ordinul nr. 353/2007 al ministrului internelor publicat în MO nr. 824/3.XII.2007?",
             "Normele de aplicare a Legii serviciilor de transport public local nr. 92/2007"),

    # ── MO_PI_74_2017-01-30 ──────────────────────────────────────────────────
    ("Q49", "Cine a fost numită judecător la Judecătoria Zalău prin Decretul prezidențial nr. 20 din 26 ianuarie 2017, publicat în MO nr. 74/30.I.2017?",
             "Geanina-Ioana Marincaș"),
    ("Q50", "Cine a fost numit judecător la Judecătoria Sectorului 4 București prin Decretul prezidențial nr. 24 din 26 ianuarie 2017, publicat în MO nr. 74/30.I.2017?",
             "George-Alexandru Lazăr"),

    # ── MO_PI_294_2026-04-14 (distinct from 294Bis) ──────────────────────────
    ("Q51", "La ce dispoziții din Codul de procedură penală se referă excepția de neconstituționalitate soluționată prin Decizia CCR nr. 598 din 11 noiembrie 2025, publicată în MO nr. 294/14.IV.2026?",
             "Art. 328 alin. (1) fraza întâi și art. 347 alin. (4) din Codul de procedură penală"),
    ("Q52", "Ce hotărâre de guvern a fost modificată prin HG nr. 205/2026 privind organizarea Ministerului Energiei, publicată în MO nr. 294/14.IV.2026?",
             "HG nr. 316/2021"),

    # ── MO_PI_311_2026-04-20 ─────────────────────────────────────────────────
    ("Q53", "Ce lege a făcut obiectul excepției de neconstituționalitate soluționate prin Decizia CCR nr. 574 din 6 noiembrie 2025, publicată în MO nr. 311/20.IV.2026?",
             "Legea nr. 165/2013 privind măsurile pentru finalizarea procesului de restituire a imobilelor preluate abuziv"),
    ("Q54", "Ce ordonanță de guvern a constituit obiectul excepției de neconstituționalitate din Decizia CCR nr. 384/2025 publicată în MO nr. 311/20.IV.2026, privind regimul drumurilor?",
             "Art. II alin. (1) din Ordonanța Guvernului nr. 7/2010 pentru modificarea OG nr. 43/1997 privind regimul drumurilor"),
]

NOISE = ["Warning", "matched successfully", "seq_len", "it/s", "deprecated", "cache"]

_NOISE_PAT = re.compile("|".join(re.escape(n) for n in NOISE))


def _clean(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not _NOISE_PAT.search(ln)]
    return "\n".join(lines).strip()


def _score(answer: str, expected: str) -> str:
    """
    Keyword-based scoring: split expected on punctuation/spaces into tokens,
    count how many appear in the (lowercased, diacritic-folded) answer.
    Returns CORECT / PARTIAL / GRESIT.
    """
    import unicodedata

    def fold(s: str) -> str:
        s = unicodedata.normalize("NFD", s.lower())
        return "".join(c for c in s if unicodedata.category(c) != "Mn")

    tokens = re.findall(r"[\w]+", expected)
    # filter very short / stopword tokens
    tokens = [t for t in tokens if len(t) >= 3]
    if not tokens:
        return "NECUNOSCUT"

    folded_answer = fold(answer)
    hits = sum(1 for t in tokens if fold(t) in folded_answer)
    ratio = hits / len(tokens)

    if ratio >= 0.6:
        return "CORECT"
    elif ratio >= 0.3:
        return "PARTIAL"
    else:
        return "GRESIT"


results = []
for qid, question, expected in QUESTIONS:
    print(f"\n{'='*60}")
    print(f"=== {qid} ===")
    print(f"Întrebare: {question}")
    print(f"Așteptat:  {expected}")
    print("--- Răspuns ---")

    result = subprocess.run(
        ["uv", "run", "legalro", "query", "--no-agentic", question],
        capture_output=True, text=True, timeout=300
    )

    answer = _clean(result.stdout)
    print(answer)

    if result.returncode != 0:
        err = _clean(result.stderr)[-500:] if result.stderr else "(no stderr)"
        print(f"[EROARE]: {err}")
        status = "EROARE"
    else:
        status = _score(answer, expected)
        print(f"[{status}]")

    results.append((qid, status, expected, answer))
    time.sleep(8)

print("\n" + "="*60)
print("SUMAR REZULTATE")
print("="*60)
counts = {"CORECT": 0, "PARTIAL": 0, "GRESIT": 0, "EROARE": 0, "NECUNOSCUT": 0}
for qid, status, expected, answer in results:
    counts[status] = counts.get(status, 0) + 1
    print(f"{qid}: {status}")
    if status not in ("CORECT",):
        print(f"     Așteptat: {expected}")

total = len(results)
print(f"\nTotal: {total}")
print(f"  CORECT:  {counts['CORECT']}/{total}")
print(f"  PARTIAL: {counts['PARTIAL']}/{total}")
print(f"  GRESIT:  {counts['GRESIT']}/{total}")
print(f"  EROARE:  {counts['EROARE']}/{total}")
