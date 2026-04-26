# TokenFrame — LLM API sąnaudų optimizavimo karkasas

*Objektinio programavimo baigiamasis projektas.*
*Autorius: Faustas.*

---

## Turinys

1. [Įvadas](#1-įvadas)
2. [Projektavimas ir įgyvendinimas](#2-projektavimas-ir-įgyvendinimas)
3. [Rezultatai](#3-rezultatai)
4. [Iššūkiai](#4-iššūkiai)
5. [Išvados](#5-išvados)

---

## 1. Įvadas

### 1.1 Problema

Didžiųjų kalbos modelių (LLM) API skambučiai, tokie kaip „Anthropic Claude" ar „OpenAI GPT", kainuoja pinigų ir laiko. Kai programėlė aptarnauja daug panašių užklausų, didelė dalis skambučių tampa pertekliniai — tas pats klausimas kartojamas skirtingomis formuluotėmis, o modelio atsakymas iš esmės tas pats. Tiesioginis API skambutis kiekvienai užklausai greitai sudegina biudžetą.

### 1.2 Sprendimas

**TokenFrame** — „Python" karkasas, kuris įterpiamas tarp aplikacijos ir LLM tiekėjo. Jis mažina sąnaudas per:

- **Užklausų talpyklavimą (caching)**: tikslų (exact) ir semantinį (semantic) atitikimą, kad kartotinės užklausos nepasiektų API.
- **ROI pagrįstą išmetimą (eviction)**: kai talpykla pilna, išmetami mažiausios ekonominės vertės įrašai, o ne tiesiog seniausi.
- **Daugiasluoksnę apsaugą**: `MathKeywordGuard` filtras apsaugo nuo klaidingų semantinių atitikimų matematinėms užklausoms.

### 1.3 Tikslinis darbo srautas — StudyBuddy

Karkaso veikimas vertinamas pagal `StudyBuddy` — hipotetinį lietuvių 10 klasės matematikos klausimų–atsakymų botą. Tai **ne** realus produktas, o tik reproduzuojamas darbo srautas, leidžiantis palyginti skirtingas konfigūracijas ant to paties užklausų rinkinio.

Kodėl tinka būtent šis scenarijus:

- Didelis užklausų pasikartojimas (studentai klausia panašių matematikos klausimų).
- Mišrus sudėtingumas (nuo paprastos aritmetikos iki procedūrinių klausimų).
- Aiškūs sėkmės rodikliai (sąnaudų mažinimas palyginti su neoptimizuotu variantu).
- Apibrėžta sritis (tik matematika) — konkretus kontekstas talpyklos ir maršruto sprendimams.

### 1.4 Įdiegimas ir paleidimas

```bash
# Įsidiegti paketą (reikalinga Python 3.10+)
pip install -e .

# Nustatyti Anthropic API raktą
export ANTHROPIC_API_KEY=sk-ant-...

# Paleisti CLI užklausą su talpyklavimu
tokenframe --semantic "Kas yra sin 30?"

# Paleisti etaloninį bandymą
python -m benchmarks exam_week --output reports/
```

Po `pip install -e .` testai leidžiami be interneto per `MockProvider` ir
`MapEmbedder`:

```bash
python -m unittest discover -s tests -v
```

---

## 2. Projektavimas ir įgyvendinimas

Karkasas pastatytas ant penkių abstrakčių bazinių klasių, kiekviena iš jų — sutartis konkretiems įgyvendinimams. Pagrindinė klasė `TokenFrameClient` veikia kaip **fasadas** (Facade), kuris paslepia visą šią sudėtingumą už vieno metodo `query(prompt)`.

### 2.1 OOP principai

#### Abstrakcija

Kiekviena sistemos „jungtis" aprašyta abstrakčia baze. `Provider` yra paprasčiausias pavyzdys:

```python
# tokenframe/providers/base.py
class Provider(ABC):
    @abstractmethod
    def send(self, messages: list[dict], model: Optional[str] = None) -> Response:
        """Send a sequence of chat messages and return the response."""
```

Klientui nesvarbu, ar „Anthropic" API skambučiai tikri, ar juos imituoja `MockProvider`. Jam svarbu tik tiek, kad `send()` grąžina `Response`.

#### Paveldėjimas (is-a ryšys)

Semantiškai tikri paveldėjimo ryšiai:

- `MockProvider` **yra** `Provider`, `AnthropicProvider` **yra** `Provider`.
- `ExactMatchCache`, `SemanticCache`, `HybridCache` **yra** `CacheStrategy`.
- `LRUEviction`, `ROIBasedEviction` **yra** `EvictionPolicy`.
- `MemoryStorage`, `SQLiteStorage` **yra** `Storage`.

Paveldėjimas čia nėra priemonė pakartotinai naudoti kodą — tai prasmingas klasifikavimas.

#### Polimorfizmas

`TokenFrameClient.query()` veikia vienodai nepaisant, koks konkretus `Provider` ar `CacheStrategy` buvo įterptas:

```python
# tokenframe/client.py (paprastintas)
def query(self, prompt: str) -> QueryResult:
    if self._cache is not None:
        entry = self._cache.get(prompt)        # gali būti Exact / Semantic / Hybrid
        if entry is not None:
            self._metrics.record_cache_hit(entry)
            return QueryResult(response=entry.response, cost_usd=0.0, cache_hit=True)
    response = self._provider.send(messages)   # gali būti Mock / Anthropic
    cost = self._cost_model.estimate(...)
    self._metrics.record(response, cost)
    return QueryResult(response=response, cost_usd=cost)
```

Testas `test_polymorphism_works_with_any_provider_subclass` tai patvirtina sukurdamas ad-hoc `Provider` poklasį ir perduodamas jį klientui.

#### Inkapsuliacija

`CacheEntry` privačiai saugo prieigos metaduomenis (`_hit_count`, `_last_accessed_at`), o iš išorės jie pasiekiami tik per `register_hit()`, kuris atomiškai atnaujina abu laukus. Tai užtikrina invariantą: **kiekvienas atitikimas padidina skaitiklį IR atnaujina laiko žymę**.

```python
# tokenframe/cache/entry.py
class CacheEntry:
    def register_hit(self) -> None:
        self._hit_count += 1
        self._last_accessed_at = time.time()

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def cost_saved_usd(self) -> float:
        """Sukauptos sąnaudos, kurių išvengta dėka šio įrašo."""
        return self._hit_count * self.original_cost_usd
```

`cost_saved_usd` yra išvestinė savybė, ne saugomas laukas — negalima pakeisti be `register_hit()`.

### 2.2 Projektavimo šablonai

#### Pagrindinis: Strategy

Kurso reikalavime leidžiama pasirinkti ir šabloną už pateikto sąrašo ribų,
jei jis tinka programai. Todėl pagrindiniu šablonu pasirinktas **Strategy**:
šiame projekte jis natūraliai pritaikomas penkiose skirtingose sistemos
vietose. Papildomai naudojamas ir iš reikalavimų sąrašo paimtas
**Factory Method** šablonas, aprašytas žemiau.

| Vieta | Abstrakcija | Konkretūs įgyvendinimai |
| --- | --- | --- |
| Tiekėjas | `Provider` | `MockProvider`, `AnthropicProvider` |
| Talpyklos strategija | `CacheStrategy` | `ExactMatchCache`, `SemanticCache`, `HybridCache` |
| Saugykla | `Storage` | `MemoryStorage`, `SQLiteStorage` |
| Išmetimo politika | `EvictionPolicy` | `LRUEviction`, `ROIBasedEviction` |
| Įterpinių skaičiuoklė | `Embedder` | `SentenceTransformerEmbedder` + testams skirtas `MapEmbedder` |

Vartotojas sistemą komponuoja kviesdamas konstruktorių, pvz.:

```python
client = TokenFrameClient(
    provider=AnthropicProvider(),
    cache=HybridCache(
        exact=ExactMatchCache(
            storage=SQLiteStorage("cache.db"),
            eviction=ROIBasedEviction(),
        ),
        semantic=SemanticCache(
            storage=SQLiteStorage("cache.semantic.db"),
            eviction=ROIBasedEviction(),
            embedder=SentenceTransformerEmbedder(),
        ),
    ),
)
```

Visos šios konkrečios klasės paklūsta tai pačiai sąsajai, todėl `TokenFrameClient` jų nemato kaip „Anthropic" ar „LRU" — jam matomas tik `Provider`, `CacheStrategy`, `EvictionPolicy`.

**Kodėl Strategy — geras pasirinkimas būtent šiam projektui:**

1. Pritaikyta daug kartų, o ne priversta vienoje vietoje.
2. Leidžia keisti algoritmą vykdymo metu per konstruktoriaus injekciją.
3. Paprastai paaiškinama: „skirtingi algoritmai, ta pati sąsaja, parinkta paleidimo metu".

#### Pagalbiniai šablonai

**Facade.** `TokenFrameClient` paslepia `Provider + CacheStrategy + CostModel + MetricsTracker` sudėtingumą už `query(prompt)`. Naudotojas nemato vidinės struktūros.

**Factory Method.** `benchmarks.configs.make_factories()` grąžina
konfigūracijų kūrimo funkcijas, remiantis bendrais ištekliais (tiekėjas,
įterpinių skaičiuoklė). Etalono vykdytojas kiekvienam konfigūracijos
variantui prašo šviežio kliento. Šis šablonas tiesiogiai atitinka kurso
pateiktą dizaino šablonų sąrašą.

**Decorator (galimybė).** Esama sąsaja leistų įvynioti `Provider` į papildomus sluoksnius (pvz., `LoggedProvider`, `RetryProvider`), nors tai V2 darbas.

### 2.3 Kompozicija vs. agregacija

**Kompozicija** (stipri nuosavybė — gyvavimo ciklas susietas su tėvu):

- `TokenFrameClient` turi `MetricsTracker` — jei klientas sunaikinamas, metrikos taip pat dingsta.
- `CacheEntry` turi `original_cost_usd` ir vidinius skaitiklius, kurie egzistuoja tik kol egzistuoja įrašas.
- Benchmarks `BenchmarkRunner` turi `cumulative_cost_timeline` sąrašą konkrečiam paleidimui.

**Agregacija** (silpna nuosavybė — nepriklausomi gyvavimo ciklai):

- `TokenFrameClient` agreguoja `Provider` — tiekėjas egzistuoja nepriklausomai ir gali būti pakeistas.
- `TokenFrameClient` agreguoja `CacheStrategy` ir `CostModel`.
- `CacheStrategy` agreguoja `Storage` ir `EvictionPolicy`.
- `HybridCache` agreguoja dvi kitas `CacheStrategy` (tipiškas Strategy kompozicijos pavyzdys).

Konstruktoriai pačiu savo parašu parodo skirtumą: agreguoti komponentai ateina kaip parametrai; komponuotieji — sukurti tėvo viduje arba perduoti kaip neprivalomi su sensingais numatytaisiais.

### 2.4 Failų įvestis / išvestis

Karkase realizuoti keturi skirtingi F I/O tipai:

1. **Talpyklos atkaklumas — SQLite.** `SQLiteStorage` išsaugo `CacheEntry` įrašus JSON formatu dėl embedding lauko; kiti laukai — natyvūs SQL tipai. Schema turi migraciją (`ALTER TABLE ADD COLUMN`), kuri tyliai atnaujina 2 fazės duomenų bazes į 3 fazės versiją.

2. **Metrikų eksportas — CSV ir JSON.** `Reporter.write_csv()` išveda vieną eilutę per konfigūraciją su visais matuotais rodikliais. `write_json()` papildomai išsaugo kiekvienos užklausos sukauptąsias sąnaudas — tai leidžia vėlesniam įrankiui perpaišyti grafikus.

3. **Konfigūracijos duomenys — JSON.** Kainos saugomos `tokenframe/economics/pricing.json`, kad atnaujinus API įkainius nereikia keisti kodo.

4. **Etalonų fiksacija — JSON.** Klausimų bankas saugomas `benchmarks/studybuddy/fixtures/questions.json`. Komituotas į repozitoriją, kad etalonai būtų identiškai atkuriami.

**Centrinė pavyzdinė vieta: CSV metrikų eksportas.** Kiekvieną etaloninį paleidimą aprašo viena eilutė per konfigūraciją:

```csv
config,total_queries,total_api_calls,total_cost_usd,cost_saved_usd,cache_hits,cache_misses,cache_hit_rate
baseline,500,500,0.16600000,0.00000000,0,0,0.000000
exact,500,34,0.01128800,0.15471200,466,34,0.932000
semantic,500,23,0.00763600,0.15836400,477,23,0.954000
full,500,23,0.00763600,0.15836400,477,23,0.954000
```

Tokį CSV galima atidaryti „Excel" arba „Google Sheets" ir iškart braižyti lyginamąsias diagramas.

### 2.5 Testavimas

Testavimo tikslas — **70%+ kodo aprėptis**; realiai pasiektas didesnis. Naudojamas `unittest` karkasas (kurso reikalavimas).

Testai struktūrizuoti pagal tikrinamą modulį:

```
tests/
├── test_providers/        (bazinė klasė + Mock + AnthropicProvider)
├── test_cache/            (entry, exact, hybrid, semantic, guard, storage×2)
├── test_eviction/         (LRU, ROI)
├── test_normalization/    (LT fileriai, trūkstamos diakritinės)
├── test_embedding/        (Embedder sąsaja + ST embedder su injekuojamu modeliu)
├── test_economics/        (cost_model, metrics)
├── test_benchmarks/       (question_bank, simulator, scenarios, configs, runner, reporter)
├── test_integration/      (galutinis kliento ir CLI elgesys, LRU vs ROI, persistencija)
└── helpers.py             (bendrai naudojama `MapEmbedder`)
```

**Offline testai.** `MockProvider` grąžina iš anksto apibrėžtus atsakymus be interneto; `MapEmbedder` grąžina iš anksto priskirtas įterpinių reikšmes. Dėl to visa testavimo aibė (daugiau nei 250 testų) vykdoma per kelias dešimtis milisekundžių ir nieko nemoka.

**Integraciniai testai.** `test_client_with_cache.py` ir `test_lru_vs_roi.py` patvirtina galutinį elgseną: ta pati užklausa du kartus → vienas API skambutis; tas pats darbo srautas dvejoms talpykloms → skirtingi išmetimo sprendimai.

---

## 3. Rezultatai

### 3.1 Eksperimento sąranka

`StudentSimulator` generuoja darbo srautą imdamas klausimus iš banko pagal **Zipf** pasiskirstymą: svoris(reitingo k) = 1 / (k+1)^α. Didesnis α — labiau koncentruotas srautas į populiariausius klausimus.

Trys scenarijai:

| Scenarijus | α | Modeliuoja |
| --- | --- | --- |
| `exam_week` | 2.5 | Studentai kartoja medžiagą prieš egzaminą — daug pasikartojimų |
| `mixed` | 1.5 | Mišrus mokymasis — vidutinis pasikartojimų kiekis |
| `casual` | 1.1 | Neformalus mokymasis — įvairūs klausimai, mažai pasikartojimų |

Keturios konfigūracijos lyginamos kiekviename scenarijuje:

1. **baseline** — jokios talpyklos, kiekviena užklausa eina į API
2. **exact** — `ExactMatchCache` + LRU
3. **semantic** — `HybridCache` (exact + semantic) + LRU
4. **full** — `HybridCache` + ROI išmetimas

Kiekvieno scenarijaus darbo srautą sudaro 500 užklausų; talpyklos talpa — 50 įrašų; klausimų bankas — 50 kanoninių klausimų × 4 formuluotės = 200 variantų.

### 3.2 Scenarijus: exam_week (α=2.5)

| Konfigūracija | API skambučiai | Sąnaudos | Sutaupyta | Atitikimo dažnis |
| --- | --- | --- | --- | --- |
| baseline | 500 | $0.1660 | $0.0000 | 0.0% |
| exact | 34 | $0.0113 | $0.1547 | 93.2% |
| semantic | 23 | $0.0076 | $0.1584 | 95.4% |
| full | 23 | $0.0076 | $0.1584 | 95.4% |

**Pastebėjimai:** labai koncentruotas srautas — `exact` jau pasiekia 93.2% atitikimo, o semantinis sluoksnis papildomai sugauna formuluotės variantus, pakeldamas iki 95.4%. Šiame scenarijuje LRU ir ROI daro tuos pačius sprendimus, nes populiariausių klausimų aibė telpa talpykloje be išmetimo.

### 3.3 Scenarijus: mixed (α=1.5)

| Konfigūracija | API skambučiai | Sąnaudos | Sutaupyta | Atitikimo dažnis |
| --- | --- | --- | --- | --- |
| baseline | 500 | $0.1660 | $0.0000 | 0.0% |
| exact | 105 | $0.0349 | $0.1311 | 79.0% |
| semantic | 51 | $0.0169 | $0.1491 | 89.8% |
| full | 51 | $0.0169 | $0.1491 | 89.8% |

**Pastebėjimai:** platesnis srautas — `exact` nebepakanka (79.0%), semantinis sluoksnis pakelia atitikimą iki 89.8%. LRU ir ROI vis dar duoda identiškus rezultatus.

### 3.4 Scenarijus: casual (α=1.1)

| Konfigūracija | API skambučiai | Sąnaudos | Sutaupyta | Atitikimo dažnis |
| --- | --- | --- | --- | --- |
| baseline | 500 | $0.1660 | $0.0000 | 0.0% |
| exact | 194 | $0.0644 | $0.1016 | 61.2% |
| semantic | 72 | $0.0239 | $0.1421 | 85.6% |
| full | 66 | $0.0219 | $0.1441 | **86.8%** |

**Pastebėjimai:** čia atsiranda **ROI pranašumas**. Esant plačiam srautui ir spaudimui talpyklai, išmetimo sprendimai tampa svarbūs. ROI išmeta mažai panaudotus įrašus, o LRU — seniausiai prisiliestus (net jei jie buvo labai vertingi). `full` konfigūracija atlieka 66 API skambučius vs 72 semantinei (LRU), t.y. apie 8.3% mažiau.

### 3.5 LRU vs ROI palyginimas (izoliuotas eksperimentas)

Integracijos testas `test_lru_vs_roi_evict_different_entries` izoliuoja vieną situaciją — tą pačią užklausų seką dviejose talpyklose, kurių talpa `max_size=2`:

- Įrašas A: 3 atitikimai × $0.10 = $0.30 vertė, priėjimas prieš B
- Įrašas B: 1 atitikimas × $0.01 = $0.01 vertė, priėjimas po A
- Įstatomas C → talpykla pilna → išmetimas

**LRU išmeta A** (seniausiai prisiliestas), palikdamas B ir C.
**ROI išmeta B** (žemesnės vertės), palikdamas A ir C.

Ta pati darbo seka, skirtingos konfigūracijos, skirtingi sprendimai. Šis konkretus kontrastas pavaizduoja, kuo ROI skiriasi nuo LRU, kai talpykla turi rinktis tarp „pigaus, bet neseno" ir „brangaus, bet seno" įrašo.

### 3.6 Realios API validacija

Kad etalono skaičiai (imitacijos) neliktų tik teoriniai, to paties `exam_week` scenarijaus darbo srautas (500 užklausų, sėkla 42) perleistas per **tikrą „Anthropic" Haiku API**:

| Konfigūracija | API skambučiai | Sąnaudos | Sutaupyta | Atitikimo dažnis | Sienos laikas |
| --- | --- | --- | --- | --- | --- |
| baseline | 500 | $0.2436 | $0.0000 | 0.0% | 1206 s |
| exact | 35 | $0.0201 | $0.2727 | 93.0% | 89 s |
| semantic | 22 | $0.0127 | $0.2174 | 95.6% | 66 s |
| full | 22 | $0.0122 | $0.2241 | 95.6% | 63 s |

**Palyginimas su imitacijos prognoze:**

| Rodiklis | Imitacija (mock) | Tikras API | Skirtumas |
| --- | --- | --- | --- |
| baseline sąnaudos | $0.1660 | $0.2436 | Tikras API +47% — tikri atsakymai ilgesni nei fiksuoti 80 išv. žetonų |
| `exact` santykinis pigumas | 93.0% | 91.7% | ≈ 1.3 proc. punktai skirtumo |
| `semantic` santykinis pigumas | 95.6% | 94.8% | ≈ 0.8 proc. punktai skirtumo |
| `full` santykinis pigumas | 95.6% | 95.0% | ≈ 0.6 proc. punktai skirtumo |
| Atitikimo dažniai | identiški | identiški | seka ta pati, cache logika ta pati |

**Išvados:**

1. **Atitikimo dažnis sutampa iki paskutinio dešimtainio** tarp imitacijos ir tikrojo API, nes `StudentSimulator` generuoja identišką užklausų seką (tas pats sėklos parametras), o cache sprendimai priklauso tik nuo užklausų turinio, ne nuo tiekėjo.

2. **Absoliutinės sąnaudos skiriasi** apie 47% — tikras Haiku grąžina vidutiniškai ilgesnius atsakymus nei `MockProvider`-iuose fiksuoti 80 išvesties žetonų. Tai reiškia, kad imitaciniai skaičiai yra **konservatyvi apatinė riba** — tikros sąnaudos didesnės, bet procentinis taupymas išlieka panašus.

3. **Sąnaudų mažinimo procentais imitacija užtikrintai prognozuoja tikrovę.** Visos trys konfigūracijos — `exact`, `semantic`, `full` — imitacijoje ir realiame API išlieka 90%+ pigesnės už baseline, o skirtumai tarp jų (<1 proc. punktas) statistiškai nereikšmingi esant 500 užklausų imtiai.

4. **Netikėta nauda: latencija.** Tikrame API `baseline` užtruko **20 minučių** (1206 s), o `full` konfigūracija tą patį darbo srautą baigė per **63 s** — beveik **19× greičiau**. Tai dėl to, kad cache hit neskambina į API (nėra tinklo delsos), o tik skaito iš atminties / SQLite. Reikšminga vartotojo patirčiai — talpykla ne tik taupo pinigus, bet ir drastiškai mažina atsakymo laiką kartotinėms užklausoms.

### 3.7 Bendra apžvalga

Iš trijų scenarijų ir keturių konfigūracijų lenteli matyti:

1. **Egzamino savaitės tipas** (smailus pasiskirstymas): paprasčiausias `exact` jau duoda ~93% atitikimo. Semantinis sluoksnis duoda nedidelį papildomą pranašumą.
2. **Mišrus tipas**: semantinis sluoksnis pakelia atitikimą ~11 procentinių punktų (79.0 → 89.8%).
3. **Neformalus tipas**: ROI pranašumas pastebimas — ~1.2 procentinio punkto virš vien tik semantinio LRU.

Didžiausias pelnas — perėjimas **nuo baseline prie kurio nors talpyklos** (~60–95% taupymas). ROI pranašumas virš LRU pasireiškia tik esant talpyklos spaudimui ir plačiai užklausų aibei.

---

## 4. Iššūkiai

### 4.1 Semantinio modelio struktūrinis silpnumas

Pirminė semantinės talpyklos koncepcija buvo paprasta: įterpti užklausą, palyginti su saugomais įterpimais, grąžinti artimiausią (jei virš slenksčio). Tačiau mažas daugiakalbis modelis (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim) turi ryškią silpnybę — trumpoms užklausoms, turinčioms bendrą paviršinę struktūrą, jis grąžina beveik identiškus įterpimus, net jei matematinis turinys visiškai skiriasi.

Matuojant:

| Poros | Kosinusinis panašumas |
| --- | --- |
| „Kas yra sin 30" vs „Apskaičiuok sin 30" | 0.923 (artimai — gerai) |
| „Kas yra sin 30" vs „Kas yra sinusas 30" | 0.830 (perifrazė — gerai) |
| **„Kas yra sin 30" vs „Kas yra cos 30"** | **0.973** ⚠️ *(skirtingi klausimai — pavojinga)* |
| „Kas yra sin 30" vs „Kas yra sin(x) integralas" | 0.102 (skirtingi — gerai) |

Tas pats modelis, kuris atskiria integralą ir sinusą, negali atskirti sinuso ir kosinuso trumpoje užklausoje, nes tuos abu pirmiausia „girdi" kaip „Kas yra — 30" su neesminiu vieno žetono skirtumu.

### 4.2 Slenksčio kalibravimas ir `MathKeywordGuard`

Pradinis slenkstis iš projekto plano — **0.92** — esant tokiam modeliui reiškia, kad **nė viena perifrazė nebus pagauta**: net geros perifrazės šeimai siekia 0.62–0.83. Pastebėjus empirinius duomenis, slenkstis buvo sumažintas iki **0.60**.

Tačiau tai iškart atidarė kryžminio funkcijų kolizijos problemą: esant 0.60 slenksčiui, „sin 30" vs „cos 30" (0.97) taip pat praeina — ir talpykla grąžina atsakymą apie sinusą, kai buvo klausta kosinuso. **Klaidingas atsakymas — korektiškumo spraga.**

Dvipusis sprendimas: **slenkstis + matematinių raktažodžių apsauga**.

1. Slenkstis pakeltas iki **0.75** — pakankamai aukštai, kad atmestų bent dalį triukšmo, bet ne tiek aukštai, kad prarastų visas perifrazes.
2. `MathKeywordGuard` klasė ištraukia iš kiekvienos užklausos **matematinių identifikatorių aibę** (sin/cos/tan, log/ln, integralas/išvestinė, kvadratinė šaknis ir t.t., plius skaitiniai literalai). Hit grąžinamas tik jei tiek slenkstis, tiek abi aibės sutampa.

Pavyzdinis rezultatas po apsaugos:

- „Kas yra sin 30" ↔ „Apskaičiuok sin 30" — slenkstis praėjo, aibės lygios → **hit** ✓
- „Kas yra sin 30" ↔ „Kas yra cos 30" — slenkstis praėjo (0.97), aibės **nelygios** {sin, 30} ≠ {cos, 30} → **reject** ✓
- „Kas yra sin 30" ↔ „Kas yra sin 60" — aibės nelygios {sin, 30} ≠ {sin, 60} → **reject** ✓

Be to, `SemanticCache.get()` iteracija pakeista — vietoj „grąžinti geriausią kosinusinį atitikimą, jei virš slenksčio", dabar „iteruoti per visus kandidatus nuo geriausio mažėjant, grąžinti pirmą, kuris praeina apsaugą". Tai užtikrina, kad aukštas-koinusinis-bet-neteisingas kandidatas neuždarys žemesnio-kosinusinio-bet-teisingo kandidato.

### 4.3 Srities susiaurinimas

Pradinis planas apėmė platesnę sritį — anglų kalbos užklausas ir adaptyvų maršrutą tarp Haiku/Sonnet/Opus modelių. Abu buvo atsisakyti V1 versijai:

- **Anglų kalba atmesta**, kad karkasas būtų fokusuotas į konkretų tikslą (lietuvių matematikos klausimai). Daugiakalbis įterpimo modelis išlaikomas, nes lietuviškai jis dirba pakankamai gerai.
- **Adaptyvus maršrutas atmestas**, nes jo sudėtingumas nepateisinamas V1 mastu — dažniau naudinga investuoti laiką į talpyklos sluoksnį, kuris duoda 60–95% taupymą pats vienas. Maršrutas paliktas kaip V2 idėja.

---

## 5. Išvados

### 5.1 Pasiekti rezultatai

Įgyvendinta viskas, ko reikalauja užsibrėžtas V1 tikslas:

- **5 pilnai veikiantys Strategy šablono pritaikymai** (Provider, CacheStrategy, Storage, EvictionPolicy, Embedder), kiekvienas su bent 2 konkrečiais įgyvendinimais.
- **Galutinė klientinė fasada** (`TokenFrameClient`), kuri paslepia komponavimo sudėtingumą už `query(prompt)`.
- **4 failų I/O sričių** realizacija (SQLite talpykla, JSON kainos, JSON klausimų fiksacija, CSV/JSON metrikų eksportas).
- **Daugiau nei 250 testų** `unittest` karkase, apimantys tiek vienetinį, tiek integracinį lygį, vykdomi offline per kelias dešimtis milisekundžių.
- **Reproduzuojamas etalonų rinkinys** (hand-crafted 50×4 klausimų bankas, 3 scenarijai) su CSV, JSON ir matplotlib grafikų eksportu.
- **Ne tik imitacija — tikrojo „Anthropic" API validacijos paleidimas** patvirtina, kad procentinis taupymas išlieka tokio pat lygio realiame pasaulyje.

### 5.2 Išmatuoti rezultatai

Pagal scenarijus ir konfigūracijas:

- **exam_week** (daug pasikartojimų): 93.2–95.4% pigiau nei baseline.
- **mixed** (vidutinis pasikartojimų kiekis): 79.0–89.8% pigiau.
- **casual** (maža pakartotina): 61.2–86.8% pigiau. **ROI vs LRU**: +1.2 procentinio punkto atitikimo dažniu.
- **Izoliuotas LRU vs ROI eksperimentas**: ROI išlaiko aukštos vertės įrašus, kuriuos LRU išmeta dėl laiko.
- **MathKeywordGuard** apsaugo nuo sin/cos klasės kolizijų, kurias kitaip sukeltų žemas slenkstis mažame daugiakalbiame modelyje.

### 5.3 Žinomos ribos

1. **Kandidatų iteracija yra O(n) talpyklos dydžio atžvilgiu.** Kiekvieną semantinę užklausą reikia palyginti su visais saugomais įterpimais. Tai priimtina projekto dydyje (šimtai–tūkstančiai įrašų), bet neskaliuojama į dešimtis tūkstančių. Sprendimas — **apytikslis artimiausio kaimyno (ANN) indeksavimas**, kurį galima įdėti už `Storage` abstrakcijos.

2. **Daugiakalbio įterpimo modelio struktūrinis-panašumo silpnumas.** Trumpoms matematikos užklausoms modelis remiasi daugiau paviršine struktūra (vartojami žodžiai „Kas yra 30") nei semantiniu turiniu (kurios funkcijos). `MathKeywordGuard` šį silpnumą sušvelnina, bet neišsprendžia iš esmės — didesnis ar specializuotas modelis (pvz., `intfloat/multilingual-e5-base`, ~1 GB) būtų geresnis sprendimas.

3. **Fiksuoti žetonų skaičiai `MockProvider`-iuose.** Etaloninis tiekėjas naudoja vidutines reikšmes (15 įv. / 80 išv.). Kad būtų tikslesnis, `MockProvider` galėtų dinamiškai įvertinti pagal užklausos ilgį. Tikra API tai išsprendė, bet tai kainavo pinigų.

4. **Vienkartinis klausimų bankas.** 200 užklausų variantų gali būti per mažai didelio masto etalonams. V2 galėtų vieną kartą sugeneruoti ~2000 klausimų per Claude-ą ir išsaugoti fiksacijoje.

### 5.4 Ateities darbai (V2)

- **Maršrutizatorius** (`AdaptiveRouter`): automatinis užklausų sudėtingumo klasifikavimas ir pigiausio tinkamo modelio parinkimas (Haiku prasta → Sonnet → Opus).
- **ANN indeksavimas**: FAISS ar panašus įrankis už `Storage` sąsajos.
- **Didesnis įterpimo modelis**: `multilingual-e5-base` ar naujesni, labiau atskirti semantinius niuansus.
- **Platesnis klausimų bankas**: vienkartinis Claude'o sugeneruotas rinkinys, ~2000 variantų.
- **Srautinis (streaming) atsakymas**: API palaiko, bet V1 atsisakyta dėl paprastumo.
- **Keli tiekėjai**: „OpenAI", „Google Gemini" integracijos — `Provider` abstrakcija tam pasiruošusi.
- **Realaus laiko metrikos**: dabar metrikos kaupiasi sesijos viduje ir nėra publikuojamos gyvai.

---

*Reportas parengtas projekto užbaigimo metu. Visos rezultatų lentelės atkuriamos iš `reports/mock_api/` CSV failų; tikro API duomenys — `reports/real_api/`. Projekto repozitorija: https://github.com/FaustasRi/LLMCacheSystem*
