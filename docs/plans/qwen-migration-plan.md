# Piano: aggiungere Qwen3.6-35B-A3B a TurboFieldfare

- Versione: 1 (2026-11-08)
- Commit di base analizzato: `296fded` su `main`
- Branch di lavoro: **`qwen-support`**, uno solo per tutto il piano
- Esecutore: Claude CLI su macOS, collegato al modello locale
  Qwen3.8-Flash-Next (256K token di contesto)
- Report delle sezioni: `docs/plans/qwen-migration/reports/SN.md`
- Script di supporto (Python): `docs/plans/qwen-migration/tools/`

## In breve

Oggi TurboFieldfare esegue solo Gemma 4 26B-A4B (4 e 8 bit). Questo piano
aggiunge **Qwen3.6-35B-A3B** (checkpoint MLX a 4 bit, solo testo):
installazione, runtime Metal, tokenizer, formato chat, CLI, server e Mac app.
Gemma deve restare identico, byte per byte e token per token.

Qwen3.6 differisce da Gemma in tre punti, ed è lì che sta il lavoro:

1. **30 layer su 40 non usano l'attention ma Gated DeltaNet**, una "attention
   lineare" con uno stato ricorrente di dimensione fissa. Servono kernel nuovi
   e una cache nuova (stato fisso più una piccola KV per i 10 layer di
   attention classica).
2. **Tokenizer e formato chat diversi**: BPE byte-level, formato ChatML con
   blocco di ragionamento `<think>…</think>` e tool call in XML.
3. **MoE diverso**: 256 expert piccoli (8 attivi, attivazione SiLU) più uno
   shared expert con un gate sigmoide.

Il lavoro è diviso in **27 sezioni piccole (S1–S27)**, da eseguire in ordine,
una per sessione di Claude CLI. Ogni sezione finisce con report, commit e push
su `qwen-support`: la sezione successiva può partire su qualunque Mac.

## Indice

- Parte 1 — Per te: come usare il piano
- Parte 2 — Regole per l'agente
- Parte 3 — Riferimento tecnico
- Parte 4 — Le sezioni S1–S27
- Appendici A–D

## Glossario minimo

| Termine | Significato |
| --- | --- |
| `.gturbo` | La cartella del modello installato nel formato di TurboFieldfare |
| Repack | L'installer: scarica il checkpoint MLX e lo riscrive in `.gturbo` |
| Golden | Risultati di riferimento prodotti da una libreria esterna (Hugging Face `tokenizers`/`transformers`, `mlx-lm`), con cui confrontiamo il nostro codice |
| Gated DeltaNet | Il "mixer" dei layer lineari di Qwen: stato ricorrente 128×128 per testa, aggiornato a ogni token |
| KV cache | Chiavi e valori dei token passati, solo per i 10 layer di attention classica |
| Riferimento CPU | Implementazione Swift semplice e lenta di un kernel, usata nei test per verificare il kernel Metal |

---

# Parte 1 — Per te: come usare il piano

## 1.1 Le sezioni

| Sezione | Cosa fa | Modello reale | Download | Tuo controllo |
| --- | --- | --- | --- | --- |
| S1 | Verifica i fatti del checkpoint e registra lo stato dei test | no | metadati (pochi MB) | ⛔ |
| S2 | Golden di tokenizer e chat (Python) | no | tokenizer (~25 MB), pacchetti Python | |
| S3 | Contratti: famiglia del modello, manifest, configurazione | no | — | |
| S4 | Catalogo: variante `qwen36` nell'installer | no | — | |
| S5 | Repack 1: lettura di `config.json` e manifest Qwen | no | — | |
| S6 | Repack 2: mappa dei tensori, tokenizer, dry-run | no | metadati | |
| S7 | Installazione reale di `qwen36.gturbo` | sì (installa) | ~20 GB | |
| S8 | Golden del modello con `mlx-lm` | `mlx-lm` | ~20 GB + `mlx-lm` | ⛔ |
| S9 | Tokenizer Qwen in Swift | no | tokenizer, se manca | |
| S10 | Formato chat Qwen in Swift | no | — | |
| S11 | Parser del ragionamento e delle tool call | no | — | |
| S12 | Kernel 1: norme (e cartella Metal Qwen) | no | — | |
| S13 | Kernel 2: router, embedding, testa finale | no | — | |
| S14 | Kernel 3: DeltaNet, convoluzione e gate | no | — | |
| S15 | Kernel 4: DeltaNet, ricorrenza e uscita | no | — | |
| S16 | Kernel 5: attention con gate | no | — | |
| S17 | Kernel 6: MoE SiLU e shared expert | no | — | |
| S18 | Caricamento del modello e cache ibrida | sì (solo caricamento) | — | |
| S19 | Runner: decode token per token | sì | — | ⛔ |
| S20 | Runner: prefill a blocchi e riuso del prefisso | sì | — | |
| S21 | Generazione indipendente dal modello | sì (prova breve) | — | |
| S22 | CLI con Qwen | sì (prova breve) | — | |
| S23 | Server OpenAI-compatibile con Qwen | sì (prova breve) | — | |
| S24 | Mac app 1: catalogo, installazione, memoria | no | — | |
| S25 | Mac app 2: conversazione e ragionamento | no | — | |
| S26 | Prestazioni di Qwen: prefill più veloce | sì | — | |
| S27 | Validazione finale, misure, benchmark, documentazione | sì | — | ⛔ |

"Modello reale: sì" vuol dire che su quel Mac deve esserci
`scratch/qwen36.gturbo` installato (S7). S8 richiede invece il checkpoint MLX
e un Mac con almeno 32 GB di RAM. Consiglio: esegui S7, S8, S18–S23, S26 e S27
sullo stesso Mac, quello con più memoria.

## 1.2 Preparazione (una volta sola)

1. Metti il piano su GitHub nel branch di lavoro (o chiedi a Claude di farlo):

   ```bash
   git checkout -b qwen-support main
   git add docs/plans/qwen-migration-plan.md
   git commit -m "Add the Qwen3.6 migration plan"
   git push -u origin qwen-support
   ```

   Alla fine `git status` deve essere pulito: ogni sezione lo controlla.
2. Su ogni Mac che userai: clone del fork, macOS 26 o successivo, Swift 6.2 o
   successivo (Xcode o Command Line Tools), Python 3, chiave SSH per fare push
   su GitHub. Poi `git checkout qwen-support`.
3. Claude CLI: avvialo dalla radice del repository, collegato al modello
   locale come fai di solito.
   - Usa una **sessione nuova per ogni sezione** (`/clear`, oppure esci e
     rilancia `claude`): il contesto di 256K basta per una sezione, non per due.
   - Lascia attive le conferme per i comandi non previsti. Puoi pre-approvare
     `swift build`, `swift run`, `Scripts/test.sh`, `git add`, `git commit`,
     `git push`, `grep`, `sed -n`.
   - Non usare `--dangerously-skip-permissions`.

## 1.3 Eseguire una sezione

1. Sul Mac scelto:

   ```bash
   git fetch origin && git checkout qwen-support && git pull --ff-only
   ```

2. Se la sezione richiede il modello reale e su quel Mac manca, vedi 1.5.
3. Avvia Claude CLI e incolla questo prompt, con il numero della sezione al
   posto di `N` (per esempio `S7`):

   ```text
   Esegui la sezione SN del piano docs/plans/qwen-migration-plan.md.
   Leggi prima AGENTS.md e la Parte 2 del piano e seguile alla lettera; poi
   leggi la scheda SN nella Parte 4 e, della Parte 3, solo ciò che la scheda
   indica. Il piano è lungo: trova le righe di ogni parte con
   grep -n '^# Parte\|^## \|^### S' docs/plans/qwen-migration-plan.md
   e leggi solo quegli intervalli.
   Lavora solo sul branch qwen-support e modifica solo i file che la
   scheda permette.
   Dopo ogni passo aggiorna docs/plans/qwen-migration/reports/SN.md, fai commit
   e push.
   Alla fine segui la Parte 2.9 (report finale, commit, push) e fermati: non
   iniziare la sezione successiva.
   ```

4. Quando l'agente si ferma, apri il report
   `docs/plans/qwen-migration/reports/SN.md`:
   - `Stato: Completata` → puoi passare alla sezione successiva, anche su un
     altro Mac (dopo i controlli ⛔ di 1.7, dove previsti);
   - `Stato: Bloccata` → leggi il motivo e vedi 1.6.

**Più sezioni di fila, senza intervento.** Chiudi Claude CLI e, dalla radice
del repository, lancia
[`run-sections.sh`](qwen-migration/run-sections.sh):

```bash
docs/plans/qwen-migration/run-sections.sh S9        # da S9 fino al prossimo ⛔
docs/plans/qwen-migration/run-sections.sh S9 S11    # da S9 a S11
```

Ogni sezione gira in una sessione nuova (contesto vuoto), con lo stesso prompt
di questo passo 3. Se una sezione si interrompe a metà, lo script la riprende
con il prompt di 1.4 finché fa progressi; se Claude CLI finisce con un errore
(per esempio del server del modello), aspetta un minuto prima di riprovare. Si
ferma quando una sezione diventa `Bloccata`, dopo 3 riprese di fila senza
progressi (nuovi commit o file modificati; o 20 tentativi in tutto) e dopo
ogni sezione ⛔. Se il modello chiude il turno con solo ragionamento, lo
script continua la stessa sessione invece di aprirne una nuova. Se la scheda chiede
una sessione per passo (S19, S20, S25, S26), ogni passo è un tentativo con
il contesto vuoto.
Le istruzioni complete sono in testa allo script.

## 1.4 Riprendere una sezione interrotta

Crash, contesto esaurito, Claude CLI chiuso: apri una sessione nuova e incolla

```text
Riprendi la sezione SN del piano docs/plans/qwen-migration-plan.md.
Leggi AGENTS.md, la Parte 2 del piano (in particolare 2.8, Ripresa), la scheda
SN e il report docs/plans/qwen-migration/reports/SN.md; poi continua dal primo
passo non completato.
```

L'agente fa push dopo ogni passo, quindi puoi riprendere anche su un altro Mac.
Se il Mac dove si è fermato è ancora acceso, controlla prima lì che non ci
siano commit non pushati: `git log origin/qwen-support..HEAD`.

## 1.5 Cambiare Mac: cosa viaggia e cosa no

Viaggia nel branch: codice, test, report, script e golden piccoli
(`Tests/QwenFixtures/`).

Non viaggia (sta in `scratch/`, che git ignora):

| Cosa | Serve in | Come averlo su un altro Mac |
| --- | --- | --- |
| `scratch/qwen36.gturbo` (~20 GB) | S18–S23, S26, S27 | reinstallalo (comando sotto), copialo tu da un altro Mac, oppure crea un link a dove già si trova |
| `scratch/gemma4.gturbo` | S27 (regressione di Gemma) | come oggi per Gemma |
| `scratch/qwen36-tokenizer/`, `scratch/qwen36-venv/` | S2, S8, S9 | li ricrea l'agente (piccoli) |
| checkpoint MLX nella cache di Hugging Face | solo S8 | — |

```bash
swift run -c release TurboFieldfareRepack --variant qwen36 --output scratch/qwen36.gturbo
ln -s /percorso/del/modello/qwen36.gturbo scratch/qwen36.gturbo
```

Il primo comando funziona da S6 in poi. L'agente non installa il modello da
solo, salvo in S7.

## 1.6 Se una sezione si blocca

- **Manca qualcosa sul Mac** (modello, spazio, un processo del modello già
  attivo): sistemalo e usa il prompt di ripresa.
- **Il piano è sbagliato** (un fatto, una formula, un contratto): non forzare.
  Fai correggere il piano (per esempio a Claude), commit e push della
  correzione su `qwen-support`, poi riprendi la sezione.
- **Un test non passa dopo 3 tentativi**: decidi tu se riprendere, rivedere la
  scheda o spezzarla.

## 1.7 I tuoi controlli (⛔)

| Dopo | Controlla |
| --- | --- |
| S1 | Nella tabella dei fatti del report ogni riga è "confermato". Se qualcosa differisce, il report è "Bloccata": fai correggere la Parte 3 prima di S2 |
| S8 | I golden sono in `Tests/QwenFixtures/model/` e il report dice se le formule della Parte 3.5 sono confermate. Se riporta "Correzioni al piano: sì", fai aggiornare la 3.5 prima di S12 |
| S19 | Il testo generato nel report è sensato e i confronti con i golden rispettano le tolleranze (C8) |
| S27 | Benchmark, regressione di Gemma, prova dell'app con l'elenco del report; poi decidi tu il merge in `main` |

## 1.8 Decisioni già prese

| # | Decisione |
| --- | --- |
| U1 | Download approvati, e solo questi: metadati e header degli shard; file del tokenizer; pacchetti Python `transformers`, `jinja2`, `huggingface_hub`, `mlx-lm` (o `mlx-vlm`) in `scratch/qwen36-venv`; installazione di `qwen36.gturbo` (~20 GB, S7); checkpoint MLX alla revisione pinnata per `mlx-lm` (~20 GB, solo S8) |
| U2 | Solo testo e solo 4 bit. Esclusi: immagini e video con Qwen, MTP e decodifica speculativa, Qwen a 8 bit, contesto oltre 262.144 token |
| U3 | Ragionamento (`<think>`) attivo per default, come nel template; disattivabile in CLI, server e app |
| U4 | Sampling di default per Qwen da `generation_config.json`: temperature 1.0, top-k 20, top-p 0.95. Gemma resta 0.2 / 64 / 0.95 |
| U5 | Un solo branch, `qwen-support`; commit e push dopo ogni passo e alla fine di ogni sezione; merge in `main` solo su tua decisione dopo S27 |
| U6 | Esecutore: Claude CLI su macOS con il modello locale Qwen3.8-Flash-Next (256K) |

U1–U4 vengono dal piano precedente (confermate il 2026-11-08); U5 e U6 da
questa richiesta.

---

# Parte 2 — Regole per l'agente

Questa parte vale per **tutte** le sezioni. Leggila per intero all'inizio di
ogni sessione.

## 2.1 Contesto

- Sei Claude CLI su un Mac, nella radice del repository TurboFieldfare. Il
  modello che ti guida ha 256K token di contesto; ogni sezione è pensata per
  starci con ampio margine, se segui le regole di 2.3 sul contesto.
- AGENTS.md vale sempre. AGENTS.md vieta di modificare il codice senza una
  richiesta dell'utente: questo piano **è** la richiesta, ma solo per i file
  elencati in "Puoi modificare" della tua scheda.
- Codice, commenti, nomi, messaggi di commit e testi dell'interfaccia: in
  inglese, nello stile del repository. Report: in italiano.
- Se un report precedente corregge esplicitamente il piano, vale il report.

## 2.2 Inizio sezione (sempre, in quest'ordine)

1. Controlla l'ambiente; se un controllo fallisce, fermati e dillo:

   ```bash
   uname -s                      # deve stampare Darwin
   sw_vers -productVersion       # 26 o più
   swift --version | head -1     # Swift 6.2 o più
   ls AGENTS.md Package.swift    # devono esistere
   ```

2. `git status --porcelain` deve essere vuoto. Se non lo è: fermati ed elenca i
   file (non scartare modifiche di altri).
3. Allinea il branch:

   ```bash
   git fetch origin
   git checkout qwen-support
   git pull --ff-only origin qwen-support
   ```

   Se `qwen-support` non esiste su `origin`: fermati e chiedi
   all'utente di fare la Parte 1.2.
4. Tranne in S1: il report della sezione precedente,
   `docs/plans/qwen-migration/reports/S<N-1>.md`, deve esistere e dire
   `Stato: Completata`. Se no, fermati: "la sezione precedente non risulta
   completata".
5. Se `docs/plans/qwen-migration/reports/SN.md` esiste già, è una ripresa:
   vai a 2.8.
6. Crea il report con il modello di 2.7 (`Stato: In corso`, Mac e RAM da
   `sysctl -n hw.model` e `sysctl -n hw.memsize`, versioni, commit di partenza
   da `git rev-parse --short HEAD`, la lista dei passi della scheda non
   spuntati). Commit `Start the SN report` e push.
7. Leggi ciò che la scheda elenca in "Leggi", e nient'altro di grande. Il
   piano supera le 1.900 righe: non leggerlo per intero. Trova gli intervalli
   con `grep -n '^# Parte\|^## \|^### S' docs/plans/qwen-migration-plan.md`
   e leggi solo quelli (Parte 2, la tua scheda, le voci della Parte 3 che la
   scheda nomina).

## 2.3 Regole durante il lavoro

**Perimetro**

- Modifica solo i file della lista "Puoi modificare". Se serve toccarne altri,
  fermati (2.6).
- **Gemma intoccato.** Non cambiare comportamento, default o output di Gemma.
  Non modificare `Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift`
  né kernel e file `.metal` esistenti, salvo le righe che la scheda autorizza
  per nome. A fine sezione questo comando non deve stampare nulla:

  ```bash
  git diff --stat <commit di partenza> -- Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift Sources/TurboFieldfare/Kernels Sources/TurboFieldfare/Metal ':!Sources/TurboFieldfare/Kernels/Qwen' ':!Sources/TurboFieldfare/Metal/Qwen'
  ```

- API pubbliche: si aggiunge, non si rinomina e non si rimuove.
- Niente ottimizzazioni non richieste: prima corretto, poi veloce (S26
  velocizza il prefill, S27 misura).
- **Non inventare API.** Prima di usare un tipo o una funzione esistente,
  trovane la firma:
  `grep -rn "func nome\|struct Nome\|class Nome\|enum Nome\|protocol Nome" Sources`.

**Contesto (il tuo budget)**

- Non leggere per intero file sopra 400 righe: usa `grep -n` e poi
  `sed -n '<a>,<b>p'`. Mai interi: `RealForwardRunner.swift`, `prefill.metal`,
  `ServerInference.swift`, `MultimodalConversation.swift`, `HTTPServer.swift`.
- Taglia sempre l'output dei comandi (`| tail -n 60`, `| head -n 60`) o
  mandalo in un log in `scratch/logs/` e leggine solo le righe utili.
- Scrivi nel report, sezione "Note", ogni scoperta che servirà più avanti
  (nomi di funzioni, offset, decisioni). Se il contesto viene compattato,
  rileggi il report prima di continuare.

**Comandi lunghi**

- `swift build`, `swift run` e `Scripts/test.sh` durano minuti: usa per il
  tool Bash un timeout di 600000 ms e manda l'output in un log:

  ```bash
  mkdir -p scratch/logs
  swift build -c release > scratch/logs/SN-build.log 2>&1; echo "exit=$?"; tail -n 40 scratch/logs/SN-build.log
  Scripts/test.sh --filter 'NomeSuite' > scratch/logs/SN-test.log 2>&1; echo "exit=$?"; grep -E "error:|✘|failed|passed|Test run" scratch/logs/SN-test.log | tail -n 60
  ```

- Comandi che possono superare 10 minuti (installazione del modello, download
  grandi, benchmark, a volte la suite completa): lanciali in background
  (`run_in_background`) con output su log e aspetta la notifica di fine. Non
  controllare il log in un ciclo stretto.
- Non lanciare `swift build` mentre gira `Scripts/test.sh`, né il contrario.

**Test**

- I test usano Swift Testing (`@Suite`, `@Test`, `#expect`), come quelli
  esistenti. I test nuovi stanno in cartelle `Qwen/` (contratto C2).
- Esegui i test solo con `Scripts/test.sh` (seriale): con `--filter` durante il
  lavoro, la suite completa a fine sezione.
- Fallimenti nella suite completa: quelli in suite che hai toccato, o che usano
  codice che hai modificato, vanno risolti. Quelli già elencati nel report di
  S1 come "baseline", o chiaramente legati all'ambiente (per esempio il formato
  numerico della regione), vanno solo elencati nel report con il messaggio di
  errore. Qualunque altro: fermati (2.6).
- Ogni kernel nuovo: prima il riferimento CPU, poi il test GPU contro di esso,
  con dimensioni piccole **e** con le dimensioni reali di 3.2.
- I test che richiedono il modello installato si abilitano solo se esiste
  `scratch/qwen36.gturbo/manifest.json` (contratto C9): senza modello la suite
  resta verde.

**Git**

- Un commit per passo, messaggio in inglese all'imperativo come nel
  repository, per esempio `Add the Qwen gated RMSNorm kernel`.
- Dopo **ogni** commit: `git push origin qwen-support`. Mai `--force`,
  mai altri branch, mai pull request, mai merge.
- Non committare nulla sotto `scratch/`, né pesi, né file sopra i 5 MB.

**Sicurezza (da AGENTS.md)**

- Un solo processo che usa il modello alla volta. Non terminare mai processi
  che non hai avviato tu.
- Nessun download oltre a quelli che la scheda elenca (decisione U1). Niente
  copie di `.gturbo`, niente svuotamento di cache, niente worktree.
- Server solo su `127.0.0.1`; alla fine chiudi il server che hai avviato tu, e
  solo quello.
- SwiftUI: `@StoredState` e `PreviewProvider`; mai `@State` diretto né
  `#Preview`.
- Controlli Ruby sempre con `LC_ALL=en_US.UTF-8`.
- Script Python: `python3 -I` oppure `scratch/qwen36-venv/bin/python -I`. I file
  scaricati vanno in una cartella nuova e vuota sotto `scratch/` e non si
  eseguono mai.

## 2.4 Prima di ogni esecuzione con il modello reale

```bash
df -h . | tail -1
memory_pressure -Q | tail -1
ls scratch/qwen36.gturbo/manifest.json
pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
```

- `pgrep` non deve stampare nulla (ignora la riga della tua stessa shell, se
  compare). Se trova un processo: non terminarlo; aspetta o chiedi all'utente.
- Se la memoria libera è sotto il 20%: avvisa e fermati.
- Se manca il modello: fermati e indica all'utente le opzioni della Parte 1.5.
  Non installarlo, salvo in S7.
- AGENTS.md nomina solo i modelli Gemma: per Qwen il modello richiesto è
  `scratch/qwen36.gturbo` (deviazione prevista dal piano; S27 aggiorna
  AGENTS.md).
- Annota nel report il comando esatto, l'exit code e i tempi.

## 2.5 Tentativi e debug

- Se lo stesso test fallisce ancora dopo 3 tentativi di correzione: fermati.
- Kernel GPU diverso dal riferimento CPU: riduci le dimensioni (1 testa,
  1 token, 1 riga), confronta elemento per elemento e annota il primo indice
  che diverge prima di cambiare codice.
- Modello diverso dai golden: confronta layer per layer per trovare il primo
  layer che diverge, poi le grandezze interne di quel layer (golden di S8).

## 2.6 Quando fermarsi (`Stato: Bloccata`)

Scrivi il motivo nel report, fai commit e push, avvisa l'utente e fermati
quando:

- un controllo di 2.2 o 2.4 fallisce;
- serve un file non elencato, un download non previsto o una decisione che il
  piano non prende;
- un fatto o un contratto del piano risulta sbagliato e correggerlo cambierebbe
  altre sezioni;
- scatta la regola dei 3 tentativi (2.5);
- la suite completa ha fallimenti che non sono tuoi, né baseline, né
  ambientali.

## 2.7 Modello del report

```markdown
# SN — <titolo della scheda>

- Stato: In corso | Completata | Bloccata (<motivo>)
- Mac: <hw.model>, <RAM> GB, macOS <versione>, Swift <versione>
- Commit di partenza: <hash> · Commit finale: <hash>

## Passi
- [x] 1. <titolo> — commit <hash>
- [ ] 2. <titolo>

## Test
<comandi esatti, exit code, numero di test; per i kernel l'errore massimo>

## Deviazioni dal piano
<cosa, perché, impatto sulle sezioni successive; "Nessuna" se non ce ne sono>

## Valori misurati o fissati

## Note per le sezioni successive

## Problemi aperti
```

## 2.8 Ripresa di una sezione

1. Esegui i passi 1–3 di 2.2.
2. Leggi il report della sezione e `git log --oneline -15`.
3. Esegui `pgrep` di 2.4: se un processo del modello è ancora attivo (per
   esempio un test lanciato prima dell'interruzione), non terminarlo: aspetta
   che finisca o chiedi all'utente.
4. Se `git status` mostra modifiche non committate, non scartarle da solo:
   mostra `git status` e `git diff --stat` all'utente e chiedi se tenerle o
   scartarle.
5. Riparti dal primo passo non spuntato. Non rifare i passi spuntati.

## 2.9 Fine sezione

1. Tutti i passi spuntati. Suite completa:
   `Scripts/test.sh > scratch/logs/SN-full.log 2>&1` (in background se serve).
2. Il controllo "Gemma intoccato" di 2.3 non stampa nulla.
3. Se hai toccato SwiftUI:
   `LC_ALL=en_US.UTF-8 ruby Scripts/check_swiftui_macros.rb --self-test` e
   `LC_ALL=en_US.UTF-8 ruby Scripts/tests/swift_macro_probe_test.rb`. Se hai
   toccato file Markdown: `LC_ALL=en_US.UTF-8 ruby Scripts/check_markdown_links.rb`.
4. Report: `Stato: Completata`, commit finale, test, deviazioni, valori, note.
5. Commit `Complete the SN report` e push.
6. Verifica: `git status --porcelain` vuoto e
   `git log origin/qwen-support..HEAD` vuoto.
7. Scrivi all'utente un riepilogo di 3–5 righe e "Prossima sezione: S<N+1>".
   Non iniziarla.

---

# Parte 3 — Riferimento tecnico

Tutte le formule e i fatti sono qui: nessuna sezione deve cercarli sul web,
salvo dove la scheda lo dice.

## 3.1 Il checkpoint (verificato il 2026-11-08; S1 lo riverifica)

| Voce | Valore |
| --- | --- |
| Modello originale | `Qwen/Qwen3.6-35B-A3B` (BF16, licenza Apache 2.0) |
| Sorgente da installare | `mlx-community/Qwen3.6-35B-A3B-4bit` |
| Revisione pinnata | `38740b847e4cb78f352aba30aa41c76e08e6eb46` |
| SHA-256 di `model.safetensors.index.json` | `0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea` |
| Shard | 4 (`model-0000N-of-00004.safetensors`), `total_size` 20.401.929.952 B |
| Tensori nell'indice | 2.090: prefisso `language_model.` (1.757) e `vision_tower.` (333) |
| Quantizzazione | affine, 4 bit, gruppi da 64, scale e bias BF16 |
| Eccezioni a 8 bit | `mlp.gate` (router) e `mlp.shared_expert_gate` in tutti i 40 layer |
| Non quantizzati | norme, `linear_attn.conv1d.weight`, `linear_attn.A_log`, `linear_attn.dt_bias` (dtype da S1) |
| MTP | nessun tensore `mtp.*` nel checkpoint MLX |
| Altri file | `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `vocab.json`, `preprocessor_config.json`, `processor_config.json`, `video_preprocessor_config.json` |
| `model_type` | `qwen3_5_moe` (testo: `qwen3_5_moe_text`); classe `Qwen3_5MoeForConditionalGeneration` |

## 3.2 Architettura (solo testo, da `config.json` → `text_config`)

| Campo | Valore |
| --- | --- |
| Layer | 40; `full_attention_interval` 4: tre layer lineari, poi uno di attention |
| Layer di attention (indice da 0) | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 |
| `hidden_size` | 2048 |
| Gated DeltaNet | 16 teste chiave × 128, 32 teste valore × 128, conv causale kernel 4, stato in FP32 (`mamba_ssm_dtype`) |
| Attention | 16 teste query, 2 teste KV, `head_dim` 256, gate d'uscita (`attn_output_gate`), `q_norm` e `k_norm` |
| RoPE | `rope_theta` 10.000.000, `partial_rotary_factor` 0,25 (64 dimensioni ruotate); mRoPE `[11, 11, 10]` che, con solo testo, equivale a RoPE 1D |
| MoE | 256 expert, 8 attivi, `moe_intermediate_size` 512, attivazione SiLU |
| Shared expert | 1, intermedio 512, gate sigmoide scalare |
| Vocabolario | 248.320 |
| Embedding | **non** legati: `embed_tokens` e `lm_head` separati |
| `rms_norm_eps` | 1e-6 |
| `max_position_embeddings` | 262.144 |
| Token | `bos` 248044 (`<|endoftext|>`); `eos` 248046 (`<|im_end|>`) e 248044; `pad` 248044 |
| Sampling consigliato | `generation_config.json`: temperature 1.0, top-k 20, top-p 0.95 |

## 3.3 Nomi dei tensori (prefisso `language_model.`)

```text
model.embed_tokens.{weight,scales,biases}
lm_head.{weight,scales,biases}
model.norm.weight
model.layers.N.input_layernorm.weight
model.layers.N.post_attention_layernorm.weight
# layer lineari (N non multiplo di 4 più 3)
model.layers.N.linear_attn.in_proj_qkv.{weight,scales,biases}
model.layers.N.linear_attn.in_proj_z.{weight,scales,biases}
model.layers.N.linear_attn.in_proj_a.{weight,scales,biases}
model.layers.N.linear_attn.in_proj_b.{weight,scales,biases}
model.layers.N.linear_attn.conv1d.weight
model.layers.N.linear_attn.A_log
model.layers.N.linear_attn.dt_bias
model.layers.N.linear_attn.norm.weight
model.layers.N.linear_attn.out_proj.{weight,scales,biases}
# layer di attention (N = 3, 7, ..., 39)
model.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj}.{weight,scales,biases}
model.layers.N.self_attn.{q_norm,k_norm}.weight
# MoE (tutti i layer)
model.layers.N.mlp.gate.{weight,scales,biases}                  (8 bit)
model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}
model.layers.N.mlp.shared_expert.{gate_proj,up_proj,down_proj}.{weight,scales,biases}
model.layers.N.mlp.shared_expert_gate.{weight,scales,biases}    (8 bit)
```

Due punti aperti, che cambiano le formule (S1 li misura, S8 li conferma):

1. **Norme "spostate".** Nel codice Hugging Face le norme dei layer, la norma
   finale, `q_norm` e `k_norm` moltiplicano per `(1 + w)`. La conversione MLX
   può aver già sommato 1 ai pesi. Se la media di
   `model.layers.0.input_layernorm.weight` è vicina a 1, i pesi sono già
   spostati (`normWeightsShifted = true`, si usa `x̂ · w`); se è vicina a 0, si
   usa `x̂ · (1 + w)`. La norma di `linear_attn.norm` usa sempre `w`.
2. **Layout di `conv1d.weight`.** Hugging Face: `[8192, 1, 4]`; MLX
   probabilmente `[8192, 4, 1]`. Conta la forma reale e l'ordine dei tap.

## 3.4 Dimensioni derivate (4 bit, gruppi da 64)

| Voce | Valore |
| --- | --- |
| Byte per expert | gate 589.824 + up 589.824 + down 589.824 = **1.769.472 B** (108 pagine da 16 KiB) |
| Expert totali | 256 × 40 × 1.769.472 = 18.119.393.280 B (≈ 16,9 GiB) |
| Lettura expert per token a cache fredda | 8 × 40 × 1,77 MB ≈ 566 MB |
| Pesi residenti (stima) | embedding ≈ 286 MB, `lm_head` ≈ 286 MB, DeltaNet 30 × 18,95 MB, attention 10 × 15,3 MB, shared expert 40 × 1,77 MB, router 40 × 0,56 MB: ≈ 1,39 GB |
| `.gturbo` atteso | ≈ 19,5 GB |
| KV FP16 (10 layer) | 10 × 2 × 2 × 256 × 2 B = **20.480 B per token**: 640 MiB a 32K, 2,5 GiB a 128K, 5 GiB a 262.144 |
| Stato DeltaNet (FP32) | 30 × 32 × 128 × 128 × 4 B = 60 MiB, costante |
| Stato conv (FP32) | 30 × 3 × 8192 × 4 B ≈ 2,8 MiB, costante |

Proiezioni: `in_proj_qkv` 2048 → 8192 (q 2048 | k 2048 | v 4096),
`in_proj_z` 2048 → 4096, `in_proj_a` e `in_proj_b` 2048 → 32, `out_proj`
4096 → 2048; `q_proj` 2048 → 8192 (query e gate), `k_proj` e `v_proj`
2048 → 512, `o_proj` 4096 → 2048; router 2048 → 256; `shared_expert_gate`
2048 → 1.

## 3.5 Formule

Notazione: `h` stato residuo (2048 valori), `RMSNorm(x; w) = x̂ · w'` con
`x̂ = x · rsqrt(mean(x²) + 1e-6)` e `w' = w` oppure `1 + w` secondo
`normWeightsShifted`. Calcoli interni in FP32; attivazioni tra kernel in FP16
come nel resto del runtime, salvo dove indicato.

**Struttura di ogni layer.** `input_layernorm` → mixer (DeltaNet o attention)
→ somma al residuo → `post_attention_layernorm` → MoE → somma al residuo.
Niente norme "sandwich", scalari per layer, scale per expert, softcap,
finestre scorrevoli o K = V: sono tutte cose di Gemma.

### Embedding e testa

1. `h = dequant(embed_tokens)[token]`, **senza** moltiplicare per
   `sqrt(hidden)`.
2. Dopo l'ultimo layer: `y = RMSNorm(h; model.norm)`,
   `logits = dequant(lm_head) · y` (248.320 righe), **senza** softcap.

### Layer lineare (Gated DeltaNet)

Dimensioni: `Hk = 16`, `Hv = 32`, `dk = dv = 128`, `conv_dim = 8192`.

```text
x   = RMSNorm(h; input_layernorm)
qkv = W_qkv · x            (8192)
z   = W_z   · x            (4096 = 32 × 128)
b   = W_b   · x            (32)
a   = W_a   · x            (32)

# conv causale depthwise sui soli 8192 canali di qkv, kernel 4
# u_c[t] = qkv del token t sul canale c; lo stato tiene u_c[t-3], u_c[t-2], u_c[t-1]
y_c = silu( Σ_{j=0..3} w_c[j] · u_c[t-3+j] )
stato_conv ← ultimi 3 input (prima dell'attivazione)

q = y[0:2048]    → 16 × 128
k = y[2048:4096] → 16 × 128
v = y[4096:8192] → 32 × 128

q_i = q_i · rsqrt(Σ q_i² + 1e-6)     (per testa)
k_i = k_i · rsqrt(Σ k_i² + 1e-6)
q   = q / sqrt(128)

# la testa di valore j usa la testa di chiave j / 2 (divisione intera)
beta_j  = sigmoid(b_j)
g_j     = -exp(A_log_j) · softplus(a_j + dt_bias_j)
decay_j = exp(g_j)

# stato S_j ∈ R^{128×128} in FP32: riga i = dimensione chiave, colonna c = dimensione valore
S_j      = decay_j · S_j
kv_mem_c = Σ_i S_j[i][c] · k_{j/2}[i]
delta_c  = (v_j[c] - kv_mem_c) · beta_j
S_j[i][c] += k_{j/2}[i] · delta_c
o_j[c]   = Σ_i S_j[i][c] · q_{j/2}[i]

# norma gated per testa (128 valori), peso condiviso tra le teste
o_j = RMSNorm(o_j; linear_attn.norm, sempre w) · silu(z_j)
h  += W_out · concat(o_0 … o_31)      (4096 → 2048)
```

Proprietà utile per i kernel: **ogni colonna `c` di ogni `S_j` evolve in modo
indipendente**, dati `k`, `q`, `beta`, `decay` e `v_j[c]`. Un SIMD-group può
tenere una colonna (32 lane × 4 righe = 128 righe) nei registri e scorrere i
token in ordine; `kv_mem_c` e `o_j[c]` sono `simd_sum` sulle 32 lane. Lo stesso
kernel serve il decode (`T = 1`) e il prefill (`T > 1`).

Da confermare in S8: ordine dei tap della conv, mappa testa di valore → testa
di chiave (`j / 2`), rinormalizzazione del router.

### Layer di attention

```text
x  = RMSNorm(h; input_layernorm)
qg = W_q · x → 16 × 512; per ogni testa: q = qg[0:256], gate = qg[256:512]
k  = W_k · x → 2 × 256
v  = W_v · x → 2 × 256
q  = RMSNorm(q; q_norm)    (per testa, 256)
k  = RMSNorm(k; k_norm)
RoPE sulle prime 64 dimensioni di q e k, stile "rotate_half":
  coppie (i, i + 32) per i in 0..31, inv_freq_i = 1e7^(-2i/64), posizione p
attention causale; la testa query h usa la testa KV h / 8; scala 1/sqrt(256);
  softmax in FP32
o  = attn · sigmoid(gate)          (per testa, elemento per elemento)
h += W_o · concat(o_0 … o_15)      (4096 → 2048)
```

### Blocco MoE (tutti i layer)

```text
x      = RMSNorm(h; post_attention_layernorm)
p      = softmax(W_gate · x)           (256 valori, FP32; W_gate a 8 bit)
top    = gli 8 indici con p più alto
w_e    = p_e / Σ_{top} p               (rinormalizzati)
routed = Σ_{e∈top} w_e · W_down_e · ( silu(W_gate_e · x) ⊙ (W_up_e · x) )
shared = W_down_s · ( silu(W_gate_s · x) ⊙ (W_up_s · x) )
sg     = sigmoid(W_sg · x)             (scalare; W_sg 1 × 2048, 8 bit)
h     += routed + sg · shared
```

## 3.6 Formato chat

Dal `chat_template.jinja` del checkpoint (S2 produce i golden esatti):

- Turni: `<|im_start|>ROLE\n … <|im_end|>\n`, ruoli `system`, `user`,
  `assistant`. I risultati dei tool stanno in un turno `user`, ciascuno dentro
  `<tool_response>\n … \n</tool_response>`.
- Tool disponibili: nel system, un blocco `# Tools … <tools> … </tools>` con un
  JSON per tool.
- Tool call emesse dal modello (XML, non JSON):

  ```text
  <tool_call>
  <function=NAME>
  <parameter=PARAM>
  VALUE
  </parameter>
  </function>
  </tool_call>
  ```

- Ragionamento: `<think>\n…\n</think>\n\n` prima del contenuto. Prompt di
  generazione: `<|im_start|>assistant\n<think>\n` con ragionamento attivo;
  `<|im_start|>assistant\n<think>\n\n</think>\n\n` con
  `enable_thinking=false`.
- Nei turni assistant precedenti all'ultima domanda dell'utente il template
  toglie il ragionamento. Quindi ri-rendere tutta la storia **non** dà gli
  stessi token che stanno già nella cache: le continuazioni (nuovo turno
  utente, risultati dei tool) si costruiscono in coda ai token già in cache,
  come fa oggi Gemma.
- Immagini: `<|vision_start|><|image_pad|><|vision_end|>` (fuori perimetro).

## 3.7 Mappa del codice: cosa si riusa, cosa è solo Gemma

Percorsi verificati sul commit `296fded`; i numeri di riga sono indicativi.
Prefisso `Sources/TurboFieldfare/` dove non indicato.

| File | Cosa contiene | Per Qwen |
| --- | --- | --- |
| `Sources/TurboFieldfareFormat/GTurboManifestV1.swift` | manifest; `arch` obbligatorio con campi Gemma | campi opzionali nuovi (D1) |
| `Infrastructure/ModelIO/ModelTypes.swift` | `ArchConfig`, `.gemma4_26B_A4B`, `ModelError` | costante Qwen (D2) |
| `Infrastructure/ModelIO/ManifestReader.swift` | `decode(data:expecting:)`, `validateArch` campo per campo | validazione per famiglia |
| `Runtime/Inference/Model.swift` | `Model.load(...)` (~riga 345), `validateRuntimeSchema` con nomi Gemma (~riga 524), accessori (`lmHead` = embedding legato) | dispatch per famiglia + schema Qwen in file nuovo |
| `Runtime/Inference/RealForwardRunner.swift` | runner Gemma (~2000 righe) | **non si tocca** |
| `Runtime/Generation/LogitProducer.swift` | `LogitProducer`, `ContinuableLogitProducer`, `ChunkedPrefillRunner`, `ContextWindowReporting` | il runner Qwen li implementa |
| `Runtime/Generation/RawCompletion.swift` | `runRawCompletion(producer:tokenizer: GFTokenizer, ...)` (~riga 273); usa `GFDetokenizer`, `stopTokenIDs`, `endOfTurnID`, `toolResponseID`, `producer as? RealForwardRunner` | overload con un protocollo (S21) |
| `Runtime/Generation/MultimodalConversation.swift` | actor legato a `Model`, `GFTokenizer`, `RealForwardRunner` (usa `reset`, `rewind`, `continuationPosition`) | generalizzato in S25 |
| `Tokenization/Tokenizer.swift` | `GFTokenizer`: carica il tokenizer dalla cartella `tokenizer/` del `.gturbo` (`tokenizerFolder`), verifica il decoder SentencePiece di Gemma; `Message`, `applyChatTemplate`, `encodeToolChat`, `encodeTextContinuation`, `encodeToolResultContinuation` | `QwenTokenizer` con gli stessi nomi di metodo |
| `Tokenization/Detokenizer.swift` | `GFDetokenizer` (decodifica Gemma, `push`/`flush`) | `QwenDetokenizer` byte-level |
| `Tokenization/StructuredAssistantDecoder.swift`, `GemmaToolCallParser.swift` | canali e tool call di Gemma → `StructuredAssistantEvent` (`.content`, `.toolCall(ParsedToolCall)`) | parser Qwen che produce gli stessi eventi |
| `Kernels/Quant/DequantInt4GEMV.swift` | GEMV INT4 **generico** in `m`, `n` (`n` multiplo di 64); forme non specializzate usano la pipeline generica | riusabile per tutte le proiezioni Qwen in decode |
| `Kernels/Quant/DequantInt8GEMV.swift` | GEMV INT8, stesso schema | router e `shared_expert_gate` |
| `Kernels/Quant/EmbedLookupInt4.swift` | lookup INT4 generico in `d` | embedding Qwen, se i test lo confermano |
| `Kernels/MoE/MoE.swift`, `Metal/MoE/moe.metal`, `Kernels/MoE/SharedExpertInt4.swift` | expert con GELU-tanh e dimensioni Gemma | kernel Qwen nuovi (SiLU) |
| `Infrastructure/Streaming/PreadExpertStreamer.swift`, `Runtime/Inference/ModelExpertIO.swift` | streaming degli expert, cache a slot: generici | si riusano |
| `Kernels/Attention/Attention.swift`, `Metal/Attention/attention.metal` | attention Gemma (finestra e full) | kernel Qwen nuovi, salvo riuso senza modifiche |
| `Runtime/KVCache/KVCacheManager.swift` | anello scorrevole e layer full di Gemma | cache Qwen nuova (D5) |
| `Runtime/Configuration/ContextAdmission.swift` | `ModelFootprint.forProfile`, `fp16KVBytes(config:)` con geometria Gemma | formula per famiglia (S24) |
| `Infrastructure/Metal/MetalContext.swift` | `shaderModules` e `shaderSubdirectories` (~righe 131–170): tutti i `.metal` sono concatenati in **un'unica sorgente** | nomi Metal univoci, prefisso `qwen_` |
| `Sources/TurboFieldfareRepack/Core/Remote/SupportedModelVariant.swift` | catalogo `int4`/`int8` (repo, revisione, SHA dell'indice, dimensioni, cartella) | `case qwen36` (D9) |
| `Sources/TurboFieldfareRepack/Core/Format/ArchInfo.swift` | lettura di `text_config` con chiavi Gemma | lettura Qwen separata |
| `Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift` | `classify`: expert riconosciuti da `.experts.switch_glu.`; torre visiva esclusa | expert da `.mlp.switch_mlp.` |
| `Sources/TurboFieldfareRepack/Core/Remote/RemoteStreamingRepacker.swift` | lista dei file del tokenizer copiati in `tokenizer/` (~riga 463) | stessi file per Qwen |
| `Sources/TurboFieldfareRepack/Command/main.swift` | `--variant` letto con `SupportedModelVariant(rawValue:)`; `--dry-run`, `--resume`, `--verify-install --input-gturbo` | `--variant qwen36` |
| `Sources/TurboFieldfareCLI/Run.swift` | `Model.load` + `RealForwardRunner` (~riga 190) | factory per famiglia (S22) |
| `Sources/TurboFieldfareServer/Core/ServerInference.swift` | `load` (~riga 553): `GFTokenizer`, `Model.load`, `RealForwardRunner`, prompt cache | factory per famiglia (S23) |
| `Sources/TurboFieldfareApp/Core/Installation/AppModelVariant.swift` | alias di `SupportedModelVariant`, profilo e footprint | S4 minimo, S24 completo |
| `Sources/TurboFieldfareApp/Core/Inference/RealInferenceClient.swift` | `Model.load` + `RealForwardRunner` (~riga 438), `MultimodalConversation` (~righe 376, 674) | S25 |
| `Sources/TurboFieldfareValidation/Support/Reference/` | riferimenti CPU (Attention, MoE, Primitives, Quant, Sampling) e `Tolerance/` | riferimenti Qwen in `Reference/Qwen/` |

Esempi da imitare nei test:
`Tests/TurboFieldfare/Core/Runtime/Vision/VisionResourceLimitTests.swift`
(test abilitato solo se c'è un modello in `scratch/`, con
`@Test(.enabled(if:))`) e
`Tests/TurboFieldfare/Core/Kernels/MoE/MoERoutedWeightBitsTests.swift`
(percorso di un file del repository ricavato da `#filePath`).

## 3.8 Decisioni di design

- **D1 Manifest.** `arch` resta obbligatorio; per Qwen contiene i valori
  "mappati" qui sotto. Due campi opzionali nuovi: `modelFamily`
  (`"qwen3_5_moe"`; assente = Gemma) e il blocco `qwen35` (C4). Un manifest
  Gemma si legge e si riscrive **identico**. Una build vecchia che apre un
  manifest Qwen fallisce con `archMismatch` su `hiddenSize`: fallimento
  chiuso, come deve essere.

  | Campo di `arch` | Qwen |
  | --- | --- |
  | `hiddenSize` / `vocabSize` / `numLayers` | 2048 / 248320 / 40 |
  | `ffnIntermediate` / `moeIntermediateSize` | 512 / 512 |
  | `numHeads` / `numKVHeads` / `numFullKVHeads` | 16 / 2 / 2 |
  | `headDim` / `fullHeadDim` | 256 / 256 |
  | `slidingWindow` / `finalLogitSoftcap` | 0 / 0 |
  | `ropeTheta` / `fullRopeTheta` / `partialRotaryFactor` | 10000000 / 10000000 / 0.25 |
  | `numExperts` / `topKExperts` | 256 / 8 |
  | `tieWordEmbeddings` / `attentionKEqV` | false / false |
  | `hiddenActivation` | `"silu"` |
  | `fullAttentionLayerMask` | 1 nei layer 3, 7, …, 39; 0 altrove |

- **D2 Configurazione runtime.** `ArchConfig.qwen36_35B_A3B` (stesso struct,
  valori di D1) più `QwenHybridConfig.qwen36_35B_A3B` per il blocco `qwen35`.
  `Model.load` legge `modelFamily`: per Qwen valida contro queste due costanti
  (ignorando il default Gemma di `expecting`) e poi contro lo schema dei
  tensori Qwen; per Gemma tutto resta com'è.
- **D3 Runner separato.** `QwenHybridForwardRunner`, in file nuovi,
  conforme ai protocolli esistenti. Una sola factory sceglie il runner in base
  alla famiglia (S21); CLI, server e app la usano.
- **D4 Kernel nuovi in file nuovi.** Tutto il Metal di Qwen sta in
  `Sources/TurboFieldfare/Metal/Qwen/`, con nomi `qwen_…` e function constant
  200–249 (C3). Un kernel esistente si può **chiamare** se accetta già le
  dimensioni Qwen come parametri (per esempio `DequantInt4GEMV`), mai
  modificare.
- **D5 Cache.** `QwenHybridCache`: KV FP16 solo per i 10 layer di attention,
  senza anello, layout `[layer di attention][posizione][testa KV][256]`; stato
  DeltaNet FP32 per i 30 layer lineari, layout
  `[layer lineare][testa valore][colonna c][riga i]` (una colonna = 128 float
  contigui); stato conv FP32 `[layer lineare][3][8192]`.
- **D6 Riuso del prefisso.** Lo stato DeltaNet non si può "riavvolgere". Regole:
  continuare in coda alla posizione corrente funziona sempre; alla fine del
  prefill di ogni prompt si salva uno snapshot (stato DeltaNet + conv,
  ≈ 63 MiB, al massimo 2); un prompt che estende esattamente un prompt con
  snapshot riparte da lì (ripristino dello stato e troncamento della KV); in
  ogni altro caso `rewind` lancia un errore e il chiamante rifà il prefill
  completo.
- **D7 Tokenizer.** `QwenTokenizer` usa `AutoTokenizer` di `swift-transformers`
  sui file della cartella `tokenizer/` del `.gturbo`, verifica la pipeline
  attesa (BPE byte-level) e fallisce in modo chiuso se cambia. Espone gli
  stessi nomi di metodo di `GFTokenizer` usati da generazione, server e app.
- **D8 Chat.** Prima scelta: rendere `chat_template.jinja` con il motore Jinja
  di `swift-transformers`, come già fa il percorso tool di Gemma. Se il
  template usa costrutti non supportati: renderer Swift del sottoinsieme di
  3.6, verificato sugli stessi golden.
- **D9 Catalogo.** Nuovo `case qwen36` in `SupportedModelVariant` (valore
  `"qwen36"`) con una proprietà `family`. Installazione:
  `TurboFieldfareRepack --variant qwen36 --output scratch/qwen36.gturbo`.
  Cartella di default `qwen36.gturbo`. ID del modello nel server = `modelID`
  del manifest = `mlx-community/Qwen3.6-35B-A3B-4bit`.
- **D10 Sampling.** Default per famiglia (U4).
- **D11 Ragionamento.** Il testo dentro `<think>` non finisce mai nel
  contenuto visibile. CLI: ragionamento su stderr, contenuto su stdout, flag
  `--no-thinking`. Server: il ragionamento viene tolto da `content` (come oggi
  i "thought" di Gemma); `chat_template_kwargs.enable_thinking: false` lo
  disattiva. App: indicatore "Thinking…" durante il blocco e interruttore nelle
  impostazioni (solo Qwen).
- **D12 Immagini con Qwen.** Non disponibili: CLI `--image` → errore chiaro;
  server → errore 400 "image input is not available for this model"; app →
  allegati disattivati con lo stesso messaggio. Mai accettare un'immagine e
  poi ignorarla.
- **D13 Generazione indipendente dal modello.** Un protocollo `ChatTokenizer`
  raccoglie ciò che `runRawCompletion`, server e conversazione chiedono al
  tokenizer; `GFTokenizer` lo adotta con un'estensione, senza cambiare
  comportamento. Le funzioni esistenti restano con la stessa firma e chiamano
  le nuove.

## 3.9 Contratti condivisi

**C1 — Nomi.** `GTurboModelFamily` (formato, `package`), `ModelFamily`
(runtime, `public`: `.gemma4`, `.qwen35Moe`), `QwenHybridConfig`,
`QwenTokenizer`, `QwenDetokenizer`, `QwenChatFormat`, `QwenReasoningSplitter`,
`QwenToolCallParser`, `QwenAssistantDecoder`, `QwenHybridCache`,
`QwenHybridForwardRunner`, `ChatTokenizer`, `LanguageModelRunnerFactory`.

**C2 — Cartelle.**

| Cosa | Dove |
| --- | --- |
| Metal | `Sources/TurboFieldfare/Metal/Qwen/`: `qwen_norms.metal`, `qwen_head.metal`, `qwen_deltanet.metal`, `qwen_attention.metal`, `qwen_moe.metal` |
| Wrapper Swift dei kernel | `Sources/TurboFieldfare/Kernels/Qwen/` |
| Riferimenti CPU | `Sources/TurboFieldfareValidation/Support/Reference/Qwen/` |
| Runtime | `Sources/TurboFieldfare/Runtime/Inference/Qwen/`, `Sources/TurboFieldfare/Runtime/KVCache/Qwen/` |
| Tokenizer e chat | `Sources/TurboFieldfare/Tokenization/Qwen/` |
| Test | `Tests/<target>/…/Qwen/`, per esempio `Tests/TurboFieldfare/Core/Kernels/Qwen/` |
| Golden (in git) | `Tests/QwenFixtures/` (fuori da ogni target di test; si leggono con un percorso ricavato da `#filePath`) |
| Script | `docs/plans/qwen-migration/tools/` |
| Solo locale (fuori git) | `scratch/qwen36.gturbo`, `scratch/qwen36-tokenizer/`, `scratch/qwen36-venv/`, `scratch/qwen36-facts/`, `scratch/logs/` |

**C3 — Function constant Metal.** Usa solo gli indici 200–249: S12 200–204,
S13 205–209, S14 210–219, S15 220–229, S16 230–239, S17 240–249. Gli indici
fino a 139 sono già usati da Gemma.

**C4 — Chiavi del manifest Qwen.**

```json
"modelFamily": "qwen3_5_moe",
"qwen35": {
  "linearNumKeyHeads": 16, "linearNumValueHeads": 32,
  "linearKeyHeadDim": 128, "linearValueHeadDim": 128, "linearConvKernel": 4,
  "rmsNormEps": 1e-6, "normWeightsShifted": true,
  "attentionOutputGate": true, "sharedExpertGate": true
}
```

`normWeightsShifted` lo misura S1 (3.3, punto 1). Il resto della geometria sta
in `arch` (D1).

**C5 — Expert nel `.gturbo`.** Un blob per expert, nell'ordine gate, up,
down × weight, scales, biases (lo stesso del planner Gemma); stride
1.769.472 B; un file per layer in `packed_experts/`.

**C6 — Tokenizer nel `.gturbo`.** Cartella `tokenizer/` con gli stessi file
che il repack copia per Gemma (almeno `tokenizer.json`,
`tokenizer_config.json`, `chat_template.jinja`).

**C7 — Golden in `Tests/QwenFixtures/`.**

| File | Contenuto | Da |
| --- | --- | --- |
| `checkpoint/config.json`, `checkpoint/generation_config.json` | copie dei file del checkpoint alla revisione pinnata | S1 |
| `checkpoint/headers.json` | nome, shard, dtype, forma e byte di ogni tensore (output di Appendice A) | S1 |
| `tokenizer/tokenizer_golden.json` | `[{text, ids, decoded}]` | S2 |
| `chat/chat_golden.json` | `[{name, messages, tools?, enable_thinking?, add_generation_prompt, rendered, ids}]` | S2 |
| `model/prompts.json` | i 6 prompt di Appendice D con i loro `ids` | S8 |
| `model/logits_topk.json` | per prompt e posizione: `top_ids` e `top_logits` (20), teacher forcing | S8 |
| `model/logits_last.json` | logit completi dell'ultima posizione dei prompt p0 e p1 | S8 |
| `model/greedy.json` | 32 token greedy per prompt | S8 |
| `model/hidden_p0.json` | stati di p0: dopo l'embedding, dopo i layer 0, 3, 39, dopo la norma finale | S8 |
| `model/layer0_internals.json`, `model/layer3_internals.json` | grandezze interne di Appendice D, primi 8 token di p0 | S8 |

Array numerici: oggetto `{"shape": [...], "f32b64": "<base64>"}` (float32
little-endian) oppure `"f16b64"` (float16, solo per i logit completi).

**C8 — Tolleranze.**

- Kernel contro riferimento CPU: le `Tolerance` esistenti (`fp16Reduction`
  5e-3, `fp16ChainedReduction` 1e-2, `quantInt4` 1,5e-3).
- Stato DeltaNet FP32, kernel contro riferimento CPU: errore relativo ≤ 1e-4
  dopo 256 passi.
- Modello contro i golden di `mlx-lm` (stessi pesi): coseno dei logit completi
  ≥ 0,999 sull'ultima posizione; top-1 uguale in almeno il 98% delle posizioni
  (teacher forcing); continuazione greedy identica per almeno 4 prompt su 6
  (ogni divergenza va spiegata come quasi-pareggio, con i due logit in gioco).
- Diagnostica per layer: coseno degli stati nascosti ≥ 0,995 dopo i layer 0 e
  3, ≥ 0,99 dopo il layer 39.

**C9 — Test con il modello o con il tokenizer.** Un helper di test
`QwenTestResources` (in `Tests/TurboFieldfare/Core/Runtime/Qwen/`, creato in
S9) restituisce: l'URL di `scratch/qwen36.gturbo` se esiste `manifest.json`,
altrimenti `nil`; la cartella del tokenizer (`scratch/qwen36.gturbo/tokenizer/`
oppure `scratch/qwen36-tokenizer/`) o `nil`; l'URL di `Tests/QwenFixtures/`.
I test che ne hanno bisogno usano `@Test(.enabled(if: …))`.

---

# Parte 4 — Le sezioni

Ogni scheda ha la stessa forma: obiettivo, requisiti, cosa leggere, cosa puoi
modificare, passi, criterio di "fatto". Il report è sempre
`docs/plans/qwen-migration/reports/SN.md` e si può sempre modificare.

### S1 — Verifica dei fatti e stato dei test

**Obiettivo.** Confermare i fatti di 3.1–3.4 sui metadati reali, misurare i
due punti aperti di 3.3 e registrare quali test falliscono già prima di
iniziare. Nessun file Swift.

**Requisiti.** Modello reale: no. Download: risposta dell'API di Hugging Face,
`config.json`, `generation_config.json`, indice, header degli shard (range
request), pochi KB di quattro norme.

**Leggi.** Parte 3.1–3.4; Appendice A.

**Puoi modificare.** `docs/plans/qwen-migration/tools/headers.py`,
`docs/plans/qwen-migration/tools/facts.py`, `Tests/QwenFixtures/checkpoint/`;
in locale `scratch/qwen36-facts/`.

**Passi.**

1. Crea `tools/headers.py` dall'Appendice A. Scarica in `scratch/qwen36-facts/`
   la risposta dell'API, `config.json`, `generation_config.json`,
   `model.safetensors.index.json` e gli header. Confronta revisione e SHA-256
   dell'indice con 3.1. Copia `config.json`, `generation_config.json` e
   `headers.json` in `Tests/QwenFixtures/checkpoint/` (C7): così S5 e S6 li
   trovano su qualunque Mac.
2. Scrivi `tools/facts.py`, che dagli header e da `config.json` verifica: nomi
   di 3.3 per tutti i 40 layer (lineari e di attention secondo la maschera),
   forme e dtype, tensori a 8 bit (dalla larghezza impacchettata:
   `colonne × bit / 32`), assenza di `mtp.*`, numero di tensori della torre
   visiva, byte per expert e totali di 3.4, valori di 3.2.
3. Punti aperti di 3.3: con range request (funzione `fetch` di Appendice A)
   leggi i byte BF16 di `model.layers.0.input_layernorm.weight`,
   `model.norm.weight`, `model.layers.3.self_attn.q_norm.weight` e
   `model.layers.0.linear_attn.norm.weight`; calcola media e deviazione
   standard e decidi `normWeightsShifted`. Annota forma e dtype di
   `conv1d.weight`, `A_log` e `dt_bias`.
4. Esegui la suite completa (`Scripts/test.sh`, in background) ed elenca nel
   report i test che falliscono: è la "baseline".
5. Nel report, tabella "fatto | piano | osservato | esito". Se tutto coincide:
   `Stato: Completata`. Se qualcosa differisce: `Stato: Bloccata` con la
   correzione proposta (l'utente aggiorna la Parte 3).

**Fatto quando.** Tutti i fatti hanno un esito; `normWeightsShifted` deciso;
baseline dei test registrata; script committati.

### S2 — Golden di tokenizer e chat

**Obiettivo.** Produrre con Hugging Face i golden con cui S9 e S10
confronteranno il codice Swift.

**Requisiti.** Modello reale: no. Download: file del tokenizer (~25 MB) e
pacchetti `transformers`, `jinja2`, `huggingface_hub` nel venv.

**Leggi.** Parte 3.6; C7; Appendice B.

**Puoi modificare.** `docs/plans/qwen-migration/tools/` (file nuovi
`fetch_tokenizer.py`, `tok_golden.py`, `chat_golden.py`,
`cases/tokenizer_cases.json`, `cases/chat_cases.json`);
`Tests/QwenFixtures/tokenizer/`, `Tests/QwenFixtures/chat/`.

**Passi.**

1. `tools/fetch_tokenizer.py`: scarica alla revisione pinnata `tokenizer.json`,
   `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json` in
   `scratch/qwen36-tokenizer/` (cartella nuova e vuota) e stampa lo SHA-256 di
   ogni file. Annota gli SHA nel report.
2. Venv: `python3 -m venv scratch/qwen36-venv`, poi
   `scratch/qwen36-venv/bin/pip install transformers jinja2 huggingface_hub`
   (niente `torch`). Annota le versioni.
3. Descrivi nel report la pipeline di `tokenizer.json`: normalizer,
   pre-tokenizer (con la regex), tipo di modello, decoder, e gli ID di
   `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`,
   `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`.
   Servono a S9.
4. Casi del tokenizer: almeno 60 testi (italiano, inglese, cinese, emoji,
   codice, spazi e a capo multipli, numeri lunghi, token speciali scritti come
   testo). Golden con `tools/tok_golden.py`.
5. Casi della chat: almeno 14 conversazioni: con e senza system; con tool; con
   risposte dei tool; più tool call; ragionamento attivo e disattivato; più
   turni con ragionamento nei turni assistant precedenti; con e senza
   `add_generation_prompt`. Golden con `tools/chat_golden.py`.
6. Rilancia i due script e controlla che l'output sia identico (golden
   deterministici).

**Fatto quando.** I due golden sono in `Tests/QwenFixtures/`; pipeline e ID
speciali sono nel report.

### S3 — Contratti: famiglia, manifest, configurazione

**Obiettivo.** Il formato e il runtime riconoscono un manifest Qwen e lo
validano, senza alcuna logica di inferenza. Gemma invariato.

**Requisiti.** Modello reale: no. Download: nessuno. Report da leggere: S1
(`normWeightsShifted`).

**Leggi.** 3.7 (righe Format, ModelTypes, ManifestReader); D1, D2; C1, C4.
`Sources/TurboFieldfareFormat/GTurboManifestV1.swift`,
`Sources/TurboFieldfareFormat/GTurboFormatV1.swift`, `ModelTypes.swift`,
`ManifestReader.swift` (con `grep -n`), i test in `Tests/TurboFieldfareFormat/`
e `Tests/TurboFieldfareFormatCompatibility/`.

**Puoi modificare.** `Sources/TurboFieldfareFormat/GTurboManifestV1.swift` e un
file nuovo `GTurboModelFamilyV1.swift` nella stessa cartella;
`Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift` (solo
aggiunte), `ManifestReader.swift`, file nuovo `QwenHybridConfig.swift` nella
stessa cartella; test in `Tests/TurboFieldfareFormat/`,
`Tests/TurboFieldfareFormatCompatibility/`,
`Tests/TurboFieldfare/Core/Infrastructure/ModelIO/Qwen/`.

**Passi.**

1. `GTurboModelFamily` e i campi opzionali `modelFamily` e `qwen35` nel
   manifest (C4), con default `nil` nell'inizializzatore, così le chiamate
   esistenti compilano invariate. Test: ogni fixture di
   `Tests/TurboFieldfareFormatCompatibility/Fixtures/v1` si decodifica e si
   ricodifica come prima.
2. Runtime: `ModelFamily`, `QwenHybridConfig.qwen36_35B_A3B`,
   `ArchConfig.qwen36_35B_A3B` (valori di D1; `normWeightsShifted` da S1).
3. `ManifestReader`: un manifest Qwen si valida contro le costanti Qwen; un
   campo diverso dà `archMismatch` con il nome del campo (per esempio
   `qwen35.linearNumValueHeads`). Il percorso Gemma resta identico.
4. Test di fallimento chiuso: un manifest Qwen validato contro le attese di
   Gemma dà `archMismatch` su `hiddenSize`; un manifest con `modelFamily`
   sconosciuto viene rifiutato con un messaggio chiaro.

**Fatto quando.** Test di formato e di ModelIO verdi; fixture Gemma identiche.

### S4 — Catalogo: variante `qwen36`

**Obiettivo.** L'installer conosce il checkpoint Qwen pinnato. Nessuna
installazione ancora possibile: S5–S6 la rendono reale.

**Requisiti.** Modello reale: no. Download: nessuno. Report da leggere: S1.

**Leggi.** 3.1; D9; `Sources/TurboFieldfareRepack/Core/Remote/SupportedModelVariant.swift`,
`Sources/TurboFieldfareRepack/Core/Verification/SourceFingerprint.swift`,
`Sources/TurboFieldfareRepack/Command/main.swift` (`grep -n variant`),
`Sources/TurboFieldfareApp/Core/Installation/AppModelVariant.swift`.

**Puoi modificare.** `SupportedModelVariant.swift`, `SourceFingerprint.swift`,
`Sources/TurboFieldfareRepack/Command/main.swift`; ogni file dove il
compilatore chiede di gestire il nuovo `case` (solo il ramo minimo, niente
funzionalità nuove); test in `Tests/TurboFieldfareRepack/Core/`,
`Tests/TurboFieldfareApp/Core/`, `Tests/TurboFieldfare/Core/CLI/`.

**Passi.**

1. `case qwen36` con: `displayName` "Qwen3.6 35B-A3B 4-bit", `shortLabel`
   "Qwen3.6", `repoID`, `revision`, `sourceIndexSHA256` (3.1, confermati da
   S1), `approximateDownloadBytes` 20_401_929_952 e `installedBytes`
   19_500_000_000 provvisori (S6 li sostituisce), `defaultDirectoryName`
   `qwen36.gturbo`, `recommendedMinimumHostMemoryBytes` 16 GB provvisorio, e la
   proprietà `family` (`GTurboModelFamily` di S3).
2. `TurboFieldfareRepack --variant qwen36` viene accettato, ma finché S6 non è
   fatta l'installazione e il dry-run falliscono subito con "Qwen3.6
   installation is not implemented yet". Il default resta `int4`.
3. Ramificazioni minime richieste dal compilatore (app, CLI, server). Nell'app
   il selettore del modello **non** deve ancora mostrare Qwen: filtra le
   varianti con `family == .gemma4` (S24 toglie il filtro).
4. Test: valori del catalogo; `variant(forRepoID:)` e
   `variant(forSourceIndexSHA256:)` trovano `qwen36`; parsing di
   `--variant qwen36`; elenco dell'app invariato.

**Fatto quando.** Test verdi; l'help del repack elenca `qwen36`; il
comportamento con Gemma è identico.

### S5 — Repack 1: lettura di `config.json` e manifest Qwen

**Obiettivo.** Dal `config.json` di Qwen il repack ricava `arch` (D1) e il
blocco `qwen35` (C4), e li scrive nel manifest.

**Requisiti.** Modello reale: no. Download: nessuno (usa
`Tests/QwenFixtures/checkpoint/config.json`, salvato da S1).

**Leggi.** 3.2; D1; C4; `Sources/TurboFieldfareRepack/Core/Format/ArchInfo.swift`,
`Sources/TurboFieldfareRepack/Core/Format/GTurboJSON.swift`, e dove il
planner costruisce il manifest (`grep -rn "GTurboManifestV1(" Sources/TurboFieldfareRepack`).

**Puoi modificare.** File in `Sources/TurboFieldfareRepack/Core/Format/` e
`Sources/TurboFieldfareRepack/Core/Planning/` (solo la parte che costruisce il
manifest); test in `Tests/TurboFieldfareRepack/Core/Qwen/` (leggono
`Tests/QwenFixtures/checkpoint/config.json` con un percorso da `#filePath`).

**Passi.**

1. Lettore Qwen separato, per esempio `ArchInfo.loadQwen35(configPath:)`,
   scelto quando `model_type` è `qwen3_5_moe`. Il lettore Gemma non cambia.
2. Dati del blocco `qwen35` (C4) nel piano di repack, con
   `normWeightsShifted` da S1.
3. Il manifest scritto per Qwen contiene `modelFamily`, `qwen35` e `arch` di D1.
4. Test: il `config.json` reale dà esattamente i valori di D1 e C4; un
   `config.json` con un campo diverso viene rifiutato; i test del repack Gemma
   restano verdi e producono gli stessi manifest.

**Fatto quando.** Test verdi; manifest Qwen di test conforme a C4.

### S6 — Repack 2: mappa dei tensori, tokenizer e dry-run

**Obiettivo.** Il planner riconosce tutti i tensori Qwen; il dry-run reale
funziona e fissa le dimensioni del catalogo.

**Requisiti.** Modello reale: no. Download: metadati e header (dry-run).
Report da leggere: S1, S5.

**Leggi.** 3.3, 3.4; C5, C6; in `RepackPlanner.swift` le funzioni `classify`,
`routedExpertRole`, `plan`, `residentQuantizedWeightAlignment`
(con `grep -n`); `RepackPlanQuantSlots.swift`; in
`RemoteStreamingRepacker.swift` la lista dei file del tokenizer (~riga 463);
i test esistenti del planner (`grep -rln "RepackPlanner" Tests/TurboFieldfareRepack`).

**Puoi modificare.** File in `Sources/TurboFieldfareRepack/Core/` (Format,
Planning, Remote, Verification); `Sources/TurboFieldfareRepack/Command/main.swift`
(togliere il blocco di S4); `SupportedModelVariant.swift` (dimensioni); test
in `Tests/TurboFieldfareRepack/Core/Qwen/`.

**Passi.**

1. Classificazione per Qwen: expert da `.mlp.switch_mlp.{gate,up,down}_proj.`
   (attenzione: `mlp.gate.` è il router, non un expert); `mlp.gate`,
   `shared_expert*`, `linear_attn.*`, `self_attn.*`, norme, `lm_head` e
   `embed_tokens` residenti; `vision_tower.*` esclusi; qualunque altro nome →
   errore, come oggi.
2. Slot di quantizzazione: router = `mlp.gate` e `mlp.shared_expert_gate`
   (8 bit); embedding = `embed_tokens` e `lm_head`; attention = proiezioni di
   `linear_attn` e `self_attn`; tensori non quantizzati copiati come sono.
3. Expert impacchettati come C5, tokenizer come C6.
4. Test con safetensors sintetici piccoli (2 layer: uno lineare e uno di
   attention, 4 expert, hidden 64): piano, scrittura, manifest, stride degli
   expert, esclusione della visione, tensori BF16 e FP32 intatti. Un secondo
   test passa alla classificazione tutti i 2.090 nomi reali di
   `Tests/QwenFixtures/checkpoint/headers.json`: nessun nome "sconosciuto",
   40 × 3 nomi base di expert (senza `.scales` e `.biases`), 333 tensori della
   torre visiva esclusi.
5. Togli il blocco di S4 e fai il dry-run reale:
   `swift run -c release TurboFieldfareRepack --variant qwen36 --dry-run --output scratch/qwen36.gturbo`.
   Metti nel catalogo i valori esatti di `approximateDownloadBytes` e
   `installedBytes`, con un commento sul calcolo come per `int8`.

**Fatto quando.** Dry-run coerente con 3.4 (scostamenti spiegati nel report);
test verdi; piani Gemma invariati.

### S7 — Installazione reale di `qwen36.gturbo`

**Obiettivo.** Il modello è installato e verificato su questo Mac.

**Requisiti.** Modello reale: lo installa questa sezione. Download: ~20 GB
(decisione U1). Spazio libero: almeno 25 GB. Report da leggere: S6.

**Leggi.** Parte 2.4; Parte 1.5; Appendice C.

**Puoi modificare.** Se l'installazione fallisce per un errore del repack:
solo i file del repack toccati in S5–S6 e i loro test (massimo 3 tentativi).
In locale: `scratch/qwen36.gturbo`.

**Passi.**

1. Controlli di 2.4 (tranne il modello) e `df -h .`.
2. `swift build -c release` (log).
3. Installazione, in background:
   `.build/release/TurboFieldfareRepack --variant qwen36 --output scratch/qwen36.gturbo > scratch/logs/S7-install.log 2>&1`.
   Se si interrompe, ripeti con `--resume`. Se l'utente ha chiesto un'altra
   cartella, installa lì e crea il link `scratch/qwen36.gturbo`.
4. Verifica:
   `.build/release/TurboFieldfareRepack --verify-install --input-gturbo scratch/qwen36.gturbo`.
5. Nel report: tempi, dimensione (`du -sh`), `modelFamily` e `qwen35` letti da
   `manifest.json`, confronto con il dry-run di S6.

**Fatto quando.** Verifica superata; report con dimensioni e tempi.

### S8 — Golden del modello con `mlx-lm` ⛔

**Obiettivo.** Produrre i golden del modello (C7) e verificare le formule di
3.5 sul riferimento, prima di scrivere i kernel.

**Requisiti.** Mac con almeno 32 GB di RAM. Download: `mlx-lm` nel venv e il
checkpoint MLX alla revisione pinnata (~20 GB, cache di Hugging Face). Nessun
processo TurboFieldfare attivo (2.4). Report da leggere: S1.

**Leggi.** 3.2–3.5; C7, C8; Appendice D.

**Puoi modificare.** `docs/plans/qwen-migration/tools/make_model_golden.py`,
`tools/check_formulas.py`, `tools/cases/model_prompts.json`;
`Tests/QwenFixtures/model/`.

**Passi.**

1. Venv (crealo se manca, come in S2); installa `mlx-lm` e `huggingface_hub`;
   annota le versioni. Scarica il checkpoint con `snapshot_download` alla
   revisione pinnata.
2. Carica il modello con `mlx_lm.load`. Se `qwen3_5_moe` non è supportato,
   prova `mlx-vlm` e annotalo. Trova il file Python del modello installato
   (cartella `mlx_lm/models/`, file `qwen3_5*` o `qwen3_next*`) e annota nel
   report i nomi di classi e attributi del layer, di DeltaNet, dell'attention
   e del MoE. Leggi solo quelle classi.
3. `make_model_golden.py` (scheletro in Appendice D): per i 6 prompt, ID,
   logit top-20 per posizione, logit completi dell'ultima posizione di p0 e
   p1, 32 token greedy; per p0 gli stati nascosti; per i layer 0 e 3 le
   grandezze interne, ricostruite passo per passo con i sottomoduli del layer.
   Controllo obbligatorio: la ricostruzione passo per passo coincide con
   l'output del layer intero.
4. `check_formulas.py`: implementa le formule di 3.5 in NumPy (float32) con i
   pesi dequantizzati e confrontale con le grandezze interne (errore relativo
   ≤ 1e-2). Chiudi i punti aperti: ordine dei tap, mappa delle teste,
   `normWeightsShifted`, rinormalizzazione del router.
5. Dimensione totale dei golden sotto i 5 MB; commit.
6. Nel report: versioni, tempi, esito di ogni formula. Se una formula di 3.5 è
   sbagliata, scrivi quella giusta e la riga "Correzioni al piano: sì"
   (l'utente aggiorna la 3.5 prima di S12); altrimenti "Correzioni al piano:
   no". In entrambi i casi, a golden prodotti: `Stato: Completata`.

**Fatto quando.** Golden in git; formule verificate o corrette nel report;
nessun processo `mlx` rimasto attivo.

### S9 — Tokenizer Qwen in Swift

**Obiettivo.** `QwenTokenizer` e `QwenDetokenizer` danno gli stessi ID e lo
stesso testo dei golden di S2.

**Requisiti.** Modello reale: no. Download: i file del tokenizer, solo se non
ci sono né `scratch/qwen36.gturbo/tokenizer/` né `scratch/qwen36-tokenizer/`
(con `tools/fetch_tokenizer.py`). Report da leggere: S2.

**Leggi.** 3.6; D7; C1, C2, C7, C9; in `Tokenization/Tokenizer.swift` le righe
1–260 (caricamento, `encode`, `decode`); `Tokenization/Detokenizer.swift`;
`Tests/TurboFieldfare/Core/Tokenization/TokenizerTests.swift` (stile).

**Puoi modificare.** File nuovi in `Sources/TurboFieldfare/Tokenization/Qwen/`;
test in `Tests/TurboFieldfare/Core/Tokenization/Qwen/`; l'helper
`QwenTestResources` in `Tests/TurboFieldfare/Core/Runtime/Qwen/` (C9).

**Passi.**

1. `QwenTestResources` (C9).
2. `QwenTokenizer.load(from:)` con `AutoTokenizer` (stesso percorso di
   `GFTokenizer`: `LanguageModelConfigurationFromHub(modelFolder:)`); verifica
   della pipeline descritta in S2; errore chiaro se diversa.
3. ID speciali da `added_tokens` (nomi di S2); `stopTokenIDs` =
   `<|im_end|>` e `<|endoftext|>`; `encode(_:addBOS:)` (Qwen non usa BOS: il
   parametro viene ignorato, documentalo) e `decode(_:skipSpecialTokens:)`.
4. `QwenDetokenizer` in streaming, con `push`/`flush` come `GFDetokenizer`:
   mappa byte-level di GPT-2, sequenze UTF-8 incomplete trattenute finché si
   completano, token "barriera" che svuotano il buffer.
5. Test: tutti i golden (ID e testo); decodifica token per token uguale a
   quella in blocco; emoji e caratteri cinesi spezzati su più token.

**Fatto quando.** 100% dei golden uguali. Se `swift-transformers` dà ID
diversi e non si può correggere senza toccare la dipendenza: `Bloccata`, con
l'elenco dei casi.

### S10 — Formato chat Qwen in Swift

**Obiettivo.** Il prompt di chat (con tool e ragionamento) coincide con i
golden; le continuazioni si costruiscono in coda ai token già in cache.

**Requisiti.** Modello reale: no. Report da leggere: S2, S9.

**Leggi.** 3.6; D8; C7; in `Tokenizer.swift` (con `grep -n`) `Message`,
`FunctionDefinition`, `applyChatTemplate`, `encodeToolChat`,
`encodeTextContinuation`, `encodeToolResultContinuation`;
`Tests/TurboFieldfare/Core/Tokenization/ChatTemplateTests.swift` (stile).

**Puoi modificare.** File nuovi in `Sources/TurboFieldfare/Tokenization/Qwen/`;
test in `Tests/TurboFieldfare/Core/Tokenization/Qwen/`.

**Passi.**

1. `QwenChatFormat`: rende `chat_template.jinja` con `swift-transformers`
   (`messages`, `tools`, `enable_thinking`, `add_generation_prompt`). Se il
   motore Jinja non regge il template, renderer Swift del sottoinsieme di 3.6.
2. Metodi su `QwenTokenizer` con gli stessi nomi e lo stesso significato di
   quelli di `GFTokenizer` (stessi tipi `GFTokenizer.Message` e
   `GFTokenizer.FunctionDefinition`), più un parametro `enableThinking` con
   default `true`.
3. Continuazioni (nuovo turno utente, risultati dei tool): token da aggiungere
   dopo un turno assistant già in cache (vedi l'ultima regola di 3.6). Studia
   cosa fanno quelle di Gemma e replica la stessa semantica.
4. Test: tutti i golden della chat (testo e ID); per ogni golden con più
   turni, "prompt del turno precedente + risposta + continuazione" = prompt
   successivo, dove il template lo permette (annota i casi in cui il template
   toglie il ragionamento).

**Fatto quando.** Golden della chat uguali; test Gemma della chat invariati.

### S11 — Parser del ragionamento e delle tool call

**Obiettivo.** Dal testo generato in streaming separare ragionamento,
contenuto e tool call, producendo gli stessi eventi che oggi produce Gemma.

**Requisiti.** Modello reale: no.

**Leggi.** 3.6; D11; `Tokenization/StructuredAssistantDecoder.swift`,
`Tokenization/GemmaToolCallParser.swift` (con `grep -n`: tipi
`StructuredAssistantEvent`, `ParsedToolCall`).

**Puoi modificare.** File nuovi in `Sources/TurboFieldfare/Tokenization/Qwen/`;
test in `Tests/TurboFieldfare/Core/Tokenization/Qwen/`.

**Passi.**

1. `QwenReasoningSplitter`: riceve frammenti di testo, emette ragionamento e
   contenuto; parte già "dentro" il ragionamento se il prompt finisce con
   `<think>\n`, "fuori" se finisce con `</think>\n\n`; regge tag spezzati tra
   due frammenti.
2. `QwenToolCallParser`: formato XML di 3.6 → `ParsedToolCall`. Il valore di
   un parametro si prova prima come JSON (numeri, booleani, oggetti, liste),
   altrimenti è una stringa; togli un solo a capo iniziale e finale. XML
   malformato → errore chiaro.
3. `QwenAssistantDecoder`: unisce i due e produce `StructuredAssistantEvent`
   (`.content`, `.toolCall`). Il ragionamento esce da una proprietà o callback
   separata: **non** aggiungere casi all'enum pubblico.
4. Test: tag spezzati in ogni posizione possibile; più tool call; contenuto
   prima di una tool call; ragionamento disattivato; XML malformato.

**Fatto quando.** Test verdi; nessuna modifica ai file Gemma.

### S12 — Kernel 1: norme (e cartella Metal Qwen)

**Obiettivo.** I kernel di normalizzazione di Qwen, verificati contro i
riferimenti CPU.

**Requisiti.** Modello reale: no. Report da leggere: S1, S8 (eventuali
correzioni alla 3.5 devono essere già nel piano).

**Leggi.** 3.5 (notazione, DeltaNet, attention); D4; C2, C3, C8;
`Kernels/Primitives/RMSNorm.swift`, `Metal/Primitives/rmsnorm.metal`,
`Infrastructure/Metal/MetalContext.swift` righe 125–175; un riferimento CPU
esistente in `Sources/TurboFieldfareValidation/Support/Reference/Primitives/`
e il suo test.

**Puoi modificare.** I cinque file di C2 in `Sources/TurboFieldfare/Metal/Qwen/`
(creali come segnaposto: `#include <metal_stdlib>`, `using namespace metal;`
e un commento con la sezione proprietaria); in `MetalContext.swift` solo
`shaderModules` e `shaderSubdirectories` (aggiunte in coda); file nuovi in
`Sources/TurboFieldfare/Kernels/Qwen/` e
`Sources/TurboFieldfareValidation/Support/Reference/Qwen/`; test in
`Tests/TurboFieldfare/Core/Kernels/Qwen/`.

**Passi.**

1. Segnaposto Metal registrati in coda; la libreria compila (i test esistenti
   di `MetalContext` restano verdi).
2. Kernel in `qwen_norms.metal`: RMSNorm di riga con `w` o `1 + w`
   (function constant 200); RMSNorm per testa (q e k, 256 valori); RMSNorm
   gated per testa con `silu(z)` (128 valori); norma L2 per testa (128 valori,
   eps 1e-6). Ingresso e uscita FP16, accumulo FP32; una riga (decode) e `T`
   righe (prefill).
3. Riferimenti CPU e test, con dimensioni piccole e reali.

**Fatto quando.** Test verdi entro C8; controllo "Gemma intoccato" pulito.

### S13 — Kernel 2: router, embedding, testa finale

**Obiettivo.** Embedding, testa finale non legata, router softmax top-8 e gate
scalare dello shared expert.

**Requisiti.** Modello reale: no.

**Leggi.** 3.5 (embedding e testa, MoE: router e `sg`); C3, C8;
`Kernels/Quant/DequantInt4GEMV.swift`, `Kernels/Quant/DequantInt8GEMV.swift`,
`Kernels/Quant/EmbedLookupInt4.swift`; in `Kernels/Sampling/LogitOutput.swift`
come si leggono i logit (con `grep -n`).

**Puoi modificare.** `Sources/TurboFieldfare/Metal/Qwen/qwen_head.metal`; file
nuovi in `Kernels/Qwen/` e in `Reference/Qwen/`; test in
`Tests/TurboFieldfare/Core/Kernels/Qwen/`.

**Passi.**

1. Embedding: usa `EmbedLookupInt4` se accetta vocabolario 248.320 e
   `d` 2048 senza modifiche (verificalo con un test); nessuna scala.
2. Testa: `DequantInt4GEMV` generico, `m` 248.320, `n` 2048 → logit FP16.
3. Router: `DequantInt8GEMV` (`m` 256, `n` 2048) + kernel `qwen_router_topk`
   (softmax FP32 su 256, top-8, rinormalizzazione → indici `uint` e pesi). A
   parità di valore vince l'indice più basso: documentalo.
4. Gate dello shared expert: `DequantInt8GEMV` (`m` 1) + sigmoide.
5. Riferimenti CPU e test, piccoli e reali (per la testa basta un sottoinsieme
   di righe nel confronto CPU).

**Fatto quando.** Test verdi entro C8.

### S14 — Kernel 3: DeltaNet, convoluzione e gate

**Obiettivo.** La prima metà del layer lineare: convoluzione causale con stato,
gate `beta`/`decay`, preparazione di `q`, `k`, `v`.

**Requisiti.** Modello reale: no. Report da leggere: S1 e S8 (layout e ordine
dei tap della conv).

**Leggi.** 3.5 (layer lineare fino a `decay`); D5; C3, C8; i wrapper di S12
(`grep -n "func encode\|init(" Sources/TurboFieldfare/Kernels/Qwen/*.swift`).

**Puoi modificare.** `Metal/Qwen/qwen_deltanet.metal`; file nuovi in
`Kernels/Qwen/` e in `Reference/Qwen/`; test in `Kernels/Qwen/`.

**Passi.**

1. Convoluzione causale depthwise (8192 canali, kernel 4) con SiLU: decode
   (`T = 1`, aggiorna lo stato) e prefill (`T > 1`, stato in ingresso e in
   uscita). Stato FP32 con il layout di D5.
2. Gate per testa (32): `beta = sigmoid(b)`, `g = -exp(A_log) · softplus(a + dt_bias)`,
   `decay = exp(g)`, in FP32.
3. Preparazione: separa `q`, `k`, `v`; norma L2 di `q` e `k` (kernel di S12);
   `q / sqrt(128)`.
4. Riferimenti CPU e test: `T = 1`, `T = 7`, `T = 300`; continuità (prefill di
   `T1` token, poi `T2` passi di decode = prefill di `T1 + T2`).

**Fatto quando.** Test verdi entro C8.

### S15 — Kernel 4: DeltaNet, ricorrenza e uscita

**Obiettivo.** La scansione ricorrente dello stato e l'uscita del layer
lineare.

**Requisiti.** Modello reale: no.

**Leggi.** 3.5 (layer lineare, da "stato S_j" in poi, e la nota sui kernel);
D5; C3, C8; i wrapper di S12 e S14.

**Puoi modificare.** `Metal/Qwen/qwen_deltanet.metal` (in coda a quanto fatto
in S14); file nuovi in `Kernels/Qwen/` e in `Reference/Qwen/`; test in
`Kernels/Qwen/`.

**Passi.**

1. Kernel di scansione: un SIMD-group per (testa di valore, colonna `c`), la
   colonna nei registri (32 lane × 4 righe), i token in ordine; stato FP32 nel
   buffer di D5; uscita `o` per token e testa. Stesso kernel per `T = 1` e
   `T > 1`.
2. Uscita: RMSNorm gated per testa (S12) con `silu(z)`, poi `out_proj` con
   `DequantInt4GEMV` (4096 → 2048) e somma al residuo.
3. Riferimenti CPU e test: `T = 1`, `T = 7`, `T = 300`, teste ridotte e
   dimensioni reali; errore dello stato dopo 256 passi (C8); continuità come in
   S14.

**Fatto quando.** Test verdi entro C8.

### S16 — Kernel 5: attention con gate

**Obiettivo.** Il mixer dei 10 layer di attention.

**Requisiti.** Modello reale: no.

**Leggi.** 3.5 (layer di attention); D4, D5; C3, C8; in
`Kernels/Attention/Attention.swift` le firme di `encodeFull` e
`splitGeometry`; in `Kernels/Primitives/RoPE.swift` come è fatta la RoPE
parziale (solo per capire se si può chiamare).

**Puoi modificare.** `Metal/Qwen/qwen_attention.metal`; file nuovi in
`Kernels/Qwen/` e in `Reference/Qwen/`; test in `Kernels/Qwen/`.

**Passi.**

1. Separazione di query e gate dall'uscita di `q_proj` (per testa: 256 di query
   e 256 di gate); `q_norm` e `k_norm` (S12).
2. RoPE parziale: prime 64 dimensioni, coppie `(i, i + 32)`, theta 1e7.
3. Scrittura di K e V nel layout KV di D5 (buffer e stride come parametri).
4. Attention causale GQA (16 query, 2 KV, rapporto 8), `head_dim` 256, scala
   1/16, softmax FP32; decode (una query) e prefill (`T` query). Riusa un
   kernel esistente solo se si può chiamare senza modifiche; altrimenti un
   kernel `qwen_` semplice e corretto.
5. Gate d'uscita `sigmoid(gate) ⊙ o`.
6. Riferimenti CPU e test, piccoli e reali, anche con 4096 posizioni in
   cache.

**Fatto quando.** Test verdi entro C8.

### S17 — Kernel 6: MoE SiLU e shared expert

**Obiettivo.** Il blocco MoE completo di un layer.

**Requisiti.** Modello reale: no.

**Leggi.** 3.4 (byte per expert), 3.5 (MoE); C3, C5, C8; in
`Kernels/MoE/MoE.swift` come arrivano i buffer degli expert e gli slot
(`grep -n "func encode"`); in `Metal/MoE/moe.metal` solo la struttura;
`Kernels/MoE/SharedExpertInt4.swift`; l'API pubblica di
`Infrastructure/Streaming/PreadExpertStreamer.swift` (`grep -n "public\|func "`).

**Puoi modificare.** `Metal/Qwen/qwen_moe.metal`; file nuovi in
`Kernels/Qwen/` e in `Reference/Qwen/`; test in `Kernels/Qwen/`.

**Passi.**

1. Expert routed INT4 (H 2048, I 512, SiLU) letti da un blob con gli offset di
   C5; 8 expert per token, somma pesata.
2. Shared expert SwiGLU INT4 con il gate scalare di S13.
3. Uscita del MoE e somma al residuo.
4. Prefill: un ciclo sui token va bene (corretto prima che veloce).
5. Test con blob sintetici in buffer Metal (senza streaming), piccoli e con
   dimensioni reali.

**Fatto quando.** Test verdi entro C8.

### S18 — Caricamento del modello e cache ibrida

**Obiettivo.** `Model.load` apre un `.gturbo` Qwen e ne verifica lo schema;
esiste `QwenHybridCache`.

**Requisiti.** Modello reale: sì, solo per il test di caricamento (2.4).

**Leggi.** 3.3, 3.4; D1, D2, D5, D6; C9; in `Runtime/Inference/Model.swift`
le righe 29–130, 340–500 e 524–610; `ManifestReader.swift` (con `grep -n`);
`Runtime/KVCache/KVCacheManager.swift` (con `grep -n`, per lo stile).

**Puoi modificare.** `Runtime/Inference/Model.swift` (solo il punto in `load`
che sceglie la validazione per famiglia); file nuovi in
`Runtime/Inference/Qwen/` (per esempio `Model+Qwen.swift` con schema e
accessori) e in `Runtime/KVCache/Qwen/`; test in
`Tests/TurboFieldfare/Core/Runtime/Qwen/`.

**Passi.**

1. In `Model.load`: se il manifest è Qwen, validazione contro le costanti Qwen
   (D2) e `validateQwenRuntimeSchema` (tutti i tensori di 3.3 con le forme di
   3.4). Gemma resta identico.
2. Accessori: `model.family`, la configurazione Qwen, i tensori per layer, la
   testa `lm_head` separata.
3. `QwenHybridCache`: allocazione per `maxContext`, `reset`, `snapshot()`,
   `restore(_:)`, `truncate(to:)` (D5, D6).
4. Test: caricamento reale riuscito (abilitato solo con il modello, C9);
   dimensioni della cache a 4K e 32K uguali a 3.4; snapshot e ripristino
   identici byte per byte.

**Fatto quando.** Test verdi; caricamento reale annotato nel report (tempo).

### S19 — Runner: decode token per token ⛔

**Obiettivo.** Il modello Qwen genera token, e i logit coincidono con i
golden di `mlx-lm`.

**Requisiti.** Modello reale: sì (2.4), dal passo 2. Report da leggere: S18
(solo "Note per le sezioni successive"); S8 solo se un confronto diverge.

**Una sessione per passo.** Scrivere `QwenHybridForwardRunner` richiede di
conoscere molto codice, più di quanto stia in un contesto: per questo la
sezione è divisa in passi piccoli, e la mappa del passo 1 sostituisce la
rilettura del codice. Dopo il commit e il push di ogni passo, aggiungi alla
mappa ciò che il passo successivo deve sapere (nomi, buffer, scoperte) e
chiudi la sessione con una riga di riepilogo: la sezione riprende dal passo
successivo con il contesto vuoto (`run-sections.sh` lo fa da solo; a mano, con
il prompt di 1.4). Dal passo 2 in poi leggi solo la scheda, il report, la
mappa e ciò che il passo elenca; mai `RealForwardRunner.swift`.

**Leggi.** Ogni passo dice cosa leggere. Le grandezze dei golden sono in C7 e
in Appendice D ("Grandezze interne"); le tolleranze in C8; i test con il
modello seguono C9.

**Puoi modificare.** File nuovi in `Runtime/Inference/Qwen/`; test in
`Tests/TurboFieldfare/Core/Runtime/Qwen/`; la mappa
`docs/plans/qwen-migration/reports/S19-map.md`. CLI, server e app non si
toccano.

**Passi.**

1. Mappa, senza codice Swift. Scrivi `reports/S19-map.md`, al massimo 150
   righe: per ogni chiamata che il runner farà, la firma esatta e
   `file:riga`; per un layer DeltaNet, un layer di attention, il MoE e la
   testa, l'ordine delle operazioni di 3.5 con il wrapper che esegue ciascuna
   e le forme dei buffer; come si leggono gli expert instradati (C5). Leggi:
   3.5; D3, D5; C5; `Runtime/Generation/LogitProducer.swift`; le firme dei
   wrapper (`grep -n "func encode\|init(" Sources/TurboFieldfare/Kernels/Qwen/*.swift Sources/TurboFieldfare/Kernels/Quant/*.swift`);
   accessori di `Runtime/Inference/Qwen/Model+Qwen.swift` e di
   `Runtime/Inference/ModelExpertIO.swift`, API di `QwenHybridCache` (con
   `grep -n "public\|func "`); nomi e forme delle chiavi dei golden di C7 (con
   `python3`, senza stampare i valori).
2. Scheletro. `QwenHybridForwardRunner` in
   `Runtime/Inference/Qwen/QwenHybridForwardRunner.swift`: inizializzazione
   (viste sui tensori, cache, buffer di lavoro, pipeline), conforme a
   `LogitProducer`, `ContinuableLogitProducer` e `ContextWindowReporting`.
   Per ora `produce(token:position:into:)` fa embedding → norma finale →
   `lm_head` → logit FP16, senza layer; prefill v1 = un token alla volta.
   Prese di debug (solo per i test) per lo stato nascosto dopo l'embedding o
   dopo un layer scelto, e per le grandezze interne dei passi 3–5. Test con il
   modello: embedding dei token di p0 contro `hidden_p0.json` → `embedding`
   (coseno ≥ 0,9999). Leggi: la mappa.
3. Mixer DeltaNet (layer lineari): norma d'ingresso, proiezioni, conv e SiLU,
   preparazione, scansione con lo stato in `QwenHybridCache`, norma gated,
   proiezione d'uscita, residuo. Test sul layer 0 con i token di p0: `x_norm`,
   `qkv_raw`, `z`, `a`, `b`, `conv_out`, `q`, `k`, `v`, `beta`, `g`,
   `core_out`, `gated_out`, `mixer_out` e `state_heads0to3` di
   `layer0_internals.json`: coseno ≥ 0,995 per ciascuna, come i layer in C8.
   Per lo stato annota nel report anche l'errore relativo: la soglia 1e-4 di
   C8 vale per il kernel contro il riferimento CPU (S15), non contro
   `mlx-lm`, che arriva allo stato con un'aritmetica diversa. Leggi: la
   mappa; in 3.5 la parte DeltaNet.
4. MoE e layer 0 completo: norma prima del MoE, router top-8, expert
   instradati letti dal file del layer (C5), shared expert con il suo gate,
   somma e residuo. Test: `moe_x`, `router_top_ids` (uguali, salvo
   quasi-pareggi annotati con i due pesi), `router_top_weights`,
   `routed_out`, `shared_out`, `layer_out` di `layer0_internals.json`; stato
   dopo il layer 0 contro `hidden_p0.json` → `layer0` (coseno ≥ 0,995, C8).
   Leggi: la mappa; in 3.5 la parte MoE; C5.
5. Attention (layer 3): norma, proiezioni di q, gate, k e v, norme per testa,
   RoPE, scrittura nella cache KV, attention, gate, proiezione d'uscita,
   residuo e il MoE del passo 4. I layer 1 e 2 sono lineari come lo 0:
   esegui i layer 0–2 e confronta il layer 3 con tutte le chiavi di
   `layer3_internals.json` (coseno ≥ 0,995) e con `hidden_p0.json` →
   `layer3` (coseno ≥ 0,995, C8). Leggi: la mappa; in 3.5 la parte
   attention.
6. Modello completo: i 40 layer secondo la maschera, norma finale, `lm_head`;
   reset e posizioni per un prompt nuovo. Test: `hidden_p0.json` → `layer39`
   (coseno ≥ 0,99) e `final_norm`; top-20 su tutte le posizioni di tutti i
   prompt contro `logits_topk.json` (top-1 uguale in almeno il 98%); logit
   completi dell'ultima posizione di p0 e p1 contro `logits_last.json`
   (coseno ≥ 0,999). Leggi: la mappa.
7. Greedy: il ciclo sta nel test (con `QwenTokenizer`), non nel CLI. 32 token
   per i 6 prompt contro `greedy.json`, identici per almeno 4 prompt su 6
   (C8); ogni divergenza spiegata come quasi-pareggio, con i due logit in
   gioco. Leggi: la mappa; C9.
8. Nel report: la continuazione greedy di p0 e p4 decodificata in testo, i
   numeri di C8, il tempo per token. Poi 2.9.

**Fatto quando.** Tolleranze C8 rispettate; testo sensato nel report. Se un
confronto diverge, segui 2.5 (layer per layer, poi grandezze interne).

### S20 — Runner: prefill a blocchi e riuso del prefisso

**Obiettivo.** Prefill di più token per volta, continuazione e riuso del
prefisso con gli snapshot.

**Requisiti.** Modello reale: sì (2.4), dal passo 2. Report da leggere: S19
(solo "Note per le sezioni successive").

**Una sessione per passo.** Come S19: il prefill estende un runner di circa
1000 righe, e il codice da conoscere non sta in un contesto. La mappa del
passo 1 sostituisce la rilettura del codice. Dopo il commit e il push di ogni
passo, aggiungi alla mappa ciò che il passo successivo deve sapere e chiudi la
sessione con una riga di riepilogo: la sezione riprende dal passo successivo
con il contesto vuoto. Dal passo 2 in poi leggi solo la scheda, il report, la
mappa, `S19-map.md` e ciò che il passo elenca. Di
`QwenHybridForwardRunner.swift` leggi solo gli intervalli di righe che la mappa
indica; `RealForwardRunner.swift` mai, se non nel passo 1.

**Leggi.** Ogni passo dice cosa leggere. Le grandezze dei golden sono in C7 e
in Appendice D; le tolleranze in C8; i test con il modello seguono C9.

**Puoi modificare.** I file Qwen di S12–S19; file nuovi in
`Runtime/Inference/Qwen/` e `Runtime/Prefill/Qwen/`; test in
`Tests/TurboFieldfare/Core/Runtime/Qwen/` e `Kernels/Qwen/`; la mappa
`docs/plans/qwen-migration/reports/S20-map.md`. Un kernel Metal nuovo solo se
quello esistente non accetta più righe (D4). Gemma, CLI, server e app non si
toccano.

**Passi.**

1. Mappa, senza codice Swift. Scrivi `reports/S20-map.md`, al massimo 150
   righe:
   - le firme esatte, con `file:riga`, di `ChunkedPrefillRunner`,
     `PrefillOutputMode`, `PrefillSeed`, `PrefillResult` e
     `PrefillRuntimeConfig` (dimensione del blocco);
   - la semantica di `reset`, `rewind(to:)`, `continuationPosition`,
     `prepareForContinuation` e `prefillChunked` in `RealForwardRunner.swift`,
     di cui leggi **solo** le righe 455–500 e 572–640;
   - per ogni kernel Qwen che il runner usa, il parametro del numero di righe
     (`tokens`, `rows`, `queryTokens`) e il layout di `T` righe; come
     `QwenMoE.encodePrefill` vuole il routing di `T` token (unione degli
     expert, slot, pesi);
   - se `MPPPrefillInt4QMM.encode` accetta i pesi Qwen così come sono (layout
     di pesi, scale e bias; `k` multiplo di `tileK`; `m ≥ 64` per Apple10).
     Scrivi la decisione: GEMM, oppure GEMV per token;
   - i buffer di lavoro del runner S19 dimensionati per una riga, da portare
     a `T` righe, e i punti in cui il runner aspetta la CPU, con gli
     intervalli di righe di `QwenHybridForwardRunner.swift` da leggere nei
     passi successivi.

   Leggi: `S19-map.md`; D5, D6; `Runtime/Generation/LogitProducer.swift`;
   `Runtime/Prefill/PrefillRuntimeConfig.swift` e
   `Kernels/TensorCore/MPPPrefillInt4QMM.swift` (con `grep -n`); le firme dei
   wrapper (`grep -n "func encode" Sources/TurboFieldfare/Kernels/Qwen/*.swift`);
   `QwenHybridForwardRunner.swift` con `grep -n "func \|let .*Buffer\|var .*Buffer"`.
2. Scheletro. Conformità a `ChunkedPrefillRunner` in un file nuovo, per
   esempio `QwenHybridForwardRunner+Prefill.swift`: buffer di lavoro per `T`
   righe (`T` = blocco di `PrefillRuntimeConfig`), ciclo sui blocchi con
   `onProgress`, embedding di `T` token. Presa di debug (solo per i test) per
   lo stato nascosto di tutte le righe del blocco. Test con il modello:
   embedding di tutte le posizioni di p0 contro `hidden_p0.json` →
   `embedding` (coseno ≥ 0,9999), a blocchi di 2 token (2+2+1: più blocchi,
   l'ultimo parziale) e in un blocco unico; `newPosition` = 5. Un blocco di
   qualunque dimensione si ottiene con `replacingChunkTokens(_:)`. Leggi: la
   mappa.
3. Mixer DeltaNet su `T` righe: norma d'ingresso, proiezioni (GEMM o GEMV,
   come dice la mappa), conv con il suo stato, gate, scansione con
   `tokens: T`, norma gated, proiezione d'uscita, residuo. Test sul layer 0
   con p0, a blocchi di 2: `mixer_out` di `layer0_internals.json` su tutte le
   posizioni (coseno ≥ 0,995); stato DeltaNet e stato conv a fine prompt
   contro quelli del decode token per token del runner S19 (coseno ≥ 0,9999;
   annota l'errore relativo nel report). Leggi: la mappa; in 3.5 la parte
   DeltaNet.
4. MoE su `T` righe: norma prima del MoE, router per `T` token con una sola
   lettura degli id sulla CPU per layer, expert dell'unione degli id letti una
   volta per blocco a gruppi di al massimo 16, gli slot del layer (C5): ogni
   gruppo si legge, si esegue e si attende prima di leggere il successivo.
   Poi shared expert e residuo per token; `QwenMoE.encodePrefill` non si usa
   (presume gli stessi 8 expert per tutti i token). Test a blocchi di 2:
   layer 0 completo contro `hidden_p0.json` → `layer0` su tutte le
   posizioni (coseno ≥ 0,995, C8) e contro il decode token per token (coseno
   ≥ 0,9999). Leggi: la mappa; in 3.5 la parte MoE; C5.
5. Attention su `T` righe (layer 3): norma, proiezioni di q, gate, k e v,
   norme per testa, RoPE con le posizioni `start…start+T−1`, scrittura di `T`
   righe nella cache KV, attention causale con `queryTokens: T`, gate,
   proiezione d'uscita, residuo, MoE del passo 4. Test: layer 0–3 a blocchi di
   2, `hidden_p0.json` → `layer3` su tutte le posizioni (coseno ≥ 0,995) e
   contro il decode token per token (coseno ≥ 0,9999). Leggi: la mappa; in 3.5
   la parte attention.
6. Modello completo ed equivalenza: i 40 layer secondo la maschera, norma
   finale e `lm_head` sull'ultima riga, `outputMode` (`PrefillSeed`),
   `PrefillResult.newPosition`. Test per i 6 prompt (da 5 a 28 token), a
   blocchi di 4, così ogni prompt occupa più blocchi, e in un blocco unico:
   stato dopo la norma finale su tutte le posizioni contro il decode token per
   token (coseno ≥ 0,9999); logit dell'ultima posizione
   contro il decode (coseno ≥ 0,9999, top-1 uguale) e, per p0 e p1, contro
   `logits_last.json` (coseno ≥ 0,998, la deviazione approvata in S19). I test
   di S19 restano verdi. Leggi: la mappa.
7. Continuazione e snapshot secondo D6: `continuationPosition`,
   `prepareForContinuation`, `rewind(to:)`; uno snapshot a fine prefill
   (stato DeltaNet e conv, circa 63 MiB, al massimo 2: esce il più vecchio);
   errore nuovo `rewindUnsupported` quando non c'è uno snapshot utilizzabile.
   Test:
   - il prefill di p0 seguito dal prefill in coda di altri token dà gli stessi
     logit del prefill completo (coseno ≥ 0,9999);
   - prefill di un prompt A, poi di un prompt B; `rewind` alla fine di A e
     prefill della coda di A+X danno gli stessi logit del prefill completo di
     A+X;
   - `rewind` a una posizione senza snapshot lancia `rewindUnsupported`, e
     dopo `reset` il runner funziona.

   Leggi: la mappa; D6.
8. Tempi e chiusura: token al secondo in prefill con 512 e 4096 token (blocco
   di default di `PrefillRuntimeConfig`) e in decode, indicando il tipo di
   build. Scrivili nel report, poi 2.9.

**Fatto quando.** Test verdi, compresi quelli di S19; tempi nel report. Se un
confronto diverge, segui 2.5 (layer per layer, poi grandezze interne).

### S21 — Generazione indipendente dal modello

**Obiettivo.** Il codice di generazione accetta tokenizer e runner di entrambe
le famiglie; una factory sceglie il runner. Gemma invariato.

**Requisiti.** Modello reale: sì, per una prova breve.

**Leggi.** D3, D7, D13; in `Runtime/Generation/RawCompletion.swift` le righe
270–420; `Tokenization/Detokenizer.swift`; gli usi di `GFTokenizer` nel
modulo (`grep -rn "GFTokenizer" Sources/TurboFieldfare/Runtime`); i protocolli
di `Runtime/Generation/LogitProducer.swift`; il report di S20 (solo "Note per
le sezioni successive").

**Puoi modificare.** File nuovi `Sources/TurboFieldfare/Tokenization/ChatTokenizer.swift`
e `Sources/TurboFieldfare/Runtime/Inference/LanguageModelRunnerFactory.swift`;
`RawCompletion.swift` (solo overload e inoltro); i file Qwen in
`Tokenization/Qwen/`; test in `Tests/TurboFieldfare/Core/Runtime/Generation/Qwen/`.
Le estensioni di conformità stanno nei due file nuovi, così `Tokenizer.swift`,
`Detokenizer.swift` e `RealForwardRunner.swift` non cambiano.

**Passi.**

1. Protocollo pubblico `ChatTokenizer` con ciò che `runRawCompletion` usa:
   `vocabSize`, `stopTokenIDs`, `endOfTurnID`, `structuralMarkerIDs`,
   `encode(_:addBOS:)`, `decode(_:skipSpecialTokens:)`, più due requisiti con
   nomi nuovi, così nessuna proprietà esistente cambia:
   - `toolResultWaitID: Int32?`, l'ID che indica "aspetto un risultato di
     tool": per Gemma il suo `toolResponseID`, per Qwen `nil` (il turno si
     chiude con `<|im_end|>`). Un requisito `toolResponseID: Int32?` non
     funziona: in entrambi i tokenizer `toolResponseID` è già un `Int32` non
     opzionale.
   - `makeStreamingDetokenizer(barrierTokenIDs: Set<Int32>) -> any
     StreamingDetokenizer`, con il protocollo pubblico `StreamingDetokenizer`
     (`mutating func push(_ id: Int32) -> String`, `mutating func flush() ->
     String`). `GFDetokenizer` e `QwenDetokenizer` hanno già questa forma: lo
     adottano con un'estensione vuota e restano interni. Oggi
     `runRawCompletion` crea un `GFDetokenizer` concreto: la versione nuova
     chiama questo metodo.

   I requisiti non hanno valori di default, quindi chi chiama passa `addBOS` e
   `skipSpecialTokens` espliciti: per `skipSpecialTokens` i default di Gemma e
   Qwen sono diversi (`true` e `false`). `GFTokenizer` adotta il protocollo
   con un'estensione; anche `QwenTokenizer` (`endOfTurnID` = `imEndID`).
2. Overload di `runRawCompletion` con `any ChatTokenizer`; la versione con
   `GFTokenizer` chiama la nuova e si comporta come prima.
3. Protocollo pubblico `LanguageModelRunner: ContinuableLogitProducer` con
   `maxContext` e `rewind(to:) throws`: è ciò che CLI, server e app chiamano
   oggi su `RealForwardRunner`, oltre a `reset` e `continuationPosition`, che
   sono già in `ContinuableLogitProducer`. Entrambi i runner lo adottano con
   un'estensione vuota, perché i membri esistono già e sono pubblici.
   - `LanguageModelRunnerFactory.make(model:context:maxContext:runtimeConfiguration:)
     throws -> any LanguageModelRunner` sceglie con `model.family`
     (`ModelFamily`, `.gemma4` o `.qwen35Moe`): `RealForwardRunner` per Gemma,
     con `runtimeConfiguration`; `QwenHybridForwardRunner` per Qwen, senza,
     perché il runner non ha quel parametro. Annota nel report, per S22–S24,
     che per Qwen la configurazione runtime non arriva al runner.
   - `LanguageModelRunnerFactory.loadTokenizer(family:forModelDirectory:)
     async throws -> any ChatTokenizer` chiama `load(forModelDirectory:)` di
     `GFTokenizer` o di `QwenTokenizer`.
   - Prefill a blocchi e immagini restano come oggi: `runRawCompletion` cerca
     `ChunkedPrefillRunner` con `as?`, e le immagini restano di
     `RealForwardRunner` (D12). Per Qwen `rewind` lancia
     `QwenRewindError.rewindUnsupported` fuori dagli snapshot di D6: gestirlo
     è compito di S23 e S25, non di S21.
4. Test: generazione con `ScriptedLogitProducer` (in
   `Sources/TurboFieldfareValidation/Support/Fixtures/`) e tokenizer Qwen: stop
   su `<|im_end|>`; prova reale breve (32 token greedy di p0 con la factory).
   I test Gemma della generazione restano verdi senza modifiche.

**Fatto quando.** Test verdi; nessuna differenza di comportamento per Gemma.

### S22 — CLI con Qwen

**Obiettivo.** `TurboFieldfareCLI --model scratch/qwen36.gturbo` funziona in
modalità testo libero e chat.

**Requisiti.** Modello reale: sì, prove brevi.

**Leggi.** D10, D11, D12; `Sources/TurboFieldfareCLI/Run.swift` (con
`grep -n`), `Sources/TurboFieldfareCLI/Args.swift` (con `grep -n`),
`Sources/TurboFieldfareCLI/ModelVariant.swift`;
`Tests/TurboFieldfare/Core/CLI/` (stile).

**Puoi modificare.** `Sources/TurboFieldfareCLI/`; test in
`Tests/TurboFieldfare/Core/CLI/`.

**Passi.**

1. Caricamento tramite la factory di S21; la riga che stampa la variante
   mostra anche la famiglia.
2. `--prompt` (testo libero) e `--chat-prompt` / `--messages-file` (formato
   Qwen di S10); flag nuovo `--no-thinking`; ragionamento su stderr e
   contenuto su stdout (D11), con il parser di S11.
3. Default di sampling per famiglia quando l'utente non passa i flag (U4).
4. `--image` con Qwen: errore chiaro (D12).
5. Test di parsing e di instradamento; prove reali brevi, per esempio:
   `swift run -c release TurboFieldfareCLI --model scratch/qwen36.gturbo --prompt "The capital of France is" --max-new 32 --temperature 0`
   e un `--chat-prompt` con e senza `--no-thinking`. Output nel report.

**Fatto quando.** Test verdi; prove reali nel report; Gemma invariato.

### S23 — Server OpenAI-compatibile con Qwen

**Obiettivo.** Il server serve Qwen con chat, tool call e ragionamento, con le
stesse regole di sicurezza di oggi.

**Requisiti.** Modello reale: sì, prove brevi; server solo su `127.0.0.1`.

**Leggi.** D6, D11, D12; `docs/OPENAI_SERVER.md` (parti su tool e prompt
cache); in `Sources/TurboFieldfareServer/Core/ServerInference.swift` (con
`grep -n`) `load`, prompt cache e tool; `ServerModelVariant.swift`;
`GemmaToolSchema.swift` e `OpenAIModels.swift` (con `grep -n`, compresa la
lista `unsupportedKeys`); `Tests/TurboFieldfareServer/` (stile).

**Puoi modificare.** `Sources/TurboFieldfareServer/`, `docs/OPENAI_SERVER.md`;
test in `Tests/TurboFieldfareServer/`.

**Passi.**

1. Caricamento con la factory; famiglia e variante nel log; `/v1/models` con
   l'ID Qwen; `model_not_found` per altri ID.
2. Chat con il formato Qwen; tool resi dal template; tool call XML → campo
   OpenAI `tool_calls`; risultati dei tool nel formato Qwen; ragionamento tolto
   da `content`; `chat_template_kwargs.enable_thinking` accettato (solo per
   Qwen; con Gemma, stesso comportamento di oggi per le chiavi non supportate).
3. Prompt cache: con Qwen, riuso solo secondo D6; altrimenti prefill completo.
4. Immagini con Qwen: errore 400 (D12).
5. Test con `ScriptedLogitProducer` e fixture; prova reale breve con `curl`
   (server avviato da te, chiuso alla fine). Aggiorna `docs/OPENAI_SERVER.md`.

**Fatto quando.** Test verdi; prova reale nel report; controllo dei link del
Markdown pulito.

### S24 — Mac app 1: catalogo, installazione, memoria

**Obiettivo.** Nell'app si può scegliere, installare e ammettere in memoria
Qwen3.6.

**Requisiti.** Modello reale: no (le prove nell'app le fa l'utente).

**Leggi.** AGENTS.md (parte SwiftUI); D9, D10, D12; 3.4 (KV per token, stati);
`Sources/TurboFieldfareApp/Core/Installation/` (`AppModelVariant.swift`,
`AppModelInstallDescriptor.swift`, `AppModelLocation.swift`),
`Core/Configuration/MacAppSettings.swift` (con `grep -n variant`),
`Sources/TurboFieldfare/Runtime/Configuration/ContextAdmission.swift`; la vista
delle impostazioni (`grep -rln "variant" Sources/TurboFieldfareApp/MacPresentation`).

**Puoi modificare.** `Sources/TurboFieldfareApp/`,
`Sources/TurboFieldfareDecodeService/`, `Sources/TurboFieldfareDecodeProtocol/`,
`ContextAdmission.swift`; test in `Tests/TurboFieldfareApp/`,
`Tests/TurboFieldfareDecodeService/`, `Tests/TurboFieldfare/Core/Runtime/Configuration/`.

**Passi.**

1. Togli il filtro di S4: il selettore mostra Qwen3.6; descrittore di
   installazione e cartella `qwen36.gturbo`; download con l'installer
   esistente.
2. `ContextAdmission` per famiglia. Qwen: KV 20.480 B per token, stato DeltaNet
   e conv 64 MiB, non-KV **provvisorio** 2,5 GiB, riserva 3 GiB (S27 misura).
   Test a 4K, 32K, 128K e 262.144.
3. Default di sampling per famiglia (U4): se l'utente non ha cambiato i valori
   si usano quelli della famiglia; se li ha cambiati restano i suoi. Descrivi
   nel report il comportamento scelto.
4. Sezione immagini: con Qwen "not available for this model" (D12).
5. Controlli Ruby SwiftUI (2.9); test; costruisci l'app con
   `Scripts/build_app.sh`.

**Fatto quando.** Test verdi; app costruita; Gemma invariato.

### S25 — Mac app 2: conversazione e ragionamento

**Obiettivo.** Una chat con Qwen nell'app funziona: più turni, ragionamento,
cronologia separata per modello.

**Requisiti.** Modello reale: no (le prove le fa l'utente). Report da leggere:
S21, S23 e S24 (solo "Note per le sezioni successive").

**Una sessione per passo.** I file da conoscere sono grandi
(`MultimodalConversation.swift` 852 righe, `RealInferenceClient.swift` 1042,
`AppModelHistory.swift` 1135): la mappa `reports/S25-map.md` del passo 1
sostituisce la loro rilettura. Dopo il commit e il push di ogni passo, aggiungi
alla mappa ciò che il passo successivo deve sapere e chiudi la sessione con una
riga di riepilogo. Dal passo 2 in poi leggi solo la scheda, il report, la mappa
e gli intervalli di righe che la mappa indica; mai un file grande per intero.

**Leggi.** Ogni passo dice cosa leggere; D6, D11, D12, D13 quando il passo li
nomina. Per SwiftUI valgono le regole di AGENTS.md: `@StoredState`,
`PreviewProvider`, niente attributi `State` diretti né macro `Preview`.

**Puoi modificare.** `MultimodalConversation.swift` (generalizzazione, con
l'inizializzatore esistente invariato) e un file nuovo accanto,
`Runtime/Generation/ConversationPromptRenderer.swift`;
`Sources/TurboFieldfareApp/`, `Sources/TurboFieldfareDecodeService/`,
`Sources/TurboFieldfareDecodeProtocol/`; test in `Tests/TurboFieldfareApp/`,
`Tests/TurboFieldfareDecodeService/`, `Tests/TurboFieldfare/Core/Runtime/Generation/`;
la mappa `docs/plans/qwen-migration/reports/S25-map.md`. Il runtime di Gemma,
CLI e server non si toccano. Un campo nuovo nel protocollo di decodifica è
opzionale: un messaggio senza quel campo si legge come oggi.

**Passi.**

1. Mappa, senza codice Swift. Scrivi `reports/S25-map.md`, al massimo 150
   righe, con `file:riga` e firme esatte:
   - `MultimodalConversation.swift`: l'inizializzatore (riga ~316); ogni uso di
     `tokenizer`, `runner`, `model`, `GFTokenizer`, `RealForwardRunner`; dove
     rende il prompt (`applyChatTemplate`, `encodeTextContinuation`,
     `encodeMultimodalUserContinuation`, righe ~800–830); dove chiama
     `rewind` e `reset` (righe ~180–225, ~590–625) e cosa succede se
     `rewind` lancia un errore.
   - `RealInferenceClient.swift`: caricamento (`Model.load`, `RealForwardRunner(`,
     `loadTokenizer`, righe ~430–500); le due creazioni di
     `MultimodalConversation` (righe ~376 e ~674); dove i delta di testo
     diventano eventi per l'app.
   - `DecodeProtocol.swift`: `DecodeGenerationRequest` (campi di sampling) e
     gli eventi che portano il testo; in `DecodeService/Entry.swift` come il
     servizio usa il client.
   - L'app: dove stanno le impostazioni di sampling (per il nuovo
     interruttore), dove la HUD mostra lo stato della generazione (per
     "Thinking…"), come la cronologia rifiuta di continuare una chat di
     un'altra variante (`grep -n variant` in `ConversationRecord.swift` e
     `AppModelHistory.swift`), chi scrive il campo `thinking` di
     `ConversationRecord` (righe ~195–196).
   - Le API Qwen da usare: `LanguageModelRunner` e
     `LanguageModelRunnerFactory` (S21), `ChatTokenizer`, `QwenTokenizer`
     (`applyChatTemplate`, `encodeTextContinuation` con `enableThinking`),
     `QwenAssistantDecoder` e `onReasoning` (S11), `QwenRewindError` (S20).

   Leggi: i report indicati sopra; `grep -n` e intervalli brevi dei file
   elencati.
2. Prompt per famiglia. Nel file nuovo, un protocollo
   `ConversationPromptRenderer` con quello che `MultimodalConversation` chiede
   al tokenizer per il primo turno e per la continuazione, adottato da Gemma
   (stesse chiamate di oggi) e da Qwen (con `enableThinking`).
   `MultimodalConversation` riceve un inizializzatore nuovo con il renderer,
   `any ChatTokenizer` e `any LanguageModelRunner`; quello esistente resta e
   chiama il nuovo. Le immagini restano possibili solo con un runner
   `MultimodalPrefillRunner`: con Qwen, un turno con immagini è un errore D12
   prima di qualunque lavoro. Test: i test esistenti di
   `MultimodalConversation` verdi senza modifiche; nuovi test con
   `ScriptedLogitProducer` e tokenizer Qwen: due turni, il secondo continua
   la cache senza `reset`. Leggi: la mappa; D13, D12.
3. `rewind` con Qwen (D6). Dove `MultimodalConversation` chiama `rewind`
   (stop-string, errori, Stop), `QwenRewindError.rewindUnsupported` non deve
   rompere la conversazione: `reset` e rigioco dei token registrati, come già
   fa il ripristino di una chat. Test con un runner finto che adotta
   `LanguageModelRunner` e lancia `rewindUnsupported`: dopo il rewind
   mancato, la posizione corrisponde ai token registrati e il turno
   successivo funziona. Leggi: la mappa; D6.
4. Caricamento nel decode service. `RealInferenceClient` carica con la
   factory (`make` e `loadTokenizer(family:forModelDirectory:)`) e crea la
   conversazione con l'inizializzatore nuovo; con Gemma tutto come oggi.
   Immagini con Qwen: errore D12 prima di qualunque lavoro sul modello. Test:
   i test dell'app e del decode service restano verdi; un test della scelta
   per famiglia con un manifest Qwen di prova, senza modello. Leggi: la mappa.
5. Ragionamento nella pipeline. Con Qwen il testo dentro `<think>` non arriva
   mai al contenuto visibile (D11), con `QwenAssistantDecoder`; l'app riceve
   uno stato "sta ragionando" finché il blocco è aperto. Un campo opzionale
   `enableThinking` in `DecodeGenerationRequest` (assente = acceso) arriva
   fino al renderer del passo 2. Test: andata e ritorno del protocollo con e
   senza il campo; con `ScriptedLogitProducer`, il ragionamento resta fuori
   dal testo e lo stato si spegne a `</think>`. Leggi: la mappa; D11.
6. Ragionamento nell'interfaccia: indicatore "Thinking…" mentre lo stato del
   passo 5 è acceso; interruttore nelle impostazioni, visibile solo con Qwen
   caricato, che diventa `enableThinking`. Con Gemma l'interfaccia non
   cambia. Test di presentazione; `ruby Scripts/check_swiftui_macros.rb
   --self-test` e `ruby Scripts/tests/swift_macro_probe_test.rb` con
   `LC_ALL=en_US.UTF-8`. Leggi: la mappa; le regole SwiftUI di AGENTS.md.
7. Cronologia: una chat registrata con un altro modello non si continua, come
   già succede tra varianti (la variante `qwen36` di S24 potrebbe già
   bastare: verificalo prima di aggiungere qualcosa). Test: una chat Gemma
   con Qwen caricato, e viceversa, si apre in sola lettura. Leggi: la mappa.
8. Chiusura: app costruita (`swift build -c release` e `Scripts/build_app.sh`),
   controlli Ruby, suite completa. Nel report, l'elenco delle prove che
   l'utente deve fare nell'app con Qwen e con Gemma. Poi 2.9.

**Fatto quando.** Test verdi; app costruita; elenco delle prove nel report.

### S26 — Prestazioni di Qwen: prefill più veloce

**Obiettivo.** Il prefill di Qwen diventa più veloce senza cambiare i
risultati. S20 misura 51 tok/s a 4096 token (release, Mac16,9), circa 4 volte
il decode. Il decode non peggiora.

**Requisiti.** Modello reale: sì (2.4), dal passo 1. Tutta la sezione gira
sullo stesso Mac, perché i passi 2–4 confrontano i logit bit per bit con quelli
del passo 1. Report da leggere: S20 (solo "Valori misurati o fissati" e "Note
per le sezioni successive").

**Una sessione per passo.** Come S19 e S20: la mappa
`reports/S26-map.md` del passo 1 sostituisce la rilettura del codice. Dopo il
commit e il push di ogni passo, aggiungi alla mappa ciò che il passo
successivo deve sapere e chiudi la sessione con una riga di riepilogo. Di
`QwenHybridForwardRunner.swift` e `QwenHybridForwardRunner+Prefill.swift` leggi
solo gli intervalli di righe che la mappa indica.

**Leggi.** Ogni passo dice cosa leggere. Le misure seguono AGENTS.md: build
release (`Scripts/test.sh -c release --filter …`), niente controlli
sperimentali, niente Instruments, e la variante (Qwen3.6 4-bit) nominata in
ogni risultato.

**Puoi modificare.** I file Qwen di S12–S20: runner, kernel in `Kernels/Qwen/`
e `Metal/Qwen/` (D4), test Qwen; la mappa
`docs/plans/qwen-migration/reports/S26-map.md`. Gemma, `ModelExpertIO.swift`,
`PreadExpertStreamer.swift`, CLI, server e app non si toccano. I default di
runtime (dimensione del blocco, slot degli expert) non cambiano: se una misura
suggerisce un default diverso, scrivilo nel report come proposta.

**Regola dei risultati.** I passi 2–4 non cambiano l'aritmetica: gli hash dei
logit salvati al passo 1 devono restare identici. Il passo 5 cambia
l'aritmetica e segue le soglie di S20: C8 contro i golden, e contro il decode
le soglie approvate nel passo 6 di S20.

**Passi.**

1. Mappa e misure di partenza, senza ottimizzare.
   - Scrivi `reports/S26-map.md`, al massimo 120 righe. Per ogni costo del
     prefill indica dove nasce, con `file:riga`: lettura degli expert a gruppi
     e attesa; un `makeBlobsArgumentBuffer` per ogni coppia (token, slot); le
     chiamate `rows: 1` dei kernel per testa (S20, passo 3); lo staging dello
     shared expert; le attese della GPU per layer. Aggiungi le firme esatte di
     `planRoutedExperts(layer:experts:avoidingSlots:)`,
     `fetchRoutedExperts(plan:)`, `adviseRoutedExperts(plan:)` e
     `routedExpertCacheSlotCount(layer:)` di `ModelExpertIO`, e scrivi se
     `MPPPrefillInt4QMM` può leggere gate, up e down dal blob C5 (offset di
     `QwenMoE.expertOffsets`).
   - Test di misura: estendi `prefillAndDecodeThroughputAreMeasured` di S20.
     Oltre ai tok/s, ripartisce il tempo di un blocco: lettura degli expert,
     tempo GPU dei command buffer (`gpuStartTime`/`gpuEndTime`), tempo CPU di
     codifica, e per layer il numero di gruppi, di expert letti e di dispatch.
     I contatori vivono nel runner solo per i test, come le prese di debug.
     Misura il prefill di 512 e 4096 token a blocchi di 128 e di 256, e il
     decode di 128 token.
   - Hash di riferimento: SHA-256 dei logit FP16 dell'ultima posizione dei 6
     prompt, a blocchi di 4 e di 128, e dei logit di 8 passi di decode greedy
     dopo p0. Salvali in `scratch/S26/baseline.json` e nel report, con il
     modello di Mac.

   Nel report: la tabella delle misure e le tre voci più costose. Leggi: il
   report di S20; `S20-map.md` (MoE e test); `ModelExpertIO.swift` con
   `grep -n "func "`.
2. Meno allocazioni e meno dispatch.
   - Un argument buffer per expert del gruppo (al massimo 16), riusato da tutti
     i token che lo usano, al posto di uno per coppia (token, slot); uno per
     layer per lo shared expert.
   - I kernel per testa che oggi girano con `rows: 1` per riga accettano un
     passo di riga, così una sola chiamata copre le `T` righe. Il default
     resta quello di oggi, quindi il decode non cambia.

   Test: hash identici al passo 1; test di S19, S20 e dei kernel Qwen verdi;
   misura come al passo 1. Leggi: la mappa.
3. Shared expert pronto per tutti i layer: lo staging nel blob C5 (S19) si fa
   una volta, all'inizializzazione, in 40 blob (circa 71 MB), invece che a ogni
   cambio di layer. Vale per decode e prefill; annota la memoria in più nel
   report per S27. Test e misura come al passo 2. Leggi: la mappa.
4. Lettura e calcolo in parallelo: mentre la GPU esegue un gruppo di expert,
   la CPU legge il gruppo successivo in slot diversi, con
   `planRoutedExperts(…avoidingSlots:)` e `fetchRoutedExperts(plan:)` (per
   esempio due metà da 8 slot che si alternano). Mai leggere in uno slot che la
   GPU sta ancora usando. Se la mappa lo indica utile, chiedi in anticipo il
   gruppo successivo con `adviseRoutedExperts(plan:)`. Test e misura come al
   passo 2. Leggi: la mappa.
5. Expert per colonne, solo se le misure dei passi 1–4 dicono che il calcolo
   GPU del MoE domina. Per ogni expert del gruppo: le righe dei token che lo
   usano in un buffer contiguo; gate e up con `MPPPrefillInt4QMM` (`m` = quei
   token); SiLU·mul; down; somma pesata in `routedOut[t][j]` nello stesso
   ordine di S19. I kernel piccoli che servono sono nuovi, in `Metal/Qwen/`
   (D4). Test: le soglie di S20 contro golden e decode, e top-1 uguale al
   decode in 12 casi su 12. Se le misure dicono che non conviene, salta il
   passo e scrivi perché nel report. Leggi: la mappa; C5.
6. Misure finali e chiusura: la tabella del passo 1 prima e dopo (prefill di
   512 e 4096 token a blocchi di 128 e 256, decode); nel report una proposta
   motivata sul blocco di default per Qwen, senza cambiarlo. Poi 2.9.

**Fatto quando.** Hash identici dopo i passi 2–4; test di S19 e S20 verdi;
tabella prima e dopo nel report. Obiettivo indicativo: prefill a 4096 token
almeno il doppio di S20 (≥ 100 tok/s) e decode non sotto 12,5 tok/s. Se
l'obiettivo non si raggiunge, la sezione si chiude comunque, con le misure e
le cause rimaste nel report: non è un motivo per `Bloccata`.

### S27 — Validazione finale, misure, benchmark, documentazione ⛔

**Obiettivo.** Correttezza finale, Gemma senza regressioni, memoria misurata,
benchmark e documentazione.

**Requisiti.** Modello reale: sì, Qwen e Gemma 4-bit (`scratch/gemma4.gturbo`)
sullo stesso Mac.

**Leggi.** C8; `docs/COMMUNITY_BENCHMARKS.md`; i report S19–S26.

**Puoi modificare.** `ContextAdmission.swift` e i suoi test (valori misurati
per Qwen); `README.md`, `AGENTS.md`, `docs/BENCHMARKS.md`,
`docs/SYSTEM_DESIGN.md`, `docs/RUNTIME_CONTROLS.md`; script e log in
`scratch/`.

**Passi.**

1. Correttezza: tutti i test Qwen con il modello (C8).
2. Regressione di Gemma: output greedy 4-bit sui prompt del benchmark
   identico a quello di `main`. Procedura senza worktree: annota l'hash di
   `origin/main`, `git checkout --detach origin/main`, build release, salva gli
   output in `scratch/`, `git checkout qwen-support`, build release,
   confronta.
3. Memoria: picco reale a 4K, 32K e 128K; `ContextAdmission` per Qwen passa
   da provvisorio a misurato (con un commento sul metodo).
4. Benchmark secondo `docs/COMMUNITY_BENCHMARKS.md`, Qwen e Gemma 4-bit, senza
   controlli sperimentali; nomina la variante in ogni risultato.
5. Documentazione: README (installazione e uso di Qwen), AGENTS.md (modello
   `scratch/qwen36.gturbo` nei controlli e nei comandi), BENCHMARKS,
   SYSTEM_DESIGN (DeltaNet e cache ibrida), RUNTIME_CONTROLS.
6. Report finale: risultati, deviazioni, elenco delle prove nell'app per
   l'utente. Il merge in `main` lo decide l'utente.

**Fatto quando.** Tutto verde; documentazione aggiornata; report finale
completo.

---

## Appendice A — `tools/headers.py` e lettura di un tensore

Legge gli header safetensors con range request: pochi MB, nessun peso.

```python
# usage: python3 -I headers.py <out_dir>
import json, struct, sys, urllib.request, os

REPO = "mlx-community/Qwen3.6-35B-A3B-4bit"
REV = "38740b847e4cb78f352aba30aa41c76e08e6eb46"
SHARDS = 4

def url(name):
    return f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"

def fetch(name, start, end):
    req = urllib.request.Request(url(name), headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req).read()

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tensors = {}
    for i in range(1, SHARDS + 1):
        name = f"model-{i:05d}-of-{SHARDS:05d}.safetensors"
        header_len = struct.unpack("<Q", fetch(name, 0, 7))[0]
        header = json.loads(fetch(name, 8, 8 + header_len - 1))
        header.pop("__metadata__", None)
        for key, value in header.items():
            start, end = value["data_offsets"]
            tensors[key] = {"shard": name, "dtype": value["dtype"], "shape": value["shape"],
                            "start": 8 + header_len + start, "bytes": end - start}
    with open(os.path.join(out_dir, "headers.json"), "w") as f:
        json.dump(tensors, f)
    print(len(tensors))

if __name__ == "__main__":
    main(sys.argv[1])
```

Per leggere i byte di un tensore (per esempio una norma BF16):
`fetch(t["shard"], t["start"], t["start"] + t["bytes"] - 1)`. Un BF16 si
converte in float32 spostando i suoi 16 bit nella metà alta di un `uint32`.

Revisione e SHA dell'indice:

```bash
curl -sL https://huggingface.co/api/models/mlx-community/Qwen3.6-35B-A3B-4bit | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["sha"])'
curl -sL https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/resolve/38740b847e4cb78f352aba30aa41c76e08e6eb46/model.safetensors.index.json -o scratch/qwen36-facts/index.json
shasum -a 256 scratch/qwen36-facts/index.json
```

## Appendice B — Golden di tokenizer e chat

Si eseguono con `scratch/qwen36-venv/bin/python -I`.

```python
# usage: python -I tok_golden.py <tokenizer_dir> <cases.json> <out.json>
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(f"{sys.argv[1]}/tokenizer.json")
out = []
for text in json.load(open(sys.argv[2], encoding="utf-8")):
    ids = tok.encode(text, add_special_tokens=False).ids
    out.append({"text": text, "ids": ids,
                "decoded": tok.decode(ids, skip_special_tokens=False)})
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out))
```

```python
# usage: python -I chat_golden.py <tokenizer_dir> <cases.json> <out.json>
# cases.json: lista di {"name", "messages", "tools"?, "enable_thinking"?, "add_generation_prompt"}
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
out = []
for case in json.load(open(sys.argv[2], encoding="utf-8")):
    kwargs = {"tokenize": False,
              "add_generation_prompt": case.get("add_generation_prompt", True)}
    if "tools" in case:
        kwargs["tools"] = case["tools"]
    if "enable_thinking" in case:
        kwargs["enable_thinking"] = case["enable_thinking"]
    text = tok.apply_chat_template(case["messages"], **kwargs)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    out.append(dict(case, rendered=text, ids=ids))
json.dump(out, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out))
```

## Appendice C — Comandi utili

```bash
Scripts/test.sh --filter <NomeSuite>
Scripts/test.sh
LC_ALL=en_US.UTF-8 ruby Scripts/check_markdown_links.rb
LC_ALL=en_US.UTF-8 ruby Scripts/check_swiftui_macros.rb --self-test
memory_pressure -Q
swift build -c release
swift run -c release TurboFieldfareRepack --variant qwen36 --dry-run --output scratch/qwen36.gturbo
swift run -c release TurboFieldfareRepack --variant qwen36 --output scratch/qwen36.gturbo
swift run -c release TurboFieldfareRepack --verify-install --input-gturbo scratch/qwen36.gturbo
swift run -c release TurboFieldfareCLI --model scratch/qwen36.gturbo --prompt "The capital of France is" --max-new 32
```

`--variant qwen36` esiste da S4 (installazione reale da S6); il CLI con Qwen
funziona da S22.

## Appendice D — Golden del modello (S8)

**Prompt** (`tools/cases/model_prompts.json`):

| Nome | Tipo | Testo |
| --- | --- | --- |
| p0 | testo libero | `The capital of France is` |
| p1 | testo libero | `def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n` |
| p2 | testo libero | `La Divina Commedia è un poema scritto da` |
| p3 | testo libero | `人工智能的发展历史可以追溯到` |
| p4 | chat, ragionamento disattivato | utente: `What is 17 * 23? Answer with the number only.` |
| p5 | chat, ragionamento attivo | utente: `Name three primary colors.` |

I prompt di chat si rendono con `apply_chat_template(add_generation_prompt=True, enable_thinking=…)`
e si tokenizzano senza token speciali aggiunti.

**Grandezze interne** (primi 8 token di p0, chiavi dei file di C7):

- `layer0_internals.json`: `x_norm` [T,2048]; `qkv_raw` [T,8192]; `z`
  [T,4096]; `a`, `b` [T,32]; `conv_out` [T,8192] (dopo SiLU); `q`, `k`
  [T,16,128] (dopo la norma L2; `q` già diviso per sqrt(128)); `v` [T,32,128];
  `beta`, `g` [T,32]; `core_out` [T,32,128] (prima della norma gated);
  `gated_out` [T,32,128]; `mixer_out` [T,2048]; `moe_x` [T,2048];
  `router_top_ids`, `router_top_weights` [T,8]; `routed_out`, `shared_out`
  [T,2048] (`shared_out` già moltiplicato per il gate); `layer_out` [T,2048];
  `state_heads0to3` [4,128,128] (stato finale delle teste 0–3, indici
  `[testa][riga i][colonna c]`).
- `layer3_internals.json`: `x_norm`; `q_pre_norm`, `gate` [T,16,256];
  `k_pre_norm`, `v` [T,2,256] (`q_pre_norm` e `k_pre_norm`, nonostante il
  nome, sono q e k dopo la norma per testa e prima del RoPE);
  `q_rope` [T,16,256], `k_rope` [T,2,256] (dopo norma e RoPE); `attn_out` [T,16,256] (prima del gate); `gated` [T,4096];
  `mixer_out`, `layer_out` [T,2048].

**Scheletro di `make_model_golden.py`** (da completare leggendo le classi del
modello in `mlx_lm/models/`):

```python
# usage: scratch/qwen36-venv/bin/python -I make_model_golden.py <model_dir> <prompts.json> <out_dir>
import base64, json, os, sys
import mlx.core as mx
import numpy as np
from mlx_lm import load

def enc(a, f16=False):
    a = np.asarray(a, dtype=np.float32)
    if f16:
        return {"shape": list(a.shape), "f16b64": base64.b64encode(a.astype("<f2").tobytes()).decode()}
    return {"shape": list(a.shape), "f32b64": base64.b64encode(a.astype("<f4").tobytes()).decode()}

def full_logits(model, ids):
    out = model(mx.array([ids]))          # [1, T, V], teacher forcing senza cache
    mx.eval(out)
    return np.array(out[0].astype(mx.float32))

def greedy(model, ids, n=32):
    ids = list(ids)
    new = []
    for _ in range(n):                    # semplice: ricalcola tutto a ogni passo
        nxt = int(np.argmax(full_logits(model, ids)[-1]))
        ids.append(nxt)
        new.append(nxt)
    return new

model, tok = load(sys.argv[1])
prompts = json.load(open(sys.argv[2], encoding="utf-8"))
out_dir = sys.argv[3]
# 1. ids di ogni prompt (testo libero: tok.encode senza token speciali;
#    chat: tok.apply_chat_template(..., tokenize=False) poi tok.encode)
# 2. logits_topk.json: per prompt e posizione, top-20 id e valori
# 3. logits_last.json: logit completi dell'ultima posizione di p0 e p1 (f16)
# 4. greedy.json: greedy(model, ids) per ogni prompt
# 5. hidden_p0.json: copia il ciclo dei layer dal __call__ del modello di testo
#    della versione installata (stesse maschere e cache) e cattura gli stati
# 6. layer0/layer3_internals.json: ricostruisci il layer con i suoi sottomoduli,
#    seguendo il codice di mlx-lm, e verifica che l'uscita coincida con layer(...)
```
