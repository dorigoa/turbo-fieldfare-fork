# Piano: inferenza di Qwen3.6-35B-A3B in TurboFieldfare (agente locale)

- Versione piano: 2 (decisioni dell'utente U1–U6 confermate il 2026-11-08,
  §10)
- Data analisi: 2026-11-08
- Commit base analizzato: `badac9f` su `main` (contiene tutto il lavoro 8-bit,
  S0–S10)
- Host dell'analisi: MacBook Air M5 (`Mac17,3`), 32 GB, macOS 27.2, Swift
  6.4.2, 202 GiB liberi sul volume dati
- Modelli installati: `/Users/dorigo_a/LLM/TurboFieldfare/gemma4.gturbo`
  (13 GB) e `gemma4-8bit.gturbo` (25 GB); nessun pack immagini
- Destinazione di Qwen: `/Users/dorigo_a/LLM/TurboFieldfare/qwen36.gturbo`
- Esecutore previsto: un agente di coding **sul Mac** che usa
  Qwen3.8-Flash-Next servito da Strata sul PC (contesto massimo 256K token)
- Il lavoro resta nel fork: nessun push e nessuna PR senza richiesta
  esplicita dell'utente

Questo documento serve a un LLM locale che implementa la funzionalità in
**sessioni sequenziali**. Ogni sessione legge AGENTS.md, le sezioni 1–7 di
questo piano e la propria scheda (§8), niente di più. Le formule del modello
sono tutte in §3: nessuna sessione deve cercarle sul web, salvo dove la scheda
lo dice.

---

## 0. Come usare questo piano

### 0.1 Dove gira l'agente

TurboFieldfare si compila e si prova solo su macOS 26+ con Metal. Il PC con
Strata non può compilare né eseguire i test. Quindi:

- **Strata gira sul PC** e resta in ascolto solo su `127.0.0.1` (secondo il
  README la porta di default è 8080: verificala). Non esporlo sulla rete
  locale: non ha autenticazione.
- **L'agente (harness) gira sul Mac**, nella cartella del repository, e
  raggiunge Strata con un tunnel SSH che resta aperto in un terminale
  dedicato:

  ```bash
  ssh -N -L 8080:127.0.0.1:8080 <utente>@<pc>
  ```

  Nell'harness: provider OpenAI-compatible, base URL
  `http://127.0.0.1:8080/v1`, model ID quello che restituisce
  `curl -s http://127.0.0.1:8080/v1/models`.
- L'harness deve avere: shell, lettura e scrittura di file, modifica di file,
  git. Configuralo perché chieda conferma prima di `rm -rf`, `git push`,
  `git reset --hard`, `git clean`, `kill`, `pkill` e download.
- Il modello locale non usa memoria del Mac. Sul Mac può girare un solo
  processo TurboFieldfare (o `mlx_lm`) alla volta, come chiede AGENTS.md.

### 0.2 Ordine, branch e merge

Le sessioni sono **sequenziali** (un solo agente alla volta):

```text
S00 → S01 → S02 → S03 → S04 → S05 → S06 → S07 → S08 → S09 → S10 → S11 → S12 → S13 → S14 (→ S15 facoltativa)
```

- S00 crea il branch di integrazione `feat/qwen36` da `main`.
- Ogni sessione crea `feat/qwen36-sNN-<slug>` da `feat/qwen36` aggiornato,
  lavora lì e alla fine chiede all'utente il merge. L'utente rivede il report
  e fa (o chiede di fare) `git checkout feat/qwen36 && git merge --no-ff
  feat/qwen36-sNN-<slug>`.
- `main` riceve `feat/qwen36` solo dopo S14 e solo su richiesta dell'utente.
- Niente worktree (AGENTS.md).

### 0.3 Controlli dell'utente (gate)

| Gate | Dopo | L'utente controlla |
| --- | --- | --- |
| G0 | S00 | fatti di §2 confermati o corretti; golden di tokenizer e chat |
| G1 | S03 | modello installato e verificato; golden del modello prodotti |
| G2 | S10 | i primi logit reali coincidono con il riferimento |
| G3 | S14 | prova dell'app, benchmark, merge in `main` |

Al gate la sessione si ferma e aspetta. La successiva parte solo dopo l'ok.

### 0.4 Budget di contesto (256K)

Ogni sessione è dimensionata per restare sotto circa 120K token. Regole:

- Non leggere per intero file oltre 400 righe: usa `grep -n` e poi
  `sed -n '<a>,<b>p'` sulle righe che servono. `RealForwardRunner.swift` ha
  quasi 2000 righe: non va mai letto tutto.
- Taglia l'output dei comandi: `swift build ... 2>&1 | tail -n 80`; per i
  test `Scripts/test.sh --filter <Suite> 2>&1 | grep -E "error|failed|passed|Test Suite|✘|✔" | tail -n 80`.
- Scrivi le scoperte nel file di avanzamento (§0.5) invece di tenerle a mente.
  Se l'harness compatta il contesto, rileggi quel file.
- Prima di usare una funzione o un tipo esistente, trova la firma con
  `grep -rn "func <nome>\|struct <nome>\|class <nome>" Sources`. Non inventare
  API.

### 0.5 Avanzamento, interruzioni e ripresa

Ogni sessione tiene `docs/plans/qwen36-reports/SNN.progress.md` e lo aggiorna
**con un commit alla fine di ogni task** della scheda:

```text
- Task 2 completato: <cosa, file toccati>, commit <hash>, test <comando> verdi
- Prossimo: task 3 (<titolo>)
- Note: <fatti scoperti che servono dopo>
```

Per riprendere una sessione interrotta usa il prompt di ripresa (§0.6).
L'agente deve:

1. `git status`, `git branch --show-current`,
   `git log --oneline feat/qwen36..HEAD`;
2. leggere il file di avanzamento e il report parziale, se esiste;
3. controllare i processi con il `pgrep` di §7.1. Se un processo del modello
   è ancora attivo (per esempio un test lanciato prima dell'interruzione),
   **non terminarlo**: aspetta che finisca o chiedi all'utente;
4. ripartire dal primo task non completato, senza rifare i task completati.

Una sessione si ferma e scrive il report con stato "Bloccata" quando:

- lo stesso test fallisce ancora dopo 3 tentativi di correzione;
- un fatto o un contratto del piano risulta sbagliato e la correzione
  cambierebbe altre sessioni;
- serve un download, un permesso o una decisione non previsti dalla scheda.

### 0.6 Prompt di avvio

Prompt generico (ogni scheda in §8 ha il suo già compilato):

```text
Lavori nel repository TurboFieldfare in /Users/dorigo_a/turbo-fieldfare-fork,
sul Mac. Leggi AGENTS.md, poi docs/plans/qwen36-model-family-plan.md: le
sezioni 0-7 e la scheda della sessione <SNN> in §8. L'utente ti autorizza a
modificare e creare solo i file che la scheda indica. Crea il branch
feat/qwen36-<snn>-<slug> da feat/qwen36. Segui le regole comuni (§7). Dopo
ogni task aggiorna docs/plans/qwen36-reports/<SNN>.progress.md e fai un commit.
Alla fine scrivi docs/plans/qwen36-reports/<SNN>.md con il formato di §7.5 e
fermati: il merge lo decide l'utente.
```

Prompt di ripresa:

```text
Riprendi la sessione <SNN> del piano docs/plans/qwen36-model-family-plan.md
nel repository /Users/dorigo_a/turbo-fieldfare-fork. Leggi AGENTS.md, le
sezioni 0-7 del piano, la scheda <SNN> e
docs/plans/qwen36-reports/<SNN>.progress.md. Segui la procedura di §0.5:
controlla branch, commit e processi, poi continua dal primo task non
completato.
```

---

## 1. Obiettivo e perimetro

**Obiettivo.** TurboFieldfare installa ed esegue **Qwen3.6-35B-A3B**
(checkpoint MLX affine 4-bit, solo testo) accanto a Gemma 4. CLI, server e Mac
app scelgono il modello; il percorso Gemma resta identico.

**Definizione di fatto (DoD globale).**

- `TurboFieldfareRepack --family qwen36` installa e verifica `qwen36.gturbo`
  in streaming, come oggi per Gemma.
- Il runtime carica il modello Qwen e produce logit che coincidono con il
  riferimento `mlx-lm` sugli stessi pesi entro le tolleranze di C8.
- Il tokenizer produce gli stessi ID di Hugging Face `tokenizers`; il formato
  chat coincide con `apply_chat_template` del checkpoint.
- CLI e server: chat con ragionamento (`<think>`), tool call nel formato XML
  di Qwen, stop corretti, `/v1/models` con l'ID Qwen.
- Mac app: scelta del modello (Gemma 4 / Qwen3.6), download, posizione,
  ammissione di memoria e cronologia separate per modello.
- Gemma invariato: stesso output greedy token per token prima e dopo.
- Test di pacchetto verdi; benchmark e documentazione (S14).

**Non obiettivi.** Immagini e video con Qwen (il supporto immagini risulta
"not available for this model"); MTP e decodifica speculativa; Qwen 8-bit;
contesto oltre 262.144 token (YaRN); prefill DeltaNet "chunked" (solo S15, se
misurato necessario); KV quantizzata; ottimizzazioni di Gemma; qualunque
modifica ai default di Gemma.

---

## 2. Fatti (2026-11-08)

Ricavati da `config.json`, dall'indice dei pesi e dalle pagine dei repository
Hugging Face. **S00 li riverifica tutti** sugli header reali e annota ogni
differenza nel proprio report.

### 2.1 Checkpoint

| Voce | Valore |
| --- | --- |
| Modello originale | `Qwen/Qwen3.6-35B-A3B` (BF16, Apache 2.0) |
| Sorgente da installare | `mlx-community/Qwen3.6-35B-A3B-4bit` |
| Revisione osservata | `38740b847e4cb78f352aba30aa41c76e08e6eb46` (da confermare con l'API in S00) |
| Shard | 4: 5,29 + 5,37 + 5,37 + 4,38 GB = 20,41 GB |
| Altri file | `tokenizer.json` (20 MB), `tokenizer_config.json`, `chat_template.jinja` (7,76 kB), `vocab.json`, `generation_config.json`, `preprocessor_config.json`, `processor_config.json` |
| Quantizzazione | affine, 4 bit, gruppi da 64, scale e bias BF16 |
| Eccezioni a 8 bit | `mlp.gate` (router) e `mlp.shared_expert_gate` in tutti i 40 layer |
| Non quantizzati (attesi) | norme, `linear_attn.conv1d.weight`, `linear_attn.A_log`, `linear_attn.dt_bias` |
| Torre visiva | presente nel checkpoint; il repack la **salta** |
| MTP | `mtp_num_hidden_layers: 1` nel modello originale; S00 verifica se il checkpoint MLX contiene tensori `mtp.*` (il repack li salta comunque) |

### 2.2 Architettura (testo)

| Campo | Valore |
| --- | --- |
| `model_type` | `qwen3_5_moe` (`Qwen3_5MoeForConditionalGeneration`) |
| Layer | 40: schema ripetuto 3 × linear attention + 1 × full attention |
| Layer full attention (indice da 0) | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 |
| `hidden_size` | 2048 |
| Gated DeltaNet | 16 teste QK × 128, 32 teste V × 128, conv causale con kernel 4 |
| Full attention | 16 teste Q, 2 teste KV, `head_dim` 256, gate d'uscita, `q_norm`/`k_norm` |
| RoPE | `rope_theta` 10.000.000, `partial_rotary_factor` 0,25 (64 dimensioni ruotate), mRoPE interleaved `[11, 11, 10]` |
| MoE | 256 expert, 8 attivi, `moe_intermediate_size` 512, SiLU |
| Shared expert | 1, intermedio 512, gate sigmoide scalare |
| Vocabolario | 248.320 |
| Embedding | **non** legati (`tie_word_embeddings: false`): `embed_tokens` e `lm_head` separati |
| `rms_norm_eps` | 1e-6 |
| `max_position_embeddings` | 262.144 |
| Token | `bos` = `eos` = 248044 (`<|endoftext|>`); stop da `generation_config`: 248046 (`<|im_end|>`) e 248044 |
| Sampling consigliato | `generation_config.json`: temperature 1.0, top-k 20, top-p 0.95 |

### 2.3 Nomi dei tensori nel checkpoint MLX (prefisso `language_model.`)

Visti nell'indice; S00 li conferma sugli header (forme e dtype).

```text
language_model.model.embed_tokens.{weight,scales,biases}
language_model.lm_head.{weight,scales,biases}
language_model.model.norm.weight                                  (da confermare)
language_model.model.layers.N.input_layernorm.weight
language_model.model.layers.N.post_attention_layernorm.weight
# layer linear attention
language_model.model.layers.N.linear_attn.in_proj_qkv.{weight,scales,biases}
language_model.model.layers.N.linear_attn.in_proj_z.{weight,scales,biases}
language_model.model.layers.N.linear_attn.in_proj_a.{weight,scales,biases}
language_model.model.layers.N.linear_attn.in_proj_b.{weight,scales,biases}
language_model.model.layers.N.linear_attn.conv1d.weight
language_model.model.layers.N.linear_attn.A_log
language_model.model.layers.N.linear_attn.dt_bias
language_model.model.layers.N.linear_attn.norm.weight
language_model.model.layers.N.linear_attn.out_proj.{weight,scales,biases}
# layer full attention
language_model.model.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj}.{weight,scales,biases}
language_model.model.layers.N.self_attn.{q_norm,k_norm}.weight
# MoE (tutti i layer)
language_model.model.layers.N.mlp.gate.{weight,scales,biases}                 (8 bit)
language_model.model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}
language_model.model.layers.N.mlp.shared_expert.{gate_proj,up_proj,down_proj}.{weight,scales,biases}
language_model.model.layers.N.mlp.shared_expert_gate.{weight,scales,biases}   (8 bit)
```

Due punti che S00 deve chiarire, perché cambiano le formule:

1. **Norme "zero-centered".** Nel codice Hugging Face le norme dei layer,
   la norma finale, `q_norm` e `k_norm` moltiplicano per `(1 + w)`. La
   conversione MLX potrebbe aver già sommato 1 ai pesi. Verifica empirica:
   leggi `input_layernorm.weight` del layer 0. Media vicina a 1 → pesi già
   spostati, usa RMSNorm standard (`x̂ · w`). Media vicina a 0 → usa
   `x̂ · (1 + w)`. La norma gated di `linear_attn.norm` usa sempre `w`
   (inizializzata a 1).
2. **Layout di `conv1d.weight`.** Hugging Face: `[8192, 1, 4]`. MLX:
   probabilmente `[8192, 4, 1]`. Annota la forma reale e l'ordine dei tap.

### 2.4 Dimensioni derivate (4-bit, gruppi da 64)

| Voce | Valore |
| --- | --- |
| Byte per expert | gate 589.824 + up 589.824 + down 589.824 = **1.769.472 B** (108 pagine da 16 KiB, già allineato) |
| Expert totali | 256 × 40 × 1.769.472 = 18.119.393.280 B (≈ 16,9 GiB) |
| Lettura expert per token a cache fredda | 8 × 40 × 1,77 MB ≈ 566 MB (Gemma 4-bit: 8 × 30 × 3,36 MB ≈ 806 MB) |
| Pesi residenti (stima) | embed ≈ 286 MB, `lm_head` ≈ 286 MB, DeltaNet 30 × 18,9 MB ≈ 568 MB, attention 10 × 15,3 MB ≈ 153 MB, shared expert 40 × 1,77 MB ≈ 71 MB, router ≈ 22 MB: totale ≈ 1,39 GB |
| `.gturbo` atteso | ≈ 19,5 GB |
| KV FP16 (10 layer full) | 10 × 2 × 256 × 2 × 2 B = **20.480 B per token**: 640 MiB a 32K, 2,5 GiB a 128K, 5 GiB a 262.144 |
| Stato DeltaNet (FP32) | 30 × 32 × 128 × 128 × 4 B ≈ 60 MiB, costante |
| Stato conv (FP32) | 30 × 3 × 8192 × 4 B ≈ 2,8 MiB, costante |

Dimensioni dei proiettori: `in_proj_qkv` 2048 → 8192 (q 2048 | k 2048 |
v 4096), `in_proj_z` 2048 → 4096, `in_proj_a` e `in_proj_b` 2048 → 32,
`out_proj` 4096 → 2048, `q_proj` 2048 → 8192 (query e gate, §3.3),
`k_proj` e `v_proj` 2048 → 512, `o_proj` 4096 → 2048.

### 2.5 Formato chat

Dal `chat_template.jinja` del checkpoint (S00 produce i golden esatti):

- Turni: `<|im_start|>ROLE\n … <|im_end|>\n`; ruoli `system`, `user`,
  `assistant`. I risultati dei tool stanno in un turno `user`, ciascuno in
  `<tool_response>\n … \n</tool_response>`.
- Tool: blocco nel system `# Tools … <tools> … </tools>` con un JSON per tool.
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

- Ragionamento: `<think>\n…\n</think>\n\n` prima del contenuto. Il prompt di
  generazione è `<|im_start|>assistant\n<think>\n` con ragionamento attivo,
  `<|im_start|>assistant\n<think>\n\n</think>\n\n` con `enable_thinking=false`.
  Nei turni precedenti all'ultima domanda dell'utente il ragionamento viene
  tolto, salvo `preserve_thinking`.
- Immagini: `<|vision_start|><|image_pad|><|vision_end|>` (fuori perimetro).

### 2.6 Strumenti già disponibili

- Il pacchetto dipende da `swift-transformers` (≥ 1.3.0): `GFTokenizer` usa
  `AutoTokenizer` e il motore Jinja della libreria (`applyChatTemplate` con
  `tools` e `additionalContext`). Lo stesso percorso può caricare un tokenizer
  BPE byte-level come quello di Qwen e rendere il suo template.
- Sul Mac c'è Python 3 con `tokenizers` 0.23.2. `mlx-lm` **non** è installato.
- `TurboFieldfareValidation` contiene riferimenti CPU (RMSNorm, RoPE,
  attention, MoE, dequant INT4/INT8) e le tolleranze (`Tolerance`).

---

## 3. Riferimento matematico

Notazione: `h` stato residuo (2048), `x̂ = x · rsqrt(mean(x²) + 1e-6)`.
Calcoli interni in FP32; attivazioni tra kernel in FP16 come nel resto del
runtime, salvo dove indicato.

### 3.1 Embedding e testa finale

1. `h = dequant(embed_tokens)[token]`. **Nessuna** moltiplicazione per
   `sqrt(hidden)` (a differenza di Gemma).
2. Dopo l'ultimo layer: `y = RMSNorm(h; model.norm)`,
   `logits = dequant(lm_head) · y` (248.320 righe). **Nessun** softcap.

### 3.2 Layer linear attention (Gated DeltaNet)

Dimensioni: `Hk = 16`, `Hv = 32`, `dk = dv = 128`, `conv_dim = 8192`.

```text
x      = RMSNorm(h; input_layernorm)
qkv    = W_qkv · x                    (8192)
z      = W_z   · x                    (4096 = 32 × 128)
b      = W_b   · x                    (32)
a      = W_a   · x                    (32)

# conv causale depthwise sui soli 8192 canali di qkv, kernel 4
# s_c = ultimi 3 input del canale c (stato conv), u_t = qkv del token t
y_c    = silu( Σ_{j=0..3} w_c[j] · u_c[t-3+j] )   # u_c[t-3..t-1] dallo stato
stato_conv ← ultimi 3 input (prima dell'attivazione)

q = y[0:2048]    → 16 × 128
k = y[2048:4096] → 16 × 128
v = y[4096:8192] → 32 × 128

q_i = q_i · rsqrt(Σ q_i² + 1e-6)     (per testa)
k_i = k_i · rsqrt(Σ k_i² + 1e-6)
q   = q / sqrt(128)

# la testa di valore j usa la testa di chiave j / 2 (repeat_interleave × 2)
beta_j  = sigmoid(b_j)
g_j     = -exp(A_log_j) · softplus(a_j + dt_bias_j)
decay_j = exp(g_j)

# stato S_j ∈ R^{128×128} in FP32, S_j[i][c]: riga i = dimensione chiave,
# colonna c = dimensione valore
S_j      = decay_j · S_j
kv_mem_c = Σ_i S_j[i][c] · k_{j/2}[i]
delta_c  = (v_j[c] - kv_mem_c) · beta_j
S_j[i][c] += k_{j/2}[i] · delta_c
o_j[c]   = Σ_i S_j[i][c] · q_{j/2}[i]

# norma gated per testa (128), peso condiviso tra le teste
o_j = RMSNorm(o_j) · w_norm · silu(z_j)
h  += W_out · concat(o_0 … o_31)      (4096 → 2048)
```

Proprietà utile per i kernel: **ogni colonna `c` di ogni `S_j` evolve in modo
indipendente** dato `k`, `q`, `beta`, `decay` e `v_j[c]`. Un SIMD-group può
tenere una colonna (32 lane × 4 righe = 128 righe) nei registri e scorrere i
token in sequenza; `kv_mem_c` e `o_j[c]` sono `simd_sum` sulle 32 lane. Lo
stesso kernel serve decode (`T = 1`) e prefill (`T > 1`, scansione
sequenziale).

Da verificare in S00 contro il riferimento: l'ordine dei tap della conv e la
mappa testa di valore → testa di chiave (`j / 2` con `repeat_interleave`).

### 3.3 Layer full attention

```text
x  = RMSNorm(h; input_layernorm)
qg = W_q · x → 16 × 512; per ogni testa: q = qg[0:256], gate = qg[256:512]
k  = W_k · x → 2 × 256
v  = W_v · x → 2 × 256
q  = RMSNorm(q; q_norm)   (per testa, 256)
k  = RMSNorm(k; k_norm)
RoPE sulle prime 64 dimensioni di q e k, stile "rotate_half":
  coppie (i, i + 32) per i in 0..31, inv_freq_i = 1e7^(-2i/64), posizione p
  (con solo testo le tre componenti di mRoPE coincidono: è RoPE 1D)
attention causale, GQA: la testa q h usa la testa kv h / 8; scala 1/sqrt(256);
  softmax in FP32
o  = attn · sigmoid(gate)          (per testa, elemento per elemento)
h += W_o · concat(o_0 … o_15)       (4096 → 2048)
```

Niente finestra scorrevole e niente K = V (a differenza di Gemma).

### 3.4 Blocco MoE (tutti i layer)

```text
x      = RMSNorm(h; post_attention_layernorm)
p      = softmax(W_gate · x)           (256, FP32; W_gate 8 bit)
top    = 8 indici con p più alto
w_e    = p_e / Σ_{top} p               (sempre rinormalizzati)
routed = Σ_{e∈top} w_e · W_down_e · ( silu(W_gate_e · x) ⊙ (W_up_e · x) )
shared = W_down_s · ( silu(W_gate_s · x) ⊙ (W_up_s · x) )
sg     = sigmoid(W_sg · x)             (scalare; W_sg 1 × 2048, 8 bit)
h     += routed + sg · shared
```

Ordine del layer: `input_layernorm` → mixer (DeltaNet o attention) →
residuo → `post_attention_layernorm` → MoE → residuo. Nessuna norma
"sandwich", nessuno scalare per layer, nessuna scala per expert (tutte
cose di Gemma).

---

## 4. Stato del codice: cosa si riusa e cosa è cablato a Gemma

Da confermare in S00 con `grep`; i numeri di riga possono cambiare.

| Area | Stato | Conseguenza |
| --- | --- | --- |
| `TurboFieldfareFormat` (`GTurboManifestV1`) | `GTurboManifestArchV1` ha campi solo Gemma (`attentionKEqV`, `finalLogitSoftcap`, `slidingWindow`, `fullAttentionLayerMask`) | serve un discriminatore di famiglia e un blocco arch per Qwen (D1) |
| `Infrastructure/ModelIO` | `ArchConfig.gemma4_26B_A4B`, `ManifestReader.validateArch` campo per campo | config Qwen separata e dispatch per famiglia |
| `Tokenization` | `GFTokenizer` verifica il decoder SentencePiece di Gemma; `GemmaDecoding`, `GemmaToolCallParser`, cornice chat Gemma | tokenizer, rendering chat e parser tool di Qwen (D6, D7) |
| `Runtime/Inference/RealForwardRunner.swift` (≈ 2000 righe) | layer Gemma: norme sandwich, `layerScalar`, K = V, `perExpertScale`, softcap, finestra scorrevole | runner Qwen **nuovo** (D2); `RealForwardRunner` non si tocca |
| `Runtime/Generation` | protocolli `LogitProducer`, `ContinuableLogitProducer`, `ChunkedPrefillRunner` | il runner Qwen li implementa; il resto della generazione si riusa |
| Kernel affini INT4/INT8 | GEMV e GEMM riusabili, ma diversi wrapper Swift hanno dimensioni Gemma cablate (2816, 704, 2112, 262144) | wrapper Qwen nuovi; quelli Gemma non si toccano |
| MoE | streaming degli expert, cache a slot, `pread`, layout packed: generici; kernel con GELU-tanh | variante SiLU e router softmax-top8 (D3) |
| `Runtime/KVCache` | anello scorrevole + layer full di Gemma | cache Qwen: KV per 10 layer + stato DeltaNet (D4) |
| Repack | infrastruttura (Hugging Face, range, verifica, transazione) generica; `RepackPlanner` mappa i nomi Gemma | planner Qwen nuovo (S02) |
| `ContextAdmission` | geometria Gemma (anello scorrevole) | formula per famiglia (S01) |
| App, CLI, server | catalogo varianti Gemma (int4/int8), ID modello, identità delle conversazioni | catalogo per famiglia + variante (S12, S13) |
| Vision | solo Gemma | Qwen: "not available for this model" |

---

## 5. Decisioni di design

- **D1 Formato.** Resta `.gturbo` 1.x. Il manifest Qwen ha `modelFamily:
  "qwen3_5_moe"` e un blocco `qwen35Arch` (C2); manca `arch`. Un manifest
  senza `modelFamily` è Gemma e si legge **esattamente** come oggi (stessi
  byte, stesso SHA). Una build vecchia deve rifiutare il manifest Qwen; S01
  verifica con un test che l'errore sia chiaro.
- **D2 Runner separato.** `QwenHybridForwardRunner` in file nuovi, conforme ai
  protocolli esistenti. Il caricamento sceglie il runner in base a
  `modelFamily` in un solo punto. Nessuna modifica a `RealForwardRunner` oltre
  a quel punto, se proprio serve.
- **D3 Kernel nuovi, file nuovi.** Tutto il Metal di Qwen sta in
  `Sources/TurboFieldfare/Metal/Qwen/`, con simboli prefissati `qwen_` e
  function constant 200–249 (C5). I kernel Gemma non si modificano. Un kernel
  Gemma si può **chiamare** se accetta già le dimensioni come parametri (per
  esempio un GEMV INT4 generico); altrimenti si scrive la versione Qwen.
- **D4 Cache.** KV FP16 solo per i 10 layer full attention, senza anello.
  Stato DeltaNet (FP32) e stato conv per i 30 layer linear, in buffer Metal.
- **D5 Riuso del prefisso.** Lo stato DeltaNet non si può "riavvolgere".
  Regole v1:
  - continuare in coda alla posizione corrente funziona sempre;
  - alla fine del prefill di ogni prompt si salva uno **snapshot** di stato
    DeltaNet + conv (≈ 63 MiB), al massimo 2 snapshot;
  - un prompt nuovo che estende esattamente un prompt con snapshot riparte da
    lì: si ripristina lo stato e si tronca la KV a quella posizione;
  - in ogni altro caso: prefill completo.

  Questo copre il ciclo tool del server e la chat a più turni, dove la storia
  fino all'ultimo prompt resta identica.
- **D6 Tokenizer.** Prima scelta: `AutoTokenizer` di `swift-transformers` con
  i file del checkpoint, avvolto in un tipo Qwen che verifica la pipeline
  attesa (BPE byte-level) e fallisce in modo chiuso se cambia. Il tokenizer
  vive in una cartella `tokenizer/` dentro `qwen36.gturbo` (sidecar già
  supportato da `GFTokenizer.tokenizerFolder`), copiata dal repack con
  revisione pinnata. Se i golden di S00 non coincidono e non si può
  correggere senza toccare la dipendenza, S04 si ferma e lo segnala.
- **D7 Chat.** Prima scelta: rendere `chat_template.jinja` con il motore
  Jinja di `swift-transformers`, come fa già il percorso tool di Gemma. Se il
  template usa costrutti non supportati, scrivere un renderer Swift del
  sottoinsieme in §2.5 e verificarlo sugli stessi golden. Parser streaming per
  `<think>…</think>` e per le tool call XML.
- **D8 Oracolo.** Il riferimento è `mlx-lm` (o `mlx-vlm` se `mlx-lm` non
  carica `qwen3_5_moe`) sugli **stessi pesi 4-bit**, in un venv in
  `scratch/qwen36-venv`. Produce logit e stati nascosti di riferimento (S03).
  È un processo modello: mai in parallelo a TurboFieldfare.
- **D9 Catalogo.** Il repack riceve `--family gemma4|qwen36` (default
  `gemma4`, così i comandi esistenti non cambiano) e `--variant` (Qwen: solo
  `int4`). Posizione di default: `qwen36.gturbo` accanto ai modelli Gemma.
  L'ID del modello nel server è il repo sorgente:
  `mlx-community/Qwen3.6-35B-A3B-4bit`.
- **D10 Sampling.** Default per famiglia. Gemma resta 0.2 / 64 / 0.95. Qwen:
  i valori di `generation_config.json` (1.0 / 20 / 0.95), confermati in U4.

---

## 6. Contratti condivisi tra sessioni

### C1 — Famiglia di modello

- In `TurboFieldfareFormat`: `package enum GTurboModelFamily: String {
  case gemma4; case qwen35Moe = "qwen3_5_moe" }`; assente nel manifest =
  `gemma4`.
- In `TurboFieldfareRepackCore`: il catalogo pinna per Qwen repo, revisione,
  SHA dell'indice sorgente, dimensioni attese (da S00/S02), cartella di
  default `qwen36.gturbo`.

### C2 — Manifest Qwen

Chiavi JSON nuove (S01 le fissa nel codec e nei test di compatibilità):

```json
"modelFamily": "qwen3_5_moe",
"qwen35Arch": {
  "hiddenSize": 2048, "numLayers": 40, "vocabSize": 248320,
  "layerTypes": ["linear", "linear", "linear", "full", "..."],
  "linearNumKeyHeads": 16, "linearNumValueHeads": 32,
  "linearKeyHeadDim": 128, "linearValueHeadDim": 128, "linearConvKernel": 4,
  "numHeads": 16, "numKVHeads": 2, "headDim": 256,
  "partialRotaryFactor": 0.25, "ropeTheta": 10000000,
  "numExperts": 256, "topKExperts": 8, "moeIntermediateSize": 512,
  "sharedExpertIntermediateSize": 512,
  "rmsNormEps": 1e-6, "normWeightsShifted": true,
  "tieWordEmbeddings": false, "maxPositionEmbeddings": 262144
}
```

`normWeightsShifted` vale `true` se i pesi delle norme contengono già il `+1`
(§2.3, punto 1). Il valore lo decide S00.

### C3 — Configurazione runtime

`public struct QwenHybridArchConfig: Sendable, Equatable` in
`Sources/TurboFieldfare/Infrastructure/ModelIO/` con gli stessi campi di C2 e
la costante `.qwen36_35B_A3B`. Il caricamento confronta manifest e costante
campo per campo, come per Gemma.

### C4 — Nomi nel `.gturbo`

- I tensori residenti mantengono il nome sorgente (con `language_model.`),
  come fa Gemma.
- Expert: un blob per expert nell'ordine gate, up, down × weights, scales,
  biases (lo stesso ordine del planner Gemma); stride 1.769.472 B.
- Tokenizer: `qwen36.gturbo/tokenizer/{tokenizer.json, tokenizer_config.json,
  chat_template.jinja}`.

### C5 — Metal

- File in `Sources/TurboFieldfare/Metal/Qwen/`: `qwen_norms.metal`,
  `qwen_deltanet.metal`, `qwen_attention.metal`, `qwen_moe.metal`,
  `qwen_head.metal`. S01 li crea come segnaposto (solo `#include
  <metal_stdlib>`, `using namespace metal;` e un commento con la sessione
  proprietaria) e li registra **in coda** a `shaderModules` in
  `MetalContext.swift`, con la directory in `shaderSubdirectories`.
- Ogni simbolo nuovo inizia con `qwen_`: la sorgente condivisa è una sola.
- Function constant: indici 200–249 (S06 200–209, S07 210–219, S08 220–229,
  S09 230–239, S11 240–249). Quelli in uso sono 0–3, 10–13, 20–26, 30–31,
  40–43, 50–53, 60–65, 69–73, 76, 80–86 e 100–139.
- Wrapper Swift in `Sources/TurboFieldfare/Kernels/Qwen/`.
- Riferimenti CPU in
  `Sources/TurboFieldfareValidation/Support/Reference/Qwen/`.

### C6 — Golden

- Piccoli (in git), da S00: `Tests/TurboFieldfare/Core/Tokenization/Qwen/Fixtures/`
  con `tokenizer_golden.json` e `chat_golden.json`. Dichiara la cartella come
  risorsa del target di test in `Package.swift`, come le altre `Fixtures`.
- Grandi (fuori git), da S03: `scratch/qwen36-golden/` (`scratch/` è già in
  `.gitignore`). Contiene, per 6 prompt: ID dei token, top-20 dei logit per
  posizione (teacher forcing), stati nascosti FP16 dopo i layer 0, 3 e 39 per
  il primo prompt, continuazioni greedy di 64 token.

### C7 — Nomi di codice

`QwenHybridForwardRunner`, `QwenHybridCache` (KV + stato DeltaNet),
`QwenTokenizer`, `QwenChatFormat`, `QwenToolCallParser`,
`QwenReasoningSplitter`. Test in `Tests/TurboFieldfare/Core/<area>/Qwen/`.

### C8 — Tolleranze

- Kernel contro riferimento CPU: `Tolerance` esistente
  (`fp16Reduction` 5e-3, `fp16ChainedReduction` 1e-2, `quantInt4` 1,5e-3).
- Stato DeltaNet in FP32: errore relativo ≤ 1e-4 dopo 256 passi.
- Modello contro `mlx-lm` (stessi pesi): coseno dei logit ≥ 0,999 sull'ultima
  posizione; accordo top-1 ≥ 98% delle posizioni in teacher forcing;
  continuazione greedy identica per almeno 4 prompt su 6 (le divergenze
  vanno spiegate come quasi-pareggi, con i due logit in gioco).

---

## 7. Regole comuni

### 7.1 Da AGENTS.md (vincolanti)

- Modifica solo ciò che la scheda autorizza; non cambiare i default del
  runtime; niente ottimizzazioni non richieste.
- Test solo con `Scripts/test.sh` (seriale), eventualmente
  `--filter <Suite>`. Un solo processo app, CLI o test che usa il modello alla
  volta. Non lanciare `swift build` mentre girano i test (il build appare come
  `swiftpm-testing-helper` e falsa i controlli).
- Prima di ogni esecuzione su modello reale verifica macOS 26+, Swift 6.2+,
  spazio disco, `memory_pressure -Q`, `.gturbo` completo e l'assenza di
  processi da:

  ```bash
  pgrep -fl 'TurboFieldfareServer|TurboFieldfareMac|TurboFieldfareDecodeService|TurboFieldfareCLI|TurboFieldfarePackageTests|swiftpm-testing-helper|mlx_lm|mlx-lm'
  ```

  Escludi dal risultato la shell stessa che lancia il comando. Se un controllo
  fallisce: avvisa e fermati. Non terminare processi, non cancellare e non
  reinstallare modelli.
- Niente download di checkpoint completi oltre a quelli approvati (U1), niente
  copie di `.gturbo`, niente svuotamento di cache.
- SwiftUI: `@StoredState` e `PreviewProvider`, mai `@State` diretto né
  `#Preview`. Se tocchi queste superfici esegui
  `ruby Scripts/check_swiftui_macros.rb --self-test` e
  `ruby Scripts/tests/swift_macro_probe_test.rb`.
- Controlli Ruby con `LC_ALL=en_US.UTF-8`.
- Server solo su `127.0.0.1`.
- Immagini fail-closed: con Qwen il supporto immagini è "not available for
  this model"; un'immagine non si accetta mai per poi ignorarla.

### 7.2 Specifiche di questo piano

- Codice, commenti, nomi, messaggi di commit e testi UI in **inglese**, nello
  stile del repository. Report e file di avanzamento in italiano.
- Gemma intoccato. A fine sessione deve essere vuoto:

  ```bash
  git diff feat/qwen36 -- Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift Sources/TurboFieldfare/Metal/Quant Sources/TurboFieldfare/Metal/MoE Sources/TurboFieldfare/Metal/Prefill Sources/TurboFieldfare/Metal/TensorCore Sources/TurboFieldfare/Metal/Sampling Sources/TurboFieldfare/Metal/Attention Sources/TurboFieldfare/Metal/Primitives Sources/TurboFieldfare/Metal/Fusions
  ```

  Fanno eccezione solo le righe che la scheda autorizza esplicitamente.
- API pubblica compatibile: `PublicAPISignatureCompatibilityTests` resta
  verde. Si aggiunge, non si rimuove né si rinomina.
- Nessun peso reale nello heap Swift: fixture piccole sì, pesi no.
- Modelli reali in `/Users/dorigo_a/LLM/TurboFieldfare/`. Per il controllo
  di AGENTS.md usa symlink in `scratch/` (`scratch/qwen36.gturbo` dopo S03).
  Registra nel report come deviazione il fatto che AGENTS.md nomina solo i
  modelli Gemma.

### 7.3 Test

- Ogni kernel nuovo ha prima il riferimento CPU (in `TurboFieldfareValidation`)
  e poi il test GPU contro di esso, con dimensioni piccole **e** con le
  dimensioni reali di §2.2.
- I test che richiedono il modello installato si saltano da soli se
  `scratch/qwen36.gturbo` manca, come fanno quelli di Gemma.
- A fine sessione: le suite toccate con `--filter`, poi l'intera
  `Scripts/test.sh`. Sono noti 7 fallimenti di `MacPresentationTests` dovuti
  al formato numerico della regione svizzera: non contano, ma vanno elencati.

### 7.4 Commit

- Commit piccoli sul branch della sessione, messaggi in inglese, uno per task.
- Aggiungi una riga di attribuzione solo se l'harness la richiede.
- Nessun push, nessuna PR.

### 7.5 Formato del report (`docs/plans/qwen36-reports/SNN.md`)

1. Stato: Completata / Bloccata (con il motivo).
2. Riepilogo (3–5 righe).
3. File creati o modificati.
4. Test eseguiti: comandi esatti ed esito; per i kernel, errori massimi.
5. Deviazioni dal piano, con motivo e impatto sulle sessioni successive.
6. Valori misurati o fissati.
7. Problemi aperti.

---

## 8. Schede delle sessioni

### S00 — Verifica dei fatti e golden di tokenizer e chat

**Obiettivo.** Confermare §2–§4 sui dati reali e produrre i golden piccoli.
Nessun codice Swift.

**Leggi.** AGENTS.md; questo piano §0–§7; Appendice B del piano 8-bit
(`docs/plans/8bit-model-variant-plan.md`, sezione "Appendice B").

**Task.**

1. Crea `feat/qwen36` da `main`, poi `feat/qwen36-s00-facts`.
2. Risolvi la revisione con
   `curl -sL https://huggingface.co/api/models/mlx-community/Qwen3.6-35B-A3B-4bit | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])'`.
3. Leggi gli header degli shard con lo script dell'Appendice B del piano
   8-bit, cambiando l'URL con la revisione pinnata:
   `python3 -I headers.py Qwen3.6-35B-A3B-4bit 4 scratch/qwen36-facts/headers.json`.
   Salva lo script in `scratch/qwen36-facts/`, non nel repository.
4. Con un piccolo script Python (in `scratch/qwen36-facts/`) verifica: nomi
   di §2.3, forme, dtype; tensori quantizzati e non; presenza di `mtp.*` e
   prefissi della torre visiva; layout di `conv1d.weight`; byte per expert
   (§2.4); somme dei byte per categoria.
5. Norme: scarica con una range request solo i byte di
   `language_model.model.layers.0.input_layernorm.weight` e di `q_norm` del
   layer 3, calcola media e deviazione standard, decidi `normWeightsShifted`.
6. Scarica in `scratch/qwen36-facts/tokenizer/` solo `tokenizer.json`,
   `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`
   dalla revisione pinnata. Annota nel report la pipeline di `tokenizer.json`
   (normalizer, pre-tokenizer con la regex, decoder, `added_tokens` con gli
   ID).
7. Crea il venv: `python3 -m venv scratch/qwen36-venv` e installa
   `transformers` (serve solo il tokenizer, non `torch`) e `jinja2` (U1).
8. Golden del tokenizer (Appendice C): almeno 60 testi (italiano, inglese,
   cinese, emoji, codice, spazi e a capo multipli, numeri lunghi, testo con
   token speciali letterali). Golden della chat: almeno 12 conversazioni
   (con e senza system, con tool, con risposte dei tool, ragionamento attivo
   e disattivo, più turni con ragionamento nei turni precedenti). Salva in
   `Tests/TurboFieldfare/Core/Tokenization/Qwen/Fixtures/` (C6).
9. Verifica con `grep` le righe di §4 e annota le differenze.
10. Report. **Gate G0.**

**DoD.** Report con ogni fatto confermato o corretto; i due golden in git;
nessun file Swift modificato.

**Prompt.**

```text
Lavori nel repository TurboFieldfare in /Users/dorigo_a/turbo-fieldfare-fork,
sul Mac. Leggi AGENTS.md, poi docs/plans/qwen36-model-family-plan.md: le
sezioni 0-7 e la scheda S00 in §8. Sei autorizzato a creare i branch
feat/qwen36 e feat/qwen36-s00-facts, i file sotto scratch/qwen36-facts/ e
scratch/qwen36-venv/, i golden in
Tests/TurboFieldfare/Core/Tokenization/Qwen/Fixtures/ e i file in
docs/plans/qwen36-reports/. Puoi scaricare solo gli header degli shard e i
file del tokenizer indicati nella scheda. Non modificare file Swift. Dopo ogni
task aggiorna docs/plans/qwen36-reports/S00.progress.md e fai un commit. Alla
fine scrivi docs/plans/qwen36-reports/S00.md (§7.5) e fermati per il gate G0.
```

### S01 — Contratti: famiglia, manifest, configurazione, segnaposto Metal

**Obiettivo.** Fissare C1–C5 nel codice, senza logica di inferenza.

**Leggi.** `Sources/TurboFieldfareFormat/GTurboManifestV1.swift`,
`GTurboFormatV1.swift`; `Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift`;
`ManifestReader.swift` (con `grep -n`); `Runtime/Configuration/ContextAdmission.swift`;
`Infrastructure/Metal/MetalContext.swift` righe 120–200; i test in
`Tests/TurboFieldfareFormat*` e `Tests/TurboFieldfare/Core/Runtime/Configuration/`.

**Task.**

1. `GTurboModelFamily` e campi `modelFamily` / `qwen35Arch` nel manifest
   (C1, C2). I manifest Gemma esistenti si decodificano e si ricodificano
   identici: aggiungi un test sulle fixture di
   `Tests/TurboFieldfareFormatCompatibility/Fixtures/v1`.
2. `QwenHybridArchConfig.qwen36_35B_A3B` (C3) e validazione del manifest Qwen
   campo per campo; un manifest Qwen con un campo diverso dà
   `archMismatch`.
3. Il caricamento di un manifest Qwen con questa build, finché il runner non
   esiste (S10), fallisce con un messaggio chiaro: "this build recognizes
   qwen3_5_moe models but cannot run them yet".
4. `ContextAdmission`: footprint per famiglia. Per Qwen: KV di §2.4 (solo 10
   layer, niente anello), più stato DeltaNet 64 MiB, più non-KV **provvisorio**
   2,5 GiB e riserva 3 GiB (si misurano in S14). Test con i valori attesi a
   4K, 32K, 128K e 262.144.
5. Segnaposto Metal (C5) registrati in coda; test che la libreria compila.
6. Report.

**DoD.** Test di formato, configurazione e ammissione verdi; Gemma invariato.

**Prompt.** Prompt generico di §0.6 con `<SNN>` = S01, slug `contracts`.

### S02 — Planner del repack per Qwen

**Obiettivo.** `TurboFieldfareRepack --family qwen36 --dry-run` legge solo
header e metadati e stampa il piano; l'installazione vera è in S03.

**Leggi.** `Sources/TurboFieldfareRepack/Core/Planning/RepackPlanner.swift`
(con `grep -n`, poi le funzioni che servono),
`Core/Remote/SupportedModelVariant.swift`, `Core/Format/ArchInfo.swift`,
`Command/main.swift` (gestione di `--variant`), i test del planner in
`Tests/TurboFieldfareRepack/`.

**Task.**

1. Catalogo: voce Qwen (repo, revisione, SHA dell'indice, dimensioni attese
   da S00), `--family gemma4|qwen36` con default `gemma4` (D9). I comandi
   Gemma esistenti producono lo stesso piano di prima (test).
2. Lettura di `config.json` Qwen (`text_config`) e confronto con C2.
3. Mappatura dei tensori (§2.3, C4): residenti, expert packed da
   `switch_mlp`, esclusione di torre visiva e `mtp.*`.
4. Sidecar del tokenizer: i tre file di C4 copiati in `tokenizer/`, con hash
   nel manifest come gli altri file.
5. Test con safetensors sintetici piccoli (2 layer, 4 expert) che coprono i
   due tipi di layer, gli 8 bit di `mlp.gate` e lo skip della visione.
6. `--dry-run` reale (solo header): dimensioni previste nel report.

**DoD.** Dry-run reale coerente con §2.4; test verdi; piani Gemma invariati.

**Prompt.** Generico, S02, slug `repack-planner`.

### S03 — Installazione reale e golden del modello

**Obiettivo.** `qwen36.gturbo` installato e verificato; golden grandi da
`mlx-lm`. **Download approvati da U1.**

**Task.**

1. Preflight di §7.1 e spazio disco (servono circa 45 GB: 19,5 GB di
   `.gturbo` più 20,4 GB di checkpoint MLX nella cache di Hugging Face).
2. `swift run -c release TurboFieldfareRepack --family qwen36 --output /Users/dorigo_a/LLM/TurboFieldfare/qwen36.gturbo`
   (con `--resume` se si interrompe), poi `--verify-install`. Symlink
   `scratch/qwen36.gturbo`.
3. Nel venv installa `mlx-lm` (o `mlx-vlm`, D8), scarica il checkpoint MLX
   alla revisione pinnata. Leggi il file del modello `qwen3_5_moe` (o
   `qwen3_next`) della versione installata per capire come chiamare i layer,
   poi scrivi `scratch/qwen36-golden/make_golden.py` che produce C6.
4. Verifica sul riferimento i due punti aperti di §3.2 (ordine dei tap della
   conv, mappa delle teste) confrontando con un'implementazione NumPy delle
   formule su un layer.
5. Report. **Gate G1.**

**DoD.** Install verificata; golden in `scratch/qwen36-golden/`; §3 confermato
o corretto nel report.

**Prompt.** Generico, S03, slug `install-golden`, più: "Sei autorizzato ai due
download di U1 e all'installazione di mlx-lm nel venv".

### S04 — Tokenizer Qwen

**Obiettivo.** `QwenTokenizer` identico ai golden di S00.

**Leggi.** `Sources/TurboFieldfare/Tokenization/Tokenizer.swift` (righe
1–200 e le funzioni chiamate), `Detokenizer.swift`, i test del tokenizer
Gemma.

**Task.**

1. `QwenTokenizer` che carica il sidecar con `AutoTokenizer` (D6), verifica
   la pipeline (BPE, pre-tokenizer byte-level, decoder ByteLevel) e fallisce
   in modo chiuso se cambia.
2. Token speciali e ID da `added_tokens`; stop di §2.2.
3. Detokenizzazione in streaming: un carattere UTF-8 diviso tra più token non
   deve uscire spezzato.
4. Test: tutti i golden di S00 (ID e testo decodificato); tempo di
   caricamento nel report.
5. Report.

**DoD.** 100% dei golden uguali; nessuna modifica al tokenizer Gemma.

**Prompt.** Generico, S04, slug `tokenizer`.

### S05 — Formato chat, ragionamento e tool call

**Obiettivo.** `QwenChatFormat` identico ai golden della chat; parser del
ragionamento e delle tool call.

**Leggi.** In `Tokenizer.swift` le funzioni di rendering della chat (cerca
`applyChatTemplate`), `StructuredAssistantDecoder.swift`,
`GemmaToolCallParser.swift`, `MultimodalConversation.swift` (con `grep -n`).

**Task.**

1. Un protocollo per il formato chat (se non c'è già) con l'implementazione
   Gemma che avvolge il codice esistente **senza cambiarne l'output**, e
   l'implementazione Qwen (D7).
2. `QwenReasoningSplitter`: separa `<think>…</think>` dal contenuto, anche in
   streaming e anche quando `<think>\n` sta già nel prompt di generazione.
3. `QwenToolCallParser`: parsing in streaming del formato XML di §2.5;
   argomenti non stringa da JSON; errori chiari su XML malformato.
4. Test: golden della chat, casi di streaming spezzato in punti difficili,
   tool call multiple.
5. Report.

**DoD.** Golden uguali; test Gemma del formato chat invariati e verdi.

**Prompt.** Generico, S05, slug `chat-format`.

### S06 — Kernel I: norme, router, embedding e testa

**Leggi.** `Kernels/Primitives/RMSNorm.swift`, `Metal/Primitives/rmsnorm.metal`,
`Kernels/MoE/MoE.swift` (router, con `grep -n`), i wrapper dell'embedding e
della testa (cerca `EmbedLookup` e `LMHeadChainInt4`), i riferimenti CPU
esistenti.

**Task.**

1. RMSNorm secondo `normWeightsShifted`; norma per testa (q/k, 256); norma
   gated per testa (128) con `silu(z)`; L2-norm per testa (128).
2. Router: softmax su 256 + top-8 + rinormalizzazione, con `W_gate` a 8 bit.
3. Embedding INT4 con vocabolario 248.320, senza scala; testa `lm_head` INT4
   non legata, senza softcap, con l'argmax/top-k che usa il sampler.
4. Riferimenti CPU e test (piccoli e reali).
5. Report.

**Prompt.** Generico, S06, slug `kernels-basic`.

### S07 — Kernel II: Gated DeltaNet

**Leggi.** §3.2; i kernel di S06; un kernel Metal esistente con `simd_sum`
come esempio (cerca con `grep -rn simd_sum Sources/TurboFieldfare/Metal`).

**Task.**

1. Conv causale: decode (aggiorna lo stato di 3 input) e prefill su `T` token
   con stato in ingresso e in uscita; SiLU.
2. Gate: `beta` e `decay` per testa.
3. Scansione ricorrente (§3.2): un kernel per `T ≥ 1`, stato FP32 in un buffer
   Metal, una colonna per SIMD-group.
4. Uscita: norma gated per testa e preparazione per `out_proj`.
5. Riferimenti CPU e test: `T = 1`, `T = 7`, `T = 300`; continuità
   (prefill di `T1` poi `T2` passi di decode = prefill di `T1 + T2`); errore
   dello stato dopo 256 passi (C8).
6. Con il modello installato: confronto di un layer 0 completo con i golden
   di S03 (stato nascosto dopo il layer 0, teacher forcing).
7. Report.

**Prompt.** Generico, S07, slug `kernels-deltanet`.

### S08 — Kernel III: full attention con gate

**Leggi.** §3.3; `Kernels/Attention/Attention.swift`,
`PrefillAttention.swift`, `Kernels/Primitives/RoPE.swift` e i loro `.metal`,
solo per capire se accettano `head_dim` 256 con rapporto GQA 8 e senza
finestra.

**Task.**

1. Separazione query/gate da `q_proj` (§3.3), `q_norm`/`k_norm`.
2. RoPE parziale (64 dimensioni, `rotate_half`, theta 1e7).
3. Attention causale GQA 16/2, `head_dim` 256, decode e prefill, sulla KV
   FP16 di D4. Riusa un kernel esistente solo se lo supporta già senza
   modifiche.
4. Gate d'uscita `sigmoid(gate)`.
5. Riferimenti CPU e test; con il modello, confronto del layer 3 con i golden.
6. Report.

**Prompt.** Generico, S08, slug `kernels-attention`.

### S09 — Kernel IV: MoE SiLU e shared expert con gate

**Leggi.** §3.4; `Kernels/MoE/MoE.swift`, `SharedExpertInt4.swift`,
`Metal/MoE/moe.metal` (solo la struttura), il percorso di streaming degli
expert (`Runtime/Inference/ModelExpertIO.swift`, `Infrastructure/Streaming/`).

**Task.**

1. Expert routed INT4 (H 2048, I 512) con SiLU, sui blob di C4 letti dalla
   cache di slot esistente.
2. Shared expert SwiGLU + gate sigmoide scalare (8 bit).
3. Somma pesata e residuo.
4. Variante prefill per `T` token (v1: anche un ciclo per token va bene, se
   corretto).
5. Riferimenti CPU e test.
6. Report.

**Prompt.** Generico, S09, slug `kernels-moe`.

### S10 — Runtime: decode

**Obiettivo.** Primo modello Qwen che genera token.

**Leggi.** `Runtime/Inference/Model.swift` (caricamento, con `grep -n`),
`Runtime/Generation/LogitProducer.swift`, `Runtime/KVCache/KVCacheManager.swift`,
in `RealForwardRunner.swift` **solo** le funzioni di decode per un token
(trovale con `grep -n "func "`).

**Task.**

1. Caricamento del modello Qwen: tensori residenti, layout degli expert,
   tokenizer sidecar.
2. `QwenHybridCache` (D4).
3. `QwenHybridForwardRunner.produce(token:position:into:)`: 40 layer secondo
   `layerTypes`, testa finale. Prefill v1: un token alla volta.
4. Dispatch per famiglia nel punto di caricamento (D2); il messaggio di S01
   sparisce per Qwen.
5. CLI in modalità raw (`--prompt`) con `scratch/qwen36.gturbo`.
6. Confronto con i golden: stati dopo i layer 0, 3, 39 e logit finali
   (teacher forcing); continuazioni greedy (C8).
7. Report. **Gate G2.**

**Prompt.** Generico, S10, slug `runtime-decode`.

### S11 — Runtime: prefill a blocchi, continuazione e riuso del prefisso

**Leggi.** `ChunkedPrefillRunner` in `LogitProducer.swift`,
`Runtime/Prefill/PrefillRuntimeConfig.swift`, il runner di S10.

**Task.**

1. Prefill a blocchi: conv e scansione DeltaNet sul blocco, attention
   prefill, MoE prefill di S09.
2. Equivalenza: prefill a blocchi contro token per token (top-1, coseno),
   come in S9 dell'8-bit.
3. `ContinuableLogitProducer`: continuazione in coda.
4. Snapshot e ripristino (D5) con troncamento della KV; test che un prompt
   esteso riparte dallo snapshot con gli stessi logit del prefill completo.
5. Tempi di prefill nel report (per decidere S15).
6. Report.

**Prompt.** Generico, S11, slug `runtime-prefill`.

### S12 — CLI e server

**Leggi.** `Sources/TurboFieldfareCLI/` (con `grep -n`),
`Sources/TurboFieldfareServer/Core/ServerInference.swift` (con `grep -n`),
`docs/OPENAI_SERVER.md`.

**Task.**

1. Il CLI e il server leggono la famiglia dal manifest e la stampano con la
   variante.
2. Chat con formato Qwen; opzione per il ragionamento (attivo per default,
   U3); stop di §2.2.
3. Server: `/v1/models` con l'ID Qwen; `model_not_found` per gli altri ID;
   tool call XML convertite in `tool_calls` OpenAI e risultati dei tool resi
   nel formato Qwen; ragionamento nel campo che il server usa già per Gemma.
4. Immagini con Qwen: errore chiaro, nessuna accettazione silenziosa.
5. Test del server con `ScriptedLogitProducer` e fixture; una prova reale
   breve (un solo processo).
6. Aggiorna `docs/OPENAI_SERVER.md`.
7. Report.

**Prompt.** Generico, S12, slug `cli-server`.

### S13 — Mac app

**Leggi.** Il report S6 dell'8-bit (`docs/plans/8bit-reports/S6.md`) per
capire come è stata aggiunta la scelta della variante; poi solo i file che
cita.

**Task.**

1. Scelta del modello nei Settings (Gemma 4 / Qwen3.6); per Gemma resta il
   picker della variante 4-bit/8-bit.
2. Descrittore di installazione, posizione `qwen36.gturbo`, download,
   ammissione di memoria con il footprint Qwen.
3. Cronologia: una chat registrata con un altro modello non continua e lo
   dice, come già succede tra varianti.
4. Ragionamento mostrato con la UI che l'app usa già.
5. Sezione Image Support: "not available for this model" con Qwen.
6. Default di sampling per famiglia (D10, U4).
7. Controlli SwiftUI di §7.1 e test; app costruita con `Scripts/build_app.sh`.
8. Report.

**Prompt.** Generico, S13, slug `mac-app`.

### S14 — Validazione, misure, benchmark e documentazione

**Task.**

1. Correttezza finale contro i golden (C8) e regressione di Gemma: output
   greedy 4-bit identico a quello di `main` sui prompt del benchmark.
2. Memoria: picco reale a 4K, 32K e 128K; `ContextAdmission` per Qwen
   passa da provvisorio a misurato.
3. Benchmark secondo `docs/COMMUNITY_BENCHMARKS.md`, Qwen e Gemma 4-bit sullo
   stesso Mac, senza controlli sperimentali.
4. Documentazione: `README.md`, `docs/BENCHMARKS.md`, `docs/SYSTEM_DESIGN.md`,
   `docs/RUNTIME_CONTROLS.md`.
5. Report finale con l'elenco di ciò che l'utente deve provare nell'app.
   **Gate G3.**

**Prompt.** Generico, S14, slug `validation`.

### S15 (facoltativa) — Prefill DeltaNet "chunked"

Solo se S11 e S14 mostrano che il prefill DeltaNet domina il tempo al primo
token. Algoritmo a blocchi da 64 con sistema triangolare (vedi
`chunk_gated_delta_rule` nel codice Hugging Face); stessi test di
equivalenza di S11. Da pianificare a parte, con nuovi fatti.

---

## 9. Rischi e mitigazioni

| Rischio | Mitigazione |
| --- | --- |
| Formule DeltaNet sbagliate (tap della conv, teste, norme) | S00 e S03 le verificano sui pesi reali e su NumPy prima di scrivere kernel; confronto per layer con i golden (S07, S08, S10) |
| `swift-transformers` non riproduce il tokenizer o il template Qwen | golden di S00; fallback documentato in D6 e D7; stop se serve toccare la dipendenza |
| Riuso del prefisso con stato ricorrente | snapshot a fine prompt (D5); in ogni altro caso prefill completo, più lento ma corretto |
| 256 expert per layer: meno colpi nella cache degli slot | misura in S14; nessuna ottimizzazione in questo piano |
| Regressioni di Gemma | file Gemma intoccati (§7.2); confronto greedy in S14 |
| L'agente locale perde il filo in sessioni lunghe | sessioni piccole, file di avanzamento, gate umani, regola dei 3 tentativi |
| Interruzioni a metà di un test reale | procedura di §0.5: non terminare processi, aspettare, riprendere |
| Spazio disco | circa 45 GB in S03; 202 GiB liberi al momento dell'analisi |

---

## 10. Decisioni dell'utente (confermate il 2026-11-08)

L'utente ha confermato tutte le proposte così come sono. Una sessione che
avrebbe bisogno di una scelta diversa si ferma e lo chiede (§0.5).

| # | Domanda | Decisione |
| --- | --- | --- |
| U1 | Download | Approvati: header e tokenizer (S00, pochi MB); `transformers` e `jinja2` nel venv (S00); install di `qwen36.gturbo` (≈ 19,5 GB) e checkpoint MLX per `mlx-lm` (≈ 20,4 GB) più `mlx-lm` nel venv (S03). Nessun altro download |
| U2 | Perimetro | Solo testo e solo 4-bit; immagini, video, MTP e 8-bit esclusi |
| U3 | Ragionamento | Attivo per default come nel template, disattivabile nel CLI, nel server (`enable_thinking`) e nell'app |
| U4 | Sampling Qwen | `generation_config.json`: temperature 1.0, top-k 20, top-p 0.95; Gemma invariato |
| U5 | Branch | Integrazione su `feat/qwen36`; merge delle sessioni deciso dall'utente; `main` solo dopo G3 |
| U6 | Esecuzione | Harness sul Mac, Strata sul PC tramite tunnel SSH (§0.1) |

---

## Appendice A — Comandi utili

```bash
Scripts/test.sh --filter <SuiteName>
Scripts/test.sh
LC_ALL=en_US.UTF-8 ruby Scripts/check_markdown_links.rb
LC_ALL=en_US.UTF-8 ruby Scripts/check_swiftui_macros.rb --self-test
memory_pressure -Q
df -h /Users/dorigo_a/LLM
swift build -c release 2>&1 | tail -n 80
swift run -c release TurboFieldfareRepack --family qwen36 --dry-run --output /Users/dorigo_a/LLM/TurboFieldfare/qwen36.gturbo
swift run -c release TurboFieldfareRepack --family qwen36 --output /Users/dorigo_a/LLM/TurboFieldfare/qwen36.gturbo
swift run -c release TurboFieldfareRepack --verify-install --input-gturbo /Users/dorigo_a/LLM/TurboFieldfare/qwen36.gturbo
ln -s /Users/dorigo_a/LLM/TurboFieldfare/qwen36.gturbo scratch/qwen36.gturbo
swift run -c release TurboFieldfareCLI --model scratch/qwen36.gturbo --prompt "The capital of France is" --max-new 32
```

`--family` esiste solo dopo S02; il CLI su Qwen funziona solo dopo S10.

## Appendice B — Header degli shard

Usa lo script dell'Appendice B di `docs/plans/8bit-model-variant-plan.md`
con due modifiche: sostituisci `resolve/main` con `resolve/<revisione>` e
passa `Qwen3.6-35B-A3B-4bit` con 4 shard.

## Appendice C — Golden di tokenizer e chat

Da eseguire con il Python del venv (`scratch/qwen36-venv/bin/python -I`),
fuori dal repository (`scratch/qwen36-facts/`).

```python
# usage: python -I tok_golden.py <tokenizer_dir> <cases.json> <out.json>
# cases.json: lista di stringhe
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
# cases.json: lista di {"name", "messages", "tools"?, "enable_thinking"?,
#                       "add_generation_prompt"}
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
