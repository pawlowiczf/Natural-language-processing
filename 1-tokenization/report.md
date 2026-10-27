# Porównanie tokenizerów: BPE, whitespace i SentencePiece

Dla trzech tokenizerów trenuję ten sam mały model językowy na polskiej Wikipedii i oceniam go na tekście, którego tokenizery nie widziały podczas uczenia.

## 1. Dane

Korpus to polska Wikipedia (SpeakLeash `plwiki`), pierwsze ok. 60 mln znaków. Co 10. dokument trafia do zbioru ewaluacyjnego, reszta do treningowego, więc oba zbiory są rozłączne.

- **Trening:** `plwiki_train.txt`, do uczenia tokenizerów i modeli (modele widzą pierwsze 200 tys. linii).
- **Ewaluacja:** `plwiki_eval.txt`, 21 390 linii, 841 878 słów, 6 193 473 znaków (ok. 6,2 MB). Ten sam zbiór służy też do śledzenia strat podczas treningu, ale nie strojono na nim hiperparametrów.

## 2. Tokenizery

Wszystkie trzy mają słownik N = 50 257 tokenów (tyle co GPT-2), więc macierz embeddingów ma ten sam rozmiar.

**BPE.** Pretrenowany tokenizer GPT-2 z HuggingFace, bez dotrenowania. Działa na bajtach, więc nie gubi informacji, ale jego słownik powstał głównie na angielskim.

```python
bpe = AutoTokenizer.from_pretrained("gpt2")
```

**Whitespace.** Własna implementacja (`custom_tokenizer.py`). Tekst jest zamieniany na małe litery i dzielony na ciągi liter/cyfr oraz pojedyncze znaki interpunkcji. Słownik to `<UNK>` i 50 256 najczęstszych tokenów ze zbioru treningowego. Słowa spoza słownika dostają id `<UNK>`.

```python
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize(self, text):
    if self.lowercase:
        text = text.lower()
    return TOKEN_PATTERN.findall(text)

def encode(self, text):
    unk_id = self.token_to_id[UNK]
    return [self.token_to_id.get(t, unk_id) for t in self.tokenize(text)]
```

**SentencePiece.** Model unigramowy wytrenowany na zbiorze treningowym (próbka 300 tys. linii).

```python
spm.SentencePieceTrainer.train(
    sentence_iterator=train_lines(),
    model_prefix="data/sp",
    vocab_size=50257,
    model_type="unigram",
    input_sentence_size=300_000,
    shuffle_input_sentence=True,
)
```

## 3. Model i trening

Model jest taki sam dla wszystkich tokenizerów, zmienia się tylko tokenizer. To dekoder typu GPT zbudowany z `nn.TransformerEncoder` z maską przyczynową, z embeddingami współdzielonymi z warstwą wyjściową.

| parametr | wartość |
|---|---|
| parametry | 30,0 mln |
| warstwy / głowice / wymiar | 6 / 6 / 384 |
| kontekst | 256 tokenów |
| dropout | 0,1 |
| optymalizator | AdamW, lr 3e-4, weight decay 0,1 |
| harmonogram lr | rozgrzewka 100 kroków, potem cosinus do 10% lr |
| batch | 16 sekwencji (4096 tokenów na krok) |
| kroki | 15 000 (61,4 mln tokenów) |
| ziarno | 42 |

Czasy pochodzą z logów Slurma (`runs/*.out`). Obejmują 61 ewaluacji w trakcie treningu, więc BPE, który ma najdłuższy zbiór ewaluacyjny w tokenach, jest tym trochę obciążony.

| tokenizer | tokeny treningowe | czas treningu | tokenów/s |
|---|---|---|---|
| BPE | 28 488 267 | 2641 s (44 min) | ok. 23 tys. |
| whitespace | 9 488 374 | 1768 s (29 min) | ok. 35 tys. |
| SentencePiece | 11 512 600 | 1890 s (32 min) | ok. 33 tys. |

## 4. Sprzęt

- **Trening:** klaster PLGrid, partycja `plgrid-gpu-v100`, 1 × Tesla V100-SXM2 32 GB, 4 rdzenie CPU, 40 GB RAM, PyTorch 2.14.1 (CUDA 12.6).
- **Ewaluacja i pomiary tokenizerów:** komputer lokalny, Intel Core i5-9400F, 15 GB RAM, GeForce GTX 1660 Ti 6 GB.

## 5. Wyniki

### 5.1 Perplexity

Liczę perplexity na słowo i na znak, nie na token. Tokenizery dzielą ten sam tekst na różną liczbę tokenów, więc strata na token nie jest porównywalna. Całkowitą stratę zbioru (strata na token razy liczba tokenów) dzielę przez liczbę słów albo znaków i biorę `exp`.

```python
nats, _ = evaluate(model, ids, config, device, n_chars)
word_ppl = math.exp(nats * len(ids) / n_words)
char_ppl = math.exp(nats * len(ids) / n_chars)
```

| tokenizer | perplexity na słowo | perplexity na znak |
|---|---|---|
| BPE | 6 672 | 3,31 |
| whitespace | 221 | 2,08 |
| SentencePiece | 1 779 | 2,77 |

Obie kolumny to ta sama strata w różnych jednostkach: `ppl_słowo = ppl_znak ^ 7,36`, bo słowo ma średnio 7,36 znaku (np. 3,31^7,36 ≈ 6 700). Dlatego kolejność jest w obu kolumnach taka sama, a perplexity na słowo jest duża.

### 5.2 OOV (whitespace)

| | OOV | wszystkich | % |
|---|---|---|---|
| wystąpienia słów | 108 060 | 846 769 | 12,76% |
| unikalne słowa | 72 292 | 115 583 | 62,55% |

Częste słowa są w słowniku, więc wśród wystąpień OOV jest niewiele. Rzadkich form (fleksja, nazwiska, liczby) jest za to bardzo dużo i prawie wszystkie są spoza słownika.

### 5.3 Statystyki tokenizacji

Pomiar na całym zbiorze ewaluacyjnym (6,2 MB), czyli tekście spoza zbioru treningowego tokenizerów.

| tokenizer | tokeny/słowo | tokeny/znak | tokeny/s | słowa jako 1 token |
|---|---|---|---|---|
| BPE | 3,71 | 0,505 | ok. 0,9 mln | 5 712 (4,4%) |
| whitespace | 1,23 | 0,168 | ok. 2,2 mln | 54 316 (42,2%) |
| SentencePiece | 1,52 | 0,207 | ok. 2,3 mln | 32 512 (25,3%) |

- „Słowa jako 1 token” to unikalne słowa z eval (ok. 130 tys.) kodowane jednym tokenem, nie `<UNK>`.
- Szybkość mierzę na pierwszym ok. 1 MB tekstu, linia po linii. Zmienia się o kilkadziesiąt procent między uruchomieniami, więc liczy się rząd wielkości. BPE to Rust (HF), SentencePiece to C++, whitespace to czysty Python z regexem.
- Whitespace ma więcej niż 1 token na słowo, bo interpunkcja jest osobnym tokenem.

### 5.4 Przykłady

Trzy teksty po ok. 50 słów: fragment „Pana Tadeusza” (A), definicja fotosyntezy (B) i fragment o Krakowie (C).

| tekst | tokenizer | tokeny/słowo | słowa kodowane bezpośrednio |
|---|---|---|---|
| A (poezja) | BPE | 3,94 | 14% |
| | whitespace | 1,22 | 61% |
| | SentencePiece | 2,00 | 53% |
| B (definicja) | BPE | 3,64 | 21% |
| | whitespace | 1,15 | 94% |
| | SentencePiece | 1,30 | 87% |
| C (artykuł) | BPE | 3,49 | 23% |
| | whitespace | 1,17 | 96% |
| | SentencePiece | 1,30 | 89% |

Fragment tekstu B (`chloroplastach, zawierających`):

```
BPE:           Ġchlor | op | last | ach | , | Ġz | aw | ier | aj | Ä | ħ | cy | ch
whitespace:    <UNK> | , | zawierających
SentencePiece: ▁chloro | plast | ach | , | ▁zawierających
```

Pełne tokenizacje są w `notebook_02_evaluation.ipynb`.

## 6. Dyskusja

**Granularność.** BPE tnie polskie słowa na 3–4 kawałki, często na pojedyncze bajty (`ą`, `ę`, `ł`, `ś` to po 2 tokeny). SentencePiece, nauczony na polskim, trzyma całe słowa albo dzieli je według morfemów (`chloro|plast|ach`). Whitespace daje jeden token na słowo ze słownika.

**OOV.** Whitespace nie ma fallbacku. Rzadkie słowa (`chloroplastach`, `zaciekłe`, liczby) zamieniają się w `<UNK>` i informacja przepada, co widać w wygenerowanym tekście:

```
albert einstein był stanem stałym <UNK> po tym, że pracował w zakładzie <UNK>, a następnie jako <UNK>, z <UNK> <UNK> <UNK>. ...
```

BPE i SentencePiece mogą zawsze rozbić słowo na mniejsze części, kosztem dłuższej sekwencji.

**Perplexity.** Whitespace ma najniższą perplexity (221, 2,08), ale to wynik nieporównywalny z resztą. Model przewiduje tekst bez wielkich liter, a 12,8% słów zastępuje jednym łatwym tokenem `<UNK>`. Uczciwie porównywać można BPE z SentencePiece, bo oba rozróżniają wielkość liter i nie mają `<UNK>`. SentencePiece jest wyraźnie lepszy: 2,77 wobec 3,31 na znak.

**Dlaczego SentencePiece wygrywa z BPE.** Jego słownik pasuje do polskiego, ale część przewagi to budżet treningu. Wszystkie modele dostały tyle samo tokenów (4096 na krok), a tokeny mają różną długość. Na krok model widzi ok. 8,1 tys. znaków (BPE), 19,8 tys. (SentencePiece) i 24,4 tys. (whitespace). Okno 256 tokenów obejmuje też ok. 500 znaków w BPE i ok. 1200 w SentencePiece. Zbiór treningowy modele przeszły inną liczbę razy: BPE ok. 2,2, SentencePiece ok. 5,3, whitespace ok. 6,5.

**Koszt.** BPE ma 2,4–3 razy więcej tokenów na ten sam tekst, więc trenuje się najdłużej (44 minuty wobec 29–32) i tokenizuje najwolniej. SentencePiece jest kompromisem: sekwencje dłuższe niż we whitespace, ale bez `<UNK>`, i najlepsza perplexity wśród tokenizerów bez utraty informacji.

## 7. Ograniczenia

- Model (30 mln parametrów, 15 000 kroków) jest niedotrenowany. Perplexity służy do porównania tokenizerów, nie do oceny jakości dużych modeli.
- Jeden przebieg i jedno ziarno na tokenizer, bez przedziałów ufności.
- Korpus jest sklejony bez znaczników granic dokumentów, tak samo dla wszystkich tokenizerów.
- Perplexity liczę na pełnych blokach po 256 tokenów, więc ostatni niepełny blok jest pomijany. Różnica jest znikoma.
