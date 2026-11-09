#

Ile mamy danych

- fineweb.bin ma ok. 1,0 GB, czyli ok. 500 mln tokenów (uint16). Po odjęciu 10 mln na walidację na trening zostaje ok. 490 mln tokenów.
- Jeden krok to BATCH_SIZE × BLOCK_SIZE = 64 × 256 = 16 384 tokenów.
- 5000 kroków to ok. 82 mln tokenów, czyli ok. 0,17 epoki. Model nie widzi nawet jednej piątej danych.
- Pełna epoka to ok. 30 000 kroków.
