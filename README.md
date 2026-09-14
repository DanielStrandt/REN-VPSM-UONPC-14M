# REN-VPSM-UONPC-14M

Tiny local language model for short, in-world Ultima Online NPC villager
dialogue.

This repository is the frozen `v7-e14` inference release. It contains the
14-million-parameter VPSM model, its matching 5,000-ID tokenizer, the reusable
single-turn runtime, a small desktop chat application, and release
verification metadata.

## What it does

The model generates the NPC's spoken reply only. The UO server remains the
authority for quests, combat, movement, inventory, economy, permissions, and
game state.

Each request is independent:

```text
NPC persona/state + current player speech -> one NPC reply
```

Do not send the visible chat transcript back into the model. If the game needs
memory, keep structured facts in the server and provide only the relevant facts
for the current turn.

## Release

- Release: `v7-e14`
- Parameters: `13,660,176`
- Vocabulary: `5,000` frozen tokenizer IDs
- Architecture: attention-free VPSM v5 scan recurrent language model
- Default generation: greedy, 18 new tokens
- Intended input: short game chat, roughly 30 model tokens or less
- Intended output: one short reply; the game may split a longer reply into at
  most two speech events at sentence/newline boundaries

The checkpoint is approximately 55 MB and is included at
`model/vpsm_v5_uo_npc_final.pt`.

## Quick start

Requires Python 3.10+ and the packages in `requirements.txt`.

```text
python -m pip install -r requirements.txt
python verify_bundle.py
python vpsm_npc_chat_gui.py
```

The GUI uses bundle-relative paths, fresh recurrent state for every message,
and an application-layer identity guard for name, town, and profession.

For game integration, import `runtime.vpsm_npc_runtime` and reuse one loaded
model instance. Call `generate_single_turn()` once per player utterance.

## Verification

`verify_bundle.py` checks the frozen core files by SHA-256, validates the
tokenizer fingerprint, instantiates the checkpoint configuration, and
strict-loads the model state dictionary on CPU.

The release's included 720-case inference report recorded 95 mechanical flags
(13.2%), including 3 empty responses and 25 generation-cap hits. Those flags
are diagnostics rather than a semantic score; the human-review columns in the
copied report are retained for transparency.

## Repository layout

```text
model/       Frozen checkpoint
tokenizer/   Matching tokenizer and metadata
runtime/     Model architecture and reusable inference core
provenance/  Evaluation summaries and review worksheet
vpsm_npc_chat_gui.py          Production-oriented 18-token GUI
vpsm_npc_chat_gui_longer.py   Experimental adjustable-cap GUI
reference/   Legacy reference implementation; not the production entry point
```

The two GUI files are convenience applications. A live UO server should
enforce its own input and two-output speech limits and split replies without
cutting through words.

## License

Copyright (c) 2026 Daniel Strandt.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

