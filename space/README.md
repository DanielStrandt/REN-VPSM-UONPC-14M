---
title: REN VPSM UO NPC 14M
emoji: 🏰
colorFrom: purple
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# REN VPSM UO NPC 14M

This Space is a Gradio demonstration of the frozen **v7-e14** VPSM v5 model
release for short, in-world Ultima Online NPC villager dialogue.

The UI mirrors the production contract:

- each player message is an independent single-turn query;
- the display-only conversation is never fed back into the model;
- input is intended for roughly 30 model tokens or fewer;
- greedy generation is capped at 18 model tokens;
- a longer reply is shown as at most two speech events.

The UO server remains authoritative for quests, combat, movement, inventory,
economy, permissions, and all game state. The model only produces spoken NPC
text.

The complete release, verification bundle, provenance, and integration notes
are available in the companion [GitHub repository](https://github.com/DanielStrandt/REN-VPSM-UONPC-14M).

Licensed under the Apache License, Version 2.0.
