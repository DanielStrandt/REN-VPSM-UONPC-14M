VPSM v5 UO NPC Final / v7-e14
================================

This folder is the frozen standalone inference bundle for the final Ultima Online
ambient villager model.

FINAL MODEL
-----------
model\vpsm_v5_uo_npc_final.pt

This is the epoch-14 snapshot selected for production after the 720-case
comprehensive inference test. Keep this file unchanged.

FROZEN TOKENIZER
----------------
tokenizer\tokenizer.json

The entire original tokenizer directory is copied into tokenizer\. The tokenizer
has exactly 5,000 IDs. Do not retrain, expand, replace, or reorder its vocabulary.

MODEL SOURCE
------------
runtime\uo_vpsm_model_ren_v5_scan.py

This is the exact VPSM v5 architecture source required to instantiate the
checkpoint.

REUSABLE INFERENCE CORE
-----------------------
runtime\vpsm_npc_runtime.py

This contains the frozen tokenizer normalization/encoding behavior, checkpoint
loader, training-compatible prompt serialization, and single-turn generation.
A new GUI or game integration should import this module rather than duplicating
those internals.

IMPORTANT INFERENCE RULE
------------------------
The production model is used as a SINGLE-TURN NPC responder:

    persona + current player utterance -> one NPC response
    reset recurrent state
    persona + next player utterance    -> one NPC response

Do not feed old player/NPC turns back into the model. The visible UI may show a
chat transcript, but prior turns should remain UI-only unless a later model is
explicitly trained and validated for multi-turn use.

IDENTITY PROTECTION
-------------------
Identity protection belongs in the application/inference layer. The next chat
GUI can validate/repair name, profession, and town claims after raw generation.
Do not change the model weights to implement that protection.

VERIFY THE BUNDLE
-----------------
From this directory:

    python verify_bundle.py

The verifier checks every tracked core file by SHA-256, verifies the tokenizer
fingerprint, instantiates the VPSM model from the checkpoint config, and strict-
loads the final state dict on CPU.

DEPENDENCIES
------------
Python 3.10+ recommended.
Required Python packages:

    torch
    tokenizers

The exact versions present when this bundle was assembled are recorded in
runtime_environment.json. For CUDA inference, install a PyTorch CUDA build that
matches the target machine rather than blindly installing a CPU wheel.

FOLDER LAYOUT
-------------
model\
    vpsm_v5_uo_npc_final.pt

tokenizer\
    tokenizer.json
    ... any other original tokenizer files ...

runtime\
    __init__.py
    uo_vpsm_model_ren_v5_scan.py
    vpsm_npc_runtime.py

apps\
    Reserved for the updated chat GUI / game-facing inference app.

provenance\
    Optional copied evaluation/training summaries when available.

reference\
    Optional copy of the prior single-turn GUI for reference only.

bundle_manifest.json
runtime_environment.json
FINAL_MODEL.txt
requirements.txt
verify_bundle.py

WHAT IS INTENTIONALLY NOT INCLUDED
----------------------------------
Training datasets, replay corpora, optimizer state, old checkpoints, cached token
streams, and training scripts are not required to run the final NPC and are not
part of this inference release.
