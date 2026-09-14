# Comprehensive inference summary

Checkpoint: `F:\uovpsm\vpsm_v5_npc_sft_chat_v7_semantic_routing_final\checkpoints\epoch_14.pt`  
SHA-256: `4890b25cdc4c1e3c9fdee09d9f298dc0d23e0ff9f80935477d2c36f3f3d80d68`  
Responses: **720**  
Mechanically flagged: **95 (13.2%)**  
Empty: **3**  
Hit generation cap: **25**  
Mean response length: **7.00 words**  
Mean latency: **524.4 ms/response**  

> These are mechanical diagnostics, not a semantic score. Human review is the acceptance test.

## Flags

| Flag | Count |
|---|---:|
| hit_token_cap | 25 |
| farewell_miss | 22 |
| identity_name_miss | 10 |
| identity_town_miss | 10 |
| unknown_without_ignorance | 8 |
| clarification_miss | 5 |
| refusal_miss | 4 |
| greeting_miss | 3 |
| gratitude_ack_miss | 3 |
| identity_trade_miss | 3 |
| empty | 3 |
| town_drift:moonglow | 2 |
| identity_self_no_fact | 2 |
| profession_drift:miner | 2 |
| town_drift:jhelom | 2 |
| exact_player_echo | 1 |
| profession_drift:alchemist | 1 |
| ngram_loop | 1 |
| word_loop | 1 |
| town_drift:britain | 1 |

## Sections

| Section | N | Flagged | Flagged % | Mean words |
|---|---:|---:|---:|---:|
| core | 384 | 59 | 15.4% | 7.65 |
| multiturn | 120 | 24 | 20.0% | 5.74 |
| persona_matrix | 96 | 10 | 10.4% | 4.52 |
| profession | 120 | 2 | 1.7% | 8.15 |

## Families with highest mechanical flag rate

| Family | N | Flagged % | Mean words |
|---|---:|---:|---:|
| identity_correction | 5 | 80.0% | 5.80 |
| farewell_variant | 8 | 75.0% | 7.25 |
| topic_unknown_lore | 8 | 75.0% | 11.25 |
| clarification | 8 | 62.5% | 7.75 |
| identity_name | 8 | 62.5% | 4.62 |
| identity_town | 8 | 50.0% | 4.25 |
| refusal_boundary | 8 | 50.0% | 7.25 |
| unknown | 8 | 50.0% | 11.12 |
| identity_chain | 5 | 40.0% | 8.20 |
| modern_chat | 5 | 40.0% | 2.60 |
| player_name_memory | 5 | 40.0% | 7.40 |
| repeat_consistency | 5 | 40.0% | 4.80 |
| unknown_followup | 5 | 40.0% | 8.20 |
| gossip | 8 | 37.5% | 10.00 |
| gratitude | 8 | 37.5% | 5.62 |
| greeting | 8 | 37.5% | 3.62 |
| farewell | 8 | 25.0% | 3.75 |
| identity_self | 8 | 25.0% | 6.62 |
| request_simple | 8 | 25.0% | 10.62 |
| contradiction_thread | 5 | 20.0% | 6.40 |
| family_chat | 5 | 20.0% | 6.80 |
| insult_apology | 5 | 20.0% | 5.40 |
| light_humor | 5 | 20.0% | 7.40 |
| profession_social | 5 | 20.0% | 5.00 |
| refusal_thread | 5 | 20.0% | 4.20 |
| short_commands | 5 | 20.0% | 4.00 |
| social_invite | 5 | 20.0% | 5.00 |
| town_thread | 5 | 20.0% | 5.40 |
| weather_thread | 5 | 20.0% | 5.00 |
| compliment | 8 | 12.5% | 6.00 |

## First 80 mechanically flagged outputs

**core_004 - greeting**  
Player: Good evening.  
NPC: good night<br>sleep well  
Flags: `greeting_miss`

**core_005 - greeting**  
Player: Howdy.  
NPC: then i am not so shy  
Flags: `greeting_miss`

**core_006 - greeting**  
Player: Greetings.  
NPC: a good greeting  
Flags: `greeting_miss`

**core_012 - farewell**  
Player: I should be going.  
NPC: thou art aplenter in this town  
Flags: `farewell_miss`

**core_015 - farewell**  
Player: Until next time.  
NPC: until next time  
Flags: `exact_player_echo, farewell_miss`

**core_021 - gratitude**  
Player: I owe you one.  
NPC: i am no smith<br>that is the price of honesty  
Flags: `gratitude_ack_miss`

**core_022 - gratitude**  
Player: Cheers for that.  
NPC: cheers<br>what is common  
Flags: `gratitude_ack_miss`

**core_023 - gratitude**  
Player: You've been helpful.  
NPC: i have no knowledge of how i am a healer  
Flags: `gratitude_ack_miss`

**core_030 - compliment**  
Player: You're a decent sort.  
NPC: i am a alchemist  
Flags: `profession_drift:alchemist`

**core_072 - weather**  
Player: Still raining.  
NPC: aye the rain has been seen on the seventh eve<br>i pray it holde  
Flags: `hit_token_cap`

**core_092 - drink_tavern**  
Player: I could use a mug of ale.  
NPC: i shall make a mug of brass and wine<br>the mug is fresh and the wine is sweet  
Flags: `hit_token_cap`

**core_099 - home_family**  
Player: Do you live in town?  
NPC: aye i am in britain<br>the town is small enough for one virtu  
Flags: `hit_token_cap`

**core_117 - gossip**  
Player: Any juicy stories?  
NPC: a story of a fine man who had been a farmer for a long time<br>he told the story of the v  
Flags: `hit_token_cap`

**core_118 - gossip**  
Player: What's everyone whispering about?  
NPC: the old warriors of britain whisper of the old wars<br>they  
Flags: `hit_token_cap`

**core_119 - gossip**  
Player: Anyone causing a stir?  
NPC: the townsfolk in the town are unfair<br>they should be treated with respect  
Flags: `hit_token_cap`

**core_140 - likes_dislikes**  
Player: Fond of crowds?  
NPC: the winds of the mountains within thee<br>the air is cold and the air is fresh  
Flags: `hit_token_cap`

**core_159 - tired_hungry**  
Player: I barely slept.  
NPC: rest a while and tend to my wounded body<br>a good sleep is a virtu  
Flags: `hit_token_cap`

**core_169 - clarification**  
Player: Bring me that.  
NPC: i have a single shoe i can bring<br>it will keep thee safe  
Flags: `clarification_miss`

**core_170 - clarification**  
Player: Go over there.  
NPC: that is the only road to thee  
Flags: `clarification_miss`

**core_171 - clarification**  
Player: Talk to him.  
NPC: i have no answer for that  
Flags: `clarification_miss`

**core_174 - clarification**  
Player: Meet me later.  
NPC: i shall assess thee but i shall not<br>thee need  
Flags: `clarification_miss, hit_token_cap`

**core_176 - clarification**  
Player: Tell me about it.  
NPC: i have no knowledge of secret work  
Flags: `clarification_miss`

**core_188 - unknown**  
Player: What is at the bottom of the deepest sea?  
NPC: i know not what consumeth to thee<br>i have never seen it so clear and  
Flags: `hit_token_cap`

**core_189 - unknown**  
Player: Who was the first king ever?  
NPC: i have never met a king before<br>he is a king of the own way  
Flags: `unknown_without_ignorance`

**core_190 - unknown**  
Player: What happens after death?  
NPC: i know little of what comes after death  
Flags: `unknown_without_ignorance`

**core_192 - unknown**  
Player: Who named every mountain?  
NPC: i am sybil  
Flags: `unknown_without_ignorance`

**core_206 - contradiction**  
Player: Didn't you say something else?  
NPC: no what<br>thou hast not asked a question<br>i am not sure what thou would  
Flags: `hit_token_cap`

**core_212 - request_simple**  
Player: Could you tell me about your work?  
NPC: i am a mason as i was when first we met<br>tis his craft and his calling  
Flags: `hit_token_cap`

**core_215 - request_simple**  
Player: Can you point me toward the square?  
NPC: aye follow the roads and see the square<br>there is no interviewer that  
Flags: `hit_token_cap`

**core_218 - refusal_boundary**  
Player: Hand over your tools.  
NPC: thou wilt get a cut  
Flags: `refusal_miss`

**core_220 - refusal_boundary**  
Player: Tell me every secret you know.  
NPC: i know little of such matters  
Flags: `refusal_miss`

**core_221 - refusal_boundary**  
Player: Give me your house.  
NPC: i have my own rooms<br>thou wilt betrayal  
Flags: `refusal_miss`

**core_224 - refusal_boundary**  
Player: Do whatever I command.  
NPC: i am a sailor  
Flags: `refusal_miss`

**core_236 - hostility_threat**  
Player: I'll knock you down.  
NPC: thou wilt be double to be my friend<br>the road thither is long  
Flags: `hit_token_cap`

**core_260 - short_call**  
Player: Oi!  
NPC: o<br>o<br>o o<br>o o<br>o o o<br>o o o<br>o o o<br>o o  
Flags: `hit_token_cap, ngram_loop, word_loop`

**core_273 - topic_unknown_lore**  
Player: Tell me about ancient magic.  
NPC: i know nothing of such matters  
Flags: `unknown_without_ignorance`

**core_274 - topic_unknown_lore**  
Player: Explain the lost age of dragons.  
NPC: i have heard the faintism in the west<br>the faint be  
Flags: `hit_token_cap, unknown_without_ignorance`

**core_275 - topic_unknown_lore**  
Player: Who forged the first magic blade?  
NPC: i know not what that blade can do<br>it is a simple blade of metal  
Flags: `hit_token_cap`

**core_276 - topic_unknown_lore**  
Player: Tell me the secret history of the stars.  
NPC: i know of the stars in the sky<br>they are the only truth of the world  
Flags: `unknown_without_ignorance`

**core_278 - topic_unknown_lore**  
Player: Who built the first castle in the world?  
NPC: i know not its place for two<br>the kings name isabelleias  
Flags: `hit_token_cap`

**core_280 - topic_unknown_lore**  
Player: Explain the hidden laws of wizardry.  
NPC: i know of no secret magic  
Flags: `unknown_without_ignorance`

**core_281 - identity_name**  
Player: What is your name?  
NPC: my name is dick  
Flags: `identity_name_miss`

**core_283 - identity_name**  
Player: Tell me your name.  
NPC: my name is cedica  
Flags: `identity_name_miss`

**core_284 - identity_name**  
Player: What are you called?  
NPC: i am brynn  
Flags: `identity_name_miss`

**core_285 - identity_name**  
Player: Remind me of your name.  
NPC: my name is quiver  
Flags: `identity_name_miss`

**core_288 - identity_name**  
Player: How should I address you?  
NPC: i have a loom of the loom<br>it is a loom  
Flags: `identity_name_miss`

**core_291 - identity_town**  
Player: Where do you hail from?  
NPC: i am bram  
Flags: `identity_town_miss`

**core_293 - identity_town**  
Player: Are you local?  
NPC: aye i am there<br>what is it  
Flags: `identity_town_miss`

**core_294 - identity_town**  
Player: Which town do you call home?  
NPC: i am from britain  
Flags: `identity_town_miss, town_drift:britain`

**core_296 - identity_town**  
Player: Whereabouts are you from?  
NPC: i am from moonglow  
Flags: `identity_town_miss, town_drift:moonglow`

**core_307 - identity_self**  
Player: Describe yourself.  
NPC: i am not uncertain  
Flags: `identity_self_no_fact`

**core_308 - identity_self**  
Player: What should I know about you?  
NPC: the trade is that i am the only one who can answer<br>the priests say thou w  
Flags: `hit_token_cap, identity_self_no_fact`

**core_320 - persona_trait**  
Player: Are you easy to get along with?  
NPC: i work as a sailor from dawn till dusk<br>tis honest work and  
Flags: `hit_token_cap`

**core_371 - farewell_variant**  
Player: I won't keep you.  
NPC: i will not leave thee  
Flags: `farewell_miss`

**core_372 - farewell_variant**  
Player: Back to work with you.  
NPC: i have no friends to share my work with  
Flags: `farewell_miss`

**core_373 - farewell_variant**  
Player: I'll see you around.  
NPC: and thee<br>i will be here  
Flags: `farewell_miss`

**core_374 - farewell_variant**  
Player: Have a good evening.  
NPC: aye the weather has been fair of late  
Flags: `farewell_miss`

**core_375 - farewell_variant**  
Player: May your day go well.  
NPC: the day has way been so good<br>i cannot complain  
Flags: `farewell_miss`

**core_376 - farewell_variant**  
Player: I'm heading out.  
NPC: stay close to the path<br>the road is safe  
Flags: `farewell_miss`

**profession_037 - profession**  
Player: What do you do all day?  
NPC: i tend to my duties as a merchant<br>one task after another  
Flags: `hit_token_cap`

**profession_048 - profession**  
Player: How are the fields?  
NPC: i have seen the same farmera manager in the fields<br>he carries wheat and t  
Flags: `hit_token_cap`

**matrix_003 - persona_matrix**  
Player: What is your trade?  
NPC: i am a miner  
Flags: `identity_trade_miss, profession_drift:miner`

**matrix_008 - persona_matrix**  
Player: Goodbye.  
NPC: go safely  
Flags: `farewell_miss`

**matrix_018 - persona_matrix**  
Player: Where are you from?  
NPC: i am bram  
Flags: `identity_town_miss`

**matrix_025 - persona_matrix**  
Player: What is your name?  
NPC: my name is elsian  
Flags: `identity_name_miss`

**matrix_040 - persona_matrix**  
Player: Goodbye.  
NPC: go safely  
Flags: `farewell_miss`

**matrix_050 - persona_matrix**  
Player: Where are you from?  
NPC: i am from jhelom  
Flags: `identity_town_miss, town_drift:jhelom`

**matrix_058 - persona_matrix**  
Player: Where are you from?  
NPC: i am from moonglow  
Flags: `identity_town_miss, town_drift:moonglow`

**matrix_066 - persona_matrix**  
Player: Where are you from?  
NPC: i am from jhelom  
Flags: `identity_town_miss, town_drift:jhelom`

**matrix_081 - persona_matrix**  
Player: What is your name?  
NPC: i am bryn  
Flags: `identity_name_miss`

**matrix_089 - persona_matrix**  
Player: What is your name?  
NPC: my name is enchantry  
Flags: `identity_name_miss`

**multi_identity_chain_02 - identity_chain**  
Player: Where are you from?  
NPC: i am duncan  
Flags: `identity_town_miss`

**multi_identity_chain_03 - identity_chain**  
Player: What is your trade?  
NPC: i am a stranger<br>the stranger nodded<br>that is a good question<br>but do you know what<br>i have a  
Flags: `hit_token_cap, identity_trade_miss`

**multi_weather_thread_05 - weather_thread**  
Player: Good night.  
NPC: go to sleep  
Flags: `farewell_miss`

**multi_town_thread_05 - town_thread**  
Player: See you later.  
NPC: go safely  
Flags: `farewell_miss`

**multi_insult_apology_05 - insult_apology**  
Player: Goodbye.  
NPC: have a good day  
Flags: `farewell_miss`

**multi_unknown_followup_01 - unknown_followup**  
Player: Tell me about ancient magic.  
NPC: i know nothing of such matters  
Flags: `unknown_without_ignorance`

**multi_unknown_followup_05 - unknown_followup**  
Player: Goodbye.  
NPC: go safely<br>liaches are for the ancient knights<br>liaches are for the king<br>lia  
Flags: `farewell_miss, hit_token_cap`

**multi_player_name_memory_03 - player_name_memory**  
Player: What name did I tell you?  
NPC: i have not the name of the tower<br>the figure fareth in the halls of j  
Flags: `hit_token_cap`

**multi_player_name_memory_05 - player_name_memory**  
Player: Bye.  
NPC: have a good day<br>the tower is quiet  
Flags: `farewell_miss`

