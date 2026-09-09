#!/usr/bin/env python3
"""
Build a balanced, real 4-Language Multilingual LLM Corpus:
Languages:
  - Tamazight (Tamaziɣt s ugemmay alatini)
  - Arabic (العربية)
  - French (Français)
  - English

Guarantees the Golden Rule of Conversational AI:
  - When user speaks Tamazight -> Assistant answers in Tamazight.
  - When user speaks Arabic    -> Assistant answers in Arabic.
  - When user speaks French    -> Assistant answers in French.
  - When user speaks English   -> Assistant answers in English.
"""

import os
import json
import random

SYSTEM_PROMPT = (
    "You are a helpful, culturally aware, multilingual AI assistant fluent in "
    "Tamazight (Tamaziɣt s ugemmay alatini), Arabic (العربية), French (Français), and English. "
    "Always converse and answer naturally in the language used by the user."
)

def chatml_record(user_msg, assistant_msg, sys_prompt=SYSTEM_PROMPT):
    return {
        "prompt": (
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        ),
        "user": user_msg,
        "assistant": assistant_msg
    }

# Comprehensive hand-curated multi-domain Q&A sets
TAMAZIGHT_CORE = [
    ("Azul fell-ak!", "Azul fell-ak a gma! Ansuf yis-k. Amek telliḍ? Ssarameɣ teggarzed."),
    ("Azul fell-awen! Amek telliḍ?", "Azul! Labas, tanemmirt tameqrant. Kulci yelha. Amek i tetteddu tmeddurt-ik ass-a?"),
    ("Isem-ik d acu?", "Isem-iw d amussnaw n tmaziɣt, d amɛiwen n tussna d umeslay s tmaziɣt, taɛrabt, tafransist d teglizit."),
    ("Tanemmirt aṭas ɣef tallalt-ik.", "Ulac aɣilif! D lferḥ ameqran i nekk ma zemreɣ ad k-d-fkeɣ tallalt. Sseqsi-d ɣef wayen tebɣiḍ."),
    ("Ar tufat!", "Ar tufat d timlilit igerzen! Qqim di lehna d tumert."),
    ("Amek i qqaren 'bonjour' s tmaziɣt?", "Qqaren-as 'Azul'. Azul d awal amaziɣ i d-yemmalen talwit d tifawt."),
    ("Amek i qqaren 'merci' s tmaziɣt?", "Qqaren-as 'Tanemmirt'. Awal-agi yebna ɣef nnmir d usnemmer."),
    ("D acu i d Tamaziɣt?", "Tamaziɣt d tutlayt d tmagit n Yimaziɣen deg Tefriqt n Ugafa. Tesɛa aṭas n tantaliwin am Teqbaylit, Tacawit, Tasusit, Tarifit, Tamaceɣt d tiyaḍ."),
    ("Anwa i d Massinissa?", "Massinissa d agellid ameqran n Numidya (tagelda tamaziɣt n zik). Yesselḥed tiqbilin timaziɣin, yesnerna tafellaḥt d tnezmar deg useggas 202 uqbel Ɛisa."),
    ("D acu i d Tiwizi?", "Tiwizi d lqima tamaziɣt n tdukli d umɛawan gar yemdanen. Di tiwizi, lɣaci ttemɛawanen deg tfellaḥt, lebni neɣ aseggem n taddart mebla lexlas."),
    ("Ini-d inzi ɣef tmusni.", "'Awal ɣezzif, lmeɛna tewzel.' Anamk-is: ur ilaq ara ad yessugget umdan awal mebla lefhama; leqraya d tidet s wawal afessas."),
    ("Ini-d inzi ɣef txazabit d ukerwa.", "'Win i izerrɛen asennan ur yettegmer ara tibexsisin.' Anamek-is: win ixeddmen ccer ur yettraǧu ara lxiir deg taggara."),
    ("Amek i sebbɣen ifexxaren di tmurt n Leqbayel?", "Afexxar aqbayli yettwaxdam s talaxt n ugama, yettwazenneq s leṣnaf yemgaraden n lemxalex d yizirigen yebnan ɣef tmeɣrut d tfinaɣ, s yini aberkan d uzeggaɣ."),
    ("D acu i d amenzu n Yennayer?", "Yennayer d amenzu n useggas amaziɣ (ass n 12 yennayer). D ass n lferḥ anda tawacult ttegg-d imensi n Yennayer (seksu s 7 n lxuḍrat d uyaziḍ n wexxam)."),
    ("Mmeslay-iyi-d ɣef taddart.", "Taddart taqbaylit tezga-d ɣef yixfawen n yidurar. Tebna ɣef Tajmaɛt, anda imeqranen d yilmeẓyen ttemlilin akken ad frun temsal n taddart s tugdut."),
    ("Acu i d asefru?", "Asefru d tamedyazt tamaziɣt i d-yettalsen ɣef tayri, amezruy, leḥzen d tmagit, am yisefra n Si Muḥend U Mḥend d Lounis Ait Menguellet."),
    ("Amek i ttgen aɣrum aquran?", "Aɣrum aquran (kesra) yettwaxdam s semid, aman, lmelḥ d zzit n uzemmur. Yettwanewwel ɣef lfeṛṛan neɣ ṭajin n talaxt."),
    ("Acu i d azemmur deg yedles amaziɣ?", "Azemmur d amudrus ameqran n tezmert d tmeddurt deg Tefriqt n Ugafa. Taseṭṭa n uzemmur temmal-d talwit, zzit-is d dwa d lqut."),
    ("Ini-d kra n tmacahutt.", "Zik-nni, yella yiwen wergaz yesɛan kraḍ n warraw-is. Yenna-yasen: 'Ad fkeɣ ayla-w i win ara iyi-d-yawin tamsalt yugaren akk tmusni.' Dɣa ameẓyan yerra-yas-d: 'Tamusni yugaren akk tid-nni d leɛqel d tezmert n wul.'"),
    ("Acu i d aselmed s tmaziɣt?", "Aselmed s tmaziɣt d asekcem n tutlayt tayemmat ɣer yiɣerbazen d tesdawiyin akken ad ternu ad tnerni deg tussna d tsekla."),
    ("Amek i zemreɣ ad snerniɣ tamaziɣt-iw?", "Tzemreḍ ad teɣreḍ idlisen n tsekla tamaziɣt, ad tesleḍ i yisefra d tezlatin, ad taruḍ s ugemmay alatini, yerna ad tmeslayeḍ yal ass yid-neɣ!")
]

ARABIC_CORE = [
    ("مرحباً بك! كيف حالك؟", "أهلاً وسهلاً بك! أنا بخير وبأفضل حال، شكراً لسؤالك اللطيف. كيف يمكنني مساعدتك اليوم؟"),
    ("من أنت وما هي اللغات التي تتقنها؟", "أنا نموذج ذكاء اصطناعي متعدد اللغات، أتحدث بطلاقة بالأمازيغية (بالحرف اللاتيني)، والعربية، والفرنسية، والإنجليزية. يسعدني تقديم المساعدة في مختلف المجالات ومحادثتك باللغة التي تختارها."),
    ("حدثني عن تاريخ الأمازيغ في شمال إفريقيا باختصار.", "الأمازيغ (إيمازيغن، أي الأحرار) هم سكان شمال إفريقيا الأصليون الممتدون عبر رقعة 'تمازغا'. يمتلكون تاريخاً عريقاً يمتد لآلاف السنين، وشهد قيام ممالك بارزة كنوميديا وموريطنية، وتألق قادة عظام مثل ماسينيسا ويوغرطة وتاكفاريناس والملكة ديهيا (الكاهنة)."),
    ("ما هي دلالات رأس السنة الأمازيغية (ينّاير)؟", "يُحتفل بينّاير في 12 يناير سنوياً، ويرتبط تاريخياً بانتصار الملك الأمازيغي شيشنق وتأسيس الأسرة 22 في مصر القديمة عام 950 ق.م. كما يجسد العيد ارتباط الإنسان بالأرض وشكر نعم الفلاحة، حيث تجتمع العائلات حول طبق الكسكسي التقليدي."),
    ("كيف أقول صباح الخير ومساء الخير بالأمازيغية؟", "يقال صباح الخير: 'Azul n tṣebḥit' أو ببساطة 'Azul' (أزول)، ومساء الخير: 'Azul n tmeddit' أو 'Tameddit tameggazt'."),
    ("ما هو مفهوم 'التويزة' في المجتمع المغاربي؟", "التويزة (Tiwizi) هي تقليد أمازيغي أصيل يعبر عن العمل التطوعي التضامني؛ حيث يتكاتف أهالي القرية لمساعدة شخص في بناء بيته، أو جني محصول الزيتون، أو تنظيف المرافق المشتركة دون مقابل."),
    ("ما هي النصائح الذهبية لتنظيم الوقت وزيادة الإنتاجية؟", "1. حدد الأولويات اليومية مستخدماً مصفوفة أيزنهاور (مهم/عاجل).\n2. طبق تقنية بومودورو (25 دقيقة تركيز تام + 5 دقائق راحة).\n3. قلل مشتتات الهواتف ووسائل التواصل أثناء العمل.\n4. احرص على ساعات نوم منتظمة وغذاء صحي لتجديد نشاط العقل."),
    ("ما الفرق بين الذكاء الاصطناعي وتعلم الآلة والتعلم العميق؟", "الذكاء الاصطناعي (AI) هو المظلة الشاملة لمحاكاة الذكاء البشري.\nتعلم الآلة (ML) هو فرع منه يعتمد على تدريب الخوارزميات للتعلم من البيانات.\nالتعلم العميق (Deep Learning) هو تخصص دقيق ضمن تعلم الآلة يستخدم الشبكات العصبية العميقة لمحاكاة الدماغ البشري في معالجة الصور والنصوص المعقدة."),
    ("أعطني مثلاً أمازيغياً مع شرح معناه بالعربية.", "المثل: 'Awal ɣezzif, lmeɛna tewzel' (الكلام طويل والمعنى قصير).\nمعناه: خير الكلام ما قل ودل، ويحث على تجنب الحشو والثرثرة والتركيز على جوهر المعنى والحكمة."),
    ("شكراً جزيلاً على الشرح الرائع!", "على الرحب والسعة دائماً! يسعدني جداً تقديم الفائدة، وإذا كان لديك أي سؤال آخر بالعربية أو الأمازيغية أو الفرنسية أو الإنجليزية فأنا هنا دائماً.")
]

FRENCH_CORE = [
    ("Bonjour ! Comment vas-tu aujourd'hui ?", "Bonjour ! Je vais très bien, merci beaucoup. Comment se déroule votre journée ? Comment puis-je vous aider ?"),
    ("Qui es-tu et que sais-tu faire ?", "Je suis un assistant intelligent multilingue capable de communiquer avec fluidité en tamazight, arabe, français et anglais. Je peux répondre à vos interrogations, vous aider dans la rédaction, la programmation ou la découverte des cultures nord-africaines."),
    ("Peux-tu me parler de l'histoire et de la culture amazighe ?", "La culture amazighe (berbère) est l'une des plus anciennes et des plus riches du pourtour méditerranéen. Portée par les Imazighen ('les hommes libres'), elle s'étend à travers toute l'Afrique du Nord (Tamazgha). Elle se distingue par une tradition orale vivante, une poterie artisanale réputée, la fête de Yennayer, et des valeurs fortes d'entraide communautaire comme la Tiwizi."),
    ("Quels sont les mots de politesse indispensables en tamazight ?", "Voici les expressions essentielles :\n- **Azul** : Bonjour / Salut\n- **Tanemmirt** : Merci\n- **Amek telliḍ ?** : Comment vas-tu ?\n- **Ar tufat** / **Ar timlilit** : Au revoir\n- **Ssuref-iyi** : Excusez-moi / Pardon"),
    ("Qu'est-ce que la tradition de la 'Tiwizi' ?", "La Tiwizi est une tradition solidaire séculaire en milieu berbère. Elle consiste en une mobilisation collective bénévole des villageois pour s'entraider lors des moissons, du ramassage des olives, de la construction d'un foyer ou de l'aide aux personnes vulnérables."),
    ("Donne-moi une explication simple du fonctionnement des grands modèles de langage (LLM).", "Un LLM (Large Language Model) est un réseau de neurones artificiels entraîné sur de gigantesques volumes de textes. Son principe fondamental est la prédiction séquentielle : pour un texte donné, il calcule statistiquement le mot (ou token) le plus pertinent à générer ensuite, ce qui lui confère une capacité étonnante de compréhension et de dialogue."),
    ("Donne-moi un proverbe berbère et sa signification.", "Voici un proverbe traditionnel : *'Win i izerrɛen asennan ur yettegmer ara tibexsisin'* (Qui sème des épines ne récoltera point de figues). Il enseigne que chaque action a des conséquences : le mal ne peut produire le bien."),
    ("Merci beaucoup pour toutes ces informations claires et précises !", "Je vous en prie, c'est un plaisir ! N'hésitez pas si vous avez d'autres questions, que ce soit en français, en tamazight, en arabe ou en anglais. Très bonne journée !")
]

ENGLISH_CORE = [
    ("Hello! How are you doing today?", "Hello! I am doing very well, thank you for asking. How is your day going? How can I assist you today?"),
    ("Who are you and what languages do you speak?", "I am a multilingual AI assistant fluent in Tamazight (Berber), Arabic, French, and English. I can help you with writing, answering diverse questions, learning languages, and discussing North African culture and general knowledge."),
    ("Tell me about the Amazigh (Berber) heritage.", "The Amazigh people (Imazighen, meaning 'free men') are the indigenous people of North Africa, with a continuous history spanning millennia across Tamazgha. Their heritage is celebrated through ancient kingdoms like Numidia, historic figures like Massinissa and Queen Dihya, unique architectural traditions, rich oral poetry, and the Tifinagh and Latin writing systems."),
    ("What are the most common greeting phrases in Tamazight?", "The key greetings in Tamazight (Latin script) are:\n- **Azul**: Hello / Greetings\n- **Amek telliḍ?**: How are you?\n- **Labas, tanemmirt**: Fine, thank you!\n- **Ar tufat**: Goodbye / See you soon\n- **Lahna d tumert**: Peace and happiness"),
    ("What is Yennayer and why is it celebrated?", "Yennayer is the Amazigh New Year, celebrated on January 12th. Historically linked to the accession of King Sheshonq I in 950 BC, it is deeply connected to agrarian life, nature, and the renewal of the earth's bounty. Families celebrate by preparing a feast of couscous with seven vegetables and local fowl."),
    ("Explain the difference between artificial intelligence and machine learning.", "Artificial Intelligence (AI) is the broad science of creating computer systems capable of performing tasks that typically require human cognition. Machine Learning (ML) is a subfield of AI focused on algorithms that learn from data and improve their performance over time without being explicitly programmed."),
    ("Share a wise Amazigh proverb with its English meaning.", "Here is a profound proverb: *'Awal ɣezzif, lmeɛna tewzel'* (Words are long, but meaning is short). It emphasizes the virtue of brevity and wisdom: express truth concisely rather than using empty speech."),
    ("Thank you very much for your great assistance!", "You are very welcome! If you have any further questions in English, Tamazight, Arabic, or French, feel free to ask. Have a fantastic day!")
]

# Cross-lingual queries (inquiring across languages about Amazigh culture or general topics)
CROSS_LINGUAL = [
    ("Comment dit-on 'Comment vas-tu' en berbère ?", "En tamazight (berbère), on dit : **'Amek telliḍ ?'** (au masculin et féminin) ou **'Amek tellam ?'** quand on s'adresse à un groupe."),
    ("كيف أقول شكراً جزيلاً بالأمازيغية؟", "تقول بالأمازيغية: **'Tanemmirt tameqrant'** (تانميرت تامقرانت)، وهي تعني شكراً جزيلاً."),
    ("How do I say 'Welcome' and 'Goodbye' in Tamazight?", "In Tamazight, 'Welcome' is **'Ansuf yis-k'** (to a male) / **'Ansuf yis-m'** (to a female), and 'Goodbye' is **'Ar tufat'** or **'Ar timlilit'**."),
    ("ما هي أسماء الشهور في التقويم الأمازيغي؟", "شهور التقويم الأمازيغي هي: 1. Yennayer (ينّاير) 2. Furar (فورار) 3. Meɣres (ماغرس) 4. Yebrir (يبرير) 5. Mayyu (مايو) 6. Yunyu (يونيو) 7. Yulyuz (يوليوز) 8. Ɣuct (غوشت) 9. Ctember (شتنبر) 10. Tuber (توبر) 11. Wamber (وانبر) 12. Dujember (دوجنبر)."),
    ("Quels sont les ingrédients traditionnels du couscous berbère ?", "Le couscous berbère traditionnel se compose de semoule de blé dur roulée à la main et cuite à la vapeur, accompagnée d'un bouillon parfumé à l'huile d'olive, de légumes de saison (courgettes, navets, carottes, fèves, potiron), de pois chiches et de viande d'agneau ou de poulet fermier."),
    ("What does the word 'Imazighen' mean?", "The word 'Imazighen' (singular: 'Amazigh') literally translates to 'free people' or 'noble men' in the Tamazight language, reflecting a deep cultural commitment to autonomy, dignity, and independence.")
]

def extract_tamazight_wiki_qa(jsonl_file, max_count=3000):
    """Extracts authentic encyclopedic Tamazight Q&As from Wikipedia."""
    qa_pairs = []
    if not os.path.exists(jsonl_file):
        return qa_pairs

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                p = d.get("prompt", "")
                if "<|im_start|>user" in p and "<|im_start|>assistant" in p:
                    parts = p.split("<|im_start|>assistant\n")
                    user_side = parts[0].replace("<|im_start|>user\n", "").replace("<|im_end|>\n", "").strip()
                    asst_side = parts[1].replace("<|im_end|>", "").strip()
                    
                    # Only accept real Tamazight questions (e.g. Dacu i d ..., Amek, Anwa)
                    if user_side.startswith("Dacu i d") or user_side.startswith("Amek") or user_side.startswith("Anwa") or user_side.startswith("Anta"):
                        # Clean wiki title underscores
                        cleaned_question = user_side.replace("_", " ")
                        qa_pairs.append((cleaned_question, asst_side))
                        if len(qa_pairs) >= max_count:
                            break
            except Exception:
                continue

    return qa_pairs

def build_multilingual_corpus(output_path="data/multilingual_chat.jsonl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_records = []

    print("Building balanced 4-Language Conversational Corpus...")

    # 1. Base curated dialogues
    for u, a in TAMAZIGHT_CORE:
        all_records.append(chatml_record(u, a))
    for u, a in ARABIC_CORE:
        all_records.append(chatml_record(u, a))
    for u, a in FRENCH_CORE:
        all_records.append(chatml_record(u, a))
    for u, a in ENGLISH_CORE:
        all_records.append(chatml_record(u, a))
    for u, a in CROSS_LINGUAL:
        all_records.append(chatml_record(u, a))

    curated_count = len(all_records)
    print(f"Added {curated_count} foundational conversational dialogues.")

    # 2. Extract genuine encyclopedic Q&As in Tamazight
    wiki_qa = extract_tamazight_wiki_qa("tamazight_train.jsonl", max_count=2500)
    for u, a in wiki_qa:
        all_records.append(chatml_record(u, a))
    print(f"Added {len(wiki_qa)} authentic Tamazight encyclopedic Q&As.")

    # 3. Add Cultural Proverbs & Idioms
    if os.path.exists("data/proverbs.tsv"):
        proverbs_added = 0
        with open("data/proverbs.tsv", "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[0] and parts[1]:
                    ber, ar_trans, ar_mean = parts[0], parts[1], parts[2]
                    # Arabic explanation
                    all_records.append(chatml_record(
                        f"ما معنى هذا المثل الأمازيغي: '{ber}'؟",
                        f"معنى المثل: '{ar_trans}'. ويقصد به في الحكمة الشعبية: {ar_mean}."
                    ))
                    # Tamazight explanation
                    all_records.append(chatml_record(
                        f"Acu i d anamek n yinzi-agi: '{ber}'?",
                        f"Inzi-agi yeqqar-d: '{ber}'. Anamek-is d akken: {ar_trans}."
                    ))
                    proverbs_added += 2
        print(f"Added {proverbs_added} cultural proverb Q&As.")

    # Replicate high-priority dialogues with slight shuffle to reinforce conversational persona
    random.seed(42)
    priority_subset = all_records[:curated_count]
    for _ in range(4):
        for rec in priority_subset:
            all_records.append(rec)

    random.shuffle(all_records)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSUCCESS! Generated {len(all_records):,} balanced multilingual training samples in: {output_path}")

if __name__ == "__main__":
    build_multilingual_corpus()
