"""
build_corpus.py — assembles the toy corpus, once, deterministically.

Run manually; the OUTPUT (`corpus/documents.jsonl`, `corpus/sources.json`) is
committed and is what the demo consumes. The demo never calls this file, so the
corpus is a frozen input to the run rather than something regenerated inside it.

    python tools/build_corpus.py

Why a hand-authored corpus. The assignment says plainly: "The implementation may
use a small corpus, tokenizer and model. The goal is not scale." Points are
awarded for the execution path, not for corpus realism, and downloading a real
dataset would put a network dependency inside a command that must run "without
manual intervention". What the corpus DOES have to be is structurally varied,
because packing policies and loss masks are graded per data type. So every lane
here carries the structure its policy has to cope with:

    plain prose        -> concatenate-and-chop is safe
    source code        -> boundaries matter, best-fit
    agentic traces     -> role spans; tool output must not be loss-bearing
    reasoning traces   -> must not be cut mid-argument
    Indic scripts      -> multi-byte, so the byte-level tokenizer is exercised

Three documents are planted deliberately:
    * `test-bench-002`  a benchmark item, never-train
    * `eng-contam-01`   a TRAINING document that shares a 13-gram with that
                        benchmark item, so the contamination scanner has
                        something real to catch
    * `val-holdout-01`  a validation document: readable for eval, never
                        gradient-bearing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

A6 = Path(__file__).resolve().parents[1]
CORPUS = A6 / "corpus"

# ---------------------------------------------------------------------------
# Sources. Mirrors the shape of the Session-5 inventory: every document traces
# to a source id that carries a licence tier and a provenance tag.
# ---------------------------------------------------------------------------
SOURCES = [
    dict(source_id="src-web-edu",      lane="english",           license="permissive",
         provenance="[PUB]", note="FineWeb-Edu/DCLM-grade web (proxy)"),
    dict(source_id="src-books-en",     lane="english",           license="share-alike",
         provenance="[EST]", note="long-form English (proxy)"),
    dict(source_id="src-stack-perm",   lane="code",              license="permissive",
         provenance="[PUB]", note="Stack-v2 permissive, lint-gated (proxy)"),
    dict(source_id="src-math-worked",  lane="math",              license="permissive",
         provenance="[PUB]", note="FineMath-grade worked solutions (proxy)"),
    dict(source_id="src-sangraha-ver", lane="indic_verified",    license="gov-open",
         provenance="[PUB]", note="Sangraha-verified native core (proxy)"),
    dict(source_id="src-indic-gov",    lane="indic_unverified",  license="gov-open",
         provenance="[EST]", note="judgments / gov records (proxy)"),
    dict(source_id="src-bpcc-parallel", lane="indic_translated", license="permissive",
         provenance="[PUB]", note="BPCC/Samanantar parallel corpus (proxy)"),
    dict(source_id="src-synth-translit", lane="indic_synthetic", license="own",
         provenance="[EST]", note="transliteration doubles, generated (proxy)"),
    dict(source_id="src-sandbox-selfplay", lane="agentic",       license="own",
         provenance="[EST]", note="sandbox self-play, execution-verified (proxy)"),
    dict(source_id="src-cot-verified", lane="reasoning",         license="permissive",
         provenance="[PUB]", note="answer-verified long CoT (proxy)"),
    dict(source_id="src-forums-codemix", lane="forums",          license="permissive",
         provenance="[EST]", note="romanized/code-mixed forums (proxy)"),
    dict(source_id="src-culturax",     lane="multilingual",      license="permissive",
         provenance="[PUB]", note="CulturaX non-Indic (proxy)"),
    dict(source_id="src-benchmarks",   lane="eval",              license="restricted",
         provenance="[PUB]", note="held-out benchmark items — NEVER TRAIN"),
]

LANGUAGE = {
    "english": ("en", "Latn"), "code": ("en", "Latn"), "math": ("en", "Latn"),
    "indic_verified": ("hi", "Deva"), "indic_unverified": ("hi", "Deva"),
    "indic_translated": ("hi", "Deva"), "indic_synthetic": ("hi", "Latn"),
    "agentic": ("en", "Latn"), "reasoning": ("en", "Latn"),
    "forums": ("hi-Latn", "Latn"), "multilingual": ("mul", "Latn"),
    "eval": ("en", "Latn"),
}

# ---------------------------------------------------------------------------
# ENGLISH — plain expository prose
# ---------------------------------------------------------------------------
ENGLISH = [
    "The southwest monsoon reaches the Kerala coast in the first week of June and "
    "spreads across the subcontinent over the following six weeks. Its arrival is "
    "declared only after a set of objective criteria are met: a threshold of rainfall "
    "at a fixed network of stations, a wind field of the right depth, and outgoing "
    "longwave radiation below a stated value. Declaring the onset early is costly, "
    "because sowing decisions follow the announcement across millions of hectares.",

    "A railway timetable is a scheduling problem disguised as a printed document. Every "
    "train occupies a block of track for an interval, and no two trains may hold the "
    "same block at once. Adding a single fast service can force dozens of slower "
    "services to be retimed, because the fast train consumes capacity out of proportion "
    "to the distance it covers. Planners therefore speak of paths rather than trains.",

    "Movable type did not spread because it produced better looking pages. Early printed "
    "books were often judged inferior to manuscripts. It spread because the marginal cost "
    "of the second copy collapsed. Once a forme was set, the expensive work was done, and "
    "each additional impression cost little more than paper and labour. Any technology "
    "that flattens the marginal cost curve reorganises the institutions built on top of it.",

    "Photosynthesis converts light energy into chemical energy stored in carbohydrate "
    "bonds. The light dependent reactions occur in the thylakoid membrane and produce ATP "
    "and NADPH. The carbon fixation reactions then use those products to reduce carbon "
    "dioxide into three carbon sugars. The two halves are often taught separately, which "
    "obscures the fact that neither runs for long without the other.",

    "In chess, material advantage is a heuristic rather than a rule. A rook is conventionally "
    "worth five pawns, but a rook trapped behind its own unmoved pieces may be worth less "
    "than an advanced passed pawn. Strong players evaluate positions rather than pieces, and "
    "the point values survive mainly as a teaching device for players who do not yet have "
    "the pattern library to evaluate directly.",

    "A city's water supply is limited less by total rainfall than by storage and "
    "distribution. A region may receive abundant rain in eight weeks and then face "
    "shortage in March, because the reservoirs that could have held the surplus were "
    "silted, or because distribution losses ran near forty percent. Investment in "
    "measurement usually returns more than investment in new sources.",

    "Standard time was adopted because railways made local solar time unworkable. When "
    "every town kept its own noon, a timetable had to be annotated with dozens of small "
    "offsets, and a missed connection could be blamed on arithmetic. The railways imposed "
    "a single reference, and the rest of civil life followed the railways rather than the "
    "other way round.",

    "Peer review is a filter with a known false negative rate. Important results have been "
    "rejected, and weak results have been accepted, in every field that has looked. The "
    "defensible claim is not that review identifies truth, but that it raises the cost of "
    "publishing an unexamined claim, and that the record it leaves lets a later reader "
    "reconstruct what was known at the time.",

    "Double entry bookkeeping survives because it is an error detecting code rather than a "
    "record keeping convention. Every transaction is written twice, once as a debit and once "
    "as a credit, and the two columns must agree. A single mistyped figure breaks the "
    "agreement and announces itself. The method spread through trading cities long before "
    "anyone described it in those terms.",

    "Vaccination protects two populations at once. The individual who receives the dose gains "
    "direct protection, and the people around that individual gain indirect protection because "
    "a chain of transmission is harder to sustain. The second effect is why coverage thresholds "
    "matter: below a certain fraction, the indirect benefit collapses even though every "
    "individual dose still worked exactly as intended.",

    "A bridge is designed against the load it will never quite see. Engineers size members for "
    "a combination of dead load, live load, wind and a factor of safety that absorbs the gap "
    "between the model and the world. The factor is not ignorance dressed up as arithmetic; it "
    "is an explicit statement of how much unmodelled variation the structure is expected to "
    "tolerate before anything yields.",

    "Soil fertility is depleted by harvest and restored by rotation, residue and fixation. A "
    "field cropped continuously with the same cereal loses nitrogen faster than the "
    "atmosphere replaces it, so yields decline even when rainfall is good. Legumes in the "
    "rotation host bacteria that fix nitrogen directly, which is why the practice predates any "
    "understanding of the chemistry involved.",

    "Compression works by removing redundancy, so a file that has already been compressed "
    "cannot usually be compressed again. This is not a limitation of a particular algorithm "
    "but a counting argument: there are fewer short strings than long ones, so no scheme can "
    "shorten every input. Every real compressor shortens the inputs it expects and lengthens "
    "the ones it does not.",

    "The distinction between accuracy and precision matters in every measurement. An "
    "instrument that reads 2.4713 kilograms every time is precise, but if the true mass is "
    "2.9 kilograms it is not accurate. Precision without accuracy is the more dangerous "
    "failure, because the repeatability of the readings looks like evidence that they are "
    "correct.",

    "Urban land value follows access rather than area. Two plots of identical size can differ "
    "in price by an order of magnitude because one sits ten minutes from a station and the "
    "other forty. Transport investment therefore redistributes value as much as it creates it, "
    "which is why the question of who captures the uplift is argued as fiercely as the "
    "question of where the line should run.",
]

# ---------------------------------------------------------------------------
# CODE — small self-contained Python units. Boundaries matter here.
# ---------------------------------------------------------------------------
CODE = [
    'def binary_search(items, target):\n'
    '    """Return the index of target in a sorted list, or -1."""\n'
    '    lo, hi = 0, len(items) - 1\n'
    '    while lo <= hi:\n'
    '        mid = (lo + hi) // 2\n'
    '        if items[mid] == target:\n'
    '            return mid\n'
    '        if items[mid] < target:\n'
    '            lo = mid + 1\n'
    '        else:\n'
    '            hi = mid - 1\n'
    '    return -1\n',

    'class LruCache:\n'
    '    """Fixed-capacity cache that evicts the least recently used key."""\n'
    '    def __init__(self, capacity):\n'
    '        self.capacity = capacity\n'
    '        self._data = {}\n'
    '        self._order = []\n'
    '\n'
    '    def get(self, key):\n'
    '        if key not in self._data:\n'
    '            return None\n'
    '        self._order.remove(key)\n'
    '        self._order.append(key)\n'
    '        return self._data[key]\n'
    '\n'
    '    def put(self, key, value):\n'
    '        if key in self._data:\n'
    '            self._order.remove(key)\n'
    '        elif len(self._data) >= self.capacity:\n'
    '            oldest = self._order.pop(0)\n'
    '            del self._data[oldest]\n'
    '        self._data[key] = value\n'
    '        self._order.append(key)\n',

    'def largest_remainder(shares, total):\n'
    '    """Allocate `total` integer slots across float shares without drift."""\n'
    '    exact = {k: v * total for k, v in shares.items()}\n'
    '    out = {k: int(v) for k, v in exact.items()}\n'
    '    short = total - sum(out.values())\n'
    '    order = sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))\n'
    '    for k in order[:short]:\n'
    '        out[k] += 1\n'
    '    return out\n',

    'def retry(times, exceptions=(OSError,)):\n'
    '    """Decorator that retries a call a fixed number of times."""\n'
    '    def wrap(fn):\n'
    '        def inner(*args, **kwargs):\n'
    '            last = None\n'
    '            for attempt in range(times):\n'
    '                try:\n'
    '                    return fn(*args, **kwargs)\n'
    '                except exceptions as exc:\n'
    '                    last = exc\n'
    '            raise last\n'
    '        return inner\n'
    '    return wrap\n',

    'def chunk(seq, size):\n'
    '    """Split a sequence into consecutive chunks of at most `size`."""\n'
    '    if size <= 0:\n'
    '        raise ValueError("size must be positive")\n'
    '    return [seq[i:i + size] for i in range(0, len(seq), size)]\n'
    '\n'
    '\n'
    'def flatten(nested):\n'
    '    """One level of flattening, preserving order."""\n'
    '    out = []\n'
    '    for part in nested:\n'
    '        out.extend(part)\n'
    '    return out\n',

    'import hashlib\n'
    '\n'
    '\n'
    'def content_hash(payload):\n'
    '    """Stable hash of a mapping, independent of key insertion order."""\n'
    '    items = sorted(payload.items())\n'
    '    joined = "|".join(f"{k}={v}" for k, v in items)\n'
    '    return hashlib.sha256(joined.encode("utf-8")).hexdigest()\n',

    'def moving_average(values, window):\n'
    '    """Simple trailing mean; returns a list shorter by window - 1."""\n'
    '    if window > len(values):\n'
    '        return []\n'
    '    out = []\n'
    '    running = sum(values[:window])\n'
    '    out.append(running / window)\n'
    '    for i in range(window, len(values)):\n'
    '        running += values[i] - values[i - window]\n'
    '        out.append(running / window)\n'
    '    return out\n',

    'def parse_csv_line(line, sep=","):\n'
    '    """Split one CSV line, honouring double quotes."""\n'
    '    fields, current, in_quotes = [], [], False\n'
    '    for ch in line:\n'
    '        if ch == \'"\':\n'
    '            in_quotes = not in_quotes\n'
    '        elif ch == sep and not in_quotes:\n'
    '            fields.append("".join(current))\n'
    '            current = []\n'
    '        else:\n'
    '            current.append(ch)\n'
    '    fields.append("".join(current))\n'
    '    return fields\n',
]

# ---------------------------------------------------------------------------
# MATH — worked solutions, Session-5 difficulty band 1-2
# ---------------------------------------------------------------------------
MATH = [
    "Problem. A train travels 240 km in 3 hours, then 180 km in 2 hours. Find the average "
    "speed for the whole journey.\nSolution. Total distance is 240 + 180 = 420 km. Total "
    "time is 3 + 2 = 5 hours. Average speed is total distance over total time, 420 / 5 = "
    "84 km per hour. Note that this is not the mean of 80 and 90, because the two legs "
    "took different amounts of time.",

    "Problem. Solve 3x + 7 = 2x + 15.\nSolution. Subtract 2x from both sides to get "
    "x + 7 = 15. Subtract 7 from both sides to get x = 8. Check by substitution: "
    "3(8) + 7 = 31 and 2(8) + 15 = 31, so the two sides agree and x = 8 is correct.",

    "Problem. A shopkeeper buys an item for 450 rupees and sells it for 540 rupees. Find "
    "the profit percentage.\nSolution. Profit is 540 - 450 = 90 rupees. Profit percentage "
    "is profit divided by cost price, times one hundred: 90 / 450 = 0.2, so the profit is "
    "20 percent. The selling price is not the denominator; using it would give 16.7 percent, "
    "which is the margin, a different quantity.",

    "Problem. Find the area of a triangle with vertices at (0, 0), (4, 0) and (0, 3).\n"
    "Solution. Two sides lie along the axes, so the base is 4 and the height is 3. Area is "
    "one half base times height, which is 0.5 * 4 * 3 = 6 square units. The shoelace formula "
    "gives the same answer and generalises to vertices in any position.",

    "Problem. A projectile is launched at 20 m/s at 30 degrees above the horizontal. Find the "
    "time of flight, taking g as 10 m/s squared.\nSolution. The vertical component of the "
    "initial velocity is 20 sin 30 = 10 m/s. Time to the apex is 10 / 10 = 1 second. Flight is "
    "symmetric about the apex, so the total time of flight is 2 seconds. Check the magnitude: "
    "a two second flight at this speed is physically reasonable.",

    "Problem. What is the probability of drawing two aces in a row from a standard deck, "
    "without replacement?\nSolution. The first draw is an ace with probability 4 / 52. Given "
    "that, only 3 aces remain among 51 cards, so the second draw is an ace with probability "
    "3 / 51. Multiply: (4 / 52) * (3 / 51) = 12 / 2652 = 1 / 221, about 0.45 percent.",
]

# ---------------------------------------------------------------------------
# INDIC — real Devanagari and Tamil, so the byte-level tokenizer is exercised
# on multi-byte scripts rather than on ASCII only.
# ---------------------------------------------------------------------------
INDIC_VERIFIED = [
    "भारत में मानसून जून के पहले सप्ताह में केरल तट पर पहुँचता है और लगभग छह सप्ताह में पूरे देश "
    "में फैल जाता है। किसान इसी वर्षा पर निर्भर रहते हैं, इसलिए मौसम विभाग की भविष्यवाणी हर "
    "वर्ष ध्यान से पढ़ी जाती है। समय से पहले घोषणा करने की कीमत बहुत बड़ी होती है।",

    "पुस्तकालय केवल पुस्तकों का संग्रह नहीं है। वह एक व्यवस्था है जिसमें खोजना, उधार लेना और "
    "लौटाना सरल बनाया जाता है। यदि सूची ठीक न हो तो सबसे अच्छी पुस्तक भी पाठक तक नहीं "
    "पहुँचती। इसलिए ग्रंथालय विज्ञान में वर्गीकरण को सबसे बुनियादी कार्य माना गया है।",

    "தமிழ்நாட்டின் கிழக்குக் கடற்கரையில் வடகிழக்குப் பருவமழை அக்டோபர் மாதத்தில் "
    "தொடங்குகிறது. சென்னை நகரின் குடிநீர்த் தேவையில் பெரும் பகுதி இந்த மழையிலிருந்தே "
    "நிரப்பப்படுகிறது. ஏரிகள் தூர் வாராமல் இருந்தால் அதிக மழையும் பயனற்றுப் போகும்.",

    "गणित की कक्षा में सबसे कठिन काम उत्तर निकालना नहीं, बल्कि यह समझाना है कि उत्तर सही "
    "क्यों है। विद्यार्थी अक्सर सूत्र याद कर लेते हैं और तर्क छोड़ देते हैं। परीक्षा में यह चल जाता है, "
    "परन्तु अगली कक्षा में यही आदत भारी पड़ती है।",

    "நூலகம் என்பது புத்தகங்களின் குவியல் மட்டுமல்ல. அது ஒரு ஒழுங்கமைப்பு. "
    "வகைப்படுத்தல் சரியாக இல்லையென்றால் சிறந்த நூலும் வாசகரைச் சென்றடையாது.",
]

INDIC_UNVERIFIED = [
    "उच्च न्यायालय ने अपने आदेश में कहा कि याचिकाकर्ता को सुनवाई का उचित अवसर नहीं दिया "
    "गया था। इस आधार पर पूर्व आदेश निरस्त किया जाता है और मामला पुनः विचार के लिए "
    "सक्षम प्राधिकारी को भेजा जाता है। चार सप्ताह के भीतर निर्णय अपेक्षित है।",

    "संसदीय समिति ने अपनी रिपोर्ट में कहा कि योजना का क्रियान्वयन धीमा रहा है। आवंटित "
    "राशि का केवल तिरपन प्रतिशत ही व्यय हुआ है। समिति ने सिफारिश की है कि मासिक "
    "समीक्षा बैठक अनिवार्य की जाए और आँकड़े सार्वजनिक रूप से प्रकाशित किए जाएँ।",

    "नगर निगम की अधिसूचना के अनुसार जल कर की दरें अगले वित्तीय वर्ष से संशोधित होंगी। "
    "घरेलू उपभोक्ताओं के लिए पहले बीस किलोलीटर पर दर अपरिवर्तित रहेगी। व्यावसायिक "
    "उपभोक्ताओं के लिए दर में वृद्धि प्रस्तावित है।",

    "राज्य सरकार ने ग्रामीण सड़क योजना के अंतर्गत सात सौ किलोमीटर सड़क के निर्माण को "
    "स्वीकृति दी है। कार्य तीन चरणों में पूरा किया जाएगा। गुणवत्ता जाँच के लिए तृतीय पक्ष "
    "निरीक्षण अनिवार्य किया गया है।",

    "मौसम विभाग ने अगले दो दिनों के लिए भारी वर्षा की चेतावनी जारी की है। तटीय जिलों में "
    "मछुआरों को समुद्र में न जाने की सलाह दी गई है। जिला प्रशासन ने राहत शिविर तैयार "
    "रखने के निर्देश दिए हैं।",
]

INDIC_TRANSLATED = [
    "The library is not merely a collection of books; it is a system.\n"
    "पुस्तकालय केवल पुस्तकों का संग्रह नहीं है; वह एक व्यवस्था है।",

    "Average speed is total distance divided by total time.\n"
    "औसत चाल कुल दूरी को कुल समय से भाग देने पर प्राप्त होती है।",

    "The committee recommended that monthly review meetings be made compulsory.\n"
    "समिति ने सिफारिश की कि मासिक समीक्षा बैठक अनिवार्य की जाए।",

    "Investment in measurement usually returns more than investment in new sources.\n"
    "मापन में किया गया निवेश प्रायः नए स्रोतों में किए गए निवेश से अधिक लाभ देता है।",
]

INDIC_SYNTHETIC = [
    "bharat mein mansoon june ke pehle saptah mein kerala tat par pahunchta hai.\n"
    "भारत में मानसून जून के पहले सप्ताह में केरल तट पर पहुँचता है।",

    "pustakalay keval pustakon ka sangrah nahin hai, vah ek vyavastha hai.\n"
    "पुस्तकालय केवल पुस्तकों का संग्रह नहीं है, वह एक व्यवस्था है।",

    "ausat chaal kul doori ko kul samay se bhag dene par prapt hoti hai.\n"
    "औसत चाल कुल दूरी को कुल समय से भाग देने पर प्राप्त होती है।",

    "mausam vibhag ne bhari varsha ki chetavani jari ki hai.\n"
    "मौसम विभाग ने भारी वर्षा की चेतावनी जारी की है।",
]

# ---------------------------------------------------------------------------
# FORUMS — romanized / code-mixed
# ---------------------------------------------------------------------------
FORUMS = [
    "yaar ye code chal hi nahi raha, import error aa raha hai. pip install karne ke baad "
    "bhi same problem. koi bataye kya galat hai?\nreply: virtual environment activate kiya "
    "tha? mostly wahi issue hota hai. terminal me which python chala ke dekh lo.",

    "question: 12th ke baad direct data science karna theek hai ya pehle CS degree?\n"
    "reply: degree se zyada portfolio matter karta hai, but pehli job ke liye degree filter "
    "ban jati hai. dono chalao, projects side me karte raho.",

    "monsoon me train late hone ka koi pattern hai kya? mujhe har saal july me delay milta hai.\n"
    "reply: haan, ghat sections me speed restriction lagta hai barish me. official reason "
    "track safety hai, aur wo genuine hai.",

    "koi acha hindi OCR bataye, purani scanned books ke liye. tesseract ka output bekar aa "
    "raha hai matras ke saath.\nreply: pehle image ko deskew aur binarize karo, accuracy "
    "kaafi improve hoti hai. legacy font ho to alag pipeline chahiye.",

    "interview me pucha gaya ki cache eviction kaise implement karoge. LRU bola, follow up "
    "aaya thread safe kaise banaoge.\nreply: lock lagana simplest hai, but contention badhta "
    "hai. sharded cache ya per bucket lock better scale karta hai.",
]

# ---------------------------------------------------------------------------
# MULTILINGUAL — non-Indic
# ---------------------------------------------------------------------------
MULTILINGUAL = [
    "La velocidad media de un viaje no es el promedio de las velocidades de cada tramo. "
    "Es la distancia total dividida entre el tiempo total. Cuando los tramos duran tiempos "
    "distintos, los dos cálculos no coinciden, y el segundo es el correcto.",

    "Une bibliothèque n'est pas seulement une collection de livres, c'est un système de "
    "classement. Si le catalogue est mauvais, le meilleur ouvrage n'atteindra jamais son "
    "lecteur. C'est pourquoi la classification est la tâche fondamentale.",

    "Musim hujan di kawasan tropis menentukan waktu tanam. Petani menunggu curah hujan "
    "yang cukup sebelum menyemai. Ramalan yang keliru merugikan banyak orang sekaligus, "
    "karena keputusan diambil serentak di wilayah yang luas.",

    "Die Einführung der Standardzeit folgte dem Eisenbahnverkehr. Solange jede Stadt ihre "
    "eigene Mittagszeit führte, war ein Fahrplan nur mit vielen kleinen Korrekturen lesbar. "
    "Die Eisenbahn setzte eine gemeinsame Referenz durch.",
]

# ---------------------------------------------------------------------------
# AGENTIC — trajectories. The point of this lane is the ROLE STRUCTURE:
# `tool_obs` is environment output and must never carry loss.
# ---------------------------------------------------------------------------
AGENTIC_TASKS = [
    dict(user="Find the total cost of 3 notebooks at 45 rupees each, then apply a 10 percent discount.",
         steps=[("think", "I should not do arithmetic in my head when a calculator is available. Multiply first, then discount."),
                ("tool_call", 'calc(expr="3*45")'),
                ("tool_obs", "135"),
                ("think", "Now take ten percent off that."),
                ("tool_call", 'calc(expr="135*0.9")'),
                ("tool_obs", "121.5")],
         final="Three notebooks cost 135 rupees, and after a 10 percent discount the total is 121.5 rupees."),

    dict(user="What is the current status of order A-4471?",
         steps=[("think", "I need to look this up rather than guess an order status."),
                ("tool_call", 'orders.lookup(id="A-4471")'),
                ("tool_obs", '{"id": "A-4471", "status": "shipped", "eta": "2026-02-14"}')],
         final="Order A-4471 has shipped and is expected to arrive on 14 February 2026."),

    dict(user="Book a train ticket from Pune to Nagpur under 800 rupees and confirm the PNR.",
         steps=[("think", "First search for services under the budget, then book, then verify the PNR rather than assuming success."),
                ("tool_call", 'rail.search(src="PNQ", dst="NGP", max_fare=800)'),
                ("tool_obs", '[{"train": "12135", "fare": 745, "class": "SL"}]'),
                ("tool_call", 'rail.book(train="12135", cls="SL")'),
                ("tool_obs", '{"pnr": "8842119736", "status": "CNF"}'),
                ("think", "The booking returned a PNR, but I should verify it independently before reporting success."),
                ("tool_call", 'rail.pnr(number="8842119736")'),
                ("tool_obs", '{"pnr": "8842119736", "status": "CNF", "berth": "S4-32"}')],
         final="Booked train 12135 in sleeper class for 745 rupees. PNR 8842119736 is confirmed, berth S4-32."),

    dict(user="How many rows are in the sales table for last quarter?",
         steps=[("think", "This needs a query against the database, not an estimate."),
                ("tool_call", 'sql.run(q="SELECT COUNT(*) FROM sales WHERE quarter = \'Q4\'")'),
                ("tool_obs", '{"count": 18422}')],
         final="There are 18,422 rows in the sales table for Q4."),

    dict(user="The build is failing. Find out why and fix it.",
         steps=[("think", "Read the failure output before changing anything."),
                ("tool_call", 'shell.run(cmd="make build")'),
                ("tool_obs", "src/parse.c:41: error: implicit declaration of function 'strdup'"),
                ("think", "strdup needs string.h. That is a missing include, not a logic error."),
                ("tool_call", 'edit.insert(file="src/parse.c", line=3, text="#include <string.h>")'),
                ("tool_obs", "ok"),
                ("tool_call", 'shell.run(cmd="make build")'),
                ("tool_obs", "build succeeded")],
         final="The build failed because src/parse.c used strdup without including string.h. I added the include and the build now succeeds."),

    dict(user="Convert 250 US dollars to rupees and tell me what that buys in monthly data.",
         steps=[("think", "I need a live rate; I must not recall one from memory."),
                ("tool_call", 'fx.rate(base="USD", quote="INR")'),
                ("tool_obs", '{"rate": 83.4}'),
                ("tool_call", 'calc(expr="250*83.4")'),
                ("tool_obs", "20850.0")],
         final="250 US dollars is about 20,850 rupees at the current rate of 83.4, which would cover a premium mobile data plan for roughly two years."),

    dict(user="Check whether the file report.csv has any rows with a missing region.",
         steps=[("think", "Read the file and count blanks rather than assuming the data is clean."),
                ("tool_call", 'shell.run(cmd="awk -F, \'$3 == \\"\\"\' report.csv | wc -l")'),
                ("tool_obs", "7")],
         final="Seven rows in report.csv have an empty region field."),

    dict(user="Summarise yesterday's error log and tell me the most common failure.",
         steps=[("think", "Aggregate the log rather than reading it line by line."),
                ("tool_call", 'logs.top(day="yesterday", field="error_code", n=3)'),
                ("tool_obs", '[["ETIMEDOUT", 412], ["ECONNRESET", 118], ["EACCES", 9]]'),
                ("think", "Timeouts dominate by a wide margin, so that is the finding worth leading with.")],
         final="There were 539 errors yesterday. ETIMEDOUT accounts for 412 of them, roughly three quarters, so connection timeouts are the dominant failure."),
]

# ---------------------------------------------------------------------------
# REASONING — long CoT, band 2-3. Must not be cut mid-argument.
# ---------------------------------------------------------------------------
REASONING = [
    dict(user="A tank fills in 6 hours through pipe A and 9 hours through pipe B. How long with both open?",
         think="Work out rates rather than times, because rates add and times do not. Pipe A fills one sixth of the "
               "tank per hour. Pipe B fills one ninth per hour. Together they fill one sixth plus one ninth. The "
               "common denominator is eighteen, so that is three eighteenths plus two eighteenths, which is five "
               "eighteenths per hour. Time is the reciprocal of the rate, so eighteen fifths hours. That is 3.6 hours, "
               "or three hours and thirty six minutes. Sanity check: the answer must be less than six, the faster "
               "single pipe, and it is.",
         final="Both pipes together fill the tank in 18/5 hours, that is 3 hours and 36 minutes."),

    dict(user="Is it cheaper to buy a monthly pass at 1100 rupees or pay 45 rupees per trip if I travel 22 working days, twice a day?",
         think="Count the trips first. Twenty two working days at two trips a day is forty four trips. At forty five "
               "rupees each that is forty four times forty five. Forty times forty five is eighteen hundred, and four "
               "times forty five is one hundred eighty, so one thousand nine hundred eighty rupees. The pass costs "
               "eleven hundred. The pass is cheaper by eight hundred eighty rupees. The break even point is eleven "
               "hundred divided by forty five, about twenty four and a half trips, so the pass wins from about "
               "thirteen working days onward.",
         final="Pay-per-trip costs 1,980 rupees for 44 trips, so the 1,100 rupee pass saves 880 rupees. The pass pays for itself after about 25 trips."),

    dict(user="Why does adding a second index sometimes make a database query slower?",
         think="An index speeds up lookups but must be maintained on every write. That is the first cost. The second "
               "cost is subtler: the query planner chooses a plan using statistics, and a new index changes the space "
               "of plans it considers. If the statistics are stale or the index is only weakly selective, the planner "
               "may pick an index scan where a sequential scan would have been faster, because it estimates the row "
               "count badly. So the slowdown is not caused by the index doing work; it is caused by the planner "
               "choosing differently.",
         final="Two reasons: every write must now maintain the extra index, and the planner may switch to a worse plan if the new index looks selective in stale statistics but is not."),

    dict(user="A test is 95 percent accurate and 1 in 1000 people have the condition. Someone tests positive. What is the chance they have it?",
         think="This is base rate reasoning. Take one hundred thousand people. One hundred of them have the condition; "
               "ninety nine thousand nine hundred do not. Of the hundred who have it, the test correctly flags ninety "
               "five. Of the ninety nine thousand nine hundred who do not, five percent are false positives, which is "
               "about four thousand nine hundred ninety five. So positives total roughly five thousand ninety, of which "
               "ninety five are true. Ninety five divided by five thousand ninety is about 1.9 percent. The intuition "
               "that a ninety five percent accurate test means ninety five percent confidence is badly wrong when the "
               "condition is rare.",
         final="About 1.9 percent. The false positives from the large healthy population swamp the true positives, because the condition is rare."),

    dict(user="Should a training run repeat scarce data or generate synthetic replacements?",
         think="Repetition and generation fail in different ways. Repetition gives back exactly the distribution you "
               "already have, and the returns fall off as passes accumulate; past roughly four passes the extra compute "
               "buys very little. Generation adds new surface but can add error that no gate catches. So the deciding "
               "question is whether an independent verifier exists. For code, execution is such a verifier. For a "
               "mathematical answer, checking the answer is. For fluent prose there is no verifier that was not itself "
               "generated, so generation there is unbounded risk and repetition is the safer choice.",
         final="Repeat when no independent verifier exists, generate when one does. Execution and answer-checking are real gates; fluency is not."),

    dict(user="Why is average speed not the average of the two speeds?",
         think="Because averaging speeds implicitly weights each leg equally, and the correct weight is time, not "
               "distance or count. If a car covers a leg at eighty and another at ninety but spends three hours on the "
               "first and two on the second, the eighty applies for longer, so the true average must sit below the "
               "midpoint of eighty five. Compute it properly: total distance two hundred forty plus one hundred eighty "
               "is four hundred twenty, over five hours, is eighty four. Eighty four is indeed below eighty five, which "
               "confirms the weighting argument.",
         final="Because the correct average weights each leg by the time spent on it, not by treating both legs equally."),
]

# ---------------------------------------------------------------------------
# EVAL / VALIDATION / TEST — the firewall's reason to exist
# ---------------------------------------------------------------------------
# `test-bench-002` and `eng-contam-01` deliberately share a long verbatim run of
# words, so the contamination scanner has a real 13-gram to catch. If you edit
# one, edit the other, or the firewall demonstration stops demonstrating anything.
CONTAMINATED_SPAN = (
    "the capital city of the state of odisha is bhubaneswar and it was designed "
    "by the german architect otto koenigsberger in nineteen forty eight"
)

EVAL_DOCS = [
    dict(doc_id="test-bench-001", split="test", benchmark="milu-mini", version="v1.0",
         text="Question: Which river flows through the city of Varanasi? Answer: The Ganga."),
    dict(doc_id="test-bench-002", split="test", benchmark="milu-mini", version="v1.0",
         text="Question: What is the capital of Odisha and who designed it? Answer: "
              + CONTAMINATED_SPAN + "."),
    dict(doc_id="test-bench-003", split="test", benchmark="humaneval-mini", version="v1.0",
         text="Write a function that returns the n-th Fibonacci number using memoisation. "
              "Assert that fib(10) equals 55."),
    dict(doc_id="test-canary-001", split="test", benchmark="canary", version="v1.0",
         text="CANARY-ERAV5-9f3a71c2-DO-NOT-TRAIN-ON-THIS-STRING-EVER"),
    dict(doc_id="val-holdout-01", split="validation", benchmark="perplexity-holdout", version="v1.0",
         text="A timetable is a scheduling problem in printed form, and every added service "
              "consumes capacity from the services around it."),
    dict(doc_id="val-holdout-02", split="validation", benchmark="indic-perplexity-holdout", version="v1.0",
         text="राज्य सरकार ने ग्रामीण क्षेत्रों में पेयजल आपूर्ति के लिए नई योजना की घोषणा की है।"),
]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def doc(doc_id, lane, kind, segments, source_id, *, split="train",
        contamination="clean", license_override=None, extra=None):
    lang, script = LANGUAGE[lane]
    src = next(s for s in SOURCES if s["source_id"] == source_id)
    rec = dict(
        doc_id=doc_id,
        lane=lane,
        kind=kind,
        split=split,
        language=lang,
        script=script,
        source_id=source_id,
        segments=segments,
        meta=dict(
            license=license_override or src["license"],
            provenance=src["provenance"],
            tier="A" if lane in ("indic_verified", "reasoning", "agentic", "math") else "B",
            cleaning_pipeline="erav4-clean@4.2.1",
            dedup_status="exact+fuzzy-passed",
            contamination_status=contamination,
            collected="2026-01-14",
        ),
    )
    if extra:
        rec["meta"].update(extra)
    return rec


def plain(prefix, lane, texts, source_id):
    return [
        doc(f"{prefix}-{i:03d}", lane, "plain", [dict(role="text", text=t)], source_id)
        for i, t in enumerate(texts)
    ]


def build():
    docs = []

    docs += plain("eng", "english", ENGLISH, "src-web-edu")
    docs += plain("code", "code", CODE, "src-stack-perm")
    docs += plain("math", "math", MATH, "src-math-worked")
    docs += plain("iv", "indic_verified", INDIC_VERIFIED, "src-sangraha-ver")
    docs += plain("iu", "indic_unverified", INDIC_UNVERIFIED, "src-indic-gov")
    docs += plain("it", "indic_translated", INDIC_TRANSLATED, "src-bpcc-parallel")
    docs += plain("is", "indic_synthetic", INDIC_SYNTHETIC, "src-synth-translit")
    docs += plain("for", "forums", FORUMS, "src-forums-codemix")
    docs += plain("mul", "multilingual", MULTILINGUAL, "src-culturax")

    # Agentic: role-structured trajectories.
    for i, task in enumerate(AGENTIC_TASKS):
        segments = [dict(role="user", text=task["user"])]
        for role, text in task["steps"]:
            segments.append(dict(role=role, text=text))
        segments.append(dict(role="assistant", text=task["final"]))
        docs.append(doc(f"agt-{i:03d}", "agentic", "trajectory", segments,
                        "src-sandbox-selfplay",
                        extra=dict(execution_verified=True)))

    # Reasoning: think + answer, no tool observations.
    for i, item in enumerate(REASONING):
        segments = [
            dict(role="user", text=item["user"]),
            dict(role="think", text=item["think"]),
            dict(role="assistant", text=item["final"]),
        ]
        docs.append(doc(f"rsn-{i:03d}", "reasoning", "reasoning", segments,
                        "src-cot-verified",
                        extra=dict(answer_verified=True)))

    # The planted contaminated training document. It is a genuine-looking
    # English document that happens to contain a long verbatim run from a
    # benchmark item. Its contamination_status is left "unscanned" on purpose:
    # the SCANNER has to discover the overlap, not the corpus author.
    docs.append(doc(
        "eng-contam-01", "english", "plain",
        [dict(role="text",
              text="A short note on state capitals. Bhubaneswar replaced Cuttack as the "
                   "administrative centre shortly after independence. For the record, "
                   + CONTAMINATED_SPAN + ". The plan was unusual for its time in "
                     "separating residential neighbourhoods from the administrative core.")],
        "src-web-edu", contamination="unscanned"))

    # A document whose licence the admission gate must refuse.
    docs.append(doc(
        "eng-badlicense-01", "english", "plain",
        [dict(role="text",
              text="An excerpt of unclear provenance whose licence could not be established "
                   "during collection. It reads plausibly, which is precisely why the gate "
                   "must decide on metadata rather than on how the text looks.")],
        "src-web-edu", license_override="unknown"))

    # Evaluation, validation and test documents.
    for e in EVAL_DOCS:
        rec = dict(
            doc_id=e["doc_id"], lane="eval", kind="plain", split=e["split"],
            language="hi" if "indic" in e["benchmark"] else "en",
            script="Deva" if "indic" in e["benchmark"] else "Latn",
            source_id="src-benchmarks",
            segments=[dict(role="text", text=e["text"])],
            meta=dict(license="restricted", provenance="[PUB]", tier="A",
                      cleaning_pipeline="erav4-clean@4.2.1",
                      dedup_status="exact+fuzzy-passed",
                      contamination_status="is_eval",
                      collected="2026-01-14",
                      benchmark=e["benchmark"], benchmark_version=e["version"],
                      never_train=(e["split"] == "test")),
        )
        docs.append(rec)

    return docs


def main():
    CORPUS.mkdir(parents=True, exist_ok=True)
    docs = build()

    out = CORPUS / "documents.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")

    with open(CORPUS / "sources.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(SOURCES, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    by_lane, by_split = {}, {}
    chars = 0
    for d in docs:
        by_lane[d["lane"]] = by_lane.get(d["lane"], 0) + 1
        by_split[d["split"]] = by_split.get(d["split"], 0) + 1
        chars += sum(len(s["text"]) for s in d["segments"])

    print(f"wrote {len(docs)} documents, {chars:,} characters -> {out}")
    for lane in sorted(by_lane):
        print(f"  {lane:20s} {by_lane[lane]:3d}")
    print("  splits:", by_split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
