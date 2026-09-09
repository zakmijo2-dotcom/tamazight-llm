# Tamazight Multilingual LLM (Tamaziɣt · العربية · Français · English) ⵣ

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/zakmijo2-dotcom/tamazight-llm)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

نموذج ذكاء اصطناعي محادثي حقيقي متعدد اللغات (Multilingual Conversational LLM) يتقن التحاور والإجابة بطبيعية بأربع لغات رئيسية:
- **الأمازيغية بالحرف اللاتيني (Tamaziɣt s ugemmay alatini)**
- **العربية الفصحى (العربية)**
- **الفرنسية (Français)**
- **الإنجليزية (English)**

مصمم ليعمل بسرعة عالية على الأجهزة المحمولة عبر **Termux** بصيغة GGUF، وعلى معالجات **Codespaces** وكروت الشاشة المجانية في **Google Colab**.

## 🚀 البدء السريع (Quickstart)

### 1. تثبيت المتطلبات (Environment Setup)
```bash
pip install -r requirements.txt
```

### 2. تجربة المحادثة والاستدلال (Inference & Chat)
يمكنك تجربة النموذج مباشرة (سواء النموذج الأساسي أو بعد التدريب):
```bash
# وضع المحادثة التفاعلية في الطرفية
python3 chat.py

# أو استعلام فردي مباشر
python3 infer.py --prompt "Azul fell-awen! Amek telliḍ?"

# أو الترجمة الفورية من الفرنسية إلى الأمازيغية
python3 infer.py --translate "Bonjour mon ami, comment vas-tu ?"
```

### 3. تدريب النموذج (Model Training & Fine-Tuning)
```bash
# تدريب قياسي مع تقسيم التحقق وحساب الخسارة على ردود المساعد فقط (SFT Masking)
python3 train.py --epochs 3 --batch-size 8 --lr 2e-4

# تدريب سريع لاختبار سير العمل (مثال 500 عينة):
python3 train.py --max-samples 500 --epochs 1 --batch-size 4
```

### 4. تقييم أداء النموذج (Evaluation & Benchmarks)
```bash
python3 evaluate.py --model-path tamazight-smollm-135m-final
```
يقوم بحساب:
- الحيرة اللغوية (Perplexity - PPL) على نصوص أمازيغية لم يرها النموذج.
- دقة الترجمة على حزمة اختبار قياسية (French -> Tamazight).
- جودة المحادثة والأمثال الثقافية.

### 5. تصدير النموذج للهواتف بصيغة GGUF (Mobile Export)
```bash
python3 export_gguf.py --model-dir tamazight-smollm-135m-final --outtype f16
```

### 6. الرفع على Hugging Face Hub
```bash
python3 upload_to_hf.py --repo-id "YOUR_USERNAME/tamazight-smollm-135m" --token "hf_YOUR_TOKEN"
```
---

## 📊 بيانات التدريب (Training Data)

تم جمع وتنظيف وتنسيق أضخم قاعدة بيانات أمازيغية لاتينية في ملف جاهز (`tamazight_train.jsonl`):
- **115,382** جملة مترجمة أمازيغية-فرنسية من ويكاموس والبيانات الموازية (`Sifal/Kabyle-French`).
- **8,897** مقالة كاملة من ويكيبيديا الأمازيغية القبائلية (`Sifal/KabyleWikipedia`).
- أمثال وحكم وألغاز من المعجم الأمازيغي (`Tamazight-NLP/AmaWar`).
- قوالب حوارية يومية (Azul, Amek telliḍ, Tanemmirt, ... إلخ).
- إجمالي العينات المعدة: **126,929 عينة** بحجم يتجاوز **33 ميغابايت**.

---

## 📁 هيكل المشروع (Repository Structure)

```text
├── .devcontainer/
│   └── devcontainer.json        # تهيئة بيئة Codespaces السحابية
├── data/                        # البيانات المعجمية الأصلية
│   ├── kabyle_french.tsv        # الجمل المترجمة الأصلية
│   ├── proverbs.tsv             # الأمثال الأمازيغية
│   ├── idioms.tsv               # التعابير
│   ├── riddles.tsv              # الألغاز
│   └── examples.tsv             # الأمثلة السياقية
├── tamazight_train.jsonl        # ملف التدريب المجهز الكامل (126,929 سطر)
├── train.py                     # سكربت التدريب الشامل والمتقدم
├── train_codespaces.py          # خط تدريب Codespaces والحوسبة السحابية
├── infer.py                     # واجهة المحادثة والاستدلال والترجمة
├── chat.py                      # اختصار لتشغيل المحادثة التفاعلية
├── evaluate.py                  # حزمة تقييم الحيرة اللغوية والترجمة (Benchmarking)
├── export_gguf.py               # تحويل النموذج إلى صيغة GGUF لتشغيله على الهواتف
├── upload_to_hf.py              # سكربت الرفع التلقائي على Hugging Face
├── requirements.txt             # متطلبات بايثون
└── README.md                    # دليل الاستخدام الشامل
```

---

## 📱 تشغيل النموذج في Termux (Android)

النموذج مبني على معمارية `SmolLM2-135M` ليعمل بأقل استهلاك لذاكرة الوصول العشوائي (أقل من 150MB في صيغة GGUF Q4_K_M) وبسرعة تفوق 40 كلمة في الثانية على هواتف أندرويد عبر `llama.cpp`:

```bash
# تشغيل الاستدلال السريع في Termux
llama-cli -m SmolLM2-135M-Instruct-Q4_K_M.gguf -p "Azul fell-awen! Amek telliḍ?" -n 50 --temp 0.7
```

---

## 🤝 المساهمة
المشروع مفتوح المصدر لدعم المحتوى واللغة الأمازيغية في مجال الذكاء الاصطناعي ومعالجة اللغات الطبيعية (NLP).
