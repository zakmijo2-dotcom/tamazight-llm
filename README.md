# Tamazight LLM (SmolLM2-135M) ⵣ

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/zakmijo2-dotcom/tamazight-llm)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

مشروع تدريب وتطوير نموذج ذكاء اصطناعي خفيف ومفتوح المصدر للغة **الأمازيغية بالحرف اللاتيني (Tamaziɣt s ugemmay alatini)**، مصمم ليعمل بسرعة فائقة على الأجهزة المحمولة عبر **Termux** ومناسب للتدريب الكامل في **GitHub Codespaces**.

---

## 🚀 التشغيل المباشر في GitHub Codespaces

اضغط على الزر أعلاه أو اتبع الخطوات التالية لمواصلة التدريب السريع على معالجات وسيرفرات GitHub السحابية (مع دعم GPU/CUDA مجاناً):

```bash
# 1. تثبيت المتطلبات (يتم تلقائياً عبر .devcontainer)
pip install -r requirements.txt

# 2. بدء التدريب الكامل على معالج الرسوميات أو المعالج السحابي
python3 train_codespaces.py --epochs 3 --batch-size 8 --lr 2e-4

# 3. الرفع مباشرة على Hugging Face بعد انتهاء التدريب
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
│   └── devcontainer.json        # تهيئة بيئة Codespaces السحابية تلقائياً
├── data/
│   ├── kabyle_french.tsv        # الجمل المترجمة الأصلية
│   ├── proverbs.tsv             # الأمثال الأمازيغية
│   ├── idioms.tsv               # التعابير
│   └── ...
├── tamazight_train.jsonl        # ملف التدريب المجهز الكامل (126,929 سطر)
├── train_codespaces.py          # سكربت التدريب الاحترافي في Codespaces
├── train_lora_tamazight.py      # سكربت تدريب LoRA المخصص لـ Termux
├── upload_to_hf.py              # سكربت الرفع التلقائي على Hugging Face
├── requirements.txt             # متطلبات بايثون
└── README.md                    # دليل الاستخدام
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
