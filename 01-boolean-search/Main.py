import json
from hazm import Normalizer, word_tokenize, stopwords_list, Lemmatizer
from tqdm import tqdm


class BooleanSearchEngine:
    def __init__(self):
        self.inverted_index = {}
        self.documents = {}
        self.total_docs = 0

        # ابزارهای پیش‌پردازش هضم
        self.normalizer = Normalizer()
        self.lemmatizer = Lemmatizer()
        # دریافت لیست استاپ‌وردهای فارسی
        self.stopwords = set(stopwords_list())

    def preprocess(self, text):
        """
        پیش‌پردازش متن شامل:
        ۱. نرمال‌سازی (اصلاح نیم‌فاصله‌ها و ی/ک)
        ۲. توکنایز کردن
        ۳. حذف استاپ‌وردها
        ۴. ریشه‌یابی (Lemmatization)
        """
        if not isinstance(text, str):
            return []

        # نرمال‌سازی
        normalized_text = self.normalizer.normalize(text)
        # قطعه‌قطعه کردن متن به کلمات
        tokens = word_tokenize(normalized_text)

        processed_tokens = []
        for token in tokens:
            # حذف علائم نگارشی و استاپ‌وردها
            if token not in self.stopwords and token.isalnum():
                # ریشه‌یابی کلمه برای بهبود کیفیت ایندکس
                lemmatized_token = self.lemmatizer.lemmatize(token)
                # در هضم، ریشه‌یابی افعال گاهی به فرم "بن‌ماضی#بن‌مضارع" برمی‌گردد.
                # ما بخش اول (بن ماضی) یا کل آن را در نظر می‌گیریم.
                if '#' in lemmatized_token:
                    lemmatized_token = lemmatized_token.split('#')[0]
                processed_tokens.append(lemmatized_token)

        return processed_tokens

    def build_index(self, json_file_path):
        """ساخت Inverted Index از فایل JSON"""
        print("Loading data and building index...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        docs = data.get("documents", [])
        self.total_docs = len(docs)

        # استفاده از tqdm برای نمایش نوار پیشرفت
        for doc_id, doc in enumerate(tqdm(docs, desc="Indexing Documents")):
            self.documents[doc_id] = doc

            # استخراج و تجمیع متن از فیلدهای مهم سند
            content_parts = []
            if "title" in doc and doc["title"]:
                content_parts.append(doc["title"])
            if "abstract" in doc and doc["abstract"]:
                content_parts.append(doc["abstract"])
            if "keywords" in doc and doc["keywords"]:
                content_parts.extend(doc["keywords"])

            full_text = " ".join(content_parts)

            # پیش‌پردازش متن برای استخراج توکن‌های نهایی
            tokens = self.preprocess(full_text)

            # اضافه کردن توکن‌ها به ایندکس (استفاده از set برای جلوگیری از تکرار doc_id در یک کلمه)
            for token in set(tokens):
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append(doc_id)

        print(f"\nIndex built successfully! Total documents indexed: {self.total_docs}")
        print(f"Total unique tokens in vocabulary: {len(self.inverted_index)}\n")

    def search(self, query, query_type="AND"):
        """
        پردازش Query و جستجوی بولی (AND / OR)
        """
        query_tokens = self.preprocess(query)

        # هندل کردن کوئری خالی یا کوئری که فقط شامل استاپ‌ورد باشد (بدون ارور)
        if not query_tokens:
            return []

        print(f"Processed Query Tokens: {query_tokens}")

        # بازیابی لیست شناسه‌ها (doc_ids) برای هر کلمه پرس‌وجو
        results_sets = []
        for token in query_tokens:
            # اگر کلمه در ایندکس نبود، با استفاده از get یک لیست خالی برمی‌گردد و ارور نمی‌دهد
            docs_with_token = set(self.inverted_index.get(token, []))
            results_sets.append(docs_with_token)

        # انجام عملیات بولی
        if query_type == "AND":
            # اشتراک تمام مجموعه‌ها
            final_results = set.intersection(*results_sets)
        elif query_type == "OR":
            # اجتماع تمام مجموعه‌ها
            final_results = set.union(*results_sets)
        else:
            raise ValueError("Query type must be 'AND' or 'OR'")

        # برگرداندن لیست شناسه‌ها به صورت مرتب شده
        return sorted(list(final_results))

    def display_results(self, doc_ids):
        """نمایش جزئیات اسناد یافت شده"""
        if not doc_ids:
            print("No documents found.")
            return

        print(f"--- Found {len(doc_ids)} documents ---")
        for doc_id in doc_ids:
            doc = self.documents[doc_id]
            print(f"Doc ID: {doc_id} | Title: {doc.get('title', 'No Title')}")
        print("-" * 40)


# =========================================
if __name__ == "__main__":
    engine = BooleanSearchEngine()

    engine.build_index("ganj_results_2026.json")

    print("\n" + "=" * 50)

    # تست ۱: جستجوی AND (باید کلماتی رو بیاره که هر دو در سند باشن)
    print("\n>>> Test 1: AND Query ('مسئولیت مدنی')")
    and_results = engine.search("مسئولیت مدنی", query_type="AND")
    engine.display_results(and_results)

    # تست ۲: جستجوی OR (باید اسنادی رو بیاره که حداقل یکی از کلمات رو داشته باشن)
    print("\n>>> Test 2: OR Query ('پزشکی درمان')")
    or_results = engine.search("پزشکی درمان", query_type="OR")
    engine.display_results(or_results)

    # تست ۳: تست کلمه ناموجود (نباید ارور بده، باید لیست خالی برگردونه)
    print("\n>>> Test 3: Non-existent Word")
    none_results = engine.search("واژه‌عجیب‌تست‌نشده", query_type="AND")
    engine.display_results(none_results)

    # تست ۴: تست کوئری کاملا خالی یا کوئری شامل فقط استاپ‌ورد (نباید ارور بده)
    print("\n>>> Test 4: Empty Query / Only Stopwords ('از و با')")
    empty_results = engine.search("از و با", query_type="AND")
    engine.display_results(empty_results)