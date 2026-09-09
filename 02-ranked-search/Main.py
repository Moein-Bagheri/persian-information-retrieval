import json
import math
from hazm import Normalizer, word_tokenize, stopwords_list, Lemmatizer
from tqdm import tqdm


class BooleanSearchEngine:
    def __init__(self):
        self.inverted_index = {}
        self.documents = {}
        self.total_docs = 0
        self.term_frequencies = {}
        self.document_lengths = {}
        self.average_document_length = 0
        self.k1 = 1.2
        self.b = 0.75

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

            # استخراج متن
            content_parts = []

            if "title" in doc and doc["title"]:
                content_parts.append(doc["title"])

            if "abstract" in doc and doc["abstract"]:
                content_parts.append(doc["abstract"])

            if "keywords" in doc and doc["keywords"]:
                content_parts.extend(doc["keywords"])

            full_text = " ".join(content_parts)

            # پیش پردازش
            tokens = self.preprocess(full_text)
           
            # طول سند
            self.document_lengths[doc_id] = len(tokens)

            # فراوانی واژه ها
            self.term_frequencies[doc_id] = {}

            for token in tokens:
                if token not in self.term_frequencies[doc_id]:
                    self.term_frequencies[doc_id][token] = 0

                self.term_frequencies[doc_id][token] += 1

            # ساخت Inverted Index
            for token in self.term_frequencies[doc_id]:
                if token not in self.inverted_index:
                    self.inverted_index[token] = []

                self.inverted_index[token].append(doc_id)

        # میانگین طول اسناد
        self.average_document_length = (
            sum(self.document_lengths.values()) / self.total_docs
        )

        print(f"\nIndex built successfully! Total documents indexed: {self.total_docs}")
        print(f"Total unique tokens in vocabulary: {len(self.inverted_index)}\n")
                
    def calculate_tf(self, term, doc_id):
            """
            محاسبه TF یک کلمه در یک سند
            """
            if doc_id not in self.term_frequencies:
                return 0

            return self.term_frequencies[doc_id].get(term, 0)
    
    def calculate_idf(self, term):
        """
        محاسبه IDF یک کلمه
        """
        if term not in self.inverted_index:
            return 0

        document_frequency = len(self.inverted_index[term])

        return math.log(self.total_docs / document_frequency)
    
    def calculate_bm25_idf(self, term):
        """
        محاسبه IDF مخصوص BM25
        """
        if term not in self.inverted_index:
            return 0

        df = len(self.inverted_index[term])

        return math.log(
            (self.total_docs - df + 0.5) /
            (df + 0.5)
        )
    def calculate_bm25_score(self, query_terms, doc_id):
        """
        محاسبه امتیاز BM25 برای یک سند
        """
        score = 0

        doc_length = self.document_lengths[doc_id]

        for term in query_terms:

            if term not in self.inverted_index:
                continue

            tf = self.calculate_tf(term, doc_id)

            if tf == 0:
                continue

            idf = self.calculate_bm25_idf(term)

            numerator = tf * (self.k1 + 1)

            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.average_document_length)
            )

            score += idf * (numerator / denominator)

        return score
    
    def bm25_search(self, query):
        """
        جستجو با استفاده از BM25
        """
        query_terms = self.preprocess(query)

        if not query_terms:
            return []

        scores = {}

        for doc_id in self.documents:
            score = self.calculate_bm25_score(query_terms, doc_id)

            if score > 0:
                scores[doc_id] = score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def tfidf_search(self, query):
        """
        جستجو با استفاده از TF-IDF
        """
        query_tokens = self.preprocess(query)

        if not query_tokens:
            return []

        scores = {}

        for term in query_tokens:
            idf = self.calculate_idf(term)

            if term not in self.inverted_index:
                continue

            for doc_id in self.inverted_index[term]:
                tf = self.calculate_tf(term, doc_id)
                score = tf * idf

                if doc_id not in scores:
                    scores[doc_id] = 0

                scores[doc_id] += score

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

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


    print("\n>>> Test 5: TF-IDF Search ('مسئولیت مدنی')")

    tfidf_results = engine.tfidf_search("مسئولیت مدنی")

    print("\nResults:\n")

    for doc_id, score in tfidf_results:
        print(f"doc_{doc_id}    score: {score:.2f}")


    print("\n>>> Test 5-2: TF-IDF Search ('پزشکی درمان')")

    tfidf_results2 = engine.tfidf_search("پزشکی درمان")    
    print("\nResults:\n")

    for doc_id, score in tfidf_results2:
        print(f"doc_{doc_id}    score: {score:.2f}")


    print("\n>>> Test 6: BM25 Search ('مسئولیت مدنی')")

    bm25_results = engine.bm25_search("مسئولیت مدنی")

    print("\nResults:\n")

    for doc_id, score in bm25_results:
        print(f"doc_{doc_id}    score: {score:.2f}")

    print("\n>>> Test 6-2: BM25 Search ('پزشکی درمان')")

    bm25_results2 = engine.bm25_search("پزشکی درمان")

    print("\nResults:\n")

    for doc_id, score in bm25_results2:
        print(f"doc_{doc_id}    score: {score:.2f}")    



    print("\n" + "=" * 50)
    print("Interactive Search")
while True:
    query = input("\nEnter your query (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    print("\nChoose search method:")
    print("1. Boolean AND")
    print("2. Boolean OR")
    print("3. TF-IDF")
    print("4. BM25")

    choice = input("Enter your choice (1-4): ")

    if choice == "1":
        results = engine.search(query, query_type="AND")
        engine.display_results(results)

    elif choice == "2":
        results = engine.search(query, query_type="OR")
        engine.display_results(results)

    elif choice == "3":
        results = engine.tfidf_search(query)

        print("\nResults:\n")

        if not results:
            print("No documents found.")
        else:
            for doc_id, score in results:
                print(f"doc_{doc_id}    score: {score:.2f}")

    elif choice == "4":
        results = engine.bm25_search(query)

        print("\nResults:\n")

        if not results:
            print("No documents found.")
        else:
            for doc_id, score in results:
                print(f"doc_{doc_id}    score: {score:.2f}")

    else:
        print("Invalid choice! Please enter a number between 1 and 4.")