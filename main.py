from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
import time
import requests
from requests.models import PreparedRequest
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.analyzer import Analyzer
from janome.tokenfilter import CompoundNounFilter, POSStopFilter, POSKeepFilter
from janome.charfilter import UnicodeNormalizeCharFilter, RegexReplaceCharFilter
import pickle
from pprint import pprint


@dataclass(eq=True)
class Document:
    keyword: str
    text: str

    def __init__(self, keyword: str):
        self.raw_keyword = keyword
        self.keyword = self._preprocess_text(self.raw_keyword)
        # self.text = self.get_googled_text(keyword)
        self.text = self.get_wikipedia_article_text(self.keyword)

    @classmethod
    def get_googled_text(cls, keywords: str) -> str:
        time.sleep(0.2)
        params = {
            "q": keywords,
            # 'hl': 'ja', # 表示言語
            # 'lr': 'lang_ja', # 検索言語
            "num": 99,  # 1ページあたり検索結果件数
            "filter": 1,  # 類似ページの除外ON
            "pwd": False,  # パーソナライズ検索の無効化
        }
        url = cls.unparse_url("https://www.google.com/search", params)
        print(url)
        text = cls._get_page_contents(url)
        text = cls._preprocess_text(text)
        return text

    @classmethod
    def get_wikipedia_article_text(cls, keyword: str) -> str:
        time.sleep(0.2)
        article_ja = cls.get_wikipedia_article_by_country(keyword, "ja")
        if not article_ja:
            article_en = cls.get_wikipedia_article_by_country(keyword, "en")
            if not article_en:
                raise ValueError("No wikipedia article found.")
            else:
                url = article_en
        else:
            url = article_ja

        print(url)
        text = cls._get_page_contents(url)
        text = cls._preprocess_text(text)
        return text

    @classmethod
    def get_wikipedia_article_by_country(cls, keyword: str, country_code: Literal["ja", "en"]):
        time.sleep(0.2)
        params = {
            "q": keyword,
            "limit": 1,
        }
        url = cls.unparse_url(
            f"https://{country_code}.wikipedia.org/w/rest.php/v1/search/page", params
        )
        # print(url)
        response = requests.get(url).json()
        if response.get("pages"):
            article_id = response["pages"][0]["id"]
            params = {
                "curid": article_id,
            }
            url = cls.unparse_url("https://ja.wikipedia.org/w/index.php", params)
            return url
        else:
            return None

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """NFKC標準化、分かち書き、ストップワード除去"""
        char_filters = [
            UnicodeNormalizeCharFilter(),
            RegexReplaceCharFilter(r"\d+", "0"),  # 数字を全て0に置換
        ]
        token_filters = [
            CompoundNounFilter(),
            # POSStopFilter(["記号", "助詞", "助動詞"])
            POSKeepFilter(["名詞", "動詞", "形容詞", "副詞"])
        ]
        analyzer = Analyzer(char_filters=char_filters, token_filters=token_filters)
        tokens = analyzer.analyze(text)
        processed_text = " ".join([t.surface for t in tokens])
        return processed_text

    @staticmethod
    def unparse_url(base_url: str, params: dict):
        req = PreparedRequest()
        req.prepare_url(base_url, params)
        return req.url

    @staticmethod
    def _get_page_contents(url: str) -> str:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.text


@dataclass
class Documents:
    googled_docs: List[Document]

    def __init__(self, keywords: List[str] = []):
        assert isinstance(keywords, list), "keywords must be a list of strings."
        self.keywords = keywords
        self.googled_docs: List[Document] = None

    def __iter__(self):
        yield from self.googled_docs

    def __len__(self):
        return len(self.googled_docs)

    def save(self, path: str):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self.googled_docs, f)

    def load(self, path: Optional[str] = None):
        if path is None:
            self.googled_docs: List[Document] = self.get_googled_texts(self.keywords)
        else:
            self.googled_docs: List[Document] = self._load(path)
        return self

    @staticmethod
    def _load(path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data, list), "data must be a list of Documents."
        assert isinstance(data[0], Document), "data must be a list of Documents."
        return data

    @staticmethod
    def get_googled_texts(keywords: List[str]) -> List[Document]:
        res = []
        for s in keywords:
            res.append(Document(s))
        return res

    def list_texts(self) -> List[str]:
        return [doc.text for doc in self.googled_docs]

    def list_keywords(self) -> List[str]:
        return [doc.keyword for doc in self.googled_docs]

    def get_document(self, keyword: str = None, text: str = None):
        if keyword is None and text is None:
            raise ValueError("Either `keyword` or `text` must be specified.")
        elif keyword:
            return [doc for doc in self.googled_docs if doc.keyword == keyword][0]
        else:
            return [doc for doc in self.googled_docs if doc.text == text][0]


class SearchEngine_BM25:
    def __init__(self, b=0.75, k1=1.2):
        self.vectorizer = TfidfVectorizer(
            smooth_idf=False,
            max_df=0.8,
        )
        self.b = b
        self.k1 = k1

    def fit(self, docs: Documents):
        self.docs = docs
        self.vectorizer.fit(self.docs.list_texts())

        # TFIDF行列作成、平均文書長を計算
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.doc_term_matrix = self.vectorizer.transform(self.docs.list_texts())
        self.average_document_length = self.doc_term_matrix.sum(1).mean()

    def error_before_fit(func):
        def wrapper(self, *args, **kwargs):
            if self.doc_term_matrix is None:
                raise ValueError("`fit` function must be called beforehand.")
            return func(self, *args, **kwargs)
        return wrapper

    @error_before_fit
    def transform(self, query: str):
        """クエリと文書間のBM25指標を計算"""
        b, k1 = self.b, self.k1

        # クエリ単語のTFIDF
        q = self.vectorizer.transform([query])
        f = self.doc_term_matrix.tocsc()[:, q.indices]

        # 相対文書長
        doc_length = self.doc_term_matrix.sum(1).A1
        relative_doc_length = doc_length / self.average_document_length

        # bm25式
        denom = f + (k1 * (1 - b + b * relative_doc_length))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.0
        numer = f.multiply(np.broadcast_to(idf, f.shape)) * (k1 + 1)
        bm25 = (numer / denom).sum(1).A1
        return bm25

    @error_before_fit
    def rank_document(self, query, n: int = 10) -> List[Tuple[float, Document]]:
        similarity = self.transform(query)
        res = sorted(zip(similarity, self.docs), key=lambda x: x[0], reverse=True)
        return res[:n]

    @error_before_fit
    def get_shared_word_importances(self, query: str, doc: Document, max_results: int = 10):
        """クエリと文書の両方に出現する単語とTFIDF値の大きい順に返す"""
        query_mtx = self.vectorizer.transform([query]).toarray().flatten()
        doc_mtx = self.vectorizer.transform([doc.text]).toarray().flatten()
        feats = self.feature_names

        # クエリと文書の両方でTFIDFが非ゼロの要素を足し合わせ降順にソート
        intersect = np.where((query_mtx!=0) & (doc_mtx!=0), query_mtx+doc_mtx, 0)
        non_zero_cnt = (intersect > 0).sum()
        idx = np.argsort(intersect, axis=None)[::-1][:non_zero_cnt]
        res = list(zip(feats[idx], intersect[idx]))
        return res[:max_results]

    @staticmethod
    def intersect_lists(A, B):
        intersected_list = []
        for item in A:
            if item in B:
                intersected_list.append(item)
        return intersected_list


def extract_closest_document_and_reason(
    query: str, search_target: Documents, top_n: int = 5
):
    # BM25初期化
    bm = SearchEngine_BM25()
    bm.fit(search_target)
    print("query:", query)

    res = []
    # 類似度が最も高い文書を取得
    for score, closest_doc in bm.rank_document(query, top_n):
        # クエリとの重要な共通単語を取得
        important_feats = bm.get_shared_word_importances(query, closest_doc)
        res.append((closest_doc.keyword, score, important_feats))

    return res


if __name__ == "__main__":
    # print("start")
    # documents = Documents([
    #     "インターステラ",
    #     "ロードオブザリング",
    #     "ターミネーター",
    #     "コマンドー",
    #     "ハリーポッター",
    # ]).load()
    # documents.save("documents.pkl")
    # print("saved")

    # query = Document("アーノルド・シュワルツェネッガー")
    # res = extract_closest_document_and_reason(query, documents)

    # pprint(res)
    # del documents

    # print("refresh documents")
    # query = Document("アーノルド・シュワルツェネッガー")
    # documents = Documents().load("documents.pkl")
    # pprint(documents.list_keywords())
    # res = extract_closest_document_and_reason(query, documents)
    # pprint(res)

    # print("start")
    # # documents = Documents(
    # #     [
    # #         "西瓜",
    # #         "梨",
    # #         "桃",
    # #         "パン",
    # #         "コカコーラ",
    # #     ]
    # # ).load()
    # # documents.save("fruit.pkl")
    # documents = Documents().load("fruit.pkl")
    # query = Document("赤い果物")
    # res = extract_closest_document_and_reason(query.keyword, documents)

    # pprint(res)

    # print("start")
    # documents = Documents(
    #     [
    #         "蓮花",
    #         "ひまわり",
    #         "百合",
    #         "カーネーション",
    #         "パクチー",
    #         "きゅうり"
    #     ]
    # ).load()
    # documents.save("vege.pkl")
    # documents = Documents().load("vege.pkl")
    # query = Document("花 黄色 太陽")
    # res = extract_closest_document_and_reason(query.keyword, documents)
    # pprint(res)

    print("start")
    documents = Documents(
        [
            "眼鏡",
            "コンタクトレンズ",
            "万年筆",
            "鉛筆",
            "定規",
            "自転車"
        ]
    ).load()
    query = Document("文房具")
    res = extract_closest_document_and_reason(query.keyword, documents)
    pprint(res)