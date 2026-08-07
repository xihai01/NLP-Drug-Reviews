import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
import math
import numpy as np

class BM25Index:
    """
    BM25Index builds a BM25 search index over a list of documents and supports
    scoring queries to retrieve the most relevant documents.
    """

    def __init__(self, docs, k1=1.2, b=0.75):
        """
        Initialize the BM25 index and build internal structures.

        Parameters
        ----------
        docs : list[dict]
            Each document is a dict with keys:
              - 'id' (str/int): unique document identifier
              - 'text' (str): document content
        k1 : float, optional
            BM25 term-frequency saturation parameter (default: 1.2).
        b : float, optional
            BM25 length normalization parameter in [0, 1] (default: 0.75).

        Builds:
        -------
        - Document lengths and average length
        - Inverted index (term -> postings)
        - Document frequency per term
        """
        self.k1 = k1
        self.b = b
        self.docs = docs  # list[Dict[str, str]]
        self.N = len(docs)
        self.doc_len = {} # {id -> num}
        self.avgdl = 0.0
        self.df = defaultdict(int)
        self.inv = defaultdict(list)  # term -> list of (doc_id, tf)
        # Built once here instead of per _tokenize() call - loading these
        # from NLTK repeatedly (once per document) was the main indexing
        # bottleneck.
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words('english'))
        self._build()

    def _tokenize(self, text):
        """
        Normalize and tokenize text.

        Steps:
        - Lowercase the text
        - Remove punctuation
        - Lemmatize tokens (simple lemmatization)
        - Remove stopwords

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        list[str]
            List of normalized tokens.
        """
        lemmatizer = self._lemmatizer
        stop_words = self._stop_words
        # convert to lowercase
        text = text.lower()
        # remove newlines
        text = text.replace('\n', ' ')
        # keep only letters (removes punctuation, digits, special chars)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # tokenize
        tokens = word_tokenize(text)
        # lemmatize and remove stopwords
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
        return tokens

    def _build(self):
        """
        Build the BM25 index over the provided documents.

        Tasks:
        - Tokenize each document
        - Compute document length and average length
        - Build inverted index (self.inv) and document frequency (self.df) map
        """
        total = 0.0
        for doc in self.docs:
            term_list = self._tokenize(doc["text"])
            term_counts = Counter(term_list)
            # store doc length
            self.doc_len[doc["id"]] = len(term_list)
            # tf/df calculation
            for term, count in term_counts.items():
                # store avg doc length
                total += count
                # tf: raw term frequency. BM25's saturation curve (applied in
                # score()) already handles diminishing returns for repeated
                # terms - pre-dampening it here with log10 would saturate twice.
                self.inv[term].append({"doc_id": doc["id"], "tf": count})
                # df
                self.df[term] += 1
        self.avgdl = total/self.N

    def _idf(self, term):
        """
        Compute BM25 inverse document frequency (IDF) for a term.
        Uses document frequency (self.df) map and self.N
        n_q is the number of documents containing the term
        Formula:
        idf(term) = ln( (N - n_q + 0.5) / (n_q + 0.5) + 1 )

        Parameters
        ----------
        term : str
            Query term.


        Returns
        -------
        float
            IDF weight for the term.
        """
        return math.log1p((self.N - self.df[term] + 0.5)/(self.df[term] + 0.5))


    def score(self, query, top_k=10):
        """
        Score documents against a multi-term query using BM25 and return top-k results.

        BM25 Formula
        ------------
        For query Q = {q1, q2, ..., qm}, the BM25 score of document d is:

            BM25(d, Q) = sum_{t in Q} IDF(t) * [ TF(t, d) * (k1 + 1) ] /
                                       [ TF(t, d) + k1 * (1 - b + b * |d| / avgdl) ]

        Where:
        - TF(t, d) : raw term frequency of term t in document d
        - |d|      : document length (number of tokens in d)
        - avgdl    : average document length in the corpus
        - k1       : term-frequency saturation parameter (typically 1.2–2.0)
        - b        : length normalization parameter in [0, 1] (typically 0.75)
        - IDF(t)   : inverse document frequency as defined in _idf()

        Steps
        -----
        - Tokenize the query
        - For each query term:
            * Get postings (doc_id, tf) from the inverted index
            * Compute IDF via _idf(term)
            * Accumulate BM25 contribution for each document containing the term
        - Sort documents by total score in descending order
        - Return the top_k (doc_id, score) pairs

        Parameters
        ----------
        query : str
            Raw query string.
        top_k : int, optional
            Number of top results to return (default: 10).

        Returns
        -------
        list[tuple[str, float]]
            Sorted (doc_id, score) pairs in descending order.
        """
        # store accumulated scores
        score_per_doc = defaultdict(float) # {id -> acc. score}
        query_list = self._tokenize(query)
        for term in query_list:
            # compute IDF
            idf = self._idf(term)
            for posting in self.inv[term]:
                bm25 = idf * (posting["tf"] * (self.k1 + 1)) / (posting["tf"] + self.k1 * (1 - self.b + self.b * (self.doc_len[posting["doc_id"]] / self.avgdl)))
                score_per_doc[posting["doc_id"]] += bm25

        # return a list of top k documents sorted from high to low scores
        res = Counter(score_per_doc)
        return res.most_common(top_k)

    def info(self):
        """
        Return a short summary of the index configuration.

        Returns
        -------
        str
            Example: "BM25Index(k1=1.2, b=0.75, N=100)".
        """
        res = f"BM25Index(k1={self.k1}, b={self.b}, N={self.N})"
        return res

    """
    Processes TREC style files.

    Returns
    -------
    List of dictionaries consisting of id and text
    """
    def load_text_documents(self, starts_with, file_path):
        documents = []
        current_doc = None

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                cleaned_line = line.strip()

                # Check if this line is the start of a new text document
                if cleaned_line.startswith(starts_with):
                    # Save the previous document before starting the new one
                    if current_doc is not None:
                        documents.append(current_doc)

                    # Split the header info (e.g., "*TEXT 017 01/04/63 PAGE 020")
                    header_parts = cleaned_line.split()

                    current_doc = {
                        "id": header_parts[1] if len(header_parts) > 1 else None,
                        "text": []
                    }
                else:
                    # Add text lines to the active document content
                    if current_doc is not None and cleaned_line:
                        current_doc["text"].append(cleaned_line)

            # Don't forget to append the final document in the file
            if current_doc is not None:
                documents.append(current_doc)

        # Clean up content list into a single paragraph string for each document
        for doc in documents:
            doc["text"] = " ".join(doc["text"])

        return documents

    @staticmethod
    def load_druglib_csv(path, text_col="full_review", id_col="Unnamed: 0"):
        """
        Load a druglib_processed_*.csv file (UCI/Kaggle drug review dataset,
        Druglib.com variant) into BM25Index's doc format.

        Parameters
        ----------
        path : str
            Path to a druglib_processed_*.csv file (test, train, or combined).
        text_col : str, optional
            Which column to index: 'full_review' (raw text, recommended so
            BM25Index's own _tokenize() controls preprocessing) or
            'processed_review' (already lemmatized/stopword-stripped).
        id_col : str, optional
            Unique row identifier column (default: 'Unnamed: 0', the original
            row index preserved from the raw dataset).

        Returns
        -------
        list[dict]
            Each entry has "id", "text", and a "meta" dict (drug name,
            condition, rating, effectiveness) carried along for display
            purposes. Note: BM25Index itself only reads "id" and "text".
        """
        df = pd.read_csv(path)
        df = df.dropna(subset=[text_col])

        documents = []
        for _, row in df.iterrows():
            documents.append({
                "id": str(row[id_col]),
                "text": str(row[text_col]),
                "meta": {
                    "drug": row["urlDrugName"],
                    "condition": row["condition"],
                    "rating": row["rating"],
                    "effectiveness": row["effectiveness"],
                },
            })
        return documents

    def load_simple_queries(self, path):
        """
        Load queries from a plain '<qid>\\t<query text>' file (one query per
        line) - simpler than the TREC *FIND format handled by
        load_text_documents, useful for datasets (like drug reviews) that
        don't come in TREC format.

        Returns
        -------
        list[dict]
            Each entry has "id" (str) and "text" (str).
        """
        queries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                qid, text = line.split("\t", 1)
                queries.append({"id": qid.strip(), "text": text.strip()})
        return queries

    def run_queries(self, query_file="time.QUE", output_file="time/run.txt",
                    top_k=100, run_tag="BM25", base_dir="time", query_format="trec"):
        """
        Read queries from a file (default: 'time.QUE'), retrieve documents for each query
        using BM25 scoring, and save the results into 'run.txt'.

        Purpose
        -------
        Automate evaluation by running a batch of queries and emitting a run file
        suitable for analysis or submission (e.g., TREC-style format).

        Output File Format (run.txt)
        ----------------------------
        The output is a plain text file that lists the top-k retrieved documents
        per query, commonly in a TREC run format (one line per result):

            <query_id> Q0 <doc_id> <rank> <score> <run_tag>

        Example:
            Q1 Q0 DOC_123 1 7.532 BM25
            Q1 Q0 DOC_045 2 7.228 BM25
            Q2 Q0 DOC_301 1 5.114 BM25

        Steps
        -----
        - Read queries from `query_file`:
        - For each query:
            * Call `self.score(qtext, top_k=top_k)` to get top results.
            * Write results to `output_file` in the format above, using `run_tag` to label the run.
        - Handle edge cases:
            * If a query yields no hits, you may skip or write no lines for that query.
            * Ensure consistent types for `doc_id` (string) and numeric `score`.

        Parameters
        ----------
        query_file : str, optional
            Path to the file containing queries (default: 'time.QUE').
        output_file : str, optional
            Path to save the run output (default: 'run.txt').
        top_k : int, optional
            Number of top documents to retrieve per query (default: 100).
        run_tag : str, optional
            A short label identifying this run (default: 'BM25').
        base_dir : str, optional
            Directory prepended to `query_file` (default: 'time', preserving
            the original TIME-collection layout). Pass "" or None to treat
            `query_file` as a full/relative path as-is (e.g. for the drug
            review dataset's query file, which doesn't live under 'time/').
        query_format : str, optional
            'trec' (default) parses `query_file` with load_text_documents's
            *FIND TREC format. 'simple' parses a plain '<qid>\\t<text>' file
            via load_simple_queries - use this for the drug review dataset.

        Returns
        -------
        None
            Writes results to `output_file`.

        Notes
        -----
        - Ensure your document IDs are strings when writing the run file.
        - Overwrites `output_file` rather than appending, so re-running
          doesn't duplicate results from a previous run.
        """
        path = f"{base_dir}/{str.upper(query_file)}" if base_dir else query_file
        if query_format == "trec":
            queries = self.load_text_documents("*FIND", path)
        elif query_format == "simple":
            # simple format shouldn't be uppercased like a TREC filename
            path = f"{base_dir}/{query_file}" if base_dir else query_file
            queries = self.load_simple_queries(path)
        else:
            raise ValueError(f"Unknown query_format: {query_format!r} (expected 'trec' or 'simple')")

        with open(output_file, "w", encoding="utf-8") as file:
            for query in queries:
                results = self.score(query["text"], top_k)
                for rank, (doc_id, score) in enumerate(results, start=1):
                    file.write(f"{query['id']} Q0 {doc_id} {rank} {score} {run_tag}\n")

    def parse_rel_fn(self, rel_file="time.REL", base_dir="time"):
        path = f"{base_dir}/{rel_file}" if base_dir else rel_file
        data_dict = {}
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue
                key = int(parts[0])
                values = [int(num) for num in parts[1:]]
                data_dict[key] = values
        return data_dict

    def parse_run_fn(self, run_file="run.txt", base_dir="time"):
        data_dict = defaultdict(list)
        path = f"{base_dir}/{run_file}" if base_dir else run_file
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue

                # 1. Extract the key (the first item) as an integer
                key = int(parts[0])

                # 2. Extract the doc id
                doc_id = int(parts[2])

                # 3. Append the extracted data as a tuple or sub-list into the main key
                data_dict[key].append(doc_id)
        return dict(data_dict)

    def _precision(self, k, rel_i):
        """
        1) Precision@k:
            P@k(q) = (1/k) * sum_{i=1..k} rel_i
            (If fewer than k documents were retrieved, use the number retrieved as k̂,
             or keep k fixed; state your chosen convention.)
        """
        if k == 0: return 0
        if len(rel_i) < k: k = len(rel_i)
        total = 0.0
        for i in range(k):
            total += rel_i[i]
        return total * (1 / k)

    def _recall(self, r_q, k, rel_i):
        """
        2) Recall@k:
            R@k(q) = (1/R_q) * sum_{i=1..k} rel_i
            (If R_q = 0, define R@k(q) = 0 and note the query has no relevant docs.)
        """
        if r_q == 0: return 0
        if len(rel_i) < k: k = len(rel_i)
        total = 0.0
        for i in range(k):
            total += rel_i[i]
        return total * (1 / r_q)

    def _avg_precision(self, r_q, k, rel_i):
        """
        3) Average Precision@k (AP@k; truncated AP):
            AP@k(q) = (1 / min(R_q, k)) * sum_{i=1..k} [ P@i(q) * rel_i ]
            (Common alternative: divide by R_q. Here we use min(R_q, k) to avoid
             penalizing truncation when k < R_q. )
        """
        if k == 0: return 0
        if len(rel_i) < k: k = len(rel_i)
        total = 0.0
        for i in range(k):
            total += (self._precision(i+1, rel_i) * rel_i[i])
        return total * (1 / min(r_q, k))

    def _reciprocal_rank(self, k, rel_i):
        """
        4) Reciprocal Rank@k (RR@k):
            RR@k(q) = 1 / r*, where r* is the rank of the first relevant doc within top k;
                      if no relevant doc in top k, RR@k(q) = 0.
        """
        if len(rel_i) < k: k = len(rel_i)
        for i in range(k):
            if rel_i[i]: return 1 / (i + 1)
        return 0.0

    def _calculate_ndcg_at_k(self, predictions, k):
        """Calculates nDCG@k for a relevance scale of 0 to 1.

        predictions: list/array of actual relevance scores in ranked order.
        ideal_relevance: list/array of true relevance scores sorted in ideal descending
        order.
        k: top elements cut-off.
        """
        # 1. Take top k elements
        pred_k = np.array(predictions[:k])
        ideal_k = np.ones(len(pred_k))

        # 2. Compute DCG@k
        ranks = np.arange(1, len(pred_k) + 1)
        discounts = np.log2(ranks + 1)
        dcg = np.sum(pred_k / discounts)

        # 3. Compute IDCG@k
        idcg = np.sum(ideal_k / discounts)

        # 4. Handle edge case: if perfect score is 0, return 0 to avoid div by zero
        if idcg == 0:
            return 0.0

        return dcg / idcg

    def eval(self, run_file="run.txt", rel_file="time.REL", k=10, base_dir="time"):
        """
        Evaluate retrieval effectiveness at cutoff k using a run file and relevance file.

        Inputs
        ------
        - run_file (default: 'run.txt'):
            * Assumed TREC-style run file produced by `run_queries`:
              <query_id> Q0 <doc_id> <rank> <score> <run_tag>
            * One line per retrieved document.
            * Query IDs should match those in `rel_file`.

        - rel_file (default: 'time.REL'):
            * Contains relevance assessments (qrels) for each query.


        Metrics (per-query, at cutoff k)
        --------------------------------
        Let:
          • R_q = total number of relevant documents for query q (from rel_file)
          • D_q = ordered list of retrieved doc_ids for q (from run_file)
          • rel_i in {0,1} indicate relevance of the document at rank i (1-based)
            rel_i = 1 if D_q[i] in relevant set for q, else 0
          • k = evaluation cutoff

        1) Precision@k:
            P@k(q) = (1/k) * sum_{i=1..k} rel_i
            (If fewer than k documents were retrieved, use the number retrieved as k̂,
             or keep k fixed; state your chosen convention.)

        2) Recall@k:
            R@k(q) = (1/R_q) * sum_{i=1..k} rel_i
            (If R_q = 0, define R@k(q) = 0 and note the query has no relevant docs.)

        3) Average Precision@k (AP@k; truncated AP):
            AP@k(q) = (1 / min(R_q, k)) * sum_{i=1..k} [ P@i(q) * rel_i ]
            (Common alternative: divide by R_q. Here we use min(R_q, k) to avoid
             penalizing truncation when k < R_q. )

        4) Reciprocal Rank@k (RR@k):
            RR@k(q) = 1 / r*, where r* is the rank of the first relevant doc within top k;
                      if no relevant doc in top k, RR@k(q) = 0.

        Aggregation (macro)
        -------------------
        - MAP@k  = mean over queries of AP@k(q)
        - MRR@k  = mean over queries of RR@k(q)
        - Mean P@k = mean over queries of P@k(q)
        - Mean R@k = mean over queries of R@k(q)

        Steps
        -----
        1) Parse run_file:
           - Build an ordered mapping: qid -> [doc_id_1, doc_id_2, ..., doc_id_k]
             respecting rank order; ignore lines beyond k if desired.
        2) Parse rel_file:
           - If parse_rel_fn is provided, call it to obtain qrels: dict[qid] -> set[doc_id].
           - Else, use the simple parser described above.
        3) For each query q present in either file (intersection or union, per policy):
           - Compute rel_i, P@i for i = 1..k, then P@k, R@k, AP@k, RR@k.
        4) Aggregate macro metrics across queries.
        5) Return a structured report.

        Parameters
        ----------
        run_file : str, optional
            Path to run file (default: 'run.txt').
        rel_file : str, optional
            Path to relevance file (default: 'time.REL').
        k : int, optional
            Evaluation cutoff (default: 10).


        Returns
        -------
        dict
            Example structure:
            {
              "k": k,
              "num_queries": N,
              "per_query": {
                 "Q1": {"P@k": 0.5, "R@k": 0.25, "AP@k": 0.42, "RR@k": 1.0},
                 "Q2": {"P@k": 0.3, "R@k": 0.6,  "AP@k": 0.28, "RR@k": 0.5},
                 ...
              },
              "macro": {"P@k": 0.4, "R@k": 0.45, "MAP@k": 0.35, "MRR@k": 0.62}
            }

    """
        per_query = defaultdict(dict)
        macro = defaultdict(float)
        run_file = self.parse_run_fn(run_file, base_dir=base_dir)
        rel_file = self.parse_rel_fn(rel_file, base_dir=base_dir)
        for q_id, docs in run_file.items():
            # compute the intersection for each query
            rel_i = []
            for doc in docs:
                if doc in rel_file[q_id]:
                    rel_i.append(1)
                else:
                    rel_i.append(0)
            # build the metrics per query
            metrics = {}
            # precision
            p = self._precision(k, rel_i)
            # recall
            r = self._recall(len(rel_file[q_id]), k, rel_i)
            # avg. precision
            ap = self._avg_precision(len(rel_file[q_id]), k, rel_i)
            # reciprocal rank
            rr = self._reciprocal_rank(k, rel_i)
            # ndcg
            ndcg = self._calculate_ndcg_at_k(rel_i, k)
            metrics["P@k"] = p
            metrics["R@k"] = r
            metrics["AP@k"] = ap
            metrics["RR@k"] = rr
            metrics["nDCG@k"] = ndcg
            # aggregations
            macro["P@k"] += p
            macro["R@k"] += r
            macro["MAP@k"] += ap
            macro["MRR@k"] += rr
            macro["MnDCG@k"] += ndcg
            per_query[f"Q{q_id}"] = metrics

        length_of_queries = len(run_file)
        macro["P@k"] /= length_of_queries
        macro["R@k"] /= length_of_queries
        macro["MAP@k"] /= length_of_queries
        macro["MRR@k"] /= length_of_queries
        macro["MnDCG@k"] /= length_of_queries

        return {
            "k": k,
            "num_queries": length_of_queries,
            "per_query": per_query,
            "macro": macro,
        }
