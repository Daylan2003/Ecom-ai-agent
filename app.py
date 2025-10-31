import pandas as pd
import numpy as np
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("products.csv")

@st.cache_data
#This function loads data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("products.csv not found in the working directory")
    df = pd.read_csv(path)
    needed = {"product_id","product_name","category","price","rating","sales_volume","description"}

    #If one of the needed columns is not present it is labelled as missing
    missing = needed - set(df.columns)

    #If there are missing values it is replaced with an empty string
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["product_name"] = df["product_name"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["text"] = (df["product_name"].str.strip() + " " + df["description"].str.strip()).str.lower()
    return df

df = load_data(DATA_PATH)

@st.cache_resource




#This function builds the vectorizer
#It converts a  collection of product descriptions into a numerical TF-IDF feature vectors use in AI models
#First it creates a vector that ignores rare words
#Then it removes common English Words
#It includes both single words and double word phrases
#It learns the vocabulary and computes each product's TF-IDF representation
def build_vectorizer(corpus: pd.Series):
    #Creates the IDF vectorizer with specified parameters
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(corpus.values)
    return vec, X

vec, X = build_vectorizer(df["text"])

#Searches the product database for a given product name.
#It checks for an exact match and resturns it' index if not found
#Otherwise it looks for partial matches containing input text.
#IF neither search succeeds it returns None
def find_index_by_name(name: str) -> int | None:
    name_l = name.strip().lower()
    exact = df.index[df["product_name"].str.lower() == name_l]
    if len(exact) > 0:
        return int(exact[0])
    mask = df["product_name"].str.lower().str.contains(name_l)
    cand = df.index[mask]
    if len(cand) > 0:
        return int(cand[0])
    return None

#This produces a list of products most similar to a given product based on cosine similarity of their TF-IDF vectors.
#It first retrieves the targest product index, computes similarity scores against all others, filters bycategory if requested, sorts by similarity and returns the top k matching products along woth the metadata and similarity scores
def recommend(product_name: str, k: int = 5, same_category: bool = False) -> pd.DataFrame:
    idx = find_index_by_name(product_name)
    if idx is None:
        return pd.DataFrame(columns=["product_id","product_name","category","price","rating","sales_volume","similarity"])
    sims = cosine_similarity(X[idx], X).ravel()
    if same_category:
        cat = df.loc[idx, "category"]
        eligible = df.index[(df["category"] == cat) & (df.index != idx)]
    else:
        eligible = df.index[df.index != idx]
    order = np.argsort(-sims[eligible])
    top_idx = eligible[order][:k]
    out = df.loc[top_idx, ["product_id","product_name","category","price","rating","sales_volume"]].copy()
    out["similarity"] = np.round(sims[top_idx], 3)
    return out.reset_index(drop=True)


#Defines a wrapper around a linear regression model for price suggestion
class DynamicPricingModel:
    def __init__(self):
        self.model = LinearRegression()
        self.trained = False
        self.features = ["price","rating","sales_volume"]
        self.category_dummies: list[str] = []
        self.model_mae: float | None = None
        self.baseline_mae: float | None = None

    def _prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        data = raw.copy()
        if "optimal_price" not in data.columns:
            raise ValueError("Column 'optimal_price' missing for supervised pricing.")
        data = pd.get_dummies(data, columns=["category"], drop_first=True)
        self.category_dummies = [c for c in data.columns if c.startswith("category_")]
        return data

    def fit(self, raw: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> None:
        data = self._prepare(raw)
        X = data[self.features + self.category_dummies]
        y = data["optimal_price"]
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=test_size, random_state=seed)
        self.model.fit(X_tr, y_tr)
        y_pred = self.model.predict(X_va)
        self.model_mae = mean_absolute_error(y_va, y_pred)
        self.baseline_mae = mean_absolute_error(y_va, X_va["price"])
        self.trained = True

    def _row_to_X(self, row: dict) -> pd.DataFrame:
        base = {f: row.get(f, 0.0) for f in self.features}
        for d in self.category_dummies:
            base[d] = 1.0 if d == f"category_{row.get('category')}" else 0.0
        return pd.DataFrame([base])

    def predict(self, row: dict) -> float:
        if not self.trained:
            raise RuntimeError("DynamicPricingModel not trained yet.")
        X_one = self._row_to_X(row)
        return float(self.model.predict(X_one)[0])



@st.cache_resource
#Creates and fits the pricing model
def build_pricing_model(data: pd.DataFrame) -> DynamicPricingModel:
    model = DynamicPricingModel()

    #trains on the entire dataset
    model.fit(data)
    return model

pricing_model = build_pricing_model(df)



#After here the steamlit app is built


#Page icon title and layout
st.set_page_config(page_title="E-commerce AI Agent", page_icon="🤖", layout="wide")
st.title("E-commerce AI Agent: Recommendations + Dynamic Pricing")

#For sidebar
with st.sidebar:
    st.header("Model metrics")
    st.metric("Pricing MAE (model)", f"{pricing_model.model_mae:.2f}")
    st.metric("Pricing MAE (baseline=price)", f"{pricing_model.baseline_mae:.2f}")
    st.caption("Lower is better. Baseline predicts the current price.")

#Creates 2 tabs
tab1, tab2 = st.tabs(["Recommend similar products", "Dynamic price suggestion"])

#tab1 is the recommender 
with tab1:
    st.subheader("Find similar items")
    #Three columns
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        product = st.selectbox("Product", sorted(df["product_name"].unique()))
    with c2:
        #K is a ML term for number of nearest neighbours
        k = st.slider("Top-K", 3, 10, 5, 1)
    with c3:
        same_cat = st.toggle("Same category only", False)
    if st.button("Recommend"):
        try:
            #Calls the recommended function with the correct UI inputs
            results = recommend(product, k=k, same_category=same_cat)
            if results.empty:
                st.warning("No recommendations found. Try disabling the category filter or choose another product.")
            else:
                st.dataframe(results, use_container_width=True)
                if not same_cat:
                    anchor_idx = find_index_by_name(product)
                    pct = 100.0 * (results["category"] == df.loc[anchor_idx, "category"]).mean()
                    st.caption(f"Sanity check: {pct:.0f}% of results share the input category.")
        except Exception as e:
            st.error(str(e))

#Tab2 is dynamic pricing
with tab2:
    st.subheader("Suggest an adjusted price")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        category = st.selectbox("Category", sorted(df["category"].unique()))
    with c2:
        default_price = float(np.round(df["price"].median(), 2))
        price = st.number_input("Current price", min_value=1.0, max_value=1000.0,
                                value=default_price, step=0.5, format="%.2f")
    with c3:
        default_rating = float(np.round(df["rating"].median(), 2))
        rating = st.slider("Rating", 1.0, 5.0, default_rating, 0.1)
    with c4:
        default_sales = int(df["sales_volume"].median())
        sales = st.number_input("Sales volume", min_value=0, max_value=100000,
                                value=default_sales, step=10)

    #Button for suggested price
    if st.button("Suggest price"):
        try:
            row = {"category": category, "price": price, "rating": rating, "sales_volume": int(sales)}
            pred = pricing_model.predict(row)
            delta_pct = 100.0 * (pred - price) / price
            st.success(f"Suggested price: ${pred:.2f}")
            st.write(f"Delta vs current: {delta_pct:+.1f}%")
        except Exception as e:
            st.error(str(e))

st.divider()
st.caption("Data are synthetic. Replace products.csv with real catalog data to demo on actual SKUs. "
           "The pricing model is a simple linear baseline for interview purposes.")
