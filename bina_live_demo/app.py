import streamlit as st
import pandas as pd
import joblib
import sys

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

MODEL_PATH = "final_price_model.joblib"


def _patch_sklearn_compat():
    """
    Known scikit-learn cross-version pickle bug: a ColumnTransformer saved
    with one sklearn version can fail to load on another with:
    AttributeError: Can't get attribute '_RemainderColsList' on
    <module 'sklearn.compose._column_transformer'>
    (see scikit-learn/scikit-learn#32090). This restores the missing
    attribute so joblib.load can unpickle the model regardless of the
    installed sklearn version.
    """
    try:
        import sklearn.compose._column_transformer as ct_module

        if not hasattr(ct_module, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            ct_module._RemainderColsList = _RemainderColsList
    except Exception:
        pass


def _iter_sub_estimators(estimator):
    """Recursively yield every fitted sub-estimator inside a Pipeline /
    ColumnTransformer / FeatureUnion, regardless of nesting depth."""
    yield estimator

    # Pipeline
    steps = getattr(estimator, "steps", None)
    if steps:
        for _, sub in steps:
            if sub is not None and sub != "passthrough" and sub != "drop":
                yield from _iter_sub_estimators(sub)

    # ColumnTransformer (fitted -> transformers_, unfitted -> transformers)
    for attr in ("transformers_", "transformers"):
        transformers = getattr(estimator, attr, None)
        if transformers:
            for item in transformers:
                sub = item[1]
                if sub is not None and sub != "passthrough" and sub != "drop":
                    yield from _iter_sub_estimators(sub)
            break  # don't process both attrs if both happen to exist

    # FeatureUnion
    tlist = getattr(estimator, "transformer_list", None)
    if tlist:
        for _, sub in tlist:
            if sub is not None:
                yield from _iter_sub_estimators(sub)


def _fix_loaded_model(model):
    """
    Known scikit-learn cross-version pickle issue: transformers fitted with
    an older/newer sklearn can be missing private attributes the installed
    version now expects (e.g. SimpleImputer._fill_dtype, or similar private
    attrs on other transformers). This walks every nested estimator in the
    pipeline and restores any such missing attributes so the model can run.
    """
    try:
        from sklearn.impute import SimpleImputer

        for est in _iter_sub_estimators(model):
            if isinstance(est, SimpleImputer) and not hasattr(est, "_fill_dtype"):
                stats = getattr(est, "statistics_", None)
                if stats is not None:
                    est._fill_dtype = getattr(stats, "dtype", None)
    except Exception as e:
        st.warning(f"Compatibility fix-up warning: {e}")
    return model


@st.cache_resource
def load_model():
    _patch_sklearn_compat()
    model = joblib.load(MODEL_PATH)
    model = _fix_loaded_model(model)
    return model

st.title("🏠 House Price Predictor (bina.az)")
st.write(
    "This demo uses a **RandomForestRegressor** model trained on bina.az real estate "
    "listings to estimate a property's market price in AZN."
)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error(
        f"Model file not found: `{MODEL_PATH}`. "
        "Please place `final_price_model.joblib` in the same folder as this app.py."
    )

st.subheader("Enter property details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (m²)", min_value=10.0, max_value=2000.0, value=80.0, step=1.0)
    rooms = st.number_input("Number of rooms", min_value=1.0, max_value=15.0, value=3.0, step=1.0)
    views = st.number_input("Views", min_value=0, max_value=100000, value=500, step=10)

with col2:
    floor_current = st.number_input("Floor (current)", min_value=1, max_value=60, value=5, step=1)
    floor_total = st.number_input("Total floors in building", min_value=1, max_value=60, value=12, step=1)

kateqoriya = st.selectbox(
    "Category",
    ["Yeni tikili", "Köhnə tikili", "Ofis", "Həyət evi/Bağ evi", "Villa", "Qaraj", "Torpaq", "Other"],
    help="Yeni tikili = New building, Köhnə tikili = Old building, Ofis = Office, "
         "Həyət evi/Bağ evi = House/Cottage, Torpaq = Land",
)

location_type = st.selectbox(
    "Location type",
    ["District center", "Near metro", "Settlement", "Other", "Unknown"],
)

seller_type = st.selectbox("Seller type", ["Agency", "Individual", "Unknown"])

temir = st.selectbox(
    "Renovation status",
    ["Təmirli", "Orta təmirli", "Təmirsiz", "Az.təmirə ehtiyacı var", "Unknown"],
    help="Təmirli = Renovated, Orta təmirli = Partially renovated, "
         "Təmirsiz = Not renovated, Az.təmirə ehtiyacı var = Needs minor repair",
)

city = st.text_input("City", value="Baku")

st.divider()

if st.button("Estimate price", type="primary", disabled=not model_loaded):
    floor_ratio = round(floor_current / floor_total, 2) if floor_total else None

    input_df = pd.DataFrame([{
        "Sahə_numeric_capped": area,
        "Otaq sayı": rooms,
        "views_capped": views,
        "floor_ratio": floor_ratio,
        "Kateqoriya": kateqoriya,
        "location_type": location_type,
        "seller_type": seller_type,
        "Təmir": temir,
        "city": city,
    }])

    pred = model.predict(input_df)[0]

    st.success(f"### Estimated price: **{pred:,.0f} AZN**")
    st.caption(
        "This is a model-based estimate derived from historical listing data "
        "and may differ from the actual market price."
    )

    with st.expander("Input values sent to the model"):
        st.dataframe(input_df)

st.divider()
st.caption(
    "I built this demo from the house.csv week4 tasks."
)
