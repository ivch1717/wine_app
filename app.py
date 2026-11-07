import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

BASE_DIR = Path(__file__).resolve().parent
LABELS = {
    "fixed_acidity": "Фиксированная кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный диоксид серы",
    "total_sulfur_dioxide": "Общий диоксид серы",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь",
    "is_red": "Красное вино",
}


@st.cache_resource
def load_model():
    # Общий объект модели. Обучение внутри приложения не выполняется.
    return joblib.load(BASE_DIR / "model.joblib")


@st.cache_data
def load_test():
    return pd.read_csv(BASE_DIR / "test.csv")


@st.cache_data
def predict_proba_cached(_model, X):
    # Порог не входит в аргументы, поэтому слайдер не пересчитывает вероятности.
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def read_uploaded_csv(contents):
    return pd.read_csv(io.BytesIO(contents))


def prepare_batch(data, feature_names):
    missing = [name for name in feature_names if name not in data.columns]
    if missing:
        raise ValueError("В CSV не хватает столбцов: " + ", ".join(missing))
    if data.empty:
        raise ValueError("В CSV нет строк с данными.")
    try:
        X = data.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("Признаки вина должны содержать числа, а не текст.") from error
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("В признаках есть пропуски или бесконечные значения.")
    if not X["is_red"].isin([0, 1]).all():
        raise ValueError("Столбец is_red должен содержать только 0 (белое) или 1 (красное).")
    return X


st.set_page_config(page_title="Оценка качества вина", page_icon="🍷", layout="wide")
st.title("Оценка качества вина")
st.caption("Хорошее вино: экспертная оценка не ниже 7. Модель оценивает его по лабораторным признакам.")

model = load_model()
test = load_test()
config = json.loads((BASE_DIR / "config.json").read_text(encoding="utf-8"))
features = list(model.feature_names_in_)

threshold = st.sidebar.slider(
    "Порог рекомендации", min_value=0.0, max_value=1.0,
    value=float(config["threshold"]), step=0.01, key="threshold",
)
st.sidebar.caption("Вино попадает в подборку, если его вероятность не ниже порога. Порог общий для всех вкладок.")

tab_single, tab_metrics, tab_batch, tab_weights = st.tabs([
    "Оценка одного вина", "Модельные метрики",
    "Пакетное предсказание", "Что выучила модель",
])

with tab_single:
    st.subheader("Введите характеристики вина")
    inputs = {}
    columns = st.columns(2)
    for i, feature in enumerate(features):
        column = columns[i % 2]
        if feature == "is_red":
            wine_type = column.radio(
                "Тип вина", ["Белое", "Красное"],
                index=int(test["is_red"].median()), horizontal=True,
            )
            inputs[feature] = int(wine_type == "Красное")
        else:
            steps = {"density": 0.0001, "chlorides": 0.001}
            inputs[feature] = column.number_input(
                LABELS.get(feature, feature),
                min_value=float(test[feature].min()),
                max_value=float(test[feature].max()),
                value=float(test[feature].median()),
                step=steps.get(feature, 0.01), format="%.4f", key=f"input_{feature}",
            )
    X_input = pd.DataFrame([inputs], columns=features)
    probability = float(predict_proba_cached(model, X_input)[0])
    st.metric("Вероятность хорошего вина", f"{probability:.1%}")
    if probability >= threshold:
        st.success("Модель рекомендует вино в подборку")
    else:
        st.warning("Модель не рекомендует вино в подборку")

with tab_metrics:
    st.subheader("Качество на тестовой выборке")
    X_test = test.loc[:, features]
    y_test = test["target"]
    probabilities = predict_proba_cached(model, X_test)
    predictions = (probabilities >= threshold).astype(int)
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{precision:.3f}")
    col2.metric("Recall", f"{recall:.3f}")
    col3.metric("F1", f"{f1:.3f}")
    st.caption(f"В подборку попало {int(predictions.sum())} из {len(predictions)} вин при пороге {threshold:.2f}.")

    cm = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=["Не хорошее", "Хорошее"]).plot(
        ax=ax_cm, colorbar=False, cmap="Blues",
    )
    ax_cm.set_xlabel("Предсказанный класс")
    ax_cm.set_ylabel("Реальный класс")
    fig_cm.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    fpr, tpr, _ = metrics.roc_curve(y_test, probabilities)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, probabilities)
    point_fpr = fp / (fp + tn) if fp + tn else 0.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(fpr, tpr, label="ROC-кривая")
    axes[0].scatter(point_fpr, recall, color="red", zorder=3, label=f"Порог {threshold:.2f}")
    axes[0].set(title="ROC-кривая", xlabel="FPR", ylabel="TPR", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend()
    axes[1].plot(pr_recall, pr_precision, label="PR-кривая")
    axes[1].axhline(y_test.mean(), color="gray", linestyle="--", label="Случайный отбор")
    axes[1].scatter(recall, precision, color="red", zorder=3, label=f"Порог {threshold:.2f}")
    axes[1].set(title="PR-кривая", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.info(
        "При увеличении порога подборка становится меньше: меньше посредственных вин попадает в неё, "
        "но больше хороших вин пропускается. При уменьшении порога подборка расширяется: "
        "больше хороших вин находится, но растёт и число ошибочных рекомендаций."
    )

with tab_batch:
    st.subheader("Загрузите CSV с характеристиками вин")
    st.write("Нужны те же 12 признаков, что и для одного вина. Столбец target не требуется.")
    with st.expander("Названия обязательных столбцов"):
        st.code(", ".join(features), language=None)
    uploaded_file = st.file_uploader("CSV-файл", type=["csv"], key="batch_csv")
    if uploaded_file is not None:
        try:
            batch_data = read_uploaded_csv(uploaded_file.getvalue())
            batch_X = prepare_batch(batch_data, features)
            batch_probabilities = predict_proba_cached(model, batch_X)
            batch_predictions = (batch_probabilities >= threshold).astype(int)
            batch_result = batch_data.copy()
            batch_result["probability_good"] = batch_probabilities
            batch_result["prediction"] = batch_predictions
            st.caption(f"Порог {threshold:.2f}. Рекомендовано {int(batch_predictions.sum())} из {len(batch_predictions)} вин.")
            st.dataframe(batch_result, hide_index=True)
            st.download_button(
                "Скачать прогнозы CSV",
                data=batch_result.to_csv(index=False).encode("utf-8-sig"),
                file_name="wine_predictions.csv", mime="text/csv",
            )
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as error:
            st.error(f"Не удалось обработать файл. {error}")

with tab_weights:
    st.subheader("Коэффициенты логистической регрессии")
    coefficients = pd.Series(model.steps[-1][1].coef_[0], index=features).sort_values()
    names = [LABELS.get(name, name) for name in coefficients.index]
    colors = ["#c44e52" if value < 0 else "#228b67" for value in coefficients]
    fig_weights, ax_weights = plt.subplots(figsize=(10, 6))
    ax_weights.barh(names, coefficients.to_numpy(), color=colors)
    ax_weights.axvline(0, color="black", linewidth=0.8)
    ax_weights.set_xlabel("Коэффициент после масштабирования")
    ax_weights.set_title("Как признаки связаны с решением модели")
    fig_weights.tight_layout()
    st.pyplot(fig_weights)
    plt.close(fig_weights)
    strongest_for = coefficients.nlargest(3).index
    strongest_against = coefficients.nsmallest(3).index
    st.write("Сильнее всего за хорошее вино: " + ", ".join(LABELS.get(name, name) for name in strongest_for) + ".")
    st.write("Сильнее всего против: " + ", ".join(LABELS.get(name, name) for name in strongest_against) + ".")
    st.caption(
        "Положительный коэффициент повышает оценку хорошего вина, отрицательный понижает её "
        "при фиксированных остальных признаках."
    )
