import streamlit as st

st.set_page_config(
    page_title="엔진 데이터 회귀분석",
    layout="wide"
)

st.title("엔진 운전 데이터 회귀분석 프로그램")

st.write("""
CSV 또는 Excel 데이터 파일을 불러와 원하는 변수를 선택하고,
선형회귀 및 비선형회귀 분석을 수행하는 데이터 분석 프로그램입니다.
""")

st.subheader("데이터 파일 업로드")

uploaded_file = st.file_uploader(
    "CSV 또는 Excel 파일을 선택하세요.",
    type=["csv", "xlsx"])

import pandas as pd

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.subheader("데이터 확인")
    st.dataframe(data)
    st.subheader("분석 변수 선택")

    numeric_columns = data.select_dtypes(include="number").columns

    x_column = st.selectbox("X축 데이터를 선택하세요.", numeric_columns)
    y_column = st.selectbox("Y축 데이터를 선택하세요.", numeric_columns, index=1)

    st.subheader(f"{x_column}과 {y_column}의 관계")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.scatter(data["RPM"], data["Power"])
    ax.set_xlabel("RPM")
    ax.set_ylabel("Power")
    ax.set_title("RPM vs Power")

    st.pyplot(fig)
    st.subheader("RPM과 출력의 관계")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.scatter(data[x_column], data[y_column])
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"{x_column} vs {y_column}")

    st.pyplot(fig)
    from sklearn.linear_model import LinearRegression

    X = data[[x_column]]
    y = data[y_column]

    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    st.pyplot(fig)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X,y)

    st.subheader("선형회귀 분석 결과")

    st.write(f"회귀식: {y_column} = {slope:.4f} × {x_column} + {intercept:.4f}")
    st.write(f"R² 결정계수: {r2:.4f}")

    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y, model.predict(X))
    rmse = mse ** 0.5
    st.write(f"RMSE 오차: {rmse:.4f}")

    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)

    poly_predicted = poly_model.predict(X_poly)
    poly_r2 = poly_model.score(X_poly, y)

    fig2, ax2 = plt.subplots()

    ax2.scatter(data[x_column], data[y_column], label="Actual Data")
    ax2.plot(data[x_column], poly_predicted, label="Polynomial Regression")

    ax2.set_xlabel(x_column)
    ax2.set_ylabel(y_column)

    ax2.set_title("Nonlinear Regression")
    ax2.legend()

    st.subheader("비선형회귀 분석")
    a = poly_model.coef_[2]
    b = poly_model.coef_[1]
    c = poly_model.intercept_

    st.write(
        f"회귀식: {y_column} = {a:.6f} × {x_column}² "
        f"+ {b:.4f} × {x_column} + {c:.4f}"
    )
    st.pyplot(fig2)

    st.write(f"R² 결정계수: {poly_r2:.4f}")

    poly_mse = mean_squared_error(y, poly_predicted)
    poly_rmse = poly_mse ** 0.5
    st.write(f"RMSE 오차: {poly_rmse:.4f}")
