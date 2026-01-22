import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 데이터 로드 및 전처리
file_path = 'dir/korea_person.csv'
df = pd.read_csv(file_path)
if 'Unnamed: 2' in df.columns:
    df = df.rename(columns={'Unnamed: 2': 'Population'})

age_order = ['5-', '5~9', '10~14', '15~19', '20~24', '25~29', '30~34',
             '35~39', '40~44', '45~49', '50~54', '55~59', '60~64',
             '65~69', '70~74', '75~79', '80+']
df_pivot = df.pivot(index='Year', columns='Age_Grp', values='Population')
df_pivot = df_pivot[age_order]
years = df_pivot.index.sort_values()

# 2. 과거의 변화율(CCR, CWR) 데이터 수집
ccr_data = []
cwr_data = []

idx_15_49_start = 3
idx_15_49_end = 9

# (1) CWR(출산비율) 시계열 수집
for t in years:
    pop_t = df_pivot.loc[t]
    child_pop = pop_t['5-']
    parent_pop = pop_t.iloc[idx_15_49_start: idx_15_49_end + 1].sum()
    if parent_pop > 0:
        cwr_data.append({'Year': t, 'CWR': child_pop / parent_pop})

# (2) CCR(생존이동률) 시계열 수집
for t in years:
    if t + 5 in years:
        pop_t = df_pivot.loc[t]
        pop_t5 = df_pivot.loc[t + 5]
        row = {'Year': t}

        # 일반 연령대 이동
        for i in range(len(age_order) - 2):
            if pop_t.iloc[i] > 0:
                row[f'CCR_{i}'] = pop_t5.iloc[i + 1] / pop_t.iloc[i]
            else:
                row[f'CCR_{i}'] = 1.0

        # 고령층(80+) 이동
        pop_old_start = pop_t.iloc[-2] + pop_t.iloc[-1]
        pop_old_end = pop_t5.iloc[-1]
        val_last = 1.0
        if pop_old_start > 0:
            val_last = pop_old_end / pop_old_start
        row['CCR_last'] = val_last

        ccr_data.append(row)

df_cwr_ts = pd.DataFrame(cwr_data)
df_ccr_ts = pd.DataFrame(ccr_data)

# 3. 변화율 자체를 예측 (Regression)
# 미래 적용 시점: 2025년(->2030), 2030년(->2035)
future_start_years = [years.max(), years.max() + 5]
predicted_ccrs = {}
predicted_cwrs = {}

# (1) CWR 예측 (로그-선형 회귀로 음수 방지 및 감소세 반영)
X_cwr = df_cwr_ts[['Year']].values
y_cwr_log = np.log(df_cwr_ts['CWR'].values)
model_cwr = LinearRegression()
model_cwr.fit(X_cwr, y_cwr_log)

target_years_cwr = np.array([years.max() + 5, years.max() + 10]).reshape(-1, 1)
pred_cwr_log = model_cwr.predict(target_years_cwr)
pred_cwr = np.exp(pred_cwr_log)

predicted_cwrs[target_years_cwr[0][0]] = pred_cwr[0]  # 2030년 CWR
predicted_cwrs[target_years_cwr[1][0]] = pred_cwr[1]  # 2035년 CWR

print(f"예측된 미래 출산비율: 2030년({pred_cwr[0]:.4f}) -> 2035년({pred_cwr[1]:.4f})")

print(predicted_ccrs)
print(predicted_cwrs )

# (2) CCR 예측 (선형 회귀)
ccr_cols = [c for c in df_ccr_ts.columns if 'CCR' in c]
X_ccr = df_ccr_ts[['Year']].values
future_X_ccr = np.array(future_start_years).reshape(-1, 1)

# 미래 연도별로 빈 배열 생성
for start_year in future_start_years:
    predicted_ccrs[start_year] = np.zeros(len(ccr_cols))

for idx, col in enumerate(ccr_cols):
    y_ccr = df_ccr_ts[col].values
    model_ccr = LinearRegression()
    model_ccr.fit(X_ccr, y_ccr)

    pred = model_ccr.predict(future_X_ccr)
    # 안전장치: 비율이 너무 튀지 않도록 0.5 ~ 1.5로 제한 (Clip)
    pred = np.clip(pred, 0.5, 1.5)

    predicted_ccrs[future_start_years[0]][idx] = pred[0]
    predicted_ccrs[future_start_years[1]][idx] = pred[1]

# 4. 동적 변화율을 적용한 인구 추계
last_year = years.max()
predictions_dynamic = {}
current_pop = df_pivot.loc[last_year].values.copy()

for i, future_year in enumerate([last_year + 5, last_year + 10]):
    # 해당 시점에 맞는 예측된 비율(CCR, CWR) 가져오기
    start_year_for_step = last_year + (i * 5)
    ccr_vec = predicted_ccrs[start_year_for_step]
    cwr_val = predicted_cwrs[future_year]

    next_pop = np.zeros_like(current_pop)

    # 코호트 이동
    for k in range(len(age_order) - 2):
        next_pop[k + 1] = current_pop[k] * ccr_vec[k]

    # 고령층
    next_pop[-1] = (current_pop[-2] + current_pop[-1]) * ccr_vec[-1]

    # 신생아
    parent_sum = np.sum(next_pop[idx_15_49_start: idx_15_49_end + 1])
    next_pop[0] = parent_sum * cwr_val

    predictions_dynamic[future_year] = next_pop
    current_pop = next_pop

# 5. 결과 저장 및 시각화
years_all = list(years) + [last_year + 5, last_year + 10]
data_all = list(df_pivot.values) + [predictions_dynamic[y] for y in [last_year + 5, last_year + 10]]
df_dynamic = pd.DataFrame(data_all, index=years_all, columns=age_order)
df_dynamic_interp = df_dynamic.reindex(range(years.min(), last_year + 11)).interpolate(method='linear').astype(int)

df_dynamic_interp.to_csv('cohort_prediction_dynamic.csv')
print("저장 완료: cohort_prediction_dynamic.csv")

plt.figure(figsize=(10, 5))
plt.plot(df_dynamic_interp.index, df_dynamic_interp['5-'], label='0-4 (Projected Rates)', marker='o')
plt.plot(df_dynamic_interp.index, df_dynamic_interp['80+'], label='80+ (Projected Rates)', marker='x')
plt.axvline(x=last_year, color='red', linestyle=':', label='Forecast Start')
plt.title('Dynamic Cohort Forecast (Predicted Rates)')
plt.legend()
plt.grid(True)
plt.show()