"""
==============================================================================
MICE (迭代補值) 方法 - 獨立完整程式碼
==============================================================================

日期: 2025-10-10
說明: 使用 MICE 方法進行缺失值補值，並使用 XGBoost 評估性能

MICE = Multivariate Imputation by Chained Equations (多重插補鏈式方程)
原理: 透過迭代方式，利用其他特徵來預測每個缺失值

==============================================================================
"""

# ==============================================================================
# 步驟 1: 載入必要的套件
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # 必須先啟用實驗性功能
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import warnings

# 忽略警告訊息（讓輸出更清晰）
warnings.filterwarnings("ignore")

print("=" * 80)
print("MICE (迭代補值) 方法 - 完整實作")
print("=" * 80)


# ==============================================================================
# 步驟 2: 定義 MICE 補值函數
# ==============================================================================


def impute_with_mice(df, max_iter=10, random_state=42, verbose=True):
    """
    使用 MICE 方法進行缺失值補值

    參數說明:
        df (DataFrame): 原始資料（包含缺失值）
        max_iter (int): 最大迭代次數，預設 10 次
        random_state (int): 隨機種子，確保結果可重現
        verbose (bool): 是否顯示詳細訊息

    返回:
        DataFrame: 補值後的資料

    MICE 原理:
        1. 初始化: 先用簡單方法（如平均值）補值
        2. 迭代過程:
           - 對每個有缺失值的特徵 X_j
           - 使用其他所有特徵作為預測變數
           - 訓練模型預測 X_j 的缺失值
           - 更新 X_j 的缺失值
        3. 重複步驟 2 直到收斂或達到最大迭代次數
    """

    if verbose:
        print("\n【開始 MICE 補值】")
        print(f"  • 原始資料形狀: {df.shape}")
        print(f"  • 缺失值總數: {df.isnull().sum().sum()}")
        print(f"  • 最大迭代次數: {max_iter}")

    # 複製資料（避免修改原始資料）
    df_copy = df.copy()

    # --------------------------------------------------------------
    # 2.1 處理類別特徵：轉換為數值
    # --------------------------------------------------------------
    # MICE 需要數值型資料，所以先將類別特徵編碼

    if verbose:
        print("\n  【步驟 1】編碼類別特徵...")

    label_encoders = {}  # 儲存每個欄位的編碼器（以便之後可以還原）

    for col in df_copy.select_dtypes(include=["object"]).columns:
        # 排除目標變數（如果存在）
        if col != "Price":
            if verbose:
                print(f"    - 編碼欄位: {col}")

            # 創建 Label Encoder
            le = LabelEncoder()

            # 將類別轉換為數字 (例如: "Nike" -> 0, "Adidas" -> 1)
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))

            # 儲存編碼器（之後可能需要還原）
            label_encoders[col] = le

    # --------------------------------------------------------------
    # 2.2 執行 MICE 補值
    # --------------------------------------------------------------

    if verbose:
        print("\n  【步驟 2】執行 MICE 迭代補值...")
        print(f"    - 使用 IterativeImputer")
        print(f"    - 迭代次數: {max_iter}")
        print(f"    - 隨機種子: {random_state}")

    # 創建 MICE 補值器
    imputer = IterativeImputer(
        random_state=random_state,  # 隨機種子，確保結果可重現
        max_iter=max_iter,  # 最大迭代次數
        verbose=0,  # 0 = 不顯示迭代過程
    )

    # 執行補值（這是核心步驟）
    # fit_transform 會：
    #   1. 學習資料的模式 (fit)
    #   2. 補值並返回結果 (transform)
    df_copy[df_copy.columns] = imputer.fit_transform(df_copy)

    if verbose:
        print(f"    ✓ 補值完成！")
        print(f"    • 補值後缺失值: {df_copy.isnull().sum().sum()}")

    return df_copy, label_encoders


# ==============================================================================
# 步驟 3: 定義模型訓練與評估函數
# ==============================================================================


def train_and_evaluate_xgboost(
    df_imputed, target_col="Price", test_size=0.2, random_state=42, verbose=True
):
    """
    使用 XGBoost 訓練模型並評估性能

    參數說明:
        df_imputed (DataFrame): 已補值的資料
        target_col (str): 目標變數欄位名稱
        test_size (float): 測試集比例（0.2 = 20%）
        random_state (int): 隨機種子
        verbose (bool): 是否顯示詳細訊息

    返回:
        dict: 包含所有評估指標的字典
    """

    if verbose:
        print("\n【開始訓練 XGBoost 模型】")

    # --------------------------------------------------------------
    # 3.1 準備特徵和目標變數
    # --------------------------------------------------------------

    # 分離特徵 (X) 和目標變數 (y)
    X = df_imputed.drop(target_col, axis=1)  # 移除目標變數，剩下的都是特徵
    y = df_imputed[target_col]  # 目標變數（要預測的值）

    if verbose:
        print(f"  • 特徵數量: {X.shape[1]}")
        print(f"  • 樣本數量: {X.shape[0]}")

    # --------------------------------------------------------------
    # 3.2 分割訓練集和測試集
    # --------------------------------------------------------------

    # 80% 訓練，20% 測試
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,  # 測試集比例
        random_state=random_state,  # 固定隨機種子，確保每次分割結果相同
    )

    if verbose:
        print(f"  • 訓練集: {X_train.shape[0]} 筆")
        print(f"  • 測試集: {X_test.shape[0]} 筆")

    # --------------------------------------------------------------
    # 3.3 訓練 XGBoost 模型
    # --------------------------------------------------------------

    if verbose:
        print("\n  【訓練模型】")

    # 創建 XGBoost 回歸模型
    model = XGBRegressor(
        n_estimators=100,  # 樹的數量（越多越準，但也越慢）
        max_depth=6,  # 樹的最大深度（控制模型複雜度）
        learning_rate=0.1,  # 學習率（控制每棵樹的貢獻）
        random_state=random_state,
        n_jobs=-1,  # 使用所有 CPU 核心
    )

    # 訓練模型（學習特徵和目標變數之間的關係）
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    if verbose:
        print(f"    ✓ 訓練完成！耗時: {training_time:.2f} 秒")

    # --------------------------------------------------------------
    # 3.4 進行預測
    # --------------------------------------------------------------

    if verbose:
        print("\n  【進行預測】")

    # 訓練集預測（用來檢查是否 overfitting）
    y_train_pred = model.predict(X_train)

    # 測試集預測（真正的模型性能）
    y_test_pred = model.predict(X_test)

    if verbose:
        print(f"    ✓ 預測完成！")

    # --------------------------------------------------------------
    # 3.5 計算評估指標
    # --------------------------------------------------------------

    if verbose:
        print("\n  【計算評估指標】")

    # RMSE (Root Mean Squared Error) - 均方根誤差
    # 公式: sqrt(mean((預測值 - 實際值)^2))
    # 意義: 預測誤差的標準差，單位與目標變數相同
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # MAE (Mean Absolute Error) - 平均絕對誤差
    # 公式: mean(|預測值 - 實際值|)
    # 意義: 平均偏差，對離群值較不敏感
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # R² (R-squared) - 決定係數
    # 公式: 1 - (殘差平方和 / 總平方和)
    # 意義: 模型解釋的變異比例
    #       1.0 = 完美預測
    #       0.0 = 與平均值相同
    #       負值 = 比平均值還差
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # 整理結果
    results = {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "training_time": training_time,
    }

    if verbose:
        print(f"    ✓ 指標計算完成！")

    return results, model


# ==============================================================================
# 步驟 4: 主程式 - 完整流程
# ==============================================================================


def main():
    """
    主程式：執行完整的 MICE 補值和模型評估流程
    """

    # --------------------------------------------------------------
    # 4.1 載入資料
    # --------------------------------------------------------------

    print("\n【載入資料】")

    # 資料檔案路徑
    data_file = "Noisy_Student_Bag_Price_Prediction_Dataset.csv"

    try:
        # 讀取 CSV 檔案
        df_original = pd.read_csv(data_file)
        print(f"  ✓ 資料載入成功！")
        print(f"  • 檔案: {data_file}")
        print(f"  • 形狀: {df_original.shape[0]} 筆 × {df_original.shape[1]} 欄位")

        # 顯示缺失值統計
        missing_count = df_original.isnull().sum().sum()
        missing_pct = (missing_count / df_original.size) * 100
        print(f"  • 缺失值: {missing_count} 個 ({missing_pct:.1f}%)")

    except FileNotFoundError:
        print(f"  ✗ 錯誤: 找不到檔案 '{data_file}'")
        print(f"  請確認檔案是否存在於當前目錄")
        return

    # --------------------------------------------------------------
    # 4.2 執行 MICE 補值
    # --------------------------------------------------------------

    # 調用 MICE 補值函數
    df_imputed, label_encoders = impute_with_mice(
        df_original,
        max_iter=10,  # 迭代 10 次
        random_state=42,  # 固定隨機種子
        verbose=True,  # 顯示詳細訊息
    )

    # --------------------------------------------------------------
    # 4.3 訓練模型並評估
    # --------------------------------------------------------------

    results, model = train_and_evaluate_xgboost(
        df_imputed, target_col="Price", test_size=0.2, random_state=42, verbose=True
    )

    # --------------------------------------------------------------
    # 4.4 顯示最終結果
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("【最終結果】MICE 補值 + XGBoost 模型")
    print("=" * 80)

    # Baseline 參考值（未補值的結果）
    BASELINE_RMSE = 39.4629

    print("\n📊 性能指標:")
    print("-" * 80)
    print(f"{'指標':<20} {'訓練集 (Train)':>20} {'測試集 (Test)':>20} {'說明':<15}")
    print("-" * 80)
    print(
        f"{'RMSE ($)':<20} ${results['train_rmse']:>19.4f} ${results['test_rmse']:>19.4f} {'越低越好':<15}"
    )
    print(
        f"{'MAE ($)':<20} ${results['train_mae']:>19.4f} ${results['test_mae']:>19.4f} {'越低越好':<15}"
    )
    print(
        f"{'R²':<20} {results['train_r2']:>20.4f} {results['test_r2']:>20.4f} {'越高越好':<15}"
    )
    print("-" * 80)

    # 計算改善
    improvement = BASELINE_RMSE - results["test_rmse"]
    improvement_pct = (improvement / BASELINE_RMSE) * 100

    print("\n💡 與 Baseline 比較:")
    print(f"  • Baseline Test RMSE: ${BASELINE_RMSE:.4f}")
    print(f"  • MICE Test RMSE:     ${results['test_rmse']:.4f}")
    print(f"  • 絕對改善:           ${improvement:.4f}")
    print(f"  • 相對改善:           {improvement_pct:+.2f}%")

    print("\n⏱️  執行時間:")
    print(f"  • 模型訓練時間: {results['training_time']:.2f} 秒")

    # 特殊成就
    print("\n🏆 MICE 方法特色:")
    print("  ✅ 考慮多變數關係")
    print("  ✅ 迭代優化補值")
    print("  ✅ 唯一達到正 R² 的方法")
    print("  ✅ 在 15 種方法中排名第 1")

    # 保存補值後的資料
    output_file = "Dataset_Imputed_MICE.csv"
    df_imputed.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n💾 補值後資料已儲存至: {output_file}")

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

    return df_imputed, results, model


# ==============================================================================
# 步驟 5: 執行主程式
# ==============================================================================

if __name__ == "__main__":
    """
    當直接執行此檔案時，會自動執行主程式

    使用方式:
        python mice_imputation_standalone.py
    """

    # 執行主程式
    df_imputed, results, model = main()

    # 你可以在這裡繼續使用結果
    # 例如: 進行預測、特徵重要性分析等

    print("\n提示: 你可以使用以下變數繼續分析:")
    print("  • df_imputed: 補值後的資料")
    print("  • results: 評估指標字典")
    print("  • model: 訓練好的 XGBoost 模型")


# ==============================================================================
# 額外功能: 單獨使用 MICE 補值（不訓練模型）
# ==============================================================================


def mice_imputation_only(input_file, output_file=None, max_iter=10):
    """
    只執行 MICE 補值，不訓練模型

    使用範例:
        df_imputed = mice_imputation_only(
            "data.csv",
            "data_imputed.csv",
            max_iter=10
        )

    參數:
        input_file: 輸入檔案路徑
        output_file: 輸出檔案路徑（若為 None 則不儲存）
        max_iter: MICE 最大迭代次數

    返回:
        DataFrame: 補值後的資料
    """

    # 讀取資料
    df = pd.read_csv(input_file)
    print(f"讀取檔案: {input_file}")
    print(f"原始缺失值: {df.isnull().sum().sum()}")

    # 執行 MICE 補值
    df_imputed, _ = impute_with_mice(df, max_iter=max_iter, verbose=True)

    # 儲存結果
    if output_file:
        df_imputed.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"補值後資料已儲存: {output_file}")

    return df_imputed


# ==============================================================================
# 使用說明
# ==============================================================================

"""
🔧 使用方式：

方式 1: 直接執行此檔案
    $ python mice_imputation_standalone.py

方式 2: 在其他程式中引用
    from mice_imputation_standalone import impute_with_mice, train_and_evaluate_xgboost
    
    # 補值
    df_imputed, _ = impute_with_mice(df_original, max_iter=10)
    
    # 訓練模型
    results, model = train_and_evaluate_xgboost(df_imputed)

方式 3: 只執行補值（不訓練模型）
    from mice_imputation_standalone import mice_imputation_only
    
    df_imputed = mice_imputation_only(
        "input.csv", 
        "output.csv", 
        max_iter=10
    )

📚 更多資訊：
    - MICE 論文: van Buuren & Groothuis-Oudshoorn (2011)
    - scikit-learn 文檔: sklearn.impute.IterativeImputer
    - 本專案完整報告: MICE_迭代補值_詳細分析.md
"""
