

'''
AD_SensorArray_project/
├── main.py                    # 主入口脚本
├── config.py                  # 配置参数（文件路径、模型参数等）
├── data_loader.py            # 数据加载和预处理
├── model_training.py         # 模型训练和评估
├── visualization.py          # 可视化函数
├── utils.py                  # 工具函数
└── results/                  # 结果保存目录
    ├── models/               # 保存训练好的模型
    ├── figures/              # 保存生成的图表
    └── metrics/              # 保存评估指标
'''
import os
import pandas as pd

from data_loader import load_data
from model_training import train_and_evaluate, tune_best_models
from config import RESULTS_DIR, TASK_TYPE


def main():
    # =========================================================
    # 1. 数据加载与预处理
    # =========================================================
    X_train, X_test, y_train, y_test, scaler, label_info, feat_names = load_data()

    if TASK_TYPE == "binary":
        class_names = ["non-AD", "AD"]
    else:
        # multiclass 时 label_info 是 LabelEncoder
        class_names = label_info.classes_.tolist()

    print("==================================================")
    print(f"任务类型: {TASK_TYPE}")
    print("数据加载完成")
    print(f"训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")
    print("特征数: ", len(feat_names))
    print("==================================================")

    # =========================================================
    # 2. 随机搜索调参
    # =========================================================
    print("\n=== 随机搜索调参阶段 ===")
    best_params = tune_best_models(X_train, y_train)

    # =========================================================
    # 3. 模型训练与测试集评估
    # =========================================================
    print("\n=== 模型训练与评估阶段 ===")
    results, best_model_info = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder=class_names,
        feature_names=feat_names,
        tuned_params=best_params
    )

    # =========================================================
    # 4. 保存汇总结果
    # =========================================================
    df = pd.DataFrame(results)

    if TASK_TYPE == "binary":
        sort_col = "test_auc"
    else:
        sort_col = "test_f1_macro"

    df = df.sort_values(by=sort_col, ascending=False)

    summary_path = os.path.join(RESULTS_DIR, "summary_results.csv")
    df.to_csv(summary_path, index=False)

    print("\n==== 模型最终结果 ====")
    show_cols = ["model", "test_accuracy", "test_f1_macro", "test_auc"]
    print(df[show_cols])

    print("\n==== 最优模型 ====")
    print(f"模型名称: {best_model_info['model_name']}")
    print(f"测试集主指标: {best_model_info['primary_metric_name']} = {best_model_info['primary_metric_value']:.4f}")

    print(f"\n结果已保存至: {RESULTS_DIR}")
    print(f"汇总表: {summary_path}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main()