import fitz
import os


def create_pdf_paper(title: str, category: str, output_path: str) -> None:
    """
    Create a dummy PDF paper with title and category.
    创建一个带有标题和类别的虚拟 PDF 论文。

    Args:
        title (str): Title of the paper. 论文标题。
        category (str): Category of the paper. 论文类别。
        output_path (str): Output path for the PDF. PDF 输出路径。
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), title, fontsize=20)
    page.insert_text((72, 110), f"Category: {category}", fontsize=12)
    page.insert_text(
        (72, 150),
        "This is a generated test paper for functional testing.\n"
        "It contains placeholder content including title and category.",
        fontsize=11,
    )
    doc.save(output_path)
    doc.close()
    print(f"Created {output_path}")


def setup_test_data_pdf() -> None:
    """
    Generate a set of test PDF papers.
    生成一组测试 PDF 论文。
    """
    root = os.getcwd()
    pdf_dir = os.path.join(root, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)

    # Direction: Computer Vision
    # 方向：计算机视觉
    for i in range(1, 4):
        filename = os.path.join(pdf_dir, f"CV_paper_{i:02d}.pdf")
        create_pdf_paper(
            title=f"Computer Vision Test Paper {i}",
            category="Computer Vision",
            output_path=filename,
        )

    # Direction: Natural Language Processing
    # 方向：自然语言处理
    for i in range(1, 4):
        filename = os.path.join(pdf_dir, f"NLP_paper_{i:02d}.pdf")
        create_pdf_paper(
            title=f"Natural Language Processing Test Paper {i}",
            category="Natural Language Processing",
            output_path=filename,
        )

    # Direction: Reinforcement Learning
    # 方向：强化学习
    for i in range(1, 4):
        filename = os.path.join(pdf_dir, f"RL_paper_{i:02d}.pdf")
        create_pdf_paper(
            title=f"Reinforcement Learning Test Paper {i}",
            category="Reinforcement Learning",
            output_path=filename,
        )

    # Direction: Data Mining
    # 方向：数据挖掘
    for i in range(1, 4):
        filename = os.path.join(pdf_dir, f"DM_paper_{i:02d}.pdf")
        create_pdf_paper(
            title=f"Data Mining Test Paper {i}",
            category="Data Mining",
            output_path=filename,
        )

    # Direction: Machine Learning
    # 方向：机器学习
    for i in range(1, 4):
        filename = os.path.join(pdf_dir, f"ML_paper_{i:02d}.pdf")
        create_pdf_paper(
            title=f"Machine Learning Test Paper {i}",
            category="Machine Learning",
            output_path=filename,
        )


if __name__ == "__main__":
    setup_test_data_pdf()
# [CV_paper_01.pdf]:[目标检测,图像分割,特征提取]
# [CV_paper_02.pdf]:[图像分类,卷积神经网络,边缘检测]
# [CV_paper_03.pdf]:[物体识别,姿态估计,特征匹配]
# [NLP_paper_01.pdf]:[文本分类,语言模型,词向量]
# [NLP_paper_02.pdf]:[命名实体识别,序列标注,依存句法]
# [NLP_paper_03.pdf]:[机器翻译,情感分析,预训练模型]
# [RL_paper_01.pdf]:[Q学习,价值函数,探索-利用]
# [RL_paper_02.pdf]:[策略梯度,Actor-Critic,奖励设计]
# [RL_paper_03.pdf]:[蒙特卡洛,时序差分,环境建模]
# [DM_paper_01.pdf]:[关联规则,频繁项集,市场篮分析]
# [DM_paper_02.pdf]:[聚类分析,KMeans,层次聚类]
# [DM_paper_03.pdf]:[异常检测,离群点,数据清洗]
# [ML_paper_01.pdf]:[监督学习,交叉验证,正则化]
# [ML_paper_02.pdf]:[无监督学习,降维,特征选择]
# [ML_paper_03.pdf]:[模型评估,偏差-方差,超参数调优]
